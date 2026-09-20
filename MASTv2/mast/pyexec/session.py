"""会话工作目录 —— 一次分析任务的家。

为什么是「会话」而不是「一次性脚本」
==================================
数据分析是探索性的：先看形状、再试个滤波、发现不对再换个方法。一次性脚本逼着
模型一口气写对，写错就整段重来；而**上一步存下的中间结果下一步还在**，就能拆成
几个小步骤逐步逼近。这也是长代码的答案 —— 与其把 500 行塞进一次 tool 调用
（`max_tokens` 那里先撞墙），不如写五个 100 行的步骤。

每次执行仍然是**全新的子进程**：崩了、跑飞了、被杀了，会话目录都还在。

目录形状
========
::

    <data>/dp-sessions/20260819-run7-3f2a/
      manifest.json      会话级：id / created / experiment_id / sample_id / steps[]
      sitecustomize.py   ← 审计钩子的加载点（解释器一启动就装）
      _mast_audit.py     ← 钩子实现
      mastdata.py        ← 薄 helper：scan_dirs / load / out / savefig / save_result
      mastkit.py         ← 便利别名
      mast/              ← 12 个纯分析模块的字节拷贝（可 import）
      source/            ← 整个 mast/ 源码只读视图（读，不 import）
      inputs/            ← py_stage_data 的产物 + manifest.json（含物理尺度）
      code/stepNN.py     ← 历次脚本，全部留档
      logs/stepNN.{out,err}
      out/               ← 推荐的产物落点（harvest 从这里收）
      tmp/  .mpl/

``out/`` 是**推荐**落点不是唯一能写的地方：审计钩子只保护已存在的测量文件，脚本
想往别处写派生结果是允许的（只是不会被自动收走，要自己在 result.json 里报路径）。

会话身份从**框架注入**的东西派生，不从模型
=========================================
``experiment_id``（state）+ ``thread_id``（config）。让模型自己起名字，就会有
「上次那个会话叫什么来着」这类问题，而它答错的代价是丢掉全部中间结果。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# 保留策略。都是**上限**，不是预算 —— 越界时删最旧的，且永远不动活动会话。
MAX_SESSIONS = 50
MAX_TOTAL_BYTES = 20 * 1024 ** 3

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_lock = threading.RLock()


def sessions_root(*, create: bool = True) -> Path:
    """``<data>/dp-sessions/``。

    落在**数据目录**而不是安装目录：安装目录在 Program Files 下，Inno 只给五个
    具名目录 users-modify（``mast2_setup.iss:88-93``）；而且 OTA delta 只碰安装
    目录，卸载器也只删 ``{app}`` —— 会话是用户的数据，三条都得躲开。
    """
    raw = os.environ.get("MAST_DP_SESSIONS_DIR", "").strip()
    if raw:
        d = Path(raw).expanduser()
    else:
        from mast._runtime_paths import project_root
        d = project_root() / "dp-sessions"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(s: str, n: int = 12) -> str:
    return (_SLUG_RE.sub("-", str(s or "")).strip("-") or "x")[:n]


def derive_session_id(experiment_id: str = "", thread_id: str = "",
                      *, day: str = "") -> str:
    """``<日期>-<实验>-<线程哈希>``。同一个实验的同一条对话线程 ⇒ 同一个会话。"""
    day = day or datetime.now(timezone.utc).astimezone().strftime("%Y%m%d")
    exp = _slug(experiment_id or "noexp")
    h = hashlib.sha256(str(thread_id or "nothread").encode("utf-8")).hexdigest()[:6]
    return f"{day}-{exp}-{h}"


@dataclass
class PySession:
    root: Path
    sid: str
    experiment_id: str = ""
    sample_id: str = ""
    created: str = ""
    steps: list[dict] = field(default_factory=list)

    # ── 目录 ─────────────────────────────────────────────────────────
    @property
    def code_dir(self) -> Path:
        return self.root / "code"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    # ── manifest ─────────────────────────────────────────────────────
    def save(self) -> None:
        """原子替换（``.part`` + ``os.replace``）—— 任意时刻拔电源，盘上的都自洽。

        这是本仓的 INCREMENTAL-ONLY 不变式（``docs/v2/architecture/v2.md:147-162``
        第 6 条）在这里的形态。
        """
        data = {
            "schema": 1,
            "session_id": self.sid,
            "created": self.created,
            "experiment_id": self.experiment_id,
            "sample_id": self.sample_id,
            "steps": self.steps,
        }
        tmp = self.manifest_path.with_suffix(".json.part")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    @classmethod
    def _load(cls, root: Path) -> "PySession | None":
        p = root / "manifest.json"
        if not p.is_file():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — 坏 manifest 不该让会话不可用
            logger.warning("会话 manifest 读不出来，按新会话处理：%s", p)
            return None
        return cls(root=root, sid=str(d.get("session_id") or root.name),
                   experiment_id=str(d.get("experiment_id") or ""),
                   sample_id=str(d.get("sample_id") or ""),
                   created=str(d.get("created") or ""),
                   steps=list(d.get("steps") or []))

    # ── step ─────────────────────────────────────────────────────────
    def next_step_name(self) -> str:
        return f"step{len(self.steps) + 1:02d}.py"

    def record_step(self, entry: dict) -> None:
        self.steps.append(entry)
        self.save()

    def size_bytes(self) -> int:
        total = 0
        for p in self.root.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
        return total


def _provision(session: PySession) -> None:
    """把会话目录布置好：子目录 + 审计钩子 + mastkit + 源码视图。

    幂等 —— 每次 ``py_run`` 都会调，因为一个被手工删掉的 ``sitecustomize.py``
    会让**保护静默消失**，而那是最不能靠「应该还在吧」的东西。
    """
    from mast.pyexec import kit_manifest as km

    for d in (session.code_dir, session.logs_dir, session.out_dir,
              session.inputs_dir, session.tmp_dir, session.root / ".mpl"):
        d.mkdir(parents=True, exist_ok=True)

    # ⚠️ 不能用 Path(__file__).parent：打包版里这些 .py 在 MAST.exe 的 PYZ 归档
    # 里，磁盘上不存在。而我们要的是**文件副本**（子进程里没有 MAST，只能拷文件）。
    # 找不到就抛 —— 静默跳过意味着审计钩子不装、保护消失，而一切看起来正常。
    from mast.pyexec._srcfiles import pyexec_file

    for src_name, dst_name in (("child_audit.py", "_mast_audit.py"),
                               ("child_sitecustomize.py", "sitecustomize.py"),
                               ("runtime_helper.py", "mastdata.py")):
        shutil.copy2(pyexec_file(src_name), session.root / dst_name)

    if km.verify(session.root):          # 缺文件或漂移了才重装
        km.install(session.root)

    _link_source(session)


def _link_source(session: PySession) -> None:
    """整个 ``mast/`` 源码的**只读视图** —— 给 agent 当教材，不是给它 import 的。

    那些 skill 里全是「怎么处理 STM 数据」的实战代码（平面扣除、去条纹、峰拟合、
    晶格检测），比让它从零推导强得多。

    优先建目录联结（Windows junction / POSIX symlink），失败就退回不建 —— **不**
    退回成整目录拷贝：每个会话拷 14 MB 是白花的磁盘，而这只是个参考资料。
    """
    dst = session.source_dir
    if dst.exists():
        return
    try:
        from mast.pyexec._srcfiles import source_root
        dst.symlink_to(source_root(), target_is_directory=True)
    except Exception as exc:  # noqa: BLE001 — 这一条**确实**只是参考资料
        logger.debug("source/ 视图没建起来（不影响分析）：%s", exc)


def get_session(*, experiment_id: str = "", thread_id: str = "",
                reset: bool = False) -> PySession:
    """拿到（必要时新建）这条线程的会话。

    ``reset=True`` **归档而不是删除** —— 用户的中间结果是数据，不是垃圾。
    """
    sid = derive_session_id(experiment_id, thread_id)
    root = sessions_root() / sid
    with _lock:
        if reset and root.exists():
            stamp = datetime.now(timezone.utc).astimezone().strftime("%H%M%S")
            archive = sessions_root() / "_archive" / f"{sid}-{stamp}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root), str(archive))
            logger.info("会话 %s 已归档到 %s（没有删除）", sid, archive)

        existing = PySession._load(root) if root.exists() else None
        if existing is None:
            root.mkdir(parents=True, exist_ok=True)
            existing = PySession(
                root=root, sid=sid, experiment_id=experiment_id,
                created=datetime.now(timezone.utc).astimezone().isoformat(
                    timespec="seconds"))
            existing.save()
            _prune(keep=sid)
        else:
            # 实验换了就更新（同一条线程可能跨实验）
            if experiment_id and existing.experiment_id != experiment_id:
                existing.experiment_id = experiment_id
                existing.save()
        _provision(existing)
        return existing


def _prune(*, keep: str = "") -> None:
    """按数量和总量剪最旧的会话。**永不动活动的那个，每次删除都记日志。**

    归档目录（``_archive/``）里的 ``code/`` 是 agent 攒下来的分析脚本，几十 KB，
    删它省不下什么 —— 所以剪的时候整目录留着，只在超总量时才动。
    """
    root = sessions_root(create=False)
    if not root.is_dir():
        return
    try:
        dirs = [d for d in root.iterdir()
                if d.is_dir() and d.name != "_archive" and d.name != keep]
    except OSError:
        return
    dirs.sort(key=lambda d: d.stat().st_mtime)

    def _drop(d: Path, why: str) -> None:
        try:
            size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        except OSError:
            size = 0
        shutil.rmtree(d, ignore_errors=True)
        logger.info("剪掉旧会话 %s（%s，%.1f MiB）", d.name, why, size / 1024 ** 2)

    while len(dirs) > MAX_SESSIONS:
        _drop(dirs.pop(0), f"超过 {MAX_SESSIONS} 个会话")

    def _total() -> int:
        t = 0
        for d in dirs:
            try:
                t += sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
            except OSError:
                pass
        return t

    while dirs and _total() > MAX_TOTAL_BYTES:
        _drop(dirs.pop(0), f"超过 {MAX_TOTAL_BYTES / 1024**3:.0f} GiB 总量")


__all__ = ["MAX_SESSIONS", "MAX_TOTAL_BYTES", "PySession", "derive_session_id",
           "get_session", "sessions_root"]
