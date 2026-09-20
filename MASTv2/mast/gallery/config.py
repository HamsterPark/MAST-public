# -*- coding: utf-8 -*-
"""图库配置：要索引哪些数据根（设计文档 D2 / D4）。

**显式配置，不隐式扫描。** 打开页面不会遍历任何目录；:func:`suggest_roots` 只给建议，
点一下才加。几十 GB 的数据树不该因为有人点开一个标签页就被遍历，测试也不会因此误扫
真实数据（陷阱 T3）。

根名是每个条目 id 的第一段（``<根名>/<相对路径>``）。数据搬家时改 ``path`` 不改 ``name``，
标记跟着走。
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mast.gallery import paths as _paths

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1
MAX_WORKERS = 16


def default_workers() -> int:
    return min(4, max(1, (os.cpu_count() or 2) // 2))


@dataclass(frozen=True)
class RootSpec:
    name: str
    path: str
    enabled: bool = True


@dataclass
class GalleryConfig:
    roots: list[RootSpec] = field(default_factory=list)
    workers: int = field(default_factory=default_workers)


# ── 读写 ───────────────────────────────────────────────────────────────


def load_config(lay: _paths.Layout | None = None) -> GalleryConfig:
    """读配置。文件不存在 ⇒ 空根列表 + 默认 workers。**不创建任何东西。**"""
    lay = lay or _paths.layout()
    raw = _paths.read_json(lay.config, None)
    if raw is None:
        if lay.config.exists():
            logger.warning("gallery config unreadable, treating as empty: %s", lay.config)
        return GalleryConfig()
    if not isinstance(raw, dict):
        return GalleryConfig()
    roots: list[RootSpec] = []
    for r in raw.get("roots") or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "").strip()
        path = str(r.get("path") or "").strip()
        if name and path:
            roots.append(RootSpec(name=name, path=path, enabled=bool(r.get("enabled", True))))
    try:
        workers = int(raw.get("workers"))
    except (TypeError, ValueError):
        workers = default_workers()
    return GalleryConfig(roots=roots, workers=min(MAX_WORKERS, max(1, workers)))


def save_config(cfg: GalleryConfig, lay: _paths.Layout | None = None) -> None:
    lay = lay or _paths.layout()
    _paths.ensure_state_dir(lay)
    _paths.atomic_write_json(lay.config, {
        "version": CONFIG_VERSION,
        "roots": [asdict(r) for r in cfg.roots],
        "workers": int(min(MAX_WORKERS, max(1, cfg.workers))),
    }, indent=1)


# ── 根名与路径校验 ─────────────────────────────────────────────────────

_NAME_OK = re.compile(r"^[A-Za-z0-9_.\-㐀-鿿豈-﫿]+$")
_WIN_RESERVED = {"con", "prn", "aux", "nul",
                 *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def validate_name(name: str) -> str | None:
    """合法返回 None，否则返回原因（中文，给操作员看）。

    根名会成为 ``thumbs/<根名>/`` 目录名与 URL 的一段，所以除了 D4 的字符集，还要避开
    Windows 保留设备名与结尾的点（Windows 会悄悄吃掉结尾的点）。"""
    if not name:
        return "根名为空"
    if name[0] in "_.":
        return f"根名「{name}」不能以 _ 或 . 开头"
    if name.endswith("."):
        return f"根名「{name}」不能以 . 结尾"
    if not _NAME_OK.match(name):
        return f"根名「{name}」只能含字母、数字、_ . - 与中文"
    if name.lower() in _WIN_RESERVED:
        return f"根名「{name}」是 Windows 保留名"
    return None


def derive_name(path: str) -> str:
    """从目录名派生一个合法根名。"""
    base = os.path.basename(os.path.normpath(path)) or "root"
    base = re.sub(r"[^A-Za-z0-9_.\-㐀-鿿豈-﫿]+", "_", base)
    base = base.lstrip("_.").rstrip(".")
    if not base:
        base = "root"
    if base.lower() in _WIN_RESERVED:
        base = base + "_data"
    return base


def _norm(p: str | Path) -> str:
    try:
        return os.path.normcase(os.path.realpath(str(p)))
    except OSError:
        return os.path.normcase(os.path.abspath(str(p)))


def _within(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip("\\/") + os.sep)


def normalise_roots(inputs: list[dict]) -> tuple[list[RootSpec], list[str]]:
    """校验并规整一组根。``(roots, errors)``；有错误时调用方不应写盘。

    规则（D4）：路径必须是已存在的目录，统一成绝对路径；不许重复、不许互相嵌套
    （嵌套的两个根会让同一个文件得到两个 id）；不许落在图库状态目录里；显式给的根名
    必须合法且不重复；没给名字的从目录名派生并去重（不抢显式名字）。
    """
    errors: list[str] = []
    out: list[RootSpec] = []
    used: list[tuple[str, str]] = []            # (规范化路径, 根名)
    explicit = {str(r.get("name") or "").strip().casefold()
                for r in inputs if isinstance(r, dict) and str(r.get("name") or "").strip()}
    taken: set[str] = set()
    try:
        state_key = _norm(_paths.state_dir())
    except Exception:  # noqa: BLE001 — 解析不了状态目录不该挡住配置
        state_key = None

    for i, r in enumerate(inputs, 1):
        if not isinstance(r, dict):
            errors.append(f"第 {i} 个数据根格式不对")
            continue
        raw_path = str(r.get("path") or "").strip()
        if not raw_path:
            errors.append(f"第 {i} 个数据根没有填路径")
            continue
        ap = os.path.abspath(os.path.expanduser(raw_path))
        if not os.path.isdir(ap):
            errors.append(f"不是已存在的目录：{raw_path}")
            continue
        key = _norm(ap)
        if state_key and _within(key, state_key):
            errors.append(f"数据根不能放在图库状态目录里：{raw_path}")
            continue
        clash = next(((k, n) for k, n in used if k == key or _within(key, k) or _within(k, key)), None)
        if clash is not None:
            if clash[0] == key:
                errors.append(f"数据根重复：{raw_path}")
            else:
                errors.append(f"数据根不能互相嵌套：{raw_path} 与根「{clash[1]}」")
            continue
        name = str(r.get("name") or "").strip()
        if name:
            why = validate_name(name)
            if why:
                errors.append(why)
                continue
            if name.casefold() in taken:
                errors.append(f"根名重复：{name}")
                continue
        else:
            base = derive_name(ap)
            name, n = base, 2
            while name.casefold() in taken or name.casefold() in explicit:
                name = f"{base}_{n}"
                n += 1
        taken.add(name.casefold())
        used.append((key, name))
        out.append(RootSpec(name=name, path=ap, enabled=bool(r.get("enabled", True))))
    return out, errors


# ── 建议 ───────────────────────────────────────────────────────────────

_Y4 = re.compile(r"^\d{4}$")
_YM6 = re.compile(r"^\d{6}$")
_YMD8 = re.compile(r"^\d{8}$")


def suggest_roots(lay: _paths.Layout | None = None) -> list[tuple[str, str]]:
    """值得加进来的目录。只给**现在存在**的；已经配过的不再建议。**只读。**

    会话目录只从本进程已经记录过的 ``scan_registry.known_scan_dirs()`` 里拿 ——
    **绝不为了给建议去连仪器**。Nanonis 默认的会话目录形如 ``.../YYYY/YYYYMM/YYYYMMDD``，
    这时上溯三级就是整棵数据树的根，那才是图库该索引的东西。
    """
    lay = lay or _paths.layout()
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        seen.update(_norm(r.path) for r in load_config(lay).roots)
    except Exception:  # noqa: BLE001
        pass

    def add(p: str | Path | None, why: str) -> None:
        if not p:
            return
        try:
            if not os.path.isdir(p):
                return
        except OSError:
            return
        k = _norm(p)
        if k in seen:
            return
        seen.add(k)
        out.append((str(p), why))

    try:
        from mast.core.scan_registry import known_scan_dirs

        for d in known_scan_dirs():
            p = Path(d)
            if (_YMD8.match(p.name) and _YM6.match(p.parent.name)
                    and _Y4.match(p.parent.parent.name)):
                add(p.parent.parent.parent,
                    f"Nanonis 数据树的根（会话目录 {p} 上溯三级）")
            add(p, "Nanonis 会话目录（本次运行里记录过）")
    except Exception:  # noqa: BLE001 — 建议是锦上添花
        pass
    try:
        from mast._runtime_paths import project_root

        add(project_root() / "working-sessions", "MAST 扫描技能的默认落盘目录")
    except Exception:  # noqa: BLE001
        pass
    try:
        from mast.core.experiment_paths import experiment_root

        add(experiment_root(),
            "实验根（自动拷贝进实验文件夹的副本；与原文件同时加入时会折叠成一条）")
    except Exception:  # noqa: BLE001
        pass
    return out
