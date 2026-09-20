"""实验文件夹的落盘器 —— 把 Nanonis 原始文件复制进当前样品的 ``raw/``。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §7

用户诉求的核心一句：**即使用户在 Nanonis 里设了独立保存目录，实验文件夹内也
必须有一份副本。** 在此之前，全仓库没有任何代码把 .sxm/.dat 复制到实验目录
（唯一按实验归档文件的是事后导出 ``rocrate.py``）。

INCREMENTAL-ONLY（硬不变式)
---------------------------

实验文件夹里没有任何东西依赖"结束"事件 —— 实验永不归档，用户可能十年后重启它。
**任意时刻拔电源，磁盘上已有的内容必须自洽、可用、可继续。**
所有落盘只有两种形态：**原子追加**（jsonl 单行 + flush）或**原子替换**
（``.part`` + ``os.replace``）。

**禁止**在本模块（或 manifest.py）编写 ``on_experiment_end`` / ``on_archive`` /
``finalize_*`` 形式的批量收尾函数。整目录校验用 :meth:`ExperimentFileStore.verify_folder`
**随时按 manifest 重算**，不产出任何封存文件。

防自我复制：四区分类，不是"排除整个 experiment_root"
----------------------------------------------------

天真做法是「路径在 experiment_root 内就跳过」。**那是错的，而且静默**：一旦
用户开启原位模式（Nanonis 直接写进实验文件夹），所有新文件都在 root 内 →
全部被跳过 → 一个文件都收不进记录，没有任何报错，只是 ``scan_files`` 永远为 0
（恰好是这个项目已经踩过一次的坑）。

正确做法见 :func:`classify`：只**无条件忽略 MAST 自管区**
（``raw/{sxm,dat,3ds,other}/``、``derived/`` 等）。我们写出去的副本永远落在
自管区，所以 copy→detect→copy 的环在结构上不成立；而 Nanonis 写进
``raw/nanonis/`` 的原位文件仍然会被收进记录。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from mast.core.experiment_paths import (
    MANAGED_SUBDIRS,
    RAW_KIND_DIRS,
    RAW_NANONIS_DIR,
    RAW_OTHER_DIR,
    experiment_root,
    is_within,
)

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20                     # 1 MiB 流式块
_MIN_FREE_BYTES = 256 << 20          # 复制前要求的空闲余量
_DEFAULT_MAX_BYTES = 2 << 30         # 单文件上限，防失控的 3ds 网格写满盘

#: 静置门：文件 mtime 距今不足这么久就认为"可能还在写"，跳过等下一跳。
#: 沿用 runtime._map_spectrum_tick 已在用的 5 s 规则，按类型细分。
SETTLE_S: dict[str, float] = {".sxm": 2.0, ".dat": 2.0, ".3ds": 5.0}
#: 原位模式下没有原子落地保护（半写的文件就是规范文件），静置门加倍。
#: 登记只读一遍算 hash，很便宜，可以更保守。
SETTLE_S_INPLACE: dict[str, float] = {".sxm": 4.0, ".dat": 4.0, ".3ds": 10.0}
_SETTLE_DEFAULT = 2.0
_SETTLE_DEFAULT_INPLACE = 4.0


class Zone(str, Enum):
    """一个路径相对实验文件夹体系的位置。见模块 docstring。"""

    EXTERNAL = "external"                # 不在 experiment_root 内 → 常规复制
    INPLACE_ACTIVE = "inplace_active"    # 当前活跃样品的 raw/nanonis/ → 原位登记
    INPLACE_FOREIGN = "inplace_foreign"  # root 内但不是活跃样品的 raw/nanonis/ → 复制到活跃样品 + 告警
    MANAGED = "managed"                  # MAST 自管区 → 无条件忽略（断递归的那一刀）


@dataclass(frozen=True)
class IngestResult:
    """一次 ingest 的结果。绝不通过异常表达失败 —— 调用方永远拿得到一个结果。"""

    origin: Path
    disposition: str                 # new|duplicate|inplace|skipped|retry|quarantined|failed
    sha256: str = ""
    size_bytes: int = 0
    dest: Path | None = None         # 实验文件夹内的规范副本（原位时 == origin）
    rel_path: str = ""               # POSIX，相对实验目录 → 目录整体移动零成本
    zone: Zone = Zone.EXTERNAL
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.disposition in ("new", "duplicate", "inplace")


# ── 分区判定 ──────────────────────────────────────────────────────────

def _managed_markers(p: Path) -> bool:
    """路径中是否出现 MAST 自管区的目录名。"""
    parts = [x.lower() for x in p.parts]
    if any(m.lower() in parts for m in MANAGED_SUBDIRS):
        return True
    # raw/<kind>/ 里除 nanonis 之外全是自管区
    for i, seg in enumerate(parts):
        if seg == "raw" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt != RAW_NANONIS_DIR and nxt in {*RAW_KIND_DIRS.values(), RAW_OTHER_DIR}:
                return True
    return False


def classify(path: str | Path, *, active_sample_dir: Path | None,
             root: Path | None = None) -> Zone:
    """这个路径属于哪个区。永不抛。

    判定顺序刻意是 MANAGED 优先：自管区的判定必须压过一切，否则递归闸门失效。
    """
    try:
        p = Path(path)
        r = Path(root) if root is not None else experiment_root()
    except (OSError, ValueError):
        return Zone.EXTERNAL
    if not is_within(p, r):
        return Zone.EXTERNAL
    if _managed_markers(p):
        return Zone.MANAGED
    if active_sample_dir is not None:
        inplace = Path(active_sample_dir) / "raw" / RAW_NANONIS_DIR
        if is_within(p, inplace):
            return Zone.INPLACE_ACTIVE
    # root 内、非自管区、又不在活跃样品的原位目录下：
    # 多半是用户换了样品但 Nanonis 的 session path 还指着上一个样品。
    # 归属真源是【活跃样品】而不是路径 —— 由调用方复制到活跃样品并告警。
    return Zone.INPLACE_FOREIGN


# ── 原子复制 + 流式哈希 ───────────────────────────────────────────────

def copy_hash_atomic(src: Path, dest: Path, *, chunk: int = _CHUNK) -> tuple[str, int]:
    """把 *src* 复制到 *dest* 并同时算出 sha256。返回 ``(sha256, size)``。

    read chunk → ``sha256.update`` → write，**只读一遍源文件**（``cas.py`` 的
    hash-then-copy 要读两遍；对 200 MB 的 .3ds 这是实打实的一倍 I/O）。

    先写 ``<dest>.part-<pid>-<n>``，成功后 ``os.replace()`` 原子落地 ——
    实验文件夹里**永远不会出现半个文件**；崩溃只留 .part，由
    :func:`sweep_partials` 清掉。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.part-{os.getpid()}-{threading.get_ident():x}")
    h = hashlib.sha256()
    size = 0
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while True:
                buf = fin.read(chunk)
                if not buf:
                    break
                h.update(buf)
                fout.write(buf)
                size += len(buf)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    try:
        shutil.copystat(src, dest)
    except OSError:
        pass  # 时间戳是锦上添花，复制成功才是要紧事
    return h.hexdigest(), size


def sha256_of(path: Path, *, chunk: int = _CHUNK) -> tuple[str, int]:
    """只算摘要不复制（原位登记用）。"""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
            size += len(buf)
    return h.hexdigest(), size


def sweep_partials(root: Path) -> int:
    """删掉遗留的 ``.part-*`` 文件（上次崩溃留下的）。返回删除个数。"""
    n = 0
    try:
        for p in Path(root).rglob("*.part-*"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    except OSError:
        pass
    return n


# ── 落盘器 ────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _settle_budget(suffix: str, *, inplace: bool) -> float:
    table = SETTLE_S_INPLACE if inplace else SETTLE_S
    default = _SETTLE_DEFAULT_INPLACE if inplace else _SETTLE_DEFAULT
    return table.get(str(suffix or "").lower(), default)


class ExperimentFileStore:
    """一个实验文件夹的落盘器。

    单写者持有（ingest worker 线程），不做跨进程锁。sha 索引三层：
    内存 set → ``.mast/sha256.idx`` → ``raw/_manifest.jsonl``（权威，可重建索引）。
    """

    def __init__(self, exp_dir: str | Path) -> None:
        self.exp_dir = Path(exp_dir)
        self._sha: set[str] | None = None
        self._lock = threading.RLock()
        # 已经记过"第二次目击"的 (sha256, source)。没有它，一个 watcher 反复
        # 发现同一个文件（seen 丢失 / 首次启动 / seen 被裁剪）会每一跳都往
        # manifest 追一行 duplicate，把台账淹掉。第一次目击仍然记录 —— 那是
        # 有信息量的（同一份字节被另一条路径又看到了一次）。
        self._dup_noted: set[tuple[str, str]] = set()

    # ── sha 索引 ──────────────────────────────────────────────────────

    @property
    def _idx_path(self) -> Path:
        return self.exp_dir / ".mast" / "sha256.idx"

    def _load_index(self) -> set[str]:
        if self._sha is not None:
            return self._sha
        seen: set[str] = set()
        try:
            with open(self._idx_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.split("\t", 1)[0].strip()
                    if s:
                        seen.add(s)
        except FileNotFoundError:
            seen = self._rebuild_index()
        except OSError as exc:
            logger.warning("sha256.idx unreadable (%s) — rebuilding from manifests", exc)
            seen = self._rebuild_index()
        self._sha = seen
        return seen

    def _rebuild_index(self) -> set[str]:
        """从各样品的 ``raw/_manifest.jsonl`` 重建 sha 索引。

        manifest 是权威：索引丢了可以重建，manifest 丢了才是真丢。
        """
        seen: set[str] = set()
        try:
            for mf in (self.exp_dir / "samples").glob("*/raw/_manifest.jsonl"):
                for row in read_manifest(mf):
                    s = str(row.get("sha256") or "")
                    if s:
                        seen.add(s)
        except OSError:
            pass
        return seen

    def _remember(self, sha: str, rel_path: str) -> None:
        with self._lock:
            idx = self._load_index()
            if sha in idx:
                return
            idx.add(sha)
            try:
                self._idx_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._idx_path, "a", encoding="utf-8") as f:
                    f.write(f"{sha}\t{rel_path}\n")
                    f.flush()
            except OSError as exc:
                logger.debug("sha256.idx append failed (%s) — manifest still authoritative", exc)

    def has_sha(self, sha256: str) -> bool:
        with self._lock:
            return sha256 in self._load_index()

    # ── 落盘 ──────────────────────────────────────────────────────────

    def ingest(
        self,
        src: str | Path,
        *,
        sample_dir_name: str,
        source: str,
        action_id: str | None = None,
        skill: str = "",
        copy_mode: str = "copy",
        zone: Zone | None = None,
        active_sample_dir: Path | None = None,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        settle_s: float | None = None,
    ) -> IngestResult:
        """把 *src* 收进 ``samples/<sample_dir_name>/raw/``。

        ``source`` ∈ ``skill|manual|migrated|import|inplace|stray``。
        ``copy_mode`` ∈ ``copy|hardlink|verify``（``verify`` = 原位登记，不搬字节）。

        **绝不抛异常** —— 扫描流程的正确性不能依赖磁盘有空间。失败返回
        ``disposition="failed"`` 并带上原因，调用方据此把 ``current_path`` 记成
        origin（记录必须活下来，哪怕副本没成）。
        """
        p = Path(src)
        try:
            return self._ingest_inner(
                p, sample_dir_name=sample_dir_name, source=source,
                action_id=action_id, skill=skill, copy_mode=copy_mode,
                zone=zone, active_sample_dir=active_sample_dir,
                max_bytes=max_bytes, settle_s=settle_s,
            )
        except Exception as exc:  # noqa: BLE001 — 见 docstring
            logger.warning("ingest failed for %s: %r", p, exc, exc_info=True)
            return IngestResult(origin=p, disposition="failed",
                                reason=f"{type(exc).__name__}: {exc}")

    def _ingest_inner(
        self, p: Path, *, sample_dir_name: str, source: str,
        action_id: str | None, skill: str, copy_mode: str,
        zone: Zone | None, active_sample_dir: Path | None,
        max_bytes: int, settle_s: float | None,
    ) -> IngestResult:
        if not p.is_file():
            return IngestResult(origin=p, disposition="skipped", reason="not_a_file")

        z = zone if zone is not None else classify(p, active_sample_dir=active_sample_dir)
        if z is Zone.MANAGED:
            # 递归闸门。我们自己写出去的副本走到这里就停 —— copy→detect→copy
            # 的环在结构上不成立。
            return IngestResult(origin=p, disposition="skipped",
                                zone=z, reason="managed_area")

        inplace = z is Zone.INPLACE_ACTIVE
        suffix = p.suffix.lower()
        budget = settle_s if settle_s is not None else _settle_budget(suffix, inplace=inplace)

        # ① 静置门 + ② 尺寸稳定：拿一次基线，等下复制完再比。
        stable = _stat_stable(p, budget)
        if stable is None:
            return IngestResult(origin=p, disposition="retry", zone=z,
                                reason="still_being_written")
        size0, mtime0 = stable

        if size0 <= 0:
            return IngestResult(origin=p, disposition="retry", zone=z, reason="empty_file")
        if size0 > max_bytes:
            return IngestResult(origin=p, disposition="skipped", zone=z,
                                size_bytes=size0, reason="over_max_bytes")

        sample_root = self.exp_dir / "samples" / sample_dir_name

        # 原位模式：不搬字节，只算摘要并登记。
        if inplace or copy_mode == "verify":
            sha, size = sha256_of(p)
            after = _stat_pair(p)
            if after != (size0, mtime0):
                # ③ 登记后重校：算摘要途中文件又变了 → 丢弃本次，下跳重试。
                # 未登记的文件对下游不可见（resolve() 查不到），这是天然闸门。
                return IngestResult(origin=p, disposition="retry", zone=z,
                                    reason="changed_during_hash")
            rel = _rel_posix(p, self.exp_dir)
            self._remember(sha, rel)
            self._append_manifest(sample_root, {
                "sha256": sha, "size": size, "rel_path": rel,
                "origin_path": str(p), "source": source or "inplace",
                "action_id": action_id or "", "skill": skill,
                "ingested_at": _now_iso(), "origin_mtime": mtime0,
                "inplace": True,
            })
            return IngestResult(origin=p, disposition="inplace", sha256=sha,
                                size_bytes=size, dest=p, rel_path=rel, zone=z)

        # 常规复制路径。先预检空间：宁可不复制也不能把盘写满导致扫描无处落脚。
        free = _free_bytes(self.exp_dir)
        if free is not None and free < size0 * 2 + _MIN_FREE_BYTES:
            return IngestResult(origin=p, disposition="failed", zone=z, size_bytes=size0,
                                reason="disk_low")

        dest_dir = _raw_dir_for(sample_root, suffix)
        dest = _resolve_dest(dest_dir, p.name)

        if copy_mode == "hardlink":
            sha, size, linked = _try_hardlink(p, dest)
            if not linked:
                sha, size = copy_hash_atomic(p, dest)
        else:
            sha, size = copy_hash_atomic(p, dest)

        # ③ 复制后重校 —— 唯一能抓住"复制途中 Nanonis 还在写"的手段。
        after = _stat_pair(p)
        if after != (size0, mtime0):
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            return IngestResult(origin=p, disposition="retry", zone=z,
                                reason="changed_during_copy")

        rel = _rel_posix(dest, self.exp_dir)

        # 去重：同一份字节已经在本实验里了 —— 仍然往 manifest 追一行记录这次
        # 目击（source 可能不同），但不产生第二份副本。
        if self.has_sha(sha) and _has_other_copy(self.exp_dir, sha, rel):
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            existing = _find_rel_by_sha(self.exp_dir, sha) or ""
            dup_key = (sha, source)
            with self._lock:
                first_sighting = dup_key not in self._dup_noted
                self._dup_noted.add(dup_key)
            if first_sighting:
                self._append_manifest(sample_root, {
                    "sha256": sha, "size": size, "rel_path": existing,
                    "origin_path": str(p), "source": source,
                    "action_id": action_id or "", "skill": skill,
                    "ingested_at": _now_iso(), "origin_mtime": mtime0,
                    "duplicate_of": existing,
                })
            return IngestResult(origin=p, disposition="duplicate", sha256=sha,
                                size_bytes=size, dest=self.exp_dir / existing if existing else None,
                                rel_path=existing, zone=z, reason="sha_already_present")

        self._remember(sha, rel)
        row = {
            "sha256": sha, "size": size, "rel_path": rel,
            "origin_path": str(p), "source": source,
            "action_id": action_id or "", "skill": skill,
            "ingested_at": _now_iso(), "origin_mtime": mtime0,
        }
        if z is Zone.INPLACE_FOREIGN:
            row["note"] = "Nanonis session path 未随样品切换而更新"
        self._append_manifest(sample_root, row)
        return IngestResult(origin=p, disposition="new", sha256=sha, size_bytes=size,
                            dest=dest, rel_path=rel, zone=z)

    # ── manifest ──────────────────────────────────────────────────────

    def _append_manifest(self, sample_root: Path, row: dict) -> None:
        append_manifest(sample_root / "raw" / "_manifest.jsonl", row)

    def note_stray(self, stray_sample_dir_name: str, row: dict) -> None:
        """在**源所在样品**的 manifest 里记一行 ``source="stray"``。

        场景：用户把 Nanonis 指向 S01，然后开始做 S02，Nanonis 仍往 S01 的
        目录写。副本按活跃样品落到 S02（归属真源是活跃样品，不是路径），但
        S01 的台账**不能静默地包含一个不属于它的文件** —— 所以这里补一行，
        指明它实际归属谁。
        """
        try:
            self._append_manifest(self.exp_dir / "samples" / stray_sample_dir_name,
                                  {**row, "source": "stray"})
        except OSError as exc:
            logger.debug("stray note failed: %r", exc)

    # ── 校验 ──────────────────────────────────────────────────────────

    def verify_folder(self, *, deep: bool = False) -> dict:
        """按 manifest 随时重算整目录的完整性。

        **不产出任何封存文件**（没有 MANIFEST.sha256 那种东西）—— 实验永不归档，
        校验是一个随时可跑的动作，不是收尾步骤。``deep=True`` 时重算 sha256。
        """
        out = {"files": 0, "missing": [], "size_mismatch": [], "sha_mismatch": []}
        for mf in (self.exp_dir / "samples").glob("*/raw/_manifest.jsonl"):
            for row in read_manifest(mf):
                rel = str(row.get("rel_path") or "")
                if not rel or row.get("duplicate_of"):
                    continue
                out["files"] += 1
                p = self.exp_dir / rel
                if not p.is_file():
                    out["missing"].append(rel)
                    continue
                try:
                    if int(row.get("size") or 0) != p.stat().st_size:
                        out["size_mismatch"].append(rel)
                        continue
                except OSError:
                    out["missing"].append(rel)
                    continue
                if deep:
                    try:
                        got, _ = sha256_of(p)
                        if got != row.get("sha256"):
                            out["sha_mismatch"].append(rel)
                    except OSError:
                        out["missing"].append(rel)
        return out


# ── manifest 读写（模块级，迁移/重建也要用） ─────────────────────────

def append_manifest(path: Path, row: dict) -> None:
    """往 jsonl 追一行。单写者，完整 JSON + ``\\n`` + flush。

    崩溃最多丢最后一行 —— 读取端 :func:`read_manifest` 会丢弃不可解析的尾行，
    索引可从剩余行重建。这是 INCREMENTAL-ONLY 的落地形式之一。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except OSError as exc:
        logger.warning("manifest append failed (%s): %r", path, exc)


def read_manifest(path: Path) -> list[dict]:
    """读 jsonl，**丢弃不可解析的行**（崩溃留下的半行）。永不抛。"""
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue      # 尾部半行 —— 正常，不是错误
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError:
        return rows
    return rows


# ── 路径解析 ──────────────────────────────────────────────────────────

def resolve(sha256: str, exp_dir: str | Path) -> Path | None:
    """按 sha 在实验文件夹里找规范副本。找不到返回 None。"""
    rel = _find_rel_by_sha(Path(exp_dir), sha256)
    if not rel:
        return None
    p = Path(exp_dir) / rel
    return p if p.is_file() else None


def _find_rel_by_sha(exp_dir: Path, sha256: str) -> str | None:
    try:
        with open(exp_dir / ".mast" / "sha256.idx", "r", encoding="utf-8") as f:
            for line in f:
                s, _, rel = line.partition("\t")
                if s.strip() == sha256:
                    return rel.strip()
    except OSError:
        pass
    for mf in (exp_dir / "samples").glob("*/raw/_manifest.jsonl"):
        for row in read_manifest(mf):
            if row.get("sha256") == sha256 and row.get("rel_path"):
                return str(row["rel_path"])
    return None


def _has_other_copy(exp_dir: Path, sha256: str, exclude_rel: str) -> bool:
    """除了 *exclude_rel* 之外，本实验里还有没有这份字节的副本。"""
    rel = _find_rel_by_sha(exp_dir, sha256)
    if not rel or rel == exclude_rel:
        return False
    return (exp_dir / rel).is_file()


# ── 小工具 ────────────────────────────────────────────────────────────

def _rel_posix(p: Path, base: Path) -> str:
    try:
        return Path(p).relative_to(base).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _raw_dir_for(sample_root: Path, suffix: str) -> Path:
    kind = RAW_KIND_DIRS.get(str(suffix or "").lower(), RAW_OTHER_DIR)
    return sample_root / "raw" / kind


def _resolve_dest(dest_dir: Path, name: str) -> Path:
    """目标路径。同名不同内容 → ``<stem>.<sha8><ext>``。

    命名规则与 ``agents/paper_writing/tools.py::_resolve_figure_dest`` 同源
    （那边是 figures 的同名冲突处理）。

    ``raw/`` 内**保留 Nanonis 原始 basename**，不用 ``files/<sha256>`` 那套：
    用户认得 ``Au111_mica_001.sxm``，而且 ``scan_registry._derive_scan_id``
    用的就是文件 stem，改名会打断 ``scan_id → path`` 的解析。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if not dest.exists():
        return dest
    # 同名已存在：让调用方先算 sha 再定夺代价太大，这里直接用一个基于内容的
    # 后备名；真正的去重在 ingest 里按 sha 判定。
    p = Path(name)
    n = 1
    while True:
        cand = dest_dir / f"{p.stem}__{n:03d}{p.suffix}"
        if not cand.exists():
            return cand
        n += 1


def _stat_pair(p: Path) -> tuple[int, int] | None:
    try:
        st = p.stat()
        return st.st_size, st.st_mtime_ns
    except OSError:
        return None


def _stat_stable(p: Path, settle_s: float) -> tuple[int, int] | None:
    """静置门 + 尺寸稳定。返回 ``(size, mtime_ns)``，未稳定返回 None。

    ``age < 0``（mtime 在未来）不算"还在写"：时钟不同步、从别的机器拷进来的
    文件、NAS 的时间偏移都会造成未来时间戳。按字面判定的话这种文件**永远**
    过不了静置门，会被无限重试直到放弃 —— 一次真实的测量就这样丢了。
    """
    pair = _stat_pair(p)
    if pair is None:
        return None
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        return None
    if 0 <= age < settle_s:
        return None
    return pair


def _free_bytes(path: Path) -> int | None:
    try:
        probe = path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        return shutil.disk_usage(str(probe)).free
    except (OSError, ValueError):
        return None


def _try_hardlink(src: Path, dest: Path) -> tuple[str, int, bool]:
    """同卷硬链接（省盘）。失败返回 ``linked=False`` 让调用方走复制。

    降级逻辑与 ``cas.py:117-121`` 同源。
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(str(src), str(dest))
    except OSError:
        return "", 0, False
    try:
        sha, size = sha256_of(dest)
        return sha, size, True
    except OSError:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return "", 0, False
