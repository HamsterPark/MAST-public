"""升级之后，把用户自己攒的文献增量并回大库。

KNOWN_ISSUES §3.1
=================
安装包里带一份出厂 ``literature_index``，安装器用 ``ignoreversion`` 覆盖
``{app}\\MASTv2\\artifacts\\literature_index\\`` —— 而 agent 自主取文写进去的增量
**就在同一处**（``ingest._promote_to_big``：往 ``metadata.parquet`` 追加一行、往
``vectors.npy`` 追加一行向量、``abstracts.parquet`` upsert）。manifest 没有任何
合并逻辑，所以一次升级 = 用户攒的全部文献静默消失。

当前的对策是「升级前手动备份」。**靠记得备份不是方案。**

这个模块 + 安装器里的一段快照，一起把它变成不需要记性的事：

1. ``mast2_setup.iss`` 在覆盖之前，把现有 ``literature_index\\`` 整个 **复制** 成
   ``literature_index.pre-upgrade-<时间戳>\\``（复制不是移动 —— 任何一刻机器都不
   能处于「没有索引」的状态，哪怕合并没跑成）。
2. 启动时 ``merge_pending_snapshots()`` 把快照里**新库没有的那些行**并回去，
   然后把快照改名成 ``.merged-<时间戳>``。

快照怎么处理才既不丢东西又不涨盘
================================
合并结果为 **0 行新增** 的快照按定义是冗余的 —— 它里面的每一行新库都有 —— 所以
删掉它不可能丢任何字节的信息，这条才是「绝不丢字节」的正确读法。真加了行的快照
一律保留，并把路径写进日志：那是唯一一份「合并之前长什么样」的证据。

合并规则（保守，且偏向出厂数据）
================================
* ``metadata.parquet`` / ``vectors.npy`` 行行对齐（第 i 行元数据 ↔ 第 i 行向量）。
  以**新库为基**，只追加快照里 ``(work_id, kind)`` 在新库中不存在的行。
  ``kind`` 必须进 key：``_promote_to_big`` 对已存在的 work_id 会再追加一行
  ``kind="user_abstract"``（Decision 2「两份都留」），按 work_id 去重会把它吃掉。
* 向量维度不一致 → **整条向量合并放弃并大声报**。把 1024 维和别的维度拼在一起
  会得到一个能加载、检索结果却是垃圾的索引 —— 比合并失败糟得多。
* ``abstracts.parquet`` 按 work_id 逐字段合并：新库有值的字段不动，只把新库为空
  而快照有值的字段填回去（用户摘要 / 全文摘录就是这样活下来的）。
* ``classified.parquet`` 只补新库没有的 work_id，绝不覆盖出厂分类。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 安装器写下的快照目录前缀（``mast2_setup.iss`` 与这里是同一个字面量）。
SNAPSHOT_PREFIX = "literature_index.pre-upgrade-"
#: 合并完成后改成这个前缀 —— 已处理，且还留在盘上当证据。
MERGED_PREFIX = "literature_index.merged-"

_SNAPSHOT_RE = re.compile(re.escape(SNAPSHOT_PREFIX) + r"(.+)$")

#: 用户增量能落在哪些字段上（abstracts.parquet 的逐字段合并只回填这些）。
_USER_ABSTRACT_FIELDS = (
    "abstract", "user_abstract", "fulltext_excerpt", "abstract_provenance",
    "source", "year", "doi", "title",
)


@dataclass
class MergeReport:
    """一次合并实际做了什么。字段都是**观察**，不是意图。"""

    snapshot: Path
    index_dir: Path
    rows_added: int = 0
    abstracts_updated: int = 0
    classified_added: int = 0
    restored_wholesale: bool = False
    errors: list[str] = field(default_factory=list)
    snapshot_kept: bool = True

    @property
    def ok(self) -> bool:
        return not self.errors

    def describe(self) -> str:
        if self.restored_wholesale:
            return f"新库缺失，已从快照整份恢复（{self.snapshot.name}）"
        bits = [f"并回 {self.rows_added} 条文献"]
        if self.abstracts_updated:
            bits.append(f"补回 {self.abstracts_updated} 条摘要字段")
        if self.classified_added:
            bits.append(f"补回 {self.classified_added} 条分类")
        if self.errors:
            bits.append("错误：" + "；".join(self.errors))
        bits.append("快照已保留" if self.snapshot_kept else "快照零新增，已删除")
        return "；".join(bits)


def _index_dir() -> Path:
    from mast.knowledge.literature_index import _INDEX_DIR

    return Path(_INDEX_DIR)


def find_snapshots(index_dir: Path | None = None) -> list[Path]:
    """待合并的升级快照，按名字（＝时间戳）排序。"""
    d = Path(index_dir) if index_dir is not None else _index_dir()
    parent = d.parent
    if not parent.is_dir():
        return []
    out = [p for p in parent.iterdir()
           if p.is_dir() and p.name.startswith(SNAPSHOT_PREFIX)]
    return sorted(out, key=lambda p: p.name)


def _key_series(meta: Any) -> list[tuple[str, str]]:
    """``(work_id, kind)`` —— kind 必须进 key，见模块 docstring。"""
    if "work_id" not in meta.columns:
        return []
    ids = meta["work_id"].astype(str).tolist()
    if "kind" in meta.columns:
        kinds = meta["kind"].astype(str).tolist()
    else:
        kinds = [""] * len(ids)
    return list(zip(ids, kinds))


def _aligned_len(meta: Any, vec: Any) -> int:
    """行对齐的有效长度。写入是「先 meta 后 vec」，崩在中间会留下多的 meta 行。"""
    return min(len(meta), int(getattr(vec, "shape", (0,))[0]))


def _merge_vectors_and_metadata(
    index_dir: Path, snap: Path, report: MergeReport
) -> None:
    import numpy as np
    import pandas as pd

    from mast.knowledge.ingest import _atomic_write_npy, _atomic_write_parquet

    base_meta_p = index_dir / "metadata.parquet"
    base_vec_p = index_dir / "vectors.npy"
    snap_meta_p = snap / "metadata.parquet"
    snap_vec_p = snap / "vectors.npy"

    if not (snap_meta_p.exists() and snap_vec_p.exists()):
        report.errors.append("快照里没有 metadata.parquet / vectors.npy")
        return

    snap_meta = pd.read_parquet(snap_meta_p)
    snap_vec = np.load(snap_vec_p)
    n_snap = _aligned_len(snap_meta, snap_vec)

    if not (base_meta_p.exists() and base_vec_p.exists()):
        # 升级没带索引（或索引没装上）。整份放回去 —— 这不是合并，是恢复。
        shutil.copy2(snap_meta_p, base_meta_p)
        shutil.copy2(snap_vec_p, base_vec_p)
        for extra in ("abstracts.parquet", "classified.parquet", "manifest.json"):
            src = snap / extra
            if src.exists() and not (index_dir / extra).exists():
                shutil.copy2(src, index_dir / extra)
        report.restored_wholesale = True
        report.rows_added = n_snap
        return

    base_meta = pd.read_parquet(base_meta_p)
    base_vec = np.load(base_vec_p)
    n_base = _aligned_len(base_meta, base_vec)

    if base_vec.ndim != 2 or snap_vec.ndim != 2 or \
            base_vec.shape[1] != snap_vec.shape[1]:
        # 维度不同的向量拼起来会得到一个「加载得了、检索是垃圾」的索引。
        report.errors.append(
            f"向量维度不一致（新库 {getattr(base_vec, 'shape', None)} vs 快照 "
            f"{getattr(snap_vec, 'shape', None)}）—— 未合并任何向量"
        )
        return

    have = set(_key_series(base_meta.iloc[:n_base]))
    snap_keys = _key_series(snap_meta.iloc[:n_snap])
    add_idx = [i for i, k in enumerate(snap_keys) if k not in have]
    if not add_idx:
        return

    add_meta = snap_meta.iloc[add_idx]
    add_vec = snap_vec[add_idx]
    new_meta = pd.concat([base_meta.iloc[:n_base], add_meta], ignore_index=True)
    new_vec = np.vstack([base_vec[:n_base], add_vec]).astype(base_vec.dtype)

    # 与 ingest 同序：先 meta 后 vec。中途崩溃留下的是「多一行 meta」，
    # 加载端的对齐截断会把它无害地丢掉；反过来则是一行没有元数据的向量。
    _atomic_write_parquet(base_meta_p, new_meta)
    _atomic_write_npy(base_vec_p, new_vec)
    report.rows_added = len(add_idx)


def _merge_abstracts(index_dir: Path, snap: Path, report: MergeReport) -> None:
    import pandas as pd

    from mast.knowledge.ingest import _atomic_write_parquet

    snap_p = snap / "abstracts.parquet"
    if not snap_p.exists():
        return
    base_p = index_dir / "abstracts.parquet"
    snap_df = pd.read_parquet(snap_p)
    if "work_id" not in snap_df.columns:
        return
    if not base_p.exists():
        _atomic_write_parquet(base_p, snap_df)
        report.abstracts_updated = len(snap_df)
        return

    base_df = pd.read_parquet(base_p)
    if "work_id" not in base_df.columns:
        return
    for col in _USER_ABSTRACT_FIELDS:
        if col in snap_df.columns and col not in base_df.columns:
            base_df[col] = ""

    by_id = {str(w): i for i, w in enumerate(base_df["work_id"].astype(str))}
    updated = 0
    new_rows = []
    for _, row in snap_df.iterrows():
        wid = str(row.get("work_id", ""))
        if not wid:
            continue
        idx = by_id.get(wid)
        if idx is None:
            new_rows.append(row)
            updated += 1
            continue
        touched = False
        for col in _USER_ABSTRACT_FIELDS:
            if col not in snap_df.columns or col not in base_df.columns:
                continue
            snap_val = row.get(col)
            if snap_val is None or str(snap_val).strip() in ("", "nan", "0"):
                continue
            base_val = base_df.at[base_df.index[idx], col]
            # 只回填新库为空的字段 —— 出厂值优先，用户增量补空。
            if str(base_val or "").strip() in ("", "nan"):
                base_df.at[base_df.index[idx], col] = snap_val
                touched = True
        if touched:
            updated += 1
    if new_rows:
        base_df = pd.concat(
            [base_df, pd.DataFrame(new_rows)], ignore_index=True
        )
    if updated:
        _atomic_write_parquet(base_p, base_df)
    report.abstracts_updated = updated


def _merge_classified(index_dir: Path, snap: Path, report: MergeReport) -> None:
    import pandas as pd

    from mast.knowledge.ingest import _atomic_write_parquet

    snap_p = snap / "classified.parquet"
    if not snap_p.exists():
        return
    base_p = index_dir / "classified.parquet"
    snap_df = pd.read_parquet(snap_p)
    if "work_id" not in snap_df.columns:
        return
    if not base_p.exists():
        _atomic_write_parquet(base_p, snap_df)
        report.classified_added = len(snap_df)
        return
    base_df = pd.read_parquet(base_p)
    if "work_id" not in base_df.columns:
        return
    have = set(base_df["work_id"].astype(str))
    extra = snap_df[~snap_df["work_id"].astype(str).isin(have)]
    if len(extra) == 0:
        return
    # 只补，绝不覆盖出厂分类。
    _atomic_write_parquet(base_p, pd.concat([base_df, extra], ignore_index=True))
    report.classified_added = int(len(extra))


def merge_snapshot(snapshot: Path, index_dir: Path | None = None) -> MergeReport:
    """把一个升级快照并回大库。绝不抛异常 —— 失败写进 report.errors。"""
    d = Path(index_dir) if index_dir is not None else _index_dir()
    report = MergeReport(snapshot=Path(snapshot), index_dir=d)
    try:
        d.mkdir(parents=True, exist_ok=True)
        _merge_vectors_and_metadata(d, Path(snapshot), report)
        _merge_abstracts(d, Path(snapshot), report)
        _merge_classified(d, Path(snapshot), report)
    except Exception as exc:  # noqa: BLE001 — 合并失败绝不能变成启动失败
        logger.exception("literature index snapshot merge failed")
        report.errors.append(f"{type(exc).__name__}: {exc}")
    return report


def _retire(snapshot: Path, report: MergeReport) -> None:
    """合并完的快照：零新增就删（按定义冗余），否则改名保留。"""
    if report.errors:
        report.snapshot_kept = True
        return
    nothing_added = (
        report.rows_added == 0
        and report.abstracts_updated == 0
        and report.classified_added == 0
        and not report.restored_wholesale
    )
    try:
        if nothing_added:
            shutil.rmtree(snapshot, ignore_errors=True)
            report.snapshot_kept = snapshot.exists()
            return
        m = _SNAPSHOT_RE.match(snapshot.name)
        stamp = m.group(1) if m else time.strftime("%Y%m%d-%H%M%S")
        target = snapshot.parent / f"{MERGED_PREFIX}{stamp}"
        if target.exists():
            target = snapshot.parent / f"{MERGED_PREFIX}{stamp}-{int(time.time())}"
        snapshot.rename(target)
        report.snapshot = target
    except OSError as exc:
        logger.warning("could not retire snapshot %s: %s", snapshot, exc)


def merge_pending_snapshots(index_dir: Path | None = None) -> list[MergeReport]:
    """启动时调用：把所有待合并的升级快照并回去。绝不抛异常。"""
    reports: list[MergeReport] = []
    try:
        snapshots = find_snapshots(index_dir)
    except Exception:  # noqa: BLE001
        logger.debug("literature index snapshot scan failed", exc_info=True)
        return reports
    for snap in snapshots:
        report = merge_snapshot(snap, index_dir)
        _retire(snap, report)
        reports.append(report)
        if report.ok:
            level = logger.info if not report.rows_added else logger.warning
            level("文献大库升级合并：%s", report.describe())
        else:
            logger.error("文献大库升级合并**未完成**：%s（快照保留在 %s）",
                         report.describe(), report.snapshot)
        try:
            from mast.core.diagnostics import record as diag_record

            diag_record("note", subject="literature_index_merge",
                        reason=report.describe(),
                        snapshot=str(report.snapshot),
                        rows_added=report.rows_added,
                        errors=report.errors)
        except Exception:  # noqa: BLE001
            logger.debug("merge diagnostics record failed", exc_info=True)
    if reports:
        try:
            from mast.knowledge import literature_index

            literature_index.invalidate_caches()
        except Exception:  # noqa: BLE001
            logger.debug("cache invalidation after merge failed", exc_info=True)
    return reports


def merge_pending_snapshots_async(index_dir: Path | None = None) -> Any:
    """守护线程里跑合并 —— 205 MB 的索引不该记在启动时间上。"""
    import threading

    def _body() -> None:
        try:
            merge_pending_snapshots(index_dir)
        except Exception:  # noqa: BLE001
            logger.debug("async index merge failed", exc_info=True)

    th = threading.Thread(target=_body, name="lit-index-merge", daemon=True)
    th.start()
    return th


def snapshot_manifest(snapshot: Path) -> dict:
    """快照里的 manifest.json（读不到返回 {}）—— 报告用途。"""
    p = Path(snapshot) / "manifest.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "MERGED_PREFIX",
    "SNAPSHOT_PREFIX",
    "MergeReport",
    "find_snapshots",
    "merge_pending_snapshots",
    "merge_pending_snapshots_async",
    "merge_snapshot",
    "snapshot_manifest",
]
