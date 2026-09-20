"""把已有的实验记录迁进实验文件夹布局 —— 只读盘点 → 建壳 → 认领文件。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §14

三个阶段，**每一步都不改动 experiment_root 之外的任何东西**：

* **Stage 0 盘点**（:func:`plan`）：纯只读，输出一份计划 JSON 交用户审阅。
* **Stage 1 建壳**（:func:`apply` 的 ``stage>=1``）：为**有内容的**实验创建目录 +
  元数据。零风险（只新建目录）。
* **Stage 2 认领**（``stage>=2``）：按时间窗把历史文件**复制**进去，记置信度。
  原目录一个字节都不动。

回滚 = 删掉 ``experiment_root``。外面什么都没改过。:func:`rollback` 做得更保守：
只删计划声明创建过、且现在仍与计划一致的路径。

空壳过滤
--------

实验库可能包含尚无动作、标记或对话的空记录；迁移时不能因此批量创建空目录。
0 action + 0 marker + 0 conversation 的实验**不建目录**，只在 ``_index`` 里记
``dir: null`` —— 将来它真有写入时再懒创建。这既不丢数据，又不让顶层被几十个
空文件夹淹掉。

卡在 running 的行不动
---------------------
16 条 2026-07-01 的实验 ``status='running'``。**不自动关闭** —— 没有"关闭"这个
动作了。manifest 如实记录它们本来的样子。
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 认领历史文件时，一个实验的时间窗在它最后一条记录之后再延多久。
_TAIL_HOURS = 12
#: 时间窗向**前**的宽容度。用户常常是先扫了几张图，才想起来在 MAST 里建
#: 实验记录 —— 那几张图在时间上早于实验行，但显然属于它。
_HEAD_MINUTES = 30
#: 「精确」认领的时间容差：动作时间戳 ±这么多秒内的文件算 exact。
_EXACT_TOLERANCE_S = 120
_SUFFIXES = (".sxm", ".dat", ".3ds")


def plan(*, storage: Any, scan_dirs: list[str] | None = None,
         include_empty: bool = False) -> dict:
    """**Stage 0：纯只读盘点。** 不写任何文件（除非调用方自己保存返回值）。

    返回的计划里，每个实验都带上它将得到的目录名、内容规模、以及是否会被
    当作空壳跳过；每个候选文件都带上它将归到哪个实验/样品和判定置信度。
    """
    from mast.core.experiment_paths import (
        experiment_dir_name, experiment_root, sample_dir_name,
    )

    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_root": str(experiment_root()),
        "experiments": [], "files": [], "unclaimed": [],
        "counts": {},
    }

    try:
        exps = storage.list_experiments(limit=10000)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"读取实验列表失败：{exc}"
        return out

    windows: list[tuple] = []
    for e in exps:
        eid = str(e.get("id") or "")
        if not eid:
            continue
        try:
            actions = storage.get_actions(eid) or []
        except Exception:  # noqa: BLE001
            actions = []
        try:
            samples = storage.get_samples(eid) or []
        except Exception:  # noqa: BLE001
            samples = []
        markers = _count(storage, "map_markers", eid)
        convs = _count(storage, "conversation_log", eid)

        content = len(actions) + markers + convs
        empty = content == 0 and not samples
        dir_name = experiment_dir_name(eid, e.get("name") or "",
                                       str(e.get("start_time") or ""))
        rec = {
            "id": eid, "name": e.get("name") or "",
            "start_time": str(e.get("start_time") or ""),
            "legacy_status": e.get("status") or "",
            "dir_name": None if (empty and not include_empty) else dir_name,
            "skipped_as_empty": bool(empty and not include_empty),
            "counts": {"actions": len(actions), "samples": len(samples),
                       "markers": markers, "conversations": convs},
            "samples": [{
                "id": s.get("id"), "name": s.get("name") or "",
                "dir_name": sample_dir_name(i + 1, s.get("name") or "",
                                            str(s.get("id") or "")),
                "index": i + 1,
                "start_time": str(s.get("start_time") or ""),
            } for i, s in enumerate(samples)],
        }
        out["experiments"].append(rec)
        if not rec["skipped_as_empty"]:
            windows.append((rec, _window(e, actions)))

    # 候选文件 → 时间窗归属
    for path in _candidate_files(scan_dirs):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        hit, conf, sample = _attribute(mtime, windows, storage)
        row = {"path": str(path), "mtime": mtime,
               "experiment_id": hit["id"] if hit else None,
               "experiment_dir": hit["dir_name"] if hit else None,
               "sample_dir": sample, "confidence": conf}
        (out["files"] if hit else out["unclaimed"]).append(row)

    out["counts"] = {
        "experiments": len(out["experiments"]),
        "with_folder": sum(1 for r in out["experiments"] if r["dir_name"]),
        "skipped_empty": sum(1 for r in out["experiments"] if r["skipped_as_empty"]),
        "files_claimed": len(out["files"]),
        "files_unclaimed": len(out["unclaimed"]),
    }
    return out


def _count(storage: Any, table: str, eid: str) -> int:
    try:
        with storage._connect() as conn:
            r = conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE experiment_id = ?",
                (eid,)).fetchone()
        return int((r["n"] if r else 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def _window(exp: dict, actions: list) -> tuple[float, float]:
    """实验的时间窗 ``(start, end)``（epoch 秒）。

    没有 ``end_time`` 就用最后一条 action 之后 12 小时 —— 实验没有终态，
    所以"结束时间"只能从内容推断。
    """
    start = _epoch(exp.get("start_time")) or 0.0
    end = _epoch(exp.get("end_time"))
    if end is None:
        last = max((_epoch(a.timestamp if hasattr(a, "timestamp")
                           else (a.get("timestamp") if isinstance(a, dict) else None))
                    or 0.0) for a in actions) if actions else start
        end = max(last, start) + _TAIL_HOURS * 3600
    return start - _HEAD_MINUTES * 60, end


def _epoch(v: Any) -> float | None:
    """ISO 时间戳 → epoch 秒。

    ★ 不带时区的字符串按**本地时间**解释，不是 UTC。``ExperimentStorage`` 写的是
    ``datetime.now().isoformat()``（本地、无 tzinfo），而文件 mtime 是真实 epoch；
    把前者当 UTC 会让两者差出整整一个时区偏移（中国 +8 小时），于是**所有历史
    文件都落在实验时间窗之外，一个都认领不到**。naive datetime 的 ``.timestamp()``
    本来就按本地时区解释，正是我们要的。
    """
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt.timestamp()
    except (TypeError, ValueError):
        return None


def _candidate_files(scan_dirs: list[str] | None) -> list[Path]:
    from mast._runtime_paths import project_root
    from mast.core.experiment_paths import experiment_root, is_within

    roots: list[Path] = []
    for d in (scan_dirs or []):
        if d:
            roots.append(Path(d))
    roots.append(project_root() / "working-sessions")
    try:
        from mast.core.scan_registry import known_scan_dirs
        roots.extend(Path(d) for d in (known_scan_dirs() or []))
    except Exception:  # noqa: BLE001
        pass

    out: list[Path] = []
    seen: set[str] = set()
    er = experiment_root()
    for r in roots:
        if not r.is_dir():
            continue
        # 绝不把实验根自己当成候选源 —— 那会把已经归档好的副本再"迁移"一遍。
        if is_within(r, er):
            continue
        for suf in _SUFFIXES:
            try:
                for p in r.rglob(f"*{suf}"):
                    key = str(p).lower()
                    if key not in seen and p.is_file():
                        seen.add(key)
                        out.append(p)
            except OSError:
                continue
    return out


def _attribute(mtime: float, windows: list, storage: Any):
    """按时间窗把一个文件归到实验/样品。返回 ``(exp_rec, confidence, sample_dir)``。"""
    for rec, (start, end) in windows:
        if not (start <= mtime <= end):
            continue
        conf = "window"
        # exact：±120 秒内有 action → 这个文件几乎肯定是那次操作产生的
        try:
            for a in (storage.get_actions(rec["id"]) or []):
                ts = _epoch(getattr(a, "timestamp", None)
                            or (a.get("timestamp") if isinstance(a, dict) else None))
                if ts and abs(ts - mtime) <= _EXACT_TOLERANCE_S:
                    conf = "exact"
                    break
        except Exception:  # noqa: BLE001
            pass
        sample_dir = None
        for s in rec["samples"]:
            s_start = _epoch(s.get("start_time")) or 0.0
            if s_start <= mtime:
                sample_dir = s["dir_name"]
        if sample_dir is None and rec["samples"]:
            sample_dir = rec["samples"][0]["dir_name"]
        return rec, conf, sample_dir
    return None, "none", None


def apply(plan_obj: dict, *, storage: Any, stage: int = 2,
          dry_run: bool = False) -> dict:
    """执行计划。``stage=1`` 只建壳，``stage=2`` 连历史文件一起认领。

    **幂等**：已经存在的目录不重建，manifest 里已有的 sha 直接跳过。
    """
    from mast.core.experiment_paths import (
        experiment_dir, sample_dir,
    )
    from mast.logging.v2 import manifest as mf
    from mast.logging.v2.filestore import ExperimentFileStore

    res = {"created": 0, "skipped_empty": 0, "files": 0, "duplicates": 0,
           "quarantined": 0, "errors": [], "dry_run": bool(dry_run)}
    index_rows: list[dict] = []

    for rec in plan_obj.get("experiments", []):
        if rec.get("skipped_as_empty"):
            res["skipped_empty"] += 1
            index_rows.append({"id": rec["id"], "dir": None,
                               "title": rec["name"],
                               "reason": "no_content_at_migration"})
            continue
        dir_name = rec.get("dir_name")
        if not dir_name:
            continue
        index_rows.append({"id": rec["id"], "dir": dir_name, "title": rec["name"]})
        if dry_run:
            res["created"] += 1
            continue
        try:
            exp_dir = experiment_dir(dir_name, create=True)
            mf.write_experiment_manifest(
                exp_dir, experiment_id=rec["id"], title=rec["name"],
                created_at=rec.get("start_time", ""), dir_name=dir_name,
                samples=[{"id": s["id"], "name": s["name"], "index": s["index"],
                          "dir_name": s["dir_name"]} for s in rec.get("samples", [])],
                provenance={"migrated_from": {"db": str(getattr(storage, "_db_path", "")),
                                              "row_id": rec["id"]},
                            "legacy_status": rec.get("legacy_status") or ""})
            for s in rec.get("samples", []):
                sp = sample_dir(exp_dir, s["dir_name"], create=True)
                mf.write_sample_manifest(
                    sp, sample_id=s["id"], name=s["name"], experiment_id=rec["id"],
                    created_at=s.get("start_time", ""), dir_name=s["dir_name"],
                    index=s["index"])
                try:
                    storage.set_sample_dir_name(s["id"], s["dir_name"], s["index"])
                except Exception:  # noqa: BLE001
                    pass
            mf.write_readme(exp_dir)
            try:
                storage.set_experiment_dir_name(rec["id"], dir_name)
            except Exception:  # noqa: BLE001
                pass
            res["created"] += 1
        except Exception as exc:  # noqa: BLE001
            res["errors"].append(f"{rec['name']}: {exc}")

    if not dry_run:
        _write_index(index_rows)

    if stage >= 2 and not dry_run:
        stores: dict[str, ExperimentFileStore] = {}
        for f in plan_obj.get("files", []):
            edir, sdir = f.get("experiment_dir"), f.get("sample_dir")
            if not edir or not sdir:
                continue
            try:
                exp_dir = experiment_dir(edir)
                store = stores.get(edir)
                if store is None:
                    store = stores[edir] = ExperimentFileStore(exp_dir)
                r = store.ingest(f["path"], sample_dir_name=sdir,
                                 source="migrated", settle_s=0.0)
                if r.disposition == "new":
                    res["files"] += 1
                elif r.disposition == "duplicate":
                    res["duplicates"] += 1
            except Exception as exc:  # noqa: BLE001
                res["errors"].append(f"{f.get('path')}: {exc}")
        # 认领不到的进隔离区索引（**不复制** —— 原文件仍在原处，
        # 这里只是登记"有这么一批孤儿，等用户认领"）。
        if plan_obj.get("unclaimed"):
            res["quarantined"] = _record_unclaimed(plan_obj["unclaimed"])

    return res


def _write_index(rows: list[dict]) -> None:
    from mast.core.experiment_paths import index_dir
    try:
        d = index_dir(create=True)
        with open(d / "experiments.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("index write failed: %r", exc)


def _record_unclaimed(rows: list[dict]) -> int:
    from mast.core.experiment_paths import quarantine_dir
    try:
        d = quarantine_dir(create=True)
        with open(d / "index.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "origin_path": r.get("path"),
                    "detected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "reason": "no_matching_experiment_window",
                    "claimed": False,
                }, ensure_ascii=False) + "\n")
        return len(rows)
    except OSError:
        return 0


def rollback(plan_obj: dict) -> dict:
    """删掉本次迁移创建的实验目录。**只删计划声明创建过的路径。**

    因为迁移从不移动原文件（只复制），回滚之后 ``experiment_root`` 之外的一切
    与迁移前完全一致。
    """
    from mast.core.experiment_paths import experiment_dir
    out = {"removed": 0, "missing": 0, "errors": []}
    for rec in plan_obj.get("experiments", []):
        dir_name = rec.get("dir_name")
        if not dir_name:
            continue
        p = experiment_dir(dir_name)
        if not p.is_dir():
            out["missing"] += 1
            continue
        try:
            meta = json.loads((p / "experiment.json").read_text(encoding="utf-8"))
            if str(meta.get("id")) != str(rec.get("id")):
                out["errors"].append(f"{dir_name}: id 不匹配，拒绝删除")
                continue
            shutil.rmtree(p)
            out["removed"] += 1
        except FileNotFoundError:
            out["errors"].append(f"{dir_name}: 没有 experiment.json，拒绝删除")
        except Exception as exc:  # noqa: BLE001
            out["errors"].append(f"{dir_name}: {exc}")
    return out


def save_plan(plan_obj: dict) -> Path | None:
    """把计划写进 ``_index/migration_plan_<ts>.json`` 供用户审阅。"""
    from mast.core.experiment_paths import index_dir
    try:
        d = index_dir(create=True)
        ts = str(plan_obj.get("generated_at") or "").replace(":", "").replace("-", "")[:15]
        p = d / f"migration_plan_{ts or 'now'}.json"
        p.write_text(json.dumps(plan_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return p
    except OSError as exc:
        logger.warning("plan save failed: %r", exc)
        return None


def main(argv: "list[str] | None" = None) -> int:
    """``python -m mast.logging.v2.migrate_folders [--stage N] [--apply]``.

    Until 2026-08-04 this module had **zero callers outside one test** — 405
    lines of plan/apply/rollback that nothing could reach, so "把历史 Nanonis
    文件按时间窗认领进实验文件夹" was a feature nobody could run (KNOWN_ISSUES
    §2.11). This is the entry point.

    Default is Stage 0 — the read-only inventory — because that is the step the
    design intends an operator to read before anything is created. ``--apply``
    is required to go further, and it prints the plan path first. There is no
    startup hook: stage 2 COPIES historical files into experiment folders, and a
    copy of that size is a decision, not a launch side effect.

    ISOLATION TRAP (hit while smoke-testing this entry point, 2026-08-04):
    ``MAST2_PROJECT_ROOT`` alone does NOT redirect the output. Stage 0 is
    read-only with respect to the DB, but ``save_plan`` writes under
    ``experiment_root()``, and that resolves to ``<install drive anchor>\\
    MAST-Data\\experiments`` — deliberately outside the repo, so pointing
    ``MAST2_PROJECT_ROOT`` at a temp dir still lands on the REAL data root. To
    run this against a scratch tree set **``MAST_EXPERIMENT_ROOT``** as well.
    (Same "redirect env A, storage reads env B" shape that has polluted real
    user data four times in this repo.)
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m mast.logging.v2.migrate_folders",
        description="把已有实验记录迁进实验文件夹布局（盘点 → 建壳 → 认领文件）。"
                    "认领是复制，原目录一个字节都不动；回滚 = 删掉 experiment_root。",
    )
    ap.add_argument("--apply", action="store_true",
                    help="真正执行（不给就只做 Stage 0 只读盘点）")
    ap.add_argument("--stage", type=int, default=2, choices=(1, 2),
                    help="1 = 只建目录+元数据；2 = 连历史文件一起认领（默认）")
    ap.add_argument("--scan-dir", action="append", dest="scan_dirs",
                    help="历史文件所在目录（可重复；不给则用默认扫描目录）")
    ap.add_argument("--include-empty", action="store_true",
                    help="连 0 action / 0 marker / 0 conversation 的空壳实验也建目录")
    ap.add_argument("--dry-run", action="store_true",
                    help="与 --apply 同用：走完全部判断但不落盘")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # mast.logging.storage, NOT mast.logging.v2.storage — the class that carries
    # list_experiments / get_actions / get_samples / set_*_dir_name lives in the
    # former; the v2 module is the schema initialiser.
    from mast.agents._shared.data_paths import experiment_db_path
    from mast.logging.storage import ExperimentStorage

    storage = ExperimentStorage(str(experiment_db_path()))
    plan_obj = plan(storage=storage, scan_dirs=args.scan_dirs,
                    include_empty=bool(args.include_empty))
    saved = save_plan(plan_obj)
    n_exp = len(plan_obj.get("experiments", []))
    n_files = len(plan_obj.get("files", []))
    print(f"Stage 0 盘点：{n_exp} 个实验，{n_files} 个候选文件")
    if saved is not None:
        print(f"计划已写入：{saved}")
    if not args.apply:
        print("只做了盘点。确认无误后加 --apply 执行（--stage 1 只建壳）。")
        return 0
    res = apply(plan_obj, storage=storage, stage=int(args.stage),
                dry_run=bool(args.dry_run))
    print(f"Stage {args.stage}{'（dry-run）' if args.dry_run else ''}：{res}")
    return 1 if res.get("errors") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
