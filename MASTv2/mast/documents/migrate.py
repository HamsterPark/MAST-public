"""把旧的全局文档导入实验文件夹。一次性、幂等、支持 dry-run。

设计文档：``docs/v2/design/document_and_library_management.md`` §3.12

旧落点（全部在实验之外，与实验没有任何关联）::

    <project_root>/experiments/plans/plan_<id>.md      # 每次覆盖，无版本
    <project_root>/data/drafts/<stem>_vNNN.md          # 版本族由 LLM title 决定
    <project_root>/data/reviews/<stem>_review_vNNN.md  # 族被 draft 版本号切碎
    <project_root>/data/reports/<stem>.html            # 直接覆盖

两条纪律
--------

1. **复制导入，原文件不动。** 迁移不销毁任何东西 —— 导错了还能重来，而且旧路径
   仍然可读（``data_paths`` 的那几个函数保留就是为此）。
2. **幂等靠 ``legacy_stem`` 判重，不靠额外状态文件。** 跑两遍结果一致，第二遍全是
   skipped。计划**额外**看一眼 ``plans.doc_id``：那一列是回写的，而回写失败只记
   error 不回滚，所以单看它会在半成功的那一次之后重复导入（2026-08-02 审计）。
   两把钥匙一起用，半成功的运行会自愈而不是复制。

怎么跑
------

**没有进程内的自动触发** —— 这个模块在生产代码里零调用方（KNOWN_ISSUES §2.11）。
开发树里手动跑::

    python -m mast.documents.migrate --dry-run     # 先看会动什么
    python -m mast.documents.migrate               # 真跑

版本族的还原
------------

``<stem>_v001.md``、``<stem>_v002.md`` … 是**同一份文档的历史**，所以按 stem 归族
导成**一个文档的多个版本**（按版本号升序），而不是 N 个独立文档。这是旧命名方案
唯一还能救回来的结构信息。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from mast.documents.model import normalize_kind
from mast.documents.store import DocumentStore

logger = logging.getLogger(__name__)

_VER_RE = re.compile(r"_v(\d+)$")


def _family(stem: str) -> tuple[str, int]:
    """``report_v003`` → ``("report", 3)``；没有版本后缀的 → ``(stem, 0)``。"""
    m = _VER_RE.search(stem)
    if not m:
        return stem, 0
    return stem[: m.start()], int(m.group(1))


def _group_families(files: list[Path]) -> dict[str, list[tuple[int, Path]]]:
    out: dict[str, list[tuple[int, Path]]] = {}
    for p in files:
        base, v = _family(p.stem)
        out.setdefault(base, []).append((v, p))
    for base in out:
        out[base].sort(key=lambda t: t[0])
    return out


def _existing_legacy_stems(s: DocumentStore) -> set[str]:
    from mast.documents.store import _scan_all
    return {e.meta.legacy_stem for e in _scan_all() if e.meta.legacy_stem}


def migrate_plans(*, plan_store=None, dry_run: bool = False,
                  store: DocumentStore | None = None) -> dict:
    """把 ``experiments/plans/plan_<id>.md`` 导成各自实验的计划文档。

    归位依据是 ``plans`` 表里那一行的 ``experiment_id``；查不到实验就落
    ``_unfiled``（内容不丢，之后可以 claim）。
    """
    s = store or DocumentStore()
    res = {"imported": 0, "skipped": 0, "unfiled": 0, "errors": [], "dry_run": dry_run}

    if plan_store is None:
        try:
            from mast.agents._shared.data_paths import experiment_db_path
            from mast.planning.plan_store import PlanStore
            plan_store = PlanStore(experiment_db_path())
        except Exception as exc:  # noqa: BLE001
            res["errors"].append(f"PlanStore 不可用：{exc!r}")
            return res

    legacy_dir = getattr(plan_store, "_plans_dir", None)
    if not legacy_dir or not Path(legacy_dir).is_dir():
        return res

    # TWO idempotency keys, because either one alone has a hole.
    #
    # ``plans.doc_id`` is the natural key — but it is written by the UPDATE at the
    # BOTTOM of this loop, and that UPDATE only appends to ``errors`` when it
    # fails; it does not roll back the document that was already created. So a
    # run where the write-back failed leaves a document with no doc_id on the plan
    # row, and the next run imports the very same plan a second time.
    #
    # ``legacy_stem`` is what the document itself carries (``legacy_stem=p.stem``
    # below), so it survives that failure — it is the key ``migrate_legacy_
    # markdown`` has always used, which is why that function never had this hole.
    # Checking both means a half-committed run heals instead of duplicating.
    seen_stems = _existing_legacy_stems(s)

    for p in sorted(Path(legacy_dir).glob("plan_*.md")):
        plan_id = p.stem[len("plan_"):]
        try:
            if p.stem in seen_stems:
                res["skipped"] += 1
                continue
            if plan_store.doc_id_for(plan_id):
                res["skipped"] += 1
                continue
            plan = plan_store.load(plan_id)
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            res["errors"].append(f"{p.name}: {exc!r}")
            continue
        if not text.strip():
            res["skipped"] += 1
            continue
        eid = (getattr(plan, "experiment_id", "") or "") if plan else ""
        title = (getattr(plan, "name", "") or "") if plan else ""
        if dry_run:
            res["imported"] += 1
            if not eid:
                res["unfiled"] += 1
            continue
        out = s.save(
            text=text, kind="experiment_plan",
            title=title or f"计划 {plan_id}",
            experiment_id=eid or None,
            created_by="migrated",
            note="从旧全局 experiments/plans/ 迁入（旧格式含进度标记）",
            legacy_stem=p.stem, use_turn_context=False,
        )
        if not out.ok:
            res["errors"].append(f"{p.name}: {out.error}")
            continue
        res["imported"] += 1
        if out.root_kind == "unfiled":
            res["unfiled"] += 1
        try:
            from contextlib import closing
            with closing(plan_store._connect()) as conn, conn:
                conn.execute("UPDATE plans SET doc_id=? WHERE plan_id=?",
                             (out.doc_id, plan_id))
        except Exception as exc:  # noqa: BLE001
            res["errors"].append(f"{p.name}: doc_id 回写失败 {exc!r}")
    return res


def migrate_legacy_markdown(*, dry_run: bool = False,
                            store: DocumentStore | None = None) -> dict:
    """把 ``data/drafts`` 与 ``data/reviews`` 按版本族导成文档。

    这些文件没有实验归属（旧模型里根本没有这个概念），所以全部落 ``_unfiled``
    待认领 —— 不猜。猜错的归属比没有归属更糟：它会让人相信一个错的因果链。
    """
    s = store or DocumentStore()
    res = {"drafts": 0, "reviews": 0, "versions": 0, "skipped": 0,
           "errors": [], "dry_run": dry_run}
    try:
        from mast.agents._shared.data_paths import drafts_dir, reviews_dir
    except Exception as exc:  # noqa: BLE001
        res["errors"].append(f"data_paths 不可用：{exc!r}")
        return res

    seen = _existing_legacy_stems(s)

    for dir_, kind, counter in ((drafts_dir(), "experiment_report", "drafts"),
                                (reviews_dir(), "review", "reviews")):
        if not dir_.is_dir():
            continue
        files = [p for p in sorted(dir_.iterdir()) if p.suffix.lower() == ".md"]
        for base, members in _group_families(files).items():
            if base in seen:
                res["skipped"] += 1
                continue
            if dry_run:
                res[counter] += 1
                res["versions"] += len(members)
                continue
            doc_id = ""
            for v, p in members:
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    res["errors"].append(f"{p.name}: {exc!r}")
                    continue
                if not text.strip():
                    continue
                out = s.save(
                    text=text, kind=normalize_kind(kind), title=base.replace("_", " "),
                    doc_id=doc_id, created_by="migrated",
                    note=f"从旧 {dir_.name}/{p.name} 迁入",
                    legacy_stem=base, use_turn_context=False,
                )
                if not out.ok:
                    res["errors"].append(f"{p.name}: {out.error}")
                    continue
                doc_id = out.doc_id
                res["versions"] += 1
            if doc_id:
                res[counter] += 1
                seen.add(base)
    return res


def migrate_all(*, dry_run: bool = False) -> dict:
    """跑全部迁移。返回汇总；**永不抛**。"""
    s = DocumentStore()
    out: dict = {"dry_run": dry_run}
    try:
        out["plans"] = migrate_plans(dry_run=dry_run, store=s)
    except Exception as exc:  # noqa: BLE001
        out["plans"] = {"errors": [repr(exc)]}
    try:
        out["markdown"] = migrate_legacy_markdown(dry_run=dry_run, store=s)
    except Exception as exc:  # noqa: BLE001
        out["markdown"] = {"errors": [repr(exc)]}
    logger.info("documents.migrate_all(dry_run=%s): %s", dry_run, out)
    return out


def main(argv: "list[str] | None" = None) -> int:
    """``python -m mast.documents.migrate [--dry-run]``.

    A module with no caller is a module nobody can run. This is the entry point;
    it is deliberately NOT wired into startup — the migration creates documents
    (dozens of ``_unfiled`` ones on a rig with history), and a copy-in that size
    is an operator decision, not a side effect of launching the app. Following
    the ``monitoring/commission.py`` / ``vision/scan_prep_commission.py`` pattern
    already used for one-shot maintenance tools in this tree.
    """
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="python -m mast.documents.migrate",
        description="把旧的全局文档（plans / drafts / reviews）导入实验文件夹。"
                    "复制导入，原文件不动；跑两遍结果一致。",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="只报告会导入什么，不写任何东西")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = migrate_all(dry_run=bool(args.dry_run))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    errors = (result.get("plans", {}).get("errors", [])
              + result.get("markdown", {}).get("errors", []))
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
