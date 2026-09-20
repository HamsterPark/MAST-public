"""Open the v2 records store for LIVE writes from the running GUI.

The v2 records DB (ExperimentStoreV2) was previously written only by the
build-time migration; the live app wrote v1 only. ``open_live_v2`` builds the v2
repos and ensures a default campaign → sample → experiment so every live skill
run can land as a v2 action (the Records tab reads this store). Best-effort:
returns (None, None) on any failure so the GUI never blocks on it.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def open_live_v2(*, store: Any | None = None) -> tuple[Any, str | None]:
    """Return ``(repos, experiment_id)`` for live v2 records, or ``(None, None)``.

    Reuses the most-recent campaign if any (so sessions don't scatter), then
    creates a fresh session sample + experiment. ``store`` is injectable for
    tests; default resolves ``$MAST_DATA_DIR/experiments/mast_experiments_v2.db``.
    """
    try:
        from mast.logging.v2.repos import build_repos
        if store is None:
            from mast.logging.v2.storage import open_store
            store = open_store()
        repos = build_repos(store)

        cid = None
        try:
            cams = repos.campaigns.list(limit=1)
            if cams:
                cid = cams[0].get("id")
        except Exception:
            cid = None
        if not cid:
            cid = repos.campaigns.create(
                title="MAST 实时记录", hypothesis="",
                hypothesis_kind="exploratory", goal={}, created_by="gui")

        # Reuse an existing OPEN session experiment rather than minting a fresh
        # "实时会话" (+ "session" sample) on EVERY boot — the old code created a
        # new pair each start, so every restart accumulated an empty synthetic
        # experiment and all live records piled onto the latest one (review
        # 2026-07-03). Only create when there's no open session to reuse.
        eid = None
        try:
            for ex in (repos.experiments.list_open() or []):
                if (ex.get("title") == "实时会话"
                        or ex.get("exp_type") == "ad_hoc"):
                    eid = ex.get("id")
                    break
        except Exception:  # noqa: BLE001
            eid = None
        if not eid:
            sid = repos.samples.create(label="session", material="")
            eid = repos.experiments.start(
                campaign_id=cid, sample_id=sid, title="实时会话", exp_type="ad_hoc")
            logger.info("live v2 records open: NEW session experiment %s",
                        str(eid)[:8])
        else:
            logger.info("live v2 records open: reusing session experiment %s",
                        str(eid)[:8])
        return repos, eid
    except Exception as exc:
        logger.warning("live v2 records unavailable: %s", exc)
        return None, None


def link_scope(repos: Any, *, v1_exp: dict, v1_sample: dict | None,
               storage: Any = None) -> tuple[str | None, str | None, str | None]:
    """把 v1 的当前实验/样品钉到 v2 的实体上。

    返回 ``(campaign_id, v2_sample_id, v2_experiment_id)``，全部 best-effort。

    映射（2026-07-28）::

        v1 实验  ↔  v2 campaign
        v1 样品  ↔  v2 sample
        v2 experiment  =  该 (campaign, sample) 组合下的会话行

    在此之前 v1 和 v2 完全不联动：``open_live_v2`` 每次开机复用/新建一行叫
    「实时会话」的 experiment，**所有作用域的记录都堆在同一行上**
    （REVIEW_2026-07-03_findings.json:490 已记录）。于是 v2 侧的 campaign
    统计、observations、scan_files 全都分不出这是哪个实验、哪块样品的。

    **永不调用 ``repos.experiments.end()``** —— 实验没有终态，而 v2 的
    append-only 触发器让 ``ended_at`` 只能写一次；一旦写了那一行就再也不能
    被复用，下次切回这个作用域就只能新建，前功尽弃。

    v1 行上的 ``v2_campaign_id`` / ``v2_sample_id`` 是缓存：命中就复用，
    保证同一个 v1 实验永远对应同一个 v2 campaign。
    """
    if repos is None or not v1_exp:
        return None, None, None
    eid_v1 = str(v1_exp.get("id") or "")
    try:
        # ── campaign（v1 实验）
        cid = (v1_exp.get("v2_campaign_id") or "").strip() or None
        if cid and not repos.campaigns.get(cid):
            cid = None
        if not cid:
            cid = repos.campaigns.create(
                title=v1_exp.get("name") or "(未命名实验)",
                hypothesis=v1_exp.get("goal_text") or "",
                hypothesis_kind="exploratory", goal={}, created_by="gui")
            _remember(storage, "experiments", "v2_campaign_id", eid_v1, cid)

        # ── sample（v1 样品）。v1 没有样品时用一个占位 —— v2 的
        #    experiments.sample_id 是 NOT NULL，绕不过。
        sid_v1 = str((v1_sample or {}).get("id") or "")
        sid = ((v1_sample or {}).get("v2_sample_id") or "").strip() or None
        if sid and not repos.samples.get(sid):
            sid = None
        if not sid:
            label = (v1_sample or {}).get("name") or "未指定样品"
            sid = repos.samples.create(
                label=label,
                material=(v1_sample or {}).get("sample_type") or "",
                prep_method=(v1_sample or {}).get("description") or None)
            if sid_v1:
                _remember(storage, "samples", "v2_sample_id", sid_v1, sid)

        # ── experiment = 该 (campaign, sample) 下的会话行，复用而不是新建
        v2_eid = _find_scope_experiment(repos, cid, sid)
        if not v2_eid:
            v2_eid = repos.experiments.start(
                campaign_id=cid, sample_id=sid,
                title=_scope_title(v1_exp, v1_sample), exp_type="ad_hoc")
            logger.info("v2 scope linked: NEW experiment %s for %s / %s",
                        str(v2_eid)[:8], v1_exp.get("name"),
                        (v1_sample or {}).get("name") or "(无样品)")
        return cid, sid, v2_eid
    except Exception as exc:  # noqa: BLE001 — 记账联动失败绝不影响 v1
        logger.warning("v2 scope link failed: %s", exc)
        return None, None, None


def _scope_title(v1_exp: dict, v1_sample: dict | None) -> str:
    name = v1_exp.get("name") or "实验"
    smp = (v1_sample or {}).get("name")
    return f"{name} · {smp}" if smp else name


def _find_scope_experiment(repos: Any, campaign_id: str, sample_id: str) -> str | None:
    """该 (campaign, sample) 组合下已存在的会话行。"""
    try:
        for ex in (repos.experiments.list_open() or []):
            if (ex.get("campaign_id") == campaign_id
                    and ex.get("sample_id") == sample_id):
                return ex.get("id")
    except Exception:  # noqa: BLE001
        pass
    return None


def _remember(storage: Any, table: str, column: str, row_id: str, value: str) -> None:
    """把 v2 的 id 缓存回 v1 行上。失败无所谓 —— 下次重算即可。"""
    if storage is None or not row_id:
        return
    try:
        with storage._connect() as conn:
            conn.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?",
                         (value, row_id))
    except Exception:  # noqa: BLE001
        pass


__all__ = ["open_live_v2", "link_scope"]
