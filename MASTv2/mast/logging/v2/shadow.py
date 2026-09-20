"""Shadow-write adapter that fans a single high-level write into both the
vendored v1 ExperimentStorage and the new ExperimentStoreV2.

Per compass §7 Phase 1, this is the strangler-fig interface for the
3–4-week migration window when both schemas are written to and the v1
schema continues to back the daily GUI. After the read cutover (Phase 2)
the v1 writes can be torn out.

This module avoids the ambition of "full type-checked translation" — it
just maps the v1 ActionRecord and friends into the v2 entity model where
the semantics line up, and emits a structured warning on divergence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mast.logging.v2.repos import V2Repos
from mast.logging.v2.ulid import ulid_now

logger = logging.getLogger(__name__)


@dataclass
class ShadowConfig:
    """Tuning for the shadow-write layer."""
    write_v1: bool = True
    write_v2: bool = True
    warn_on_divergence: bool = True
    default_campaign_title: str = "MAST default campaign"
    default_campaign_hypothesis: str = "Catch-all for ad-hoc experiments"
    default_sample_label: str = "Unspecified sample"
    default_sample_material: str = "unknown"


class ShadowLogger:
    """Routes high-level lifecycle calls to v1 + v2 in lockstep.

    ``v1_log`` is an instance of the legacy ``mast.logging.experiment_log.ExperimentLog``.
    ``v2_repos`` is the ``V2Repos`` bundle built by ``repos.build_repos``.
    Both are optional — set to None to disable that side (driven by ``cfg``).
    """

    def __init__(self, v1_log: Any | None, v2_repos: V2Repos | None, cfg: ShadowConfig | None = None):
        self.v1 = v1_log
        self.v2 = v2_repos
        self.cfg = cfg or ShadowConfig()
        self._default_campaign_id: str | None = None
        self._default_sample_id: str | None = None
        self._v1_to_v2_experiment: dict[str, str] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start_experiment(
        self,
        *,
        name: str,
        goal: str = "",
        campaign_id: str | None = None,
        sample_id: str | None = None,
        sample_label: str | None = None,
        sample_material: str | None = None,
        exp_type: str = "ad_hoc",
    ) -> dict[str, str]:
        """Start an experiment on both sides.

        Returns ``{'v1': v1_id, 'v2': v2_id}`` with whichever sides are enabled.
        """
        result: dict[str, str] = {}
        if self.cfg.write_v1 and self.v1 is not None:
            v1_id = self.v1.start_experiment(name, goal)
            result["v1"] = v1_id
        if self.cfg.write_v2 and self.v2 is not None:
            cid = campaign_id or self._ensure_default_campaign()
            sid = sample_id or self._ensure_default_sample(sample_label, sample_material)
            v2_id = self.v2.experiments.start(
                campaign_id=cid, sample_id=sid, title=name, exp_type=exp_type,
            )
            result["v2"] = v2_id
            if "v1" in result:
                self._v1_to_v2_experiment[result["v1"]] = v2_id
        return result

    def end_experiment(
        self,
        *,
        v1_experiment_id: str | None = None,
        v2_experiment_id: str | None = None,
        exit_status: str = "success",
        conclusion: str | None = None,
    ) -> None:
        if self.cfg.write_v1 and self.v1 is not None and v1_experiment_id is not None:
            # v1 ExperimentLog.end_experiment(status) ends the CURRENT experiment
            # by side-effect — it takes only a status, not an id. Passing the id
            # positionally (the old code) raised TypeError on every real call,
            # which is part of why ShadowLogger was never adoptable as a live
            # writer. Map the v2 exit_status vocabulary onto v1's status words.
            v1_status = "completed" if exit_status == "success" else "aborted"
            self.v1.end_experiment(v1_status)
        if self.cfg.write_v2 and self.v2 is not None:
            v2_id = v2_experiment_id or (
                self._v1_to_v2_experiment.get(v1_experiment_id) if v1_experiment_id else None
            )
            if v2_id:
                self.v2.experiments.end(v2_id, exit_status=exit_status, conclusion=conclusion)
            elif self.cfg.warn_on_divergence:
                logger.warning("end_experiment: no v2 mapping for v1=%s", v1_experiment_id)

    # ── Actions ───────────────────────────────────────────────────────

    def log_action(
        self,
        *,
        v1_record: Any = None,
        agent_id: str | None = None,
        action_type: str | None = None,
        params: dict | None = None,
        experiment_id_v2: str | None = None,
    ) -> dict[str, str]:
        """Mirror an action write across the two systems.

        If ``v1_record`` is supplied (a v1 ``ActionRecord``), it is logged on
        the v1 side and a best-effort translation is logged on the v2 side.
        Otherwise the caller passes the structured fields and v1 side is
        skipped.
        """
        result: dict[str, str] = {}
        v1_id_for_v2_ref: str | None = None

        if self.cfg.write_v1 and self.v1 is not None and v1_record is not None:
            self.v1.log_skill_execution(v1_record)
            v1_id_for_v2_ref = getattr(v1_record, "id", None)
            agent_id = agent_id or getattr(v1_record, "approval_source", "unknown")
            action_type = action_type or getattr(v1_record, "skill_name", "unknown")
            params = params or dict(getattr(v1_record, "parameters", {}))
            result["v1"] = v1_id_for_v2_ref

        if self.cfg.write_v2 and self.v2 is not None and experiment_id_v2:
            try:
                aid = self.v2.actions.begin(
                    experiment_id=experiment_id_v2,
                    agent_id=agent_id or "unknown",
                    action_type=action_type or "unknown",
                    params=params or {},
                )
                # If a v1 record came in with state_before/state_after, treat
                # state_after.bias_v / current_a / z_pos_m as scalar observations.
                if v1_record is not None:
                    self._translate_v1_state_to_observations(
                        v1_record, action_id=aid, experiment_id=experiment_id_v2,
                    )
                if v1_record is not None and getattr(v1_record, "result", None):
                    res = v1_record.result
                    if res.success:
                        self.v2.actions.succeed(aid, duration_ms=int(res.elapsed_s * 1000))
                    else:
                        self.v2.actions.fail(aid, res.error or "unknown",
                                             duration_ms=int(res.elapsed_s * 1000))
                else:
                    self.v2.actions.succeed(aid)
                result["v2"] = aid
            except Exception as exc:
                if self.cfg.warn_on_divergence:
                    logger.warning("v2 shadow write failed: %s", exc, exc_info=True)
        return result

    # ── Internals ─────────────────────────────────────────────────────

    def _ensure_default_campaign(self) -> str:
        if self._default_campaign_id is not None:
            return self._default_campaign_id
        if self.v2 is None:
            raise RuntimeError("v2 repos unavailable")
        existing = self.v2.campaigns.list(limit=1)
        if existing:
            self._default_campaign_id = existing[0]["id"]
            return self._default_campaign_id
        self._default_campaign_id = self.v2.campaigns.create(
            title=self.cfg.default_campaign_title,
            hypothesis=self.cfg.default_campaign_hypothesis,
            hypothesis_kind="exploratory",
            goal={"target_observable": "tbd", "success_criteria": [], "budget_hours": None},
            created_by="shadow_logger",
        )
        return self._default_campaign_id

    def _ensure_default_sample(self, label: str | None, material: str | None) -> str:
        if self._default_sample_id is not None and label is None and material is None:
            return self._default_sample_id
        if self.v2 is None:
            raise RuntimeError("v2 repos unavailable")
        sid = self.v2.samples.create(
            label=label or self.cfg.default_sample_label,
            material=material or self.cfg.default_sample_material,
        )
        if label is None and material is None:
            self._default_sample_id = sid
        return sid

    def _translate_v1_state_to_observations(
        self, v1_record: Any, *, action_id: str, experiment_id: str,
    ) -> None:
        if self.v2 is None:
            return
        state_after = getattr(v1_record, "state_after", None)
        if state_after is None:
            return
        for attr, observable, units in (
            ("bias_v", "bias", "V"),
            ("current_a", "current", "A"),
            ("z_pos_m", "z_position", "m"),
            ("setpoint_a", "setpoint", "A"),
        ):
            val = getattr(state_after, attr, None)
            if val is None:
                continue
            try:
                self.v2.observations.record_scalar(
                    action_id=action_id,
                    experiment_id=experiment_id,
                    observable=observable,
                    scalar_value=float(val),
                    units=units,
                )
            except Exception as exc:
                if self.cfg.warn_on_divergence:
                    logger.debug("skipping observation %s: %s", observable, exc)


# ── Helpers for tests / scripts ───────────────────────────────────────

def build_shadow(
    v1_db_path: str | None,
    v2_db_path: str | None,
    *,
    cfg: ShadowConfig | None = None,
) -> ShadowLogger:
    """Convenience factory: open both DBs and return a wired ShadowLogger."""
    v1_log = None
    v2_repos = None

    if v1_db_path is not None:
        from mast.logging.storage import ExperimentStorage
        from mast.logging.experiment_log import ExperimentLog
        v1_log = ExperimentLog(ExperimentStorage(v1_db_path))

    if v2_db_path is not None:
        from mast.logging.v2.repos import build_repos
        from mast.logging.v2.storage import ExperimentStoreV2
        v2_repos = build_repos(ExperimentStoreV2(v2_db_path))

    return ShadowLogger(v1_log, v2_repos, cfg)
