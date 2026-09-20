"""Post-skill hook for TipShapeWithReadback: render → v2 records → vision buffer.

Producer-side GLUE (NOT a model-facing tool): wrap_skill calls this after a
successful TipShapeWithReadback execute, closed over the GUI's v2 records repos +
buffer handle. It:

  1. renders the z/current dual-curve PNG (mast.data.tip_shape_plot),
  2. registers it in the v2 records store (action → scan_file → observation) so
     the Records tab shows it,
  3. emits a TIP_SHAPE_VERDICT VisionEvent so the Vision Buffer tab surfaces the
     verdict + thumbnail.

Every step is best-effort and guarded — a render/DB/buffer failure must NEVER
turn a successful hardware action into a tool error (wrap_skill also wraps the
whole hook in try/except). The buffer is fetched lazily (it warms up async) and
null-checked; the 'agents never write the buffer' invariant holds because this
runs in the adapter glue, not in the LLM tool body.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SKILL = "TipShapeWithReadback"


def make_tip_shape_post_hook(
    *,
    repos: Any | None,
    experiment_id_getter: Callable[[], str | None],
    buffer_getter: Callable[[], Any],
    artifacts_dir: str | os.PathLike[str],
) -> Callable[[str, dict, Any], dict | None]:
    """Build the post_hook(skill_name, data, ctx) for TipShapeWithReadback.

    ``repos`` is a V2Repos facade (or None to skip records). ``*_getter`` are
    lazy so the experiment id / buffer can come into existence after build time.
    Returns ``{"scan_paths": [png]}`` so the PNG flows into agent state.
    """
    art = Path(artifacts_dir)

    def _hook(skill_name: str, data: dict, ctx: Any) -> dict | None:
        if not isinstance(data, dict):
            return None

        # This success-only hook renders the tip-shape figure, attaches it to the
        # recorded action and emits its vision verdict. General action recording
        # belongs to CoreRuntime._record_v2_action, which sees both successes and
        # failures with their actual parameters.
        if skill_name != _SKILL:
            return None
        eid = experiment_id_getter()
        ind = data.get("indent") or {}
        if not ind:
            return None

        # The action this figure belongs to. wrap_skill records the skill call
        # BEFORE calling us and hands the id over on the context, so the figure
        # attaches to the REAL row; opening our own is the standalone fallback
        # (direct callers / tests), never a second row for the same call.
        aid = getattr(ctx, "records_action_id", None)
        if not aid and repos is not None and eid:
            try:
                aid = repos.actions.begin(
                    experiment_id=eid, agent_id="instrument_control",
                    action_type=skill_name, params={})
                repos.actions.succeed(aid)
            except Exception as exc:
                logger.warning("v2 action write failed for %s: %s", skill_name, exc)
                aid = None

        try:
            from mast.data.tip_shape_plot import render_tip_shape_readback
            art.mkdir(parents=True, exist_ok=True)
            png = str(art / f"tip_shape_{uuid.uuid4().hex[:12]}.png")
            render_tip_shape_readback(data, png)
        except Exception as exc:  # nothing to show — bail, but don't raise
            logger.warning("tip_shape render failed: %s", exc)
            return None

        verdict = ind.get("verdict", "")
        delta_m = ind.get("delta_m", 0.0) or 0.0

        if repos is not None and eid and aid:
            try:
                raw = Path(png).read_bytes()
                sfid = repos.scan_files.register(
                    produced_by_action_id=aid,
                    sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw),
                    current_path=png, format_kind="png",
                    parser_spec="matplotlib-agg", meta={"verdict": verdict})
                repos.observations.record_scan(
                    action_id=aid, experiment_id=eid, observable="tip_shape_verdict",
                    scan_file_id=sfid,
                    result_summary={"verdict": verdict, "delta_m": delta_m,
                                    "advice": str(ind.get("advice", ""))[:400]})
            except Exception as exc:
                logger.warning("tip_shape scan_file write failed: %s", exc)

        try:
            buf = buffer_getter()
            if buf is not None:
                from mast.buffer.schemas import make_tip_shape_verdict
                buf.emit_event(make_tip_shape_verdict(
                    verdict, delta_m * 1e9, seqno=buf.next_seq(),
                    file_path=png, cause_ref=f"skill:{skill_name}"))
        except Exception as exc:
            logger.warning("tip_shape buffer emit failed: %s", exc)

        return {"scan_paths": [png]}

    return _hook


__all__ = ["make_tip_shape_post_hook"]
