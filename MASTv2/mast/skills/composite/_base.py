"""CompositeSkill: base class for multi-step orchestration skills.

Two flavours live here:

  • ``CompositeSkill`` — the historical v1 base (opaque Python loop). Kept
    for back-compat while we migrate. ``run_composite()`` walks sub-skills
    via ``step()`` / ``step_or_fail()`` and returns one SkillResult.

  • ``CompositeSkillGraph`` — the v2 unified framework (this file's
    centerpiece). Subclasses declare their work as a *plan* (list of
    :class:`CompositeStep`) or *dynamic plan* (generator). The
    :class:`GraphExecutor` walks the plan, emits per-step progress into
    ``MASTState.composite_progress[<skill>]``, supports resume on
    re-invocation, and lets the SqliteSaver flush after each step.

All new composite skills MUST subclass ``CompositeSkillGraph``. Existing
composites are being migrated to it.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Iterator

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.base import BaseSkill
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
    abort_facts,
)


class AbortRequested(Exception):
    """Raised by step() when the abort flag is set."""

    def __init__(self, skill_name: str = ""):
        super().__init__(f"Abort requested during {skill_name}")


class CompositeSkill(BaseSkill):
    """Base class for composite skills with orchestration primitives.

    Subclasses implement ``run_composite()`` instead of ``execute()``.
    Three primitives eliminate the per-sub-skill boilerplate:

    * ``step()``  — run a sub-skill, accumulate calls, check abort
    * ``step_or_fail()`` — same, but return immediately on failure
    * ``abortable_sleep()`` — interruptible sleep with abort checks

    Two factory methods build the final ``SkillResult``:

    * ``ok(**data)``   — success result with accumulated calls
    * ``fail(error)``  — failure result with accumulated calls
    """

    def __init__(self) -> None:
        self._all_calls: list[NanonisCallRecord] = []

    # ------------------------------------------------------------------
    # Orchestration primitives
    # ------------------------------------------------------------------

    def step(self, context, skill_name: str, params: dict) -> SkillResult:
        """Run a sub-skill, accumulate its calls, and check abort.

        Raises ``AbortRequested`` if the abort flag is set *before*
        the sub-skill runs.  The caller decides how to handle a failed
        sub-skill (continue, retry, or return).
        """
        if hasattr(context, "check_abort") and context.check_abort():
            raise AbortRequested(self._skill_name())
        result = context.run(skill_name, params)
        self._all_calls.extend(result.nanonis_calls)
        return result

    def step_or_fail(
        self, context, skill_name: str, params: dict,
    ) -> SkillResult:
        """Run a sub-skill; return a failure ``SkillResult`` if it fails.

        Convenience wrapper around ``step()`` for the common fail-fast
        pattern.  If the sub-skill succeeds the result is returned
        unchanged so callers can inspect ``result.data``.
        """
        result = self.step(context, skill_name, params)
        if not result.success:
            return self.fail(f"{skill_name} failed: {result.error}")
        return result

    # ------------------------------------------------------------------
    # Result factories
    # ------------------------------------------------------------------

    def fail(self, error: str, **data) -> SkillResult:
        """Build a failure ``SkillResult`` with accumulated calls."""
        return SkillResult(
            skill_name=self._skill_name(),
            success=False,
            error=error,
            data=data if data else {},
            nanonis_calls=list(self._all_calls),
        )

    def ok(self, **data) -> SkillResult:
        """Build a success ``SkillResult`` with accumulated calls."""
        return SkillResult(
            skill_name=self._skill_name(),
            success=True,
            data=data,
            nanonis_calls=list(self._all_calls),
        )

    # ------------------------------------------------------------------
    # Execution wrapper
    # ------------------------------------------------------------------

    def execute(self, context, params: dict) -> SkillResult:  # noqa: D102
        self._all_calls = []
        try:
            return self.run_composite(context, params)
        except AbortRequested:
            return self.fail("Aborted by user or watchdog")

    @abstractmethod
    def run_composite(self, context, params: dict) -> SkillResult:
        """Implement the composite workflow here (instead of execute)."""
        ...

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _skill_name(self) -> str:
        return self.metadata().name


# ──────────────────────────────────────────────────────────────────────────
# CompositeSkillGraph — the v2 unified framework
# ──────────────────────────────────────────────────────────────────────────


class CompositeSkillGraph(CompositeSkill):
    """Composite skill that exposes its work as a graph (plan of steps).

    Subclasses implement either :meth:`plan` (static — params alone
    determine the step sequence) or :meth:`plan_dynamic` (later step
    params depend on earlier sub-skill results).

    Optional hooks:
      * :meth:`on_step_result(step, sub_result)` — stash data into
        ``self._executor.progress.partial_data`` for the final aggregate.
      * :meth:`on_step_failed(step, msg)` — return True to continue
        (treat as optional), False to abort. Default behaviour: abort
        iff ``step.optional is False``.
      * :meth:`aggregate(sub_results)` — build the final SkillResult
        data dict from per-step results. Default: returns
        ``self._executor.progress.partial_data``.

    State machine:
      ::

          execute() → run_composite() → _graph_execute()
                                         │
                                         ├─→ plan() / plan_dynamic()
                                         │
                                         ├─→ GraphExecutor.run_plan()
                                         │     for step in plan:
                                         │       if completed → skip (resume)
                                         │       result = ctx.run(step.skill, …)
                                         │       emit_progress()
                                         │       checkpoint_flush()
                                         │
                                         └─→ aggregate()  ⇒  SkillResult
    """

    # Subclasses override exactly one of plan() / plan_dynamic().
    # Default plan() raises so a subclass that forgets to implement either
    # fails loudly rather than silently no-op'ing.

    def plan(self, params: dict) -> list[CompositeStep]:
        """Static plan: full step list derivable from params alone.

        Override this for composites where the step sequence is fixed
        once the input params are known (e.g. GridSTS: nx × ny moves).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override plan() or plan_dynamic()"
        )

    def plan_dynamic(
        self, params: dict, executor: GraphExecutor,
    ) -> Iterator[CompositeStep]:
        """Dynamic plan: later steps depend on earlier results.

        Yield steps one at a time. After each ``yield``, the executor
        has finished that step and its result is in
        ``executor.sub_results[step.step_id]``. Use
        :meth:`on_step_result` to stash data, or read
        ``executor.sub_results`` directly inside this generator.

        Default implementation: yield from :meth:`plan` (static). Override
        only when needed.
        """
        for step in self.plan(params):
            yield step

    def on_step_result(self, step: CompositeStep, sub_result: SkillResult) -> None:
        """Stash data from a successful sub-step. Default: no-op.

        Typical use: copy a measurement from sub_result.data into
        ``self._executor.set_partial(key, value)`` so the final
        SkillResult carries aggregated data."""

    def on_step_failed(self, step: CompositeStep, msg: str) -> bool:
        """Return True to continue, False to abort. Default: respect
        ``step.optional`` (delegated to executor)."""
        return step.optional

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        """Build the final SkillResult.data dict.

        Default: returns ``progress.partial_data`` so subclasses that
        used :meth:`set_partial` get their stashed data automatically.
        Override for custom aggregation."""
        return dict(progress.partial_data)

    # -- Final wiring: execute → run_composite → GraphExecutor ---------

    def run_composite(self, context, params: dict) -> SkillResult:
        """Default: drive the GraphExecutor. Subclasses rarely override."""
        return self._graph_execute(context, params)

    def _graph_execute(self, context, params: dict) -> SkillResult:
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        self._executor = executor  # expose for hooks
        plan_iter = self.plan_dynamic(params, executor)
        all_good = executor.run_plan(plan_iter)
        if all_good and not executor.progress.aborted:
            # P2-G: the run finished — drop the out-of-band sidecar so the
            # NEXT invocation starts fresh instead of resume-skipping stale
            # completed steps. Kept on abort/interrupt (that's the resume).
            executor.clear_sidecar()
        data = self.aggregate(executor.sub_results, executor.progress)
        # Promote progress into the result for downstream wrap_skill
        # adapter (which lifts it into MASTState.composite_progress).
        data["_progress"] = executor.progress.to_dict()
        # 停止这件事的机器可读版本(缺陷⑬)。放在**顶层**而不是只埋在 _progress 里:
        # 下游要判「用户喊停」还是「它自己失败了」,不该去猜一句人话的措辞 ——
        # 那正是 #46 的翻版(判据落在措辞上,措辞一改就失效)。
        data.update(abort_facts(executor.progress))
        ok, reason = self._decide_outcome(all_good, executor.progress, data)
        if ok:
            # P?-⑨ product-validity gate: "跑完/存盘" is not "produced a usable
            # product". Before we call a drained plan a success, verify the
            # terminal product isn't a crashed / all-NaN / dead-flat frame
            # (5305868e: timed_out:False + saved_path but crash_indicator:True,
            # rms:nan — reported as success). Positive-evidence-only, so a
            # composite with no recognizable product is unaffected.
            pv_ok, pv_reason = self._validate_products(data)
            if not pv_ok:
                data["degraded"] = True
                return self.fail("degraded: " + pv_reason, **data)
            return self.ok(**data)
        return self.fail(reason, **data)

    # Keys under which a composite's terminal PRODUCT (a saved scan array) may
    # surface in its aggregate data. A path here is loaded + checked; anything
    # else in data is ignored by the gate.
    _PRODUCT_PATH_KEYS: tuple[str, ...] = (
        "saved_path", "product_path", "output_path", "scan_path", "result_path",
    )

    def _validate_products(self, data: dict) -> tuple[bool, str]:
        """Positive-evidence-only check that the composite's terminal product is
        usable (feedback ⑨). Returns ``(ok, reason)``.

        Downgrades to a degraded result ONLY on unmistakable evidence:
          * an explicit ``crash_indicator: True`` (or a ``crash_check == "crash"``
            in the progress partial data) that reached the aggregate; or
          * a saved scan file under one of ``_PRODUCT_PATH_KEYS`` that loads to an
            all-NaN / dead-flat array, or cannot be opened at all.

        A composite with no recognizable product returns ``(True, "")`` — this
        gate never invents a failure. It also never raises: any error inside the
        check degrades to "valid" so the gate can't itself break a run.
        """
        if not isinstance(data, dict):
            return True, ""
        try:
            # 1. Explicit crash verdict from a sub-skill (FullScan crash check /
            #    CheckScanForCrash) that reached the aggregate.
            if data.get("crash_indicator") is True:
                reason = str(data.get("crash_status") or data.get("crash_channel")
                             or data.get("error") or "crash_indicator=True")
                return False, f"scan crashed ({reason})"
            prog = data.get("_progress")
            if isinstance(prog, dict):
                pd = prog.get("partial_data")
                if isinstance(pd, dict) and str(pd.get("crash_check")) == "crash":
                    return False, "post-scan crash check = crash"
            # 2. A saved scan product we can actually load + assess.
            for key in self._PRODUCT_PATH_KEYS:
                val = data.get(key)
                if not val or not isinstance(val, str):
                    continue
                from mast.io.product_validity import (
                    LOADABLE_PRODUCT_EXT,
                    assess_scan_file,
                )
                if not val.lower().endswith(LOADABLE_PRODUCT_EXT):
                    continue
                verdict = assess_scan_file(val)
                if not verdict.get("valid", True):
                    return False, f"product invalid ({key}): {verdict.get('reason')}"
        except Exception:  # noqa: BLE001 — the gate must never crash a run
            return True, ""
        return True, ""

    def _decide_outcome(self, all_good: bool, progress: CompositeProgress,
                        data: dict) -> tuple[bool, str]:
        """Final ok/fail decision for the composite. Base behaviour: success iff
        the plan drained without abort. ``SpecComposite`` (P5) overrides this to
        honour declarative ``success_when`` / ``succeed``-``fail`` verdict nodes."""
        if all_good:
            return True, ""
        return False, progress.aborted_reason or "composite aborted"
