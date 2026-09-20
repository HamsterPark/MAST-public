"""Hardware Spectrum Analyzer acquisition — Nanonis-side PSD via TCP.

This skill pulls one (or more) PSDs from the Spectrum Analyzer module that
Nanonis already computes in hardware. Compared with software FFT on a polled
buffer it has two advantages:

  - No Python-side polling burden; one TCP round-trip (~0.5 ms) returns
    the whole spectrum.
  - On real hardware (not the bundled simulator), f_max can go up to
    RTfreq / 2 (~10 kHz on a V5e), well above the ~1 kHz Nyquist
    achievable from TCP polling.

The simulator caps f_max at 1 kHz (six selectable ranges: 20, 50, 100,
200, 500, 1000 Hz) and df at 5 resolutions (~0.49–7.81 Hz). The skill
keeps the operator-chosen settings by default; pass ``freq_range_index``
to override a single capture, or ``freq_range_indices`` (JSON-list) to
sweep multiple ranges in one composite invocation.

Phase 7 migration (2026-05-19): AcquirePSD is now a graph-shaped composite.
The work splits into:

  * configure step — start the analyser, set channel + resolution.
  * per-range capture steps — for each requested range index, set the
    frequency range and read the PSD. Each capture step is a graph node so
    progress is per-band visible.

Requires the nanonis_spm patch in ``mast/core/nanonis_patch.py`` — without
it, the upstream library's wrong-type decode of ``SpectrumAnlzr.DataGet``
raises UnicodeDecodeError. We auto-import the patch so it's applied.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mast.core import nanonis_patch as _patch  # noqa: F401 — ensures patch is applied
from mast.core.types import (
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)

logger = logging.getLogger(__name__)


# Synthetic phase identifiers — intercepted by _AcquirePSDPhaseCtx.
_PHASE_CONFIGURE = "_phase_configure"
_PHASE_RANGE_CAPTURE_PREFIX = "_phase_range_"


class _AcquirePSDPhaseCtx:
    """Wraps the real ExecutionContext to dispatch ``_phase_*`` skill names."""

    def __init__(self, real_ctx, skill: "AcquirePSD") -> None:
        self._ctx = real_ctx
        self._skill = skill

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        return getattr(self._ctx, name)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        if skill_name.startswith("_phase_"):
            return self._skill._run_phase(skill_name, params, self._ctx)
        return self._ctx.run(skill_name, params)


class AcquirePSD(CompositeSkillGraph):
    """Pull one or more PSDs from Nanonis hardware Spectrum Analyzer.

    Supports either single-band capture (legacy v1 contract: pass
    ``freq_range_index`` and get back one ``psd`` array) or multi-band sweep
    (pass ``freq_range_indices`` as JSON list and get back a ``per_range``
    dict keyed by frequency-range index).
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquirePSD",
            version="2.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从 Nanonis 侧的 Spectrum Analyzer（硬件 FFT）读功率谱密度。"
                "每个请求的频率量程返回一组 (f0, df, psd_array)。时延：每个量程 ~0.5 ms RTT。"
                "当你需要的刷新率超出软件轮询能达到的范围时用它（真实硬件：Nyquist 最高 ~10 kHz；"
                "模拟器：上限 1 kHz）。"
            ),
            parameters=[
                ParameterSpec(
                    name="instance",
                    type="int",
                    description="Spectrum Analyzer 实例（1 或 2）。",
                    required=False, default=1,
                    min_value=1, max_value=2,
                ),
                ParameterSpec(
                    name="signal_index",
                    type="int",
                    description=(
                        "可选的待分析信号索引（0–127）。默认用 Spectrum Analyzer "
                        "前面板上已配置好的通道。"
                    ),
                    required=False, default=-1,
                    min_value=-1, max_value=127,
                ),
                ParameterSpec(
                    name="freq_range_index",
                    type="int",
                    description=(
                        "可选的、从 0 开始的索引，指向可用频率量程列表。-1 = 保持当前。本模拟器上："
                        "0=20Hz, 1=50, 2=100, 3=200, 4=500, 5=1000 Hz。给了 freq_range_indices "
                        "时本参数被忽略。"
                    ),
                    required=False, default=-1,
                    min_value=-1, max_value=15,
                ),
                ParameterSpec(
                    name="freq_range_indices",
                    type="str",
                    description=(
                        "可选的 JSON 列表，给出要扫的频率量程索引，例如 '[0, 2, 5]'。给了之后，"
                        "技能会对每个索引各采一条 PSD，并返回一个 per_range 字典。"
                    ),
                    required=False, default="",
                ),
                ParameterSpec(
                    name="freq_resolution_index",
                    type="int",
                    description=(
                        "可选的、从 0 开始的索引，指向分辨率列表。-1 = 保持当前。索引越小 = df 越细。"
                    ),
                    required=False, default=-1,
                    min_value=-1, max_value=15,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=1,
            tags=["spectrum", "psd", "fft", "hardware", "read", "composite"],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_indices(params: dict) -> list[int]:
        """Resolve the list of frequency-range indices to capture."""
        raw = params.get("freq_range_indices", "") or ""
        if raw:
            try:
                items = json.loads(raw)
                return [int(x) for x in items]
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning(
                    "AcquirePSD: invalid freq_range_indices %r (%s); "
                    "falling back to single-range mode.", raw, exc,
                )
        # Single-range fallback: use freq_range_index (-1 = keep current).
        return [int(params.get("freq_range_index", -1))]

    # ------------------------------------------------------------------
    # Plan — configure + one capture per range.
    # ------------------------------------------------------------------

    def plan(self, params: dict) -> list[CompositeStep]:
        indices = self._parse_indices(params)
        instance = int(params.get("instance", 1))
        sig_idx = int(params.get("signal_index", -1))
        res_idx = int(params.get("freq_resolution_index", -1))

        steps: list[CompositeStep] = [
            CompositeStep(
                step_id="configure",
                skill_name=_PHASE_CONFIGURE,
                params={
                    "instance": instance,
                    "signal_index": sig_idx,
                    "freq_resolution_index": res_idx,
                },
                optional=False,
                checkpoint_after=False,
                tags=("setup",),
            )
        ]
        last = len(indices) - 1
        for i, fr_idx in enumerate(indices):
            steps.append(CompositeStep(
                step_id=f"range_{i}_capture",
                skill_name=f"{_PHASE_RANGE_CAPTURE_PREFIX}{i}",
                params={
                    "instance": instance,
                    "range_index_pos": i,
                    "freq_range_index": int(fr_idx),
                    "is_last": i == last,
                },
                # Any capture failure aborts — we can't substitute zeros.
                optional=False,
                checkpoint_after=(i == last),
                tags=("capture", f"range_idx={fr_idx}", f"pos={i}"),
            ))
        return steps

    # ------------------------------------------------------------------
    # Phase dispatch
    # ------------------------------------------------------------------

    def _run_phase(self, skill_name: str, params: dict, real_ctx) -> SkillResult:
        if skill_name == _PHASE_CONFIGURE:
            return self._phase_configure(params, real_ctx)
        if skill_name.startswith(_PHASE_RANGE_CAPTURE_PREFIX):
            return self._phase_range_capture(params, real_ctx)
        return SkillResult(
            skill_name=skill_name,
            success=False,
            error=f"Unknown phase: {skill_name}",
        )

    def _phase_configure(self, params: dict, real_ctx) -> SkillResult:
        instance = int(params["instance"])
        sig_idx = int(params["signal_index"])
        res_idx = int(params["freq_resolution_index"])

        # Ensure the analyser is running. Idempotent — don't fail-fast on Run
        # errors (sometimes the module is already running and returns benign
        # warnings).
        rec = real_ctx.safe_call("SpectrumAnlzr_Run", instance)
        self._call_log.append(rec)

        if sig_idx >= 0:
            rec = real_ctx.safe_call(
                "SpectrumAnlzr_ChSet", instance, sig_idx,
            )
            self._call_log.append(rec)
            if rec.error:
                return SkillResult(
                    skill_name=_PHASE_CONFIGURE,
                    success=False,
                    error=f"ChSet failed: {rec.error}",
                )

        if res_idx >= 0:
            rec = real_ctx.safe_call(
                "SpectrumAnlzr_FreqResSet", instance, res_idx,
            )
            self._call_log.append(rec)

        # Read back the resolution + channel so the LLM/caller knows the
        # configuration. Don't fail on read errors.
        rec_res = real_ctx.safe_call("SpectrumAnlzr_FreqResGet", instance)
        self._call_log.append(rec_res)
        rec_ch = real_ctx.safe_call("SpectrumAnlzr_ChGet", instance)
        self._call_log.append(rec_ch)
        ch_d = _decoded(rec_ch.return_value)
        if ch_d:
            try:
                self._executor.set_partial("channel_index", int(ch_d[0]))
            except (TypeError, ValueError):
                pass
        self._executor.set_partial("instance", instance)
        return SkillResult(
            skill_name=_PHASE_CONFIGURE,
            success=True,
            data={"instance": instance},
        )

    def _phase_range_capture(self, params: dict, real_ctx) -> SkillResult:
        instance = int(params["instance"])
        fr_idx = int(params["freq_range_index"])
        pos = int(params["range_index_pos"])

        if fr_idx >= 0:
            rec = real_ctx.safe_call(
                "SpectrumAnlzr_FreqRangeSet", instance, fr_idx,
            )
            self._call_log.append(rec)

        rec_range = real_ctx.safe_call(
            "SpectrumAnlzr_FreqRangeGet", instance,
        )
        self._call_log.append(rec_range)
        rec_data = real_ctx.safe_call("SpectrumAnlzr_DataGet", instance)
        self._call_log.append(rec_data)
        if rec_data.error:
            return SkillResult(
                skill_name=_PHASE_RANGE_CAPTURE_PREFIX,
                success=False,
                error=f"DataGet failed: {rec_data.error}",
            )

        d = _decoded(rec_data.return_value)
        if len(d) < 4:
            return SkillResult(
                skill_name=_PHASE_RANGE_CAPTURE_PREFIX,
                success=False,
                error=f"Unexpected response shape: got {len(d)} fields",
            )
        try:
            f0 = float(d[0])
            df = float(d[1])
            n = int(d[2])
            ys_raw = d[3]
            psd = [
                float(v[0]) if isinstance(v, tuple) else float(v)
                for v in ys_raw
            ]
        except (TypeError, ValueError, IndexError) as exc:
            return SkillResult(
                skill_name=_PHASE_RANGE_CAPTURE_PREFIX,
                success=False,
                error=f"PSD decode error: {exc}",
            )

        # Pull available frequency ranges from the FreqRangeGet response —
        # cache once (the list doesn't change between captures).
        if "available_freq_ranges_hz" not in self._executor.progress.partial_data:
            ranges_d = _decoded(rec_range.return_value)
            avail_ranges: list[float] = []
            if len(ranges_d) >= 3 and isinstance(ranges_d[2], list):
                avail_ranges = [
                    float(r[0]) if isinstance(r, tuple) else float(r)
                    for r in ranges_d[2]
                ]
            self._executor.set_partial(
                "available_freq_ranges_hz", avail_ranges,
            )

        # Stash this band's PSD into partial_data, keyed by position so
        # resumed runs don't duplicate.
        per_range = dict(
            self._executor.progress.partial_data.get("per_range", {}))
        per_range[str(pos)] = {
            "freq_range_index": fr_idx,
            "f0_hz": f0,
            "df_hz": df,
            "n_bins": n,
            "f_max_hz": f0 + (n - 1) * df,
            "psd": psd,
        }
        self._executor.set_partial("per_range", per_range)
        return SkillResult(
            skill_name=_PHASE_RANGE_CAPTURE_PREFIX,
            success=True,
            data={"pos": pos, "freq_range_index": fr_idx},
        )

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        self._call_log: list[NanonisCallRecord] = []
        wrapped = _AcquirePSDPhaseCtx(context, self)
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=wrapped,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        executor.set_partial_default("per_range", {})
        self._executor = executor

        all_good = executor.run_plan(iter(self.plan(params)))

        per_range = dict(executor.progress.partial_data.get("per_range", {}))
        # Sort by integer position for deterministic ordering.
        ordered = [per_range[k] for k in sorted(per_range, key=lambda x: int(x))]
        instance = int(executor.progress.partial_data.get("instance",
                                                          params.get("instance", 1)))
        channel_index = int(executor.progress.partial_data.get(
            "channel_index", -1))
        avail = list(executor.progress.partial_data.get(
            "available_freq_ranges_hz", []))

        data: dict[str, Any] = {
            "instance": instance,
            "channel_index": channel_index,
            "per_range": ordered,
            "available_freq_ranges_hz": avail,
            "source": "nanonis_hardware_spectrum_analyzer",
            "_progress": executor.progress.to_dict(),
        }
        # Back-compat: surface the first range's data at the top level so v1
        # callers that read result.data["psd"] keep working.
        if ordered:
            first = ordered[0]
            data.update({
                "f0_hz": first["f0_hz"],
                "df_hz": first["df_hz"],
                "n_bins": first["n_bins"],
                "f_max_hz": first["f_max_hz"],
                "psd": first["psd"],
            })

        if not all_good:
            return SkillResult(
                skill_name=self._skill_name(),
                success=False,
                error=executor.progress.aborted_reason or "AcquirePSD aborted",
                data=data,
                nanonis_calls=list(self._call_log),
            )
        return SkillResult(
            skill_name=self._skill_name(),
            success=True,
            data=data,
            nanonis_calls=list(self._call_log),
        )


def _decoded(rv):
    """Unwrap (err_str, raw_bytes, [decoded_values])."""
    if isinstance(rv, tuple) and len(rv) >= 3 and isinstance(rv[2], list):
        return rv[2]
    if isinstance(rv, list):
        return rv
    return []
