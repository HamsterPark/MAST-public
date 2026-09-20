"""FullScan — one-step scan configuration, acquisition, and wait (Phase 7 graph-shaped composite).

Migration note (Phase 7):
  The original v1 layout (ConfigureScan -> SetScanSpeed -> StartScan ->
  wait_scan_complete -> raw FrameDataGrab crash check) maps cleanly onto
  the graph framework: four sub-skill steps for the main flow, then the
  post-scan crash detection stays as raw ``context.safe_call`` inside the
  run_composite override (no sub-skill exists for that and the v1
  semantic is "best-effort warning, never block result"). The wait step
  uses the ``WaitScanComplete`` builtin (added Phase 4) instead of the
  legacy ``wait_scan_complete`` helper.
"""

# K (Keep) — migrated to CompositeSkillGraph 2026-05-19

from __future__ import annotations

import logging

import numpy as np

from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.core.tip_crash_tracker import (
    crash_escape_message,
    crash_guard,
    get_tip_crash_tracker,
)
from mast.agents._shared.skill_adapter import wrap_skill

logger = logging.getLogger(__name__)


class FullScan(CompositeSkillGraph):
    """Configure, start, and wait for a scan in one step.

    Orchestrates: ConfigureScan -> SetScanSpeed -> StartScan -> wait ->
    crash detection.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="FullScan",
            version="1.2.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "一步式扫描:配置扫描区域、设定速度、启动、等它扫完。"
                "相当于 ConfigureScan + SetScanSpeed + StartScan 合成一步。"
            ),
            parameters=[
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description="扫描中心 X",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description="扫描中心 Y",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="width_m",
                    type="float",
                    description=(
                        "扫描宽度,单位**米**(SI),**不是纳米**。"
                        "换算:100 nm → 100n,50 nm → 50n。像 100 这样的裸值"
                        "(=100 m)是单位写错,会被拒绝。"
                    ),
                    unit="m",
                    required=True,
                    min_value=1e-10,
                    max_value=1e-5,   # = config scan_size_max_m (10 µm); guards nm→m slips
                ),
                ParameterSpec(
                    name="height_m",
                    type="float",
                    description=(
                        "扫描高度,单位**米**(SI),**不是纳米**。"
                        "换算:100 nm → 100n,50 nm → 50n。像 100 这样的裸值"
                        "(=100 m)是单位写错,会被拒绝。"
                    ),
                    unit="m",
                    required=True,
                    min_value=1e-10,
                    max_value=1e-5,   # = config scan_size_max_m (10 µm); guards nm→m slips
                ),
                ParameterSpec(
                    name="line_time_s",
                    type="float",
                    description=(
                        "正扫每线时间,单位秒。**除非要求说了一个数,否则"
                        "别填**:省略时,它来自用户的逐尺度档位表(这个帧尺寸"
                        "该配多快,是他们的知识,不是你的)。优先用 ScanAt,"
                        "它对每一个扫描参数都是这么定的。"
                    ),
                    unit="s",
                    required=False,
                    # 刻意 None 而不是 0.1:pydantic 会把 ParameterSpec.default
                    # 物化进参数,所以只要这里写着 0.1,「没传」和「显式传 0.1」
                    # 就永远分不开,档位表也就永远轮不到。
                    default=None,
                    min_value=0.01,
                    max_value=60.0,
                ),
                ParameterSpec(
                    name="channels",
                    type="str",
                    description="分号分隔的通道列表",
                    required=False,
                    default="",
                ),
                ParameterSpec(
                    name="wait_timeout_s",
                    type="float",
                    description=(
                        "等扫描完成最多等多久。留空则由**真实的扫描几何自动估算**"
                        "(从仪器读回的实际行数 × line_time × 2 个方向,"
                        "+30% 余量,下限 300 s)—— **不要**自己挑一个短值,"
                        "否则可能把一张慢扫 / 高分辨的图截断。超时会停掉扫描。"
                    ),
                    unit="s",
                    required=False,
                    default=None,
                    min_value=10.0,
                    max_value=3600.0,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=60.0,
            composition_level=3,
            tags=["scan", "imaging", "composite"],
        )

    # --- Scan parameters from the operator's per-scale policy table ---

    @staticmethod
    def _resolve_line_time(params: dict) -> tuple[float, str]:
        """(每线时间, 来源)。显式给了就用显式的,否则按帧尺寸查档位表。

        这条 fallback 换掉了原来写死的 0.1 s —— 那个常数对 1 µm 的巡查图和 5 nm
        的原子分辨图给的是同一个速度,而它们之间差着一两个数量级。
        """
        # 实现搬到 ``scan_policy.resolve_line_time`` 了 —— 这段逻辑原本有三份，
        # 其中两份把 0.1 s 当默认值。语义与来源标签保持不变。
        from mast.core.scan_policy import resolve_line_time
        size = max(float(params.get("width_m") or 0.0),
                   float(params.get("height_m") or 0.0))
        return resolve_line_time(params.get("line_time_s"), size)

    # --- Adaptive wait-timeout estimation ---

    @staticmethod
    def _estimate_wait_timeout_s(line_time: float, n_lines) -> float:
        """Estimate scan-completion timeout from the ACTUAL scan geometry:
        ``n_lines × line_time × 2 directions`` with 30 % headroom + 30 s, floored
        at 300 s. Replaces the old hardcoded 512-line assumption — a 512-line scan
        at 1 s/line (fwd+bwd) needs ~1024 s, so any short/fixed timeout truncated
        every high-res or slow scan (the operator saw scans cut off at ~88 %)."""
        try:
            n = int(n_lines)
            if n <= 0:
                n = 512
        except (TypeError, ValueError):
            n = 512
        try:
            lt = float(line_time)
        except (TypeError, ValueError):
            lt = 0.1
        return max(300.0, n * lt * 2 * 1.3 + 30.0)

    @staticmethod
    def _read_scan_lines(context):
        """Best-effort read of the configured scan line count via Scan_BufferGet.
        Returns an int, or None on any failure so the caller falls back to the
        512-line assumption. Scan.BufferGet Variables ==
        [num_channels, channel_indexes, pixels, lines] → lines is index 3."""
        try:
            rec = context.safe_call("Scan_BufferGet")
            if getattr(rec, "error", ""):
                return None
            from mast.io.nanonis_files import parse_buffer_get
            buf = parse_buffer_get(getattr(rec, "return_value", None))
            n = buf["lines"] if buf else None
            return n if n and n > 0 else None
        except Exception:  # noqa: BLE001 — best-effort; fall back to 512
            return None

    def _resolve_wait_timeout(self, params: dict) -> float:
        """The wait timeout actually used: an explicit ``wait_timeout_s`` wins;
        otherwise estimate from ``line_time_s`` and the real line count
        (``_n_lines``, stashed by run_composite; 512 when unknown)."""
        req = params.get("wait_timeout_s")
        if req is not None:
            return float(req)
        return self._estimate_wait_timeout_s(
            self._resolve_line_time(params)[0],
            params.get("_n_lines") or 512,
        )

    # --- Plan: static (params alone determine the 4-step flow) ---

    def plan(self, params: dict) -> list[CompositeStep]:
        center_x = params["center_x_m"]
        center_y = params["center_y_m"]
        width = params["width_m"]
        height = params["height_m"]
        # Honour the requested line time (the ParameterSpec bounds it to
        # 0.01–60 s). The old min(..., 0.1) silently clamped every slow/
        # high-quality scan back to 0.1 s/line despite the declared range
        #.
        line_time, _lt_src = self._resolve_line_time(params)
        channels = params.get("channels", "")
        # Default the wait timeout to the scan's real duration, estimated from the
        # ACTUAL line count (params["_n_lines"], read from Scan_BufferGet by
        # run_composite) × line_time × 2 directions — not a hardcoded 512 (field
        # ). An explicit wait_timeout_s is honoured as-is.
        wait_timeout = self._resolve_wait_timeout(params)

        cfg_params: dict = {
            "center_x_m": center_x,
            "center_y_m": center_y,
            "width_m": width,
            "height_m": height,
        }
        if channels:
            cfg_params["channels"] = channels

        scan_speed = width / line_time if line_time > 0 else 200e-9

        return [
            CompositeStep(
                step_id="configure",
                skill_name="ConfigureScan",
                params=cfg_params,
                optional=False,
                checkpoint_after=False,
                tags=("setup",),
            ),
            CompositeStep(
                step_id="set_speed",
                skill_name="SetScanSpeed",
                params={
                    "fwd_speed": scan_speed,
                    "bwd_speed": scan_speed,
                    "fwd_line_time": line_time,
                    "bwd_line_time": line_time,
                    "keep_const": 0,
                },
                optional=False,
                checkpoint_after=False,
                tags=("setup",),
            ),
            CompositeStep(
                step_id="start_scan",
                skill_name="StartScan",
                params={},
                optional=False,
                checkpoint_after=False,
                tags=("scan",),
            ),
            CompositeStep(
                step_id="wait_scan",
                skill_name="WaitScanComplete",
                params={"timeout_ms": int(wait_timeout * 1000)},
                optional=False,
                checkpoint_after=True,    # checkpoint once the scan finishes
                tags=("wait",),
            ),
        ]

    # --- Hooks ---

    def on_step_result(self, step: CompositeStep, sub_result) -> None:
        # WaitScanComplete reports success on EVERY way a scan can end — timeout,
        # a frame that finished, and a frame that was stopped part-way. Forward
        # all three, so "the operator pressed Stop at 24 %" cannot look identical
        # to "the frame is done" by the time run_composite decides (v6.1.2).
        if step.step_id == "wait_scan":
            data = getattr(sub_result, "data", {}) or {}
            self._executor.set_partial(
                "wait_timed_out", bool(data.get("timed_out", False)),
            )
            self._executor.set_partial(
                "wait_stopped_early", bool(data.get("stopped_early", False)),
            )
            self._executor.set_partial(
                "wait_outcome", str(data.get("outcome") or ""),
            )
            # Scalars only — this rides the checkpointer.
            for key in ("lines_done", "lines_total"):
                val = data.get(key)
                self._executor.set_partial(
                    f"scan_{key}", int(val) if val is not None else None,
                )
            self._executor.set_partial(
                "scan_lines_verified", bool(data.get("lines_verified", False)),
            )

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        return {
            "center_x_m": progress.partial_data.get("center_x_m"),
            "center_y_m": progress.partial_data.get("center_y_m"),
            "width_m": progress.partial_data.get("width_m"),
            "height_m": progress.partial_data.get("height_m"),
            "line_time_s": progress.partial_data.get("line_time_s"),
            # "ok" | "skipped" | "crash" — never silently hide that the
            # post-scan crash detector could not read any channel.
            "crash_check": progress.partial_data.get("crash_check"),
            "crash_check_channels": progress.partial_data.get(
                "crash_check_channels"
            ),
            # How the wait actually ended, and the evidence for it. Present on
            # the success path too: "512/512 行, verified" is what makes the
            # clean result a CHECKED claim rather than an assumed one.
            "wait_outcome": progress.partial_data.get("wait_outcome"),
            "scan_lines_done": progress.partial_data.get("scan_lines_done"),
            "scan_lines_total": progress.partial_data.get("scan_lines_total"),
            "scan_lines_verified": progress.partial_data.get(
                "scan_lines_verified"
            ),
        }

    # --- run_composite override: graph + post-scan crash detection ---

    def run_composite(self, context, params: dict) -> SkillResult:
        center_x = params["center_x_m"]
        center_y = params["center_y_m"]
        width = params["width_m"]
        height = params["height_m"]
        line_time, _lt_src = self._resolve_line_time(params)

        # ⑫ tip-crash state machine: refuse to scan a spot that has already
        # crashed the tip ≥ threshold times. Conditioning in place does not fix a
        # bad location — the agent must withdraw + coarse-move to escape. Refusing
        # here is what breaks the ~5-min in-place spin (field trace).
        escape = crash_guard(context, center_x, center_y)
        if escape:
            return self.fail(escape, repeated_crash=True,
                             center_x_m=center_x, center_y_m=center_y)
        # Read the ACTUAL scan resolution ONCE, up front, so the wait-timeout
        # estimate matches the real scan instead of assuming 512 lines (field
        # ). Stash it into params so self.plan() below uses the
        # same number. Only needed when the caller did not pin wait_timeout_s.
        if params.get("wait_timeout_s") is None and params.get("_n_lines") is None:
            n_lines = self._read_scan_lines(context)
            if n_lines:
                params = dict(params)
                params["_n_lines"] = n_lines
        # The wait timeout actually used (explicit value wins; else adaptive
        # estimate). Used both for plan() below and the timeout error message.
        wait_timeout = self._resolve_wait_timeout(params)

        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        # Stash scan params so aggregate() can echo them
        executor.set_partial("center_x_m", center_x)
        executor.set_partial("center_y_m", center_y)
        executor.set_partial("width_m", width)
        executor.set_partial("height_m", height)
        executor.set_partial("line_time_s", line_time)
        self._executor = executor

        all_good = executor.run_plan(iter(self.plan(params)))

        if not all_good:
            data = self.aggregate(executor.sub_results, executor.progress)
            data["_progress"] = executor.progress.to_dict()
            return self.fail(
                executor.progress.aborted_reason or "scan aborted",
                **data,
            )

        # WaitScanComplete returns success even on timeout — handle that
        # as a hard failure (v1 behaviour).
        if executor.progress.partial_data.get("wait_timed_out"):
            data = self.aggregate(executor.sub_results, executor.progress)
            data["_progress"] = executor.progress.to_dict()
            return self.fail(f"Scan timed out after {wait_timeout}s", **data)

        # ...and the scan can also just STOP. Operator Stop, a Nanonis-side halt,
        # a safety halt: the status goes to 0 with part of the frame never
        # acquired. Failing here is the whole point — the next thing this method
        # would otherwise do is run a crash check over NaN rows and report a
        # clean scan, and the caller above would save and move on. Whatever
        # stopped it, the operator has to know the frame is not the frame they
        # asked for. (2026-08-04: observed at 24 % of frame, no file produced.)
        if executor.progress.partial_data.get("wait_stopped_early"):
            data = self.aggregate(executor.sub_results, executor.progress)
            data["_progress"] = executor.progress.to_dict()
            done = executor.progress.partial_data.get("scan_lines_done")
            total = executor.progress.partial_data.get("scan_lines_total")
            where = (f"{done}/{total} 行" if done is not None and total
                     else "行数未知")
            return self.fail(
                f"扫描中途停止({where}),这一帧没有扫完。可能是用户按了 Stop、"
                f"Nanonis 自行停止,或安全停机。不要把这一帧当作完整图像使用。",
                stopped_early=True, **data,
            )

        # 5. Post-scan crash detection (raw safe_call — best-effort, no
        #    sub-skill exists for "data variance check").
        crash = self._check_scan_data(context)
        if crash is not None:
            # crash already carries _all_calls from _check_scan_data
            data = self.aggregate(executor.sub_results, executor.progress)
            data["_progress"] = executor.progress.to_dict()
            # ⑫ record this crash at the scan centre. Once the region hits the
            # block threshold, ESCALATE the error with the escape directive so the
            # agent stops retrying here and relocates (the NEXT call is refused by
            # crash_guard, but say it now too — don't make it crash a third time).
            error = crash.error
            repeated = False
            try:
                count = get_tip_crash_tracker().record_crash(center_x, center_y)
                if get_tip_crash_tracker().is_blocked(center_x, center_y):
                    repeated = True
                    error = f"{crash.error}\n{crash_escape_message(count, center_x, center_y)}"
            except Exception:  # noqa: BLE001 — tracking never breaks the result
                pass
            return self.fail(
                error,
                crash_indicator=crash.data.get("crash_indicator"),
                data_range=crash.data.get("data_range"),
                repeated_crash=repeated,
                **data,
            )

        # Clean scan at this region — the tip works here. Clear any prior crash
        # count so an old, since-resolved crash never blocks a good spot.
        try:
            get_tip_crash_tracker().note_recovery(center_x, center_y)
        except Exception:  # noqa: BLE001
            pass

        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()
        # Ground the "image quality" claim in the ACTUAL vision verdict. The
        # scan-vision monitor assessed every milestone of this scan, but its
        # conclusion never reached the agent, which then narrated "图像质量正常"
        # off nothing but the crash check while the model had said tip=bad all
        # along. Scalars only — checkpointer-safe.
        data.update(self._vision_verdict())
        return self.ok(**data)

    @staticmethod
    def _vision_verdict() -> dict:
        """Latest tip-quality verdict from the vision buffer, as plain scalars.
        Empty dict when no buffer/vision is running (offline scans degrade)."""
        try:
            from mast.buffer.active import get_active_buffer
            buf = get_active_buffer()
            if buf is None:
                return {}
            ts, _seq = buf.get_latest_tip_status()
            if ts is None:
                return {}
            q = getattr(ts, "quality", None)
            q = getattr(q, "value", q)
            conf = getattr(ts, "confidence", None)
            out: dict = {
                "vision_tip_quality": str(q) if q is not None else None,
                "vision_tip_confidence": (round(float(conf), 3)
                                          if conf is not None else None),
            }
            if str(q).lower() == "bad":
                # Wording discipline (physics-truth validation 2026-07-27): the
                # coarse label supports a tip-state TIER judgement only — it
                # cannot name the failure mode (multi-tip claims in particular
                # have no reliable detector; see vigil_truth_validation).
                out["vision_note"] = (
                    "视觉判定针尖状态为 bad（状态档位判断，不指认具体失效模式）— "
                    "不要将图像质量描述为正常；考虑 ConditionTip/TipPulse 后重扫。")
            return out
        except Exception:  # noqa: BLE001 — vision verdict is best-effort
            return {}

    # --- Crash detection: raw safe_call (no sub-skill exists) ---

    # Channels to probe for the post-scan crash check, as
    # (channel_index, label). Channel 0 is the first acquired channel
    # (usually topography); channel 14 is the Nanonis Z-controller signal.
    # A tip crash flattens Z even when channel 0 looks plausible, so we
    # must consult Z as well rather than trusting channel 0 alone.
    _CRASH_CHECK_CHANNELS: tuple[tuple[int, str], ...] = ((0, "ch0"), (14, "Z"))

    def _grab_channel_array(self, context, channel_index: int):
        """Grab forward-direction data for one channel as a 1-D ndarray.

        Returns ``None`` when the channel could not be read or carried no
        usable samples (so the caller can tell "no data" apart from a
        genuine flat scan).
        """
        rec = context.safe_call("Scan_FrameDataGrab", channel_index, 1)
        self._all_calls.append(rec)
        if rec.error or rec.return_value is None:
            return None
        # Shared robust parse of the heterogeneous [name_len, name, rows, cols,
        # data_2D, dir] body (was inlined here; now shared with scan_frame.py /
        # drift_track.py so all three stay in sync — 审查).
        from mast.io.nanonis_files import parse_frame_grab
        return parse_frame_grab(rec.return_value)

    def _resolve_crash_channels(self, context) -> list[tuple[int, str]]:
        """The channels to probe for a crash = the ACTUALLY ACQUIRED channels,
        read from ``Scan_BufferGet``. The static ``_CRASH_CHECK_CHANNELS`` hard-
        coded channel 14 for "Z", but the Z-controller signal id varies by setup
        (e.g. it is **30**, not 14, on the standard sim) — so the hardcoded probe
        hit "channel #14 is not part of the acquired selection", the read failed,
        and the crash check reported "skipped" on every scan (2026-06-29). Falls
        back to the static list if the buffer can't be read."""
        try:
            rec = context.safe_call("Scan_BufferGet")
            self._all_calls.append(rec)
            rv = getattr(rec, "return_value", None)
            if not getattr(rec, "error", "") and isinstance(rv, (list, tuple)) and len(rv) > 2:
                # 通道的单元素元组由 channel_ids_from_buffer 统一拆包；只读通道不依赖像素和行数完整。
                from mast.io.nanonis_files import channel_ids_from_buffer
                ids = channel_ids_from_buffer(rv[2])
                if ids:
                    return [(cid, f"ch{cid}") for cid in ids]
        except Exception:  # noqa: BLE001 — fall back to the static probe list
            pass
        return list(self._CRASH_CHECK_CHANNELS)

    def _check_scan_data(self, context) -> SkillResult | None:
        """Post-scan crash detection across multiple channels.

        Flags a crash if ANY probed channel has near-zero variance or NaN.
        Crucially, it does NOT report "no crash" when it simply failed to
        read the data: it records ``crash_check`` = "ok" | "skipped" |
        "crash" plus the per-channel verdicts in partial_data, so a silent
        read failure can never masquerade as a clean scan.
        """
        per_channel: dict[str, str] = {}
        readable_any = False
        crash_result: SkillResult | None = None
        crash_channel: str | None = None
        crash_range: float | None = None

        for channel_index, label in self._resolve_crash_channels(context):
            try:
                arr = self._grab_channel_array(context, channel_index)
            except Exception:
                # This channel blew up — note it and keep probing others.
                per_channel[label] = "error"
                continue
            if arr is None:
                per_channel[label] = "no_data"
                continue
            readable_any = True
            data_range = float(np.ptp(arr))
            has_nan = bool(np.any(np.isnan(arr)))
            if data_range < 1e-25 or has_nan:
                per_channel[label] = "crash"
                if crash_result is None:
                    crash_channel = label
                    crash_range = data_range
                    crash_result = self.fail(
                        "CRASH_DETECTED: scan data on channel "
                        f"{label} has near-zero variance or NaN",
                        crash_indicator=True,
                        data_range=data_range,
                        crash_channel=label,
                    )
            else:
                per_channel[label] = "ok"

        # Record an honest verdict so "couldn't read" is never silently
        # collapsed into "looks fine".
        if crash_result is not None:
            status = "crash"
        elif readable_any:
            status = "ok"
        else:
            status = "skipped"  # no channel was readable — inconclusive

        ex = getattr(self, "_executor", None)
        if ex is not None:
            ex.set_partial("crash_check", status)
            ex.set_partial("crash_check_channels", per_channel)

        if status == "skipped":
            logger.warning(
                "FullScan: post-scan crash check skipped — no channel "
                "returned usable data (%s)", per_channel,
            )
        return crash_result


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(FullScan, context_provider)
