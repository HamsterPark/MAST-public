"""Scan-progress-triggered vision monitor (Phase 9).

When a scan starts, a :class:`ScanVisionMonitor` daemon thread runs alongside
it and, at each 1/8 of the scan (12.5 %, 25 %, … 100 %), grabs the current
(partial) frame, runs the M12 vision model on it, translates the result to a
short Chinese narration with the deterministic ``buffer_summarizer.describe``
templates (NO LLM — instant, no network), and publishes structured results +
the narration into the BufferService so agents and the GUI see live vision.

Design
------
* **Self-terminating**: polls ``Scan_StatusGet``; when the scan finishes it
  fires the 100 % milestone (full coarse + fine + segment) and exits. Also
  exits on the executor abort event or a hard ``max_runtime_s`` guard.
* **Off the graph**: runs in its own daemon thread (never inside a graph.py
  node), so the no-block invariant is preserved. The ~30 s M12 cold-load
  happens in this thread, in parallel with the early scan — never on the loop.
* **Fail-safe**: every Nanonis read / vision call / publish is wrapped; a
  failure logs + skips that milestone, never killing the monitor or the scan.
  If no BufferService is active or vision is unusable, the monitor no-ops.
* **No TCP contention**: status reads use the ``monitor`` role and frame grabs
  the ``data`` role, distinct from ``WaitScanComplete``'s ``main`` role, so the
  per-role connection locks never serialize against the scan-wait loop.
* **Progress is time-estimated**: Nanonis TCP does not expose the current scan
  line, so progress = elapsed / (total_lines × line_time); 100 % is confirmed
  by ``Scan_StatusGet`` going idle.

Vision is ADVISORY only — it never issues hardware commands; all instrument
actions still go through SafetyGate / HITL.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import numpy as np

# SAFE mode suppresses the tip-repair alerts this monitor raises (see the
# call sites in _realtime_check). Unbound holder → False → historical behaviour.
from mast.core.operating_mode import safe_mode_active as _safe_override_active

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
)

# How often to MEASURE the scan front (one frame grab). Progress and every
# milestone key off this measurement, not off Nanonis's time estimate — see
# ScanVisionMonitor._measure_acquired_frac for why the estimate cannot be
# trusted (/ #94). 15 s over a multi-minute scan is a
# few dozen grabs on the dedicated "data" socket: cheap next to the 8 milestone
# grabs it makes correct.
_PROBE_EVERY_S = 15.0

# Fallback only, for an instrument that backfills stale data instead of
# NaN-filling (so the front cannot be measured): if the clock says the scan
# should be finished but Scan_StatusGet says it is still RUNNING, the estimate
# was too short. Extend it rather than pinning frac at 0.999 — a pinned frac is
# exactly what silences the vision pulse for the rest of the scan.
_ESTIMATE_GROWTH = 1.5

# A scan that goes idle is only recorded as a COMPLETED 100 % scan when at least
# this fraction of the buffer is acquired (measured from the NaN front). Below
# it, never-acquired NaN rows remain → the scan was stopped early, and recording
# it as complete is the "未扫完却记为完成" fabrication. The 0.98
# margin absorbs a one-row race at the finish line without ever passing a real
# early-abort (which leaves a large NaN region).
_COMPLETE_FRAC = 0.98


def _parsed_values(rec: Any) -> list | None:
    """Pull the parsed value list out of a NanonisCallRecord.

    Nanonis return_value is ``(error_str, raw_bytes, [parsed...])`` — the data
    lives at index 2. Returns None on any error / unexpected shape."""
    if rec is None or getattr(rec, "error", None):
        return None
    rv = getattr(rec, "return_value", None)
    if isinstance(rv, (list, tuple)) and len(rv) > 2:
        vals = rv[2]
        if isinstance(vals, (list, tuple)):
            return list(vals)
    return None


class ScanVisionMonitor:
    """Background vision monitor for one scan. See module docstring."""

    def __init__(
        self,
        pool: Any,
        *,
        scan_id: str,
        channel: int = 0,
        thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
        poll_interval_s: float = 0.5,
        abort_event: threading.Event | None = None,
        buffer: Any | None = None,
        vision_getter: Callable[[], Any] | None = None,
        translate: Callable[[str, dict], str] | None = None,
        # Backstop against a wedged monitor. The old 1 h default silently killed
        # the monitor mid-scan on any overnight/slow scan (>1 h is the norm for
        # large or slow frames), so progress stopped updating and the 100 %
        # completion milestone never fired. 24 h is a safe
        # ceiling — the monitor still exits promptly on scan-complete or abort.
        max_runtime_s: float = 86400.0,
        # explicit overrides (mostly for tests); read from Nanonis if None
        scan_size_nm: float | None = None,
        total_lines: int | None = None,
        pixels: int | None = None,
        total_time_s: float | None = None,
        time_fn: Callable[[], float] = time.monotonic,
        # 旁白出口（``mast.chat.narration.Sink``）。None = 不发旁白，行为与本参数
        # 存在之前逐字节相同。
        #
        # ⚠️ 为什么是**传进来**的，而不是在这根线程里自己取:``turn_context`` 是
        # ContextVar，靠 ``copy_context()`` 传进 langgraph 的执行器线程 —— 而这里
        # 是一根**裸 threading.Thread**，一个字节都传不进来。在监视器线程里调无参
        # ``narrate()`` 会静默 no-op，而且看起来像「旁白坏了」。
        # 绑定发生在 ``start_scan_vision_monitor`` 里，也就是**起线程之前**。
        narration_sink: Any | None = None,
    ) -> None:
        self._pool = pool
        self._scan_id = scan_id
        self._channel = int(channel)
        self._thresholds = tuple(sorted(thresholds))
        self._poll = float(poll_interval_s)
        self._abort = abort_event
        self._buffer = buffer
        self._vision_getter = vision_getter
        self._translate = translate
        self._max_runtime_s = float(max_runtime_s)
        self._scan_size_nm = scan_size_nm
        self._scan_height_nm: float | None = None
        self._total_lines = total_lines
        self._pixels = pixels
        self._total_time_s = total_time_s
        self._time = time_fn
        self._narration = narration_sink

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fired: set[float] = set()
        # Real-time classical alerts (mid-scan tip-change / feedback oscillation /
        # bad scan-lines) are emitted at most ONCE each per scan — dedup here.
        self._rt_alerted: set[str] = set()
        self._vision: Any = None
        self._channel_resolved: int | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> "ScanVisionMonitor":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name=f"scan-vision-{self._scan_id}", daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=join_timeout)

    def _aborted(self) -> bool:
        if self._stop.is_set():
            return True
        return bool(self._abort is not None and self._abort.is_set())

    # ── Nanonis reads (best-effort) ──────────────────────────────────

    def _safe_call(self, method: str, *args, role: str = "monitor"):
        try:
            return self._pool.safe_call(method, *args, role=role)
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: %s failed: %s", method, exc)
            return None

    def _scan_running(self) -> bool | None:
        """True if scanning, False if idle, None if unknown (read failed)."""
        vals = _parsed_values(self._safe_call("Scan_StatusGet", role="monitor"))
        if not vals:
            return None
        try:
            return int(vals[0]) != 0
        except (ValueError, TypeError):
            return None

    def _read_geometry(self) -> None:
        """Populate scan_size_nm / total_lines / pixels / total_time_s from
        Nanonis where not already provided. Each read is independent + lenient."""
        if self._scan_size_nm is None or self._scan_height_nm is None:
            vals = _parsed_values(self._safe_call("Scan_FrameGet", role="monitor"))
            if vals and len(vals) >= 3:
                try:
                    self._scan_size_nm = float(vals[2]) * 1e9  # width_m → nm
                except (ValueError, TypeError):
                    pass
            # 同时读取扫描高度，才能识别物理长宽比。
            # 方形像素数组未必代表方形区域；在几何退化的输入上不能沿用方图模型的针尖判断。
            if vals and len(vals) >= 4:
                try:
                    self._scan_height_nm = float(vals[3]) * 1e9  # height_m → nm
                except (ValueError, TypeError):
                    pass
        if self._total_lines is None or self._pixels is None:
            vals = _parsed_values(self._safe_call("Scan_BufferGet", role="monitor"))
            # [num_channels, [channel_indexes], pixels, lines]
            if vals and len(vals) >= 4:
                try:
                    if self._pixels is None:
                        self._pixels = int(vals[2])
                    if self._total_lines is None:
                        self._total_lines = int(vals[3])
                except (ValueError, TypeError):
                    pass
        if self._total_time_s is None:
            vals = _parsed_values(self._safe_call("Scan_SpeedGet", role="monitor"))
            # [fwd_speed, bwd_speed, fwd_time_s, bwd_time_s, keep_const, ratio]
            fwd_t = bwd_t = 0.0
            if vals and len(vals) >= 4:
                try:
                    fwd_t = float(vals[2]); bwd_t = float(vals[3])
                except (ValueError, TypeError):
                    pass
            lines = self._total_lines or 256
            per_line = (fwd_t + bwd_t) if (fwd_t + bwd_t) > 0 else 0.5
            self._total_time_s = max(1.0, lines * per_line)

    def _resolve_channel(self) -> int:
        """Pick the channel id to grab. Scan_FrameDataGrab's first arg is a
        GLOBAL channel id that must be in the acquired selection (from
        Scan_BufferGet). Prefer the topography channel ("Z (m)") since M12 was
        trained on height maps, not Current; fall back to the configured /
        first acquired channel. Cached after the first resolve."""
        if self._channel_resolved is not None:
            return self._channel_resolved
        ch = self._channel
        try:
            from mast.io.nanonis_files import channel_ids_from_buffer
            vals = _parsed_values(self._safe_call("Scan_BufferGet", role="data"))
            # vals = [num_channels, [(id,), (id,)...], pixels, lines] — the
            # 1-tuple unpack used to be inlined here. It was RIGHT here and wrong
            # in three other places (SetScanBuffer crashed on it, v6.1.1), so the
            # judgement now lives in exactly one place: channel_ids_from_buffer.
            ids = channel_ids_from_buffer(vals)
            if ids:
                ch = ids[0]  # default: first acquired channel
                for cid in ids:
                    rec = self._safe_call("Scan_FrameDataGrab", cid, 1, role="data")
                    pv = _parsed_values(rec)
                    name = next((x for x in (pv or []) if isinstance(x, str)), "")
                    nl = name.lower()
                    if any(k in nl for k in ("z (", "z [", "z(", "height", "topo")):
                        ch = cid  # topography channel — best for M12
                        break
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: channel resolve failed (%s); using %s", exc, ch)
        self._channel_resolved = ch
        logger.info("scan-vision: using channel id %s for M12", ch)
        return ch

    def _measure_acquired_frac(self) -> float | None:
        """The TRUE fraction of the frame acquired so far — measured, not estimated.

        Nanonis fills UNACQUIRED scan-buffer rows with NaN during a live scan, so
        the NaN front IS the scan front. Counting the rows that are not all-NaN is
        therefore an exact measurement of progress, and it costs one frame grab.

        Returns None when the measurement is not usable — the grab failed, or the
        frame has NO NaN rows at all (which is ambiguous: either the scan just
        finished, or this instrument backfills stale data instead of NaN-filling).
        The caller then falls back to the time estimate.

        WHY THIS EXISTS. The monitor used to schedule
        every milestone off ``elapsed / total_time`` — a Nanonis time ESTIMATE.
        When the estimate runs short (routinely), that ratio saturates: ``frac``
        pins at 0.999, all seven partial milestones burn inside the first slice of
        the real scan, and the vision pulse then goes SILENT for the rest of it.
        The operator's report is the bug read back verbatim: "读取了第一张 1/4 的
        图像之后，就不再读取新扫的图了，一直分析这张图，直到扫描结束才分析了一张
        全图。" (#94's "511/512" is the same bug's fingerprint — round(0.999·512).)
        A clock cannot know how far a scan has got. The scan buffer can.
        """
        arr = self._extract_frame(
            _parsed_values(self._safe_call(
                "Scan_FrameDataGrab", self._resolve_channel(), 1, role="data")),
            self._pixels or 0,
        )
        if arr is None or arr.size == 0 or arr.ndim != 2:
            return None
        nan_rows = np.isnan(arr).all(axis=1)
        if not nan_rows.any():
            return None          # ambiguous — never claim 100 % from this
        h = int(arr.shape[0])
        return (int((~nan_rows).sum()) / h) if h else None

    #: 物理长宽比超过这个数就不判针尖。细条是 20:1,真实矩形扫描少见超过 2:1。
    DEGENERATE_ASPECT = 4.0

    def _degenerate_frame(self) -> bool:
        """这一帧的几何是不是退化到「上面没有二维形貌可判」。

        **读不到就返回 False(不拦)** —— 不知道几何时沉默放行,比沉默拦住更接近
        「不猜」:拦住会让一台读不回 `Scan_FrameGet` 的机器永远拿不到视觉判断,
        而那是一个**因为不知道而产生的沉默失效**,正是本仓最贵的那一类。
        """
        w, h = self._scan_size_nm, self._scan_height_nm
        try:
            if not w or not h or w <= 0 or h <= 0:
                return False
            ratio = max(float(w) / float(h), float(h) / float(w))
            return ratio > self.DEGENERATE_ASPECT
        except (TypeError, ValueError, ZeroDivisionError):
            return False

    def _grab_frame(self, frac: float) -> np.ndarray | None:
        """Grab fwd+bwd for the topography channel → (2, H, W); zero rows past frac."""
        pixels = self._pixels or 0
        ch = self._resolve_channel()
        rec_f = self._safe_call("Scan_FrameDataGrab", ch, 1, role="data")
        rec_b = self._safe_call("Scan_FrameDataGrab", ch, 0, role="data")
        vals_f = _parsed_values(rec_f)
        fwd = self._extract_frame(vals_f, pixels)
        bwd = self._extract_frame(_parsed_values(rec_b), pixels)
        if fwd is None and bwd is None:
            return None
        if fwd is None:
            fwd = bwd
        if bwd is None:
            bwd = fwd
        h = min(fwd.shape[0], bwd.shape[0])
        w = min(fwd.shape[1], bwd.shape[1])
        fwd, bwd = fwd[:h, :w], bwd[:h, :w]
        # CRITICAL: Nanonis fills UNACQUIRED scan-buffer rows with NaN (not 0)
        # during a live scan. NaN poisons the whole M12 forward (→ NaN logits →
        # garbage). Sanitise to 0 first — this both removes the NaN and zeroes
        # the not-yet-scanned region (the actual scan front), so the model sees
        # [real acquired rows | zeros] exactly like a partial scan.
        had_nan = bool(np.isnan(fwd).any() or np.isnan(bwd).any())
        fwd = np.nan_to_num(fwd.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        bwd = np.nan_to_num(bwd.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # Time-estimate zeroing is ONLY a fallback for instruments that fill
        # unacquired rows with STALE data instead of NaN. When NaN-filling worked
        # (had_nan), the frame ALREADY has the correct [acquired | zeros] split —
        # applying a low time estimate on top blanked REAL rows at early
        # milestones, feeding the model a mostly-zero garbage frame (review
        # 2026-07-03). And honour the scan DIRECTION: a down-scan acquires the TOP
        # rows first (zero the bottom); an up-scan acquires the BOTTOM first.
        if not had_nan:
            n_acq = max(1, int(round(frac * h)))
            if n_acq < h:
                fwd = fwd.copy(); bwd = bwd.copy()
                scan_up = self._frame_direction(vals_f) == 1
                if scan_up:
                    fwd[: h - n_acq] = 0.0
                    bwd[: h - n_acq] = 0.0
                else:
                    fwd[n_acq:] = 0.0
                    bwd[n_acq:] = 0.0
        return np.stack([fwd, bwd])

    @staticmethod
    def _frame_direction(vals: list | None) -> int:
        """Scan direction from the Scan_FrameDataGrab reply
        ``[name_len, name, rows, cols, data_2D, direction]`` — 1=up, 0=down.
        Defaults to 0 (down) when absent."""
        if isinstance(vals, (list, tuple)) and len(vals) >= 6:
            try:
                return int(vals[5])
            except (TypeError, ValueError):
                return 0
        return 0

    @staticmethod
    def _extract_frame(vals: list | None, pixels: int) -> np.ndarray | None:
        """Extract the (H,W) data array from a Scan_FrameDataGrab parsed list.

        Nanonis returns ``[name_len, name, rows, cols, data_2D, direction]`` —
        a HETEROGENEOUS list whose data is a 2-D ndarray element (NOT a flat
        list at a fixed index, and NOT ravel-able as a whole). Prefer that 2-D
        element; fall back to reshaping a flat numeric list (stub / flat
        instruments) via :meth:`_reshape`."""
        if not vals:
            return None
        for el in vals:
            if isinstance(el, (str, bytes, int, float, bool)):
                continue
            try:
                a = np.asarray(el, dtype=np.float64)
            except Exception:  # noqa: BLE001
                continue
            if a.ndim == 2 and a.size > 0:
                return a
            if a.ndim == 1 and pixels and a.size >= pixels and a.size % pixels == 0:
                return a.reshape(-1, pixels)
        # Fallback: the whole list is a flat numeric sequence (stub / flat fmt).
        try:
            return ScanVisionMonitor._reshape([float(x) for x in vals], pixels)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _reshape(vals: list | None, pixels: int) -> np.ndarray | None:
        if not vals:
            return None
        arr = np.asarray(vals, dtype=np.float64).ravel()
        if arr.size == 0:
            return None
        if pixels and pixels > 0 and arr.size % pixels == 0:
            return arr.reshape(-1, pixels)
        # Fall back to a near-square reshape if pixels unknown.
        side = int(np.sqrt(arr.size))
        if side >= 2 and side * side <= arr.size:
            return arr[: side * side].reshape(side, side)
        return arr.reshape(1, -1)

    # ── buffer publish (thread-safe; no-op if no buffer) ─────────────

    def _buf(self):
        if self._buffer is not None:
            return self._buffer
        try:
            from mast.buffer.active import get_active_buffer
            return get_active_buffer()
        except Exception:  # noqa: BLE001
            return None

    def _describe(self, kind: str, payload: dict) -> str:
        if self._translate is not None:
            try:
                return self._translate(kind, payload)
            except Exception:  # noqa: BLE001
                pass
        try:
            from mast.agents.buffer_summarizer.node import describe
            return describe(kind, payload)
        except Exception:  # noqa: BLE001
            return f"{kind}: {payload.get('label') or payload.get('coarse_label') or ''}"

    # ── the run loop ─────────────────────────────────────────────────

    def _run(self) -> None:
        t0 = self._time()
        try:
            self._read_geometry()
        except Exception as exc:  # noqa: BLE001
            logger.warning("scan-vision: geometry read failed: %s", exc)
        # Warm the vision model in THIS thread (parallel to the early scan).
        try:
            getter = self._vision_getter
            if getter is None:
                from mast.vision.module import VisionModule
                getter = VisionModule.get
            self._vision = getter()
        except Exception as exc:  # noqa: BLE001
            logger.warning("scan-vision: VisionModule unavailable (%s); monitor exits", exc)
            return

        total_time = self._total_time_s or 60.0
        partial_thr = [t for t in self._thresholds if t < 1.0]
        saw_idle = False
        completed = False          # scan REALLY finished (not stopped early)
        last_frac = 0.0            # published progress never goes backwards
        next_probe = 0.0           # measure the scan front on its own cadence
        measured: float | None = None
        while True:
            # _stop = superseded / StopScan; abort_event = operator abort.
            if self._aborted():
                break
            elapsed = self._time() - t0
            if elapsed > self._max_runtime_s:
                break  # hard backstop only — never the progress estimate
            running = self._scan_running()
            if running is False:
                saw_idle = True  # the scan STOPPED — finished OR aborted early
                # Distinguish the two with a fresh NaN-front measurement so an
                # early-stopped scan is never recorded as a completed 100 % scan
                #. Positive evidence of incompleteness only —
                # a normal finish (buffer filled → no NaN) still confirms.
                completed = self._confirm_complete()
                break

            # ── Where is the scan REALLY? ──
            # Measure the scan front from the buffer's NaN boundary. A time
            # estimate cannot do this: when Nanonis's estimate runs short,
            # elapsed/total_time saturates at 0.999, every partial milestone
            # burns inside the first slice of the real scan, and the vision pulse
            # goes silent for the rest of it. Probing costs one frame grab, so it
            # runs on its own (slower) cadence, not on every poll.
            if elapsed >= next_probe:
                next_probe = elapsed + _PROBE_EVERY_S
                m = self._measure_acquired_frac()
                if m is not None:
                    measured = m
                    if m > 0.02 and elapsed > 0:
                        # Re-anchor the estimate to reality so the ETA and the
                        # remaining milestones are both honest from here on.
                        total_time = elapsed / m

            if measured is not None:
                frac = min(0.999, measured)
            else:
                # No usable measurement (instrument backfills stale data instead
                # of NaN-filling). Fall back to the clock — but never let it
                # saturate while Scan_StatusGet still says RUNNING: that is proof
                # the estimate was too short, and a pinned frac is what silences
                # the pulse.
                if total_time > 0 and elapsed >= total_time:
                    total_time = elapsed * _ESTIMATE_GROWTH
                    logger.info(
                        "scan-vision: time estimate ran short (%.0fs elapsed, scan "
                        "still running) — extending to %.0fs", elapsed, total_time)
                frac = min(0.999, elapsed / total_time) if total_time > 0 else 0.0

            last_frac = max(last_frac, frac)
            self._publish_progress(last_frac, elapsed, total_time)

            # ONE milestone per poll. Firing every due threshold in a single pass
            # is what let a saturated estimate burn the whole budget at once; one
            # at a time, each grab is separated by a fresh measurement.
            due = next((t for t in partial_thr
                        if last_frac >= t and t not in self._fired), None)
            if due is not None:
                self._fired.add(due)
                self._milestone(due, final=False,
                                ordinal=partial_thr.index(due) + 1)
            self._stop.wait(self._poll)

        # The 100 % milestone (full coarse+fine+segment) fires ONLY when the
        # scan actually COMPLETED — NOT on an operator abort, a supersede/stop,
        # the timeout backstop, and NOT when Scan_StatusGet went idle because the
        # scan was stopped EARLY (confirmed by the NaN front, ).
        # Those must never narrate a bogus "scan complete".
        if saw_idle and completed:
            # Loop-side progress is capped at frac 0.999 (an estimate must never
            # claim completion), so the last published line_idx is
            # round(0.999*lines) — e.g. 511/512 — and stays there forever. Now
            # that completion is confirmed, publish the authoritative 100 % so the
            # GUI bar and get_scan_progress() reach lines/lines.
            elapsed = self._time() - t0
            self._publish_progress(1.0, elapsed, min(total_time, elapsed))
            if 1.0 not in self._fired:
                self._fired.add(1.0)
                self._milestone(1.0, final=True, ordinal=8)
        elif saw_idle and not completed:
            # Stopped early: record the HONEST partial state (real fraction + a
            # WARN "ended early" event), never a fake completion. Nothing
            # downstream (records / 近期帧 borrowing) can then read it as done.
            elapsed = self._time() - t0
            self._publish_progress(last_frac, elapsed, min(total_time, elapsed))
            self._emit_incomplete(last_frac)

    def _confirm_complete(self) -> bool:
        """True if the just-idled scan really FINISHED; False if stopped early.

        The instrument-honest test is the scan buffer's NaN front: rows still
        all-NaN were never rastered, so their presence is positive proof the
        scan did not complete. A FRESH grab at the idle edge is authoritative.
        When the front cannot be measured (grab failed, buffer fully filled, or
        an instrument that backfills stale data instead of NaN-filling) we return
        True — we must not fabricate an "aborted" verdict on a scan that may well
        have finished. So this only ever DEMOTES a completion on real evidence,
        never falsely demotes a genuine finish."""
        fresh = self._measure_acquired_frac()
        if fresh is None:
            return True                    # unmeasurable / filled → assume done
        return fresh >= _COMPLETE_FRAC

    def _emit_incomplete(self, frac: float) -> None:
        """Record a scan that STOPPED EARLY honestly — a WARN 'ended early' event
        with the true partial frame, never a SCAN_COMPLETE.

        Uses the already-warmed frame grab + PNG persist (no extra model
        inference — the partial milestones already ran vision), so it is cheap
        and fail-safe. kind=FEATURE_OF_INTEREST (never SCAN_COMPLETE) means the
        近期帧 borrowing never lends it a full-scan .sxm as if it had completed."""
        buf = self._buf()
        if buf is None:
            return
        pct = int(round(max(0.0, min(1.0, frac)) * 100))
        frame_path = None
        try:
            raw = self._grab_frame(frac)
            if raw is not None:
                frame_path = self._persist_frame_png(
                    self._adapt_frame_for_backend(self._vision, raw), 8)
        except Exception:  # noqa: BLE001 — the record must survive a grab failure
            frame_path = None
        try:
            from mast.buffer.schemas import Severity, VisionEvent, VisionEventType
            payload = {
                "milestone": round(frac, 3),
                "scan_id": self._scan_id,
                "incomplete": True,
                "summary_zh": f"扫描提前结束，仅完成约 {pct}%（未记为完成扫描）",
            }
            if frame_path:
                payload["frame_path"] = frame_path
            buf.emit_event(VisionEvent(
                seqno=buf.next_seq(), kind=VisionEventType.FEATURE_OF_INTEREST,
                severity=Severity.WARN, payload=payload,
                cause_ref=f"scan#{self._scan_id}",
            ))
            # 「提前停下」和「扫完了」必须是**两句话**（# 「停止」≠「达标」）。这里发的是 scan_stopped_early，模板里连
            # 「没有记为完成」都写进句子了 —— 让用户一眼看出这不是一次成功。
            if self._narration is not None:
                self._narration.narrate(
                    "scan_stopped_early", frac=float(frac),
                    scan_id=self._scan_id,
                    image=({"src": frame_path, "origin": "milestone_png"}
                           if frame_path else None))
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: emit incomplete failed: %s", exc)

    @staticmethod
    def _accepts_channel_pair(vm: Any) -> bool:
        """True if the active vision backend understands a (2,H,W) trace/retrace
        pair (the M12 / VIGIL backend). Legacy / Mock backends do not — they take
        a single (H,W) channel. Detected by the backend's _to_fwd_bwd adapter
        (VIGILBackend-only); duck-typed so it survives backend swaps/fallbacks."""
        be = getattr(vm, "_backend", vm)
        return callable(getattr(be, "_to_fwd_bwd", None))

    def _adapt_frame_for_backend(self, vm: Any, frame: np.ndarray) -> np.ndarray:
        """Hand the (2,H,W) pair straight to a pair-aware backend; otherwise
        collapse to the forward (topography) channel as a single (H,W) image."""
        a = np.asarray(frame)
        if a.ndim == 3 and a.shape[0] == 2 and not self._accepts_channel_pair(vm):
            return a[0]  # forward channel only — universally accepted (H,W)
        return a

    def _milestone(self, frac: float, *, final: bool, ordinal: int) -> None:
        """Run vision at this milestone + publish. Fully guarded.

        ``ordinal`` is the 1-of-8 milestone index (1..8) used for frame_idx so
        it stays injective (round(frac*8) collides 0.999 and 1.0 at 8)."""
        try:
            # 旁白按帧被观察的时刻排序，而非推理完成后发言的时刻。
            # 使用与其他转录行相同来源的 wall clock，不能混用 monotonic 或局部假时钟。
            # 时钟引用必须在当前作用域可用，避免异常被外层捕获后静默丢失里程碑。
            seen_at = time.time()
            frame = self._grab_frame(frac)
            if frame is None:
                logger.debug("scan-vision: no frame at %.0f%%", frac * 100)
                return
            # 几何退化的帧应弃权，不判针尖不好。
            # 判断依据物理长宽比而非像素形状；配置阈值只是软件启发式，需按目标输入验证。
            # 当前实现高度未知时不阻断视觉分析，同时不能声称已经确认几何正常。
            if self._degenerate_frame():
                logger.info(
                    "scan-vision: 帧几何退化(%.3g nm × %.3g nm,%.0f:1)—— "
                    "**不做针尖判定**。这种细条上没有二维形貌可判,"
                    "在它上面出的判决是错误信息,不是判决。",
                    self._scan_size_nm or float("nan"),
                    self._scan_height_nm or float("nan"),
                    (self._scan_size_nm or 0) / max(self._scan_height_nm or 1e-9, 1e-9))
                return
            vm = self._vision
            # Real-time network-free check on the RAW fwd/bwd acquired region
            # (BEFORE the single-channel backend adaptation below): catch a mid-scan
            # tip change / feedback oscillation / bad scan-lines DURING the scan so
            # the operator/agent can abort early instead of burning the whole frame.
            # Emits its own deduped WARN alerts; returns a signal dict for the event.
            rt = self._realtime_check(frame, frac)
            # _grab_frame yields the (2,H,W) trace/retrace pair the M12 (VIGIL)
            # backend wants. Legacy / Mock backends only understand a single
            # (H,W) channel — handing them (2,H,W) crashes (legacy raises on
            # ndim==3, mock mis-reads shape[0] as height). Collapse to the
            # forward (topography) channel for non-VIGIL backends so the monitor
            # is fail-safe across every backend.
            raw_frame = frame                       # (2,H,W) before single-channel collapse
            frame = self._adapt_frame_for_backend(vm, frame)
            if self._scan_size_nm and self._scan_size_nm > 0:
                try:
                    vm.set_scan_size_nm(self._scan_size_nm)
                except Exception:  # noqa: BLE001
                    pass
            buf = self._buf()
            # Always: coarse tip quality (the early-abort signal).
            coarse = vm.assess_tip_coarse(frame)
            self._publish_coarse(buf, coarse, ordinal)
            # Segmentation (where features/defects are forming) — the network-free
            # CLASSICAL scale-adaptive 4-class segmenter on the ACQUIRED region,
            # NOT the deployed Head C: it does not hallucinate contamination on
            # clean lattices (the C head does — see the diagnostic) and needs no
            # model. Result is padded back to the full frame shape for the GUI
            # overlay; the decision summary (presence / counts / coverage — the
            # readout autonomy should consume) rides the milestone event.
            try:
                seg, seg_summary = self._segment_classical_frame(raw_frame)
                if seg is not None:
                    self._publish_segment(buf, seg, frac, final)
                if seg_summary is not None:
                    rt = dict(rt or {})
                    rt["surface"] = seg_summary
            except Exception as exc:  # noqa: BLE001
                logger.debug("scan-vision: segment failed at %.0f%%: %s", frac * 100, exc)
            # Fine morphology only makes sense on a (near-)complete image.
            fine = None
            if final:
                try:
                    fine = vm.assess_tip_fine(frame)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("scan-vision: fine failed: %s", exc)
            # Persist the EXACT analysed frame so the UI shows the true
            # partial image for this pulse, not a borrowed stale scan.
            frame_path = self._persist_frame_png(frame, ordinal)
            self._emit_milestone_event(buf, frac, final, coarse, fine,
                                       frame_path=frame_path, rt=rt,
                                       seen_at=seen_at)
        except Exception as exc:  # noqa: BLE001
            logger.warning("scan-vision: milestone %.0f%% failed: %s", frac * 100, exc)
            self._emit_error(exc)

    # ── real-time network-free detectors (mid-scan warnings) ──────────

    @staticmethod
    def _acquired_crop(raw: np.ndarray) -> np.ndarray | None:
        """Crop a (2,H,W) grab to the contiguous ACQUIRED rows (non-zero; the
        unscanned region is zeroed by _grab_frame). Returns None if too few rows."""
        if raw is None or raw.ndim != 3 or raw.shape[0] < 1:
            return None
        active = np.abs(raw).sum(axis=(0, 2)) > 0        # (H,) rows with any signal
        idx = np.where(active)[0]
        if idx.size < 40:                                 # need enough rows for a split
            return None
        return raw[:, int(idx.min()):int(idx.max()) + 1, :]

    def _segment_classical_frame(self, raw: np.ndarray):
        """Classical 4-class segmentation (terrace/step/defect/contam) of the
        ACQUIRED region, padded back to the full frame shape (unscanned rows →
        terrace). Scale-adaptive (2026-07-27) — validated on physics ground
        truth; never hallucinates contamination on a clean lattice the way the
        deployed Head C does. Returns (SegmentationResult, decision-summary
        dict) or (None, None). The summary (presence / physically-merged object
        counts / area fractions) is what autonomy should consume — NOT pixel
        masks of sparse targets."""
        try:
            from mast.vision.classical_seg import CLASSES, class_counts, segment_classical
            from mast.vision.module import SegmentationResult
            from mast.vision.seg_utils import encode_rle

            full = raw[0] if (raw is not None and raw.ndim == 3) else raw
            if full is None or full.ndim != 2:
                return None, None
            H, W = full.shape
            idx = np.where((np.abs(full) > 0).any(axis=1))[0]     # acquired rows
            if idx.size < 15:                                     # segmentation needs
                return None, None                                 # far fewer rows than a change-point split
            r0, r1 = int(idx.min()), int(idx.max()) + 1
            nm_per_px = (float(self._scan_size_nm) / W
                         if (self._scan_size_nm and self._scan_size_nm > 0) else None)
            seg_crop = segment_classical(full[r0:r1], nm_per_px).astype(np.uint8)
            seg_full = np.zeros((H, W), np.uint8)
            seg_full[r0:r1] = seg_crop
            summary = None
            try:
                from mast.vision.seg_scale_adaptive import summarize_segmentation
                summary = summarize_segmentation(seg_crop, nm_per_px)
            except Exception:  # noqa: BLE001
                pass
            result = SegmentationResult(
                mask_rle=encode_rle(seg_full), shape=(H, W),
                class_counts=class_counts(seg_full), level=0, classes=list(CLASSES),
                tipflag_stability_rle=b"", tipflag_transition_rle=b"")
            return result, summary
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: classical segment failed: %s", exc)
            return None, None

    def _realtime_check(self, raw: np.ndarray, frac: float) -> dict:
        """Run the vision capabilities that apply to a live scan frame on the
        acquired region + emit deduped mid-scan alerts. Returns a compact signal
        dict for the milestone event. Fully guarded — never raises, never blocks
        the monitor.

        Wired capabilities (per docs/v2/benchmarks/vision_v25_diagnostic/):
          * tip_change   — mid-scan tip change (CRITICAL → drives IC auto-abort/repair)
          * scan_artifacts — feedback oscillation / drift / bad scan-lines (WARN)
          * double_tip   — mid-scan double/multi tip (WARN)
          * tip_metrics  — FFT sharpness / resolution / fwd-bwd instability (data)
          * assess_quality — learned stm_quality_v1 score (strongest real signal; WARN on bad)
        (assess_iz/assess_iv need spectra, not a scan frame, so are N/A here.)
        """
        out: dict = {}
        # The PNG of THIS frame — written at most once no matter how many alerts
        # fire on it, and not at all when none do.
        #
        # #20/#25: every mid-scan alert used to read "反馈振荡/振铃(55
        # 周期/行)——50% 处" or "其后行不可信" with no picture attached.
        # ``_emit_alert`` was the ONE publisher that never called
        # ``_persist_frame_png``, so a verdict could be issued about an image
        # nobody had any way to see.
        png_cell: list = []

        # Read the mode ONCE per check, so a switch landing mid-check cannot make
        # one alert suppressed and the next one not, within the same frame.
        safe = _safe_override_active()

        def _alert_png() -> "str | None":
            if not png_cell:
                try:
                    png_cell.append(self._persist_frame_png(
                        self._adapt_frame_for_backend(self._vision, raw),
                        int(round(max(0.0, min(1.0, frac)) * 100))))
                except Exception:  # noqa: BLE001 — an alert must survive this
                    png_cell.append(None)
            return png_cell[0]

        try:
            crop = self._acquired_crop(raw)
            if crop is None:
                return out
            from mast.vision.classical_thresholds import get_classical_thresholds
            from mast.vision.double_tip import detect_double_tip
            from mast.vision.scan_artifacts import detect_scan_artifacts
            from mast.vision.tip_change import detect_tip_change
            from mast.vision.tip_metrics import assess_tip_classical

            # live-read per-instrument thresholds (设置 → effective NEXT check;
            # they were previously only wired into tip_quality, so an operator
            # retune never reached the mid-scan alerts — recon 2026-07-27)
            cth = get_classical_thresholds()

            # crop is (2, H_acquired, W): the pixel pitch comes from the WIDTH
            # (shape[2]) — shape[1] is the acquired row count and shrinks with
            # scan progress, which mis-binned early partial frames into the
            # meso calibration table.
            nm_per_px = (float(self._scan_size_nm) / crop.shape[2]
                         if (self._scan_size_nm and self._scan_size_nm > 0) else None)
            fwd = crop[0]

            # ── mid-scan tip change v2 (the IC auto-abort/repair trigger) ──
            # Null-calibrated lag-k detector (AUC 0.97 on visible events at
            # FPR 5 %); the old max-t baseline was random (AUC 0.51) with its
            # detections bought by an 83 % false-alarm rate — an autonomous
            # CRITICAL that aborts scans must not cry wolf.
            tc = detect_tip_change(
                crop,
                threshold=(cth.tq_change_threshold if cth.tq_change_threshold > 0 else None),
                nm_per_px=nm_per_px)
            # Nanonis Z is metres → the frame's own detection limit in pm.
            # Sanity-gate the unit: an LOD above 1 µm means the input was not
            # in metres (test rigs / arbitrary units) — no pm claim then.
            lod_pm = (float(tc.lod) * 1e12
                      if (tc.lod is not None and float(tc.lod) < 1e-6) else None)
            out["tip_change"] = {"changed": bool(tc.changed), "row": tc.change_row,
                                 "score": round(float(tc.score), 1),
                                 "lod_pm": (round(lod_pm, 1) if lod_pm is not None else None)}
            # SAFE mode: this CRITICAL is the "abort the scan and repair the tip"
            # trigger — exactly what SAFE promises not to do. Demote to INFO so the
            # record and the GUI still show it happened, but nothing halts and no
            # HITL interrupt fires. Its dedup key is SEPARATE: if the operator
            # switches back to auto/semi mid-scan, the same scan must still be able
            # to raise the real CRITICAL.
            tc_key = "tip_change@safe" if safe else "tip_change"
            if tc.changed and tc_key not in self._rt_alerted:
                self._rt_alerted.add(tc_key)
                nrow = crop.shape[1]
                if safe:
                    self._emit_alert(
                        "tip_change", "info",
                        f"（安全模式）扫描中途检出针尖状态突变（已采集 {nrow} 行中第 "
                        f"~{tc.change_row} 行，校准 z={tc.score:.1f}）——{int(frac * 100)}% 处。"
                        f"安全模式下不修针、不中止，仅记录。",
                        {"change_row": tc.change_row, "score": round(float(tc.score), 1),
                         "threshold": round(float(tc.threshold), 1),
                         "lod_pm": (round(lod_pm, 1) if lod_pm is not None else None),
                         "milestone": round(frac, 3),
                         "safe_mode_suppressed": True},
                        frame_png=_alert_png())
                else:
                    self._emit_alert(
                        "tip_change", "critical",
                        f"扫描中途针尖状态突变（已采集 {nrow} 行中第 ~{tc.change_row} 行，"
                        f"校准 z={tc.score:.1f}/阈值 {tc.threshold:.0f}）"
                        f"——{int(frac * 100)}% 处，其后行不可信，建议中止扫描并修针尖",
                        {"change_row": tc.change_row, "score": round(float(tc.score), 1),
                         "threshold": round(float(tc.threshold), 1),
                         "lod_pm": (round(lod_pm, 1) if lod_pm is not None else None),
                         "milestone": round(frac, 3),
                         "recommend": ["StopScan", "ConditionTip"]},
                        frame_png=_alert_png())
            elif not tc.changed and lod_pm is not None:
                # A falsifiable negative: "nothing seen AND this frame was
                # sensitive to row-DC jumps ≥ lod_pm" — single-atom steps are
                # ~200+ pm, typical contamination z-offsets tens of pm.
                out["tip_change"]["negative_statement"] = (
                    f"未检出针尖状态突变（本帧对 ≥{lod_pm:.0f} pm 的 z 跳变敏感）")

            # ── feedback oscillation / drift / bad scan-lines ──
            sa = detect_scan_artifacts(crop)
            out["artifacts"] = {"oscillation": bool(sa.oscillation),
                                "drift_px": sa.drift_px, "bad_row_frac": round(sa.bad_row_frac, 3)}
            if sa.oscillation and "oscillation" not in self._rt_alerted:
                self._rt_alerted.add("oscillation")
                cyc = sa.oscillation_cycles_per_line
                self._emit_alert(
                    "oscillation", "warn",
                    f"反馈振荡/振铃（{cyc:.0f} 周期/行）——{int(frac * 100)}% 处，建议降增益/调反馈参数"
                    if cyc else "反馈振荡/振铃——建议调反馈参数",
                    {"cycles_per_line": cyc, "severity": round(float(sa.oscillation_severity), 1),
                     "milestone": round(frac, 3)},
                    frame_png=_alert_png())
            if sa.bad_row_frac > 0.10 and "bad_lines" not in self._rt_alerted:
                self._rt_alerted.add("bad_lines")
                self._emit_alert(
                    "bad_lines", "warn",
                    f"坏扫描行 {sa.bad_row_frac:.0%}（尖峰/丢线）——{int(frac * 100)}% 处",
                    {"bad_row_frac": round(float(sa.bad_row_frac), 3), "milestone": round(frac, 3)},
                    frame_png=_alert_png())

            # ── double / multi tip: DATA ONLY, no alert (demoted 2026-07-27) ──
            # Physics-truth validation killed this as an alarm: AUC 0.442 on
            # VIGIL labels, 0.625 even on physically-correct injected ghosts
            # (misses ~4 of 5 at FPR 5 %; small separations are principally
            # unresolvable — the ghost overlaps its own source). Keeping the
            # WARN would be manufacturing false "repair the tip" advice. The
            # score stays visible as advisory data for a human who asks.
            dt = detect_double_tip(crop, nm_per_px=nm_per_px,
                                   threshold=cth.tq_double_threshold)
            out["double_tip"] = {"is_double": bool(dt.is_double),
                                 "score": round(float(dt.score), 2),
                                 "separation_nm": dt.separation_nm,
                                 "reliability": "unreliable"}

            # ── cheap tip-quality signals (data only, no alert) ──
            try:
                m = assess_tip_classical(crop, nm_per_px=nm_per_px)
                out["metrics"] = {"fft_sharpness": None if m.fft_sharpness is None else round(m.fft_sharpness, 1),
                                  "resolution_nm": m.resolution_nm, "has_lattice": bool(m.has_lattice),
                                  # tier judgement only, never a radius measurement;
                                  # sharpness_scale gates the Bragg-family reading
                                  "sharpness_scale": m.sharpness_scale,
                                  "circularity_dev": (None if m.circularity_dev is None
                                                      else round(m.circularity_dev, 3)),
                                  "fwd_bwd_instability": (None if m.fwd_bwd_instability is None
                                                          else round(m.fwd_bwd_instability, 3))}
            except Exception:  # noqa: BLE001
                pass

            # ── learned stm_quality_v1 (strongest real-data quality signal; optional) ──
            try:
                q = self._vision.assess_quality(fwd, scan_size_nm=self._scan_size_nm)
                out["learned_quality"] = {"score": round(float(q.score), 2), "tier": q.tier}
                # Same SAFE treatment as tip_change: the score stays visible as
                # data, the "your tip is bad" alert does not. Separate dedup key.
                lq_key = "learned_bad@safe" if safe else "learned_bad"
                if q.tier == "bad" and lq_key not in self._rt_alerted:
                    self._rt_alerted.add(lq_key)
                    if safe:
                        self._emit_alert(
                            "learned_quality", "info",
                            f"（安全模式）学习质量评分低（tier=bad, {q.score:.2f}）"
                            f"——{int(frac * 100)}% 处，仅记录，不修针。",
                            {"score": round(float(q.score), 2), "tier": q.tier,
                             "milestone": round(frac, 3), "safe_mode_suppressed": True},
                            frame_png=_alert_png())
                    else:
                        self._emit_alert(
                            "learned_quality", "warn",
                            f"学习质量评分低（tier=bad, {q.score:.2f}）——{int(frac * 100)}% 处",
                            {"score": round(float(q.score), 2), "tier": q.tier,
                             "milestone": round(frac, 3)},
                            frame_png=_alert_png())
            except Exception:  # noqa: BLE001 — scorer optional (weights/backbone may be absent)
                pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: realtime check failed at %.0f%%: %s", frac * 100, exc)
        return out

    def _emit_alert(self, signal: str, severity: str, summary_zh: str, extra: dict,
                    frame_png: str | None = None) -> None:
        """Emit a mid-scan classical WARN/CRITICAL alert (TIP_QUALITY_DROP).

        ``frame_png`` is the persisted PNG of the frame this verdict was made
        on. It is not decoration: these alerts are the ones that stop a scan and
        ask for a tip repair, and they were for a while the only vision
        publisher that shipped without one — 「这一条为什么不显示图像」.
        """
        buf = self._buf()
        if buf is None:
            return
        try:
            from mast.buffer.schemas import Severity, VisionEvent, VisionEventType
            sev = {"critical": Severity.CRITICAL, "warn": Severity.WARN,
                   "info": Severity.INFO}.get(severity, Severity.WARN)

            # 事件生产方写明 source，不让读取方猜测。
            # 相同 tip_quality_drop 类型可来自电流监控或视觉，两者的证据和中止权限不同。
            # 电流来源使用 current_monitor；此处显式标记视觉来源，便于正确通知与处置。
            payload = {"signal": signal, "scan_id": self._scan_id, "summary_zh": summary_zh,
                       "network_free": True, "source": "vision_scan_monitor", **extra}
            if frame_png:
                payload["frame_path"] = frame_png
            buf.emit_event(VisionEvent(
                seqno=buf.next_seq(), kind=VisionEventType.TIP_QUALITY_DROP, severity=sev,
                payload=payload, cause_ref=f"scan#{self._scan_id}"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: emit_alert failed: %s", exc)

    # ── publishers ───────────────────────────────────────────────────

    def _publish_progress(self, frac: float, elapsed: float, total_time: float) -> None:
        """Publish an ESTIMATED ScanProgress (Nanonis exposes no live line idx),
        so the GUI bar + get_scan_progress() are populated during the scan."""
        buf = self._buf()
        if buf is None:
            return
        try:
            from mast.buffer.schemas import ScanProgress
            lines = self._total_lines or 0
            buf.put_progress(ScanProgress(
                seqno=buf.next_seq(), scan_id=self._scan_id,
                line_idx=int(round(frac * lines)), lines_total=int(lines),
                eta_s=max(0.0, total_time - elapsed),
            ))
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: put_progress failed: %s", exc)

    def _publish_coarse(self, buf, coarse, ordinal: int) -> None:
        if buf is None:
            return
        try:
            from mast.buffer.schemas import TipQuality, TipStatus
            q = TipQuality.GOOD if coarse.label == "good" else TipQuality.BAD
            seq = buf.next_seq()
            buf.put_tip_status(TipStatus(
                seqno=seq, quality=q, confidence=float(coarse.confidence),
                embedding_sha=getattr(coarse, "embedding_sha", None),
                scan_id=self._scan_id, frame_idx=int(ordinal),
                # `coarse` carries safe_mode_raw when the facade rewrote it; mark
                # the stored row so quality/confidence read as a mode, not a
                # measurement (the raw verdict itself goes to the milestone event).
                safe_mode=getattr(coarse, "safe_mode_raw", None) is not None,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: put_tip_status failed: %s", exc)

    def _publish_segment(self, buf, seg, frac, final) -> None:
        if buf is None or seg is None:
            return
        try:
            from mast.buffer.schemas import RegionMap
            seq = buf.next_seq()
            buf.put_region(RegionMap(
                seqno=seq, scan_id=self._scan_id,
                mask_rle=seg.mask_rle, shape=tuple(seg.shape),
            ))
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: put_region failed: %s", exc)

    @staticmethod
    def _contrast_range(a: np.ndarray) -> tuple[float, float]:
        """(vmin, vmax) for displaying a PARTIAL STM frame — over the acquired
        pixels only.

        Why ("图片全黑"): the unacquired region of a partial frame
        is filled with EXACTLY 0.0 (see _grab_frame). STM topography (Z in m)
        carries a large DC offset (~1e-8) with tiny corrugation, so folding the
        0.0 fill into the colour scale makes the acquired strip collapse onto one
        end of ``afmhot`` and the whole thumbnail reads as black. Stretching the
        scale over the NON-ZERO (acquired) pixels — robustly, 1–99 percentile so
        a single spike can't blow it out — restores the real contrast; the zero
        fill then sits below vmin and renders as the dark background. Falls back
        to all-finite when too few non-zero pixels exist to trust."""
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return (0.0, 1.0)
        acquired = finite[finite != 0.0]
        basis = acquired if acquired.size >= 4 else finite
        vmin = float(np.percentile(basis, 1))
        vmax = float(np.percentile(basis, 99))
        if vmax <= vmin:
            vmin, vmax = float(basis.min()), float(basis.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        return (vmin, vmax)

    def _persist_frame_png(self, frame: np.ndarray, ordinal: int) -> str | None:
        """Write the EXACT frame this milestone analysed to a small PNG and
        return its path (None on any failure — never blocks the milestone).

        Why : milestone events used to carry no
        image, so the UI's 近期帧 strip borrowed whatever .sxm was newest on
        disk — during a scan every "pulse" showed the PREVIOUS scan's image,
        and once the new full frame was saved, ALL history thumbnails silently
        became that full image. One PNG per milestone keeps the record honest:
        what you see is what the model actually saw at that instant."""
        try:
            import re
            from mast._runtime_paths import project_root
            a = np.asarray(frame)
            if a.ndim == 3:
                a = a[0]  # forward/topography channel
            if a.ndim != 2 or a.size == 0:
                return None
            d = project_root() / "artifacts" / "vision_frames"
            d.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w一-鿿-]", "_", str(self._scan_id))[:60] or "scan"
            # Per-run uniqueness: same scan_id re-scanned later must not
            # overwrite the earlier run's frames (that is exactly the history
            # falsification #78 complains about).
            out = d / f"{safe}_{int(self._time() * 1000) & 0xFFFFFFFF:x}_f{ordinal}.png"
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            finite = a[np.isfinite(a)]
            # An all-zero / blank frame is NOT a real image: the unacquired region
            # is filled with EXACTLY 0.0 (see _grab_frame), so a frame with no
            # non-zero pixel has nothing acquired yet. Persisting it would write a
            # black square the operator reads as fabricated history (feedback
            # #141 "图片全黑" / #143). Skip it — a real milestone always has data.
            if finite.size == 0 or not np.any(finite != 0.0):
                return None
            vmin, vmax = self._contrast_range(a)
            fig = plt.figure(figsize=(1.6, 1.6), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.imshow(a, cmap="afmhot", vmin=vmin, vmax=vmax,
                      origin="lower", interpolation="nearest")
            fig.savefig(out, format="png")
            plt.close(fig)
            return str(out)
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: frame png persist failed: %s", exc)
            return None

    def _emit_milestone_event(self, buf, frac, final, coarse, fine,
                              frame_path: str | None = None, rt: dict | None = None,
                              seen_at: float | None = None) -> None:
        if buf is None:
            return
        try:
            from mast.buffer.schemas import Severity, VisionEvent, VisionEventType
            coarse_d = coarse.model_dump()
            summary = self._describe("tip_coarse", coarse_d)
            if final and fine is not None:
                summary += " " + self._describe("tip_fine", fine.model_dump())
            # A confirmed mid-scan tip change makes even a partial milestone a
            # SCAN-quality warning worth surfacing at WARN.
            # In SAFE the dedicated tip_change alert is already demoted to INFO
            # (see _realtime_check); leaving THIS event at WARN for the same frame
            # would contradict it — same finding, two severities.
            rt_changed = bool(rt and rt.get("tip_change", {}).get("changed")
                              and not _safe_override_active())
            kind = VisionEventType.SCAN_COMPLETE if final else VisionEventType.FEATURE_OF_INTEREST
            # Severity: a bad tip mid-scan is worth a WARN so it surfaces.
            sev = Severity.WARN if (coarse.label == "bad" or rt_changed) else Severity.INFO
            payload = {
                "milestone": round(frac, 3),
                "scan_id": self._scan_id,
                "summary_zh": summary,
                # In SAFE this dict carries `safe_mode_raw` = what the model
                # actually said, so the record keeps the truth even though the
                # label the system acts on reads "good".
                "tip_coarse": coarse_d,
            }
            if _safe_override_active():
                payload["safe_mode"] = True
            if rt:
                # Network-free real-time signals (mid-scan tip-change / artifacts)
                # attached for the GUI + records alongside the learned coarse label.
                payload["classical"] = rt
            if frame_path:
                # The honest thumbnail: the persisted PNG of the exact partial
                # frame this milestone analysed (see _persist_frame_png).
                payload["frame_path"] = frame_path
            if final and fine is not None:
                payload["tip_fine"] = fine.model_dump()
            buf.emit_event(VisionEvent(
                seqno=buf.next_seq(), kind=kind, severity=sev,
                payload=payload, cause_ref=f"scan#{self._scan_id}",
            ))
            # 同一份判读，第二个读者：对话流里的旁白（给**人**看的叙事视图）。
            # Vision Buffer 那一页是取证视图（全部事件 + 严重度 + 原始 payload），
            # 两者读的是同一份 payload，旁白不重新判读画面。
            self._narrate_milestone(frac, final, summary, frame_path, seen_at)
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan-vision: emit_event failed: %s", exc)

    def _narrate_milestone(self, frac: float, final: bool, summary: str,
                           frame_path: "str | None",
                           seen_at: "float | None" = None) -> None:
        """把这个里程碑说给用户听。没有 sink 就什么都不做。

        ``summary_zh`` **复用** ``buffer_summarizer.describe`` 的那句话
        （``_describe`` 的产物），不另写一套判读措辞 —— 两套措辞迟早会对同一帧
        给出两种说法，而用户没有办法知道该信哪一句。

        图**原样引用**已经落盘的那张 PNG（``_persist_frame_png`` 写的、模型真正
        看过的那一帧），绝不重渲染 —— 重渲染就把 #76/#78 那条保证毁了
        （"历史里所有缩略图会静默变成最新那张整图"）。
        """
        sink = self._narration
        if sink is None:
            return
        try:
            image = ({"src": frame_path, "origin": "milestone_png"}
                     if frame_path else None)
            if final:
                sink.narrate("scan_done", image=image, summary_zh=summary,
                             scan_id=self._scan_id, event_t=seen_at)
            else:
                sink.narrate("scan_milestone", image=image, frac=float(frac),
                             summary_zh=summary, scan_id=self._scan_id,
                             event_t=seen_at)
        except Exception as exc:  # noqa: BLE001 — 一条旁白绝不许弄坏一次扫描
            logger.debug("scan-vision: narrate failed: %s", exc)

    def _emit_error(self, exc: Exception) -> None:
        buf = self._buf()
        if buf is None:
            return
        try:
            from mast.buffer.schemas import make_vision_error
            buf.emit_event(make_vision_error(
                "model_error", str(exc), seqno=buf.next_seq(),
                cause_ref=f"scan#{self._scan_id}",
            ))
        except Exception:  # noqa: BLE001
            pass


# ── Process-level current-monitor registry ──────────────────────────
# Only one scan runs at a time on the instrument, so only one monitor should
# be live. Tracking it lets a new scan supersede a stale monitor (rapid-fire /
# re-scan / abort-restart) instead of stranding an orphan daemon that would
# contend on the TCP roles + VisionModule lock and race on scan_id writes.
_CURRENT_MONITOR: "ScanVisionMonitor | None" = None
_CURRENT_LOCK = threading.Lock()


def is_monitor_running() -> bool:
    """True iff a scan-vision monitor is CURRENTLY watching a live scan (its
    daemon thread is alive).

    A monitor is started by StartScan right after ``Scan_Action`` and self-exits
    when the scan goes idle, so this is a reliable "MAST is running this scan"
    signal. The scan-map manual watcher uses it to avoid mislabelling an agent /
    system scan as "手动开始扫描" when the state poll observes ``scan_running`` more
    than the skill-recency window after the scan-start skill returned — e.g. a
    composite scan that does long prep before ``Scan_Action``.
    Goes False once the scan ends and the thread exits, so a genuine later MANUAL
    scan is still detected."""
    with _CURRENT_LOCK:
        mon = _CURRENT_MONITOR
    if mon is None:
        return False
    t = getattr(mon, "_thread", None)
    return bool(t is not None and t.is_alive())


def stop_active_monitor(join_timeout: float = 2.0) -> None:
    """Stop the current scan-vision monitor, if any (called on StopScan)."""
    global _CURRENT_MONITOR
    with _CURRENT_LOCK:
        mon, _CURRENT_MONITOR = _CURRENT_MONITOR, None
    if mon is not None:
        try:
            mon.stop(join_timeout=join_timeout)
        except Exception:  # noqa: BLE001
            pass


def _bind_narration_sink() -> Any:
    """把当前会话绑成一个可以跨线程带走的旁白出口。**必须在这里调**。

    这个函数跑在**调用方**的线程上 —— 也就是扫描起始技能所在的那根 langgraph
    工具线程，``turn_context`` 的 ContextVar 在那里是可见的。监视器自己那根
    ``threading.Thread`` 是裸线程，``copy_context()`` 不会发生，在里面读
    ``current_turn()`` 永远是空的。

    绑在这里而不是让每个调用方各自绑，是因为「每个调用方各自记得」这种接线
    人肉找不齐 —— 本仓已经为这个形状写过一次结构闸门。这里只有一个入口，
    绑一次，所有扫描起始路径都覆盖到。

    没有会话（后台唤醒跑 / 手动 GUI 触发）时返回一个 no-op sink，调用方不用判空。
    """
    try:
        from mast.chat import narration

        return narration.bind()
    except Exception:  # noqa: BLE001 — 旁白不可用不该拦住扫描
        return None


def start_scan_vision_monitor(
    pool: Any,
    *,
    scan_id: str,
    abort_event: threading.Event | None = None,
    **kwargs: Any,
) -> ScanVisionMonitor | None:
    """Create + start a monitor IF a buffer is active and vision is enabled.

    Supersedes any previously-running monitor (only one scan at a time), so a
    rapid-fire / restarted scan never strands an orphan thread. Fail-safe
    wiring helper called from the scan-start skill: returns the running
    monitor, or ``None`` if monitoring is disabled / unavailable (scanning then
    proceeds exactly as before). Never raises.

    Also binds the CHAT NARRATION sink here (see :func:`_bind_narration_sink`) —
    this is the last point on the caller's thread before the daemon starts, and
    the ContextVar the sink needs does not cross that boundary."""
    global _CURRENT_MONITOR
    try:
        import os
        import time as _time
        if os.environ.get("MAST_SCAN_VISION_MONITOR", "1").strip() == "0":
            return None
        from mast.buffer.active import get_active_buffer
        buf = get_active_buffer()
        if buf is None:
            return None  # no buffer running → nothing to publish into
        if not scan_id:  # no active experiment → still give the stream an id
            scan_id = f"scan-{int(_time.time())}"
        # Supersede any stale predecessor before starting the new one.
        stop_active_monitor(join_timeout=1.0)
        kwargs.setdefault("narration_sink", _bind_narration_sink())
        mon = ScanVisionMonitor(
            pool, scan_id=scan_id, abort_event=abort_event, buffer=buf, **kwargs,
        )
        mon.start()
        with _CURRENT_LOCK:
            _CURRENT_MONITOR = mon
        logger.info("scan-vision monitor started for scan %s", scan_id)
        return mon
    except Exception as exc:  # noqa: BLE001
        logger.warning("scan-vision monitor could not start: %s", exc)
        return None


__all__ = [
    "ScanVisionMonitor", "start_scan_vision_monitor", "stop_active_monitor",
    "is_monitor_running", "DEFAULT_THRESHOLDS",
]
