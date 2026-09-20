"""Software-side FFT of tunneling current — polls Current_Get at high rate,
then runs ``numpy.fft.rfft`` on the buffer.

Use this when:
  - The Nanonis Spectrum Analyzer module isn't loaded on the current
    Nanonis configuration (simulators are a common case).
  - You need full control over the window function / overlap / detrending.
  - You're already polling current for some other reason (contact
    detection in ShapeTipOnSurface) and want the spectrum cheaply.

Trade-off vs ``AcquirePSD`` (hardware path):
  - Sampling rate: capped at ~2 kHz by TCP RTT → Nyquist ≈ 1 kHz.
    The hardware path can go up to RTFreq/2 (≈10 kHz on a real V5e).
  - Latency: must collect ``duration_s`` of samples before the FFT,
    so end-to-end > duration_s. Hardware path returns in ~0.5 ms.
  - Compute cost: ~22 µs for a 1024-point rfft. Negligible.
"""
from __future__ import annotations

import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill


class MonitorCurrentFFT(BaseSkill):
    """Poll Current_Get for a window, then numpy.fft.rfft the buffer."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MonitorCurrentFFT",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "通过 TCP 轮询对隧道电流做软件 FFT。以 `poll_hz` Hz 采集 `duration_s` 时长的样本，"
                "然后返回幅度（或功率）谱。时延 = duration_s + 22 µs FFT。Nyquist = poll_hz / 2。当 "
                "Nanonis Spectrum Analyzer 模块不可用、或你需要自定义窗函数时用它。"
            ),
            parameters=[
                ParameterSpec(
                    name="duration_s",
                    type="float",
                    description=(
                        "窗长，单位是 SECONDS（秒）。频率分辨率 df = 1 / duration_s 。典型 0.5–2 s。"
                    ),
                    unit="s", required=False, default=1.0,
                    min_value=0.05, max_value=30.0,
                ),
                ParameterSpec(
                    name="poll_hz",
                    type="float",
                    description=(
                        "目标轮询速率，单位 Hz。真实 Nanonis V5e + Python TCP 大约在 1500–2000 Hz "
                        "封顶；实际速率会写进结果的 `actual_fs_hz` 字段。"
                    ),
                    unit="Hz", required=False, default=1000.0,
                    min_value=10.0, max_value=4000.0,
                ),
                ParameterSpec(
                    name="window",
                    type="str",
                    description=(
                        "FFT 之前施加的窗函数。'hann'（默认）、'hamming'、'rect'（不加窗）。"
                    ),
                    required=False, default="hann",
                ),
                ParameterSpec(
                    name="detrend",
                    type="bool",
                    description="加窗前先减去缓冲区的均值。",
                    required=False, default=True,
                ),
                ParameterSpec(
                    name="output",
                    type="str",
                    description="'magnitude'（|FFT|）或 'power'（|FFT|² PSD）。",
                    required=False, default="magnitude",
                ),
            ],
            estimated_duration_s=1.1,
            composition_level=2,
            tags=["current", "fft", "psd", "software", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        duration = float(params.get("duration_s", 1.0))
        poll_hz = float(params.get("poll_hz", 1000.0))
        window_name = (params.get("window") or "hann").lower()
        detrend = bool(params.get("detrend", True))
        output = (params.get("output") or "magnitude").lower()

        try:
            import numpy as np
        except ImportError:
            return SkillResult(
                skill_name="MonitorCurrentFFT", success=False,
                error="numpy required for FFT",
            )

        # Poll Current_Get as fast as possible up to poll_hz target.
        period = 1.0 / poll_hz
        samples: list[float] = []
        timestamps: list[float] = []
        records = []
        t0 = time.perf_counter()
        next_poll = t0
        while True:
            now = time.perf_counter()
            if now - t0 >= duration:
                break
            if hasattr(context, "check_abort") and context.check_abort():
                break
            if now < next_poll:
                time.sleep(min(next_poll - now, 0.001))
                continue
            rec = context.safe_call("Current_Get")
            records.append(rec)
            if rec.error:
                next_poll += period
                continue
            d = _decoded(rec.return_value)
            if d:
                v = d[0]
                # decodeArray-style unwrap
                if isinstance(v, tuple) and v:
                    v = v[0]
                samples.append(float(v))
                timestamps.append(now - t0)
            next_poll += period

        n = len(samples)
        if n < 4:
            return SkillResult(
                skill_name="MonitorCurrentFFT", success=False,
                error=f"Too few samples ({n}); polling failed or aborted",
                nanonis_calls=records,
            )

        arr = np.array(samples, dtype=np.float64)
        actual_dur = timestamps[-1] - timestamps[0] if n > 1 else 1.0 / poll_hz
        actual_fs = (n - 1) / actual_dur if actual_dur > 0 else poll_hz

        if detrend:
            arr = arr - arr.mean()

        if window_name == "hann":
            win = np.hanning(n)
        elif window_name == "hamming":
            win = np.hamming(n)
        else:
            win = np.ones(n)
        arr_w = arr * win

        spectrum = np.fft.rfft(arr_w)
        freqs = np.fft.rfftfreq(n, d=1.0 / actual_fs)
        mag = np.abs(spectrum)

        if output == "power":
            # One-sided PSD (W/Hz), Hann normalization
            scale = 1.0 / (actual_fs * (win ** 2).sum())
            psd = (mag ** 2) * scale
            psd[1:-1] *= 2.0   # one-sided
            psd_values = psd.tolist()
        else:
            psd_values = mag.tolist()

        return SkillResult(
            skill_name="MonitorCurrentFFT", success=True,
            data={
                "n_samples": n,
                "actual_duration_s": actual_dur,
                "actual_fs_hz": actual_fs,
                "nyquist_hz": actual_fs / 2.0,
                "df_hz": actual_fs / n,
                "window": window_name,
                "output": output,
                "freqs_hz": freqs.tolist(),
                "spectrum": psd_values,
                "source": "software_fft_polled_current",
            },
            nanonis_calls=records,
        )


def _decoded(rv):
    if isinstance(rv, tuple) and len(rv) >= 3 and isinstance(rv[2], list):
        return rv[2]
    if isinstance(rv, list):
        return rv
    return []
