"""One-sided rfft of an already-captured signal trace — pure numpy, no hardware,
no Gradio. Relocated here from gui/exp_capture.py during the TS-rewrite cutover so
the API/vision route can compute FFTs without dragging in the (deleted) Gradio UI.
"""

from __future__ import annotations


def compute_fft(trace: dict, *, window: str = "hann", detrend: bool = True,
                output: str = "magnitude") -> dict:
    """One-sided rfft of an ALREADY-captured trace — no extra hardware call.

    Operates on whatever channel was captured (current or any signal index), so
    "导出 nanonis 的 FFT" is simply: capture → compute_fft → save. Returns {}
    when there are too few samples to transform.
    """
    import numpy as np

    ys = trace.get("samples") or []
    n = len(ys)
    if n < 4:
        return {}
    fs = float(trace.get("fs_hz") or 0.0)
    if fs <= 0:
        dur = float(trace.get("duration_s") or 0.0)
        fs = (n - 1) / dur if dur > 0 else float(n)
    arr = np.asarray(ys, dtype=np.float64)
    if detrend:
        arr = arr - arr.mean()
    w = (window or "hann").lower()
    if w == "hann":
        win = np.hanning(n)
    elif w == "hamming":
        win = np.hamming(n)
    else:
        win = np.ones(n)
        w = "rect"
    spec = np.fft.rfft(arr * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(spec)
    if (output or "magnitude").lower() == "power":
        denom = fs * float((win ** 2).sum())
        psd = (mag ** 2) * (1.0 / denom if denom > 0 else 1.0)
        if psd.size > 2:
            psd[1:-1] *= 2.0   # one-sided
        vals = psd
        out_kind = "power"
    else:
        vals = mag
        out_kind = "magnitude"
    return {
        "freqs_hz": freqs.tolist(),
        "spectrum": vals.tolist(),
        "n_samples": n,
        "fs_hz": fs,
        "nyquist_hz": fs / 2.0,
        "df_hz": fs / n,
        "window": w,
        "output": out_kind,
        "unit": trace.get("unit") or "A",
        "channel_name": trace.get("channel_name") or "signal",
    }


__all__ = ["compute_fft"]
