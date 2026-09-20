"""Physically-shaped synthetic tunnelling-current traces for the monitor tests.

Every component is generated the way the real noise source behaves, not the way
that makes an assertion pass:

* flicker noise is shaped in the frequency domain as 1/√f amplitude, so its PSD
  really is 1/f — a lowpassed white sequence would give the right "look" and the
  wrong slope;
* telegraph switching is a two-state Markov chain with exponential dwell times,
  so the transition rate and the dwell statistics are consistent with each other;
* spikes are exponentially-decaying transients, not single-sample impulses, so
  the median filter has something realistic to subtract;
* mains hum carries the 3rd and 5th harmonics a real ground loop produces.

This matters because the detectors are tuned against these traces. A synthetic
signal that is not physical certifies the wrong threshold — the lesson from the
VIGIL corpus, where a multiplicative lattice model produced spectra with no
fundamental and quietly invalidated a whole round of tuning.

Units are SI amps throughout; defaults sit in the picoamp range of a real STM
tunnel junction.
"""
from __future__ import annotations

import numpy as np


def pink_noise(n: int, rms_a: float, fs_hz: float, rng: np.random.Generator,
               f_min_hz: float = 0.5) -> np.ndarray:
    """1/f (flicker) noise: white spectrum shaped by 1/√f in amplitude.

    Below ``f_min_hz`` the shaping flattens — a true 1/f process diverges at DC
    and a finite record cannot represent it anyway.
    """
    if n < 4 or rms_a <= 0:
        return np.zeros(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    shape = np.zeros_like(freqs)
    nz = freqs > 0
    shape[nz] = 1.0 / np.sqrt(np.maximum(freqs[nz], f_min_hz))
    spec = (rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)) * shape
    spec[0] = 0.0
    out = np.fft.irfft(spec, n=n)
    sd = out.std()
    return out * (rms_a / sd) if sd > 0 else out


def telegraph(n: int, gap_a: float, rate_hz: float, fs_hz: float,
              rng: np.random.Generator) -> np.ndarray:
    """Two-state RTN with exponential dwell times (symmetric switching rate).

    ``rate_hz`` is the per-state switching rate, so the expected number of
    transitions over the record is ``rate_hz · duration``.
    """
    if n <= 0 or gap_a <= 0 or rate_hz <= 0:
        return np.zeros(n)
    p_switch = min(0.5, rate_hz / fs_hz)      # per-sample switch probability
    flips = rng.random(n) < p_switch
    state = np.cumsum(flips) % 2              # each flip toggles the level
    return state.astype(np.float64) * gap_a


def spike_train(n: int, count: int, amp_a: float, fs_hz: float,
                rng: np.random.Generator, tau_s: float = 1e-3) -> np.ndarray:
    """Exponentially-decaying transients at random times (alternating sign)."""
    out = np.zeros(n)
    if count <= 0 or amp_a == 0:
        return out
    decay_len = max(2, int(round(5 * tau_s * fs_hz)))
    kernel = np.exp(-np.arange(decay_len) / max(1.0, tau_s * fs_hz))
    for i in range(count):
        start = int(rng.integers(0, max(1, n - decay_len)))
        sign = 1.0 if i % 2 == 0 else -1.0
        out[start:start + decay_len] += sign * amp_a * kernel
    return out


def mains(n: int, amp_a: float, fs_hz: float, line_hz: float = 50.0,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """Line hum with the 3rd and 5th harmonics a ground loop typically carries."""
    if n <= 0 or amp_a <= 0:
        return np.zeros(n)
    t = np.arange(n) / fs_hz
    phase = float(rng.random() * 2 * np.pi) if rng is not None else 0.0
    return amp_a * (np.sin(2 * np.pi * line_hz * t + phase)
                    + 0.30 * np.sin(2 * np.pi * 3 * line_hz * t + phase)
                    + 0.15 * np.sin(2 * np.pi * 5 * line_hz * t + phase))


def synth_current(*, fs_hz: float = 20000.0, dur_s: float = 1.0,
                  mean_a: float = 100e-12,
                  white_rms_a: float = 2e-12,
                  pink_rms_a: float = 0.0,
                  rtn_gap_a: float = 0.0, rtn_rate_hz: float = 0.0,
                  spikes: int = 0, spike_amp_a: float = 50e-12,
                  spike_tau_s: float = 1e-3,
                  line_amp_a: float = 0.0, line_hz: float = 50.0,
                  sat_rail_a: float | None = None,
                  drift_a_per_s: float = 0.0,
                  seed: int = 0) -> np.ndarray:
    """Compose one segment of tunnelling current from physical components.

    Returns float64 amps of length ``round(fs_hz · dur_s)``.
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs_hz * dur_s))
    t = np.arange(n) / fs_hz

    y = np.full(n, float(mean_a))
    if white_rms_a > 0:
        y += rng.normal(0.0, white_rms_a, size=n)
    if pink_rms_a > 0:
        y += pink_noise(n, pink_rms_a, fs_hz, rng)
    if rtn_gap_a > 0 and rtn_rate_hz > 0:
        y += telegraph(n, rtn_gap_a, rtn_rate_hz, fs_hz, rng)
    if spikes > 0:
        y += spike_train(n, spikes, spike_amp_a, fs_hz, rng, spike_tau_s)
    if line_amp_a > 0:
        y += mains(n, line_amp_a, fs_hz, line_hz, rng)
    if drift_a_per_s:
        y += drift_a_per_s * t
    if sat_rail_a is not None:
        y = np.clip(y, -abs(sat_rail_a), abs(sat_rail_a))
    return y


__all__ = ["synth_current", "pink_noise", "telegraph", "spike_train", "mains"]
