"""Spectroscopic tip probes — mast.vision.spectroscopy.

I(z) approach curve (clean exponential + apparent barrier + tip-jump detection)
and I(V) tunnelling spectrum (smoothness/symmetry/tip-switch). Network-free,
non-image, independent of the imaging path. See
docs/v2/benchmarks/vision_v25_diagnostic/.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np  # noqa: E402

from mast.vision.spectroscopy import assess_iv, assess_iz  # noqa: E402

# I ∝ exp(−2κz); κ[Å⁻¹]=0.5123·√φ. For φ=4.5 eV: 2κ = 21.7 nm⁻¹.
_DECAY_45EV = 21.7


def _clean_iz(phi_ev=4.5, seed=0):
    z = np.linspace(0.0, 0.5, 60)                     # nm
    decay = 2 * 0.5123 * np.sqrt(phi_ev) * 10.0       # nm⁻¹
    I = np.exp(-decay * z) * (1 + 0.01 * np.random.RandomState(seed).randn(60))
    return z, I


def test_clean_iz_recovers_barrier():
    z, I = _clean_iz(4.5)
    r = assess_iz(z, I)
    assert r.is_clean_exponential is True
    assert r.barrier_ev is not None and abs(r.barrier_ev - 4.5) < 0.6
    assert r.fit_r2 > 0.98
    assert r.n_jumps == 0


def test_iz_tip_jump_detected():
    z, I = _clean_iz()
    I = I.copy()
    I[30:] *= 0.2                                      # sudden current drop mid-ramp
    r = assess_iz(z, I)
    assert r.n_jumps >= 1
    assert r.is_clean_exponential is False


def test_iz_too_short_graceful():
    r = assess_iz([0.0, 0.1], [1.0, 0.5])
    assert r.is_clean_exponential is False and r.fit_r2 == 0.0


def test_clean_iv_is_stable_and_symmetric():
    V = np.linspace(-1, 1, 101)
    I = np.sinh(3 * V) + 0.01 * np.random.RandomState(0).randn(101)
    q = assess_iv(V, I)
    assert q.is_stable is True
    assert q.symmetry > 0.9
    assert q.n_spikes == 0


def test_iv_tip_switch_detected():
    V = np.linspace(-1, 1, 101)
    I = np.sinh(3 * V).astype(float)
    I[50] += 5.0
    I[70] -= 4.0
    q = assess_iv(V, I)
    assert q.n_spikes >= 1
    assert q.is_stable is False


def test_iv_gap_measured():
    V = np.linspace(-1, 1, 101)
    I = np.sign(V) * np.clip(np.abs(V) - 0.3, 0, None) ** 1.5   # ~0.6 V zero-conductance gap
    q = assess_iv(V, I)
    assert q.gap_ev is not None and q.gap_ev > 0.3


def test_vision_module_spectroscopy_delegates():
    from mast.vision.module import IvResult, IzResult, VisionModule

    vm = VisionModule(backend="mock")
    z, I = _clean_iz()
    assert isinstance(vm.assess_iz(z, I), IzResult)
    V = np.linspace(-1, 1, 51)
    assert isinstance(vm.assess_iv(V, np.sinh(3 * V)), IvResult)
