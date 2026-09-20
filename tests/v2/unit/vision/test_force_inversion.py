"""``sader_jarvis`` / ``invert_force_curve``: the force behind a Δf(z) curve.

The loop is closed against a known force law: a Morse well plus a van der Waals background
goes through ``forward_df`` to make the Δf a qPlus sensor would see, and the inversion has to
give the force back. That is the only honest check — the inversion is an integral transform,
and eyeballing the shape of one curve proves nothing about its magnitude.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mast.vision.force_inversion import (forward_df, invert_force_curve,  # noqa: E402
                                         sader_jarvis)

F0, K = 25296.2, 1800.0
#: a Morse well of the depth and range a metal adatom has, in SI
D_E_J, A_PER_M, Z_E_M = 150e-3 * 1.602176634e-19, 14e9, 0.30e-9
HAMAKER_J, Z0_M, TIP_R_M = 2.0e-19, 0.3e-9, 2.0e-9


def morse_force(z):
    """dU/dz of D_e[(1−e^{−a(z−z_e)})² − 1]; negative is attractive."""
    e = np.exp(-A_PER_M * (np.asarray(z, float) - Z_E_M))
    return -2.0 * D_E_J * A_PER_M * e * (1.0 - e)


def vdw_force(z):
    return -HAMAKER_J * TIP_R_M / (6.0 * (np.asarray(z, float) + Z0_M) ** 2)


def total_force(z):
    return morse_force(z) + vdw_force(z)


def true_f_min() -> tuple[float, float]:
    """Closed form for the Morse part: F_min = −a·D_e/2 at z_e + ln2/a."""
    return -A_PER_M * D_E_J / 2.0, Z_E_M + math.log(2.0) / A_PER_M


Z = np.linspace(0.10e-9, 1.20e-9, 400)


@pytest.mark.parametrize("amp_m", [50e-12, 200e-12])
def test_the_inversion_gives_the_force_back(amp_m):
    df = forward_df(Z, morse_force, f0_hz=F0, k_n_per_m=K, amplitude_m=amp_m)
    f = sader_jarvis(Z, df, f0_hz=F0, k_n_per_m=K, amplitude_m=amp_m)
    f_true, z_true = true_f_min()
    core = slice(1, Z.size - 40)                 # the tail carries the truncation error
    i = int(np.argmin(f[core])) + core.start
    assert f[i] == pytest.approx(f_true, rel=0.08), (f[i] * 1e12, f_true * 1e12)
    assert Z[i] == pytest.approx(z_true, abs=8e-12)


def test_the_small_amplitude_limit_is_the_force_gradient():
    """With A → 0, Δf is just −f0/2k times dF/dz. If that limit is wrong nothing else can
    be right."""
    df = forward_df(Z, morse_force, f0_hz=F0, k_n_per_m=K, amplitude_m=1e-14)
    h = 0.5e-12                                  # the same step the routine takes
    grad = (morse_force(Z + h) - morse_force(Z - h)) / (2 * h)
    assert np.allclose(df, -(F0 / (2 * K)) * grad, rtol=1e-6)


def test_more_quadrature_nodes_do_not_move_the_answer():
    a = 100e-12
    lo = forward_df(Z, morse_force, f0_hz=F0, k_n_per_m=K, amplitude_m=a, n_nodes=64)
    hi = forward_df(Z, morse_force, f0_hz=F0, k_n_per_m=K, amplitude_m=a, n_nodes=512)
    assert np.allclose(lo, hi, rtol=1e-3)


def test_the_full_measurement_recovers_depth_decay_and_binding_energy():
    """What the P5 scenario is scored on, end to end: a curve on the atom, a curve on clean
    surface, and the three numbers that come out after subtracting one from the other."""
    amp = 50e-12
    df_atom = forward_df(Z, total_force, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    df_bg = forward_df(Z, vdw_force, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    res = invert_force_curve(Z, df_atom, f0_hz=F0, k_n_per_m=K, amplitude_m=amp,
                             background_df_hz=df_bg)
    assert res.verdict == "well", res.reasons
    assert res.background_used
    f_true, _ = true_f_min()
    assert res.f_min_pn == pytest.approx(f_true * 1e12, rel=0.10)
    # the well depth is D_e once the van der Waals background is gone
    assert res.e_bind_mev == pytest.approx(150.0, rel=0.20)
    # the Morse tail falls as e^{-a z}: a decay length of 1/a
    assert res.decay_length_m == pytest.approx(1.0 / A_PER_M, rel=0.35)


def test_without_the_background_the_binding_energy_is_overstated():
    """The van der Waals tail integrates into the energy, which is why the scenario demands a
    clean-surface curve as evidence and not just a curve on the atom."""
    amp = 50e-12
    df_atom = forward_df(Z, total_force, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    with_bg = invert_force_curve(Z, df_atom, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    df_bg = forward_df(Z, vdw_force, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    without = invert_force_curve(Z, df_atom, f0_hz=F0, k_n_per_m=K, amplitude_m=amp,
                                 background_df_hz=df_bg)
    assert with_bg.e_bind_mev > without.e_bind_mev * 1.2
    assert not with_bg.background_used


def test_a_sweep_that_never_reached_the_turning_point_is_undecidable():
    """A pure van der Waals background grows all the way in, so its Δf minimum sits at the
    edge of the sweep. There is no well to report a depth for, and saying "well" there would
    let a curve that stopped too early look like a measurement."""
    amp = 50e-12
    df = forward_df(Z, vdw_force, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    res = invert_force_curve(Z, df, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    assert res.verdict == "undecidable", res.f_min_pn
    assert "minimum_not_bracketed" in res.reasons


def test_a_short_curve_is_undecidable_rather_than_inverted():
    res = invert_force_curve(Z[:10], np.zeros(10), f0_hz=F0, k_n_per_m=K, amplitude_m=50e-12)
    assert res.verdict == "undecidable" and "too_few_points" in res.reasons


def test_a_mismatched_background_warns_and_is_not_used():
    amp = 50e-12
    df = forward_df(Z, total_force, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    res = invert_force_curve(Z, df, f0_hz=F0, k_n_per_m=K, amplitude_m=amp,
                             background_df_hz=np.zeros(7))
    assert not res.background_used
    assert "background_length_mismatch" in res.warnings


def test_the_force_curve_is_returned_so_it_can_be_plotted_or_checked():
    amp = 50e-12
    df = forward_df(Z, total_force, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    res = invert_force_curve(Z, df, f0_hz=F0, k_n_per_m=K, amplitude_m=amp)
    assert len(res.z_m) == len(res.force_n) == len(res.energy_ev) == Z.size
    assert res.n_points == Z.size
