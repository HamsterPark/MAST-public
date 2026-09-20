"""``fit_dispersion``: the surface-state band bottom and effective mass from standing waves.

The synthetic input is what an experimenter actually has — dI/dV against distance from a
scatterer at a handful of energies. Near a step the modulation is a Bessel J0 of twice the
wavevector; the routine has to pull k out of each energy's profile and then fit the parabola
``E = E0 + hbar^2 k^2 / 2 m*`` through the k values it found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import special

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mast.vision.standing_wave import (distance_to_line, distance_to_point,  # noqa: E402
                                       fit_dispersion)

#: hbar^2 / 2 m_e in eV nm^2 — the same constant the simulator uses
H2_2M = 0.0381
CU_E0, CU_M = -0.44, 0.38


def k_of(e_ev: float, *, e0: float = CU_E0, m_eff: float = CU_M) -> float:
    return 0.0 if e_ev <= e0 else float(np.sqrt((e_ev - e0) * m_eff / H2_2M))


def step_grid(energies, distances, *, e0=CU_E0, m_eff=CU_M, refl=0.5,
              noise=0.0, seed=0) -> np.ndarray:
    """dI/dV(E, d) near a straight step: 1 − r·J0(2kd), damped a little with distance."""
    rng = np.random.default_rng(seed)
    d = np.asarray(distances, float)
    out = np.empty((len(energies), d.size))
    for i, e in enumerate(energies):
        k = k_of(e, e0=e0, m_eff=m_eff)
        g = 1.0 - refl * special.j0(2 * k * d) * np.exp(-d / 60.0)
        out[i] = g * (1.0 + rng.normal(0.0, noise, d.size)) if noise else g
    return out


ENERGIES = list(np.linspace(-0.40, 0.10, 12))
DISTANCES = list(np.linspace(1.0, 25.0, 96))


def test_it_recovers_the_band_bottom_and_the_mass():
    g = step_grid(ENERGIES, DISTANCES)
    res = fit_dispersion(ENERGIES, DISTANCES, g, scatterer_kind="step")
    assert res.verdict == "dispersion", (res.reasons, res.notes)
    assert res.e0_ev == pytest.approx(CU_E0, abs=0.015)
    assert res.m_eff == pytest.approx(CU_M, rel=0.05)
    assert res.r2 is not None and res.r2 > 0.95
    assert res.n_energies_used >= 8


def test_it_still_works_with_two_percent_noise():
    """Two percent is what a real dI/dV map carries; the tolerances the P2 scenario asks for
    were derived assuming this much."""
    g = step_grid(ENERGIES, DISTANCES, noise=0.02, seed=3)
    res = fit_dispersion(ENERGIES, DISTANCES, g, scatterer_kind="step")
    assert res.verdict == "dispersion", res.reasons
    assert res.e0_ev == pytest.approx(CU_E0, abs=0.03)
    assert res.m_eff == pytest.approx(CU_M, rel=0.10)


def test_a_different_surface_gives_a_different_answer():
    """The whole point of drawing the truth per seed: the fit has to follow it, not recite
    Cu(111)."""
    g = step_grid(ENERGIES, DISTANCES, e0=-0.46, m_eff=0.30)
    res = fit_dispersion(ENERGIES, DISTANCES, g, scatterer_kind="step")
    assert res.verdict == "dispersion"
    assert res.e0_ev == pytest.approx(-0.46, abs=0.02)
    assert res.m_eff == pytest.approx(0.30, rel=0.08)
    assert abs(res.m_eff - CU_M) > 0.05


def test_a_flat_map_is_not_a_dispersion():
    g = np.ones((len(ENERGIES), len(DISTANCES)))
    res = fit_dispersion(ENERGIES, DISTANCES, g, scatterer_kind="step")
    assert res.verdict in ("no_standing_wave", "undecidable"), res.e0_ev


def test_pure_noise_is_not_a_dispersion():
    rng = np.random.default_rng(11)
    g = 1.0 + rng.normal(0.0, 0.05, (len(ENERGIES), len(DISTANCES)))
    res = fit_dispersion(ENERGIES, DISTANCES, g, scatterer_kind="step")
    assert res.verdict in ("no_standing_wave", "undecidable"), (res.e0_ev, res.m_eff, res.r2)


def test_a_mismatched_grid_is_refused_rather_than_guessed():
    res = fit_dispersion(ENERGIES, DISTANCES, np.ones((3, 4)), scatterer_kind="step")
    assert res.verdict == "undecidable" and "shape_mismatch" in res.reasons


def test_spectra_all_on_top_of_each_other_are_refused():
    """Everything within a nanometre of one place constrains no wavelength at all."""
    d = list(np.linspace(1.0, 1.5, 20))
    res = fit_dispersion(ENERGIES, d, step_grid(ENERGIES, d), scatterer_kind="step")
    assert res.verdict == "undecidable"
    assert "distance_span_too_short" in res.reasons


def test_the_k_table_is_returned_so_the_fit_can_be_checked():
    res = fit_dispersion(ENERGIES, DISTANCES, step_grid(ENERGIES, DISTANCES),
                         scatterer_kind="step")
    assert len(res.k_table) == res.n_energies_used
    for e, k, sigma in res.k_table:
        assert k == pytest.approx(k_of(e), abs=0.15), (e, k)
        assert sigma >= 0.0


# ── the geometry helpers ──
def test_distance_to_a_step_is_perpendicular_and_comes_back_in_nanometres():
    """Both helpers take metres and return nanometres — the unit changes across the call,
    which is exactly the kind of thing that goes unnoticed until a fit is quietly off by 1e9."""
    # a step through the origin running along y: the distance is |x|
    d = distance_to_line(np.array([3e-9, -4e-9]), np.array([7e-9, 1e-9]),
                         edge_x_m=0.0, edge_y_m=0.0, edge_angle_deg=90.0)
    assert np.allclose(np.abs(d), [3.0, 4.0], atol=1e-9)
    assert d[0] < 0 < d[1]                       # and it is signed, so sides can be told apart


def test_distance_to_a_point_is_radial():
    d = distance_to_point(np.array([3e-9]), np.array([4e-9]),
                          scatterer_x_m=0.0, scatterer_y_m=0.0)
    assert d[0] == pytest.approx(5.0, rel=1e-9)
