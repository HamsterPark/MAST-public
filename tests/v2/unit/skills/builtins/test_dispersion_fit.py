"""``FitDispersion``: the IO shell around the standing-wave fit.

The judge itself is covered in ``tests/v2/unit/vision/test_standing_wave.py``. What this file
covers is the part that reads real files and turns positions into distances — including the
``distances_nm`` entry, which exists because a long spectroscopy line drifts: the scatterer is
somewhere else by the last spectrum than it was at the first, and a single edge located once
cannot know that. A caller that tracked the geometry itself hands the distances over directly.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
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
import pytest  # noqa: E402
from scipy import special  # noqa: E402

from mast.skills.builtins.dispersion_fit import FitDispersion  # noqa: E402

H2_2M = 0.0381                      # hbar^2 / 2 m_e, eV nm^2
E0, M_EFF = -0.44, 0.38
ENERGIES = np.linspace(-0.35, 0.20, 40)


def _dat(path: Path, *, distance_nm: float, x_m: float, y_m: float, noise: float = 0.0,
         seed: int = 0) -> Path:
    """One dI/dV spectrum, in the Nanonis .dat layout the reader expects."""
    rng = np.random.default_rng(seed)
    k = np.where(ENERGIES > E0, np.sqrt(np.maximum(ENERGIES - E0, 0) * M_EFF / H2_2M), 0.0)
    y = 1.0 - 0.5 * special.j0(2 * k * distance_nm) * math.exp(-distance_nm / 60.0)
    if noise:
        y = y + rng.normal(0.0, noise, y.size)
    lines = [f"Experiment\tbias spectroscopy\t", f"X (m)\t{x_m:.6E}\t",
             f"Y (m)\t{y_m:.6E}\t", f"Z (m)\t{-1.0e-8:.6E}\t", "",
             "[DATA]", "Bias calc (V)\tLI Demod 1 X (A)"]
    lines += [f"{a:.6E}\t{b:.6E}" for a, b in zip(ENERGIES, y)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _line_of_spectra(tmp_path: Path, distances_nm, *, along=(1.0, 0.0), noise=0.0) -> list[str]:
    out = []
    for i, d in enumerate(distances_nm):
        p = _dat(tmp_path / f"sw_p{i:03d}.dat", distance_nm=d,
                 x_m=along[0] * d * 1e-9, y_m=along[1] * d * 1e-9, noise=noise, seed=i)
        out.append(str(p))
    return out


DISTANCES = list(np.linspace(1.0, 13.0, 40))


def test_it_fits_the_dispersion_from_a_line_of_dat_files(tmp_path):
    paths = _line_of_spectra(tmp_path, DISTANCES, noise=0.01)
    res = FitDispersion().execute(None, {
        "dat_paths": json.dumps(paths), "scatterer_kind": "step",
        "edge_x_m": 0.0, "edge_y_m": 0.0, "edge_angle_deg": 90.0})
    assert res.success, res.error
    assert res.data["verdict"] == "dispersion", res.data.get("reasons")
    assert res.data["e0_mev"] == pytest.approx(E0 * 1e3, abs=20.0)
    assert res.data["m_eff"] == pytest.approx(M_EFF, rel=0.05)
    assert res.data["n_spectra"] == len(paths)


def test_distances_given_by_the_caller_are_used_instead_of_the_geometry(tmp_path):
    """The positions in the headers are deliberately wrong here — all at the origin. Only the
    supplied distances can produce a fit, so a fit proves they were used."""
    paths = []
    for i, d in enumerate(DISTANCES):
        paths.append(str(_dat(tmp_path / f"d_p{i:03d}.dat", distance_nm=d,
                              x_m=0.0, y_m=0.0, noise=0.01, seed=i)))
    res = FitDispersion().execute(None, {
        "dat_paths": json.dumps(paths), "scatterer_kind": "step",
        "distances_nm": json.dumps(DISTANCES)})
    assert res.success, res.error
    assert res.data["verdict"] == "dispersion", res.data.get("reasons")
    assert res.data["m_eff"] == pytest.approx(M_EFF, rel=0.05)


def test_a_distance_list_of_the_wrong_length_is_refused(tmp_path):
    """They pair up by position. A length mismatch means nobody knows which spectrum is where,
    and quietly zipping the shorter one would put every reading at the wrong distance."""
    paths = _line_of_spectra(tmp_path, DISTANCES[:6])
    res = FitDispersion().execute(None, {
        "dat_paths": json.dumps(paths), "scatterer_kind": "step",
        "distances_nm": json.dumps([1.0, 2.0])})
    assert res.success is False
    assert "distances_nm" in (res.error or "")


def test_without_geometry_or_distances_it_says_what_is_missing(tmp_path):
    paths = _line_of_spectra(tmp_path, DISTANCES[:8])
    res = FitDispersion().execute(None, {"dat_paths": json.dumps(paths),
                                         "scatterer_kind": "step"})
    assert res.success is False
    assert "edge_x_m" in (res.error or "") and "distances_nm" in (res.error or "")


def test_spectra_on_both_sides_of_the_step_are_not_mixed(tmp_path):
    """A spectrum 3 nm to the left and one 3 nm to the right are at the same distance and on
    different terraces. Folding them together fits a wave to two different surfaces."""
    paths = _line_of_spectra(tmp_path, DISTANCES, noise=0.01)
    paths += [str(_dat(tmp_path / f"neg_{i}.dat", distance_nm=d, x_m=-d * 1e-9, y_m=0.0,
                       noise=0.01, seed=100 + i)) for i, d in enumerate(DISTANCES[:8])]
    res = FitDispersion().execute(None, {
        "dat_paths": json.dumps(paths), "scatterer_kind": "step",
        "edge_x_m": 0.0, "edge_y_m": 0.0, "edge_angle_deg": 90.0})
    assert res.success
    assert "mixed_sides" in (res.data.get("warnings") or [])
    assert res.data["n_spectra"] == len(DISTANCES)


def test_no_files_is_an_error_not_an_empty_answer(tmp_path):
    res = FitDispersion().execute(None, {"dat_paths": json.dumps([]),
                                         "scatterer_kind": "step"})
    assert res.success is False and "dat_paths" in (res.error or "")


def test_unreadable_files_are_skipped_and_counted(tmp_path):
    paths = _line_of_spectra(tmp_path, DISTANCES, noise=0.01)
    bad = tmp_path / "bad.dat"
    bad.write_text("not a spectrum\n", encoding="utf-8")
    res = FitDispersion().execute(None, {
        "dat_paths": json.dumps(paths + [str(bad)]), "scatterer_kind": "step",
        "edge_x_m": 0.0, "edge_y_m": 0.0, "edge_angle_deg": 90.0})
    assert res.success
    assert "some_spectra_unreadable" in (res.data.get("warnings") or [])
