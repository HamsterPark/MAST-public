"""``InvertForceSaderJarvis``: the IO shell — where the sensor's numbers come from.

The inversion is covered in ``tests/v2/unit/vision/test_force_inversion.py``. What this file
covers is the part with no physics in it and every opportunity to be silently wrong: the
frequency, the spring constant and the amplitude. None of the three is in a Nanonis header the
way a bias is, and inventing any of them scales the whole force curve by a constant nobody
would notice.
"""
from __future__ import annotations

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

from mast.skills.builtins.force_inversion import InvertForceSaderJarvis  # noqa: E402
from mast.vision.force_inversion import forward_df  # noqa: E402

F0, K, AMP = 25296.2, 1800.0, 50e-12
D_E_J, A_PER_M, Z_E_M = 150e-3 * 1.602176634e-19, 14e9, 0.30e-9
Z = np.linspace(0.10e-9, 1.10e-9, 300)


def morse(z):
    e = np.exp(-A_PER_M * (np.asarray(z, float) - Z_E_M))
    return -2.0 * D_E_J * A_PER_M * e * (1.0 - e)


def true_f_min_pn() -> float:
    return -A_PER_M * D_E_J / 2.0 * 1e12


def _dat(path: Path, *, df_name="OC M1 Freq. Shift (Hz)", with_header=True,
         force=morse) -> Path:
    df = forward_df(Z, force, f0_hz=F0, k_n_per_m=K, amplitude_m=AMP)
    head = ["Experiment\tZ spectroscopy\t", "X (m)\t1.000000E-08\t", "Y (m)\t0.000000E+00\t"]
    if with_header:
        head += [f"Oscillation Control>Center Frequency (Hz)\t{F0:.6E}\t",
                 f"Oscillation Control>Amplitude Setpoint (m)\t{AMP:.6E}\t"]
    # Z rel is 0 at the start and negative towards the surface, as the corpus has it
    z_rel = Z - Z[-1]
    lines = head + ["", "[DATA]", f"Z rel (m)\t{df_name}"]
    lines += [f"{a:.6E}\t{b:.6E}" for a, b in zip(z_rel, df)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_it_inverts_a_curve_and_gets_the_force_back(tmp_path):
    res = InvertForceSaderJarvis().execute(None, {
        "dat_path": str(_dat(tmp_path / "atom.dat")), "k_n_per_m": K})
    assert res.success, res.error
    assert res.data["verdict"] == "well", res.data.get("reasons")
    assert res.data["f_min_pn"] == pytest.approx(true_f_min_pn(), rel=0.15)


@pytest.mark.parametrize("df_name", ["OC M1 Freq. Shift (Hz)", "Frequency Shift (Hz)"])
def test_both_names_the_frequency_shift_column_goes_by_are_recognised(tmp_path, df_name):
    """The corpus writes the instrument's abbreviation; the protocol document writes it out."""
    res = InvertForceSaderJarvis().execute(None, {
        "dat_path": str(_dat(tmp_path / "n.dat", df_name=df_name)), "k_n_per_m": K})
    assert res.success, res.error
    assert res.data["verdict"] == "well"


def test_the_frequency_and_amplitude_come_from_the_header_when_not_given(tmp_path):
    res = InvertForceSaderJarvis().execute(None, {
        "dat_path": str(_dat(tmp_path / "h.dat")), "k_n_per_m": K})
    assert res.data["f0_hz"] == pytest.approx(F0, rel=1e-6)
    assert res.data["amplitude_m"] == pytest.approx(AMP, rel=1e-6)


def test_without_a_spring_constant_it_refuses_instead_of_assuming_one(tmp_path):
    """k is in no header — on a real rig it comes from the sensor's calibration. Assuming a
    value scales every force in the result by whatever the assumption was wrong by."""
    res = InvertForceSaderJarvis().execute(None, {"dat_path": str(_dat(tmp_path / "k.dat"))})
    assert res.success is False
    assert "k_n_per_m" in (res.error or "") or "弹性" in (res.error or "")


def test_without_a_frequency_anywhere_it_refuses(tmp_path):
    res = InvertForceSaderJarvis().execute(None, {
        "dat_path": str(_dat(tmp_path / "nof.dat", with_header=False)), "k_n_per_m": K})
    assert res.success is False


def test_a_background_curve_is_subtracted_and_reported_as_used(tmp_path):
    atom = _dat(tmp_path / "a.dat")
    bg = _dat(tmp_path / "b.dat", force=lambda z: -1e-19 * 2e-9 / (6 * (np.asarray(z, float) + 3e-10) ** 2))
    res = InvertForceSaderJarvis().execute(None, {
        "dat_path": str(atom), "background_dat_path": str(bg), "k_n_per_m": K})
    assert res.success and res.data["background_used"] is True


def test_a_missing_file_is_a_failure(tmp_path):
    res = InvertForceSaderJarvis().execute(None, {"dat_path": str(tmp_path / "no.dat"),
                                                  "k_n_per_m": K})
    assert res.success is False
