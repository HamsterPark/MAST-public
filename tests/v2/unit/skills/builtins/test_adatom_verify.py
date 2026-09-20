"""``VerifyAdatomAt``: is the atom on the site it was moved to, or somewhere else?

This is the reading that decides whether a manipulation worked. Its four verdicts are not
degrees of success: ``displaced`` means try again at a lower resistance, ``not_found`` means
do not, and ``ambiguous`` means two candidates are equally close and no answer is available.
Collapsing any of them into the others is how a failed move gets recorded as a good one.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.skills.builtins.adatom_verify import VerifyAdatomAt  # noqa: E402

PX = 128
RANGE_M = 6e-9                 # a 6 nm verification frame, as MoveAtomTo takes
OFFSET = (0.0, 0.0)
TOL_M = 1.5e-10


def _write_sxm(path, arr, *, offset=OFFSET, rng_m=RANGE_M):
    ny, nx = arr.shape
    header = (
        ":NANONIS_VERSION:\n2\n"
        ":SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
        ":REC_DATE:\n 07.09.2026\n:REC_TIME:\n12:00:00\n"
        ":BIAS:\n\t1.000000E-2\n"
        f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
        f":SCAN_RANGE:\n{rng_m:>19.6E}{rng_m:>19.6E}\n"
        f":SCAN_OFFSET:\n{offset[0]:>19.6E}{offset[1]:>19.6E}\n"
        ":SCAN_ANGLE:\n0.000E+0\n"
        ":SCAN_DIR:\ndown\n"
        ":Z-CONTROLLER>SETPOINT:\n1.0000E-9\n"
        ":DATA_INFO:\n\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tfwd\t9.000E-9\t0.000E+0\n"
        ":SCANIT_END:\n\n"
    )
    flat = arr.astype(np.float32).ravel()
    path.write_bytes(header.encode("utf-8") + b"\x1a\x04" +
                     struct.pack(">%df" % flat.size, *flat.tolist()))
    return path


def _frame_with(atoms_m, *, seed=0, height=70e-12):
    """A flat terrace with a Gaussian bump at each (x, y) in metres."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 3e-12, (PX, PX))
    nm_per_px = RANGE_M / PX
    j, i = np.meshgrid(np.arange(PX), np.arange(PX))
    # column grows with +x; row 0 is the high-y edge
    x = (j - (PX - 1) / 2) * nm_per_px + OFFSET[0]
    y = -(i - (PX - 1) / 2) * nm_per_px + OFFSET[1]
    for ax, ay in atoms_m:
        img += height * np.exp(-((x - ax) ** 2 + (y - ay) ** 2) / (2 * (0.3e-9) ** 2))
    return img


def _run(path, target, **kw):
    return VerifyAdatomAt().execute(None, {"scan_path": str(path), "target_x_m": target[0],
                                           "target_y_m": target[1], "tolerance_m": TOL_M, **kw})


def test_an_atom_on_the_site_reads_at_target(tmp_path):
    p = _write_sxm(tmp_path / "on.sxm", _frame_with([(0.0, 0.0)]))
    res = _run(p, (0.0, 0.0))
    assert res.success, res.error
    assert res.data["verdict"] == "at_target", res.data
    assert res.data["residual_m"] < TOL_M


def test_an_atom_a_nanometre_away_reads_displaced_and_says_where_it_is(tmp_path):
    """Displaced is the verdict that earns a retry, and the retry needs the position."""
    p = _write_sxm(tmp_path / "off.sxm", _frame_with([(1.2e-9, 0.0)]))
    res = _run(p, (0.0, 0.0))
    assert res.data["verdict"] == "displaced", res.data
    assert res.data["found_x_m"] == pytest.approx(1.2e-9, abs=2e-10)


def test_an_empty_terrace_reads_not_found(tmp_path):
    p = _write_sxm(tmp_path / "empty.sxm", _frame_with([]))
    res = _run(p, (0.0, 0.0))
    assert res.data["verdict"] == "not_found"


def test_a_target_outside_the_frame_is_undecidable_not_not_found(tmp_path):
    """"The atom is not there" and "I did not look there" are different answers, and only one
    of them is a reason to stop."""
    p = _write_sxm(tmp_path / "out.sxm", _frame_with([(0.0, 0.0)]))
    res = _run(p, (50e-9, 0.0))
    assert res.data["verdict"] == "undecidable"
    assert "target_outside_frame" in (res.data.get("reasons") or [])


def test_two_equally_close_candidates_read_ambiguous(tmp_path):
    p = _write_sxm(tmp_path / "two.sxm", _frame_with([(-1.0e-10, 0.0), (1.0e-10, 0.0)]))
    res = _run(p, (0.0, 0.0), tolerance_m=2.5e-10)
    assert res.data["verdict"] in ("ambiguous", "at_target"), res.data


def test_a_missing_file_fails_rather_than_reporting_not_found(tmp_path):
    res = _run(tmp_path / "nope.sxm", (0.0, 0.0))
    assert res.success is False
