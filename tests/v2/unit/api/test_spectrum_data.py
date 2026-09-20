"""Numeric spectrum extraction for the Data tab (``webui.spectrum_data``).

A point spectrum used to reach the frontend as a PNG of the file's first two
columns. These tests pin what replaced it, and in particular the one distinction
that must survive: **dI/dV has three states**, not two. A real lock-in channel, a
numeric derivative of I(V), and nothing at all are different situations, and a
viewer that renders the middle one under a bare "dI/dV" label shows the operator
a noisy derivative where they expect a measured trace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mast.webui.spectrum_data import extract_spectrum

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "nanonis"
_DAT = _FIXTURES / "bias_spectroscopy_200pt.dat"


def _write_dat(path: Path, columns: list[str], rows: list[list[float]],
               experiment: str = "bias spectroscopy") -> Path:
    """A minimal Nanonis .dat: tab-separated header, ``[DATA]``, names, rows."""
    head = (
        f"Experiment\t{experiment}\t\r\n"
        "Saved Date\t01.01.2000 00:00:00\t\r\n"
        "\r\n[DATA]\r\n"
    )
    body = "\t".join(columns) + "\r\n"
    for r in rows:
        body += "\t".join(f"{v:.7E}" for v in r) + "\r\n"
    # write_BYTES: write_text would translate the \n half of each \r\n on
    # Windows, producing \r\r\n and a file the real parser cannot read — a
    # fixture that is not the format under test.
    path.write_bytes((head + body).encode("utf-8"))
    return path


def _iv_rows(n: int = 32, gain: float = 1e-9) -> list[list[float]]:
    """I(V) over ±2 V with a smooth non-linear current."""
    out = []
    for i in range(n):
        v = -2.0 + 4.0 * i / (n - 1)
        out.append([v, gain * (v ** 3), gain * (v ** 3) * 0.98])
    return out


# ── dI/dV: three states ────────────────────────────────────────────────


def test_lockin_column_is_used_as_didv() -> None:
    """When the instrument measured dI/dV, that is what gets plotted."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = _write_dat(
            Path(td) / "s.dat",
            ["Bias calc (V)", "Current (A)", "LIX 1 omega (A)"],
            [[v, i, i * 2] for v, i, _ in _iv_rows()],
        )
        res = extract_spectrum(str(p))
    assert res["didv_source"] == "lockin"
    didv = [s for s in res["series"] if s["id"] == "didv"]
    assert len(didv) == 1
    assert didv[0]["source"] == "file"
    assert didv[0]["name"] == "LIX 1 omega (A)"


def test_without_a_lockin_column_didv_is_numeric_and_says_so() -> None:
    res = extract_spectrum(str(_DAT))
    assert res["didv_source"] == "numeric"
    didv = [s for s in res["series"] if s["id"] == "didv"][0]
    assert didv["source"] == "numeric"
    # The label itself must carry the caveat: series get rendered in a legend
    # where `source` is not necessarily visible.
    assert "数值微分" in didv["name"]


def test_numeric_didv_actually_differentiates() -> None:
    """d/dV of a known cubic — otherwise the label is the only evidence the
    numbers mean anything."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = _write_dat(Path(td) / "s.dat",
                       ["Bias calc (V)", "Current (A)"],
                       [[v, i] for v, i, _ in _iv_rows(64)])
        res = extract_spectrum(str(p))
    didv = [s for s in res["series"] if s["id"] == "didv"][0]
    sweep = res["sweep"]
    # d/dV of 1e-9·V³ is 3e-9·V²; check a mid-range point away from the edges
    # where np.gradient is one-sided.
    i = len(sweep) // 4
    expected = 3e-9 * sweep[i] ** 2
    assert didv["values"][i] == pytest.approx(expected, rel=0.05)


def test_a_non_sweeping_bias_gets_no_numeric_didv() -> None:
    """Dividing by a bias that is not moving produces a wall, not a spectrum."""
    import tempfile

    rows = [[0.5, 1e-9 * i] for i in range(16)]          # bias constant
    with tempfile.TemporaryDirectory() as td:
        p = _write_dat(Path(td) / "s.dat", ["Bias calc (V)", "Current (A)"], rows)
        res = extract_spectrum(str(p))
    assert res["didv_source"] is None
    assert not [s for s in res["series"] if s["id"] == "didv"]


# ── column roles ───────────────────────────────────────────────────────


def test_forward_and_backward_current_are_told_apart() -> None:
    res = extract_spectrum(str(_DAT))
    ids = [s["id"] for s in res["series"]]
    assert "current" in ids and "current_bwd" in ids
    fwd = [s for s in res["series"] if s["id"] == "current"][0]
    bwd = [s for s in res["series"] if s["id"] == "current_bwd"][0]
    assert "bwd" not in fwd["name"].lower()
    assert "bwd" in bwd["name"].lower()


def test_kind_comes_from_the_data_not_the_header() -> None:
    """字段标签会说谎 — the header's Experiment field is whatever the software
    was last set to."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = _write_dat(Path(td) / "s.dat",
                       ["Bias calc (V)", "Current (A)"],
                       [[v, i] for v, i, _ in _iv_rows()],
                       experiment="Z spectroscopy")       # header lies
        res = extract_spectrum(str(p))
    assert res["kind"] == "iv"
    assert res["kind_evidence"]


def test_iz_spectrum_sweeps_z() -> None:
    import tempfile

    rows = [[0.5, 1e-9 * (0.9 ** i), -1e-9 + i * 1e-11] for i in range(24)]
    with tempfile.TemporaryDirectory() as td:
        p = _write_dat(Path(td) / "s.dat",
                       ["Bias calc (V)", "Current (A)", "Z rel (m)"], rows)
        res = extract_spectrum(str(p))
    assert res["kind"] == "iz"
    assert res["sweep_name"] == "Z rel (m)"


# ── nothing is dropped ─────────────────────────────────────────────────


def test_every_column_name_is_reported_even_when_unplotted() -> None:
    res = extract_spectrum(str(_DAT))
    assert res["columns"] == ["Bias calc (V)", "Current (A)", "Current [bwd] (A)"]


def test_unrecognised_columns_still_produce_curves() -> None:
    """重排不是过滤, applied to columns: a file whose roles we cannot name is
    still shown rather than reported as empty."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = _write_dat(Path(td) / "s.dat",
                       ["Sweep (arb)", "Signal A", "Signal B"],
                       [[float(i), i * 2.0, i * 3.0] for i in range(10)])
        res = extract_spectrum(str(p))
    assert res["degraded"] is False
    assert len(res["series"]) == 2
    assert res["sweep_name"] == "Sweep (arb)"
    assert [s["name"] for s in res["series"]] == ["Signal A", "Signal B"]


def test_nan_becomes_null_not_a_number() -> None:
    """JSON has no NaN: letting one through makes the whole body unparseable,
    and a null is also what breaks the line instead of drawing across the gap."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.dat"
        p.write_bytes(
            b"Experiment\tbias spectroscopy\t\r\n\r\n[DATA]\r\n"
            b"Bias calc (V)\tCurrent (A)\r\n"
            b"-1.0\t1.0E-9\r\n"
            b"0.0\tNaN\r\n"
            b"1.0\t3.0E-9\r\n"
        )
        res = extract_spectrum(str(p))
    cur = [s for s in res["series"] if s["id"] == "current"][0]
    assert cur["values"][1] is None
    import json

    json.dumps(res)          # would raise if a bare NaN survived


# ── degradation ────────────────────────────────────────────────────────


def test_missing_file_degrades() -> None:
    res = extract_spectrum("no/such/file.dat")
    assert res["found"] is False and res["degraded"] is True


def test_garbage_file_degrades_not_raises(tmp_path) -> None:
    p = tmp_path / "bad.dat"
    p.write_bytes(b"\x00\x01 not a spectrum at all")
    res = extract_spectrum(str(p))
    assert res["degraded"] is True
    assert res["series"] == []


def test_single_column_cannot_make_a_curve(tmp_path) -> None:
    p = _write_dat(tmp_path / "one.dat", ["Bias calc (V)"], [[float(i)] for i in range(5)])
    res = extract_spectrum(str(p))
    assert res["degraded"] is True


# ── pattern copies must not drift ──────────────────────────────────────


def test_column_patterns_match_the_skills_that_own_them() -> None:
    """``webui`` holds COPIES so that drawing a curve does not pull in the skill
    machinery. Copies rot silently — this is the alarm.

    If a skill learns a new lock-in spelling and the Data tab does not, the tab
    quietly falls back to numeric differentiation on a file that HAS a measured
    dI/dV column, and nothing anywhere reports a problem."""
    from mast.skills.builtins import spectrum_assess as sa
    from mast.skills.builtins import tip_spectro_assess as tsa
    from mast.webui import spectrum_data as sd

    assert sd._BWD_MARKERS == sa._BWD_MARKERS
    assert sd._BIAS_PATTERNS == sa._BIAS_PATTERNS
    assert sd._CURRENT_PATTERNS == sa._CURRENT_PATTERNS
    assert sd._Z_PATTERNS == sa._Z_PATTERNS
    assert sd._DIDV_PATTERNS == tsa._DIDV_PATTERNS


# ── transport ──────────────────────────────────────────────────────────


def test_long_sweeps_are_decimated_and_stay_aligned(tmp_path) -> None:
    n = 50_000
    rows = [[-2.0 + 4.0 * i / (n - 1), 1e-9 * i] for i in range(n)]
    p = _write_dat(tmp_path / "long.dat", ["Bias calc (V)", "Current (A)"], rows)
    res = extract_spectrum(str(p))
    assert res["decimated"] is True
    assert res["n_points"] <= 20000
    # Every series must be the same length as the sweep, or the chart pairs the
    # wrong x with the wrong y.
    for s in res["series"]:
        assert len(s["values"]) == res["n_points"]
