"""Robustness tests for mast.io.nanonis_files parsing.

Covers self-review findings #71, #72, #73, #123, #124, #125 — corrupt /
adversarial Nanonis files must degrade gracefully (clear error or empty
result) instead of crashing with MemoryError / ValueError or silently
fabricating fake data via negative-dimension reshape.

All fixtures here are SYNTHETIC bad files built in-process; no real Nanonis
hardware, network, or LLM is required.
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# --- v2 package bootstrap (tests live outside MASTv2/) ----------------------
_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.io import nanonis_files  # noqa: E402
from mast.io.nanonis_files import (  # noqa: E402
    read_3ds,
    read_dat,
    read_sxm,
    _parse_sxm_header,
)


# ---------------------------------------------------------------------------
# helpers — build synthetic bad files on disk
# ---------------------------------------------------------------------------

def _write(suffix: str, data: bytes) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        f.write(data)
        return f.name
    finally:
        f.close()


def _make_sxm(header_text: str, body: bytes = b"") -> str:
    """Build a minimal .sxm: text header + \\x1a\\x04 marker + binary body."""
    return _write(".sxm", header_text.encode("utf-8") + b"\x1a\x04" + body)


def _make_3ds(header_text: str, body: bytes = b"") -> str:
    return _write(".3ds", header_text.encode("utf-8") + b"\r\n:HEADER_END:\r\n" + body)


def _make_dat(text: str) -> str:
    return _write(".dat", text.encode("utf-8"))


def _f32be(*vals: float) -> bytes:
    return b"".join(struct.pack(">f", float(v)) for v in vals)


# ===========================================================================
# #73  .sxm negative scan_pixels → must NOT silently fabricate data
# ===========================================================================

def test_sxm_negative_scan_pixels_returns_no_channels():
    """A negative nx/ny would be re-interpreted by reshape as a '-1 infer'
    wildcard, producing a frame of the wrong shape from arbitrary bytes.
    Guard with <=0 → empty channels."""
    # 32 floats of junk in the body so reshape(-1, ...) WOULD succeed if the
    # guard were missing.
    body = _f32be(*range(32))
    header = (
        ":SCAN_PIXELS:\n-4 8\n"
        ":DATA_INFO:\nChannel\tName\tUnit\tDirection\n1\tZ\tm\tboth\n"
    )
    path = _make_sxm(header, body)
    try:
        d = read_sxm(path)
        assert d["channels"] == {}, "negative scan_pixels must yield no channels"
        # header still parsed (or scan_pixels dropped) — never a fake frame.
        assert isinstance(d["header"], dict)
    finally:
        Path(path).unlink()


def test_sxm_zero_scan_pixels_returns_no_channels():
    header = ":SCAN_PIXELS:\n0 0\n:DATA_INFO:\nChannel\tName\n1\tZ\n"
    path = _make_sxm(header, _f32be(*range(16)))
    try:
        d = read_sxm(path)
        assert d["channels"] == {}
    finally:
        Path(path).unlink()


def test_sxm_valid_small_scan_still_reads():
    """Sanity: the guard must not break a legitimate tiny scan."""
    nx, ny = 2, 2
    frame_fwd = _f32be(1, 2, 3, 4)
    frame_bwd = _f32be(5, 6, 7, 8)
    header = (
        f":SCAN_PIXELS:\n{nx} {ny}\n"
        ":DATA_INFO:\nChannel\tName\tUnit\tDirection\n1\tZ\tm\tboth\n"
    )
    path = _make_sxm(header, frame_fwd + frame_bwd)
    try:
        d = read_sxm(path)
        assert "Z" in d["channels"]
        assert d["channels"]["Z"]["forward"].shape == (ny, nx)
        np.testing.assert_allclose(d["channels"]["Z"]["forward"], [[1, 2], [3, 4]])
    finally:
        Path(path).unlink()


# ===========================================================================
# #124  .sxm scan_pixels non-integer token → int() ValueError tolerated
# ===========================================================================

def test_sxm_non_integer_scan_pixels_does_not_raise():
    header = (
        ":SCAN_PIXELS:\n128.0 abc\n"
        ":DATA_INFO:\nChannel\tName\n1\tZ\n"
    )
    path = _make_sxm(header, _f32be(*range(8)))
    try:
        d = read_sxm(path)  # must not raise ValueError
        # scan_pixels dropped by parser → empty channels, header intact.
        assert d["channels"] == {}
        assert "scan_pixels" not in d["header"]
    finally:
        Path(path).unlink()


def test_parse_sxm_header_skips_bad_scan_pixels_tokens():
    h = _parse_sxm_header(":SCAN_PIXELS:\nNaN -\n")
    assert "scan_pixels" not in h


def test_sxm_nan_token_scan_pixels_via_reader():
    """Even if a token like 'NaN' slips through the header, read_sxm's own
    int() coercion must catch it rather than blow up."""
    header = ":SCAN_PIXELS:\nNaN 5\n:DATA_INFO:\nChannel\tName\n1\tZ\n"
    path = _make_sxm(header, _f32be(*range(8)))
    try:
        d = read_sxm(path)
        assert d["channels"] == {}
    finally:
        Path(path).unlink()


# ===========================================================================
# #123  .3ds negative Points / Grid dim → np.zeros negative dim ValueError
# ===========================================================================

@pytest.mark.parametrize("grid_line, points_line", [
    ("Grid dim=-4 x 4", "Points=51"),
    ("Grid dim=4 x -4", "Points=51"),
    ("Grid dim=4 x 4", "Points=-51"),
])
def test_3ds_negative_dims_return_empty(grid_line, points_line):
    header = (
        f"{grid_line}\r\nChannels=Current (A)\r\n{points_line}\r\n"
        "# Parameters (4 byte)=2\r\nNum Channels=1\r\n"
    )
    # Provide some junk body so the loop would otherwise run.
    path = _make_3ds(header, _f32be(*range(64)))
    try:
        d = read_3ds(path)  # must NOT raise "negative dimensions are not allowed"
        assert d["grid"].size == 0, "negative geometry must yield empty grid"
        assert d["bias"].size == 0
    finally:
        Path(path).unlink()


def test_3ds_negative_num_parameters_returns_empty():
    header = (
        "Grid dim=4 x 4\r\nChannels=Current (A)\r\nPoints=10\r\n"
        "# Parameters (4 byte)=-5\r\nNum Channels=1\r\n"
    )
    path = _make_3ds(header, _f32be(*range(64)))
    try:
        d = read_3ds(path)
        assert d["grid"].size == 0
    finally:
        Path(path).unlink()


# ===========================================================================
# #71  .3ds corrupt header with absurd grid → MemoryError must be prevented
# ===========================================================================

def test_3ds_oversized_grid_raises_clear_error_not_memoryerror():
    """A header claiming a multi-billion-element grid must be refused with a
    clear ValueError BEFORE np.zeros tries (and fails) to allocate it."""
    header = (
        "Grid dim=100000 x 100000\r\nChannels=Current (A)\r\nPoints=512\r\n"
        "# Parameters (4 byte)=2\r\nNum Channels=1\r\n"
    )
    path = _make_3ds(header, _f32be(*range(16)))
    try:
        with pytest.raises(ValueError) as ei:
            read_3ds(path)
        assert "safety limit" in str(ei.value) or "grid elements" in str(ei.value)
    finally:
        Path(path).unlink()


def test_3ds_truncated_blob_clipped_not_crashed():
    """Header declares a 4x4 grid but the binary blob only backs ~2 pixels.
    Must return a grid sized to the header with zero-fill for missing pixels,
    not crash."""
    nx, ny, n_points, n_params = 4, 4, 5, 2
    # Only write 2 full pixels' worth of data.
    floats_per_point = n_params + n_points
    body = _f32be(*([0.1] * (floats_per_point * 2)))
    header = (
        f"Grid dim={nx} x {ny}\r\nChannels=Current (A)\r\nPoints={n_points}\r\n"
        f"# Parameters (4 byte)={n_params}\r\nNum Channels=1\r\n"
    )
    path = _make_3ds(header, body)
    try:
        d = read_3ds(path)
        assert d["grid"].shape == (ny, nx, n_points)
        # First pixel populated, far pixels stayed zero-filled (no crash).
        assert np.any(d["grid"][0, 0, :] != 0.0)
        assert np.all(d["grid"][-1, -1, :] == 0.0)
    finally:
        Path(path).unlink()


def test_3ds_zero_points_returns_empty():
    header = "Grid dim=4 x 4\r\nChannels=Current (A)\r\nPoints=0\r\n# Parameters (4 byte)=2\r\n"
    path = _make_3ds(header, b"")
    try:
        d = read_3ds(path)
        assert d["grid"].size == 0
    finally:
        Path(path).unlink()


# ===========================================================================
# #72  .dat ragged rows → NaN-pad / drop instead of total failure
# ===========================================================================

def test_dat_ragged_rows_do_not_lose_all_data():
    """One short row must not nuke the whole dataset (NumPy 2.x raises
    ValueError on ragged np.array)."""
    text = (
        "Experiment\ttest\n"
        "[DATA]\n"
        "Bias (V)\tCurrent (A)\n"
        "-1\t-1e-9\n"
        "0\t0\n"
        "1\n"              # ragged: only one value
        "2\t2e-9\n"
    )
    path = _make_dat(text)
    try:
        d = read_dat(path)
        cols = d["columns"]
        assert "Bias (V)" in cols and "Current (A)" in cols
        # All 4 rows preserved.
        assert cols["Bias (V)"].shape == (4,)
        np.testing.assert_allclose(cols["Bias (V)"], [-1, 0, 1, 2])
        # The ragged row's missing Current value is NaN-filled, not dropped.
        assert np.isnan(cols["Current (A)"][2])
        np.testing.assert_allclose(
            cols["Current (A)"][[0, 1, 3]], [-1e-9, 0, 2e-9]
        )
    finally:
        Path(path).unlink()


def test_dat_extra_long_row_truncated_to_modal_width():
    """A stray over-wide row is truncated to the modal column count rather
    than corrupting the array shape."""
    text = (
        "Experiment\ttest\n"
        "[DATA]\n"
        "Bias (V)\tCurrent (A)\n"
        "-1\t-1e-9\n"
        "0\t0\t999\t999\n"     # over-wide stray row
        "1\t1e-9\n"
    )
    path = _make_dat(text)
    try:
        d = read_dat(path)
        cols = d["columns"]
        assert cols["Bias (V)"].shape == (3,)
        assert cols["Current (A)"].shape == (3,)
        # The over-wide row was truncated to 2 cols → its extras discarded.
        np.testing.assert_allclose(cols["Bias (V)"], [-1, 0, 1])
        np.testing.assert_allclose(cols["Current (A)"], [-1e-9, 0, 1e-9])
    finally:
        Path(path).unlink()


def test_dat_uniform_rows_unaffected_by_ragged_path():
    """Regression: well-formed uniform data must be untouched by the
    ragged-normalisation branch."""
    text = (
        "Experiment\ttest\n"
        "[DATA]\n"
        "Bias (V)\tCurrent (A)\n"
        "-1\t-1e-9\n"
        "0\t0\n"
        "1\t1e-9\n"
    )
    path = _make_dat(text)
    try:
        d = read_dat(path)
        np.testing.assert_allclose(d["columns"]["Bias (V)"], [-1, 0, 1])
        np.testing.assert_allclose(d["columns"]["Current (A)"], [-1e-9, 0, 1e-9])
        assert not np.any(np.isnan(d["columns"]["Current (A)"]))
    finally:
        Path(path).unlink()


# ===========================================================================
# #125  whole-file read has an upper size bound (clear error if exceeded)
# ===========================================================================

@pytest.mark.parametrize("reader, suffix, body", [
    (read_sxm, ".sxm", b":SCAN_PIXELS:\n2 2\n\x1a\x04"),
    (read_3ds, ".3ds", b"Grid dim=2 x 2\r\n:HEADER_END:\r\n"),
    (read_dat, ".dat", b"Experiment\ttest\n[DATA]\nBias\n1\n"),
])
def test_readers_refuse_oversized_files(reader, suffix, body, monkeypatch):
    """All three readers must refuse a file above MAX_FILE_BYTES with a clear
    ValueError (mentioning the size) instead of OOM-ing on f.read()."""
    path = _write(suffix, body)
    # Pretend the file is enormous without actually writing terabytes.
    monkeypatch.setattr(
        nanonis_files.os.path, "getsize", lambda p: nanonis_files.MAX_FILE_BYTES + 1
    )
    try:
        with pytest.raises(ValueError) as ei:
            reader(path)
        assert "too large" in str(ei.value).lower()
    finally:
        Path(path).unlink()


def test_size_limit_constant_is_sane():
    # Guard against an accidental tiny limit that would reject real files.
    assert nanonis_files.MAX_FILE_BYTES >= 1 * 1024 * 1024 * 1024
    assert nanonis_files.MAX_GRID_ELEMENTS >= 1_000_000


def test_readers_accept_normal_sized_files():
    """The size guard must let a legitimately-sized file through unchanged."""
    text = "Experiment\ttest\n[DATA]\nBias (V)\tI (A)\n-1\t-1e-9\n1\t1e-9\n"
    path = _make_dat(text)
    try:
        d = read_dat(path)
        assert d["columns"]["Bias (V)"].shape == (2,)
    finally:
        Path(path).unlink()
