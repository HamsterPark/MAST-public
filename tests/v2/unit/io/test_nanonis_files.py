"""Unit tests for mast.io.nanonis_files: .sxm / .dat / .3ds readers.

Fixtures under tests/v2/fixtures/nanonis/ are generated entirely from formulas
by generate.py. They contain no instrument measurements or recorded metadata.
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Force-import the v2 mast package (tests live outside MASTv2/).
_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if _MASTV2_ROOT not in sys.path:
    sys.path.insert(0, _MASTV2_ROOT)
# Evict any v1 `mast` that may already be cached.
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.io.nanonis_files import (  # noqa: E402
    load_scan_file,
    read_3ds,
    read_dat,
    read_sxm,
    _parse_3ds_header,
)

_FIX = Path(__file__).resolve().parents[2] / "fixtures" / "nanonis"
SXM = str(_FIX / "scan_topography.sxm")
DAT = str(_FIX / "bias_spectroscopy_200pt.dat")
THREEDS = str(_FIX / "grid_3x3_bias.3ds")


# ---------------------------------------------------------------------------
# .dat — real Nanonis V5e file
# ---------------------------------------------------------------------------

def test_dat_header_populated():
    """The bug fixed in 2026-05: header was empty because parser used `=`
    instead of TAB. Make sure that regression doesn't come back."""
    d = read_dat(DAT)
    assert d["header"], "header should not be empty for a real Nanonis .dat"
    # Check the synthetic header fields.
    assert d["header"]["Experiment"] == "bias spectroscopy"
    assert "Saved Date" in d["header"]
    assert "X (m)" in d["header"]


def test_dat_columns_correctly_named():
    """The earlier bug also produced column names "Filter type" / "None"
    because the parser greedily matched any tab-line as a column header.
    Real columns must be the post-[DATA] line."""
    d = read_dat(DAT)
    cols = list(d["columns"].keys())
    assert cols == ["Bias calc (V)", "Current (A)", "Current [bwd] (A)"], (
        f"unexpected columns: {cols}"
    )


def test_dat_numeric_values_sensible():
    d = read_dat(DAT)
    bias = d["columns"]["Bias calc (V)"]
    current = d["columns"]["Current (A)"]
    assert bias.shape == (200,)
    assert current.shape == (200,)
    # Sweep range -2..+2 V (configured BiasSpectr_LimitsSet).
    assert -2.01 < bias.min() < -1.99
    assert 1.99 < bias.max() < 2.01
    # Current is in nA scale on the simulator.
    assert abs(current).max() < 1e-7


def test_dat_no_data_section():
    """Header-only file (no [DATA] marker) should return empty columns."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write("Experiment\tbias spectroscopy\nKey\tValue\n")
        path = f.name
    try:
        d = read_dat(path)
        assert d["header"]["Experiment"] == "bias spectroscopy"
        assert d["columns"] == {}
    finally:
        Path(path).unlink()


def test_dat_extra_columns_get_positional_names():
    """If the column-header line is shorter than the data rows, the trailing
    columns must still be returned under positional names so callers don't
    silently lose data."""
    text = (
        "Experiment\ttest\n"
        "[DATA]\n"
        "Bias (V)\tCurrent (A)\n"          # only 2 names
        "-1\t-1e-9\t1e-9\n"                  # but 3 columns of data
        "0\t0\t0\n"
        "1\t1e-9\t-1e-9\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        d = read_dat(path)
        assert list(d["columns"]) == ["Bias (V)", "Current (A)", "column_2"]
        assert d["columns"]["column_2"].shape == (3,)
    finally:
        Path(path).unlink()


# ---------------------------------------------------------------------------
# .3ds — real file from the simulator + synthetic-legacy compatibility
# ---------------------------------------------------------------------------

def test_3ds_real_file_basic_shape():
    d = read_3ds(THREEDS)
    assert d["grid"].shape == (3, 3, 51)
    assert d["bias"].shape == (51,)
    # The synthetic grid uses sweep limits -1.0 and 1.0.
    assert d["bias"][0] == pytest.approx(-1.0, abs=1e-6)
    assert d["bias"][-1] == pytest.approx(1.0, abs=1e-6)
    # Header essentials.
    assert d["header"]["grid_dim"] == [3, 3]
    assert d["header"]["points"] == 51
    assert d["header"]["num_parameters"] == 12       # 2 fixed + 10 experiment
    assert d["header"]["channels"] == ["Current (A)"]
    assert d["header"]["fixed_parameters"] == ["Sweep Start", "Sweep End"]


def test_3ds_per_pixel_params_extracted():
    d = read_3ds(THREEDS)
    pa = d["params"]["param_array"]
    assert pa.shape == (3, 3, 12)
    # Pixel (0,0) fixed params must be the sweep limits we configured.
    sweep_start, sweep_end = pa[0, 0, 0], pa[0, 0, 1]
    assert sweep_start == pytest.approx(-1.0, abs=1e-6)
    assert sweep_end == pytest.approx(1.0, abs=1e-6)
    # Experiment param "Z (m)" must be a finite, plausible STM tip height.
    expt_names = d["params"]["experiment_param_names"]
    z_idx = 2 + expt_names.index("Z (m)")   # offset by 2 fixed params
    z_value = float(pa[0, 0, z_idx])
    assert 1e-9 < z_value < 1e-6, f"Z(m) outside plausible STM range: {z_value}"


def test_3ds_legacy_synthetic_header():
    """Older .3ds variants put Sweep Start / Sweep End directly in the
    header. The parser must still honour that path."""
    nx, ny, n_points = 4, 4, 50
    header = (
        "Grid dim=4 x 4\r\nSweep Signal=Bias (V)\r\nChannels=Current (A)\r\n"
        "Points=50\r\nNum Parameters=2\r\nNum Channels=1\r\n"
        "Sweep Start=-0.5\r\nSweep End=0.5\r\n"
    ).encode() + b"\r\n:HEADER_END:\r\n"
    bias = np.linspace(-0.5, 0.5, n_points).astype(">f4")
    floats = []
    for iy in range(ny):
        for ix in range(nx):
            floats.extend([np.float32(ix), np.float32(iy)])
            spectrum = (np.sin(bias * 3 + ix + iy) * 1e-9).astype(">f4")
            floats.extend(spectrum.tolist())
    binary = b"".join(struct.pack(">f", float(v)) for v in floats)
    with tempfile.NamedTemporaryFile(suffix=".3ds", delete=False) as f:
        f.write(header)
        f.write(binary)
        path = f.name
    try:
        d = read_3ds(path)
        assert d["grid"].shape == (4, 4, 50)
        assert d["bias"][0] == pytest.approx(-0.5)
        assert d["bias"][-1] == pytest.approx(0.5)
        expected = np.sin(np.linspace(-0.5, 0.5, n_points) * 3 + 3) * 1e-9
        got = d["grid"][2, 1, :]
        assert np.max(np.abs(expected - got)) < 1e-12
    finally:
        Path(path).unlink()


def test_3ds_multi_channel_returns_first_channel_data():
    """Channels="A;B" → 2 channels derived from the NAME list (not int('A')),
    and the reader returns the FIRST channel's spectrum per pixel. (Migrated from
    the removed mast.data.formats regression suite — canonical reader parity.)"""
    nx = ny = 1
    n_points = 3
    header = (
        "Grid dim=1 x 1\r\n"
        'Channels="Current (A);LI Demod 1 X (A)"\r\n'
        f"Points={n_points}\r\n# Parameters (4 byte)=0\r\n"
        "Sweep Start=0\r\nSweep End=1\r\n"
    ).encode() + b"\r\n:HEADER_END:\r\n"
    # First channel points, then second channel points (no per-pixel params).
    floats = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0]
    binary = b"".join(struct.pack(">f", float(v)) for v in floats)
    with tempfile.NamedTemporaryFile(suffix=".3ds", delete=False) as f:
        f.write(header)
        f.write(binary)
        path = f.name
    try:
        d = read_3ds(path)
        assert d["header"]["num_channels"] == 2
        assert d["grid"].shape == (1, 1, 3)
        np.testing.assert_allclose(d["grid"][0, 0, :], [10.0, 11.0, 12.0])
    finally:
        Path(path).unlink()


def test_parse_3ds_header_strips_quotes_and_recognises_pound_keys():
    raw = (
        'Grid dim="6 x 4"\r\n'
        'Sweep Signal="Bias (V)"\r\n'
        '# Parameters (4 byte)=7\r\n'
        'Channels="Current (A);LI Demod 1 X (A)"\r\n'
        'Fixed parameters="Sweep Start;Sweep End"\r\n'
        'Points=128\r\n'
    )
    h = _parse_3ds_header(raw)
    assert h["grid_dim"] == [6, 4]
    assert h["num_parameters"] == 7
    assert h["channels"] == ["Current (A)", "LI Demod 1 X (A)"]
    assert h["num_channels"] == 2
    assert h["fixed_parameters"] == ["Sweep Start", "Sweep End"]
    assert h["points"] == 128
    assert h["sweep_signal"] == "Bias (V)"


# ---------------------------------------------------------------------------
# .sxm — real scan
# ---------------------------------------------------------------------------

def test_sxm_real_file():
    d = read_sxm(SXM)
    assert d["header"], "sxm header should be populated"
    # Standard scan_pixels metadata present.
    assert "scan_pixels" in d["header"]
    nx, ny = d["header"]["scan_pixels"]
    assert nx > 0 and ny > 0
    # At least one channel with forward+backward frames.
    assert d["channels"], "channels dict empty"
    first_ch = next(iter(d["channels"].values()))
    assert "forward" in first_ch
    assert first_ch["forward"].shape == (ny, nx)


# ---------------------------------------------------------------------------
# load_scan_file dispatcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path, expected_keys", [
    (SXM, {"header", "channels"}),
    (DAT, {"header", "columns"}),
    (THREEDS, {"header", "grid", "bias", "params"}),
])
def test_load_scan_file_dispatch(path, expected_keys):
    d = load_scan_file(path)
    assert set(d.keys()) == expected_keys


def test_load_scan_file_rejects_unknown_extension():
    # .txt/.csv are now supported (read_txt); use a genuinely-unknown extension.
    with pytest.raises(ValueError):
        load_scan_file("/tmp/foo.bin")


# ---------------------------------------------------------------------------
# GUI rendering — make sure scan_preview.render_scan_preview produces a
# Figure (not None) for each supported extension.
# ---------------------------------------------------------------------------

def test_render_scan_preview_dispatches_all_extensions(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from mast.webui.scan_preview import render_scan_preview

    fig_sxm = render_scan_preview(SXM)
    assert fig_sxm is not None
    assert len(fig_sxm.axes) >= 1

    fig_dat = render_scan_preview(DAT)
    assert fig_dat is not None
    # .dat path is a single axis line plot.
    ax = fig_dat.axes[0]
    assert len(ax.lines) >= 1
    assert ax.get_xlabel() == "Bias calc (V)"

    fig_3ds = render_scan_preview(THREEDS)
    assert fig_3ds is not None
    # .3ds path: heatmap subplot (with colorbar) + spectrum subplot → 3 axes total.
    # Order from fig.subplots(1,2): [left=heatmap, right=spectrum, then colorbar].
    assert len(fig_3ds.axes) == 3
    images = sum(len(ax.images) for ax in fig_3ds.axes)
    lines = sum(len(ax.lines) for ax in fig_3ds.axes)
    assert images >= 1, "heatmap imshow missing"
    assert lines >= 1, "spectrum line missing"


def test_render_scan_preview_returns_placeholder_for_unknown_ext():
    import matplotlib
    matplotlib.use("Agg")
    from mast.webui.scan_preview import render_scan_preview

    # Existing file with unsupported extension → placeholder Figure (not None).
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not a scan file")
        path = f.name
    try:
        fig = render_scan_preview(path)
        assert fig is not None  # placeholder, not None
        assert len(fig.axes) == 1
    finally:
        Path(path).unlink()


def test_render_scan_preview_returns_none_for_missing_file():
    import matplotlib
    matplotlib.use("Agg")
    from mast.webui.scan_preview import render_scan_preview
    assert render_scan_preview("/nonexistent/path.sxm") is None
