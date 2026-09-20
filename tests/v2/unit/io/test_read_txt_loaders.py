"""read_txt (generic numeric text) + the shared multi-format loaders.

Covers the .txt/.csv path end to end and the load_image_2d / load_spectrum /
grid_to_spectra helpers that wire the analysis skills + data viewer to the real
Nanonis readers. .sxm/.3ds/.dat binary parsing is covered by test_nanonis_files.
"""
from __future__ import annotations

import numpy as np
import pytest

from mast.data import (
    grid_to_spectra,
    load_image_2d,
    load_spectrum,
    sample_grid_column,
)
from mast.io.nanonis_files import read_txt


# ── read_txt ────────────────────────────────────────────────────────────
class TestReadTxt:
    def test_matrix(self, tmp_path):
        p = tmp_path / "m.txt"
        np.savetxt(p, np.arange(12.0).reshape(3, 4))
        out = read_txt(str(p))
        assert out["matrix"].shape == (3, 4)
        assert len(out["columns"]) == 4

    def test_csv(self, tmp_path):
        p = tmp_path / "m.csv"
        np.savetxt(p, np.arange(6.0).reshape(3, 2), delimiter=",")
        assert read_txt(str(p))["matrix"].shape == (3, 2)

    def test_comments_and_header(self, tmp_path):
        p = tmp_path / "h.txt"
        p.write_text("# a comment\n; another\nBias\tCurrent\n-1 0.1\n0 0.2\n1 0.3\n")
        out = read_txt(str(p))
        assert out["matrix"].shape == (3, 2)
        assert list(out["columns"]) == ["Bias", "Current"]

    def test_ragged_rows_nan_padded(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("1 2 3\n4 5\n6 7 8\n")  # one short row
        out = read_txt(str(p))
        assert out["matrix"].shape == (3, 3)
        assert np.isnan(out["matrix"][1, 2])

    def test_nothing_numeric_raises_where_the_file_is_still_known(self, tmp_path):
        """A file with no parseable numbers must fail HERE, not downstream.

        This used to return an empty matrix. The emptiness then travelled until
        numpy raised "zero-size array to reduction operation fmin which has no
        identity" somewhere unrelated — an error naming neither the file nor the
        reason, so the agent could only retry the same wrong input. ``load_image_2d`` already treated empty as an
        error (see TestLoadImage2d.test_empty_raises); this just moves the check
        to where the filename and the offending line are still in hand.
        """
        p = tmp_path / "e.txt"
        p.write_text("# only comments\n# nothing numeric\n")
        with pytest.raises(ValueError, match="No numeric rows"):
            read_txt(str(p))

    def test_tool_return_dump_is_named_as_such(self, tmp_path):
        """The exact file that crashed the field run: a skill's tool-return
        sidecar (a dict repr), handed to a scan loader."""
        p = tmp_path / "AcquireSTS_767f4756.txt"
        p.write_text("{'acquisition_complete': True, 'num_points': 200, "
                     "'voltage': [-1.0, -0.98], 'current': [1e-10, 2e-10]}")
        with pytest.raises(ValueError) as ei:
            read_txt(str(p))
        msg = str(ei.value)
        assert "tool-return record" in msg, "must say WHAT the file is"
        assert "NOT a scan file" in msg, "must say what it is not"
        assert ".dat" in msg, "must point at the right file to read instead"


# ── load_image_2d ───────────────────────────────────────────────────────
class TestLoadImage2d:
    def test_npy(self, tmp_path):
        p = tmp_path / "a.npy"
        np.save(p, np.arange(12.0).reshape(3, 4))
        arr = load_image_2d(p)
        assert arr.shape == (3, 4) and arr.dtype == np.float64

    def test_txt(self, tmp_path):
        p = tmp_path / "a.txt"
        np.savetxt(p, np.ones((5, 5)))
        assert load_image_2d(p).shape == (5, 5)

    def test_npz_first_key(self, tmp_path):
        p = tmp_path / "a.npz"
        np.savez(p, img=np.zeros((2, 3)))
        assert load_image_2d(p).shape == (2, 3)

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image_2d(tmp_path / "nope.npy")

    def test_empty_raises(self, tmp_path):
        p = tmp_path / "e.txt"
        p.write_text("# nothing\n")
        with pytest.raises(ValueError):
            load_image_2d(p)


# ── load_spectrum + grid_to_spectra ─────────────────────────────────────
class TestLoadSpectrum:
    def test_npy(self, tmp_path):
        p = tmp_path / "s.npy"
        np.save(p, np.linspace(-1, 1, 64))
        assert load_spectrum(p).shape == (64,)

    def test_txt_columns(self, tmp_path):
        p = tmp_path / "s.txt"
        np.savetxt(p, np.column_stack([np.linspace(-1, 1, 50), np.random.rand(50)]))
        assert load_spectrum(p).shape == (50, 2)

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_spectrum(tmp_path / "nope.dat")


def test_grid_to_spectra():
    assert grid_to_spectra(np.zeros((4, 5, 32))).shape == (20, 32)
    assert grid_to_spectra(np.zeros((7, 32))).shape == (7, 32)
    assert grid_to_spectra(np.zeros(32)).shape == (1, 32)


def test_sample_grid_column():
    # cube (ny=2, nx=2, n=3); g[y, x] corner spectra are constant-valued 0/1/2/3
    cube = np.zeros((2, 2, 3))
    cube[0, 0] = 0.0; cube[0, 1] = 1.0; cube[1, 0] = 2.0; cube[1, 1] = 3.0
    np.testing.assert_allclose(sample_grid_column(cube, 0.5, 0.5), [1.5, 1.5, 1.5])  # center
    np.testing.assert_allclose(sample_grid_column(cube, 1, 0), [1, 1, 1])            # exact pixel
    np.testing.assert_allclose(sample_grid_column(cube, 9, 9), [3, 3, 3])            # clamp
    import pytest as _pt
    with _pt.raises(ValueError):
        sample_grid_column(np.zeros((3, 3)), 0, 0)  # not a 3-D cube
