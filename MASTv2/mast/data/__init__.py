"""MAST data processing: file I/O, image processing, quality metrics, visualization."""

from __future__ import annotations

# The .sxm / .dat / .3ds byte parsers live in ONE place — mast.io.nanonis_files
# (the hardened, size-guarded readers the data viewer + load_image_2d use). These
# names are re-exported here for convenience; the old mast.data.formats duplicate
# parser was removed (2026-07-20) so a file can only ever be read one way.
from mast.io.nanonis_files import read_3ds, read_dat, read_sxm
from .loaders import (
    SUPPORTED_IMAGE_EXT,
    SUPPORTED_SPECTRUM_EXT,
    grid_to_spectra,
    load_image_2d,
    load_spectrum,
    load_spectrum_named,
    sample_grid_column,
)
from .processors import (
    drift_estimate,
    fft2d,
    fft_filter,
    line_by_line_level,
    plane_subtract,
)
from .quality import fft_quality_score, noise_estimate, rms_roughness
from .visualization import plot_fft, plot_sts, plot_topo_overlay, plot_topography

__all__ = [
    "read_sxm",
    "read_dat",
    "read_3ds",
    "load_image_2d",
    "load_spectrum",
    "load_spectrum_named",
    "grid_to_spectra",
    "sample_grid_column",
    "SUPPORTED_IMAGE_EXT",
    "SUPPORTED_SPECTRUM_EXT",
    "plane_subtract",
    "line_by_line_level",
    "fft2d",
    "fft_filter",
    "drift_estimate",
    "fft_quality_score",
    "rms_roughness",
    "noise_estimate",
    "plot_topography",
    "plot_sts",
    "plot_fft",
    "plot_topo_overlay",
]
