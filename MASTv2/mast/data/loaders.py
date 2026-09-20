"""Format-agnostic loaders for analysis skills.

The paper-replication skills historically did a bare ``np.load(path)`` and so only
ever accepted ``.npy``. The real Nanonis readers already live in
``mast.io.nanonis_files`` (:func:`read_sxm`, :func:`read_dat`, :func:`read_3ds`,
:func:`read_txt`) — the SAME readers the data viewer uses — so these helpers
dispatch on the file extension and reuse them. A skill can then accept the formats
a user actually has on disk:

  * :func:`load_image_2d` — a 2-D topography image, from
    ``.npy`` / ``.npz`` / ``.sxm`` / ``.txt`` / ``.csv`` / ``.asc`` / ``.dat``.
  * :func:`load_spectrum` — a 1-D/2-D spectrum (or 3-D grid cube), from
    ``.npy`` / ``.npz`` / ``.dat`` / ``.txt`` / ``.csv`` / ``.3ds``.

Both RAISE on failure (``FileNotFoundError`` / ``ValueError``) — the same contract
as ``np.load`` — so they drop into the existing ``try: arr = np.load(path)`` call
sites without changing their error handling.

These use ``mast.io.nanonis_files`` — the single, hardened, size-guarded set of
canonical readers wired into the data viewer — so a skill reads a file
byte-for-byte the same way the viewer renders it. (A parallel ``mast.data.formats``
parser once existed and drifted from these; it was removed 2026-07-20 so there is
exactly one .sxm/.dat/.3ds byte parser in the system.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mast.io.nanonis_files import read_3ds, read_dat, read_sm4, read_sxm, read_txt

# Extensions handled as delimited numeric text (no Nanonis [DATA] marker needed).
_TEXT_EXT = {".txt", ".csv", ".asc", ".tsv", ".xyz"}
_SXM_EXT = ".sxm"
_SM4_EXT = ".sm4"
_3DS_EXT = ".3ds"
_DAT_EXT = ".dat"


def _pick_image_channel(channels: dict, prefer: str | None) -> str:
    """Choose the topography channel of an ``.sxm``: an explicit *prefer* substring
    first, then a Z/height/topography channel (not Current), else the first."""
    names = list(channels.keys())
    if not names:
        raise ValueError("sxm file exposes no channels")
    low = [str(c).lower() for c in names]
    if prefer:
        pl = prefer.lower()
        for i, c in enumerate(low):
            if pl in c:
                return names[i]
    for kw in ("z", "height", "topo"):
        for i, c in enumerate(low):
            if kw in c and "current" not in c:
                return names[i]
    return names[0]


def _channels_to_image(d: dict, path: Path, prefer: str | None) -> np.ndarray:
    """Pick a 2-D image from a read_sxm/read_sm4-shaped dict
    ``{channels: {name: {"forward": 2D, "backward": 2D}}}`` (forward preferred)."""
    channels = d.get("channels") or {}
    ch = _pick_image_channel(channels, prefer)
    frame = channels[ch]
    img = frame.get("forward")
    if img is None:
        img = frame.get("backward")
    if img is None:
        raise ValueError(f"{path.name}: channel {ch!r} has no forward/backward frame")
    return np.asarray(img, dtype=np.float64)


def _stack_columns(d: dict, name: str) -> np.ndarray:
    cols = d.get("columns") or {}
    if not cols:
        raise ValueError(f"{name}: no numeric data columns")
    return np.column_stack([np.asarray(v, dtype=np.float64) for v in cols.values()])


def load_spectrum_named(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """:func:`load_spectrum`, but WITHOUT throwing the column names away.

    ``read_dat`` / ``read_txt`` both return ``columns`` as a dict keyed by the
    real channel name — ``"Bias calc (V)"``, ``"Current (A)"``,
    ``"LIX 1 omega (A)"`` — and ``_stack_columns`` immediately reduces it to
    ``.values()``. Every caller downstream then has to invent placeholder names,
    and one of them (``plot_spectrum``) went on to pick its y-axis label by
    pattern-matching the placeholders it had just invented, so its "is this a
    lock-in dI/dV?" test could never fire: EVERY spectrum figure was labelled
    ``I (A)``, including pure dI/dV traces (found end-to-end 2026-07-28).

    Returns ``(array, names)``. ``names`` is empty for formats that genuinely
    carry no channel names (.npy / .npz / .3ds) — empty means "unknown", and a
    caller must not fill it with guesses that later read as data.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == _DAT_EXT:
        d = read_dat(str(p))
        return (_stack_columns(d, p.name).astype(np.float64),
                [str(k) for k in (d.get("columns") or {})])
    if ext in _TEXT_EXT:
        d = read_txt(str(p))
        cols = d.get("columns") or {}
        if cols:
            return (_stack_columns(d, p.name).astype(np.float64),
                    [str(k) for k in cols])
    return load_spectrum(p), []


def load_image_2d(path: str | Path, *, prefer_channel: str | None = None) -> np.ndarray:
    """Load a 2-D topography image as float64 from any supported format.

    For ``.sxm`` the topography channel (forward) is auto-selected (see
    :func:`_pick_image_channel`); pass *prefer_channel* (a name substring, e.g.
    ``"Z"``) to override. For ``.dat`` the numeric columns are stacked into a 2-D
    array. Raises ``FileNotFoundError`` / ``ValueError``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image file not found: {p}")
    ext = p.suffix.lower()

    if ext == ".npy":
        arr = np.load(p)
    elif ext == ".npz":
        with np.load(p) as z:
            key = prefer_channel if (prefer_channel and prefer_channel in z) else list(z.keys())[0]
            arr = np.asarray(z[key])
    elif ext == _SXM_EXT:
        arr = _channels_to_image(read_sxm(str(p)), p, prefer_channel)
    elif ext == _SM4_EXT:
        arr = _channels_to_image(read_sm4(str(p)), p, prefer_channel)
    elif ext == _DAT_EXT:
        arr = _stack_columns(read_dat(str(p)), p.name)
    elif ext in _TEXT_EXT:
        arr = read_txt(str(p))["matrix"]
    else:
        arr = np.load(p)  # last resort — let np.load raise a clear error

    arr = np.asarray(arr, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.size == 0:
        raise ValueError(f"{p.name}: parsed to an empty array")
    if arr.ndim > 2:
        # Multi-channel / stacked → flatten trailing axes so a 2-D filter still has
        # something sane to operate on.
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def load_spectrum(path: str | Path, *, prefer_channel: str | None = None) -> np.ndarray:
    """Load a spectrum (1-D / 2-D) or grid cube (3-D from ``.3ds``) as float64.

    * ``.npy`` / ``.npz`` — returned as stored.
    * ``.dat`` — Nanonis point spectra; numeric columns stacked → (n_points, n_cols)
      (column 0 is usually the bias/sweep axis).
    * ``.txt`` / ``.csv`` / ``.asc`` — delimited numeric columns.
    * ``.3ds`` — Nanonis grid spectroscopy; the (ny, nx, n_points) cube is returned
      (use :func:`grid_to_spectra` for a (ny*nx, n_points) stack).
    Raises ``FileNotFoundError`` / ``ValueError``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"spectrum file not found: {p}")
    ext = p.suffix.lower()

    if ext == ".npy":
        return np.load(p).astype(np.float64)
    if ext == ".npz":
        with np.load(p) as z:
            key = prefer_channel if (prefer_channel and prefer_channel in z) else list(z.keys())[0]
            return np.asarray(z[key], dtype=np.float64)
    if ext == _DAT_EXT:
        return _stack_columns(read_dat(str(p)), p.name).astype(np.float64)
    if ext in _TEXT_EXT:
        arr = np.asarray(read_txt(str(p))["matrix"], dtype=np.float64)
        if arr.size == 0:
            raise ValueError(f"{p.name}: no numeric data parsed")
        return arr
    if ext == _3DS_EXT:
        cube = np.asarray(read_3ds(str(p)).get("grid"), dtype=np.float64)
        if cube.size == 0:
            raise ValueError(f"{p.name}: .3ds file parsed to an empty grid")
        return cube
    return np.load(p).astype(np.float64)  # last resort


def grid_to_spectra(cube: np.ndarray) -> np.ndarray:
    """Flatten a (ny, nx, n_points) .3ds grid cube into a (ny*nx, n_points) stack of
    spectra (one row per pixel) for unmixing / clustering skills."""
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim == 3:
        ny, nx, n = cube.shape
        return cube.reshape(ny * nx, n)
    if cube.ndim == 2:
        return cube
    if cube.ndim == 1:
        return cube.reshape(1, -1)
    raise ValueError(f"cannot flatten array of shape {cube.shape} into spectra")


def sample_grid_column(grid: np.ndarray, x: float, y: float, *, wrap: bool = False) -> np.ndarray:
    """Bilinearly sample a .3ds grid cube ``(ny, nx, n_points)`` at fractional pixel
    coordinates (``x`` along nx, ``y`` along ny) → the interpolated spectrum
    ``(n_points,)``.

    Lets the data viewer / agents pull a spectrum at any lateral point instead of
    snapping to the nearest pixel. Ported from HamsterPark/metalated-gamma-graphyne-AFM
    ``make_sampler`` (PBC bilinear column sampling), adapted to read_3ds's
    ``(ny, nx, n_points)`` layout. ``wrap=False`` (default) clamps to the grid edge —
    a finite STM grid is not periodic; ``wrap=True`` restores the original PBC behaviour.
    """
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 3:
        raise ValueError(f"expected a 3-D (ny, nx, n_points) cube, got shape {g.shape}")
    ny, nx, _ = g.shape
    if nx < 1 or ny < 1:
        raise ValueError(f"degenerate grid shape {g.shape}")
    if wrap:
        fx, fy = float(x), float(y)
        x0 = int(np.floor(fx)) % nx
        y0 = int(np.floor(fy)) % ny
        x1 = (x0 + 1) % nx
        y1 = (y0 + 1) % ny
    else:
        fx = min(max(float(x), 0.0), nx - 1.0)
        fy = min(max(float(y), 0.0), ny - 1.0)
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        x1 = min(x0 + 1, nx - 1)
        y1 = min(y0 + 1, ny - 1)
    wx = fx - np.floor(fx)
    wy = fy - np.floor(fy)
    c00 = g[y0, x0]
    c01 = g[y0, x1]
    c10 = g[y1, x0]
    c11 = g[y1, x1]
    return ((1 - wx) * (1 - wy) * c00 + wx * (1 - wy) * c01
            + (1 - wx) * wy * c10 + wx * wy * c11)


# Extensions advertised in skill parameter descriptions / the data viewer.
SUPPORTED_IMAGE_EXT = (".npy", ".npz", ".sxm", ".sm4", ".txt", ".csv", ".asc", ".dat")
SUPPORTED_SPECTRUM_EXT = (".npy", ".npz", ".dat", ".txt", ".csv", ".asc", ".3ds")

__all__ = [
    "load_image_2d",
    "load_spectrum",
    "load_spectrum_named",
    "grid_to_spectra",
    "sample_grid_column",
    "SUPPORTED_IMAGE_EXT",
    "SUPPORTED_SPECTRUM_EXT",
]
