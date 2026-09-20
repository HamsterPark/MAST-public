"""read_sm4 (RHK .sm4) — ported from HamsterPark/Nanonis-RHK-SPM-PyTools.

No real .sm4 fixture exists in-repo, so we synthesise a minimal valid SM4 byte
stream (file header → page-index header → page-index array → one forward
topographic image page) that exercises the object-table + page-tree walk, then
assert read_sm4 + the shared loader + load_scan_file dispatch round-trip it.
"""
from __future__ import annotations

import struct

import numpy as np


# Byte offsets of each section in the synthetic file (see read_sm4 for the layout).
_HEADER_SIZE = 56  # so seek(header_size + 2) lands on the file object list @58
_PIH = 70          # page index header
_PIA = 98          # page index array
_HDR = 154         # page header
_DATA = 330        # page data (page header is 176 bytes: 154 + 176 = 330)


def _make_sm4(path, xsize=4, ysize=3, values=None):
    """Write a minimal one-forward-topographic-page SM4 file (little-endian)."""
    n = xsize * ysize
    if values is None:
        values = list(range(n))
    buf = bytearray(_DATA + n * 4)

    # ── file header ──
    struct.pack_into("<H", buf, 0, _HEADER_SIZE)   # header size
    struct.pack_into("<I", buf, 38, 1)             # total page count
    struct.pack_into("<I", buf, 42, 1)             # object list count
    struct.pack_into("<I", buf, 46, 12)            # object field size
    # file object list @58: (PAGE_INDEX_HEADER=1, offset=_PIH, size)
    struct.pack_into("<III", buf, 58, 1, _PIH, 28)

    # ── page index header @70 ──
    struct.pack_into("<I", buf, 70, 1)             # page count
    struct.pack_into("<I", buf, 74, 1)             # page index obj count
    # page index objects @86: (PAGE_INDEX_ARRAY=2, offset=_PIA, size)
    struct.pack_into("<III", buf, 86, 2, _PIA, 56)

    # ── page index array @98: one page entry ──
    struct.pack_into("<I", buf, 114, 0)            # page_data_type = IMAGE(0)
    struct.pack_into("<I", buf, 122, 2)            # page object count = 2
    # page objects @130: (PAGE_HEADER=3, _HDR, 176), (PAGE_DATA=4, _DATA, n*4)
    struct.pack_into("<III", buf, 130, 3, _HDR, 176)
    struct.pack_into("<III", buf, 142, 4, _DATA, n * 4)

    # ── page header @154 (176 bytes) ──
    o = _HDR
    o += 4                                          # field size + string count
    struct.pack_into("<I", buf, o, 1); o += 4       # page_type TOPOGRAPHIC
    o += 4                                           # data sub source
    struct.pack_into("<I", buf, o, 1); o += 4       # line_type = 1 (float)
    o += 8                                           # x/y corner
    struct.pack_into("<I", buf, o, xsize); o += 4
    struct.pack_into("<I", buf, o, ysize); o += 4
    o += 4                                           # image type
    struct.pack_into("<I", buf, o, 0); o += 4       # scan_type FORWARD
    o += 4                                           # group id
    struct.pack_into("<I", buf, o, n * 4); o += 4   # page_data_size
    o += 8                                           # min/max z
    struct.pack_into("<f", buf, o, 1e-9); o += 4    # xscale (>0 → no x-flip)
    struct.pack_into("<f", buf, o, -1e-9); o += 4   # yscale (≤0 → no y-flip)
    struct.pack_into("<f", buf, o, 1.0); o += 4     # zscale
    o += 12                                          # xyscale/xoffset/yoffset
    struct.pack_into("<f", buf, o, 0.0); o += 4     # zoffset
    o += 12                                          # period/bias/current
    o += 16                                          # color count + grid xy + obj count
    o += 1 + 3 + 60                                  # data flag + reserved
    assert o == _DATA, f"page header ended at {o}, expected {_DATA}"

    # ── page data @330: n float32 ──
    for i, v in enumerate(values):
        struct.pack_into("<f", buf, _DATA + i * 4, float(v))
    path.write_bytes(bytes(buf))


def test_read_sm4_roundtrip(tmp_path):
    from mast.io.nanonis_files import read_sm4
    p = tmp_path / "scan.sm4"
    _make_sm4(p, 4, 3, list(range(12)))
    d = read_sm4(str(p))
    assert "Z" in d["channels"]
    fwd = d["channels"]["Z"]["forward"]
    assert fwd.shape == (4, 3)
    np.testing.assert_allclose(fwd, np.arange(12).reshape(4, 3))
    assert d["header"]["scan_pixels"] == [4, 3]
    np.testing.assert_allclose(d["header"]["pixel_size_m"], [1e-9, 1e-9])


def test_load_scan_file_dispatches_sm4(tmp_path):
    from mast.io.nanonis_files import load_scan_file
    p = tmp_path / "scan.sm4"
    _make_sm4(p)
    d = load_scan_file(str(p))
    assert "Z" in d["channels"]


def test_load_image_2d_reads_sm4(tmp_path):
    from mast.data import load_image_2d
    p = tmp_path / "scan.sm4"
    _make_sm4(p, 4, 3, list(range(12)))
    arr = load_image_2d(p)
    assert arr.shape == (4, 3)
    np.testing.assert_allclose(arr, np.arange(12).reshape(4, 3))


def test_sm4_in_supported_image_ext():
    from mast.data.loaders import SUPPORTED_IMAGE_EXT
    assert ".sm4" in SUPPORTED_IMAGE_EXT
