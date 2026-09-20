"""Unit tests for the unified multi-class segmentation RLE codec (Phase 9)."""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.vision.seg_utils import decode_rle, encode_rle  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("nclass", [1, 2, 4, 28])
def test_round_trip_random(seed, nclass):
    rng = np.random.default_rng(seed)
    mask = rng.integers(0, nclass, size=(23, 17), dtype=np.uint8)
    out = decode_rle(encode_rle(mask), mask.shape)
    assert np.array_equal(out, mask)


def test_empty_mask_encodes_to_empty_bytes():
    assert encode_rle(np.zeros((0, 0), dtype=np.uint8)) == b""


def test_decode_empty_is_all_zero():
    out = decode_rle(b"", (5, 6))
    assert out.shape == (5, 6)
    assert int(out.sum()) == 0


def test_all_same_class_single_run():
    mask = np.full((8, 8), 3, dtype=np.uint8)
    rle = encode_rle(mask)
    # one (value,count) pair → 2 uint32 = 8 bytes
    assert len(rle) == 8
    assert np.array_equal(decode_rle(rle, mask.shape), mask)


def test_decode_truncated_pads_with_terrace():
    """A short/corrupt stream degrades to terrace, never raises."""
    mask = np.full((4, 4), 2, dtype=np.uint8)
    rle = encode_rle(mask)
    out = decode_rle(rle, (4, 8))  # ask for more pixels than encoded
    assert out.shape == (4, 8)
    assert int((out == 0).sum()) == 16  # the extra half is terrace-filled


def test_decode_overlong_is_trimmed():
    mask = np.full((4, 4), 1, dtype=np.uint8)
    rle = encode_rle(mask)
    out = decode_rle(rle, (2, 4))  # fewer pixels than encoded → trim
    assert out.shape == (2, 4)
    assert int((out == 1).sum()) == 8


def test_class_histogram_preserved():
    rng = np.random.default_rng(3)
    mask = rng.integers(0, 4, size=(40, 40), dtype=np.uint8)
    out = decode_rle(encode_rle(mask), mask.shape)
    for c in range(4):
        assert int((out == c).sum()) == int((mask == c).sum())


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
