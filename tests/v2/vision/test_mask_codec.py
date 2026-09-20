"""VIGIL v0.4 mask codec tests.

Verifies the contract from
``MAST-reference/compass_artifact_wf-7560c2c7…_text_markdown.md`` §1:

  * Round-trip decode/write_surface preserves all bits.
  * Wrong dtype triggers AssertionError at the boundary.
  * Reserved bits (12-15) and reserved surface code (15) are rejected.
  * write_surface preserves high byte (TipFlag bits).
  * set_tipflag flips a single bit without touching others.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
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

from mast.vision import mask_codec as mc  # noqa: E402


def _make_mask(h: int = 4, w: int = 4) -> np.ndarray:
    """Build a small mask with mixed surface codes + a few flag bits."""
    surf = np.array([
        [0,  1,  2,  3 ],
        [4, 16, 20, 14],
        [0,  0,  0,  0 ],
        [0,  0,  0,  0 ],
    ], dtype=np.uint8)
    mask = surf.astype(np.uint16)
    # Set bit 9 (stability) on (1, 2)
    mask[1, 2] |= mc.STABILITY_ARTIFACT
    # Set bit 11 (transition) on (1, 1)
    mask[1, 1] |= mc.TRANSITION_ZONE
    return mask


# ── decode ────────────────────────────────────────────────────────────


def test_decode_round_trip():
    mask = _make_mask()
    decoded = mc.decode(mask)
    assert decoded.surface[0, 0] == 0
    assert decoded.surface[1, 2] == 20
    # bit-9 only set at (1, 2)
    assert bool(decoded.stability_artifact[1, 2])
    assert not bool(decoded.stability_artifact[0, 0])
    # bit-11 only set at (1, 1)
    assert bool(decoded.transition_zone[1, 1])
    assert not bool(decoded.transition_zone[1, 2])


def test_decode_rejects_uint8():
    bad = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(AssertionError, match="must be uint16"):
        mc.decode(bad)


def test_decode_rejects_int32():
    bad = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(AssertionError, match="must be uint16"):
        mc.decode(bad)


def test_decode_rejects_reserved_high_bits():
    bad = np.zeros((2, 2), dtype=np.uint16)
    bad[0, 0] = np.uint16(1 << 12)  # bit 12 reserved
    with pytest.raises(AssertionError, match="reserved bits"):
        mc.decode(bad)


def test_decode_rejects_reserved_surface_code_15():
    bad = np.zeros((2, 2), dtype=np.uint16)
    bad[0, 0] = np.uint16(15)
    with pytest.raises(AssertionError, match="reserved"):
        mc.decode(bad)


def test_decode_empty_mask_ok():
    empty = np.zeros((0,), dtype=np.uint16)
    decoded = mc.decode(empty)
    assert decoded.surface.shape == (0,)
    assert decoded.stability_artifact.shape == (0,)


# ── write_surface ─────────────────────────────────────────────────────


def test_write_surface_preserves_tipflag_bits():
    mask = _make_mask()
    new_surface = np.full(mask.shape, 7, dtype=np.uint8)
    new_surface[0, 0] = 12
    new_mask = mc.write_surface(mask, new_surface)
    # Surface byte updated
    assert new_mask[0, 0] & 0x00FF == 12
    assert new_mask[1, 1] & 0x00FF == 7
    # TipFlag high byte unchanged
    assert (new_mask[1, 1] & 0xFF00) == (mask[1, 1] & 0xFF00)
    assert (new_mask[1, 2] & 0xFF00) == (mask[1, 2] & 0xFF00)


def test_write_surface_does_not_mutate_input():
    mask = _make_mask()
    snapshot = mask.copy()
    mc.write_surface(mask, np.full(mask.shape, 1, dtype=np.uint8))
    np.testing.assert_array_equal(mask, snapshot)


def test_write_surface_rejects_reserved_15():
    mask = _make_mask()
    bad = np.full(mask.shape, 15, dtype=np.uint8)
    with pytest.raises(AssertionError, match="reserved"):
        mc.write_surface(mask, bad)


def test_write_surface_rejects_dtype_mismatch():
    mask = _make_mask()
    bad = np.zeros(mask.shape, dtype=np.int32)
    with pytest.raises(AssertionError, match="surface must be uint8"):
        mc.write_surface(mask, bad)


def test_write_surface_rejects_shape_mismatch():
    mask = _make_mask()
    bad = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(AssertionError, match="shape mismatch"):
        mc.write_surface(mask, bad)


def test_write_surface_rejects_out_of_range():
    mask = _make_mask()
    bad = np.full(mask.shape, 200, dtype=np.uint8)
    with pytest.raises(AssertionError, match="0..14"):
        mc.write_surface(mask, bad)


# ── set_tipflag ───────────────────────────────────────────────────────


def test_set_tipflag_flips_bit_without_touching_others():
    mask = _make_mask()
    where = np.zeros(mask.shape, dtype=bool)
    where[0, 0] = True
    out = mc.set_tipflag(mask, bit=10, where=where)
    # bit 10 set at (0,0)
    assert (out[0, 0] & mc.TIP_CONTAMINATION) != 0
    # bit 9 still only at (1,2)
    decoded = mc.decode(out)
    assert bool(decoded.stability_artifact[1, 2])
    assert not bool(decoded.stability_artifact[0, 0])
    # surface byte unchanged
    assert (out & mc.SURFACE_MASK).tolist() == (mask & mc.SURFACE_MASK).tolist()


def test_set_tipflag_clears_off_pixels():
    mask = _make_mask()
    where = np.zeros(mask.shape, dtype=bool)
    # All False — should clear bit 9 everywhere
    out = mc.set_tipflag(mask, bit=9, where=where)
    assert int((out & mc.STABILITY_ARTIFACT).sum()) == 0


def test_set_tipflag_rejects_invalid_bit():
    mask = _make_mask()
    where = np.zeros(mask.shape, dtype=bool)
    with pytest.raises(AssertionError, match="must be in 8"):
        mc.set_tipflag(mask, bit=7, where=where)
    with pytest.raises(AssertionError, match="must be in 8"):
        mc.set_tipflag(mask, bit=12, where=where)
