"""VIGIL v0.4 mask codec — uint16 encode/decode for STM training labels.

Bit layout (uint16, little-endian semantics):
    bits 0-7   surface MaskClass
                0..14    Block 0 (atomic-scale, 5–50 nm scans)
                15       reserved, MUST be zero
                16..26   Block 1 (mesoscale, 50–500 nm scans)
    bit  8     MORPH_ARTIFACT      image-level (Head B in T1)
    bit  9     STABILITY_ARTIFACT  spatial   (Head C L2 in T1)
    bit 10     TIP_CONTAMINATION   image-level (Head B in T1)
    bit 11     TRANSITION_ZONE     spatial   (Head C L2 in T1)
    bits 12-15 reserved, MUST be zero

This file is a faithful port of the contract in
``MAST-reference/compass_artifact_wf-7560c2c7…_text_markdown.md`` §1. MAST
imports the codec when consuming a VIGIL-trained checkpoint or training
shard; we do NOT generate v0.4 shards in MAST.

The contract is the **only** way to round-trip surface labels and TipFlag
bits without one clobbering the other. ``write_surface()`` enforces that
re-labelling the surface (low byte) never touches TipFlag bits (high byte).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# ─────────────────────────────────────────────────────────────────────
# Bit masks (kept as np.uint16 so all bitops stay in uint16 space)
# ─────────────────────────────────────────────────────────────────────

SURFACE_MASK = np.uint16(0x00FF)
TIPFLAG_MASK = np.uint16(0xFF00)
RESERVED_MASK = np.uint16(0xF000)
RESERVED_SURFACE_CODE = np.uint8(15)

MORPH_ARTIFACT = np.uint16(1 << 8)
STABILITY_ARTIFACT = np.uint16(1 << 9)
TIP_CONTAMINATION = np.uint16(1 << 10)
TRANSITION_ZONE = np.uint16(1 << 11)

NUM_SURFACE_CLASSES = 27  # 0..14 (15 vals) + 16..26 (11 vals) — code 15 reserved


@dataclass(frozen=True)
class DecodedMask:
    """View over a uint16 mask split into surface + 4 TipFlag bool layers.

    Each layer is the same (H, W) shape as the input mask. Layers do not
    share memory with the input — they are the result of bitops + cast.
    """

    surface: npt.NDArray[np.uint8]                  # values in {0..14, 16..26}
    morph_artifact: npt.NDArray[np.bool_]           # bit 8
    stability_artifact: npt.NDArray[np.bool_]       # bit 9
    tip_contamination: npt.NDArray[np.bool_]        # bit 10
    transition_zone: npt.NDArray[np.bool_]          # bit 11


def decode(mask_u16: npt.NDArray[np.uint16]) -> DecodedMask:
    """Split a v0.4 uint16 mask into surface + 4 TipFlag bool layers.

    Args:
        mask_u16: numpy uint16 array of any shape.

    Raises:
        AssertionError: if dtype is not uint16, or any reserved bit (12-15)
                        is set, or any pixel encodes the reserved surface
                        code 15.

    Returns:
        DecodedMask — layers do not alias the input.
    """
    assert mask_u16.dtype == np.uint16, (
        f"mask must be uint16, got {mask_u16.dtype}. v0.4 contract violation; "
        "check the loader pipeline (HDF5 attr / DataLoader collate)."
    )
    if mask_u16.size:
        assert int((mask_u16 & RESERVED_MASK).max()) == 0, (
            "reserved bits 12-15 must be zero (v0.4 contract)"
        )
    surface = (mask_u16 & SURFACE_MASK).astype(np.uint8)
    if surface.size:
        assert int((surface == RESERVED_SURFACE_CODE).sum()) == 0, (
            f"surface code {RESERVED_SURFACE_CODE} is reserved — v0.4 contract"
        )
    return DecodedMask(
        surface=surface,
        morph_artifact=(mask_u16 & MORPH_ARTIFACT).astype(bool),
        stability_artifact=(mask_u16 & STABILITY_ARTIFACT).astype(bool),
        tip_contamination=(mask_u16 & TIP_CONTAMINATION).astype(bool),
        transition_zone=(mask_u16 & TRANSITION_ZONE).astype(bool),
    )


def write_surface(
    mask_u16: npt.NDArray[np.uint16],
    surface: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint16]:
    """Write surface codes to the low byte WITHOUT touching the high byte.

    This is the explicit contract: re-labelling surfaces must never clobber
    the TipFlag bits. Returns a new array; the input is not modified.

    Args:
        mask_u16: existing v0.4 uint16 mask.
        surface:  uint8 surface codes — values in {0..14, 16..26}.

    Returns:
        new uint16 mask with the same TipFlag bits + the new surface codes.
    """
    assert mask_u16.dtype == np.uint16, f"mask must be uint16, got {mask_u16.dtype}"
    assert surface.dtype == np.uint8, f"surface must be uint8, got {surface.dtype}"
    assert mask_u16.shape == surface.shape, (
        f"shape mismatch: mask {mask_u16.shape} vs surface {surface.shape}"
    )
    if surface.size:
        assert int((surface == RESERVED_SURFACE_CODE).sum()) == 0, (
            f"surface code {RESERVED_SURFACE_CODE} is reserved"
        )
        valid = (surface < RESERVED_SURFACE_CODE) | (
            (surface > RESERVED_SURFACE_CODE) & (surface < NUM_SURFACE_CLASSES + 1)
        )
        assert bool(valid.all()), (
            "surface values must lie in {0..14, 16..26}"
        )
    high = mask_u16 & TIPFLAG_MASK
    return (high | surface.astype(np.uint16))


def set_tipflag(
    mask_u16: npt.NDArray[np.uint16],
    *,
    bit: int,
    where: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint16]:
    """Set or clear a TipFlag bit on selected pixels.

    Args:
        mask_u16: v0.4 uint16 mask.
        bit:      the TipFlag bit number, must be in {8, 9, 10, 11}.
        where:    bool array of mask shape — True pixels get the bit set,
                  False pixels get the bit cleared.

    Returns:
        new uint16 mask. Surface (low byte) and other TipFlag bits unchanged.
    """
    assert mask_u16.dtype == np.uint16
    assert bit in (8, 9, 10, 11), f"TipFlag bit must be in 8..11, got {bit}"
    assert where.dtype == np.bool_
    assert mask_u16.shape == where.shape
    flag = np.uint16(1 << bit)
    cleared = mask_u16 & np.uint16(~int(flag) & 0xFFFF)
    return cleared | (where.astype(np.uint16) * flag)


__all__ = [
    "SURFACE_MASK",
    "TIPFLAG_MASK",
    "RESERVED_MASK",
    "RESERVED_SURFACE_CODE",
    "MORPH_ARTIFACT",
    "STABILITY_ARTIFACT",
    "TIP_CONTAMINATION",
    "TRANSITION_ZONE",
    "NUM_SURFACE_CLASSES",
    "DecodedMask",
    "decode",
    "write_surface",
    "set_tipflag",
]
