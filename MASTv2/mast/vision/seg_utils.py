"""Multi-class run-length codec for ``SegmentationResult.mask_rle`` (Phase 9).

Phase 1-8 shipped a *binary* RLE (terrace vs not) good enough for the legacy
AttentionUNet. The M12 Head C-L1 produces a genuine 4-class per-pixel map
(0=terrace, 1=step, 2=defect, 3=contamination), so the mask codec is unified
here to a **multi-class** value-run format. The binary case (legacy / L0) is
just the 2-value (0/1) special case, so every backend can share one codec and
one decoder — and ``SegmentationResult.classes`` is no longer a lie.

Format
------
A flat little-endian ``uint32`` array of alternating ``(value, count)`` pairs::

    [v0, c0, v1, c1, …]   → c0 pixels of class v0, then c1 of class v1, …

read in row-major (C) order over the (H, W) mask. An empty mask → ``b""``.
Supports class ids 0‥2³²−1 (we use 0‥27); ``decode_rle`` returns ``uint8``.
"""
from __future__ import annotations

import numpy as np


def encode_rle(mask: np.ndarray) -> bytes:
    """Encode a (H, W) integer class map to the value-run byte format."""
    flat = np.ascontiguousarray(mask).astype(np.uint32).ravel(order="C")
    if flat.size == 0:
        return b""
    change = np.nonzero(np.diff(flat))[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [flat.size]))
    values = flat[starts]
    counts = (ends - starts).astype(np.uint32)
    pairs = np.empty(values.size * 2, dtype=np.uint32)
    pairs[0::2] = values
    pairs[1::2] = counts
    return pairs.tobytes()


def decode_rle(rle: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`encode_rle` → a (H, W) ``uint8`` class map.

    Defensive: an empty/short stream yields zeros (terrace), an over-long one
    is trimmed, so a corrupt mask degrades to "all terrace" rather than raising
    inside an inference path.
    """
    h, w = int(shape[0]), int(shape[1])
    n = h * w
    if not rle:
        return np.zeros((h, w), dtype=np.uint8)
    pairs = np.frombuffer(rle, dtype=np.uint32)
    values = pairs[0::2]
    counts = pairs[1::2]
    m = min(values.size, counts.size)
    flat = np.repeat(values[:m].astype(np.uint8), counts[:m])
    if flat.size < n:
        flat = np.concatenate([flat, np.zeros(n - flat.size, dtype=np.uint8)])
    elif flat.size > n:
        flat = flat[:n]
    return flat.reshape(h, w)


__all__ = ["encode_rle", "decode_rle"]
