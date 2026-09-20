"""Crockford Base32 ULID generator.

128-bit identifier = 48-bit milliseconds-since-epoch + 80-bit randomness.
Encoded as 26 Crockford Base32 chars (sort-friendly, case-insensitive,
no ambiguous I/L/O/U).

Reference: https://github.com/ulid/spec
"""
from __future__ import annotations

import os
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford Base32
_DECODE_TABLE = {c: i for i, c in enumerate(_ALPHABET)}
# Crockford aliases (ambiguous chars map to safe equivalents)
for src, dst in (("I", "1"), ("L", "1"), ("O", "0"), ("U", "V")):
    _DECODE_TABLE[src] = _DECODE_TABLE[dst]
    _DECODE_TABLE[src.lower()] = _DECODE_TABLE[dst]


def _encode_int(n: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_ALPHABET[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def _decode_str(s: str) -> int:
    n = 0
    for c in s.upper():
        if c not in _DECODE_TABLE:
            raise ValueError(f"Invalid ULID char: {c!r}")
        n = (n << 5) | _DECODE_TABLE[c]
    return n


def ulid_now() -> str:
    """Generate a new ULID using current time + 80 bits of OS randomness."""
    ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_bytes = os.urandom(10)
    rand_int = int.from_bytes(rand_bytes, "big")
    return _encode_int(ms, 10) + _encode_int(rand_int, 16)


def ulid_from_ms(ms: int) -> str:
    """Generate a ULID with explicit timestamp (for replay / time travel)."""
    ms = ms & ((1 << 48) - 1)
    rand_bytes = os.urandom(10)
    rand_int = int.from_bytes(rand_bytes, "big")
    return _encode_int(ms, 10) + _encode_int(rand_int, 16)


def ulid_timestamp_ms(ulid: str) -> int:
    """Recover the millisecond timestamp from a ULID."""
    if len(ulid) != 26:
        raise ValueError(f"ULID must be 26 chars, got {len(ulid)}: {ulid!r}")
    return _decode_str(ulid[:10])


def is_valid_ulid(s: str) -> bool:
    """True iff *s* is a syntactically valid 26-char Crockford Base32 ULID."""
    if not isinstance(s, str) or len(s) != 26:
        return False
    try:
        _decode_str(s)
        return True
    except ValueError:
        return False
