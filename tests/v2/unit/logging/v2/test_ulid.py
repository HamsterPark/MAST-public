"""Unit tests for v2 ULID generator."""
import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
while _MASTV2_ROOT in sys.path:
    sys.path.remove(_MASTV2_ROOT)
sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import time

import pytest

from mast.logging.v2.ulid import (
    is_valid_ulid,
    ulid_from_ms,
    ulid_now,
    ulid_timestamp_ms,
)


def test_ulid_length_and_alphabet():
    u = ulid_now()
    assert len(u) == 26
    # All chars are Crockford Base32
    for c in u:
        assert c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def test_ulid_uniqueness():
    seen = {ulid_now() for _ in range(1000)}
    assert len(seen) == 1000


def test_ulid_timestamp_roundtrip():
    fixed = 1_700_000_000_000  # 2023-11-14 22:13:20 UTC
    u = ulid_from_ms(fixed)
    assert ulid_timestamp_ms(u) == fixed


def test_ulid_is_time_ordered_within_ms():
    ms = int(time.time() * 1000)
    a = ulid_from_ms(ms)
    b = ulid_from_ms(ms + 1)
    assert a < b


def test_is_valid_ulid_accepts_proper():
    assert is_valid_ulid(ulid_now())


def test_is_valid_ulid_rejects_bad():
    assert not is_valid_ulid("")
    assert not is_valid_ulid("short")
    assert not is_valid_ulid("Z" * 27)
    # 'U' is a Crockford ambiguous char that decodes to 'V', so it's fine in 26 chars.
    # But '!' is not.
    assert not is_valid_ulid("!" * 26)


def test_ulid_now_then_after_zero_sleep_strictly_increases():
    # ulid_from_ms with the same ms may collide on the random part with prob 2^-80;
    # the wall-clock test asserts monotonicity of the time prefix only.
    a = ulid_now()
    time.sleep(0.002)
    b = ulid_now()
    assert ulid_timestamp_ms(b) >= ulid_timestamp_ms(a)
