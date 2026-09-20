"""Unit tests for the Hybrid Logical Clock."""
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

from mast.logging.v2.hlc import HLC, HLCClock


def test_now_monotonic():
    c = HLCClock("test")
    prev = c.now()
    for _ in range(100):
        cur = c.now()
        assert cur.encode() > prev.encode()
        prev = cur


def test_encode_parse_roundtrip():
    h = HLC(pt=1_700_000_000_000, counter=42, node="XD")
    s = h.encode()
    assert s == "1700000000000-0042-XD"
    back = HLC.parse(s)
    assert back == h


def test_lexical_eq_causal_within_same_node():
    c = HLCClock("A")
    a = c.now()
    b = c.now()
    assert a.encode() < b.encode()
    assert a < b


def test_update_with_remote_advances_clock():
    a = HLCClock("A")
    b = HLCClock("B")
    ha = a.now()
    hb_local = b.now()
    # When B sees A's HLC it should produce one strictly greater than both.
    hb_new = b.update(ha)
    assert hb_new > hb_local
    assert hb_new > ha


def test_node_id_must_be_non_empty_no_dash():
    with pytest.raises(ValueError):
        HLCClock("")
    with pytest.raises(ValueError):
        HLCClock("bad-name")


def test_max_offset_enforced():
    c = HLCClock("A", max_offset_ms=10)
    far_future = HLC(pt=int(time.time() * 1000) + 60_000, counter=0, node="B")
    with pytest.raises(ValueError):
        c.update(far_future)


def test_counter_increments_on_same_ms(monkeypatch):
    c = HLCClock("A")
    # Force same physical-ms by patching the wall clock; monkeypatch
    # restores the original staticmethod descriptor after the test.
    fixed = int(time.time() * 1000)
    monkeypatch.setattr(HLCClock, "_wall_now_ms", staticmethod(lambda: fixed))
    a = c.now()
    b = c.now()
    d = c.now()
    assert b.pt == a.pt and b.counter == a.counter + 1
    assert d.counter == a.counter + 2


def test_parse_rejects_malformed():
    with pytest.raises(ValueError):
        HLC.parse("not-an-hlc")
    with pytest.raises(ValueError):
        HLC.parse("only-two")
