"""«扫描还在跑吗» must be answerable without guessing from a counter.

. An abort had fired; the agent called ``get_scan_progress``,
saw a NEW seqno and "line 2/512", and concluded:

    扫描实际已在运行（新 seqno，第 2/512 行）——中止只打断了等待调用，硬件扫描继续。
    改用它再等待一次。

It then waited twice more on a scan that had already stopped, and the operator
watched the run burn machine time on nothing. The inference was wrong for a
reason the tool invited:

* ``seqno`` is the vision buffer's GLOBAL sequence — every event bumps it, so it
  keeps climbing long after a scan ends;
* ``get_latest_progress`` returns the LAST PUBLISHED record, which persists
  forever after the scan stops, so ``line_idx: 2`` was a fossil, not a reading;
* the docstring said "progress is None if no scan is running", which is simply
  not what the buffer does.

What the tool now answers directly: how OLD the record is, and whether the line
number MOVED since the caller's previous look.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_scan_progress_liveness.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import time  # noqa: E402

import pytest  # noqa: E402

from mast.agents._shared.buffer_tools import make_buffer_tools  # noqa: E402
from mast.buffer.schemas import ScanProgress  # noqa: E402


class _Buf:
    """A buffer whose seqno keeps climbing while the scan stands still — the
    exact shape that fooled the agent."""

    def __init__(self):
        self.line = 2
        self.seq = 1000
        self.t_ns = time.monotonic_ns()

    def get_latest_progress(self):
        self.seq += 1                      # a global counter, not scan progress
        return (ScanProgress(seqno=self.seq, scan_id="synthetic_sample-001_",
                             line_idx=self.line, lines_total=512, eta_s=0.0,
                             t_mono_ns=self.t_ns),
                self.seq)


def _tool(buf):
    return next(t for t in make_buffer_tools(buf) if t.name == "get_scan_progress")


# ════════════════════════════════════════════════════════════════════════════
# The question the agent actually had
# ════════════════════════════════════════════════════════════════════════════

def test_a_stalled_scan_is_reported_as_not_advancing():
    buf = _Buf()
    t = _tool(buf)
    first = t.invoke({})
    assert first["advancing"] is None, "第一次调用无从比较,不该假装知道"
    second = t.invoke({})
    assert second["advancing"] is False, "行号没动却没说出来"
    assert "已经停止" in second["note"]
    # And the seqno DID climb — the trap is still present in the data, which is
    # exactly why the verdict has to be computed rather than inferred.
    assert second["seqno"] > first["seqno"]


def test_a_live_scan_is_reported_as_advancing():
    buf = _Buf()
    t = _tool(buf)
    t.invoke({})
    buf.line = 37
    buf.t_ns = time.monotonic_ns()
    out = t.invoke({})
    assert out["advancing"] is True
    assert out["note"] == "", f"扫描在推进却给了警告: {out['note']}"


def test_a_stale_record_is_flagged_by_age():
    """Even on the FIRST call — when there is nothing to compare against — a
    minutes-old record must not read as live."""
    buf = _Buf()
    buf.t_ns = time.monotonic_ns() - 300 * 1_000_000_000
    out = _tool(buf).invoke({})
    assert out["age_s"] is not None and out["age_s"] > 250
    assert "已经停止" in out["note"]


def test_no_progress_ever_published_says_so():
    class _Empty:
        def get_latest_progress(self):
            return None, 0

    out = _tool(_Empty()).invoke({})
    assert out["progress"] is None
    assert out["advancing"] is None
    assert "从未发布" in out["note"]


def test_the_progress_payload_is_still_there():
    """The fix adds signal; it must not take any away."""
    out = _tool(_Buf()).invoke({})
    p = out["progress"]
    assert p["line_idx"] == 2 and p["lines_total"] == 512
    assert p["scan_id"] == "synthetic_sample-001_"


# ════════════════════════════════════════════════════════════════════════════
# The docstring is part of the fix — it is what the model reads
# ════════════════════════════════════════════════════════════════════════════

def test_the_docstring_warns_against_the_seqno_inference():
    doc = _tool(_Buf()).description
    assert "不要用 seqno 判断" in doc, "模型还会再犯同一个推断错误"
    assert "不会消失" in doc, "没说清进度记录会一直停在旧值"
    # The false claim that started it must be gone.
    assert "None if no scan is running" not in doc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
