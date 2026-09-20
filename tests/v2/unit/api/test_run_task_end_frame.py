"""A run that FINISHES must not die on its own last chunk.

— three runs in one night ended with::

    [SUPERVISOR → __end__] …
    运行出错，已停止：IndexError: list index out of range

The work was already done and the route note was already on screen. The crash
was in the bridge's per-chunk bookkeeping:

    fanned = _parse_dispatch_targets(msgs)        # [] for an END, by design
    …
    "text": (f"并行下发 → …" if len(fanned) > 1
             else f"_supervisor → {fanned[0]}")   # ← len 0 lands HERE

Two reasons this survived so long, both pinned below:

* the existing tests run without a live ``_agents_api_state``, so ``st`` is
  None and the whole block is skipped — the crash needs the LIVE task dict;
* the ``except`` around the stream logged ``logger.warning("…: %s", exc)`` with
  no traceback, so the log said exactly what the operator's screen said and
  nothing more.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/api/test_run_task_end_frame.py -q
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

import threading  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import mast.api.routes.orchestrator as O  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# The parser's own contract — [] means END, and that is deliberate
# ════════════════════════════════════════════════════════════════════════════

def test_an_end_note_parses_to_an_empty_target_list():
    """Not None. None means "no dispatch note here"; [] means "ended"."""
    msgs = [AIMessage(content="[SUPERVISOR → __end__] 扫描被中止锁存卡住")]
    assert O._parse_dispatch_targets(msgs) == []


def test_no_note_at_all_parses_to_none():
    assert O._parse_dispatch_targets([AIMessage(content="随便说点什么")]) is None


def test_a_real_dispatch_still_parses():
    msgs = [AIMessage(content="[SUPERVISOR → instrument_control] 需要实时仪器操作")]
    assert O._parse_dispatch_targets(msgs) == ["instrument_control"]


# ════════════════════════════════════════════════════════════════════════════
# End-to-end: the END chunk, with a LIVE task dict (the condition that crashed)
# ════════════════════════════════════════════════════════════════════════════

class _EndOnlyGraph:
    """A graph whose single super-step is the supervisor ending the run."""

    def stream(self, *_a, **_kw):
        yield ((), {"supervisor": {"messages": [
            AIMessage(content="[SUPERVISOR → __end__] 任务已完成，无需继续分发")]}})


def _client_with_live_task_state():
    app = FastAPI()
    app.include_router(O.router, prefix="/api")
    live = types.SimpleNamespace()
    # The live bookkeeping dict. Without it `st` is None and the crashing block
    # is skipped entirely — which is precisely why unit tests never saw this.
    live._agents_api_state = {
        "lock": threading.Lock(),
        # IDLE. The run POPULATES this slot; pre-seeding it active=True is not
        # a real state, and since 2026-07-28 the slot write is also the
        # concurrency claim, so a pre-seeded active task reads as "another run
        # is already streaming" and the request is (correctly) refused.
        "task": None,
        "holds": {},
        "interrupts": {"pending": {}, "resolved": {}, "events": {},
                       "lock": threading.Lock()},
    }
    live._orch_abort = threading.Event()
    live._orchestrator = _EndOnlyGraph()
    app.state.ctx = types.SimpleNamespace(live_app=live)
    return TestClient(app), live


def _frames(body: str) -> list[str]:
    return [ln[len("data: "):] for ln in body.splitlines()
            if ln.startswith("data: ")]


def test_a_finished_run_does_not_report_an_error():
    client, live = _client_with_live_task_state()
    r = client.post("/api/agents/run-task", json={"task": "扫一张图"})
    assert r.status_code == 200
    body = r.text
    assert "IndexError" not in body, "跑完的任务在最后一块上崩了"
    assert "list index out of range" not in body
    # And the terminal frame must not be an error label.
    assert "运行出错" not in body


def test_the_end_is_recorded_as_an_end_not_a_handoff_to_nobody():
    client, live = _client_with_live_task_state()
    client.post("/api/agents/run-task", json={"task": "扫一张图"})
    task = live._agents_api_state["task"]
    ends = [h for h in task["handoffs"] if h.get("kind") == "end"]
    assert ends, f"没有记录结束: {task['handoffs']}"
    assert ends[-1]["targets"] == []
    assert task["active_agents"] == []


def test_a_real_dispatch_is_still_recorded_as_a_handoff():
    """The fix must not swallow the case that actually worked."""
    class _DispatchGraph:
        def stream(self, *_a, **_kw):
            yield ((), {"supervisor": {"messages": [
                AIMessage(content="[SUPERVISOR → instrument_control] 去扫图")]}})

    client, live = _client_with_live_task_state()
    live._orchestrator = _DispatchGraph()
    client.post("/api/agents/run-task", json={"task": "扫一张图"})
    task = live._agents_api_state["task"]
    hs = [h for h in task["handoffs"] if h.get("kind") == "handoff"]
    assert hs and hs[-1]["targets"] == ["instrument_control"]
    assert "_supervisor → instrument_control" in hs[-1]["text"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
