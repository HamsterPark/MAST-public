"""远程控制电脑的网络变化会导致对话卡住。

Backend half of the fix (``mast/api/sse.py``). Two independent failure modes
kept a Tailscale-remote operator staring at a dead conversation:

  * a stream that emits nothing for minutes looks dead to every idle-TCP reaper
    on the path (NAT rebind, WireGuard keepalive window, Wi-Fi roam, sleep) —
    only the HITL approval wait had a beat, everything else was silent;
  * a reverse proxy in front of MAST buffers an unmarked event-stream, which
    turns a live run into a blank screen with no bytes at all.

These pin the wrapper's behaviour AND the two properties an operator actually
depends on: the beat must be invisible to the frame parser, and the producer's
cleanup must still run when the client vanishes.
"""

from __future__ import annotations

import json
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.sse import BEAT, SSE_HEADERS, with_heartbeat


# ── the wrapper itself ────────────────────────────────────────────────────


def test_beats_fill_a_silent_gap() -> None:
    """A producer that stalls gets keep-alives; the payload frames still arrive
    in order, unmodified."""

    def slow():
        yield "data: {}\n\n"
        time.sleep(0.35)
        yield "data: {\"kind\": \"done\"}\n\n"

    out = list(with_heartbeat(slow(), interval=0.5, label="t"))
    # interval is floored at 0.5 s inside the wrapper, so 0.35 s of silence must
    # NOT produce a beat — the guard against beating a healthy fast stream.
    assert out == ["data: {}\n\n", "data: {\"kind\": \"done\"}\n\n"]

    def slower():
        yield "data: 1\n\n"
        time.sleep(1.6)
        yield "data: 2\n\n"

    out2 = list(with_heartbeat(slower(), interval=0.5, label="t"))
    assert out2[0] == "data: 1\n\n"
    assert out2[-1] == "data: 2\n\n"
    assert out2.count(BEAT) >= 2, out2


def test_beat_is_an_sse_comment_the_parser_ignores() -> None:
    """The keep-alive must be invisible: no ``data:`` line, so both EventSource
    and our hand-rolled readers skip it instead of rendering a junk bubble."""
    assert BEAT.startswith(":")
    assert "data:" not in BEAT
    assert BEAT.endswith("\n\n")
    payload = [ln for ln in BEAT.split("\n") if ln.startswith("data:")]
    assert payload == []


def test_frames_pass_through_untouched_and_in_order() -> None:
    src = [f"data: {i}\n\n" for i in range(50)]
    assert [f for f in with_heartbeat(iter(src), interval=5.0) if f != BEAT] == src


def test_producer_exception_is_re_raised() -> None:
    """Wrapping must not swallow a failure — same observable behaviour as
    iterating the generator directly."""

    def boom():
        yield "data: 1\n\n"
        raise RuntimeError("kaboom")

    got = []
    try:
        for f in with_heartbeat(boom(), interval=5.0):
            got.append(f)
    except RuntimeError as exc:
        assert "kaboom" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("exception was swallowed")
    assert got == ["data: 1\n\n"]


def test_client_disconnect_still_runs_producer_cleanup() -> None:
    """THE disconnect contract (see mast/api/sse.py).

    When the consumer goes away mid-stream the producer is drained to its
    natural end rather than having GeneratorExit thrown into it, so its finally
    — transcript terminal row, training-trajectory close, task-slot release —
    still runs, and a network blip no longer aborts a running experiment at the
    next super-step boundary."""
    cleaned = threading.Event()
    produced: list[int] = []

    def producer():
        try:
            for i in range(6):
                produced.append(i)
                yield f"data: {i}\n\n"
                time.sleep(0.02)
        finally:
            cleaned.set()

    gen = with_heartbeat(producer(), interval=5.0, label="disconnect")
    assert next(gen) == "data: 0\n\n"
    gen.close()  # what Starlette does when the client drops

    assert cleaned.wait(timeout=5.0), "producer finally never ran after disconnect"
    assert produced == list(range(6)), "producer was cut short instead of drained"


# ── the endpoints ─────────────────────────────────────────────────────────


def _headers_are_streaming_safe(headers) -> None:
    cache = headers.get("cache-control", "")
    assert "no-cache" in cache and "no-transform" in cache, cache
    # nginx buffers an event-stream unless told not to; that alone reproduces
    # "对话卡住" with a perfectly healthy backend.
    assert headers.get("x-accel-buffering") == "no"


def test_chat_stream_sets_no_buffering_headers() -> None:
    from mast.api.routes.chat_stream import router

    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    c = TestClient(app)
    with c.stream(
        "POST", "/api/agents/instrument_control/chat",
        json={"conversation_id": "c1", "user_text": "hi"},
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        _headers_are_streaming_safe(r.headers)
        frames = [
            json.loads(ln[len("data:"):].strip())
            for ln in r.iter_lines() if ln.startswith("data:")
        ]
    # the degraded path still terminates properly through the wrapper
    assert frames[-1]["kind"] == "done"


def test_run_task_sets_no_buffering_headers() -> None:
    from mast.api.routes.orchestrator import router

    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    c = TestClient(app)
    with c.stream("POST", "/api/agents/run-task", json={"task": "hello"}) as r:
        assert r.status_code == 200
        _headers_are_streaming_safe(r.headers)
        frames = [
            json.loads(ln[len("data:"):].strip())
            for ln in r.iter_lines() if ln.startswith("data:")
        ]
    assert frames[-1]["kind"] == "done"


def test_sse_headers_constant_is_complete() -> None:
    assert set(SSE_HEADERS) == {"Cache-Control", "X-Accel-Buffering", "Connection"}
