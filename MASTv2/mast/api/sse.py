"""Shared SSE plumbing — response headers + a keep-alive wrapper.

The operator drives this box over Tailscale, and a roam / re-route / sleep kills
the TCP connection under a long-running SSE stream. Two independent problems fed
that symptom, and this module fixes the backend half of both:

1. **Silence gets a live stream murdered.** A stream that emits nothing for
   minutes (one long LLM turn, one long instrument tool call, a 10-minute
   operator hold) looks dead to every idle reaper in the path — WireGuard
   keepalive windows, NAT/gateway idle timeouts, Wi-Fi roaming, laptop sleep.
   ``with_heartbeat`` injects an SSE **comment** frame (``: ping``) whenever the
   producer has been quiet for ``HEARTBEAT_S``. Comment frames are ignored by
   every SSE parser (including our hand-rolled ones — they carry no ``data:``
   line), so they cost nothing but keep the socket demonstrably alive.

2. **Silence is indistinguishable from death on the client.** Without beats a
   browser cannot tell "the backend is thinking" from "the socket died three
   minutes ago", so it cannot honestly offer a reconnect. Regular beats give the
   frontend an idle watchdog it can trust (see ``frontend/src/lib/sse.ts``).

``SSE_HEADERS`` additionally stops *proxies* from doing the same thing by
buffering: nginx/Caddy in front of MAST (docs/v2/ops/external-access.md) will
happily hold a chunked response until it has "enough" bytes unless told not to.

DISCONNECT SEMANTICS (deliberate, changed by this module)
---------------------------------------------------------
``with_heartbeat`` drives the wrapped generator on its own worker thread. When
the client goes away, the consumer generator is closed and we set ``stop`` — but
the producer is then **drained to completion with its output discarded** rather
than having ``GeneratorExit`` thrown into it. That is on purpose:

  • the producer's own ``finally`` still runs (transcript terminal row, training
    trajectory close, task-slot release) — it just runs at the run's natural end;
  • a Wi-Fi roam no longer silently aborts a running experiment mid-scan. Before
    this, losing the stream closed the orchestrator's graph generator at the next
    super-step boundary, i.e. a network blip acted exactly like pressing 中止.
    The frontend already assumed the opposite (its reconnect banner is gated on
    ``serverActive && !running`` and offers 刷新进展 / 中止后端运行) — this makes
    the backend match that contract.

Stopping a run therefore stays an explicit operator act (``/api/agents/run-task/
abort``), which is the only place hardware-affecting cancellation belongs.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

# Emit a keep-alive after this much producer silence. Matches ws.py's 15 s ping
# and orchestrator._APPROVAL_BEAT_S — comfortably inside the ~30-60 s idle
# windows that consumer NAT/gateway boxes and WireGuard keepalive use.
HEARTBEAT_S = 15.0

# An SSE *comment* line: valid per the spec, ignored by EventSource and by our
# hand-rolled readers (no ``data:`` line → empty payload → skipped). Its only job
# is to put bytes on the wire.
BEAT = ": ping\n\n"

# How often to consult ``on_idle`` while the producer is blocked. Deliberately
# much shorter than HEARTBEAT_S: this is the latency of "a tool started" / "still
# waiting on you, 40s" reaching the operator, and those are worth about a second.
# Only used when an ``on_idle`` sink is supplied; otherwise the poll clock IS the
# ping clock, exactly as before.
IDLE_POLL_S = 1.0

# Headers every SSE response should carry.
#   Cache-Control  — no intermediary may cache or (crucially) *transform* an
#                    event stream; `no-transform` is what stops content-rewriting
#                    proxies from re-chunking it.
#   X-Accel-Buffering — nginx-specific opt-out of response buffering. Without it
#                    nginx buffers the stream and the client sees nothing until
#                    the run ends, which is "对话卡住" with extra steps.
#   Connection     — hop-by-hop hint for HTTP/1.1 intermediaries.
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-store, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

# Bounded hand-off so a producer that outruns a slow client can't grow the queue
# without limit; the pump blocks on a full queue (and keeps checking `stop`).
_QUEUE_MAX = 256


class _End:
    """Terminal sentinel: carries the producer's exception, if any."""

    __slots__ = ("error",)

    def __init__(self, error: BaseException | None) -> None:
        self.error = error


def with_heartbeat(
    inner: Iterator[str],
    *,
    interval: float = HEARTBEAT_S,
    label: str = "sse",
    on_idle: "Any | None" = None,
) -> Iterator[str]:
    """Wrap a SYNC SSE generator so it never goes silent for > ``interval``.

    Yields everything ``inner`` yields, plus a ``BEAT`` comment frame whenever
    ``inner`` has produced nothing for ``interval`` seconds. Exceptions raised by
    ``inner`` are re-raised here (same observable behaviour as iterating it
    directly); its ``finally`` blocks always run — see the module docstring for
    the deliberate client-disconnect semantics.

    ``inner`` is consumed on a daemon thread, so the caller (Starlette's
    threadpool) is only ever blocked on a queue read.

    ``on_idle`` — SAY SOMETHING TRUE INSTEAD OF ``: ping`` (2026-08-27)
    ------------------------------------------------------------------
    Optional zero-arg callable returning an iterable of ready-to-send SSE
    strings. It is consulted **only** when the producer has been quiet for
    ``interval``; whatever it returns is yielded, and the ``BEAT`` comment is
    sent only if it returned nothing. So the wire is never quieter than before —
    this can only *upgrade* a ping into real information.

    It exists because a producer that is a blocked **sync generator** cannot
    yield: while ``drive_group_run`` sits in ``next(gen)`` waiting for a tool or
    for the operator to answer a HITL question, the frames it has already
    generated have no way out. A ping keeps the socket alive but says nothing,
    and "alive but silent" is what read as 卡住了. ``on_idle``
    gives those frames a second exit that does not pass through the generator.

    It runs on the CONSUMER thread and must therefore be cheap and must not
    block — it is called between two queue reads, so a slow ``on_idle`` delays
    real frames. Draining a lock-guarded list is the intended shape. Exceptions
    are swallowed (a keep-alive path must never be able to kill a live run) and
    degrade to the plain ``BEAT``.
    """
    q: queue.Queue[Any] = queue.Queue(maxsize=_QUEUE_MAX)
    stop = threading.Event()

    def _pump() -> None:
        err: BaseException | None = None
        try:
            for item in inner:
                if stop.is_set():
                    # Consumer is gone. Keep pulling so the producer runs to its
                    # natural end (and its finally with it), but drop the frames.
                    continue
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 — relayed to the consumer verbatim
            err = exc
        finally:
            try:
                q.put_nowait(_End(err))
            except queue.Full:  # consumer already gone — nothing to tell
                pass

    beat = max(0.5, float(interval))
    # With an on_idle sink we poll FASTER than we ping. "A tool started" is worth
    # ~a second of latency, not ~15; but pinging every second would be noise on an
    # idle stream. So: poll on the short clock, ping on the original one.
    # Without on_idle the two collapse to one and the behaviour is byte-identical
    # to before — this wrapper is on every SSE route in the app, and a keep-alive
    # layer is the last place that should acquire a new failure mode.
    poll = min(beat, IDLE_POLL_S) if on_idle is not None else beat
    thread = threading.Thread(target=_pump, name=f"sse-pump-{label}", daemon=True)
    thread.start()
    last_beat = time.monotonic()
    try:
        while True:
            try:
                item = q.get(timeout=poll)
            except queue.Empty:
                # Producer is quiet. Before falling back to a content-free ping,
                # ask whether anything real is waiting behind the block.
                said = False
                if on_idle is not None:
                    try:
                        for frame in (on_idle() or ()):
                            if frame:
                                said = True
                                yield frame
                    except Exception as exc:  # noqa: BLE001 — keep-alive must not kill a run
                        logger.debug("%s: on_idle failed: %s", label, exc)
                now = time.monotonic()
                if said:
                    last_beat = now          # real bytes count as the keep-alive
                elif now - last_beat >= beat:
                    last_beat = now
                    yield BEAT
                continue
            last_beat = time.monotonic()
            if isinstance(item, _End):
                if item.error is not None:
                    raise item.error
                return
            yield item
    finally:
        # Consumer closed (client disconnect / normal return). Release the pump;
        # it drains the producer to completion in the background.
        stop.set()


__all__ = ["BEAT", "HEARTBEAT_S", "IDLE_POLL_S", "SSE_HEADERS", "with_heartbeat"]
