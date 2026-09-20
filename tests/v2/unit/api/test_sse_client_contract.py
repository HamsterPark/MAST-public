"""— the FRONTEND half, pinned from Python.

There is no JS test runner in this repo (frontend/package.json has playwright
only), so the client-side invariants of the #34 fix are guarded here as source
contracts. They are cheap and they pin the exact regression:

    for (;;) { const { value, done } = await reader.read(); if (done) break; }

That loop cannot distinguish "the server finished" from "the socket died",
because nothing tracks whether a terminal frame ever arrived. Under it a
Tailscale roam ended the transcript mid-sentence with no spinner, no error and
no reconnect banner — the conversation LOOKED finished. Every SSE consumer must
therefore go through `readSseStream`, which reports how the stream ended.

The behaviour of `readSseStream` / the hierarchy grouping itself was verified by
running the compiled modules under node (esbuild bundle + scenario harness:
clean finish / truncated body / keep-alive comments / half-open socket /
user abort). That harness is not committed — these contracts are what stop the
old pattern coming back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mast.api.sse import HEARTBEAT_S

_FRONTEND = Path(__file__).resolve().parents[4] / "frontend" / "src"

# Every view that consumes one of the SSE endpoints.
_SSE_CONSUMERS = (
    "components/agents/runTaskStore.ts",
    "pages/ChatPage.tsx",
    "components/agents/AgentChatPanel.tsx",
)


def _read(rel: str) -> str:
    p = _FRONTEND / rel
    if not p.exists():  # pragma: no cover - keeps the suite honest if files move
        pytest.skip(f"frontend source not present: {p}")
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize("rel", _SSE_CONSUMERS)
def test_sse_consumer_uses_the_shared_reader(rel: str) -> None:
    src = _read(rel)
    assert "readSseStream" in src, (
        f"{rel} consumes SSE without the shared reader — it cannot tell a "
        "finished stream from a dropped connection "
    )


@pytest.mark.parametrize("rel", _SSE_CONSUMERS)
def test_no_hand_rolled_reader_loop_survives(rel: str) -> None:
    """The literal shape of the bug: a raw reader whose only exit is `done`."""
    src = _read(rel)
    assert "getReader()" not in src, (
        f"{rel} re-introduced a hand-rolled SSE reader loop; route it through "
        "readSseStream instead "
    )


@pytest.mark.parametrize("rel", _SSE_CONSUMERS)
def test_broken_stream_is_surfaced_to_the_operator(rel: str) -> None:
    """Detecting the break is only half of it — the operator has to be told.
    'UI 绝不冻结' is violated by a screen that silently claims success."""
    src = _read(rel)
    assert "sseBroke" in src, f"{rel} never checks whether the stream broke "
    assert "setStreamErr" in src or "streamErr" in src, (
        f"{rel} detects a broken stream but shows the operator nothing "
    )


def test_idle_watchdog_is_a_safe_multiple_of_the_server_beat() -> None:
    """The client may only declare a link dead after several missed beats —
    otherwise a slow-but-healthy run gets killed by its own watchdog."""
    src = _read("lib/sse.ts")
    m = re.search(r"SSE_IDLE_MS\s*=\s*([\d_]+)", src)
    assert m, "SSE_IDLE_MS not found in frontend/src/lib/sse.ts"
    idle_ms = int(m.group(1).replace("_", ""))
    assert idle_ms >= 4 * HEARTBEAT_S * 1000, (
        f"idle budget {idle_ms} ms is under 4 keep-alive intervals "
        f"({HEARTBEAT_S}s) — beat jitter would fake a dead connection"
    )


def test_reader_treats_eof_without_terminal_frame_as_a_break() -> None:
    """The one line the whole fix turns on."""
    src = _read("lib/sse.ts")
    assert "sawTerminal" in src
    assert re.search(r"sawTerminal\s*\?\s*\{\s*reason:\s*\"done\"\s*\}\s*:\s*"
                     r"\{\s*reason:\s*\"truncated\"\s*\}", src), (
        "readSseStream no longer distinguishes a terminal frame from a plain "
        "EOF — that IS the #34 bug"
    )
