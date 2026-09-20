"""Hybrid Logical Clock (HLC) — Kulkarni et al., OPODIS 2014.

Encodes (physical_ms, logical_counter, node_id) so:
- Lexical ordering of the encoded string equals causal ordering of events.
- Tolerates physical clock skew up to ``max_offset_ms`` (default 500 ms).
- Multi-agent / multi-process safe via instance-local lock.

Encoded format: ``"PPPPPPPPPPPPP-CCCC-NODE"``
- 13 zero-padded digits of physical milliseconds since epoch (good until ~year 2286)
- 4 zero-padded digits of logical counter (max 9999 events per ms per node)
- Free-form node tag (e.g. ``"XD"``, ``"IC"``, ``"OP"``)

Any two encoded HLC strings compare byte-wise in the same order as their
causal precedence (provided ``max_offset_ms`` invariant holds).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class HLC:
    """An immutable HLC instant."""

    pt: int            # physical milliseconds since epoch
    counter: int       # logical sub-millisecond counter
    node: str          # short node id (e.g. "XD", "IC")

    def encode(self) -> str:
        return f"{self.pt:013d}-{self.counter:04d}-{self.node}"

    @classmethod
    def parse(cls, encoded: str) -> "HLC":
        parts = encoded.split("-", 2)
        if len(parts) != 3:
            raise ValueError(f"Malformed HLC string: {encoded!r}")
        pt, counter, node = parts
        return cls(int(pt), int(counter), node)

    def __lt__(self, other: "HLC") -> bool:
        return (self.pt, self.counter, self.node) < (other.pt, other.counter, other.node)

    def __le__(self, other: "HLC") -> bool:
        return (self.pt, self.counter, self.node) <= (other.pt, other.counter, other.node)


class HLCClock:
    """Thread-safe HLC generator with peer-update support."""

    _COUNTER_OVERFLOW = 10_000

    def __init__(self, node_id: str, *, max_offset_ms: int = 500):
        if not node_id or "-" in node_id:
            raise ValueError(f"node_id must be non-empty and contain no '-': {node_id!r}")
        self.node_id = node_id
        self.max_offset_ms = max_offset_ms
        self._lock = threading.Lock()
        self._last_pt = 0
        self._last_counter = 0

    # ── Public API ────────────────────────────────────────────────────

    def now(self) -> HLC:
        """Mint a fresh HLC at this node."""
        with self._lock:
            phys = self._wall_now_ms()
            if phys > self._last_pt:
                self._last_pt = phys
                self._last_counter = 0
            else:
                self._last_counter += 1
                self._check_counter_overflow()
            return HLC(self._last_pt, self._last_counter, self.node_id)

    def update(self, remote: HLC) -> HLC:
        """Receive a remote HLC, fold it in, return a strictly-greater local HLC."""
        with self._lock:
            phys = self._wall_now_ms()
            self._assert_skew(phys, remote.pt)
            new_pt = max(phys, self._last_pt, remote.pt)
            if new_pt == self._last_pt == remote.pt:
                new_counter = max(self._last_counter, remote.counter) + 1
            elif new_pt == self._last_pt:
                new_counter = self._last_counter + 1
            elif new_pt == remote.pt:
                new_counter = remote.counter + 1
            else:
                new_counter = 0
            self._last_pt = new_pt
            self._last_counter = new_counter
            self._check_counter_overflow()
            return HLC(self._last_pt, self._last_counter, self.node_id)

    # ── Internals ─────────────────────────────────────────────────────

    @staticmethod
    def _wall_now_ms() -> int:
        return int(time.time() * 1000)

    def _assert_skew(self, phys: int, remote_pt: int) -> None:
        offset = abs(phys - remote_pt)
        if offset > self.max_offset_ms:
            raise ValueError(
                f"HLC offset {offset} ms between local clock and remote.pt "
                f"exceeds max_offset_ms={self.max_offset_ms}. "
                "Check NTP/chrony on this node."
            )

    def _check_counter_overflow(self) -> None:
        if self._last_counter >= self._COUNTER_OVERFLOW:
            raise RuntimeError(
                f"HLC counter overflow in 1 ms on node {self.node_id!r} "
                f"(> {self._COUNTER_OVERFLOW} events). Slow down or shard."
            )
