"""Lightweight per-agent-type run-duration statistics — the learning substrate
for smarter auto-backgrounding (item ②).

The BackgroundRunManager records each finished run's ``(agent_type, duration_s)``
through an injected ``duration_sink``. This store accumulates those durations per
type in a small BOUNDED ring, persists them to a JSON file (best-effort — a
corrupt/absent file degrades to empty, never raises), and answers the two
questions the auto-background policy asks:

  * ``count(agent_type)``  — how many samples do we have? (below a floor the
    signal is not trusted and the caller falls back to the static whitelist);
  * ``is_slow(agent_type, threshold_s, min_samples)`` — is this type CONSISTENTLY
    slow enough to be worth detaching? Returns ``None`` (== "not enough data,
    prefer the whitelist") when samples < ``min_samples``, else ``p50 > threshold``.

Deliberately dependency-light (stdlib only) and unit-testable WITHOUT the runtime:
pass ``path=None`` for a pure in-memory store, or a real path for persistence.
Thread-safe: one lock guards the in-memory rings AND the file write, because the
manager calls ``record`` from its background worker threads concurrently.

This is pure OBSERVABILITY — it never influences hardware and only ever informs a
conservative, default-off routing hint (see ``_split_auto_background``).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def _p50(values: "list[float]") -> "float | None":
    """Lower-median (no interpolation) — deterministic and dependency-free."""
    if not values:
        return None
    s = sorted(values)
    return s[(len(s) - 1) // 2]


class RunStatsStore:
    """Bounded, JSON-backed per-type duration history. Thread-safe + best-effort.

    ``max_per_type`` caps the ring so a long-lived install can't grow the file
    unbounded and so the median tracks RECENT behaviour rather than ancient runs.
    """

    def __init__(self, path: "str | Path | None" = None, *, max_per_type: int = 50) -> None:
        self._path = Path(path) if path else None
        self._max = max(4, int(max_per_type))
        self._lock = threading.Lock()
        self._durations: dict[str, list[float]] = {}
        self._load()

    # ── persistence (best-effort; never raises) ──────────────────────────────
    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            data = raw.get("durations", raw) if isinstance(raw, dict) else {}
            out: dict[str, list[float]] = {}
            for k, v in (data or {}).items():
                if isinstance(v, list):
                    vals = [float(x) for x in v
                            if isinstance(x, (int, float)) and float(x) >= 0]
                    if vals:
                        out[str(k)] = vals[-self._max:]
            self._durations = out
        except Exception as exc:  # noqa: BLE001 — a bad file must degrade to empty
            logger.debug("run stats load failed (%s): starting empty", exc)
            self._durations = {}

    def _save_locked(self) -> None:
        """Atomic-ish write (tmp + replace) under the caller-held lock."""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps({"durations": self._durations},
                                      ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("run stats save failed: %s", exc)

    # ── recording (the manager's duration_sink target) ───────────────────────
    def record(self, agent_type: str, duration_s: float) -> None:
        """Append one finished run's duration. Ignores blanks / negatives / NaNs."""
        if not agent_type:
            return
        try:
            d = float(duration_s)
        except (TypeError, ValueError):
            return
        if d < 0 or d != d:  # negative or NaN
            return
        with self._lock:
            ring = self._durations.setdefault(str(agent_type), [])
            ring.append(d)
            if len(ring) > self._max:
                del ring[: len(ring) - self._max]
            self._save_locked()

    # ── queries ──────────────────────────────────────────────────────────────
    def count(self, agent_type: str) -> int:
        with self._lock:
            return len(self._durations.get(str(agent_type), ()))

    def p50(self, agent_type: str) -> "float | None":
        with self._lock:
            return _p50(list(self._durations.get(str(agent_type), ())))

    def is_slow(self, agent_type: str, *, threshold_s: float,
                min_samples: int) -> "bool | None":
        """Is *agent_type* consistently slow enough to be worth detaching?

        PREFER-MISSING: returns ``None`` when we have fewer than ``min_samples``
        observations (the caller then falls back to its static whitelist rather
        than trusting a thin signal). Otherwise ``p50(durations) > threshold_s``.
        """
        with self._lock:
            vals = list(self._durations.get(str(agent_type), ()))
        if len(vals) < max(1, int(min_samples)):
            return None
        med = _p50(vals)
        return med is not None and med > float(threshold_s)

    def snapshot(self) -> dict:
        """JSON-safe {agent_type: {count, p50}} for UI / debug."""
        with self._lock:
            return {k: {"count": len(v), "p50": _p50(v)}
                    for k, v in self._durations.items()}


__all__ = ["RunStatsStore"]
