"""Standalone SQLite ledger for API usage/cost events.

Append-only on the hot path, aggregated on read. Its own DB file (sibling of
``mast_experiments.db``) so it is isolated + resettable without touching
experiment data. Thread-safe (one lock; the WS/agent threads + the API thread
all write) and fail-safe: :meth:`record` never raises — a billing hiccup must
never break the API call that triggered it.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    kind: str                       # llm | tts | asr | ocr | embedding
    provider: str
    model: str
    source: str = ""               # agent id, or chat/quickask/voice/ocr/…
    input_tokens: int = 0
    output_tokens: int = 0
    chars: int = 0
    seconds: float = 0.0
    cost: float = 0.0
    currency: str = "CNY"
    cost_known: bool = True
    ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    kind          TEXT    NOT NULL,
    provider      TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    source        TEXT,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    chars         INTEGER DEFAULT 0,
    seconds       REAL    DEFAULT 0,
    cost          REAL    DEFAULT 0,
    currency      TEXT,
    cost_known    INTEGER DEFAULT 1,
    meta          TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(ts);
CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_events(provider);
"""


class UsageLedger:
    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
            self._conn.commit()

    # ── write ────────────────────────────────────────────────────────────────
    def record(self, rec: UsageRecord) -> None:
        """Append one event. Fail-safe: swallows + logs, never raises."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO usage_events (ts, kind, provider, model, source, "
                    "input_tokens, output_tokens, chars, seconds, cost, currency, "
                    "cost_known, meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        float(rec.ts), rec.kind, rec.provider, rec.model, rec.source,
                        int(rec.input_tokens or 0), int(rec.output_tokens or 0),
                        int(rec.chars or 0), float(rec.seconds or 0.0),
                        float(rec.cost or 0.0), rec.currency,
                        1 if rec.cost_known else 0,
                        json.dumps(rec.meta or {}, ensure_ascii=False),
                    ),
                )
                self._conn.commit()
        except Exception:  # noqa: BLE001 — billing must never break the call
            logger.debug("UsageLedger.record failed (swallowed)", exc_info=True)

    # ── read / aggregate ─────────────────────────────────────────────────────
    def _range(self, since: Optional[float], until: Optional[float]) -> tuple[str, list]:
        clauses, args = [], []
        if since is not None:
            clauses.append("ts >= ?"); args.append(float(since))
        if until is not None:
            clauses.append("ts < ?"); args.append(float(until))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, args

    def _group(self, dim: str, where: str, args: list) -> list[dict]:
        rows = self._conn.execute(
            f"SELECT {dim} AS key, currency, "
            "COUNT(*) AS n, SUM(cost) AS cost, "
            "SUM(input_tokens) AS in_tok, SUM(output_tokens) AS out_tok, "
            "MIN(cost_known) AS all_known "
            f"FROM usage_events{where} GROUP BY {dim}, currency "
            "ORDER BY cost DESC",
            args,
        ).fetchall()
        return [
            {
                "key": r["key"] or "—",
                "currency": r["currency"] or "CNY",
                "count": r["n"],
                "cost": round(r["cost"] or 0.0, 6),
                "input_tokens": r["in_tok"] or 0,
                "output_tokens": r["out_tok"] or 0,
                "all_priced": bool(r["all_known"]),
            }
            for r in rows
        ]

    def summary(self, since: Optional[float] = None,
                until: Optional[float] = None) -> dict:
        """Aggregated spend: per-currency totals + breakdown by provider / model /
        source / kind. Read-only; safe to call concurrently with writes."""
        with self._lock:
            where, args = self._range(since, until)
            totals = self._conn.execute(
                "SELECT currency, COUNT(*) AS n, SUM(cost) AS cost, "
                "SUM(CASE WHEN cost_known=0 THEN cost ELSE 0 END) AS est_cost "
                f"FROM usage_events{where} GROUP BY currency", args,
            ).fetchall()
            by_currency = {
                (r["currency"] or "CNY"): {
                    "cost": round(r["cost"] or 0.0, 6),
                    "count": r["n"],
                    "estimated_cost": round(r["est_cost"] or 0.0, 6),
                }
                for r in totals
            }
            total_count = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM usage_events{where}", args,
            ).fetchone()["n"]
            out = {
                "since": since,
                "until": until,
                "count": total_count,
                "by_currency": by_currency,
                "by_provider": self._group("provider", where, args),
                "by_model": self._group("model", where, args),
                "by_source": self._group("source", where, args),
                "by_kind": self._group("kind", where, args),
            }
        # combined 折算 (labelled estimate; never written to the ledger)
        try:
            from mast.billing.pricing import usd_to_cny_rate
            rate = usd_to_cny_rate()
            combined = 0.0
            for cur, agg in by_currency.items():
                combined += agg["cost"] * (rate if cur == "USD" else 1.0)
            out["combined_cny"] = round(combined, 4)
            out["usd_to_cny"] = rate
        except Exception:  # noqa: BLE001
            pass
        return out

    def recent(self, limit: int = 50,
               since: Optional[float] = None) -> list[dict]:
        with self._lock:
            where, args = self._range(since, None)
            rows = self._conn.execute(
                "SELECT ts, kind, provider, model, source, input_tokens, "
                "output_tokens, chars, seconds, cost, currency, cost_known "
                f"FROM usage_events{where} ORDER BY ts DESC LIMIT ?",
                args + [int(limit)],
            ).fetchall()
        return [
            {
                "ts": r["ts"], "kind": r["kind"], "provider": r["provider"],
                "model": r["model"], "source": r["source"],
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                "chars": r["chars"], "seconds": r["seconds"],
                "cost": round(r["cost"] or 0.0, 6), "currency": r["currency"],
                "cost_known": bool(r["cost_known"]),
            }
            for r in rows
        ]

    def reset(self) -> int:
        """Wipe all events. Returns rows deleted."""
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) AS n FROM usage_events").fetchone()["n"]
            self._conn.execute("DELETE FROM usage_events")
            self._conn.commit()
        return n

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


# ── process-global singleton ─────────────────────────────────────────────────

_LEDGER: UsageLedger | None = None
_LEDGER_LOCK = threading.Lock()


def _default_path() -> Path:
    from mast._runtime_paths import project_root
    return Path(project_root()) / "experiments" / "usage_ledger.sqlite"


def get_ledger() -> UsageLedger:
    global _LEDGER
    if _LEDGER is None:
        with _LEDGER_LOCK:
            if _LEDGER is None:
                _LEDGER = UsageLedger(_default_path())
    return _LEDGER


def set_ledger_for_test(ledger: UsageLedger | None) -> None:
    """Swap the singleton (tests point it at a tmp DB)."""
    global _LEDGER
    with _LEDGER_LOCK:
        _LEDGER = ledger


__all__ = ["UsageRecord", "UsageLedger", "get_ledger", "set_ledger_for_test"]
