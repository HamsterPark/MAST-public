"""Native tunnelling-current monitoring — segmented acquisition, feature
extraction, rolling storage and advisory alerts.

Self-contained subsystem (same shape as ``mast.billing``): nothing outside
imports its internals, and every entry point degrades instead of raising so a
monitoring hiccup can never break acquisition, the API or the agents.

Layout::

    pump.py        Osci1T segment pump (hardware buffer -> ~1 s Segments)
    features.py    pure-function feature extraction (numpy/scipy only)
    store.py       standalone SQLite + .npy segment files + retention sweep
    thresholds.py  live-read operator knobs (settings -> next segment)
    alerts.py      WARN/CRITICAL adjudication, evidence PNG, event emission
    service.py     the daemon: lifecycle, context labels, suppression, telemetry
    export.py      offline corpus export (npy + sidecar + parquet/jsonl)

Design rules that outlive any single file:

* The monitor is ADVISORY. It never touches the tip, the bias, Z or the motors —
  the worst it does is emit an event that a human or an agent acts on. (Same
  invariant the vision scan monitor holds.) It DOES write to the oscilloscope's
  own display settings (Run / ChSet / TrigSet / TimebaseSet), which is why the
  scope is treated as shared state and its channel re-verified every segment.
* It reads on ``role="data"`` only, and takes no instrument token: a read-only
  observer must never queue behind a ten-minute scan.
* Arrays go to ``.npy``; only scalars ride the event bus.
"""
from __future__ import annotations

__all__ = [
    "get_store",
    "set_store_for_test",
    "get_monitor_thresholds",
    "set_monitor_thresholds",
    "get_service",
    "start_service",
    "stop_service",
]


def __getattr__(name: str):
    """Lazy re-export so ``import mast.monitoring`` stays cheap (and works even
    when numpy/scipy are unavailable — the API layer probes this package)."""
    if name in ("get_store", "set_store_for_test"):
        from mast.monitoring import store
        return getattr(store, name)
    if name in ("get_monitor_thresholds", "set_monitor_thresholds"):
        from mast.monitoring import thresholds
        return getattr(thresholds, name)
    if name in ("get_service", "start_service", "stop_service"):
        from mast.monitoring import service
        return getattr(service, name)
    raise AttributeError(name)
