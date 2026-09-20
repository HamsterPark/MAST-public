"""MAST experiment record system v2 — per compass terminal-plan.

13-entity schema, HLC time, ULID ids, append-only triggers, CAS for scan files.
Parallel to the vendored v1 logging stack; strangler migration in Phase 2+.

Public API surface:
    from mast.logging.v2 import (
        HLC, HLCClock, ulid_now, ulid_timestamp_ms,
        ExperimentStoreV2, open_store,
        repos, cas, schema,
    )
"""
from __future__ import annotations

from mast.logging.v2.hlc import HLC, HLCClock
from mast.logging.v2.ulid import ulid_from_ms, ulid_now, ulid_timestamp_ms
from mast.logging.v2.storage import ExperimentStoreV2, open_store
from mast.logging.v2 import cas, repos, schema

__all__ = [
    "HLC",
    "HLCClock",
    "ulid_now",
    "ulid_from_ms",
    "ulid_timestamp_ms",
    "ExperimentStoreV2",
    "open_store",
    "cas",
    "repos",
    "schema",
]

SCHEMA_VERSION = "2.0.0"
