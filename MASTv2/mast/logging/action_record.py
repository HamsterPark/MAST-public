"""ActionRecord serialization helpers for SQLite storage."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from mast.core.types import (
    ActionRecord,
    HardwareState,
    NanonisCallRecord,
    SkillResult,
)


def _json_default(o: Any) -> Any:
    """Fallback for json.dumps when it hits a non-JSON-serializable value.

    Handles the values that appear in NanonisCallRecord but cannot be
    natively encoded:
    - bytes / bytearray → "<bytes len=N>" (raw protocol bytes from
      NanonisCallRecord.raw_bytes; we don't need to round-trip them
      to inspect mission history, only to keep them out of the JSON)
    - tuple → list (preserves nested call return_values)
    - everything else → str(o) so the row still writes, with the type
      visible to the operator
    """
    if isinstance(o, (bytes, bytearray)):
        return f"<bytes len={len(o)}>"
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def _serialize(obj: Any) -> str:
    """Serialize a dataclass or list of dataclasses to a JSON string."""
    if obj is None:
        return ""
    if isinstance(obj, list):
        return json.dumps([asdict(item) for item in obj], default=_json_default)
    return json.dumps(asdict(obj), default=_json_default)


def _deserialize_hardware_state(raw: str) -> HardwareState | None:
    if not raw:
        return None
    return HardwareState(**json.loads(raw))


def _deserialize_skill_result(raw: str) -> SkillResult | None:
    if not raw:
        return None
    d = json.loads(raw)
    d["state_before"] = (
        HardwareState(**d["state_before"]) if d.get("state_before") else None
    )
    d["state_after"] = (
        HardwareState(**d["state_after"]) if d.get("state_after") else None
    )
    d["nanonis_calls"] = [
        NanonisCallRecord(**c) for c in (d.get("nanonis_calls") or [])
    ]
    for call in d["nanonis_calls"]:
        if isinstance(call.args, list):
            call.args = tuple(call.args)
    return SkillResult(**d)


def _deserialize_nanonis_calls(raw: str) -> list[NanonisCallRecord]:
    if not raw:
        return []
    items = json.loads(raw)
    calls = []
    for c in items:
        if isinstance(c.get("args"), list):
            c["args"] = tuple(c["args"])
        calls.append(NanonisCallRecord(**c))
    return calls


def action_to_dict(record: ActionRecord) -> dict:
    """Serialize ActionRecord to a flat dict for SQLite storage.

    Nested objects (SkillResult, HardwareState, NanonisCallRecord lists)
    are stored as JSON strings.
    """
    return {
        "id": record.id,
        "experiment_id": record.experiment_id,
        "sample_id": record.sample_id,
        "timestamp": record.timestamp,
        "skill_name": record.skill_name,
        "skill_version": record.skill_version,
        "parameters": json.dumps(record.parameters),
        "result": _serialize(record.result),
        "state_before": _serialize(record.state_before),
        "state_after": _serialize(record.state_after),
        "nanonis_calls": _serialize(record.nanonis_calls),
        "context": record.context,
        "duration_s": record.duration_s,
        "approval_source": record.approval_source,
    }


def dict_to_action(d: dict) -> ActionRecord:
    """Deserialize from SQLite row dict back to ActionRecord."""
    return ActionRecord(
        id=d["id"],
        experiment_id=d.get("experiment_id", ""),
        sample_id=d.get("sample_id") or "",
        timestamp=d.get("timestamp", ""),
        skill_name=d.get("skill_name", ""),
        skill_version=d.get("skill_version", ""),
        parameters=json.loads(d["parameters"]) if d.get("parameters") else {},
        result=_deserialize_skill_result(d.get("result", "")),
        state_before=_deserialize_hardware_state(d.get("state_before", "")),
        state_after=_deserialize_hardware_state(d.get("state_after", "")),
        nanonis_calls=_deserialize_nanonis_calls(d.get("nanonis_calls", "")),
        context=d.get("context", ""),
        duration_s=d.get("duration_s", 0.0),
        approval_source=d.get("approval_source", "auto"),
    )
