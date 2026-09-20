"""Persisted configuration for environment sensors (vacuum / temperature).

Stored as JSON at ``<project_root>/config/environment_sensors.json`` — the same
``project_root()`` resolution the API-key dir uses, so it follows the user's
relocated data dir (<data-root> via ``MAST2_PROJECT_ROOT``) instead of the
read-only install dir.

Schema::

    {
      "autodetect": true,                # scan COM ports for un-configured gauges
      "sensors": [
        {"id": "vac_main", "name": "主腔真空", "type": "dl7_vacuum",
         "port": "COM15", "address": 7, "unit": "Pa",
         "alarm": {"max": 1e-6, "warn_max": 1e-7}},
        {"id": "temp_sample", "name": "样品温度", "type": "lakeshore_temp",
         "port": "COM16", "channel": "A", "baudrate": 57600,
         "alarm": {"max": 10.0, "warn_max": 8.0}}
      ]
    }

Every accessor is best-effort: a missing / corrupt file yields the empty
default ``{"autodetect": True, "sensors": []}`` so the program never crashes
when no sensors are configured (explicit requirement #7).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "environment_sensors.json"
#: ``nanonis_signal`` reads one of the controller's 128 signal slots instead of
#: a serial port — the entry point for a magnet supply or a helium level meter
#: whose analogue output is wired into the Nanonis. It carries no ``port``, and
#: it is built by ``core.runtime`` rather than by ``environment.autodetect``,
#: because it needs a live ConnectionPool and environment/ must not import core/.
SENSOR_TYPES = ("dl7_vacuum", "lakeshore_temp", "nanonis_signal")

_DEFAULT: dict[str, Any] = {"autodetect": True, "sensors": []}


def config_path() -> Path:
    """Resolve the config file path (honours MAST2_PROJECT_ROOT / frozen exe)."""
    from mast._runtime_paths import project_root
    return project_root() / "config" / CONFIG_FILENAME


def load_config(path: Path | None = None) -> dict:
    """Load the sensor config. Returns the empty default on any error."""
    p = path or config_path()
    try:
        if not p.exists():
            return json.loads(json.dumps(_DEFAULT))  # fresh copy
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return json.loads(json.dumps(_DEFAULT))
        data.setdefault("autodetect", True)
        sensors = data.get("sensors")
        if not isinstance(sensors, list):
            data["sensors"] = []
        else:
            data["sensors"] = [s for s in sensors if isinstance(s, dict)]
        return data
    except Exception as exc:
        logger.warning("environment sensor config load failed (%s): %s", p, exc)
        return json.loads(json.dumps(_DEFAULT))


def save_config(data: dict, path: Path | None = None) -> Path:
    """Atomically write the sensor config. Creates parent dirs. Raises only on
    a genuine filesystem failure (caller surfaces it)."""
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "autodetect": bool(data.get("autodetect", True)),
        "sensors": [s for s in (data.get("sensors") or []) if isinstance(s, dict)],
    }
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp", prefix=CONFIG_FILENAME + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(p))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logger.info("Saved environment sensor config: %s (%d sensor(s))",
                p, len(payload["sensors"]))
    return p


def normalise_sensor(entry: dict, index: int = 0) -> dict:
    """Fill defaults / derive an id+name for one sensor entry."""
    e = dict(entry)
    stype = e.get("type")
    e.setdefault("id", f"{stype or 'sensor'}_{index}")
    if not e.get("name"):
        e["name"] = e["id"]
    return e


def update_sensor(sensor_id: str, *, path: Path | None = None, **fields) -> dict:
    """Merge *fields* into the sensor with *sensor_id*, persist, and return it.

    Used by the GUI "rename / set threshold" controls. Unknown sensor_id with
    a ``type`` field creates a new entry. ``alarm`` is replaced wholesale when
    provided (pass the full dict)."""
    cfg = load_config(path)
    sensors = cfg["sensors"]
    target = None
    for s in sensors:
        if s.get("id") == sensor_id:
            target = s
            break
    if target is None:
        target = {"id": sensor_id}
        sensors.append(target)
    for k, v in fields.items():
        if v is None:
            continue
        target[k] = v
    save_config(cfg, path)
    return target


def remove_sensor(sensor_id: str, *, path: Path | None = None) -> bool:
    cfg = load_config(path)
    before = len(cfg["sensors"])
    cfg["sensors"] = [s for s in cfg["sensors"] if s.get("id") != sensor_id]
    if len(cfg["sensors"]) != before:
        save_config(cfg, path)
        return True
    return False


def set_autodetect(enabled: bool, *, path: Path | None = None) -> None:
    cfg = load_config(path)
    cfg["autodetect"] = bool(enabled)
    save_config(cfg, path)


__all__ = [
    "CONFIG_FILENAME",
    "SENSOR_TYPES",
    "config_path",
    "load_config",
    "save_config",
    "normalise_sensor",
    "update_sensor",
    "remove_sensor",
    "set_autodetect",
]
