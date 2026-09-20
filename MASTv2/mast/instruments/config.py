"""Declarative optical-instrument inventory — ``config/optical_instruments.json``.

Follows the ``mast.environment.config`` pattern (external JSON manifest,
atomic save, resolved via ``project_root()``) rather than baking hardware
into ``mast/config.py``: benches differ per rig and get re-wired without a
code release.

Schema::

    {
      "devices": [
        {
          "id": "delay1",                  // unique handle
          "name": "泵浦-探测延迟线",         // display name
          "type": "pztc_nm003",            // driver key, see DEVICE_TYPES
          "enabled": true,
          // -- pztc_nm003 --
          "port": "COM17", "baudrate": 115200, "slave": 1,
          "counts_per_unit": 1.0,          // encoder counts per axis unit
          "position_tolerance": 5.0,       // counts
          "register_map": {                // one-time on-rig calibration
            "setpoint_place": null, "setpoint_speed": null,
            "place_counter": null, "closed_loop_move": null,
            "look_zero": null, "zero_status": null, "stop": null,
            "channel_stride": 0, "word_order_big": true
          },
          // -- pi_gcs --
          //   "port": "COM18", "baudrate": 115200, "model": "E-861",
          //   "servo_on_connect": true,
          // -- thorlabs_benchtop_piezo --
          //   "serial_no": "71000001", "dll_dir": null, "closed_loop": true,
          "axes": [
            {"name": "delay", "channel": 1,
             "min_pos": 0.0, "max_pos": 15000.0, "unit": "um",
             "settle_s": 0.05, "role": "delay_line"}
          ]
        }
      ],
      "delay_line": {                       // pump-probe delay conversion
        "device_id": "delay1", "axis": "delay",
        "unit_per_mm": 1000.0,              // axis native units per mm
        "ps_per_mm": 6.6713,                // optical delay per stage mm
        "zero_offset_mm": 0.0,              // stage pos (mm) of zero delay
        "sign": 1                           // +1: larger pos = later probe
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from mast._runtime_paths import project_root

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIG_FILENAME",
    "DEVICE_TYPES",
    "config_path",
    "load_config",
    "save_config",
    "device_by_id",
]

CONFIG_FILENAME = "optical_instruments.json"

#: driver keys accepted in a device entry's "type"
DEVICE_TYPES = ("pztc_nm003", "pi_gcs", "thorlabs_benchtop_piezo", "thorlabs_pdxc2")

_DEFAULT_CONFIG: dict[str, Any] = {"devices": [], "delay_line": None}


def config_path() -> Path:
    return project_root() / "config" / CONFIG_FILENAME


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read the manifest; a missing/corrupt file degrades to the empty
    inventory (never raises — no-hardware machines must boot clean)."""
    p = path or config_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return json.loads(json.dumps(_DEFAULT_CONFIG))
    except Exception as exc:
        logger.error("optical_instruments config unreadable (%s): %s", p, exc)
        return json.loads(json.dumps(_DEFAULT_CONFIG))
    if not isinstance(raw, dict):
        logger.error("optical_instruments config is not an object: %s", p)
        return json.loads(json.dumps(_DEFAULT_CONFIG))
    raw.setdefault("devices", [])
    raw.setdefault("delay_line", None)
    if not isinstance(raw["devices"], list):
        raw["devices"] = []
    return raw


def save_config(cfg: dict[str, Any], path: Path | None = None) -> Path:
    """Atomic write (tmp + replace), mirroring environment.config."""
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(cfg, ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(
        prefix=p.stem + ".", suffix=".tmp", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return p


def device_by_id(cfg: dict[str, Any], device_id: str) -> dict[str, Any] | None:
    for entry in cfg.get("devices", []):
        if isinstance(entry, dict) and entry.get("id") == device_id:
            return entry
    return None
