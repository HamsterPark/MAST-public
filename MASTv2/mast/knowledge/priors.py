"""Quantitative literature priors — read-only access to ``literature_priors.json``.

Design: literature_library_design.md §9.2.

``literature_priors.json`` carries per-material / per-phase parameter
percentiles (bias / setpoint / scan-size / temperature / …), aggregated by the
openalex-stm pipeline (``07_aggregate_priors.py``). Structure::

    {
      "by_material_phase": {
        "Au(111)": {
          "imaging": {
            "n_papers": 10,
            "sources": {...},
            "bias_v":  {"p25":…, "p50":…, "p75":…, "min":…, "max":…, "n":…},
            "temperature_k": {…},
            "scan_size_nm":  {…},
            ...
          },
          "sts": {…}
        },
        ...
      }
    }

This module is the *quantitative* complement to the literature agent's
*qualitative* prior art. It is a small, dependency-free, read-only helper that
Experiment Design (XD) and the GUI can call to seed bias/setpoint/scan-rate
suggestions for a material+phase.

Notes:
- ``mode`` here is the second-level key in ``by_material_phase[material]`` —
  i.e. the measurement *phase* ("imaging", "sts", …) as the design's §9.2
  ``literature_priors[...][M][mode]`` access path prescribes. It is NOT the
  three-tier simple/normal/expert knowledge-injection mode.
- Missing file / missing material / missing mode → graceful ``None`` / ``{}``;
  this module never raises on a lookup.
- Pure stdlib (``json``); no numpy / pandas dependency.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Path resolution (same repo-root walk as literature_index.py) ─────
_HERE = Path(__file__).resolve()
# this file lives at {repo}/MASTv2/mast/knowledge/priors.py — the json sits
# right next to it.
_PRIORS_PATH = _HERE.parent / "literature_priors.json"

# Known percentile/parameter sub-blocks (each is a {p25,p50,p75,min,max,n} dict).
# Anything that is NOT one of these top-level bookkeeping keys is treated as a
# parameter block. We do not hard-code the parameter names so newly-aggregated
# parameters surface automatically.
_NON_PARAM_KEYS = frozenset({"n_papers", "sources"})

# ── Cache (load-once, thread-safe; never holds tensors/handles) ──────
_lock = threading.Lock()
_cache: dict[str, Any] = {"data": None, "loaded": False, "mtime": None}


def _load() -> dict[str, Any]:
    """Load ``literature_priors.json`` once, caching the parsed dict.

    Returns the ``by_material_phase`` sub-dict (possibly ``{}``). Never raises:
    a missing/corrupt file degrades to an empty dict and logs a warning.
    The cache is invalidated when the file's mtime changes (cheap stat).
    """
    try:
        mtime = _PRIORS_PATH.stat().st_mtime if _PRIORS_PATH.exists() else None
    except OSError:
        mtime = None

    if _cache["loaded"] and _cache["mtime"] == mtime:
        return _cache["data"]

    with _lock:
        # Re-check inside the lock.
        if _cache["loaded"] and _cache["mtime"] == mtime:
            return _cache["data"]

        data: dict[str, Any] = {}
        if _PRIORS_PATH.exists():
            try:
                raw = json.loads(_PRIORS_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    bmp = raw.get("by_material_phase")
                    if isinstance(bmp, dict):
                        data = bmp
                    else:
                        logger.warning(
                            "literature_priors.json has no 'by_material_phase' dict"
                        )
                else:
                    logger.warning("literature_priors.json is not a JSON object")
            except (ValueError, OSError) as e:
                logger.warning("Failed to read literature_priors.json: %s", e)
        else:
            logger.info("literature_priors.json not found at %s", _PRIORS_PATH)

        _cache["data"] = data
        _cache["loaded"] = True
        _cache["mtime"] = mtime
        return data


def _is_param_block(value: Any) -> bool:
    """True if *value* looks like a percentile block ({p25,p50,p75,min,max,n})."""
    return (
        isinstance(value, dict)
        and "p50" in value
        and any(k in value for k in ("p25", "p75", "min", "max"))
    )


def get_priors(material: str, mode: str = "") -> dict[str, Any] | None:
    """Return the prior-parameter percentiles for *material* (and *mode*).

    Args:
        material: material/phase key, e.g. ``"Au(111)"`` (exact key in the
                  ``by_material_phase`` table). Required.
        mode:     measurement phase / sub-key, e.g. ``"imaging"`` or ``"sts"``.
                  When empty, the whole material record (all its modes) is
                  returned.

    Returns:
        - ``mode`` given and present → the mode dict
          (``{n_papers, sources, bias_v:{...}, temperature_k:{...}, ...}``).
        - ``mode`` empty → the full material record (``{imaging:{...}, sts:{...}}``).
        - material/mode missing, or file absent/corrupt → ``None``.

    Never raises. Safe to call without numpy/pandas.

    Example:
        >>> p = get_priors("Au(111)", "imaging")
        >>> p["temperature_k"]["p50"]
        300.0
    """
    if not isinstance(material, str) or not material:
        return None

    table = _load()
    record = table.get(material)
    if not isinstance(record, dict):
        return None

    if not mode:
        return record

    block = record.get(mode)
    if not isinstance(block, dict):
        return None
    return block


def list_materials() -> list[str]:
    """Return all material keys present in the priors table (sorted).

    Empty list if the file is missing/corrupt. Never raises.
    """
    table = _load()
    return sorted(table.keys())


def list_modes(material: str) -> list[str]:
    """Return the available modes/phases for *material* (e.g. ['imaging', 'sts']).

    Empty list if the material is unknown or the file is missing. Never raises.
    """
    record = get_priors(material, "")
    if not isinstance(record, dict):
        return []
    return sorted(k for k in record.keys() if k not in _NON_PARAM_KEYS)


def get_param(material: str, mode: str, param: str) -> dict[str, Any] | None:
    """Return a single parameter's percentile block, or ``None``.

    Args:
        material: e.g. ``"Au(111)"``.
        mode:     e.g. ``"imaging"``.
        param:    e.g. ``"bias_v"`` / ``"temperature_k"`` / ``"scan_size_nm"``.

    Returns the ``{p25,p50,p75,min,max,n}`` dict, or ``None`` if anything is
    missing or the resolved value is not a percentile block. Never raises.
    """
    block = get_priors(material, mode)
    if not isinstance(block, dict):
        return None
    val = block.get(param)
    if _is_param_block(val):
        return val
    return None


def summarize_priors(material: str, mode: str = "") -> dict[str, Any]:
    """Return a compact, JSON-safe summary of the priors for *material*/*mode*.

    Shape::

        {"material": ..., "mode": ..., "n_papers": int,
         "params": {"bias_v": {"p25":…, "p50":…, "p75":…, "n":…}, ...}}

    For an empty *mode*, the first available mode of the material is summarized
    (modes are sorted; "imaging" sorts before "sts"). Returns ``{}`` if nothing
    is found. Never raises — handy for GUI/agent display where a missing prior
    must degrade gracefully rather than throw.
    """
    if not material:
        return {}

    resolved_mode = mode
    if not resolved_mode:
        modes = list_modes(material)
        if not modes:
            return {}
        resolved_mode = modes[0]

    block = get_priors(material, resolved_mode)
    if not isinstance(block, dict):
        return {}

    params: dict[str, Any] = {}
    for key, val in block.items():
        if _is_param_block(val):
            params[key] = {
                "p25": val.get("p25"),
                "p50": val.get("p50"),
                "p75": val.get("p75"),
                "min": val.get("min"),
                "max": val.get("max"),
                "n": val.get("n"),
            }

    return {
        "material": material,
        "mode": resolved_mode,
        "n_papers": block.get("n_papers", 0),
        "params": params,
    }


def _set_priors_path_for_test(path: Path | None) -> None:
    """Override the priors file path and invalidate the cache (TEST ONLY).

    Lets tests point at a synthetic ``literature_priors.json`` without touching
    the shipped one. Pass ``None`` to restore the default.
    """
    global _PRIORS_PATH
    with _lock:
        if path is None:
            _PRIORS_PATH = _HERE.parent / "literature_priors.json"
        else:
            _PRIORS_PATH = Path(path)
        _cache["data"] = None
        _cache["loaded"] = False
        _cache["mtime"] = None


__all__ = [
    "get_priors",
    "list_materials",
    "list_modes",
    "get_param",
    "summarize_priors",
]
