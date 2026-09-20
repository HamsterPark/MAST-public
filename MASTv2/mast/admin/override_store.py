"""ConfigOverrideRegistry — JSON override layer for MAST configuration.

vendored from v1 mast/admin/override_store.py 2026-04-23.

Loads, saves, and merges JSON overrides on top of code defaults. Each override
file sits in ``config/overrides/`` and only contains the fields that differ
from code defaults. Every save creates a timestamped backup in
``config/overrides/_history/`` for undo support.

WHERE THIS DIRECTORY LIVES (KNOWN_ISSUES §3.2, fixed 2026-08-04)
================================================================
It used to be resolved ``Path(__file__).resolve().parents[3] / "config" /
"overrides"``. In a frozen build ``__file__`` is
``C:\\MAST\\_internal\\mast\\admin\\override_store.py``, so that walk landed on
**``C:\\MAST\\config\\overrides\\`` — inside the INSTALL directory**, which every
upgrade overwrites. The operator's own safety envelope was living in a folder
the installer copies files into.

It survived on a coincidence, not a design: ``mast2_setup.iss`` never cleared
the target dir, and ``dist\\MAST\\config\\`` happened to contain no
``overrides\\``. That coincidence has already failed once — on 2026-08-03 a
post-build smoke run from ``dist\\MAST`` left a test ``safety_limits.json``
(±2.52 µm) inside the build artifact; one more packaging run and it would have
shipped to every machine and silently replaced the operator's envelope.
``mast2_build.ps1`` Step 5.9 now blocks that specific dragging, but the shape —
user data inside an upgradeable directory — is what actually needed fixing.

Now it resolves through :func:`mast._runtime_paths.project_root`, which honours
``MAST2_PROJECT_ROOT`` — the launcher exports it from ``data_dir.txt``, i.e. the
user data root (``D:\\MAST-data``), the same side as the experiment library. In
a dev checkout both expressions give the same repo root, so nothing moves.

MIGRATION: an existing install already has real, hand-measured limits in the old
location, and quietly reverting a machine to the ±1.5 µm factory envelope
because its file "wasn't there any more" would be the worst possible way to fix
a data-location bug. So the first construction against an empty target copies
any legacy files across and logs it at WARNING. See ``_migrate_legacy_dir``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The directory the code walk USED to produce. In a dev checkout it equals the
# repo root; in a frozen build it is the INSTALL directory (C:\MAST) — which is
# exactly why it is now only the legacy/migration source, never the target.
_LEGACY_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_DIR = _LEGACY_ROOT / "config" / "overrides"


def _default_dir() -> Path:
    """The user-data-side ``config/overrides``. Resolved per call, not at import.

    Per call because ``MAST2_PROJECT_ROOT`` is exported by the launcher and by
    test fixtures, and a module-level constant would freeze whichever value
    happened to exist at import time. That is a failure this repo has paid for
    more than once (``huggingface_hub`` caches, the two literature key dirs) —
    "the env var works, but only if you set it before importing" is a rule
    nobody remembers at the call site.
    """
    try:
        from mast._runtime_paths import project_root

        return project_root() / "config" / "overrides"
    except Exception:  # noqa: BLE001 — never let path resolution break admin config
        logger.warning("project_root() unavailable; overrides fall back to %s",
                       _LEGACY_DIR)
        return _LEGACY_DIR

# Known override filenames
SAFETY_LIMITS = "safety_limits.json"
SAFETY_CHECKS = "safety_checks.json"
SAFETY_CONSTRAINTS = "safety_constraints.json"
SKILL_OVERRIDES = "skill_overrides.json"
KNOWLEDGE_OVERRIDES = "knowledge_overrides.json"
FAULT_DIAGNOSIS_OVERRIDES = "fault_diagnosis_overrides.json"
GUIDANCE_OVERRIDES = "guidance_overrides.json"
ENCYCLOPEDIA_OVERRIDES = "encyclopedia_overrides.json"
QUICK_PROMPTS = "quick_prompts.json"
AGENT_OVERRIDES = "agent_overrides.json"
PROMPT_OVERRIDES = "prompt_overrides.json"

# NOTE: a file missing from this list is loaded by NOTHING at startup — save()
# writes it, the next boot silently ignores it. Same failure shape as
# SettingsStore.KNOWN_KEYS. Adding a new override file means adding it HERE.
_ALL_FILES = [
    SAFETY_LIMITS,
    SAFETY_CHECKS,
    SAFETY_CONSTRAINTS,
    SKILL_OVERRIDES,
    KNOWLEDGE_OVERRIDES,
    FAULT_DIAGNOSIS_OVERRIDES,
    GUIDANCE_OVERRIDES,
    ENCYCLOPEDIA_OVERRIDES,
    QUICK_PROMPTS,
    AGENT_OVERRIDES,
    PROMPT_OVERRIDES,
]


class ConfigOverrideRegistry:
    """Singleton registry that manages JSON override files."""

    _instance: ConfigOverrideRegistry | None = None

    def __init__(self, overrides_dir: Path | None = None):
        explicit = overrides_dir is not None
        self._dir = Path(overrides_dir) if explicit else _default_dir()
        self._history_dir = self._dir / "_history"
        self._cache: dict[str, dict] = {}
        # Hot-reload subscribers. Consumers that hold derived/cached state built
        # from these overrides (e.g. SkillRegistry's per-skill metadata cache,
        # SafetyGate's resolved checks) register a no-arg callback here so a
        # 'save override' button can refresh them in-process without a restart.
        self._reload_hooks: list[Callable[[], None]] = []
        self._dir.mkdir(parents=True, exist_ok=True)
        if not explicit:
            self._migrate_legacy_dir()
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._load_all()

    @classmethod
    def get(cls, overrides_dir: Path | None = None) -> ConfigOverrideRegistry:
        """Return the singleton instance (created on first call)."""
        if cls._instance is None:
            cls._instance = cls(overrides_dir)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # ── Legacy location migration (KNOWN_ISSUES §3.2) ─────────────────

    def _migrate_legacy_dir(self) -> None:
        """Carry pre-2026-08-04 overrides across from the install directory.

        The move of ``config/overrides`` out of the install dir must not cost a
        rig its measured safety envelope. A machine upgrading from v6.0.2 has
        real numbers in ``C:\\MAST\\config\\overrides\\safety_limits.json``;
        finding an empty new directory and falling back to the ±1.5 µm factory
        placeholder would be a SILENT WIDENING of the envelope on a live rig —
        strictly worse than the bug being fixed.

        Rules, all deliberately conservative:

        * only files in ``_ALL_FILES`` (never ``_history/``, never strays);
        * never overwrite a file that already exists at the target;
        * COPY, not move — the legacy copy stays put as evidence, and a rollback
          to the previous MAST build still finds its config;
        * a failure is logged and swallowed. A migration that cannot run must
          not stop the process from starting.
        """
        try:
            legacy = _LEGACY_DIR
            if legacy.resolve() == self._dir.resolve() or not legacy.is_dir():
                return
            if any((self._dir / f).exists() for f in _ALL_FILES):
                return  # target already populated — nothing to decide
            moved: list[str] = []
            for fname in _ALL_FILES:
                src = legacy / fname
                if src.is_file():
                    shutil.copy2(src, self._dir / fname)
                    moved.append(fname)
            if moved:
                logger.warning(
                    "管理员覆写已从旧位置迁移：%s → %s（%s）。旧文件保留未删 —— "
                    "旧位置在安装目录内，会被升级覆盖（KNOWN_ISSUES §3.2）。",
                    legacy, self._dir, ", ".join(moved),
                )
        except Exception:  # noqa: BLE001 — never block startup on a migration
            logger.exception("override migration from %s failed", _LEGACY_DIR)

    # ── Load / Reload ─────────────────────────────────────────────────

    def _load_all(self) -> None:
        """Read all known JSON files into the in-memory cache."""
        self._cache.clear()
        for fname in _ALL_FILES:
            path = self._dir / fname
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self._cache[fname] = data
                    logger.debug("Loaded override: %s (%d keys)", fname, len(data))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to load %s: %s", fname, exc)

    def reload(self) -> None:
        """Re-read all JSON files from disk into the in-memory cache.

        Clears the cross-process ``.reload`` sentinel if present (a peer process
        — e.g. the v1 admin GUI sharing this directory — may have dropped it to
        ask us to re-read). This refreshes only THIS registry's cache; consumers
        with their own derived caches are refreshed via ``signal_reload`` /
        ``register_reload_hook``.
        """
        self._load_all()
        sentinel = self._dir / ".reload"
        if sentinel.exists():
            sentinel.unlink(missing_ok=True)

    def register_reload_hook(self, callback: Callable[[], None]) -> None:
        """Subscribe a no-arg callback to fire on every ``signal_reload``.

        Lets long-lived holders of derived state (SkillRegistry metadata cache,
        SafetyGate resolved checks) re-apply overrides in-process the moment a
        'save override' button persists a change — so admin GUI edits take
        effect on the running agents without a restart. Idempotent: the same
        callable is not registered twice.
        """
        if callback not in self._reload_hooks:
            self._reload_hooks.append(callback)

    def unregister_reload_hook(self, callback: Callable[[], None]) -> None:
        """Remove a previously registered reload hook (no error if absent)."""
        try:
            self._reload_hooks.remove(callback)
        except ValueError:
            pass

    # ── Save / History ────────────────────────────────────────────────

    def save(self, filename: str, data: dict) -> None:
        """Write *data* to *filename*, creating a history backup first."""
        path = self._dir / filename

        if path.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup = self._history_dir / f"{filename}.{ts}.json"
            shutil.copy2(path, backup)
            logger.info("Backed up %s → %s", filename, backup.name)

        fd, tmp = tempfile.mkstemp(
            dir=str(self._dir), suffix=".tmp", prefix=filename + "."
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        self._cache[filename] = copy.deepcopy(data)
        logger.info("Saved override: %s", filename)

    def delete(self, filename: str) -> None:
        """Delete an override file (revert to code defaults)."""
        path = self._dir / filename
        if path.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup = self._history_dir / f"{filename}.{ts}.json"
            shutil.copy2(path, backup)
            path.unlink()
        self._cache.pop(filename, None)
        logger.info("Deleted override: %s", filename)

    def get_history(self, filename: str) -> list[tuple[str, Path]]:
        prefix = filename + "."
        entries: list[tuple[str, Path]] = []
        for p in self._history_dir.iterdir():
            if p.name.startswith(prefix) and p.suffix == ".json":
                ts = p.stem.replace(filename + ".", "")
                entries.append((ts, p))
        entries.sort(key=lambda x: x[0], reverse=True)
        return entries

    def restore(self, filename: str, timestamp: str) -> dict:
        target = self._history_dir / f"{filename}.{timestamp}.json"
        if not target.exists():
            raise FileNotFoundError(f"History entry not found: {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        self.save(filename, data)
        return data

    def signal_reload(self) -> int:
        """Trigger an in-process hot-reload of everything built on overrides.

        Two effects:

        1. Re-reads all JSON files into this registry's cache so external edits
           (e.g. by a peer v1 GUI sharing the directory) are picked up.
        2. Fires every callback registered via ``register_reload_hook`` so
           consumers with derived caches (SkillRegistry metadata, SafetyGate
           resolved checks) re-apply the overrides immediately.

        Also drops a ``.reload`` file sentinel as a best-effort cross-process
        signal for any *separate* process sharing this directory; an in-process
        consumer does not need it (it is driven by the hooks above).

        Returns the number of hooks that ran WITHOUT raising — so a caller can
        report "hot-reloaded" or "重启后生效" from an observation instead of an
        assumption. **Today that number is 0 in production**: nothing registers a
        hook, so this fires into an empty list. The API layer must not paper over
        that (it did, with a hardcoded ``reloaded=True``, until 2026-08-03).
        """
        self._load_all()
        fired = 0
        for hook in list(self._reload_hooks):
            try:
                hook()
                fired += 1
            except Exception:  # noqa: BLE001 — one bad hook must not block others
                logger.exception("override reload hook failed")
        sentinel = self._dir / ".reload"
        try:
            sentinel.write_text(
                datetime.now(timezone.utc).isoformat(), encoding="utf-8"
            )
        except OSError as exc:
            logger.debug("could not write .reload sentinel: %s", exc)
        return fired

    def save_and_reload(self, filename: str, data: dict) -> int:
        """Save *data* and trigger an in-process hot-reload.

        Vendored from v1 mast/admin/override_store.py:178-181. The v2 admin
        GUI calls this from every 'save override' button (safety limits,
        checks, skill levels, material constraints, knowledge/guidance/
        encyclopedia/per-agent models). Missing it made all those buttons
        raise AttributeError at runtime — the most safety-critical being the
        SafetyGuard limit editor.

        After ``save`` updates this registry's cache, ``signal_reload`` fires
        any registered reload hooks. NOTE: consumers that have NOT registered a
        hook (currently the live SkillRegistry's per-skill metadata cache) will
        keep their pre-save values until the process restarts — callers whose
        change targets such a consumer should tell the user "重启后生效".

        Returns the hook-fire count from ``signal_reload`` so the caller can act
        on that NOTE instead of only reading it.
        """
        self.save(filename, data)
        return self.signal_reload()

    def save_or_delete_and_reload(self, filename: str, data: dict) -> int:
        """Save if *data* is non-empty, delete otherwise, then signal reload.

        Vendored from v1 mast/admin/override_store.py:183-189. Lets a 'reset
        to code defaults' (empty payload) remove the override file instead of
        persisting an empty stub.

        Returns the hook-fire count (see ``signal_reload``).
        """
        if data:
            self.save(filename, data)
        else:
            self.delete(filename)
        return self.signal_reload()

    # ── Raw access ────────────────────────────────────────────────────

    def get_raw(self, filename: str) -> dict:
        return copy.deepcopy(self._cache.get(filename, {}))

    def has_overrides(self, filename: str) -> bool:
        return bool(self._cache.get(filename))

    def override_summary(self) -> dict[str, bool]:
        return {f: self.has_overrides(f) for f in _ALL_FILES}

    # ── Domain-specific accessors ─────────────────────────────────────

    def get_safety_limits(self) -> dict:
        """Return SafetyLimits field overrides (e.g. {"bias_max_v": 5.0})."""
        return self.get_raw(SAFETY_LIMITS)

    def get_safety_checks(self) -> dict:
        """Return _GLOBAL_CHECKS overrides.

        Format::

            {
                "overrides": [{"pattern": ..., "unit": ..., "min_attr": ..., "max_attr": ...}],
                "additions": [...],
                "removals": ["pattern_name", ...]
            }
        """
        return self.get_raw(SAFETY_CHECKS)

    def get_safety_constraints(self) -> dict:
        return self.get_raw(SAFETY_CONSTRAINTS)

    def get_skill_override(self, skill_name: str) -> dict | None:
        data = self._cache.get(SKILL_OVERRIDES, {})
        ovr = data.get(skill_name)
        return copy.deepcopy(ovr) if ovr else None

    def get_all_skill_overrides(self) -> dict:
        return self.get_raw(SKILL_OVERRIDES)

    def get_knowledge_override(self, category_id: str) -> dict | None:
        data = self._cache.get(KNOWLEDGE_OVERRIDES, {})
        ovr = data.get(category_id)
        return copy.deepcopy(ovr) if ovr else None

    def get_all_knowledge_overrides(self) -> dict:
        return self.get_raw(KNOWLEDGE_OVERRIDES)

    def get_fault_diagnosis_overrides(self) -> dict:
        return self.get_raw(FAULT_DIAGNOSIS_OVERRIDES)

    def get_guidance_override(self, skill_name: str) -> dict | None:
        data = self._cache.get(GUIDANCE_OVERRIDES, {})
        extra = data.get("skill_extra", {})
        ovr = extra.get(skill_name)
        return copy.deepcopy(ovr) if ovr else None

    def get_all_guidance_overrides(self) -> dict:
        return self.get_raw(GUIDANCE_OVERRIDES)

    def get_measurement_template_override(self, key: str) -> dict | None:
        data = self._cache.get(GUIDANCE_OVERRIDES, {})
        templates = data.get("measurement_templates", {})
        ovr = templates.get(key)
        return copy.deepcopy(ovr) if ovr else None

    def get_encyclopedia_overrides(self) -> dict:
        return self.get_raw(ENCYCLOPEDIA_OVERRIDES)

    def get_quick_prompts(self) -> list[dict]:
        data = self._cache.get(QUICK_PROMPTS, {})
        return data.get("prompts", [])

    # ── Per-agent model / thinking overrides (Agents tab) ─────────────

    def get_agent_overrides(self) -> dict:
        """Return all per-agent overrides: ``{agent_id: {model, thinking}}``."""
        return self.get_raw(AGENT_OVERRIDES)

    def get_agent_override(self, agent_id: str) -> dict | None:
        """Return one agent's ``{model, thinking}`` override, or None."""
        data = self._cache.get(AGENT_OVERRIDES, {})
        ovr = data.get(agent_id)
        return copy.deepcopy(ovr) if ovr else None

    def set_agent_override(
        self, agent_id: str, *, model: str | None = None,
        thinking: str | None = None,
    ) -> dict:
        """Merge a model / thinking override for one agent and persist it.

        Used by the Agents tab's per-agent LLM picker. Only the provided
        fields are updated; the override file is the source of truth read
        back by ``GET /agents/snapshot``.
        """
        data = self.get_raw(AGENT_OVERRIDES)
        entry = dict(data.get(agent_id, {}))
        if model is not None:
            entry["model"] = model
        if thinking is not None:
            entry["thinking"] = thinking
        data[agent_id] = entry
        self.save(AGENT_OVERRIDES, data)
        return entry


# ── Merge helpers ─────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def apply_skill_metadata_override(meta: Any, override: dict) -> Any:
    """Apply override dict to a SkillMetadata dataclass (returns new instance).

    Vendored from v1 mast/admin/override_store.py:318-374. Used by SkillRegistry
    when admin GUI has saved per-skill overrides via skill_overrides.json.

    Supported override keys:
      - safety_level: str ("auto"/"confirm"/"dangerous")
      - parameters: {param_name: {field: value, ...}}
      - preconditions / postconditions: [str, ...]
      - rollback_skill: str | None
      - estimated_duration_s: float
      - description: str
    """
    from dataclasses import replace

    from mast.core.types import ParameterSpec, SafetyLevel

    kwargs: dict[str, Any] = {}

    if "safety_level" in override:
        level_str = override["safety_level"].upper()
        kwargs["safety_level"] = SafetyLevel[level_str]

    for field in ("rollback_skill", "estimated_duration_s", "description"):
        if field in override:
            kwargs[field] = override[field]

    for field in ("preconditions", "postconditions"):
        if field in override:
            kwargs[field] = list(override[field])

    if "parameters" in override:
        param_overrides = override["parameters"]
        new_params: list[ParameterSpec] = []
        for spec in meta.parameters:
            if spec.name in param_overrides:
                po = param_overrides[spec.name]
                spec_kwargs: dict[str, Any] = {}
                for attr in (
                    "min_value", "max_value", "default",
                    "required", "description", "unit",
                    "allowed_values", "type",
                ):
                    if attr in po:
                        spec_kwargs[attr] = po[attr]
                spec = replace(spec, **spec_kwargs)
            new_params.append(spec)
        kwargs["parameters"] = new_params

    if kwargs:
        return replace(meta, **kwargs)
    return meta


__all__ = [
    "ConfigOverrideRegistry",
    "deep_merge",
    "apply_skill_metadata_override",
    "SAFETY_LIMITS", "SAFETY_CHECKS", "SAFETY_CONSTRAINTS",
    "SKILL_OVERRIDES", "KNOWLEDGE_OVERRIDES",
    "FAULT_DIAGNOSIS_OVERRIDES", "GUIDANCE_OVERRIDES",
    "ENCYCLOPEDIA_OVERRIDES", "QUICK_PROMPTS", "AGENT_OVERRIDES",
]
