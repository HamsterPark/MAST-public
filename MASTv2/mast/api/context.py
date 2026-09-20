"""Runtime context for the API process — holds optional references to the live
core subsystems.

Two run modes (strangler-fig transition):

* **Standalone dev** (``python -m mast.api``): nothing heavy is wired. Pure
  config endpoints serve real data; the settings store points at
  ``<cwd>/config``; the skill registry / experiment storage are absent and
  their endpoints return ``degraded=True`` (empty but never broken).

* **Mounted next to the live app** (Phase 3+): ``app.py`` calls
  :meth:`AppContext.wire` to share the already-constructed singletons
  (SkillRegistry, ExperimentStorage, …) in-process — zero data copy.

Everything optional is fetched through this context so a missing subsystem
degrades gracefully instead of 500-ing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AppContext:
    def __init__(self, user_root: Optional[str] = None) -> None:
        self.user_root = user_root
        self._skill_registry: Any = None
        self._experiment_storage: Any = None
        self._settings_store: Any = None
        self._buffer_service: Any = None

    # ── wiring (called by app.py in mounted mode) ──────────────────────
    def wire(
        self,
        *,
        skill_registry: Any = None,
        experiment_storage: Any = None,
        settings_store: Any = None,
        buffer_service: Any = None,
    ) -> None:
        if skill_registry is not None:
            self._skill_registry = skill_registry
        if experiment_storage is not None:
            self._experiment_storage = experiment_storage
        if settings_store is not None:
            self._settings_store = settings_store
        if buffer_service is not None:
            self._buffer_service = buffer_service

    # ── accessors (graceful fallback) ──────────────────────────────────
    @property
    def skill_registry(self) -> Any:
        return self._skill_registry

    @property
    def experiment_storage(self) -> Any:
        return self._experiment_storage

    @property
    def buffer_service(self) -> Any:
        """The live BufferService (vision/scan realtime streams). None in
        standalone dev → realtime endpoints degrade to empty snapshots + pings."""
        return self._buffer_service

    @property
    def settings_store(self) -> Any:
        """A SettingsStore — lazily created against ``<user_root|cwd>/config`` in
        standalone mode. Reuses the existing pure-stdlib store; TODO(Phase 5):
        relocate SettingsStore out of ``mast.webui`` so the API has no gui dep."""
        if self._settings_store is None:
            try:
                from mast.webui.settings_store import SettingsStore, default_config_dir

                self._settings_store = SettingsStore(default_config_dir(self.user_root))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("settings store init failed: %s", exc)
        return self._settings_store


# Process-wide singleton. ``create_app`` installs it on ``app.state`` and routes
# read it from there; this module-level handle is the standalone default.
_CONTEXT: Optional[AppContext] = None


def get_context() -> AppContext:
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = AppContext()
    return _CONTEXT


def set_context(ctx: AppContext) -> None:
    global _CONTEXT
    _CONTEXT = ctx


__all__ = ["AppContext", "get_context", "set_context"]
