"""The ONE DashScope key resolver for the knowledge layer.

(): ``literature_index`` and ``ingest`` each carried
their own copy of a key loader that resolved ``api key/dashscope.env`` off the
INDEX root. That root is deliberately the executable's directory in a frozen
build (that is where the 205 MB index ships), but the key is not there: the Inno
installer writes a ``data_dir.txt`` naming a user-chosen data root, the launcher
exports it as ``MAST2_PROJECT_ROOT``, and ``api key/`` lives under THAT. So a
normal install found no key, every literature search silently degraded to
keyword matching — 68 times in one session — and the agent read the resulting
noise as relevant papers.

Resolution order (first hit wins), and there is no third place to look:

1. ``DASHSCOPE_API_KEY`` / ``ALIYUN_BAILIAN_API_KEY`` env vars.
2. ``<project root>/api key/dashscope.env`` resolved **lazily**, i.e. honouring
   whatever ``MAST2_PROJECT_ROOT`` says right now.

Step 2 goes through ``config._api_key_dir()`` rather than the more obvious
``config._load_provider_key``, because that function builds its per-provider
file map from a module-level ``_API_KEY_DIR`` captured at IMPORT time: a root
set after ``mast.config`` was first imported is ignored. Production survives
that only because the launcher exports the env var before importing ``mast`` —
an ordering constraint, not a guarantee. The lazy resolver is correct
regardless of import order while still reusing ``config``'s own directory logic
and key-file format.

Deliberately NO fallback to the import-frozen path when step 2 comes up empty.
A second place to find a key is what caused this bug in the first place: it
turns "the key is not where this install puts it" into "the key was found
somewhere else", which is precisely the failure that stayed invisible for 68
searches. If step 2 has no key, there is no key.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ENV_VARS = ("DASHSCOPE_API_KEY", "ALIYUN_BAILIAN_API_KEY")


def load_dashscope_key() -> str:
    """Return the DashScope key, or ``""`` when none is provisioned."""
    for var in _ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val:
            return val

    try:
        from mast.config import _api_key_dir, _read_key_file
    except Exception as exc:  # pragma: no cover - config import guard
        logger.warning("DashScope key: config import failed: %s", exc)
        return ""

    try:
        return (_read_key_file(_api_key_dir() / "dashscope.env") or "").strip()
    except Exception as exc:  # pragma: no cover - unreadable path
        logger.debug("DashScope key: read failed: %s", exc)
        return ""


def dashscope_key_path_hint() -> str:
    """Where an operator should PUT the key — named in every "no key" message.

    The old error said only "write api key/dashscope.env", relative to nothing;
    on the machine that degraded 68 times, the directory it meant and the
    directory the installer created were different.
    """
    try:
        from mast.config import _api_key_dir

        return str(_api_key_dir() / "dashscope.env")
    except Exception:  # pragma: no cover - defensive
        return "<project root>/api key/dashscope.env"


def require_dashscope_key() -> str:
    """:func:`load_dashscope_key`, raising ``RuntimeError`` when absent."""
    key = load_dashscope_key()
    if key:
        return key
    raise RuntimeError(
        "No DashScope key found. Set DASHSCOPE_API_KEY or write the key into "
        f"{dashscope_key_path_hint()}."
    )


__all__ = ["load_dashscope_key", "require_dashscope_key", "dashscope_key_path_hint"]
