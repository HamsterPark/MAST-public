"""Build-time defaults for the MAST push update server URL + shared token +
release signing public key.

Mirrors v1 mast.update.defaults; values live in `_defaults.py` (generated
by installer/mast2_build.ps1 from installer/push_defaults.json or a
mast2 equivalent). Missing module → empty defaults (manual config required).

``DEFAULT_SIGNING_PUBKEY`` is the Ed25519 release public key (32-byte raw hex)
baked into the client so it can verify the OTA manifest signature. Empty until
the admin runs ``python -m mast.update keygen`` and puts the hex in the build's
push-defaults JSON (see mast.update.signing).
"""

from __future__ import annotations

try:
    from mast.update._defaults import DEFAULT_SERVER_URL, DEFAULT_TOKEN
except ImportError:
    DEFAULT_SERVER_URL = ""
    DEFAULT_TOKEN = ""

try:
    from mast.update._defaults import DEFAULT_SIGNING_PUBKEY  # type: ignore
except ImportError:
    DEFAULT_SIGNING_PUBKEY = ""


def get_default_server_url() -> str:
    return DEFAULT_SERVER_URL or ""


def get_default_token() -> str:
    return DEFAULT_TOKEN or ""


def get_default_signing_pubkey() -> str:
    return DEFAULT_SIGNING_PUBKEY or ""
