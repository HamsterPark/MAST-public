"""Ed25519 release signing for the OTA manifest — closes the supply-chain
authenticity gap documented in ``client.py`` (a Bearer token proves
AUTHORIZATION and sha256 proves INTEGRITY-vs-corruption, but nothing proved the
manifest actually came from us — a MITM / compromised push server could ship
malicious bytes with a matching hash → RCE).

Chain:
  1. Admin runs ``python -m mast.update keygen`` ONCE → an Ed25519 keypair. The
     PRIVATE key stays OFFLINE with the release signer (never on the push
     server); the PUBLIC key (32-byte hex) is baked into every client build via
     ``_defaults.DEFAULT_SIGNING_PUBKEY``.
  2. At publish, the canonicalised manifest bytes are signed → ``manifest.sig``
     ships next to ``manifest.json``.
  3. The client fetches both and, when a public key is embedded, VERIFIES the
     signature over the manifest BEFORE trusting any field (version, filename,
     sha256, deltas). Only then do the existing sha256/size checks mean anything.

Canonicalisation: ``json.dumps(sort_keys=True, separators=(",", ":"))`` over the
manifest dict with any ``signature`` key removed, UTF-8 encoded. Deterministic so
signer and verifier hash identical bytes regardless of key order / whitespace.

Stdlib-only surface; the actual crypto needs ``cryptography`` (pinned in
requirements-v2.txt, bundled via mast2.spec). If it is unavailable every function
degrades safely: signing raises, verification returns False (never a silent pass).
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    HAVE_CRYPTO = True
except Exception:  # noqa: BLE001 — module must import even without cryptography
    HAVE_CRYPTO = False


def canonical_manifest_bytes(manifest: dict) -> bytes:
    """Deterministic bytes to sign/verify: the manifest dict minus any signature
    field, serialised with sorted keys and no whitespace, UTF-8."""
    payload = {k: v for k, v in (manifest or {}).items()
               if k not in ("signature", "sig")}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def generate_keypair() -> tuple[str, str]:
    """Create a fresh Ed25519 keypair. Returns (private_key_PEM, public_key_HEX).
    The admin keeps the PEM offline and bakes the hex into client builds."""
    if not HAVE_CRYPTO:
        raise RuntimeError("cryptography is required for keygen (pip install cryptography>=43)")
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    pub_raw = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return pem, pub_raw.hex()


def public_key_from_private_pem(private_key_pem: str) -> str:
    """The public key HEX that belongs to this private PEM.

    Added 2026-08-20 for skill packs. The packing tool self-checks a freshly
    signed pack twice, and the two checks answer different questions:

    * verify with THIS key  → "did signing itself work?" (a bug in the packer)
    * verify with the BAKED-IN release key → "will any client accept it?"

    Without the first, a mismatched key looks exactly like a broken packer.
    Without the second, you ship a pack that every machine refuses — and the
    refusal happens far away from the mistake.

    Verification paths are untouched: this only derives, it never verifies.
    """
    if not HAVE_CRYPTO:
        raise RuntimeError("cryptography is required")
    key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("key is not an Ed25519 private key")
    return key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def sign_manifest(manifest: dict, private_key_pem: str) -> str:
    """Sign the canonical manifest bytes with the offline private key (PEM).
    Returns the signature as hex. Raises if cryptography/key is unavailable."""
    if not HAVE_CRYPTO:
        raise RuntimeError("cryptography is required for signing")
    key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("signing key is not an Ed25519 private key")
    return key.sign(canonical_manifest_bytes(manifest)).hex()


def verify_manifest(manifest: dict, signature_hex: str, public_key_hex: str) -> bool:
    """True iff *signature_hex* is a valid Ed25519 signature over the canonical
    manifest bytes for *public_key_hex* (32-byte raw hex). NEVER raises — returns
    False on any error (missing crypto, bad key/sig, tamper), never a silent pass."""
    if not HAVE_CRYPTO:
        logger.warning("manifest signature NOT verified — cryptography unavailable")
        return False
    if not signature_hex or not public_key_hex:
        return False
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        pub.verify(bytes.fromhex(signature_hex), canonical_manifest_bytes(manifest))
        return True
    except (InvalidSignature, ValueError, TypeError) as exc:
        logger.warning("manifest signature verification FAILED: %s", type(exc).__name__)
        return False
    except Exception as exc:  # noqa: BLE001 — any crypto error = not verified
        logger.warning("manifest signature verify error: %s", exc)
        return False


def get_release_public_key() -> str:
    """The build-embedded release public key hex, or '' when not provisioned
    (transition builds: the client then skips verification with a warning)."""
    try:
        from mast.update.defaults import get_default_signing_pubkey
        return get_default_signing_pubkey() or ""
    except Exception:  # noqa: BLE001
        return ""


__all__ = [
    "HAVE_CRYPTO", "canonical_manifest_bytes", "generate_keypair",
    "public_key_from_private_pem",
    "sign_manifest", "verify_manifest", "get_release_public_key",
]
