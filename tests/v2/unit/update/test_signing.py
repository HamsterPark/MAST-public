"""OTA manifest signing (Ed25519) — mast.update.signing.

Closes the supply-chain authenticity gap: the client verifies an Ed25519
signature over the manifest against an embedded public key before trusting any
field. Non-crypto tests (canonicalisation, degradation, no-key gating) run
always; the sign/verify round-trip is skipped when `cryptography` is absent (a
pinned dep + bundled in the app, so it runs in a provisioned venv / the frozen
build).
"""
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.update import signing as S  # noqa: E402

MANIFEST = {"version": "4.5.0", "filename": "MAST-Setup-v4.5.0.exe",
            "sha256": "ab" * 32, "size_bytes": 123, "deltas": [],
            "published_at": "2026-07-02T00:00:00"}


# ── canonicalisation (no crypto) ─────────────────────────────────────────────

def test_canonical_bytes_key_order_independent_and_compact():
    a = S.canonical_manifest_bytes({"b": 1, "a": 2})
    b = S.canonical_manifest_bytes({"a": 2, "b": 1})
    assert a == b                    # sort_keys → order-independent
    assert b" " not in a             # compact separators (no whitespace)


def test_canonical_bytes_excludes_signature_field():
    base = S.canonical_manifest_bytes(MANIFEST)
    with_sig = S.canonical_manifest_bytes({**MANIFEST, "signature": "deadbeef"})
    assert base == with_sig          # the sig is never part of the signed bytes


def test_get_release_public_key_is_str():
    # Dev build: no baked key → '' (client then runs in transition/skip mode).
    assert isinstance(S.get_release_public_key(), str)


def test_verify_returns_false_on_missing_inputs():
    assert S.verify_manifest(MANIFEST, "", "abcd") is False
    assert S.verify_manifest(MANIFEST, "abcd", "") is False
    # never raises, never silently passes
    assert S.verify_manifest(MANIFEST, "zz", "zz") is False


# ── real Ed25519 round-trip (needs cryptography) ─────────────────────────────

@pytest.mark.skipif(not S.HAVE_CRYPTO, reason="cryptography not installed")
def test_sign_verify_roundtrip():
    pem, pub = S.generate_keypair()
    sig = S.sign_manifest(MANIFEST, pem)
    assert S.verify_manifest(MANIFEST, sig, pub) is True


@pytest.mark.skipif(not S.HAVE_CRYPTO, reason="cryptography not installed")
def test_tampered_manifest_fails_verification():
    pem, pub = S.generate_keypair()
    sig = S.sign_manifest(MANIFEST, pem)
    assert S.verify_manifest({**MANIFEST, "filename": "evil.exe"}, sig, pub) is False
    assert S.verify_manifest({**MANIFEST, "sha256": "00" * 32}, sig, pub) is False


@pytest.mark.skipif(not S.HAVE_CRYPTO, reason="cryptography not installed")
def test_wrong_key_fails_verification():
    pem, _pub = S.generate_keypair()
    _pem2, pub2 = S.generate_keypair()
    sig = S.sign_manifest(MANIFEST, pem)
    assert S.verify_manifest(MANIFEST, sig, pub2) is False


@pytest.mark.skipif(not S.HAVE_CRYPTO, reason="cryptography not installed")
def test_verification_survives_key_reordering():
    # The client verifies over the RAW received dict; canonicalisation must make
    # a differently-ordered-but-equal manifest still verify.
    pem, pub = S.generate_keypair()
    sig = S.sign_manifest(MANIFEST, pem)
    reordered = dict(reversed(list(MANIFEST.items())))
    assert S.verify_manifest(reordered, sig, pub) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
