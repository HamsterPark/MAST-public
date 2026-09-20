"""CAS sha256 ingest + verify."""
import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
while _MASTV2_ROOT in sys.path:
    sys.path.remove(_MASTV2_ROOT)
sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import os

import pytest

from mast.logging.v2.cas import CASStore, sha256_file


@pytest.fixture
def store(tmp_path):
    return CASStore(tmp_path / "cas")


def test_sha256_file_streaming(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world" * 1000)
    digest, size = sha256_file(p)
    assert size == len("hello world") * 1000
    # Stable digest
    again, _ = sha256_file(p)
    assert digest == again


def test_ingest_copy_deduplicates(tmp_path, store):
    a = tmp_path / "a.bin"
    a.write_bytes(b"payload-A")
    e1 = store.ingest(a, mode="copy")
    e2 = store.ingest(a, mode="copy")
    assert e1.sha256 == e2.sha256
    assert e1.cas_path == e2.cas_path
    # File is laid out as <root>/<aa>/<bb>/<sha256>
    assert e1.cas_path.parent.name == e1.sha256[2:4]
    assert e1.cas_path.parent.parent.name == e1.sha256[:2]


def test_verify(tmp_path, store):
    a = tmp_path / "a.bin"
    a.write_bytes(b"verify-me")
    e = store.ingest(a)
    assert store.verify(e.sha256)
    # Tampering invalidates
    e.cas_path.write_bytes(b"tampered")
    assert not store.verify(e.sha256)


def test_verify_unknown_returns_false(store):
    assert not store.verify("0" * 64)


def test_ingest_verify_mode_returns_no_copy(tmp_path, store):
    a = tmp_path / "a.bin"
    a.write_bytes(b"verify-only")
    e = store.ingest(a, mode="verify")
    assert e.cas_path == a
    assert not store.cas_path(e.sha256).exists()


def test_iter_all_lists_ingested(tmp_path, store):
    for i in range(3):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(f"content-{i}".encode())
        store.ingest(p)
    found = list(store.iter_all())
    assert len(found) == 3
