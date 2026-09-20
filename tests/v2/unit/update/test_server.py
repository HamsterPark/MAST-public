"""Push-update server endpoint tests — focus: /download serves published deltas
(OTA P0 fix #30). Before the fix /download 404'd anything but the full installer,
so incremental delta downloads could never succeed.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mast.update.manifest import Manifest, write_manifest  # noqa: E402
from mast.update.server import build_app, push_dir, write_token  # noqa: E402


def _setup(tmp_path):
    write_token(tmp_path, "tok")
    pd = push_dir(tmp_path)
    (pd / "MAST2_Setup_2.2.3.exe").write_bytes(b"INSTALLER-BYTES")
    (pd / "delta_2.2.2_to_2.2.3.zip").write_bytes(b"DELTA-ZIP-BYTES")
    m = Manifest(
        version="2.2.3", filename="MAST2_Setup_2.2.3.exe",
        sha256="x" * 64, size_bytes=15, published_at="2026-06-02T00:00:00",
        deltas=[{"from_version": "2.2.2",
                 "filename": "delta_2.2.2_to_2.2.3.zip",
                 "sha256": "y" * 64, "size_bytes": 14}],
    )
    write_manifest(pd / "manifest.json", m)
    return TestClient(build_app(tmp_path)), {"Authorization": "Bearer tok"}


def test_download_serves_full_installer(tmp_path):
    c, h = _setup(tmp_path)
    r = c.get("/download/MAST2_Setup_2.2.3.exe", headers=h)
    assert r.status_code == 200 and r.content == b"INSTALLER-BYTES"


def test_download_serves_published_delta(tmp_path):
    """The fix: a filename listed in manifest.deltas[] now downloads (was 404)."""
    c, h = _setup(tmp_path)
    r = c.get("/download/delta_2.2.2_to_2.2.3.zip", headers=h)
    assert r.status_code == 200 and r.content == b"DELTA-ZIP-BYTES"


def test_download_rejects_unpublished_filename(tmp_path):
    c, h = _setup(tmp_path)
    assert c.get("/download/evil.zip", headers=h).status_code == 404


def test_download_rejects_path_traversal(tmp_path):
    c, h = _setup(tmp_path)
    # backslash/forward-slash/leading-dot are rejected at 400 before any lookup
    assert c.get("/download/..%2Fsecret", headers=h).status_code in (400, 404)


def test_download_requires_auth(tmp_path):
    c, _ = _setup(tmp_path)
    assert c.get("/download/delta_2.2.2_to_2.2.3.zip").status_code == 401


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
