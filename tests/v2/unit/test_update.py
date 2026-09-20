"""Smoke test for mast.update — manifest schema, sha256, version compare."""

from __future__ import annotations

# ── path bootstrap (canonical block)
import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

# Module-level imports so the cached `mast` becomes v2 before pytest swaps it
from mast.update.manifest import (
    Manifest,
    compute_sha256,
    is_newer,
    load_manifest,
    make_manifest,
    write_manifest,
)
from mast.update.defaults import get_default_server_url, get_default_token
from mast.core.registry import SkillRegistry
from mast.skills.composite.demo_scan_and_sts import DemoScanAndSTS
from mast.skills.builtins.tip_shaper import TipShape


def test_manifest_roundtrip(tmp_path):
    setup = tmp_path / "MAST2-Setup-v2.0.0-windows-x64.exe"
    setup.write_bytes(b"x" * 1024)

    m = make_manifest(setup, version="2.0.0")
    assert m.version == "2.0.0"
    assert m.filename == setup.name
    assert m.size_bytes == 1024
    assert len(m.sha256) == 64
    assert m.sha256 == compute_sha256(setup)

    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, m)
    loaded = load_manifest(manifest_path)
    assert loaded is not None
    assert loaded.to_dict() == m.to_dict()


def test_manifest_forward_compat():
    """Unknown fields in the JSON should be ignored (forward compat)."""
    data = {
        "version": "3.0.0",
        "filename": "MAST2-Setup-v3.0.0.exe",
        "sha256": "abc",
        "size_bytes": 100,
        "published_at": "2026-05-13T12:00:00+08:00",
        "future_field": "added in v5",
    }
    m = Manifest.from_dict(data)
    assert m.version == "3.0.0"
    assert m.filename == "MAST2-Setup-v3.0.0.exe"


def test_is_newer_version():
    assert is_newer("2.0.1", "2.0.0")
    assert is_newer("3.0.0", "2.9.9")
    assert is_newer("2.1.0", "2.0.99")
    assert not is_newer("2.0.0", "2.0.0")
    assert not is_newer("1.9.0", "2.0.0")
    assert is_newer("2.0.1-rc1", "2.0.0")


def test_defaults_empty_by_default():
    assert isinstance(get_default_server_url(), str)
    assert isinstance(get_default_token(), str)


def test_demo_scan_and_sts_metadata():
    """DemoScanAndSTS exposes the v0.3.11 composite skill metadata."""
    md = DemoScanAndSTS().metadata()
    assert md.name == "DemoScanAndSTS"
    assert md.safety_level.name == "AUTO"
    assert "demo" in md.tags
    required_params = [p for p in md.parameters if p.required]
    assert required_params == [], "DemoScanAndSTS should not have any required params"


def test_demo_scan_and_sts_registered():
    """SkillRegistry.discover picks up DemoScanAndSTS."""
    r = SkillRegistry()
    r.discover("mast.skills.composite")
    names = {m.name for m in r.list_skills()}
    assert "DemoScanAndSTS" in names


def test_tip_shape_safety_bounds():
    """TipShape tip_lift_m / lift_height_m capped at ±1e-7 (±100 nm).

    Tightened from ±1e-6 (±1 µm, 500× a normal poke) in the 2026-07-03 review;
    a global tip_lift check in mast.core.safety backs it up."""
    md = TipShape().metadata()
    by_name = {p.name: p for p in md.parameters}
    for name in ("tip_lift_m", "lift_height_m"):
        ps = by_name[name]
        assert ps.min_value == -1e-7, f"{name} min_value mismatch"
        assert ps.max_value == 1e-7, f"{name} max_value mismatch"
