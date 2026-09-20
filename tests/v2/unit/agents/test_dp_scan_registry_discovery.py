"""Feedback ①: data-path single source of truth.

The 2026 field run (72e18bf1 s51-195) had the LLM盲猜 directory+filename combos
and load_scan-fail 64× because the IC save path never reached DP as a structured
product, and DP had only a "load this exact path" primitive — no way to look a
scan up by identity or SEE what was on disk.

Pins:
  * scan_registry keeps a scan_id→record map: record_scan / get_scan /
    resolve_scan_id / list_scans, and record_scan_path auto-registers by stem.
  * load_scan(scan_id=…) resolves via the registry — no exact path needed.
  * list_scan_dir / glob_scans surface files in ONE call and register every hit
    so the follow-up is load_scan(scan_id=…), not another guess.
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np
import pytest

from mast.core import scan_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    scan_registry.clear()
    yield
    scan_registry.clear()


def _make_sxm(path: Path, nx: int = 8, ny: int = 8) -> Path:
    """Minimal but REAL .sxm: header + \\x1a\\x04 marker + one 'both' Z channel."""
    header = (
        ":SCAN_PIXELS:\n"
        f"{nx} {ny}\n"
        ":SCAN_OFFSET:\n"
        "-8.53E-8 9.244E-8\n"
        ":SCAN_RANGE:\n"
        "5E-8 5E-8\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.0\t0.0\n"
        "\n"
        ":SCANIT_END:\n"
    )
    fwd = np.arange(nx * ny, dtype=">f4")
    bwd = fwd[::-1].astype(">f4")
    path.write_bytes(header.encode() + b"\x1a\x04" + fwd.tobytes() + bwd.tobytes())
    return path


# ─────────────────────────────────────────────────────────────────────
# scan_registry scan_id map
# ─────────────────────────────────────────────────────────────────────

class TestScanIdRegistry:
    def test_record_scan_path_registers_by_stem(self, tmp_path):
        p = tmp_path / "Au111_mica_001.sxm"
        p.write_bytes(b"x")
        scan_registry.record_scan_path(p)
        assert scan_registry.resolve_scan_id("Au111_mica_001") == str(p)

    def test_trailing_underscore_id_resolves(self, tmp_path):
        # Nanonis often appends a trailing '_' to the saved stem.
        p = tmp_path / "Au111_mica_001_.sxm"
        p.write_bytes(b"x")
        scan_registry.record_scan_path(p)
        # both the bare id and the trailing-underscore id resolve
        assert scan_registry.resolve_scan_id("Au111_mica_001") == str(p)
        assert scan_registry.resolve_scan_id("Au111_mica_001_") == str(p)

    def test_record_scan_stores_channels_and_frame(self, tmp_path):
        p = tmp_path / "scan_042.sxm"
        p.write_bytes(b"x")
        sid = scan_registry.record_scan(
            p, channels=["Z", "Current"],
            frame={"scan_pixels": [128, 128], "scan_range": "5E-8 5E-8"},
        )
        assert sid == "scan_042"
        rec = scan_registry.get_scan("scan_042")
        assert rec["channels"] == ["Z", "Current"]
        assert rec["frame"]["scan_pixels"] == [128, 128]

    def test_bare_record_after_rich_record_keeps_metadata(self, tmp_path):
        p = tmp_path / "s1.sxm"
        p.write_bytes(b"x")
        scan_registry.record_scan(p, channels=["Z"], frame={"scan_pixels": [64, 64]})
        scan_registry.record_scan_path(p)  # a later bare save must not wipe it
        rec = scan_registry.get_scan("s1")
        assert rec["channels"] == ["Z"]
        assert rec["frame"]["scan_pixels"] == [64, 64]

    def test_list_scans_newest_first_and_bounded(self, tmp_path):
        for i in range(60):
            scan_registry.record_scan_path(tmp_path / f"f{i}.sxm")
        scans = scan_registry.list_scans(5)
        assert [s["scan_id"] for s in scans][0] == "f59"
        assert len(scans) == 5

    def test_unknown_id_returns_none(self):
        assert scan_registry.get_scan("nope") is None
        assert scan_registry.resolve_scan_id("nope") is None
        assert scan_registry.resolve_scan_id("") is None


# ─────────────────────────────────────────────────────────────────────
# load_scan(scan_id=…)
# ─────────────────────────────────────────────────────────────────────

class TestLoadScanByScanId:
    def test_load_scan_by_scan_id(self, tmp_path):
        from mast.agents.data_processing.tools import load_scan
        sxm = _make_sxm(tmp_path / "Au111_topo.sxm", 8, 8)
        scan_registry.record_scan_path(sxm)
        out = load_scan.invoke({"scan_id": "Au111_topo"})
        assert "load_scan failed" not in out
        assert "(8, 8)" in out
        assert "channel: Z" in out

    def test_load_scan_unknown_scan_id_points_at_discovery(self):
        from mast.agents.data_processing.tools import load_scan
        out = load_scan.invoke({"scan_id": "does_not_exist"})
        assert "load_scan failed" in out
        assert "list_scan_dir" in out or "glob_scans" in out

    def test_load_scan_needs_path_or_scan_id(self):
        from mast.agents.data_processing.tools import load_scan
        out = load_scan.invoke({})
        assert "load_scan failed" in out

    def test_successful_sxm_load_enriches_registry(self, tmp_path):
        """Loading a .sxm by path fills channels+frame so a later scan_id lookup
        carries real metadata (lazy enrichment at the one place we parse)."""
        from mast.agents.data_processing.tools import load_scan
        sxm = _make_sxm(tmp_path / "enrich_me.sxm", 8, 8)
        load_scan.invoke({"path": str(sxm)})
        rec = scan_registry.get_scan("enrich_me")
        assert rec is not None
        assert "Z" in (rec.get("channels") or [])
        assert rec.get("frame", {}).get("scan_pixels") == [8, 8]


# ─────────────────────────────────────────────────────────────────────
# list_scan_dir / glob_scans discovery
# ─────────────────────────────────────────────────────────────────────

class TestDiscoveryTools:
    def test_list_scan_dir_explicit_directory(self, tmp_path):
        from mast.agents.data_processing.tools import list_scan_dir
        _make_sxm(tmp_path / "a_001.sxm")
        _make_sxm(tmp_path / "a_002.sxm")
        (tmp_path / "notes.md").write_text("ignore me")
        out = list_scan_dir.invoke({"directory": str(tmp_path)})
        assert "a_001" in out and "a_002" in out
        assert "notes.md" not in out  # non-scan files excluded
        # every listed scan is now resolvable by id
        assert scan_registry.resolve_scan_id("a_001") is not None

    def test_list_scan_dir_missing_directory(self, tmp_path):
        from mast.agents.data_processing.tools import list_scan_dir
        out = list_scan_dir.invoke({"directory": str(tmp_path / "nope")})
        assert "failed" in out

    def test_list_scan_dir_empty_uses_known_dirs(self, tmp_path):
        from mast.agents.data_processing.tools import list_scan_dir
        session = tmp_path / "session"
        session.mkdir()
        _make_sxm(session / "known_001.sxm")
        scan_registry.record_session_dir(session)
        out = list_scan_dir.invoke({})
        assert "known_001" in out

    def test_glob_scans_substring_match_across_known_dirs(self, tmp_path):
        from mast.agents.data_processing.tools import glob_scans
        session = tmp_path / "sess"
        session.mkdir()
        _make_sxm(session / "Au111_mica_007.sxm")
        _make_sxm(session / "Cu111_003.sxm")
        scan_registry.record_session_dir(session)
        out = glob_scans.invoke({"pattern": "Au111"})
        assert "Au111_mica_007" in out
        assert "Cu111_003" not in out
        # discovered → registered → loadable by id
        assert scan_registry.resolve_scan_id("Au111_mica_007") is not None

    def test_glob_scans_registers_hits_for_load_by_id(self, tmp_path):
        from mast.agents.data_processing.tools import glob_scans, load_scan
        d = tmp_path / "d"
        d.mkdir()
        _make_sxm(d / "target_010.sxm", 8, 8)
        glob_scans.invoke({"pattern": "target*", "directory": str(d)})
        out = load_scan.invoke({"scan_id": "target_010"})
        assert "load_scan failed" not in out
        assert "(8, 8)" in out

    def test_glob_scans_needs_a_pattern(self, tmp_path):
        from mast.agents.data_processing.tools import glob_scans
        out = glob_scans.invoke({"pattern": "  "})
        assert "failed" in out
