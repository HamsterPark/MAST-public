"""DP agent scan discovery + .sxm loading — .

The 2026-07-10 full-flow run saved scans into Nanonis's session dir
(D:\\Data\\fsSTM\\...), but data_processing's get_latest_scan_file (running
with context=None — no pool) searched only working-sessions and found
nothing; load_scan then failed on a dozen guessed paths and reported ".sxm
parsing not yet wired" because it looked for the ARCHIVED v1 parser.

Pins:
  * SaveScan/session resolves publish real scan locations into
    mast.core.scan_registry; context-less _candidate_save_dirs reads them.
  * load_scan reads a real .sxm via v2's own mast.io.nanonis_files.read_sxm.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
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

import struct

import numpy as np
import pytest

from mast.core import scan_registry
from mast.skills.builtins.scan_extra import GetLatestScanFile, _candidate_save_dirs


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
    blob = header.encode() + b"\x1a\x04" + fwd.tobytes() + bwd.tobytes()
    path.write_bytes(blob)
    return path


class TestScanRegistryDiscovery:
    def test_context_less_search_uses_recorded_dirs(self, tmp_path):
        """A scan recorded by SaveScan-side code becomes discoverable by the
        DP tool's context=None search."""
        session = tmp_path / "nanonis_session"
        session.mkdir()
        sxm = _make_sxm(session / "Au111_001.sxm")
        scan_registry.record_scan_path(sxm)
        dirs = _candidate_save_dirs(None)
        assert session.resolve() in [d.resolve() for d in dirs]
        res = GetLatestScanFile().execute(None, {"max_age_s": 3600})
        assert res.success
        assert res.data["path"] == str(sxm)

    def test_session_dir_record_used(self, tmp_path):
        session = tmp_path / "nanonis_session2"
        session.mkdir()
        _make_sxm(session / "scan_0001.sxm")
        scan_registry.record_session_dir(session)
        res = GetLatestScanFile().execute(None, {"max_age_s": 3600})
        assert res.success
        assert res.data["path"] is not None
        assert str(session) in res.data["path"]

    def test_registry_bounded_and_newest_first(self, tmp_path):
        for i in range(60):
            scan_registry.record_scan_path(tmp_path / f"f{i}.sxm")
        recent = scan_registry.recent_scan_paths(5)
        assert recent[0].endswith("f59.sxm")
        assert len(recent) == 5


class TestLoadScanSxm:
    def test_load_scan_reads_real_sxm(self, tmp_path):
        from mast.agents.data_processing.tools import load_scan
        sxm = _make_sxm(tmp_path / "Au111_topo.sxm", nx=8, ny=8)
        out = load_scan.invoke({"path": str(sxm)})
        assert "load_scan failed" not in out
        assert "sxm" in out
        assert "(8, 8)" in out          # reshaped to the header geometry
        assert "channel: Z" in out      # topography channel chosen
        assert "scan_offset" in out     # header geometry surfaced

    def test_load_scan_resolves_a_path_the_model_guessed_wrong(self, tmp_path):
        """The 2026-07-10 field failure, verbatim :

            load_scan failed: FileNotFoundError: file not found:
            D:\\MAST-data\\working-sessions\\Au111_mica_STM_STS_Au111_mica_001_.npy

        The scan WAS on disk. The model reconstructed a plausible path — right
        stem, wrong directory, wrong extension (.npy for what Nanonis saved as
        .sxm) — and load_scan did a bare exists() check and gave up. An LLM will
        guess paths; refusing to look strands the operator's data behind a typo.
        """
        from mast.agents.data_processing.tools import load_scan
        from mast.core import scan_registry

        real = tmp_path / "session" / "Au111_mica_STM_STS_Au111_mica_001.sxm"
        real.parent.mkdir(parents=True)
        _make_sxm(real, 8, 8)
        scan_registry.record_scan_path(real)

        # the model's guess: wrong directory AND wrong extension
        guess = tmp_path / "working-sessions" / "Au111_mica_STM_STS_Au111_mica_001_.npy"
        out = load_scan.invoke({"path": str(guess)})

        assert "load_scan failed" not in out, (
            f"the guessed path was refused instead of resolved: {out}")
        assert "(8, 8)" in out

    def test_a_genuinely_absent_file_says_where_it_looked(self, tmp_path, monkeypatch):
        """Still fails when the file really is not there — but "file not found"
        alone sent the agent round in circles guessing new paths. Say where we
        looked, and point at the tool that knows the real answer."""
        from mast.agents.data_processing.tools import load_scan
        from mast.core import scan_registry

        scan_registry.record_session_dir(tmp_path / "session")
        (tmp_path / "session").mkdir(exist_ok=True)

        out = load_scan.invoke({"path": str(tmp_path / "nope.sxm")})
        assert "load_scan failed" in out
        assert "session" in out, f"the error does not say where it looked: {out}"
        assert "get_latest_scan_file" in out, (
            f"the error does not point at the tool that knows: {out}")
