"""OCFL 1.1 export smoke test."""
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

import json

import pytest

from mast.logging.v2 import ocfl
from mast.logging.v2.repos import build_repos
from mast.logging.v2.storage import ExperimentStoreV2


@pytest.fixture
def loaded(tmp_path):
    store = ExperimentStoreV2(tmp_path / "x.db")
    repos = build_repos(store)
    cid = repos.campaigns.create(title="c", hypothesis="h", goal={},
                                  hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid,
                                   title="e", exp_type="topo_scan")
    aid = repos.actions.begin(experiment_id=eid, agent_id="agent:IC",
                               action_type="scan")
    repos.actions.succeed(aid)
    src = tmp_path / "scan.sxm"
    src.write_bytes(b"sxm-content")
    from mast.logging.v2.cas import sha256_file
    digest, size = sha256_file(src)
    repos.scan_files.register(
        produced_by_action_id=aid, sha256=digest, size_bytes=size,
        current_path=str(src), format_kind="sxm", parser_spec="t",
    )
    repos.experiments.end(eid, conclusion="ok")
    return repos, eid, tmp_path


def test_export_object_structure(loaded):
    repos, eid, tmp_path = loaded
    obj = ocfl.export(repos, eid, tmp_path / "ocfl_root")
    assert (obj / "0=ocfl_object_1.1").is_file()
    assert (obj / "inventory.json").is_file()
    assert (obj / "inventory.json.sha512").is_file()
    assert (obj / "v1" / "inventory.json").is_file()
    assert (obj / "v1" / "content" / "ro-crate-metadata.json").is_file()


def test_export_inventory_well_formed(loaded):
    repos, eid, tmp_path = loaded
    obj = ocfl.export(repos, eid, tmp_path / "ocfl_root")
    inv = json.loads((obj / "inventory.json").read_text(encoding="utf-8"))
    assert inv["head"] == "v1"
    assert inv["digestAlgorithm"] == "sha512"
    assert inv["contentDirectory"] == "content"
    assert isinstance(inv["manifest"], dict)
    assert "versions" in inv and "v1" in inv["versions"]


def test_verify_passes_on_freshly_exported(loaded):
    repos, eid, tmp_path = loaded
    obj = ocfl.export(repos, eid, tmp_path / "ocfl_root")
    ok, errors = ocfl.verify(obj)
    assert ok, errors


def test_verify_detects_tampering(loaded):
    repos, eid, tmp_path = loaded
    obj = ocfl.export(repos, eid, tmp_path / "ocfl_root")
    f = next((obj / "v1" / "content").rglob("ro-crate-metadata.json"))
    f.write_text("tampered", encoding="utf-8")
    ok, errors = ocfl.verify(obj)
    assert not ok
    assert any("digest mismatch" in e for e in errors)
