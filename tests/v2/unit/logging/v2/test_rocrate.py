"""RO-Crate export smoke test."""
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

from mast.logging.v2 import rocrate
from mast.logging.v2.repos import build_repos
from mast.logging.v2.storage import ExperimentStoreV2


@pytest.fixture
def loaded(tmp_path):
    store = ExperimentStoreV2(tmp_path / "x.db")
    repos = build_repos(store)
    cid = repos.campaigns.create(
        title="c", hypothesis="h", goal={}, hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(
        campaign_id=cid, sample_id=sid, title="e", exp_type="topo_scan")
    aid = repos.actions.begin(experiment_id=eid, agent_id="agent:IC", action_type="scan")
    repos.actions.succeed(aid)
    src = tmp_path / "scan.sxm"
    src.write_bytes(b"fake sxm bytes")
    from mast.logging.v2.cas import sha256_file
    digest, size = sha256_file(src)
    sfid = repos.scan_files.register(
        produced_by_action_id=aid, sha256=digest, size_bytes=size,
        current_path=str(src), format_kind="sxm", parser_spec="test-parser",
        meta={"bias_v": -2.0, "width_m": 5e-9},
    )
    oid = repos.observations.record_scan(
        action_id=aid, experiment_id=eid, observable="topography",
        scan_file_id=sfid,
    )
    repos.experiments.end(eid, conclusion="ok")
    return repos, eid, src, digest


def test_build_metadata_shape(loaded):
    repos, eid, _, digest = loaded
    md = rocrate.build_metadata(repos, eid)
    assert "@context" in md
    assert "@graph" in md
    types = {n.get("@type") for n in md["@graph"]}
    assert "Dataset" in types
    assert "Action" in types
    assert "Observation" in types
    assert "File" in types


def test_export_writes_metadata_and_files(loaded, tmp_path):
    repos, eid, src, digest = loaded
    out = rocrate.export(repos, eid, tmp_path / "crate")
    assert (out / "ro-crate-metadata.json").is_file()
    file_copy = out / "files" / digest
    assert file_copy.is_file()
    assert file_copy.read_bytes() == src.read_bytes()
