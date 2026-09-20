"""Parity Wave A — records_export contract tests.

Per the house test rule the router under test is NOT yet included in
``mast.api.app`` (integration wires that); we mount it on a throwaway
FastAPI app with a fresh AppContext. We assert:

  * every endpoint returns 200 + the schema-shaped body;
  * standalone (no live core wired) paths degrade — empty, never broken, never 500;
  * wiring a REAL v1 ExperimentStorage (temp SQLite) drives the timeline endpoint
    through its live path incl. per-action TCP calls + state-diff;
  * the trajectory + full-history export relays hold their contract whether or
    not an on-disk v2 store / project layout exists.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.records_export import router


# ── throwaway apps ─────────────────────────────────────────────────────


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _wired_client(tmp_path) -> tuple[TestClient, object]:
    """A client backed by a REAL v1 ExperimentStorage over a temp SQLite db."""
    from mast.logging.storage import ExperimentStorage

    storage = ExperimentStorage(str(tmp_path / "rec.db"))
    ctx = AppContext()
    ctx.wire(experiment_storage=storage)
    return _client(ctx), storage


# ── POST /api/trajectories/export ──────────────────────────────────────


def test_trajectory_export_rejects_unknown_format() -> None:
    r = _client().post("/api/trajectories/export", json={"format": "bogus"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert body["format"] == "bogus"
    assert body["rows"] == [] and body["count"] == 0


def test_trajectory_export_contract_all_formats() -> None:
    # The v2 store is reached on disk independent of ctx. Either it's absent/empty
    # (degraded) or it returns rows; either way the contract holds + never 500s.
    for fmt in ("jsonl", "sft", "dpo", "failure_mining"):
        r = _client().post("/api/trajectories/export", json={"format": fmt})
        assert r.status_code == 200, fmt
        body = r.json()
        assert body["format"] == fmt
        assert isinstance(body["rows"], list)
        assert isinstance(body["count"], int)
        if body["degraded"]:
            assert body["rows"] == [] and body["count"] == 0
        else:
            assert body["ok"] is True
            assert body["count"] >= len(body["rows"])


def test_trajectory_export_validates_limit() -> None:
    r = _client().post("/api/trajectories/export", json={"format": "jsonl", "limit": 0})
    assert r.status_code == 422  # ge=1


def test_trajectory_export_inline_off_returns_no_rows_but_count_field() -> None:
    r = _client().post(
        "/api/trajectories/export", json={"format": "jsonl", "inline": False}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == []  # inline off ⇒ never inlines rows
    assert isinstance(body["count"], int)


# ── POST /api/experiments/export ───────────────────────────────────────


def test_experiments_export_runs_into_tmp(tmp_path) -> None:
    # Point the export at an explicit dest so it never touches the real project.
    dest = tmp_path / "export.zip"
    r = _client().post(
        "/api/experiments/export",
        json={"include_heavy": False, "dest": str(dest)},
    )
    assert r.status_code == 200
    body = r.json()
    # export_all_history creates an empty-but-valid zip even with no data dirs.
    if body["degraded"]:
        assert body["ok"] is False
    else:
        assert body["ok"] is True
        assert body["dest"]
        assert body["include_heavy"] is False
        assert isinstance(body["file_count"], int)
        assert isinstance(body["total_bytes"], int)
        assert dest.exists()


# ── GET /api/scans/latest ──────────────────────────────────────────────


def test_latest_scans_empty_dir_is_empty_not_broken(tmp_path) -> None:
    r = _client().get("/api/scans/latest", params={"dir": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["scans"] == [] and body["count"] == 0
    assert str(tmp_path) in body["search_dirs"]
    assert body["degraded"] is False


def test_latest_scans_discovers_files(tmp_path) -> None:
    (tmp_path / "scan_a.sxm").write_bytes(b"fake")
    (tmp_path / "spectrum_b.dat").write_bytes(b"fake")
    (tmp_path / "export_c.txt").write_bytes(b"1 2\n3 4\n")  # generic text now supported
    (tmp_path / "ignore.log").write_bytes(b"fake")          # non-scan ext → excluded
    r = _client().get("/api/scans/latest", params={"dir": str(tmp_path), "n": 10})
    assert r.status_code == 200
    body = r.json()
    names = sorted(s["name"] for s in body["scans"])
    assert names == ["export_c.txt", "scan_a.sxm", "spectrum_b.dat"]
    assert body["count"] == 3
    for s in body["scans"]:
        assert s["ext"] in (".sxm", ".dat", ".txt")
        assert s["size_bytes"] is not None


def test_latest_scans_validates_n() -> None:
    r = _client().get("/api/scans/latest", params={"n": 0})
    assert r.status_code == 422  # ge=1


# ── ext filter (2026-08-04, 的残余 / #53 同形状) ──────────────


def _rig_shaped_dir(tmp_path):
    """Synthetic directory with newer data files and older scan files; filtering must precede limiting."""
    import os
    import time

    now = time.time()
    for i in range(30):
        p = tmp_path / f"spec_{i:02d}.dat"
        p.write_bytes(b"fake")
        os.utime(p, (now - 10 + i, now - 10 + i))     # newest
    for i in range(3):
        p = tmp_path / f"scan_{i}.sxm"
        p.write_bytes(b"fake")
        os.utime(p, (now - 1000 + i, now - 1000 + i))  # older than every .dat
    return tmp_path


def test_ext_filter_applies_before_the_n_slice(tmp_path) -> None:
    """The point of the parameter: .sxm survives being outnumbered by newer .dat.

    Without server-side filtering the client asks for the newest 5, receives 5
    .dat, keeps the .sxm ones and shows an empty screen while three sit on disk.
    """
    d = _rig_shaped_dir(tmp_path)
    r = _client().get("/api/scans/latest", params={"dir": str(d), "n": 5, "ext": ".sxm"})
    assert r.status_code == 200
    body = r.json()
    assert [s["ext"] for s in body["scans"]] == [".sxm"] * 3
    assert body["total_matched"] == 3


def test_counts_by_ext_describes_the_whole_result_not_the_page(tmp_path) -> None:
    """Chip labels come from here, so they must not shrink with ``n``."""
    d = _rig_shaped_dir(tmp_path)
    r = _client().get("/api/scans/latest", params={"dir": str(d), "n": 2, "ext": ".sxm"})
    body = r.json()
    assert body["count"] == 2                       # the page
    assert body["total_matched"] == 3               # matches beyond the page
    # …and the counts still see the .dat files the filter excluded, which is what
    # keeps a chip from reading 0 for a type that is merely filtered out.
    assert body["counts_by_ext"][".sxm"] == 3
    assert body["counts_by_ext"][".dat"] == 30


def test_ext_filter_accepts_bare_and_multiple_extensions(tmp_path) -> None:
    (tmp_path / "a.sxm").write_bytes(b"fake")
    (tmp_path / "b.dat").write_bytes(b"fake")
    (tmp_path / "c.txt").write_bytes(b"1 2\n")
    r = _client().get("/api/scans/latest", params={"dir": str(tmp_path), "ext": "sxm,.DAT"})
    body = r.json()
    assert sorted(s["ext"] for s in body["scans"]) == [".dat", ".sxm"]


def test_unknown_ext_filter_matches_nothing_rather_than_everything(tmp_path) -> None:
    """Fail closed. Returning the unfiltered list would answer a question nobody
    asked — and look exactly like the bug this parameter exists to fix."""
    (tmp_path / "a.sxm").write_bytes(b"fake")
    r = _client().get("/api/scans/latest", params={"dir": str(tmp_path), "ext": ".nope"})
    body = r.json()
    assert body["scans"] == [] and body["total_matched"] == 0
    assert body["counts_by_ext"][".sxm"] == 1       # still honest about what IS there


def test_blank_ext_is_no_filter(tmp_path) -> None:
    (tmp_path / "a.sxm").write_bytes(b"fake")
    (tmp_path / "b.dat").write_bytes(b"fake")
    r = _client().get("/api/scans/latest", params={"dir": str(tmp_path), "ext": ""})
    assert len(r.json()["scans"]) == 2


# ── paging (2026-08-21) ────────────────────────────────────────────────
#
# The client asked for a fixed 60 and printed "只显示最近 60 个" underneath —
# file 61 was unreachable by any interaction. ``offset`` pages the collapsed list.


def _numbered_dir(tmp_path, count: int = 5):
    """`count` .sxm files with strictly decreasing mtime, newest first."""
    import os
    import time

    now = time.time()
    for i in range(count):
        p = tmp_path / f"scan_{i:02d}.sxm"
        p.write_bytes(f"fake-{i}".encode())
        os.utime(p, (now - i, now - i))     # scan_00 newest
    return tmp_path


def test_offset_pages_past_the_first_window(tmp_path) -> None:
    d = _numbered_dir(tmp_path, 5)
    first = _client().get("/api/scans/latest", params={"dir": str(d), "n": 2}).json()
    second = _client().get(
        "/api/scans/latest", params={"dir": str(d), "n": 2, "offset": 2}).json()
    assert [s["name"] for s in first["scans"]] == ["scan_00.sxm", "scan_01.sxm"]
    assert [s["name"] for s in second["scans"]] == ["scan_02.sxm", "scan_03.sxm"]
    assert second["offset"] == 2
    assert first["has_more"] is True and second["has_more"] is True


def test_last_page_reports_no_more(tmp_path) -> None:
    d = _numbered_dir(tmp_path, 5)
    body = _client().get(
        "/api/scans/latest", params={"dir": str(d), "n": 2, "offset": 4}).json()
    assert [s["name"] for s in body["scans"]] == ["scan_04.sxm"]
    assert body["has_more"] is False


def test_offset_past_the_end_is_empty_not_broken(tmp_path) -> None:
    d = _numbered_dir(tmp_path, 3)
    body = _client().get(
        "/api/scans/latest", params={"dir": str(d), "n": 10, "offset": 99}).json()
    assert body["scans"] == [] and body["has_more"] is False
    assert body["degraded"] is False
    assert body["total_collapsed"] == 3      # still honest about what exists


def test_offset_rejects_negative() -> None:
    assert _client().get("/api/scans/latest", params={"offset": -1}).status_code == 422


def test_omitting_offset_is_the_old_behaviour(tmp_path) -> None:
    """Back-compat: callers that never heard of paging see exactly page one."""
    d = _numbered_dir(tmp_path, 5)
    without = _client().get("/api/scans/latest", params={"dir": str(d), "n": 3}).json()
    with_zero = _client().get(
        "/api/scans/latest", params={"dir": str(d), "n": 3, "offset": 0}).json()
    assert [s["name"] for s in without["scans"]] == [s["name"] for s in with_zero["scans"]]
    assert without["offset"] == 0


# ── copy collapse (2026-08-21) ─────────────────────────────────────────
#
# The automatic ingest copies each scan into the experiment folder and leaves the
# original in place, so the listing showed the same measurement twice with an
# identical name, size and timestamp on both cards.


def _with_ingested_copy(tmp_path):
    """A session file plus the byte-identical copy the ingest would make.

    ``shutil.copy2`` is what ``filestore`` effectively does (copy + copystat), so
    the copy carries the same mtime — which is exactly why path-only dedup could
    not see it."""
    import shutil

    session = tmp_path / "session"
    session.mkdir()
    managed = tmp_path / "exp" / "samples" / "S01" / "raw" / "sxm"
    managed.mkdir(parents=True)
    src = session / "scan_001.sxm"
    src.write_bytes(b"identical bytes")
    shutil.copy2(src, managed / "scan_001.sxm")
    return tmp_path


def test_ingested_copies_show_as_one_entry(tmp_path) -> None:
    d = _with_ingested_copy(tmp_path)
    body = _client().get("/api/scans/latest", params={"dir": str(d), "n": 10}).json()
    assert len(body["scans"]) == 1
    entry = body["scans"][0]
    assert entry["copies"] == 2
    # Both copies are NAMED. A listing that hides a file is the failure this
    # codebase already recorded twice  pointing the other way.
    assert len(entry["locations"]) == 2
    assert {Path(loc["path"]).name for loc in entry["locations"]} == {"scan_001.sxm"}


def test_collapse_keeps_the_raw_file_count_visible(tmp_path) -> None:
    """``total_matched`` still counts FILES, ``total_collapsed`` counts entries —
    the difference is how many duplicates the operator is no longer looking at."""
    d = _with_ingested_copy(tmp_path)
    body = _client().get("/api/scans/latest", params={"dir": str(d), "n": 10}).json()
    assert body["total_matched"] == 2
    assert body["total_collapsed"] == 1
    assert body["counts_by_ext"][".sxm"] == 2
    assert body["counts_by_ext_collapsed"][".sxm"] == 1


def test_same_name_different_content_stays_two_entries(tmp_path) -> None:
    """Same-named files with different contents remain separate entries."""
    import os
    import time

    a_dir, b_dir = tmp_path / "20260318", tmp_path / "20260319"
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / "unnamed0001.sxm").write_bytes(b"first measurement")
    (b_dir / "unnamed0001.sxm").write_bytes(b"second measurement!")
    now = time.time()
    os.utime(a_dir / "unnamed0001.sxm", (now - 500, now - 500))
    os.utime(b_dir / "unnamed0001.sxm", (now, now))
    body = _client().get("/api/scans/latest", params={"dir": str(tmp_path), "n": 10}).json()
    assert len(body["scans"]) == 2
    assert all(s["copies"] == 1 for s in body["scans"])


def test_single_copy_entries_carry_no_location_list(tmp_path) -> None:
    """The one path is already the entry's ``path``; repeating it on every card
    is bytes on the wire for nothing."""
    (tmp_path / "solo.sxm").write_bytes(b"fake")
    body = _client().get("/api/scans/latest", params={"dir": str(tmp_path), "n": 10}).json()
    assert body["scans"][0]["copies"] == 1
    assert body["scans"][0]["locations"] == []
    assert body["scans"][0]["kind"] in ("origin", "experiment", "quarantine")


# ── GET /api/scans/preview ─────────────────────────────────────────────


def test_scan_preview_requires_path() -> None:
    r = _client().get("/api/scans/preview")
    assert r.status_code == 422  # path is required


def test_scan_preview_missing_file_degrades(tmp_path) -> None:
    missing = tmp_path / "nope.sxm"
    r = _client().get("/api/scans/preview", params={"path": str(missing)})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is False and body["degraded"] is True
    assert body["image"] is None


def test_scan_preview_unsupported_ext_degrades(tmp_path) -> None:
    f = tmp_path / "data.txt"
    f.write_bytes(b"hello")
    r = _client().get("/api/scans/preview", params={"path": str(f)})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["rendered"] is False
    assert body["degraded"] is True


def test_scan_preview_unparseable_file_degrades_not_broken(tmp_path) -> None:
    # A .sxm with garbage bytes: present + supported ext, but render returns None.
    f = tmp_path / "bad.sxm"
    f.write_bytes(b"not a real nanonis file")
    r = _client().get("/api/scans/preview", params={"path": str(f)})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["rendered"] is False
    assert body["degraded"] is True
    assert body["image"] is None


# ── 去衬底 / flatten (2026-08-21) ───────────────────────────────────────
#
# A raw topograph is mostly sample tilt: over a 100 nm frame a fraction of a
# degree is nanometres of z while the surface structure is picometres. The
# preview rendered the raw array, so the colour range went to the ramp.

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "nanonis"
_SXM_FIXTURE = _FIXTURES / "scan_topography.sxm"
_DAT_FIXTURE = _FIXTURES / "bias_spectroscopy_200pt.dat"


def test_preview_rejects_an_unknown_flatten_mode() -> None:
    """Fail loudly rather than silently rendering something else: a client that
    asks for a mode we do not have has a bug, and a picture that looks plausible
    would hide it."""
    r = _client().get("/api/scans/preview",
                      params={"path": str(_SXM_FIXTURE), "flatten": "bogus"})
    assert r.status_code == 422


def test_sxm_preview_is_line_flattened_by_default() -> None:
    r = _client().get("/api/scans/preview", params={"path": str(_SXM_FIXTURE)})
    assert r.status_code == 200
    body = r.json()
    assert body["rendered"] is True
    assert body["flatten"] == "line"
    assert body["channel"]                       # says WHICH channel it drew
    assert body["image"].startswith("data:image/png;base64,")


def test_preview_honours_an_explicit_mode_and_reports_it_back() -> None:
    seen = {}
    for mode in ("raw", "plane", "line"):
        body = _client().get(
            "/api/scans/preview",
            params={"path": str(_SXM_FIXTURE), "flatten": mode, "size": 128},
        ).json()
        assert body["flatten"] == mode
        seen[mode] = body["image"]
    # Different processing must produce different pixels — otherwise the control
    # is decorative and the operator is told a background was removed when it was
    # not. (A flatten that silently no-ops is exactly the "guard that isn't".)
    assert seen["raw"] != seen["plane"] != seen["line"]
    assert seen["raw"] != seen["line"]


def test_auto_reports_the_method_it_chose_and_why() -> None:
    """``auto`` resolves to a concrete method — possibly one no fixed mode
    offers (poly2 / masked_line) — and explains itself in Chinese."""
    body = _client().get(
        "/api/scans/preview",
        params={"path": str(_SXM_FIXTURE), "flatten": "auto", "size": 128},
    ).json()
    assert body["rendered"] is True
    assert body["flatten"] in ("plane", "poly2", "line", "masked_line", "raw")
    assert body["flatten"] != "auto"             # never echoes the request back
    assert body["flatten_why"]                   # the operator gets a reason


def test_preview_carries_frame_scale_and_bias() -> None:
    """So a card can label the picture without a second round trip."""
    body = _client().get("/api/scans/preview", params={"path": str(_SXM_FIXTURE)}).json()
    assert body["width_nm"] and body["width_nm"] > 0
    assert body["bias_v"] is not None


def test_dat_preview_ignores_flatten(tmp_path) -> None:
    """A spectrum is a curve; a plane fit through it removes signal, not
    background. The interactive version is GET /api/scans/spectrum."""
    body = _client().get(
        "/api/scans/preview",
        params={"path": str(_DAT_FIXTURE), "flatten": "line"},
    ).json()
    assert body["rendered"] is True
    assert body["flatten"] == "raw"


def test_explicit_channel_is_honoured() -> None:
    body = _client().get(
        "/api/scans/preview",
        params={"path": str(_SXM_FIXTURE), "channel": "Current"},
    ).json()
    if body["rendered"]:                          # fixture may not carry Current
        assert body["channel"].lower().startswith("current") or body["channel"] == "Z"


# ── GET /api/scans/spectrum ────────────────────────────────────────────
#
# The Data tab could only show a spectrum as a two-column PNG with no axes and
# no zoom. This hands over the numbers.


def test_spectrum_requires_path() -> None:
    assert _client().get("/api/scans/spectrum").status_code == 422


def test_spectrum_missing_file_degrades(tmp_path) -> None:
    body = _client().get(
        "/api/scans/spectrum", params={"path": str(tmp_path / "nope.dat")}).json()
    assert body["found"] is False and body["degraded"] is True
    assert body["series"] == []


def test_spectrum_rejects_an_image_file() -> None:
    body = _client().get(
        "/api/scans/spectrum", params={"path": str(_SXM_FIXTURE)}).json()
    assert body["degraded"] is True
    assert ".sxm" in (body["detail"] or "")


def test_spectrum_returns_plottable_series() -> None:
    r = _client().get("/api/scans/spectrum", params={"path": str(_DAT_FIXTURE)})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["degraded"] is False
    assert body["kind"] == "iv"
    assert body["n_points"] > 0
    assert len(body["sweep"]) == body["n_points"]
    ids = [s["id"] for s in body["series"]]
    assert "current" in ids
    # Every column in the file is named in the response even if unplotted.
    assert len(body["columns"]) >= len(body["series"])


def test_spectrum_marks_a_numerically_derived_didv() -> None:
    """The fixture has no lock-in column, so dI/dV can only be a derivative —
    and the response has to say so rather than let it pass as measured."""
    body = _client().get(
        "/api/scans/spectrum", params={"path": str(_DAT_FIXTURE)}).json()
    assert body["didv_source"] == "numeric"
    didv = [s for s in body["series"] if s["id"] == "didv"]
    assert didv and didv[0]["source"] == "numeric"


# ── GET /api/scans/attribution ─────────────────────────────────────────


def test_attribution_requires_experiment_id() -> None:
    assert _client().get("/api/scans/attribution").status_code == 422


def test_attribution_unknown_id_is_empty_not_broken() -> None:
    r = _client().get("/api/scans/attribution", params={"experiment_id": "no-such-exp"})
    assert r.status_code == 200
    body = r.json()
    assert body["experiment_id"] == "no-such-exp"
    assert body["files"] == [] and body["count"] == 0
    # Either the store is absent (degraded) or present and simply has no rows —
    # both are valid; what matters is that neither 500s.
    assert isinstance(body["degraded"], bool)


def test_attribution_validates_limit() -> None:
    r = _client().get("/api/scans/attribution",
                      params={"experiment_id": "x", "limit": 0})
    assert r.status_code == 422


def test_attribution_reads_file_locations(tmp_path, monkeypatch) -> None:
    """A recorded location retains its sample and origin for experiment grouping."""
    from mast.logging.v2.repos import build_repos
    from mast.logging.v2.storage import ExperimentStoreV2

    store = ExperimentStoreV2(tmp_path / "v2.db")
    repos = build_repos(store)
    # file_locations.experiment_id is a real foreign key, so the chain above it
    # has to exist — which is also the point: attribution is only meaningful
    # against an experiment the ledger knows.
    cid = repos.campaigns.create(title="c", hypothesis="h", goal={},
                                 hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid,
                                  title="e", exp_type="topo_scan")
    repos.file_locations.record(
        sha256="a" * 64, experiment_id=eid, rel_path="samples/S01__x__aaa/raw/sxm/a.sxm",
        source="skill", sample_id=sid, origin_path=r"D:\sessions\a.sxm",
        size_bytes=1234,
    )

    # The handler does `from …storage import open_store` INSIDE the function, so
    # the name is looked up on the module at call time — patching the module
    # attribute reaches it without a seam in production code.
    import mast.logging.v2.storage as storage_mod

    monkeypatch.setattr(storage_mod, "open_store", lambda *a, **k: store)
    body = _client().get(
        "/api/scans/attribution", params={"experiment_id": eid}).json()
    assert body["degraded"] is False
    assert body["count"] == 1
    row = body["files"][0]
    assert row["sha256"] == "a" * 64
    assert row["sample_id"] == sid
    assert row["rel_path"].endswith("a.sxm")
    assert row["origin_path"] == r"D:\sessions\a.sxm"
    assert row["size_bytes"] == 1234


def _seeded_store(tmp_path):
    """A v2 store with one experiment that owns one recorded file location."""
    from mast.logging.v2.repos import build_repos
    from mast.logging.v2.storage import ExperimentStoreV2

    store = ExperimentStoreV2(tmp_path / "v2.db")
    repos = build_repos(store)
    cid = repos.campaigns.create(title="c", hypothesis="h", goal={},
                                 hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid,
                                  title="e", exp_type="topo_scan")
    repos.file_locations.record(
        sha256="b" * 64, experiment_id=eid,
        rel_path="samples/S01__x__aaa/raw/sxm/scan.sxm",
        source="skill", sample_id=sid, size_bytes=99,
    )
    return store, eid


def _folder_with_provenance(root, v1_id: str, v2_id: str):
    """An experiment folder named from the V1 id, as the real layout is."""
    import json

    d = root / f"2026-07-01__demo__{v1_id[:8]}"
    (d / "samples" / "S01__x__aaa" / "raw" / "sxm").mkdir(parents=True)
    (d / "experiment.json").write_text(
        json.dumps({
            "id": v1_id,
            "dir_name": d.name,
            "provenance": {"v1_row_id": v1_id, "v2_experiment_id": v2_id},
        }),
        encoding="utf-8",
    )
    return d


def test_attribution_accepts_the_v1_id_the_client_actually_has(
    tmp_path, monkeypatch
) -> None:
    """一个实验有两个 id，而客户端拿到的是另一个。

    The folder is named from the v1 row id and ``GET /api/experiments`` returns
    that one, but ``file_locations.experiment_id`` holds the v2 ULID. Querying
    the table with the id the client has therefore returned zero rows for an
    experiment with 64 of them — an empty screen indistinguishable from "this
    experiment produced no data". Found on the instrument's real store, 2026-08-24.
    """
    store, v2_id = _seeded_store(tmp_path)
    v1_id = "627c3dab-3c08-49ae-83c1-0a58f0c45685"
    root = tmp_path / "experiments"
    root.mkdir()
    _folder_with_provenance(root, v1_id, v2_id)

    import mast.core.experiment_paths as paths_mod
    import mast.logging.v2.storage as storage_mod

    monkeypatch.setattr(storage_mod, "open_store", lambda *a, **k: store)
    monkeypatch.setattr(paths_mod, "experiment_root", lambda **k: root)

    by_v2 = _client().get("/api/scans/attribution", params={"experiment_id": v2_id}).json()
    by_v1 = _client().get("/api/scans/attribution", params={"experiment_id": v1_id}).json()
    assert by_v2["count"] == 1
    assert by_v1["count"] == 1, "客户端手里的 v1 id 查不到 —— 分组视图会全空"


def test_attribution_rebuilds_an_absolute_path_that_exists(tmp_path, monkeypatch) -> None:
    """rel_path is relative to the experiment folder; without the folder the
    grouped view has nothing to preview."""
    store, v2_id = _seeded_store(tmp_path)
    v1_id = "627c3dab-3c08-49ae-83c1-0a58f0c45685"
    root = tmp_path / "experiments"
    root.mkdir()
    d = _folder_with_provenance(root, v1_id, v2_id)
    (d / "samples" / "S01__x__aaa" / "raw" / "sxm" / "scan.sxm").write_bytes(b"x")

    import mast.core.experiment_paths as paths_mod
    import mast.logging.v2.storage as storage_mod

    monkeypatch.setattr(storage_mod, "open_store", lambda *a, **k: store)
    monkeypatch.setattr(paths_mod, "experiment_root", lambda **k: root)

    row = _client().get(
        "/api/scans/attribution", params={"experiment_id": v2_id}).json()["files"][0]
    assert row["abs_path"]
    assert Path(row["abs_path"]).exists()


def test_attribution_survives_an_unlocatable_folder(tmp_path, monkeypatch) -> None:
    """No folder ⇒ abs_path is null and the row still ships. Null is honest;
    a fabricated path would send the client looking for a file that is not there."""
    store, v2_id = _seeded_store(tmp_path)
    empty = tmp_path / "nothing"
    empty.mkdir()

    import mast.core.experiment_paths as paths_mod
    import mast.logging.v2.storage as storage_mod

    monkeypatch.setattr(storage_mod, "open_store", lambda *a, **k: store)
    monkeypatch.setattr(paths_mod, "experiment_root", lambda **k: empty)

    body = _client().get(
        "/api/scans/attribution", params={"experiment_id": v2_id}).json()
    assert body["count"] == 1
    assert body["files"][0]["abs_path"] is None
    assert body["degraded"] is False


# ── GET /api/experiments/{id}/timeline ─────────────────────────────────


def test_timeline_degrades_unwired() -> None:
    r = _client().get("/api/experiments/abc/timeline")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "abc"
    assert body["degraded"] is True
    assert body["found"] is False
    assert body["entries"] == []


def test_timeline_not_found_is_empty_not_broken(tmp_path) -> None:
    client, _ = _wired_client(tmp_path)
    r = client.get("/api/experiments/does-not-exist/timeline")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is False and body["degraded"] is False
    assert body["entries"] == []


def test_timeline_wired_surfaces_tcp_and_state(tmp_path) -> None:
    from mast.core.types import (
        ActionRecord,
        HardwareState,
        NanonisCallRecord,
        SkillResult,
    )

    client, storage = _wired_client(tmp_path)
    exp_id = storage.create_experiment("Au111 run", "map terraces")

    before = HardwareState(bias_v=1.0, current_a=1e-9, z_pos_m=-5e-9)
    after = HardwareState(bias_v=0.5, current_a=2e-9, z_pos_m=-4e-9)
    calls = [
        NanonisCallRecord(method="Bias_Set", args=(0.5,), elapsed_s=0.01),
        NanonisCallRecord(method="ZCtrl_OnOffSet", args=(1,), error="port busy"),
    ]
    result = SkillResult(
        skill_name="SetBias", success=True, state_before=before,
        state_after=after, nanonis_calls=calls,
    )
    rec = ActionRecord(
        experiment_id=exp_id,
        skill_name="SetBias",
        parameters={"bias_v": 0.5},
        result=result,
        state_before=before,
        state_after=after,
        nanonis_calls=calls,
        context="operator asked to lower bias",
        duration_s=0.25,
    )
    storage.log_action(rec)

    r = client.get(f"/api/experiments/{exp_id}/timeline")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["degraded"] is False
    assert body["name"] == "Au111 run"
    assert body["action_count"] == 1
    assert body["succeeded"] == 1 and body["failed"] == 0
    assert abs(body["total_duration_s"] - 0.25) < 1e-6

    entry = body["entries"][0]
    assert entry["skill_name"] == "SetBias"
    assert entry["success"] is True
    assert entry["parameters"] == {"bias_v": 0.5}
    assert entry["context"] == "operator asked to lower bias"

    methods = [c["method"] for c in entry["tcp_calls"]]
    assert methods == ["Bias_Set", "ZCtrl_OnOffSet"]
    assert entry["tcp_calls"][1]["error"] == "port busy"

    assert entry["state_diff"]["before"]["bias_v"] == 1.0
    assert entry["state_diff"]["after"]["bias_v"] == 0.5
