"""TipShapeWithReadback post-skill hook: render → v2 records → vision buffer.

Fake repos + fake buffer (no real DB / no real BufferService). Verifies the hook
renders a PNG, registers action+scan_file+observation, emits a TIP_SHAPE_VERDICT
event, ignores other skills, and survives a records-DB failure (best-effort).
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

from mast.webui.tip_shape_records import make_tip_shape_post_hook  # noqa: E402


def _data():
    return {
        "z": {"t_s": [0.0, 0.1, 0.2, 0.3], "samples_m": [1e-9, 0.7e-9, 0.7e-9, 1e-9]},
        "current": {"t_s": [0.0, 0.1, 0.2, 0.3], "samples_a": [5e-11, 8e-10, 8e-10, 5e-11]},
        "indent": {"verdict": "no_change", "delta_m": -1e-13, "advice": "没扎上",
                   "z1_m": 1e-9, "z3_m": 1e-9, "baseline_sigma_m": 1e-14, "z_min_m": 0.7e-9},
        "stages": [{"stage": "pre_roll", "t_start": 0.0, "t_end": 0.1},
                   {"stage": "z_ramp_1_plunge", "t_start": 0.1, "t_end": 0.2}],
        "jumps": {"z": {"events": []}, "current": {"events": []}},
        "timing": {"n_z": 4, "fs_z_hz": 40},
    }


class _Buf:
    def __init__(self):
        self._n = 0
        self.events = []

    def next_seq(self):
        self._n += 1
        return self._n

    def emit_event(self, ev):
        self.events.append(ev)


def test_hook_renders_registers_and_emits(tmp_path):
    calls = {"begin": 0, "succeed": 0, "register": 0, "record_scan": 0}

    class _Actions:
        def begin(self, **kw):
            calls["begin"] += 1
            assert kw["action_type"] == "TipShapeWithReadback"
            return "act-1"

        def succeed(self, aid, **kw):
            calls["succeed"] += 1
            assert aid == "act-1"

    class _ScanFiles:
        def register(self, **kw):
            calls["register"] += 1
            assert kw["format_kind"] == "png" and kw["produced_by_action_id"] == "act-1"
            assert kw["sha256"] and kw["size_bytes"] > 0
            return "sf-1"

    class _Obs:
        def record_scan(self, **kw):
            calls["record_scan"] += 1
            assert kw["scan_file_id"] == "sf-1" and kw["observable"] == "tip_shape_verdict"
            # result_summary must be small scalars (4KiB cap), not the full traces
            assert "samples_a" not in str(kw["result_summary"])

    class _Repos:
        actions = _Actions()
        scan_files = _ScanFiles()
        observations = _Obs()

    buf = _Buf()
    hook = make_tip_shape_post_hook(
        repos=_Repos(), experiment_id_getter=lambda: "exp-1",
        buffer_getter=lambda: buf, artifacts_dir=str(tmp_path))

    out = hook("TipShapeWithReadback", _data(), None)
    assert out and out["scan_paths"]
    png = out["scan_paths"][0]
    assert Path(png).exists()
    assert calls == {"begin": 1, "succeed": 1, "register": 1, "record_scan": 1}
    assert len(buf.events) == 1
    ev = buf.events[0]
    assert ev.kind.value == "tip_shape_verdict"
    assert ev.payload["file_path"] == png and ev.payload["verdict"] == "no_change"


def test_hook_ignores_other_skills(tmp_path):
    hook = make_tip_shape_post_hook(
        repos=None, experiment_id_getter=lambda: None,
        buffer_getter=lambda: None, artifacts_dir=str(tmp_path))
    assert hook("GetBias", {"bias_v": 1.0}, None) is None
    assert hook("TipShapeWithReadback", {}, None) is None  # no indent → skip


def test_hook_does_not_record_unrelated_skills(tmp_path):
    """The hook is NOT the records writer any more ().

    It used to open a v2 action row for every skill that passed through it —
    but wrap_skill only calls a post_hook on SUCCESS, so that seam turned the
    v2 store into a success-only ledger (22 rows, all 'succeeded', for a
    session with 5 real failures). Recording moved to the recorder seam
    (CoreRuntime._record_v2_action), which fires on failure too."""
    seen = {"begin": [], "succeed": 0}

    class _Actions:
        def begin(self, **kw):
            seen["begin"].append(kw["action_type"])
            return "a"

        def succeed(self, aid, **kw):
            seen["succeed"] += 1

    class _Repos:
        actions = _Actions()

    hook = make_tip_shape_post_hook(
        repos=_Repos(), experiment_id_getter=lambda: "e",
        buffer_getter=lambda: None, artifacts_dir=str(tmp_path))
    assert hook("GetBias", {"bias_v": 1.0}, None) is None      # non-tip → no scan_paths
    assert hook("SetBias", {}, None) is None
    assert seen["begin"] == [] and seen["succeed"] == 0


def test_hook_reuses_the_action_id_wrap_skill_hands_over(tmp_path):
    """No duplicate action row for the same skill call: wrap_skill records the
    call first and passes the id on the context; the figure attaches to it."""
    class _Ctx:
        records_action_id = "real-action-1"

    seen = {"begin": 0, "register_parent": None}

    class _Actions:
        def begin(self, **kw):
            seen["begin"] += 1
            return "duplicate-row"

        def succeed(self, aid, **kw):
            pass

    class _ScanFiles:
        def register(self, **kw):
            seen["register_parent"] = kw["produced_by_action_id"]
            return "sf-1"

    class _Obs:
        def record_scan(self, **kw):
            pass

    class _Repos:
        actions = _Actions()
        scan_files = _ScanFiles()
        observations = _Obs()

    hook = make_tip_shape_post_hook(
        repos=_Repos(), experiment_id_getter=lambda: "e",
        buffer_getter=lambda: None, artifacts_dir=str(tmp_path))
    out = hook("TipShapeWithReadback", _data(), _Ctx())
    assert out and Path(out["scan_paths"][0]).exists()
    assert seen["begin"] == 0, "must not open a second action row"
    assert seen["register_parent"] == "real-action-1"


def test_hook_survives_records_failure(tmp_path):
    """A DB failure must not stop the render + buffer emit (best-effort)."""
    class _Actions:
        def begin(self, **kw):
            raise RuntimeError("db down")

    class _Repos:
        actions = _Actions()

    buf = _Buf()
    hook = make_tip_shape_post_hook(
        repos=_Repos(), experiment_id_getter=lambda: "e",
        buffer_getter=lambda: buf, artifacts_dir=str(tmp_path))
    out = hook("TipShapeWithReadback", _data(), None)
    assert out and Path(out["scan_paths"][0]).exists()   # render survived
    assert len(buf.events) == 1                            # buffer emit survived


def test_hook_no_buffer_no_repos_still_renders(tmp_path):
    hook = make_tip_shape_post_hook(
        repos=None, experiment_id_getter=lambda: None,
        buffer_getter=lambda: None, artifacts_dir=str(tmp_path))
    out = hook("TipShapeWithReadback", _data(), None)
    assert out and Path(out["scan_paths"][0]).exists()


# ── real v2 store (validates the actual repos API signatures end-to-end) ──

def test_open_live_v2_repos_api_roundtrip(tmp_path, monkeypatch):
    """open_live_v2 + the exact repos calls the hook makes, against a REAL
    ExperimentStoreV2 (tmp). Catches any signature drift the fake repos can't."""
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    import hashlib
    from mast.logging.v2.live import open_live_v2

    repos, eid = open_live_v2()
    assert repos is not None and eid, "campaigns/samples/experiments API"
    aid = repos.actions.begin(
        experiment_id=eid, agent_id="instrument_control",
        action_type="TipShapeWithReadback", params={"verdict": "no_change"})
    assert aid
    repos.actions.succeed(aid)
    png = tmp_path / "f.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    sfid = repos.scan_files.register(
        produced_by_action_id=aid,
        sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
        size_bytes=png.stat().st_size, current_path=str(png),
        format_kind="png", parser_spec="test", meta={"verdict": "no_change"})
    assert sfid
    oid = repos.observations.record_scan(
        action_id=aid, experiment_id=eid, observable="tip_shape_verdict",
        scan_file_id=sfid, result_summary={"verdict": "no_change", "delta_m": -1e-13})
    assert oid


def test_hook_writes_to_real_v2_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    from mast.logging.v2.live import open_live_v2
    repos, eid = open_live_v2()
    assert repos is not None and eid
    buf = _Buf()
    hook = make_tip_shape_post_hook(
        repos=repos, experiment_id_getter=lambda: eid,
        buffer_getter=lambda: buf, artifacts_dir=str(tmp_path / "art"))
    out = hook("TipShapeWithReadback", _data(), None)
    assert out and Path(out["scan_paths"][0]).exists()
    assert len(buf.events) == 1 and buf.events[0].payload["verdict"] == "no_change"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
