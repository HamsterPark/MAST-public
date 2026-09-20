"""Composite version store + template + flow-graph rendering tests.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_version_store.py -x -v
"""
from __future__ import annotations

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

import json
import threading

import pytest

from mast.skills.composite.spec import CompositeSpec, ParamSpec
from mast.skills.composite.version_store import (
    CompositeVersionStore, VersionConflictError, VersionStoreError,
)


def _spec(name="Demo", desc="d"):
    return CompositeSpec(
        name=name, description=desc, safety_level="confirm",
        params=[ParamSpec("n", "int", 2)],
        nodes=[{"type": "step", "id": "a", "skill": "GetBias", "params": {}}],
    )


class TestVersionStore:
    def _store(self, tmp_path):
        return CompositeVersionStore(root=tmp_path / "composite_skills")

    def test_save_bumps_version_and_archives(self, tmp_path):
        st = self._store(tmp_path)
        s = st.save(_spec())
        assert s.version == 1
        s2 = st.save(_spec(desc="d2"))
        assert s2.version == 2                       # bumped
        vers = st.list_versions("Demo")
        assert [v["version"] for v in vers] == [2, 1]  # newest first, both archived

    def test_load_current(self, tmp_path):
        st = self._store(tmp_path)
        st.save(_spec())
        st.save(_spec(desc="newer"))
        assert st.load("Demo").description == "newer"
        assert st.load("Demo").version == 2

    def test_restore_is_nondestructive(self, tmp_path):
        st = self._store(tmp_path)
        st.save(_spec(desc="v1"))
        st.save(_spec(desc="v2"))
        restored = st.restore("Demo", 1)             # roll back to v1's content
        assert restored.version == 3                 # written forward as v3
        assert st.load("Demo").description == "v1"
        assert [v["version"] for v in st.list_versions("Demo")] == [3, 2, 1]

    def test_invalid_spec_rejected(self, tmp_path):
        st = self._store(tmp_path)
        bad = CompositeSpec(name="Bad", nodes=[{"type": "step", "id": "x"}])  # no skill
        with pytest.raises(VersionStoreError):
            st.save(bad)

    def test_bad_name_rejected(self, tmp_path):
        st = self._store(tmp_path)
        with pytest.raises(VersionStoreError):
            st.save(CompositeSpec(name="../evil", nodes=[]))

    def test_clone(self, tmp_path):
        st = self._store(tmp_path)
        st.save(_spec())
        st.save(_spec(desc="v2"))
        c = st.clone("Demo", "DemoCopy", author="me")
        assert c.name == "DemoCopy" and c.version == 1 and c.author == "me"
        assert st.exists("DemoCopy")
        with pytest.raises(VersionStoreError):
            st.clone("Demo", "DemoCopy")             # already exists

    def test_diff(self, tmp_path):
        st = self._store(tmp_path)
        st.save(_spec())
        s2 = _spec(desc="changed")
        s2.nodes.append({"type": "step", "id": "b", "skill": "SetBias", "params": {}})
        st.save(s2)
        d = st.diff("Demo", 1, 2)
        assert "b" in d["added"]
        assert "description" in d["fields"]

    def test_list_specs(self, tmp_path):
        st = self._store(tmp_path)
        st.save(_spec("A"))
        st.save(_spec("B"))
        names = {s["name"] for s in st.list_specs()}
        assert names == {"A", "B"}


class TestTemplates:
    def test_builtin_templates_valid(self):
        from mast.skills.composite.templates import builtin_templates
        for spec in builtin_templates():
            assert spec.validate() == [], f"{spec.name}: {spec.validate()}"

    def test_seed(self, tmp_path):
        from mast.skills.composite.templates import seed_templates
        st = CompositeVersionStore(root=tmp_path / "cs")
        seeded = seed_templates(st)
        assert "ConditionTipUntilSharp" in seeded
        assert seed_templates(st) == []   # idempotent — already present

    def test_template_executes(self):
        # the while-loop template should run end-to-end against a fake context.
        #
        # NOTE ( 2026-07-27): this used to stub "AssessTipQuality" and
        # assert a "LogNote" call. Neither skill has ever existed — the fake
        # context answered to any name, so this test passed for months while the
        # real loader refused to register the template at every startup. Stub the
        # skills the template ACTUALLY names, and keep the guard that resolves
        # them against a live registry in
        # tests/v2/unit/skills/test_composite_templates_real_skills.py.
        from mast.skills.composite.interpreter import SpecComposite
        from mast.skills.composite.templates import builtin_templates
        spec = next(s for s in builtin_templates() if s.name == "ConditionTipUntilSharp")

        class _Res:
            def __init__(s, data): s.success = True; s.data = data; s.error = ""; s.nanonis_calls = []
        calls = []
        # AssessImageQuality returns fft_quality (0..1); 0.8 clears the default.
        quality = {"v": 0.8}   # tip becomes sharp on the first measure

        class Ctx:
            def run(s, skill, params):
                calls.append(skill)
                if skill == "AssessImageQuality":
                    return _Res({"fft_quality": quality["v"]})
                return _Res({})
        res = SpecComposite(spec).execute(Ctx(), {
            "pulse_v": 3.0, "center_x_m": 0.0, "center_y_m": 0.0,
            "check_scan_m": 2e-8, "quality_threshold": 0.3, "max_attempts": 5})
        assert res.success
        # one pulse + re-scan + assess (sharp immediately); the 'accept' branch is
        # a `succeed` node, so it ends the run instead of calling a sixth skill.
        assert calls == ["BiasPulse", "FullScan", "AssessImageQuality"]

    def test_template_gives_up_after_max_attempts(self):
        """The else-branch is a `fail` node — a blunt tip must not report success."""
        from mast.skills.composite.interpreter import SpecComposite
        from mast.skills.composite.templates import builtin_templates
        spec = next(s for s in builtin_templates() if s.name == "ConditionTipUntilSharp")

        class _Res:
            def __init__(s, data): s.success = True; s.data = data; s.error = ""; s.nanonis_calls = []

        class Ctx:
            def run(s, skill, params):
                if skill == "AssessImageQuality":
                    return _Res({"fft_quality": 0.01})   # never good enough
                return _Res({})
        res = SpecComposite(spec).execute(Ctx(), {
            "pulse_v": 3.0, "center_x_m": 0.0, "center_y_m": 0.0,
            "check_scan_m": 2e-8, "quality_threshold": 0.3, "max_attempts": 2})
        assert res.success is False


class TestGraphRender:
    def test_build_graph_if_loop(self):
        from mast.skills.composite.graph_render import build_graph, to_mermaid
        spec = CompositeSpec(name="D", nodes=[
            {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
            {"type": "if", "id": "c", "cond": "n > 1",
             "then": [{"type": "step", "id": "p", "skill": "Pulse", "params": {}}],
             "else": [{"type": "step", "id": "s", "skill": "Single", "params": {}}]},
            {"type": "loop", "id": "lp", "mode": "repeat", "count": "3", "var": "i",
             "body": [{"type": "step", "id": "m", "skill": "Move", "params": {}}]},
        ])
        g = build_graph(spec)
        kinds = [n["kind"] for n in g["nodes"]]
        assert "if" in kinds and "loop" in kinds and "start" in kinds and "end" in kinds
        # a back-edge exists for the loop
        assert any(e["label"] == "循环" for e in g["edges"])
        # branch labels present
        assert any(e["label"] == "是" for e in g["edges"])
        mm = to_mermaid(g)
        assert mm.startswith("flowchart TD")
        assert "{" in mm and "}" in mm   # decision diamond rendered

    def test_steps_to_graph(self):
        from mast.skills.composite.graph_render import steps_to_graph, to_mermaid
        from mast.skills.composite.graph_executor import CompositeStep
        steps = [CompositeStep("s1", "GetBias", {}),
                 CompositeStep("s2", "SetBias", {"bias_v": 1.0})]
        g = steps_to_graph(steps)
        assert len([n for n in g["nodes"] if n["kind"] == "step"]) == 2
        assert to_mermaid(g).startswith("flowchart TD")

    def test_spec_to_html_offline(self):
        from mast.skills.composite.graph_render import spec_to_html
        from mast.skills.composite.templates import builtin_templates
        for spec in builtin_templates():
            html = spec_to_html(spec)
            assert "开始" in html and "完成" in html
            assert "<script" not in html.lower()   # offline, no external JS
        # control flow markers present for the while-template
        cond = next(s for s in builtin_templates() if s.name == "ConditionTipUntilSharp")
        html = spec_to_html(cond)
        assert "循环" in html and "条件" in html      # loop + if rendered

    def test_spec_to_html_escapes(self):
        from mast.skills.composite.graph_render import spec_to_html
        spec = CompositeSpec(name="X", nodes=[
            {"type": "step", "id": "a", "skill": "<b>x</b>", "params": {}}])
        assert "&lt;b&gt;" in spec_to_html(spec)     # HTML-escaped (no injection)


class TestLoader:
    def test_load_registers_runnable_skill(self, tmp_path):
        from mast.core.registry import SkillRegistry
        from mast.skills.composite.loader import load_spec_skills
        st = CompositeVersionStore(root=tmp_path / "cs")
        st.save(_spec("MyComposite"))
        reg = SkillRegistry()
        loaded = load_spec_skills(reg, st, seed=False)
        assert "MyComposite" in loaded
        assert reg.has("MyComposite")
        # the registered class is instantiable + runs as a SpecComposite
        cls = reg.get("MyComposite")
        from mast.skills.composite.interpreter import SpecComposite
        assert issubclass(cls, SpecComposite)

    def test_seed_then_load(self, tmp_path):
        from mast.core.registry import SkillRegistry
        from mast.skills.composite.loader import load_spec_skills
        reg = SkillRegistry()
        st = CompositeVersionStore(root=tmp_path / "cs")
        loaded = load_spec_skills(reg, st, seed=True)
        assert "ConditionTipUntilSharp" in loaded
        assert reg.has("GridSTSWithPrecheck")


# ─────────────────────────────────────────────────────────────────────
# 修复项 (2026-06-11): optimistic concurrency (CAS) + per-root lock.
# Two browser tabs / LAN clients saving the same composite must never
# silently clobber each other or overwrite an "immutable" history snapshot.
# ─────────────────────────────────────────────────────────────────────

class TestConcurrency:
    def _store(self, tmp_path):
        return CompositeVersionStore(root=tmp_path / "composite_skills")

    def test_cas_rejects_stale_save(self, tmp_path):
        st = self._store(tmp_path)
        st.save(_spec(desc="v1"))                      # → v1
        st.save(_spec(desc="tab-A"), base_version=1)   # → v2
        with pytest.raises(VersionConflictError):
            st.save(_spec(desc="tab-B"), base_version=1)  # stale base
        assert st.load("Demo").description == "tab-A"  # winner intact

    def test_cas_passes_on_fresh_base(self, tmp_path):
        st = self._store(tmp_path)
        assert st.save(_spec(), base_version=0).version == 1   # no prior
        assert st.save(_spec(), base_version=1).version == 2

    def test_no_base_version_keeps_legacy_last_write_wins(self, tmp_path):
        st = self._store(tmp_path)
        st.save(_spec(desc="a"))
        s = st.save(_spec(desc="b"))                   # no kwarg — legacy path
        assert s.version == 2

    def test_concurrent_saves_never_clobber_history(self, tmp_path):
        """8 threads save at once: history must end up with 8 DISTINCT
        versions (1..8). Pre-lock, racers read the same prior version and the
        second os.replace overwrote the first's snapshot."""
        st = self._store(tmp_path)
        n = 8
        barrier = threading.Barrier(n)
        errors: list = []

        def worker(i):
            try:
                barrier.wait(timeout=10)
                st.save(_spec(desc=f"writer-{i}"))
            except Exception as exc:  # pragma: no cover — failure surface
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors
        versions = [v["version"] for v in st.list_versions("Demo")]
        assert versions == list(range(n, 0, -1))       # n..1, no gaps, no dupes
        assert st.load("Demo").version == n

    def test_history_snapshot_never_overwritten_by_external_file(self, tmp_path):
        """A foreign _history/<name>.v2.json (hand-copied, sync residue…) must
        survive — save() bumps past it instead of clobbering."""
        st = self._store(tmp_path)
        st.save(_spec(desc="v1"))
        foreign = (tmp_path / "composite_skills" / "_history" / "Demo.v2.json")
        foreign.parent.mkdir(parents=True, exist_ok=True)
        foreign.write_text(json.dumps({"name": "Demo", "marker": "foreign"}),
                           encoding="utf-8")
        s = st.save(_spec(desc="next"))
        assert s.version == 3                           # skipped occupied v2
        assert json.loads(foreign.read_text(encoding="utf-8"))["marker"] == "foreign"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
