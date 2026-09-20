"""Every composite template step must resolve against the actual registry, with valid skill names, parameters and required arguments."""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (canonical block for tests/v2/) ──
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

import pytest

from mast.core.registry import SkillRegistry
from mast.skills.composite.loader import (
    _collect_step_skills,
    last_rejected_specs,
    load_spec_skills,
)
from mast.skills.composite.templates import builtin_templates, seed_templates
from mast.skills.composite.version_store import CompositeVersionStore

# The names the specs used to reference. Kept literal so the test still means
# something if someone later adds a skill by one of these names.
NEVER_EXISTED = ("AssessTipQuality", "LogNote", "BiasSpectroscopy")

# The five specs the loader refused at every startup.
REFUSED_IN_FIELD = ("ConditionTipUntilSharp", "GridSTSWithPrecheck",
                    "LineProfileSTS", "ScanAssessRescan", "ScanThenConditionTip")


@pytest.fixture(scope="module")
def registry():
    r = SkillRegistry()
    r.discover()
    return r


@pytest.fixture(scope="module")
def skills_by_name(registry):
    return {m.name: m for m in registry.list_skills()}


def _step_nodes(nodes):
    """Every ``step`` node in a spec tree, including nested branches/bodies."""
    for n in nodes or []:
        if not isinstance(n, dict):
            continue
        if n.get("type") == "step":
            yield n
        for key in ("then", "else", "body", "finally", "on_error"):
            yield from _step_nodes(n.get(key) or [])
        for lst in (n.get("routes") or {}).values():
            yield from _step_nodes(lst or [])


# ──────────────────────────────────────────────────────────────────────
# Root cause: the names were invented, not removed
# ──────────────────────────────────────────────────────────────────────

def test_the_three_missing_skills_do_not_exist(registry):
    """Pins the diagnosis: spec was wrong, the skills were never there."""
    for name in NEVER_EXISTED:
        assert not registry.has(name), (
            f"{name} now exists — re-check whether the templates should use it")


def test_no_template_references_an_invented_name():
    referenced = set()
    for spec in builtin_templates():
        referenced |= _collect_step_skills(spec.nodes)
    assert referenced.isdisjoint(NEVER_EXISTED)


# ──────────────────────────────────────────────────────────────────────
# The guard: every template must resolve against the live registry
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", builtin_templates(), ids=lambda s: s.name)
class TestTemplateResolvesAgainstRegistry:

    def test_spec_is_structurally_valid(self, spec):
        assert spec.validate() == []

    def test_every_step_skill_exists(self, spec, registry):
        missing = sorted(s for s in _collect_step_skills(spec.nodes)
                         if not registry.has(s))
        assert missing == [], f"{spec.name} references non-existent {missing}"

    def test_every_step_param_is_a_real_param(self, spec, skills_by_name):
        """``MoveToXY(x=…)`` passed the loader and would have failed at runtime."""
        problems = []
        for node in _step_nodes(spec.nodes):
            md = skills_by_name.get(node["skill"])
            if md is None:
                continue
            allowed = {p.name for p in (md.parameters or [])}
            unknown = sorted(set((node.get("params") or {})) - allowed)
            if unknown:
                problems.append(f"{node.get('id')} -> {node['skill']}: {unknown}")
        assert problems == []

    def test_every_required_param_is_supplied(self, spec, skills_by_name):
        """``FullScan`` needs a centre and an extent; the specs passed ``{}``."""
        problems = []
        for node in _step_nodes(spec.nodes):
            md = skills_by_name.get(node["skill"])
            if md is None:
                continue
            required = {p.name for p in (md.parameters or []) if p.required}
            missing = sorted(required - set(node.get("params") or {}))
            if missing:
                problems.append(f"{node.get('id')} -> {node['skill']}: {missing}")
        assert problems == []


def test_result_keys_read_by_conditions_are_real(skills_by_name):
    """``assess['quality']`` — AssessImageQuality has never returned that key.

    Only the keys this suite can establish from source are pinned; the point is
    that a condition reading a made-up key is a runtime crash after hardware
    has already moved.
    """
    import re

    from mast.skills.composite import assess_quality

    src = Path(assess_quality.__file__).read_text(encoding="utf-8")
    produced = set(re.findall(r'data=\{"([a-z_]+)"', src))
    assert "fft_quality" in produced and "quality" not in produced

    for spec in builtin_templates():
        # repr() flattens the whole node tree, so nested conditions are covered.
        for key in re.findall(r"\w+\['([a-z_]+)'\]", repr(spec.nodes)):
            if key.endswith("quality"):
                assert key in produced, f"{spec.name} reads unknown key {key!r}"


# ──────────────────────────────────────────────────────────────────────
# End to end: the loader now registers all five
# ──────────────────────────────────────────────────────────────────────

def test_all_five_register_on_a_fresh_store(registry, tmp_path):
    loaded = load_spec_skills(registry, CompositeVersionStore(tmp_path / "fresh"))
    for name in REFUSED_IN_FIELD:
        assert name in loaded, f"{name} still not registered"
    assert [r for r in last_rejected_specs()
            if r["name"] in REFUSED_IN_FIELD] == []


def test_tip_repair_loops_are_callable(registry, tmp_path):
    """Tip-repair templates must be callable."""
    load_spec_skills(registry, CompositeVersionStore(tmp_path / "tip"))
    assert registry.has("ConditionTipUntilSharp")
    assert registry.has("ScanThenConditionTip")


def test_rejections_are_reportable_not_only_logged(registry, tmp_path):
    """UI could not tell "broken" from "never designed" — now it can ask."""
    store = CompositeVersionStore(tmp_path / "broken")
    spec = builtin_templates()[0]
    spec.name = "SpecWithGhostSkill"
    spec.nodes = [{"type": "step", "id": "ghost", "skill": "NoSuchSkillAtAll",
                   "params": {}}]
    store.save(spec)

    load_spec_skills(registry, store)
    rejected = {r["name"]: r for r in last_rejected_specs()}
    assert "SpecWithGhostSkill" in rejected
    assert rejected["SpecWithGhostSkill"]["reason"] == "unknown_skills"
    assert "NoSuchSkillAtAll" in rejected["SpecWithGhostSkill"]["missing_skills"]


# ──────────────────────────────────────────────────────────────────────
# Upgrades must reach installs that already stored the broken specs
# ──────────────────────────────────────────────────────────────────────

class TestStaleSeededTemplatesGetRepaired:

    @staticmethod
    def _plant_broken(store) -> None:
        """Write the pre-fix ConditionTipUntilSharp, verbatim in its essentials."""
        from mast.skills.composite.spec import CompositeSpec, ParamSpec

        store.save(CompositeSpec(
            name="ConditionTipUntilSharp",
            description="broken original",
            safety_level="confirm",
            params=[ParamSpec("pulse_v", "number", 3.0, "V", True)],
            nodes=[
                {"type": "step", "id": "pulse", "skill": "BiasPulse",
                 "params": {"bias_v": {"$expr": "pulse_v"}, "width_s": 0.1}},
                {"type": "step", "id": "q", "skill": "AssessTipQuality", "params": {}},
            ],
            tags=["tip", "conditioning", "template"],
        ))

    def test_stale_template_is_replaced(self, registry, tmp_path):
        """`if not exists: save` would have left this broken forever."""
        store = CompositeVersionStore(tmp_path / "stale")
        self._plant_broken(store)
        assert "AssessTipQuality" in _collect_step_skills(
            store.load("ConditionTipUntilSharp").nodes)

        seed_templates(store, registry)

        repaired = store.load("ConditionTipUntilSharp")
        assert "AssessTipQuality" not in _collect_step_skills(repaired.nodes)
        assert not [s for s in _collect_step_skills(repaired.nodes)
                    if not registry.has(s)]

    def test_prior_version_is_archived_not_destroyed(self, registry, tmp_path):
        store = CompositeVersionStore(tmp_path / "archive")
        self._plant_broken(store)
        seed_templates(store, registry)
        versions = store.list_versions("ConditionTipUntilSharp")
        assert len(versions) >= 2, "the operator must be able to roll back"

    def test_a_working_operator_spec_is_left_alone(self, registry, tmp_path):
        """Only unregisterable templates are repaired — never a working one."""
        from mast.skills.composite.spec import CompositeSpec

        store = CompositeVersionStore(tmp_path / "mine")
        store.save(CompositeSpec(
            name="ConditionTipUntilSharp",
            description="operator's own version",
            safety_level="confirm",
            nodes=[{"type": "step", "id": "p", "skill": "BiasPulse",
                    "params": {"bias_v": 3.0, "width_s": 0.1}}],
            tags=["tip", "template"],
        ))
        seed_templates(store, registry)
        assert store.load("ConditionTipUntilSharp").description == "operator's own version"

    def test_untagged_spec_is_never_touched(self, registry, tmp_path):
        """A spec the operator renamed/retagged is theirs, broken or not."""
        from mast.skills.composite.spec import CompositeSpec

        store = CompositeVersionStore(tmp_path / "untagged")
        store.save(CompositeSpec(
            name="ConditionTipUntilSharp",
            description="retagged by operator",
            safety_level="confirm",
            nodes=[{"type": "step", "id": "g", "skill": "AssessTipQuality",
                    "params": {}}],
            tags=["mine"],
        ))
        seed_templates(store, registry)
        assert store.load("ConditionTipUntilSharp").description == "retagged by operator"

    def test_fresh_store_still_seeds(self, registry, tmp_path):
        store = CompositeVersionStore(tmp_path / "empty")
        seeded = seed_templates(store, registry)
        assert set(seeded) == {s.name for s in builtin_templates()}
