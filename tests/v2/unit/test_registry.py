"""SkillRegistry port tests — discovery + lookup + admin override."""
from __future__ import annotations

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

import pytest

from mast.admin.override_store import ConfigOverrideRegistry
from mast.core.registry import SkillRegistry
from mast.core.types import SafetyLevel, SkillCategory


class TestSkillRegistryDiscovery:
    def setup_method(self):
        ConfigOverrideRegistry.reset()
        self.registry = SkillRegistry()

    def test_discover_finds_bias_skills(self):
        n = self.registry.discover("mast.skills.builtins")
        # Phase 4 mini vendored 6 skills in bias.py
        assert n >= 6
        assert self.registry.has("GetBias")
        assert self.registry.has("SetBias")
        assert self.registry.has("GetCurrent")
        assert self.registry.has("SetBiasCalibration")

    def test_get_skill_by_name(self):
        self.registry.discover("mast.skills.builtins")
        cls = self.registry.get("GetBias")
        assert cls.__name__ == "GetBias"

    def test_get_skill_unknown_raises(self):
        with pytest.raises(KeyError):
            self.registry.get("NonExistentSkill")

    def test_list_skills_returns_metadata(self):
        self.registry.discover("mast.skills.builtins")
        skills = self.registry.list_skills()
        names = [s.name for s in skills]
        assert "GetBias" in names
        # SafetyLevel preserved
        get_bias = next(s for s in skills if s.name == "GetBias")
        assert get_bias.safety_level == SafetyLevel.AUTO
        # SetBiasCalibration was DANGEROUS pre-v0.3.22, now CONFIRM
        # (confirm pane never wired → DANGEROUS skills stalled silently)
        sbc = next(s for s in skills if s.name == "SetBiasCalibration")
        assert sbc.safety_level == SafetyLevel.CONFIRM


class TestSkillRegistryDescribe:
    def setup_method(self):
        ConfigOverrideRegistry.reset()
        self.registry = SkillRegistry()
        self.registry.discover("mast.skills.builtins")

    def test_describe_all_skills_markdown(self):
        out = self.registry.describe_skills()
        assert "# Available skills" in out
        assert "## GetBias" in out
        assert "## SetBias" in out
        assert "AUTO" in out  # GetBias safety level
        assert "CONFIRM" in out  # SetBiasCalibration (was DANGEROUS pre-v0.3.22)

    def test_describe_filter_by_safety_level(self):
        out_confirm = self.registry.describe_skills(safety_level=SafetyLevel.CONFIRM)
        # SetBiasCalibration (formerly DANGEROUS) now appears in CONFIRM filter
        assert "## SetBiasCalibration" in out_confirm
        assert "## GetBias" not in out_confirm

    def test_describe_filter_by_category(self):
        out = self.registry.describe_skills(category=SkillCategory.READ)
        assert "## GetBias" in out
        assert "## GetCurrent" in out
        # SetBias is WRITE → should be excluded
        assert "## SetBias " not in out  # space to avoid matching SetBiasCalibration


class TestSkillRegistryToolDefinitions:
    def setup_method(self):
        ConfigOverrideRegistry.reset()
        self.registry = SkillRegistry()
        self.registry.discover("mast.skills.builtins")

    def test_to_tool_definitions_format(self):
        tools = self.registry.to_tool_definitions()
        # Each tool has Claude tool_use shape
        for t in tools:
            assert "name" in t
            assert "input_schema" in t
            assert t["input_schema"]["type"] == "object"

    def test_set_bias_tool_def_has_required_param(self):
        tools = self.registry.to_tool_definitions()
        set_bias = next(t for t in tools if t["name"] == "SetBias")
        assert "bias_v" in set_bias["input_schema"]["required"]
        assert set_bias["input_schema"]["properties"]["bias_v"]["type"] == "number"
        assert set_bias["input_schema"]["properties"]["bias_v"]["minimum"] == -10.0
        assert set_bias["input_schema"]["properties"]["bias_v"]["maximum"] == 10.0


class TestSkillRegistryAdminOverride:
    def setup_method(self):
        ConfigOverrideRegistry.reset()

    def test_admin_can_lower_set_bias_max(self, tmp_path):
        ovr_dir = tmp_path / "overrides"
        ovr_dir.mkdir()
        # Override SetBias parameters: bias_v.max_value = 3.0
        (ovr_dir / "skill_overrides.json").write_text(
            '{"SetBias": {"parameters": {"bias_v": {"max_value": 3.0}}}}',
            encoding="utf-8",
        )
        ConfigOverrideRegistry.reset()
        # Populate singleton via classmethod (constructor alone doesn't set _instance)
        ConfigOverrideRegistry.get(overrides_dir=ovr_dir)
        registry = SkillRegistry()
        registry.discover("mast.skills.builtins")
        meta = next(s for s in registry.list_skills() if s.name == "SetBias")
        bias_param = next(p for p in meta.parameters if p.name == "bias_v")
        assert bias_param.max_value == 3.0  # overridden from default 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
