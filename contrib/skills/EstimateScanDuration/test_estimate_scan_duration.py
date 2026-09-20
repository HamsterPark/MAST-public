"""EstimateScanDuration 的单元测试 —— 纯计算，不需要仪器，也不需要执行上下文。

技能按**文件路径**加载（``skill.py`` 就在旁边），不依赖 contrib 在 sys.path 上：
装到 ``config/custom_skills/`` 之后它也是这样被单独加载的。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mast.core.types import SafetyLevel, SkillCategory

_HERE = Path(__file__).resolve().parent


def _load_skill_class():
    spec = importlib.util.spec_from_file_location("contrib_estimate_scan_duration", _HERE / "skill.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EstimateScanDuration


Skill = _load_skill_class()


class _NoInstrument:
    """任何硬件访问都当场失败 —— 证明这个技能确实不碰仪器。"""

    def safe_call(self, *args, **kwargs):
        raise AssertionError("纯计算技能不该发 Nanonis 命令")

    def run(self, *args, **kwargs):
        raise AssertionError("纯计算技能不该调子技能")


def test_frame_time_is_lines_times_forward_plus_backward() -> None:
    r = Skill().execute(_NoInstrument(), {"width_m": 100e-9, "lines": 256, "speed_m_per_s": 200e-9})
    assert r.success, r.error
    assert r.data["line_time_s"] == pytest.approx(1.0)       # 2 × 100 nm ÷ 200 nm/s
    assert r.data["frame_time_s"] == pytest.approx(256.0)
    assert "4 min 16 s" in (r.summary or "")


def test_accepts_si_prefixed_strings() -> None:
    """agent 路径与直调路径都会先把 '100n' 还原成数；技能自己也认，旧调用方同样可用。"""
    r = Skill().execute(_NoInstrument(), {"width_m": "100n", "lines": 256, "speed_m_per_s": "200n"})
    assert r.success, r.error
    assert r.data["frame_time_s"] == pytest.approx(256.0)


def test_overhead_per_line_adds_up() -> None:
    r = Skill().execute(_NoInstrument(), {"width_m": 100e-9, "lines": 256, "speed_m_per_s": 200e-9,
                                          "overhead_s_per_line": 0.5})
    assert r.data["line_time_s"] == pytest.approx(1.5)
    assert r.data["frame_time_s"] == pytest.approx(384.0)


def test_pixels_change_dwell_not_duration() -> None:
    a = Skill().execute(_NoInstrument(), {"width_m": 100e-9, "lines": 64, "speed_m_per_s": 200e-9, "pixels": 256})
    b = Skill().execute(_NoInstrument(), {"width_m": 100e-9, "lines": 64, "speed_m_per_s": 200e-9, "pixels": 512})
    assert a.data["frame_time_s"] == pytest.approx(b.data["frame_time_s"])
    assert b.data["pixel_dwell_s"] == pytest.approx(a.data["pixel_dwell_s"] / 2)
    assert b.data["pixel_dwell_s"] == pytest.approx(0.5 / 512)


@pytest.mark.parametrize("bad", [
    {"width_m": 100e-9, "lines": 256, "speed_m_per_s": 0.0},
    {"width_m": -1e-9, "lines": 256, "speed_m_per_s": 200e-9},
    {"width_m": 100e-9, "lines": 0, "speed_m_per_s": 200e-9},
    {"width_m": "not a number", "lines": 256, "speed_m_per_s": 200e-9},
    {"lines": 256, "speed_m_per_s": 200e-9},
])
def test_bad_input_is_a_typed_failure_not_an_exception(bad) -> None:
    r = Skill().execute(_NoInstrument(), bad)
    assert r.success is False
    assert r.error


def test_metadata_declares_what_it_is() -> None:
    meta = Skill().metadata()
    assert meta.name == "EstimateScanDuration"
    assert meta.category is SkillCategory.ANALYSIS
    assert meta.safety_level is SafetyLevel.AUTO
    assert not meta.capabilities
    dimensioned = {p.name: p for p in meta.parameters if p.unit}
    assert set(dimensioned) == {"width_m", "speed_m_per_s", "overhead_s_per_line"}
    for p in dimensioned.values():
        assert p.min_value is not None and p.max_value is not None, p.name
