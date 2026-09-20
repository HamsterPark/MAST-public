"""``_get_metadata_raw`` vs ``_get_metadata`` —— 声明 vs 生效包络。

这两个方法看起来是冗余的，合并回去只要删四行。它们不是冗余的，区别是
**「技能作者声明了什么」** 和 **「此刻实际生效的是什么」**，而覆盖层的
「只能收紧不能放松」检查必须用前者。

反例（这就是为什么这条测试存在）：
    1. 管理员在 skill_overrides.json 里把内置 SetBias 从 dangerous 降到 confirm
       —— 这是一个有审计、有 PIN、有理由的动作，合法。
    2. 有人上传一个 overlay，也把 SetBias 声明成 confirm。
    3. 如果比对用的是 ``_get_metadata()``（已合并覆写），基线就是 confirm，
       overlay 的 confirm **合法通过**。
    4. 管理员事后撤销覆写 —— overlay 还在，SetBias 永久停在 confirm。

两层各自看都讲得通，合起来把审批门洗掉了。所以基线必须是**作者的声明**，
它不随管理员覆写移动。

Split 于 2026-08-19（skill-overlay 修复项）。相关：`mast/skills/overlay/checks.py`。
"""

from __future__ import annotations

import json

import pytest

from mast.admin.override_store import SKILL_OVERRIDES, ConfigOverrideRegistry
from mast.core.registry import SkillRegistry
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
)
from mast.skills.base import BaseSkill


class _DangerousProbe(BaseSkill):
    """一个只为这条测试存在的技能：声明 DANGEROUS + 一个有上界的参数。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="_RawMetadataProbe",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS,
            description="probe",
            parameters=[
                ParameterSpec(
                    name="bias_v", type="float", description="probe",
                    unit="V", required=True, min_value=-2.0, max_value=2.0,
                ),
            ],
        )

    def execute(self, context, params):  # pragma: no cover — never run
        raise AssertionError("probe skill must never execute")


@pytest.fixture
def relaxing_override(tmp_path):
    """一条把 probe 从 dangerous 降到 confirm、并放宽参数上界的管理员覆写。"""
    d = tmp_path / "config" / "overrides"
    d.mkdir(parents=True)
    (d / SKILL_OVERRIDES).write_text(
        json.dumps({
            "_RawMetadataProbe": {
                "safety_level": "confirm",
                "parameters": {"bias_v": {"max_value": 10.0}},
            }
        }),
        encoding="utf-8",
    )
    ConfigOverrideRegistry.reset()
    ConfigOverrideRegistry.get(d)
    yield d
    ConfigOverrideRegistry.reset()


def test_raw_metadata_ignores_admin_override(relaxing_override):
    """raw 报作者的声明；merged 报此刻生效的包络。两者必须不同。"""
    raw = SkillRegistry._get_metadata_raw(_DangerousProbe)
    merged = SkillRegistry._get_metadata(_DangerousProbe)

    assert raw.safety_level is SafetyLevel.DANGEROUS, (
        "raw 必须是作者声明的 DANGEROUS —— 它一旦跟着管理员覆写走，"
        "覆盖层的『只能收紧』就失去了不动的基线"
    )
    assert merged.safety_level is SafetyLevel.CONFIRM, (
        "merged 必须反映管理员覆写，否则这条覆写从来没生效过"
    )

    raw_bias = {p.name: p for p in raw.parameters}["bias_v"]
    merged_bias = {p.name: p for p in merged.parameters}["bias_v"]
    assert raw_bias.max_value == 2.0
    assert merged_bias.max_value == 10.0


def test_raw_and_merged_agree_when_no_override():
    """没有覆写时两者必须一致 —— 否则 raw 就不只是『少了一层』，是另一个东西。"""
    ConfigOverrideRegistry.reset()
    try:
        raw = SkillRegistry._get_metadata_raw(_DangerousProbe)
        merged = SkillRegistry._get_metadata(_DangerousProbe)
        assert raw == merged
    finally:
        ConfigOverrideRegistry.reset()


def test_registry_register_uses_the_merged_envelope(relaxing_override):
    """注册表对外仍然报生效包络 —— 修复项 只是把 raw 露出来，没有改变默认。

    这条钉住「拆分不改变现有行为」。把 ``register``/``list_skills`` 改成用
    raw，会让管理员的 DANGEROUS 升级在闸门上静默失效。
    """
    reg = SkillRegistry()
    reg.register(_DangerousProbe)
    (listed,) = [m for m in reg.list_skills() if m.name == "_RawMetadataProbe"]
    assert listed.safety_level is SafetyLevel.CONFIRM
