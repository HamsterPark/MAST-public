"""界面列的工具表，必须**就是**模型手上那份。

``webui/agents_api.py`` 里那段注释把这件事说得很直白：一份不同门的清单「成了同一件
事的另一种说谎方式」。在订阅门装上之前，那段是手抄的第二份并集 —— 于是第三道门加
进来的时候，它会静默地保持两道门，而界面照常显示一个看起来很正常的工具表。

所以这里比的不是「某个名字在不在」，而是**集合相等**：镜像分叉是集合病，抽查一个
名字的测试可以在分叉了一半的时候仍然全绿。
"""

from __future__ import annotations

import pytest

from mast.agents.instrument_control.tools import (
    build_instrument_skill_tools,
    discover_instrument_skills,
)
from mast.skills import hardware_modules as hm
from mast.skills import subscription as sub
from mast.webui import agents_api


@pytest.fixture(scope="module")
def registry():
    return discover_instrument_skills()


@pytest.fixture(autouse=True)
def _clean(subscription_store, registry):
    agents_api.set_live_registry(registry)
    agents_api.invalidate_agent_tools()
    yield
    agents_api.set_live_registry(None)
    agents_api.invalidate_agent_tools()


def _mirror_names() -> set[str]:
    """镜像里的 IC 技能名（去掉 buffer / handoff 这些非注册表工具）。

    走 ``_compute_agent_tools()`` 而不是 ``get_agent_tools()`` —— 后者只读缓存、
    从不计算（"NEVER computes/blocks"），拿它测会得到一个空 dict 的假绿。
    """
    agents_api.invalidate_agent_tools()
    listed = {t["name"] for t in agents_api._compute_agent_tools()["instrument_control"]}
    infra = set(agents_api._BUFFER_TOOL_NAMES) | {
        f"handoff_to_{t}" for t in agents_api._HANDOFF_TARGETS["instrument_control"]}
    return listed - infra


def _built_names(registry) -> set[str]:
    return {t.name for t in build_instrument_skill_tools(registry, lambda: None)}


def test_mirror_applies_the_subscription_gate(registry):
    all_names = {m.name for m in registry.list_skills()}
    assert "SetBias" in _mirror_names(), "前提没立住：它本来就该在镜像里"

    sub.set_subscribed(all_names - {"SetBias"})
    assert "SetBias" not in _mirror_names(), (
        "退订之后界面仍然把它列在 instrument_control 的工具表里 —— "
        "agents_api 少了一道门，这份清单不是模型手上那份")


@pytest.mark.parametrize("drop", [0, 1, 5])
def test_mirror_equals_the_built_face(registry, drop):
    """集合相等，三种订阅规模各来一次。"""
    all_names = sorted(m.name for m in registry.list_skills())
    if drop:
        sub.set_subscribed(set(all_names) - set(all_names[:drop]))

    built = _built_names(registry)
    mirrored = _mirror_names()
    assert mirrored == built, (
        "界面镜像与真实装配面不一致。\n"
        f"  只在界面上：{sorted(mirrored - built)[:10]}\n"
        f"  只在模型手上：{sorted(built - mirrored)[:10]}")


def test_mirror_still_composes_with_the_hardware_gate(registry):
    """新加的门不许把旧的挤掉。"""
    hm.set_enabled({})
    try:
        mirrored = _mirror_names()
        for name in list(hm.SKILL_OWNER)[:5]:
            assert name not in mirrored, f"{name} 属于关闭的硬件模块，却仍在镜像里"
    finally:
        hm.set_enabled(hm.DEFAULT_ENABLED)


def test_uncustomised_mirror_is_the_full_face(registry):
    """零回归的界面半边。"""
    assert _mirror_names() == _built_names(registry)
    assert sub.is_customised() is False
