"""订阅门 —— holder 语义、装配过滤、以及那条**零回归**钉子。

这个功能的头号风险不是「过滤错了」，是「上线当天悄悄改变了工具面」。全量默认
（``customised=False`` ⇒ 恒真门）是一句承诺，而承诺要能被验证：
:func:`test_zero_regression_wrap_fingerprint_is_unchanged` 拿 wrap 指纹逐位比对
「订阅门存在」与「订阅门不参与」两种算法的结果 —— 指纹覆盖全集（名字+模块+版本），
不是抽查。

第二条风险是**镜像分叉**：并集在三处被用（装配 / 生效探针 / UI 镜像），漏改一处
都不报错。这里钉住前两处同源（第三处在 tests/v2/unit/gui/）。
"""

from __future__ import annotations

import pytest

from mast.agents.instrument_control.tools import (
    LAST_WRAP,
    build_instrument_skill_tools,
    discover_instrument_skills,
    expected_wrap_fingerprint,
)
from mast.skills import advanced_capabilities as ac
from mast.skills import hardware_modules as hm
from mast.skills import subscription as sub
from mast.skills import tool_face
from mast.skills.overlay.provenance import fingerprint, registry_triples


@pytest.fixture(scope="module")
def registry():
    return discover_instrument_skills()


@pytest.fixture(autouse=True)
def _isolated(subscription_store):
    """每个测试一个一次性 store（进程级 holder 会漏给下一个测试）。"""
    yield


def _ctx():
    return None


def _face(registry) -> set[str]:
    return {t.name for t in build_instrument_skill_tools(registry, _ctx)}


# ─────────────────────────────────────────────────────────────────────────────
# absent = 全订阅（零回归）
# ─────────────────────────────────────────────────────────────────────────────

def test_absent_means_everything_subscribed():
    assert sub.is_customised() is False
    assert sub.subscribed_names() is None, "未定制必须返回 None，不是空集"
    assert sub.unloaded_skill_names(["A", "B"]) == frozenset()


def test_zero_regression_wrap_fingerprint_is_unchanged(registry):
    """未定制态下，工具面与「这个功能不存在」时**逐位相同**。

    比的是 wrap 指纹（名字+模块+版本的全集哈希），不是计数 —— 计数相等而内容不同
    是这类事故的常见形态。
    """
    build_instrument_skill_tools(registry, _ctx)
    with_gate = LAST_WRAP["fingerprint"]

    # 「订阅门不参与」的算法：只有原来那两道门。
    legacy_skip = hm.disabled_skill_names() | ac.disabled_skill_names()
    triples = [t for t in registry_triples(registry) if t[0] not in legacy_skip]
    without_gate = fingerprint(triples)

    assert with_gate == without_gate, (
        "出厂态（未定制）的工具面与订阅功能不存在时不一样 —— 零回归承诺已破。")


def test_uncustomised_holder_ignores_a_huge_universe():
    """恒真门：给多大的全集都不过滤。"""
    assert sub.unloaded_skill_names([f"S{i}" for i in range(500)]) == frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# 定制之后
# ─────────────────────────────────────────────────────────────────────────────

def test_customised_list_unloads_exactly_the_complement():
    universe = {"A", "B", "C", "D"}
    sub.set_subscribed({"A", "B"})
    assert sub.is_customised() is True
    assert sub.unloaded_skill_names(universe) == {"C", "D"}
    assert sub.subscribed_names() == {"A", "B"}


def test_unsubscribing_removes_it_from_the_built_face(registry):
    """先证正例再证负例 —— 负例测试必须先证明自己会红。"""
    victim = "SetBias"
    assert any(m.name == victim for m in registry.list_skills())

    before = _face(registry)
    assert victim in before, "正例先立：订阅门装上之前它本来就在工具面上"

    all_names = {m.name for m in registry.list_skills()}
    sub.unsubscribe([victim], all_names=all_names)
    after = _face(registry)

    assert victim not in after, "退订之后它仍然在 agent 工具面上"
    assert before - after == {victim}, "退订一个技能，动的却不止那一个"


def test_new_skill_after_customisation_lands_in_market_not_in_face():
    """定制之后新出现的技能只进市场 —— 这是 materialise 的副产品，不是开关。"""
    sub.subscribe([], all_names={"A", "B"})        # materialise 到 {A,B}
    assert sub.is_customised() is True
    # registry 后来多了一个 C（overlay/热注册/agent 自建都长这样）
    assert sub.unloaded_skill_names({"A", "B", "C"}) == {"C"}
    assert sub.is_subscribed("C") is False


def test_reset_goes_back_to_everything():
    sub.set_subscribed({"A"})
    assert sub.unloaded_skill_names({"A", "B"}) == {"B"}
    sub.reset_to_default()
    assert sub.is_customised() is False
    assert sub.unloaded_skill_names({"A", "B"}) == frozenset()


def test_first_mutation_without_a_universe_is_refused():
    """不知道全集就 materialise = 把没列出来的全部退订。宁可拒绝。"""
    res = sub.unsubscribe(["A"], all_names=None)
    assert res["ok"] is False
    assert sub.is_customised() is False, "拒绝之后不该留下半个定制态"


# ─────────────────────────────────────────────────────────────────────────────
# 必装豁免集
# ─────────────────────────────────────────────────────────────────────────────

def test_mandatory_skills_cannot_be_unsubscribed():
    universe = set(sub.MANDATORY_SKILLS) | {"A"}
    sub.set_subscribed(universe)
    res = sub.unsubscribe(sorted(sub.MANDATORY_SKILLS), all_names=universe)
    assert res["ok"] is True
    assert set(res["skipped_mandatory"]) == set(sub.MANDATORY_SKILLS), (
        "必装项被跳过了，但没有如实报告 —— 静默不作为比拒绝更糟")
    assert sub.unloaded_skill_names(universe) == frozenset()


def test_mandatory_survives_an_imported_list_that_omits_them():
    """别人手改过的 manifest 也砍不掉必装项（豁免不依赖 entries 的内容）。"""
    universe = set(sub.MANDATORY_SKILLS) | {"A", "B"}
    sub.set_subscribed({"A"})            # entries 里一个必装项都没有
    assert sub.unloaded_skill_names(universe) == {"B"}
    for name in sub.MANDATORY_SKILLS:
        assert sub.is_subscribed(name) is True


def test_mandatory_names_all_exist_in_the_registry(registry):
    """拼错一个名字 = 豁免了空气。这条测试是这张手工名单的对账者。"""
    real = {m.name for m in registry.list_skills()}
    missing = sorted(sub.MANDATORY_SKILLS - real)
    assert not missing, (
        f"MANDATORY_SKILLS 里这些技能名在注册表里不存在：{missing}\n"
        "名字错了就等于没豁免，而界面上「必装」徽标照样显示。")


def test_mandatory_names_the_stop_and_retract_family():
    """点名钉住：这张名单的判据是「abort 之后仍然允许的动词」。"""
    for name in ("SafeRetract", "WithdrawTip", "StopScan", "StopMotor"):
        assert name in sub.MANDATORY_SKILLS


def test_mandatory_does_not_exempt_the_hardware_gate(registry):
    """必装**只豁免订阅门**。给一台没有的硬件保留工具位，正是 hardware_modules
    开头反对的事。"""
    owned_by_a_module = set(sub.MANDATORY_SKILLS) & set(hm.SKILL_OWNER)
    if not owned_by_a_module:
        pytest.skip("当前没有必装技能属于可选硬件模块")
    hm.set_enabled({})
    try:
        skip = tool_face.compute({m.name for m in registry.list_skills()}).names
        for name in owned_by_a_module:
            assert name in skip, f"{name} 属于关闭的硬件模块，必装不该把它放进来"
    finally:
        hm.set_enabled(hm.DEFAULT_ENABLED)


# ─────────────────────────────────────────────────────────────────────────────
# 坏文件：fail-open 到全订阅，而不是空订阅
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("junk", [
    "not json at all",
    "[1, 2, 3]",
    '{"entries": "SetBias"}',
    '{"entries": 17, "customised": true}',
    "null",
])
def test_unreadable_file_falls_back_to_everything_not_nothing(subscription_store, junk):
    """方向与 hardware_modules 的 fail-closed **相反**，而且是故意的。

    订阅不是保护：往「全关」兜换不来任何安全，只换来「agent 一夜之间失能」。
    但 fail-open ≠ 装作没事 —— ``unreadable`` 必须能与「文件不存在」区分开。
    """
    subscription_store.parent.mkdir(parents=True, exist_ok=True)
    subscription_store.write_text(junk, encoding="utf-8")
    sub.reset_default_store()

    assert sub.unloaded_skill_names({"A", "B"}) == frozenset(), (
        "订阅文件读不出来，却把工具面清空了")
    assert sub.unreadable_reason(), "读不出来必须留下原因（否则界面上什么都不会说）"


def test_missing_file_is_not_reported_as_unreadable(subscription_store):
    """不存在 = 还没人定制过（正常）；读不出来 = 有人配了但我们没看懂（要说）。"""
    assert not subscription_store.exists()
    assert sub.unreadable_reason() == ""
    assert sub.is_customised() is False


# ─────────────────────────────────────────────────────────────────────────────
# 与另外两道门的合成 + 基础设施工具
# ─────────────────────────────────────────────────────────────────────────────

def test_three_gates_compose(registry):
    all_names = {m.name for m in registry.list_skills()}
    hw_victim = next((s for s in hm.SKILL_OWNER if s in all_names), None)
    keep = all_names - {"SetBias"}
    sub.set_subscribed(keep)
    hm.set_enabled({})
    try:
        face = _face(registry)
        assert "SetBias" not in face, "订阅门没生效"
        if hw_victim:
            assert hw_victim not in face, "硬件门没生效"
    finally:
        hm.set_enabled(hm.DEFAULT_ENABLED)


def test_infrastructure_tools_are_never_filtered(registry):
    """buffer / handoff 不是注册表技能 ⇒ 差集碰不到它们。

    这是**结构性**豁免，不是又一张名单：订阅门只对 registry 名字取补集。
    """
    sub.set_subscribed(set())            # 退订一切
    tools = build_instrument_skill_tools(registry, _ctx)
    assert {t.name for t in tools} <= set(sub.MANDATORY_SKILLS), (
        "退订一切之后，工具面上除了必装项还剩别的注册表技能")
    # 基础设施工具走的是另一条路（build_tools 里 + buffer/handoff），
    # 这里只钉住「订阅门没有把它们算进自己的全集」。
    assert sub.unloaded_skill_names({m.name for m in registry.list_skills()}) \
        .isdisjoint({"handoff_to_supervisor", "read_latest_tip_status"})


# ─────────────────────────────────────────────────────────────────────────────
# 生效探针必须与装配同门（漏改这里 = fingerprint_matches 永远说谎）
# ─────────────────────────────────────────────────────────────────────────────

def test_expected_fingerprint_applies_the_subscription_gate_too(registry):
    all_names = {m.name for m in registry.list_skills()}
    sub.set_subscribed(all_names - {"SetBias"})

    build_instrument_skill_tools(registry, _ctx)
    exp, _ovl = expected_wrap_fingerprint(registry)

    assert LAST_WRAP["fingerprint"] == exp, (
        "装配算的面与探针算的面不一致 —— 只给 build 加了订阅门、忘了 "
        "expected_wrap_fingerprint，界面会把「已生效」永远显示成「没跟上」。")
