"""按需加载工具：分包、检索、收窄的闸门。

## 这些测试在防什么

1. **提示词点名了一个模型看不见的工具。** IC 的系统提示词里写着「用 ScanAt」
   「调 ConditionTip」——如果这些不在核心包里，提示词就是在叫模型调一个它这一次
   收不到 schema 的东西。症状不是报错，是模型自己编一个名字或者绕路。
2. **收窄变成了门禁。** 这套机制的全部前提是「目录不是门禁」：ToolNode 仍然注册
   全部工具，任何工具都取得到。一旦哪次改动让某个包再也取不出来，就从「省
   token」变成了「削能力」，而 2026-08-20 的裁决明确否决过后者。
3. **中文查不到。** 工具名与 description 几乎全是英文，问问题的是中文。第一版
   只做英文分词，`search_tools("锁相放大器 调制幅度")` 零命中 —— 而零命中的
   agent 会去猜一个名字，正是这套机制要防的事。
4. **可见集回缩。** 取过的包必须留着：模型刚看见就丢会让它反复取；对 prompt
   cache 来说，只增不减才让工具前缀稳定。

Run from repo root::

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/prompts/test_tool_packs.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[4]
_MASTV2 = str(_REPO / "MASTv2")
if sys.path and sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)

from mast.agents._shared import tool_packs as tp  # noqa: E402
from mast.agents._shared.tool_visibility_mw import (  # noqa: E402
    STATE_KEY,
    ToolVisibilityMiddleware,
)


# ── 替身：一个小目录，形状与真的一样 ─────────────────────────────────────

def _meta(name, tags):
    return SimpleNamespace(name=name, tags=list(tags), description=f"{name} does things")


def _catalog(extra=()):
    tools = [SimpleNamespace(name=n) for n in
             ("ScanAt", "AcquirePLLFreqSweep", "ConfigureLockIn", "SetMotorFreqAmp",
              "handoff_to_supervisor", "search_tools", *extra)]
    metas = {
        "AcquirePLLFreqSweep": _meta("AcquirePLLFreqSweep", ["pll", "frequency"]),
        "ConfigureLockIn": _meta("ConfigureLockIn", ["lockin", "spectroscopy"]),
        "SetMotorFreqAmp": _meta("SetMotorFreqAmp", ["motor"]),
    }
    # 描述照真的写：真实的 description 里有 modulation / drive voltage 这些词，
    # 而检索正是靠它们区分「锁相调制幅度」与「马达驱动幅度」。替身写得太素，
    # 测出来的是替身的检索能力，不是真的那个。
    metas["ConfigureLockIn"].description = (
        "Configure the lock-in amplifier: modulation amplitude and frequency.")
    metas["SetMotorFreqAmp"].description = (
        "Set the coarse motor drive frequency and amplitude (drive voltage).")
    registry = SimpleNamespace(list_skills=lambda: list(metas.values()))
    return tp.build_catalog("instrument_control", tools, registry)


# ── 1. 核心包与提示词对账（最重要的一条）────────────────────────────────

_TOOL_NAME_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,})`|\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+){1,})\b")


def _tools_named_in_ic_prompt() -> set[str]:
    from mast.agents.instrument_control.prompts import SYSTEM_PROMPT

    found: set[str] = set()
    for m in _TOOL_NAME_RE.finditer(SYSTEM_PROMPT):
        found.add(m.group(1) or m.group(2))
    return found


def test_the_gate_actually_reads_the_prompt():
    """一个抓不到任何工具名的正则永远是绿的。先证明它抓到了东西。"""
    named = _tools_named_in_ic_prompt()
    assert len(named) >= 20, f"只从 IC 提示词里抓到 {len(named)} 个候选名，正则多半错了"
    assert "ScanAt" in named and "ConditionTip" in named


def test_every_tool_the_ic_prompt_names_is_visible_by_default():
    """提示词点名的工具必须在核心包里。

    否则提示词是在叫模型调一个它这一次收不到 schema 的东西 —— 而模型的反应
    不是报错，是编一个名字或者绕路。
    """
    from mast.agents.instrument_control.tools import discover_instrument_skills

    registry = discover_instrument_skills()
    real = {m.name for m in registry.list_skills()}
    named = _tools_named_in_ic_prompt() & real          # 只管真存在的工具
    missing = sorted(n for n in named
                     if tp.CORE not in tp.classify(n, None)
                     and not n.startswith(tp.CORE_PREFIXES)
                     and n not in tp.PROHIBITED_IN_PROMPT
                     and n not in tp.DEFERRED_NAMES)
    assert not missing, (
        "IC 系统提示词点名了这些工具，但它们不在核心包里 —— 模型默认看不见：\n  "
        + "\n  ".join(missing)
        + "\n三个出口，选一个并写明理由：加进 CORE_NAMES（它在一条「多等一轮就有"
        + "实际代价」的路径上）；加进 DEFERRED_NAMES 并在提示词里写清去哪取；"
        + "或加进 PROHIBITED_IN_PROMPT（提示词提它只是为了说「这个会被拒绝」）。")


def test_deferred_tools_are_reachable_and_the_prompt_says_how():
    """**这是 DEFERRED_NAMES 的另一半，缺了它这份名单就是在削能力。**

    「叫模型调一个它看不见的东西」是按需加载唯一真正的失败模式。放进
    DEFERRED_NAMES 就等于承诺了两件事，这条把两件都钉住：

    1. 它真的在声明的那个包里（不是靠标签碰巧落进去 —— 标签改一次，提示词里
       那句 ``load_tool_pack("tip")`` 就成了假话）；
    2. 提示词真的写了怎么取（包名 + `load_tool_pack` 字样都要出现）。
    """
    from mast.agents.instrument_control.prompts import SYSTEM_PROMPT

    assert "load_tool_pack" in SYSTEM_PROMPT, (
        "提示词里连 load_tool_pack 都没提 —— 那 DEFERRED_NAMES 里每一个都是"
        "「叫模型调一个它看不见的东西」")

    problems: list[str] = []
    for name, pack in sorted(tp.DEFERRED_NAMES.items()):
        if not tp.pack_exists(pack):
            problems.append(f"{name} 声明在不存在的包 {pack!r} 里")
            continue
        if pack not in tp.classify(name, None):
            problems.append(f"{name} 声明在 {pack}，但 classify 把它归到 "
                            f"{sorted(tp.classify(name, None))}")
        if name in SYSTEM_PROMPT and f'"{pack}"' not in SYSTEM_PROMPT:
            problems.append(f"提示词点名了 {name}，却没有一处写 "
                            f'load_tool_pack("{pack}")')
    assert not problems, "\n  ".join(["DEFERRED_NAMES 与提示词对不上："] + problems)


def test_the_deferred_list_does_not_hide_an_urgent_path():
    """**不许把「多等一轮有实际代价」的工具 defer 掉。**

    判据是路径，不是体积：ConditionTip 贵（1 652）但它在视觉告警 CRITICAL 的
    即时处置路径上（StopScan → ConditionTip）—— 多一次检索就是多让一个坏针尖
    再扫一会儿。ScanAt 更贵（4 595）但它是最高频入口。
    """
    for urgent in ("StopScan", "ConditionTip", "ScanAt", "Withdraw",
                   "ApproachTip", "SetBias", "SetSetpoint", "RelocateCoarseXY"):
        assert urgent not in tp.DEFERRED_NAMES, (
            f"{urgent} 被 defer 了 —— 它在一条多等一轮就有实际代价的路径上")
        assert tp.CORE in tp.classify(urgent, None), f"{urgent} 不在核心里"


def test_the_prohibited_list_is_not_a_blanket_exemption():
    """变异自检：豁免名单必须**窄**。

    把它写成「凡是提示词提到的都豁免」等于把上一条闸门变成摆设 —— 而一个恒绿的
    闸门和没有闸门长得一模一样。这里钉住它只收硬拦那几个，而且每一个都确实出现
    在提示词里（否则就是个没人会发现的死条目）。
    """
    from mast.agents.instrument_control.prompts import SYSTEM_PROMPT

    assert len(tp.PROHIBITED_IN_PROMPT) <= 10, "豁免名单变胖了 —— 它在替谁开路？"
    for name in tp.PROHIBITED_IN_PROMPT:
        assert name in SYSTEM_PROMPT, (
            f"{name} 在豁免名单里，但提示词根本没提它 —— 死条目，删掉。")
    for must_be_core in ("ScanAt", "ConditionTip", "RelocateCoarseXY"):
        assert must_be_core not in tp.PROHIBITED_IN_PROMPT, (
            f"{must_be_core} 是提示词叫模型**去用**的，不能进豁免名单")


# ── 2. 分包 ─────────────────────────────────────────────────────────────

def test_core_tools_are_not_also_in_a_pack():
    assert tp.classify("ScanAt", _meta("ScanAt", ["scan"])) == frozenset({tp.CORE})


def test_prefix_tools_are_core():
    for name in ("handoff_to_supervisor", "ask_user", "search_tools",
                 "load_tool_pack", "conduct_status"):
        assert tp.CORE in tp.classify(name, None), name


def test_untagged_tools_land_in_util_not_nowhere():
    """没有标签的工具必须落在某个包里 —— 落不进任何包 = 永远取不出来。"""
    assert tp.classify("WeirdUnknownThing", _meta("x", [])) == frozenset({"util"})


def test_catalog_visible_grows_with_loaded_packs():
    cat = _catalog()
    core = cat.visible([])
    assert "ScanAt" in core and "AcquirePLLFreqSweep" not in core
    with_pll = cat.visible(["pll"])
    assert core < with_pll, "取包之后必须是超集"
    assert "AcquirePLLFreqSweep" in with_pll


def test_unknown_pack_name_is_ignored_not_crashing():
    cat = _catalog()
    assert cat.visible(["no-such-pack"]) == cat.visible([])


# ── 3. 检索 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expect", [
    ("ConfigureLockIn", "ConfigureLockIn"),          # 精确
    ("lockin", "ConfigureLockIn"),                   # 英文词
    ("锁相", "ConfigureLockIn"),                      # 中文 → 同义词
    ("调制幅度", "ConfigureLockIn"),                  # 中文多词
    ("马达", "SetMotorFreqAmp"),                      # 中文
    ("共振频率", "AcquirePLLFreqSweep"),              # 中文 → freq
])
def test_search_finds_it(query, expect):
    cat = _catalog()
    names = [h["name"] for h in tp.search(cat, query)]
    assert expect in names, f"{query!r} 找不到 {expect}；命中的是 {names}"


def test_search_is_not_vacuously_matching_everything():
    """变异自检：一个「什么都命中」的检索和一个都不命中一样没用。"""
    cat = _catalog()
    hits = tp.search(cat, "锁相")
    assert hits, "至少要命中一个"
    assert len(hits) < len(cat.packs_by_tool), "命中了全部工具 = 判据是摆设"
    assert "SetMotorFreqAmp" not in [h["name"] for h in hits]


def test_search_returns_nothing_for_nonsense():
    assert tp.search(_catalog(), "zzzzqqqq") == []
    assert tp.search(_catalog(), "") == []


def test_pack_fallback_catches_a_chinese_query_no_tool_matches():
    """撞不到具体工具 ≠ 没有这个能力 —— 至少要把对的那一包捞出来。"""
    cat = _catalog()
    assert "spectroscopy" in tp.search_packs(cat, "谱学")
    assert "optics" in tp.search_packs(cat, "激光") or not cat.tools_by_pack.get("optics")


def test_packs_of_skips_core():
    hits = [{"name": "ScanAt", "packs": ["core"]},
            {"name": "X", "packs": ["pll", "core"]}]
    assert tp.packs_of(hits) == ["pll"]


# ── 4. 目录块 ───────────────────────────────────────────────────────────

def test_index_names_the_two_meta_tools_and_says_it_is_not_a_gate():
    idx = tp.render_index(_catalog())
    assert "search_tools" in idx and "load_tool_pack" in idx
    assert "不是权限限制" in idx, "目录必须说清它不是门禁，否则模型会以为自己被限权"


def test_index_does_not_list_individual_tool_names():
    """列了就等于把 schema 换成一份更差的清单 —— 省不下多少，还诱导模型猜参数。"""
    idx = tp.render_index(_catalog())
    assert "AcquirePLLFreqSweep" not in idx


def test_index_is_empty_when_there_are_no_packs():
    cat = tp.build_catalog("x", [SimpleNamespace(name="ScanAt")], None)
    assert tp.render_index(cat) == ""


# ── 5. 收窄中间件 ───────────────────────────────────────────────────────

def _req(tools, loaded=()):
    from langchain_core.messages import HumanMessage
    return SimpleNamespace(tools=list(tools), state={STATE_KEY: list(loaded)},
                           system_message=None,
                           messages=[HumanMessage(content="hi")])


def _mw(cat, **kw):
    mw = ToolVisibilityMiddleware(cat, **kw)
    mw._enabled = kw.get("enabled", True)      # 绕开 MIN_TOOLS（替身目录很小）
    return mw


def test_middleware_narrows_and_appends_the_index():
    cat = _catalog()
    tools = [SimpleNamespace(name=n) for n in cat.packs_by_tool]
    out = _mw(cat)._apply(_req(tools))
    names = {t.name for t in out.tools}
    assert "ScanAt" in names and "AcquirePLLFreqSweep" not in names
    assert tp.INDEX_HEADER in out.system_message.content


def test_middleware_is_a_subset_operation_only():
    """**绝不加工具。** 加了 factory 会 ValueError（中间件不能引入未注册工具）。"""
    cat = _catalog()
    tools = [SimpleNamespace(name=n) for n in cat.packs_by_tool]
    out = _mw(cat)._apply(_req(tools, ["pll", "spectroscopy", "motion"]))
    assert {t.name for t in out.tools} <= {t.name for t in tools}


def test_disabled_middleware_changes_nothing():
    cat = _catalog()
    tools = [SimpleNamespace(name=n) for n in cat.packs_by_tool]
    req = _req(tools)
    out = ToolVisibilityMiddleware(cat, enabled=False)._apply(req)
    assert len(out.tools) == len(tools)
    assert out.system_message is None


def test_missing_state_key_degrades_to_core_not_to_crash():
    cat = _catalog()
    tools = [SimpleNamespace(name=n) for n in cat.packs_by_tool]
    req = SimpleNamespace(tools=tools, state={}, system_message=None, messages=[])
    out = _mw(cat)._apply(req)
    assert {t.name for t in out.tools} == set(cat.visible([]))


def test_a_broken_catalog_gives_everything_rather_than_nothing():
    """收窄失败必须 fail-open。少给工具 = 让 agent 干不了活，比多给贵得多。"""
    broken = tp.Catalog(agent="x")           # 空目录：visible() 返回空集
    tools = [SimpleNamespace(name="ScanAt")]
    out = _mw(broken)._apply(_req(tools))
    assert len(out.tools) == 1


# ── 6. 真目录（不是替身）——省下来的量对得上吗 ────────────────────────

def test_real_ic_catalog_saves_most_of_the_tool_surface():
    """这是本次改动的**验收数字**，不是装饰。

    2026-08-24 实测：396 个工具 354 740 字符 → 核心 40 个 52 724 字符。
    阈值写 60% 而不是 85%，是因为加技能会改变分子分母；跌破 60% 说明核心包被
    塞胖了，那时候该看的是 CORE_NAMES 而不是这个数字。
    """
    from mast.agents.instrument_control.tools import (
        build_tools,
        discover_instrument_skills,
    )

    registry = discover_instrument_skills()
    tools = build_tools(None, lambda: SimpleNamespace(), registry=registry, targets=())
    cat = tp.build_catalog("instrument_control", tools, registry)
    assert cat.total_chars > 100_000, "工具面小得不像 IC —— 是不是没发现技能？"
    saved = 1 - cat.core_chars / cat.total_chars
    assert saved > 0.60, (
        f"只省了 {saved:.1%}（核心 {len(cat.core)} 个 / {cat.core_chars:,} 字符，"
        f"共 {cat.total_chars:,}）。核心包多半被塞胖了。")


def test_every_pack_in_the_index_can_actually_be_loaded():
    """目录里写着能取的包，必须真的取得出东西来。

    「列了却取不出」和「没列」不一样：前者会让模型取一次、拿到空、然后放弃。
    """
    from mast.agents.instrument_control.tools import (
        build_tools,
        discover_instrument_skills,
    )

    registry = discover_instrument_skills()
    tools = build_tools(None, lambda: SimpleNamespace(), registry=registry, targets=())
    cat = tp.build_catalog("instrument_control", tools, registry)
    base = cat.visible([])
    for pack in cat.known_packs():
        assert cat.visible([pack]) > base, f"取了 {pack} 之后可见集没变大"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
