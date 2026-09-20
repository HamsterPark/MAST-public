"""结构闸门:修针的**每一个结局都念得出中文**。

## 为什么需要这一条

2026-08-18 给修针外环补旁白时,结局代码(`ready` / `surface_spent` /
`round_hard_cap` …)需要一份**一行**的中文 —— 而 `forge_au_tip._OUTCOME_CN`
里那一份是给**报告**用的:每条两三句、带处置建议(「不建议直接加大预算重跑」)。
一张表两种排版,于是有了 `narration_templates.FORGE_OUTCOME_ZH`。

两张表本身不是问题(同一个代码的两种呈现,不是两份事实)。问题是**一边加了
代码另一边忘了** —— 那时旁白会把一个新结局显示成一个英文 slug,而报告里好好的。
没人会注意到:那条旁白照常出现、照常有内容,只是内容是 `pulse_rescue_exhausted`。

人肉对两张表是对不齐的,所以钉成闸门。

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/chat/test_forge_outcomes_are_all_narratable.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.chat import narration_templates as T  # noqa: E402
from mast.skills.composite import forge_au_tip as F  # noqa: E402


def test_every_report_outcome_has_a_one_line_narration():
    """报告里有的结局,旁白里必须也有。

    少一个的症状不是报错,是旁白里冒出一个英文 slug —— 而那条旁白看起来
    完全正常(有内容、有位置、有颜色)。
    """
    missing = sorted(set(F._OUTCOME_CN) - set(T.FORGE_OUTCOME_ZH))
    assert not missing, (
        f"这些结局报告里有、旁白里没有:{missing}\n"
        f"→ 往 narration_templates.FORGE_OUTCOME_ZH 里各加**一行**中文。")


def test_no_narration_outcome_is_invented():
    """反过来也要成立:旁白里不许有报告不认识的结局。

    多出来的那一条**永远不会被用到** —— 而「已经支持了」这句话在整棵树上
    没有任何证据支持(本仓 `producer_wired_consumer_absent` 的同一形状)。
    """
    extra = sorted(set(T.FORGE_OUTCOME_ZH) - set(F._OUTCOME_CN))
    assert not extra, (
        f"这些结局只有旁白认得,报告里没有:{extra}\n"
        f"→ 要么它是死代码,要么 forge_au_tip._OUTCOME_CN 漏了。")


def test_every_outcome_the_code_can_actually_set_is_in_both_tables():
    """**判据从源码派生,不从名单派生。**

    上面两条只保证两张表一致 —— 它们可以**一起**漏掉一个代码里真的会赋的值。
    这一条去源码里数 `outcome = "..."` 和 `site["outcome"] = "..."`,
    那才是「这个流程实际会产出什么」的真源。
    """
    import re

    src = Path(F.__file__).read_text(encoding="utf-8")
    # 只认字面量赋值 —— 变量赋值(`outcome = str(moved)`)本来就数不出来,
    # 而它的取值来自 `_relocate`,那几个也都在表里(下面一条断言覆盖)。
    found = set(re.findall(r'(?:site\["outcome"\]|outcome)\s*=\s*"([a-z_]+)"', src))
    both = set(F._OUTCOME_CN) & set(T.FORGE_OUTCOME_ZH)
    missing = sorted(found - both)
    assert not missing, (
        f"源码里会赋这些结局,而两张表里不全有:{missing}")
    # 闸门自校验:一条扫不到东西的闸门和没有闸门是同一件事。
    assert len(found) >= len(both) // 2, (
        f"只从源码里数到 {len(found)} 个结局 —— 正则多半失效了,这条闸门在空跑")


@pytest.mark.parametrize("code", sorted(T.FORGE_OUTCOME_ZH))
def test_each_outcome_renders_a_real_sentence(code):
    """每个代码都要渲染成一句**带中文**的话,不是把代码原样印出来。"""
    r = T.render("site_result", {"site_no": 1, "outcome": code, "rounds": 2})
    assert r is not None
    assert r.text, code
    assert any("一" <= ch <= "鿿" for ch in r.text), (
        f"{code} 渲染出来没有一个汉字:{r.text!r}")
    assert code not in r.text, (
        f"{code} 被原样印进了句子 —— 说明词表里没有它:{r.text!r}")


def test_an_unknown_outcome_shows_the_code_rather_than_a_soothing_sentence():
    """查不到就**原样用代码**。

    绝不编一句「流程已结束」之类的通用话:一个没被翻译的代码看起来刺眼,
    而一句通用话看起来完全正常 —— 后者会让一个漏掉的结局永远不被发现。
    """
    r = T.render("site_result", {"site_no": 1, "outcome": "brand_new_outcome"})
    assert r is not None
    assert "brand_new_outcome" in r.text


# ── 形参名撞车:一条纯粹因为「参数叫什么」而炸掉的调用 ──────────────────


def test_a_template_field_named_kind_does_not_collide_with_the_emitter():
    """旁白的**数据字段**可以叫 ``kind`` —— 而发射器的形参也叫 ``kind``。

    2026-08-18:``poke_decision`` 的决策码字段就叫 ``kind``,于是
    ``_say("poke_decision", kind="deeper")`` 当场抛
    ``TypeError: _say() got multiple values for argument 'kind'``,
    **63 条 composite 测试一起红**。

    修法是把三个发射器的第一个形参改成**位置限定**(``def f(kind, /, **data)``)。
    这条钉住那个 ``/``:去掉它,下面这个调用立刻抛。

    为什么值得钉:这类坏法与逻辑无关,只与**名字**有关。它躲得过 code review
    (那一行看起来完全正常),也躲得过类型检查,只在真的发这条旁白时才炸 ——
    而那条旁白可能几个月才走到一次。
    """
    import inspect

    from mast.chat import narration
    from mast.skills.composite import _tip_phases as P
    from mast.skills.composite import graph_executor as G

    # **四个**发射器全覆盖。漏掉一个的后果不是「少一道保险」,是下一个叫 kind
    # 的字段会从那一个炸出来 —— 而这一条测试是绿的。
    for fn in (narration.narrate, narration.Sink.narrate, P._say, G._narrate):
        params = list(inspect.signature(fn).parameters.values())
        # 跳过 self
        first = next(p for p in params if p.name not in ("self",))
        assert first.kind is inspect.Parameter.POSITIONAL_ONLY, (
            f"{fn.__qualname__} 的第一个形参 {first.name!r} 不是位置限定的 —— "
            f"任何数据字段叫这个名字的旁白都会抛 TypeError")

    # 端到端:真的发一条 ``kind`` 字段的旁白,不许抛。
    # (``_say`` 自己吞异常,所以这里直接调 ``render`` 之外的那一层。)
    P._say("poke_decision", kind="deeper", depth_pm=500.0, next_depth_pm=600.0)


def test_every_poke_decision_code_the_loop_emits_is_renderable():
    """扎针环里 ``_decide(entry, "...")`` 用的每一个决策码,模板都得认得。

    判据从**源码**派生:去 ``_tip_phases`` 里数 ``_decide(entry, "xxx"``,
    而不是维护一张名单。多一个决策码而模板不认,症状是旁白说
    「这个决策旁白还不认得」—— 那句话是特意写成刺眼的,但没人盯着屏幕时它照样溜过去。
    """
    import re

    from mast.chat import narration_templates as T2
    from mast.skills.composite import _tip_phases as P

    src = Path(P.__file__).read_text(encoding="utf-8")
    codes = set(re.findall(r'_decide\(\s*entry\s*,\s*"([a-z_]+)"', src))
    assert codes, "一个 _decide 调用都没数到 —— 这条闸门在空跑"
    for code in sorted(codes):
        r = T2.render("poke_decision", {"kind": code, "depth_pm": 500.0,
                                        "next_depth_pm": 300.0, "tries": 3,
                                        "streak": 2, "need": 5})
        assert r is not None and not r.degraded, code
        assert "还不认得" not in r.text, (
            f"扎针环会发 {code!r},而模板不认识它:{r.text!r}")
