"""流程默认值须与每一种可用针尖包络对账；显式输入仍按包络拒绝，不能静默夹紧。"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.core.noble_tip_workflow import (  # noqa: E402
    NOBLE_METAL_BASELINE,
    _ENVELOPE_PAIRS,
    reconcile_with_tip_envelope,
    within_envelope,
)
from mast.core.tip_conditioning_policy import ANY, resolve_policy  # noqa: E402
import mast.core.tip_conditioning_policy as policy_mod  # noqa: E402


def _all_tier_facts() -> list[tuple[str, dict]]:
    """每一档的代表性针尖事实。直接从表的键派生 —— 不另抄一份档位清单。

    表名从真源取:第一版写的是自己编的 `_TIP_POLICY`,当场 AttributeError。
    编出来的名字只有「假警报」和「碰巧对」两种结局 —— 这次是前者,而它只是恰好
    会抛异常;换成 `getattr(mod, name, {})` 就会静默变成一张空表,闸门全绿。

    **`facts=None`(未登记针尖)必须在列**(2026-08-10 补):它不是 `_POLICY` 的
    一个键,所以按键派生会漏掉它 —— 而那正是本系统**最常见**的状态,
    且通用档 `_DEFAULT` 有自己的包络(`max_abs_pulse_v=6.0`),
    出厂 `pulse_v=10` 在它上面同样违法。
    「按真源派生」防的是抄漏一档;防不了**真源里没有的那一档**。
    """
    out: list[tuple[str, dict]] = [("(未登记针尖 → 通用档 _DEFAULT)", None)]
    for key in policy_mod._POLICY:                           # noqa: SLF001
        mat, fab, form = key
        facts = {
            "name": f"tier-{mat}-{fab}-{form}",
            "material": "Zz" if mat is ANY else mat,
            "fabrication": "zz" if fab is ANY else fab,
            "form": form,
        }
        out.append((f"({mat}, {fab}, {form})", facts))
    return out


def _violations(wf, policy: dict) -> list[str]:
    """这套参数在这一档上超了哪几条、各超多少倍。

    「在不在包络内」用被测模块**自己**那个判据(`within_envelope`),不在这里另写
    一个 `>` —— 这道闸门第一次跑就是被这个咬的:`3.0*1e-9` 比 `3e-9` 大一个最低位,
    于是「换成正好等于包络的值」之后再比一次仍然判超限。两份「差不多相等」的实现
    迟早各自漂移,而闸门与被测代码漂移开的时候,红的那一方通常是对的那一方。
    """
    bad: list[str] = []
    for field_name, limit_key, _rec, scale in _ENVELOPE_PAIRS:
        limit = policy.get(limit_key)
        value = getattr(wf, field_name, None)
        if limit is None or value is None:
            continue
        limit_f, value_f = float(limit), abs(float(value)) * scale
        if not within_envelope(value_f, limit_f):
            bad.append(
                f"{field_name}={value:g} → {value_f:.3e} 超出 {limit_key}="
                f"{limit_f:.3e}（{value_f / limit_f:.2g}×）")
    return bad


# ══════════════════════════════════════════════════════════════════════
# 自检 —— 一条匹配不到任何东西的闸门会一直绿
# ══════════════════════════════════════════════════════════════════════

def test_the_gate_actually_looks_at_something():
    tiers = _all_tier_facts()
    assert len(tiers) >= 10, f"只枚举到 {len(tiers)} 档,查表方式变了"
    assert any(f and f["form"] == "qplus" for _n, f in tiers), "qPlus 档一个都没枚举到"
    assert any(f is None for _n, f in tiers), (
        "未登记针尖(通用档)不在枚举里 —— 而它是本系统最常见的状态")
    assert len(_ENVELOPE_PAIRS) >= 3, _ENVELOPE_PAIRS
    # 每一对都必须真的指向存在的字段/键,否则整个闸门在比较两个 None。
    for field_name, limit_key, rec_key, _scale in _ENVELOPE_PAIRS:
        assert hasattr(NOBLE_METAL_BASELINE, field_name), field_name
        assert any(limit_key in p for _n, p in
                   [(n, resolve_policy(f)) for n, f in tiers]), limit_key
        if rec_key:
            assert any(rec_key in p for _n, p in
                       [(n, resolve_policy(f)) for n, f in tiers]), rec_key


#: **合成的紧包络** —— 机制测试的载体,不是任何一档出厂值。
#:
#: 为什么必须是合成的:这些测试要考的是**对账机制**(会不会换、换了说不说话、
#: 用户指定的值动不动),而不是「出厂值恰好紧」。把机制绑在出厂值上,机制就会
#: 随着一次策略调整整体失效 —— 而这件事**已经发生过两次**:
#:   2026-08-10 用户把 qPlus 的 `max_abs_pulse_v` 从 3 V 定到 10 V,
#:              于是 qPlus 上再也找不到「合法但超包络」的值,测试改挂到 PtIr 丝上;
#:   2026-08-12 用户把**所有档**的包络统一拉满(10 V / 10 nm),
#:              于是 PtIr 那个载体也没了 —— 全仓一个还咬人的档都不剩。
#: 第一次是搬家,第二次就得解耦。**下一次改策略,这些测试不该再动一个字。**
_TIGHT_POLICY: dict = {"max_abs_pulse_v": 3.0, "max_poke_depth_m": 5.0e-10}


def test_the_reconciliation_machinery_can_still_detect_a_violation():
    """探针有效性:对账机制**认得出**一个越界的默认值。

    这条不再问「出厂值是不是有一档会超」——2026-08-12 起答案永远是「不会」
    (包络全部拉满)。它问的是机制本身还活着没有:喂一个紧包络,它必须报违规。

    机制活着这件事仍然重要:出厂值将来还会改,而**下一次某个默认值越界时,
    唯一会告诉我们的就是它**。
    """
    bad = _violations(NOBLE_METAL_BASELINE, _TIGHT_POLICY)
    assert bad, ("对账机制对一个明显越界的紧包络也报不出违规 —— "
                 "它已经失效了,而主闸门会因此永远是绿的。")


def test_shipped_envelopes_no_longer_clamp_anything(monkeypatch):
    """出厂值现在落在**每一档**包络内 —— 拉满之后的应然状态,钉住它。

    这条是上一条的对照:机制活着(上),而且现在没有东西要被它换(下)。
    两条一起才说得清「没有替换」是因为没必要,不是因为机制坏了。

    ⚠️ 附带后果,写下来:既然一个值都不换,`reconcile_with_tip_envelope` 的
    notes 从此恒空 —— 那句「已改用这根针方案表里的值」的提示不会再出现。
    它原本是用户了解「系统替我改了什么」的窗口。
    """
    for name, facts in _all_tier_facts():
        bad = _violations(NOBLE_METAL_BASELINE, resolve_policy(facts))
        assert not bad, f"{name} 档上仍有默认值越界: {bad}"


# ══════════════════════════════════════════════════════════════════════
# 主闸门
# ══════════════════════════════════════════════════════════════════════

def test_reconciled_defaults_fit_every_tier_envelope():
    """对账之后,默认值在**每一档**上都可执行。

    失败时逐条列出「哪个默认在哪一档上超了多少倍」。
    """
    failures: list[str] = []
    for name, facts in _all_tier_facts():
        policy = resolve_policy(facts)
        wf, _notes = reconcile_with_tip_envelope(
            NOBLE_METAL_BASELINE, policy=policy)
        bad = _violations(wf, policy)
        if bad:
            failures.append(f"  {name}: " + "; ".join(bad))
    assert not failures, (
        "对账之后仍然有默认值超出针尖包络 —— 在这些针上,修针流程出厂即不可用:\n"
        + "\n".join(failures)
        + "\n（修默认值，或给 _ENVELOPE_PAIRS 补一条映射。"
          "**不要**去放宽包络：那是用户的授权范围。）")


def test_reconciliation_never_touches_an_operator_specified_value():
    """「拒绝不夹紧」保住的正是这一条。

    用户逐字说过的数,即使违法也**原样送去过包络门**,让他看到一句精确的拒绝 ——
    夹了他会以为自己扎了 3 nm。对账只动没人指定过的默认值。
    """
    # 载体用**合成紧包络**,不再去出厂表里找「还咬人的那一档」——
    # 那种找法已经搬过一次家(qPlus→PtIr),而 2026-08-12 拉满之后一档都不剩。
    # 见 `_TIGHT_POLICY` 的说明。
    policy = _TIGHT_POLICY
    from dataclasses import replace

    asked = replace(NOBLE_METAL_BASELINE, pulse_v=9.0)   # > 紧包络的 3 V
    wf, notes = reconcile_with_tip_envelope(
        asked, specified={"pulse_v"}, policy=policy)
    assert wf.pulse_v == 9.0, "用户指定的 9 V 被悄悄换掉了"
    assert not any("pulse_v" in n for n in notes)
    # 同一个值,没被指定时才换。
    wf2, notes2 = reconcile_with_tip_envelope(asked, policy=policy)
    assert wf2.pulse_v != 9.0 and any("pulse_v" in n for n in notes2), notes2


def test_a_substitution_is_never_silent():
    """静默替换和静默夹紧一样坏 —— 每换一个值都要有一句话。

    载体是**合成紧包络**(见 `_TIGHT_POLICY`):出厂包络 2026-08-12 起全部拉满,
    真跑时不会再有替换发生,但这条规矩必须继续被钉着 —— 将来任何一次收紧包络,
    替换会立刻回来,而那时它仍然不许静默。
    """
    wf, notes = reconcile_with_tip_envelope(
        NOBLE_METAL_BASELINE, policy=_TIGHT_POLICY)
    changed = [f for f, *_ in _ENVELOPE_PAIRS
               if getattr(wf, f) != getattr(NOBLE_METAL_BASELINE, f)]
    assert changed, "这一档上什么都没换?探针失效了"
    assert len(notes) == len(changed), (changed, notes)
    for note in notes:
        assert "超出" in note and "已改用" in note, note


def test_a_broken_policy_chain_means_no_reconciliation(monkeypatch):
    """整条链不可用时不动手 —— 那时确实没有包络可对。

    注意 `policy=None` 在这个 API 里是「**去查**」而不是「没有」——
    第一版这条测试拿 `policy=None` 当「没有」，于是它测的是查表路径。
    真正的「链坏了」要让 `resolve_policy` 抛。
    """
    import mast.core.tip_conditioning_policy as pol

    def _boom(*a, **kw):
        raise RuntimeError("policy table unavailable")

    monkeypatch.setattr(pol, "resolve_policy", _boom, raising=False)
    wf, notes = reconcile_with_tip_envelope(NOBLE_METAL_BASELINE)
    assert wf is NOBLE_METAL_BASELINE and notes == []
    # 空表(读到了但什么都没有)同样不动手。
    assert reconcile_with_tip_envelope(
        NOBLE_METAL_BASELINE, policy={})[0] is NOBLE_METAL_BASELINE


def test_an_unregistered_tip_still_gets_reconciled_on_the_runtime_path(monkeypatch):
    """**未登记针尖不是「没有包络」** —— 而这条走的是运行时那条路,不传 policy。

    上面那些测试都显式传 `policy=`,所以它们**证明不了运行时会不会对账**。
    第一版的 `_tip_policy_now()` 在读不到登记时直接返回 None、不对账,于是:
      * 闸门(传 policy)绿;
      * 运行时(不传)在最常见的状态下一发脉冲都打不出去(10 V > 通用档 6 V)。
    **闸门测的不是真正会跑的那条代码** —— 那正是本仓「证据回答的不是被问的那个
    问题」那条纪律要抓的形状。
    """
    import mast.core.tip_conditioning_policy as pol
    import mast.core.tip_state as tip_state

    monkeypatch.setattr(tip_state, "current_tip_facts", lambda: None,
                        raising=False)
    # 通用档的包络 2026-08-12 起也拉满了 ⇒ 用真值跑,这条路「没有东西要换」,
    # 于是它证明不了自己有没有被走到。塞一个紧包络进**运行时会查的那个函数**,
    # 走到了就一定会有替换。这是唯一能把「路通了」和「恰好没东西要换」分开的做法。
    monkeypatch.setattr(pol, "resolve_policy",
                        lambda *a, **kw: dict(_TIGHT_POLICY), raising=False)
    wf, notes = reconcile_with_tip_envelope(NOBLE_METAL_BASELINE)   # ← 不传 policy
    assert notes, "运行时那条路没去查包络 —— 闸门测的不是真正会跑的那条代码"
    assert abs(wf.pulse_v) <= float(_TIGHT_POLICY["max_abs_pulse_v"]), (
        f"对账后 pulse_v={wf.pulse_v} 仍然超出包络 "
        f"{_TIGHT_POLICY['max_abs_pulse_v']}")
    assert not _violations(wf, _TIGHT_POLICY), _violations(wf, _TIGHT_POLICY)


def test_the_metal_wire_default_is_not_watered_down():
    """合法的默认一个都不许碰。

    10 V 是用户对金属丝针尖的做法(流程表注释里写着)。对账只在**违法**时出手;
    把它顺手降到方案表的 5 V 会悄悄改掉一条用户的做法。"""
    wire = {"name": "w", "material": "W", "fabrication": "etched",
            "form": "stm_wire"}
    wf, notes = reconcile_with_tip_envelope(
        NOBLE_METAL_BASELINE, policy=resolve_policy(wire))
    assert wf.pulse_v == NOBLE_METAL_BASELINE.pulse_v == 10.0
    assert not any("pulse_v" in n for n in notes), notes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
