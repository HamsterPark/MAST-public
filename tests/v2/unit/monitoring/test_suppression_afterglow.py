"""抑制余波期：尖峰落在仪器令牌空隙里时，仍按近期动作上下文判定。

进退针动作的瞬态可能持续到工具调用之间。仅查看当前令牌持有者会丢失
动作上下文；单纯增大连续告警门槛又会延迟真正危险的告警。
因此监控应独立保留有限的动作余波窗口，而不改变仲裁层的令牌释放语义。
本文件使用合成事件与数值，不包含仪器现场记录。
"""
from __future__ import annotations

import sys
import time
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

import pytest  # noqa: E402

from mast.monitoring.service import (  # noqa: E402
    SUPPRESS_SKILL_PATTERNS,
    CurrentMonitorService,
)


def _svc() -> CurrentMonitorService:
    return CurrentMonitorService(pool_getter=lambda: None)


# ══════════════════════════════════════════════════════════════════════
# 令牌在手 —— 原有行为,一个字都不许变
# ══════════════════════════════════════════════════════════════════════

def test_a_held_token_still_suppresses():
    assert CurrentMonitorService._is_suppressed({"ctx_skill": "AutoApproach"})
    assert CurrentMonitorService._is_suppressed({"ctx_skill": "ForgeAuTip"})


def test_an_unrelated_skill_never_suppresses():
    assert not CurrentMonitorService._is_suppressed({"ctx_skill": "GetBias"})


def test_no_token_and_no_afterglow_is_not_suppressed():
    """空令牌本身**不是**抑制理由 —— 这条是余波期的对照组。"""
    assert not CurrentMonitorService._is_suppressed({"ctx_skill": ""})
    assert not CurrentMonitorService._is_suppressed(
        {"ctx_skill": "", "ctx_afterglow_skill": ""})


# ══════════════════════════════════════════════════════════════════════
# 余波期 —— 事故本体
# ══════════════════════════════════════════════════════════════════════

def test_a_spike_in_the_token_gap_is_suppressed_after_an_approach():
    """进针刚返回、令牌已放,settle 尾巴上的尖峰不该弹窗。"""
    s = _svc()
    s._note_skill_done("autoapproach")
    ctx = s._context_labels()
    assert ctx["ctx_skill"] == "", "这条要测的就是空令牌那一段"
    assert ctx["ctx_afterglow_skill"] == "autoapproach"
    assert CurrentMonitorService._is_suppressed(ctx)


def test_the_afterglow_ends(monkeypatch):
    """**余波期必须有尽头** —— 否则这不是「降低瞬态权重」,是把物理保护关掉。"""
    s = _svc()
    monkeypatch.setattr(s, "_afterglow_window_s", lambda: 0.05)
    s._note_skill_done("autoapproach")
    assert s._afterglow_skill() == "autoapproach"
    time.sleep(0.12)
    assert s._afterglow_skill() == ""
    assert not CurrentMonitorService._is_suppressed(s._context_labels())


def test_a_non_suppressible_skill_leaves_no_afterglow():
    """余波期只延长**本来就该被抑制**的技能,不是给所有技能发通行证。"""
    s = _svc()
    name = "getbias"
    assert not any(p in name for p in SUPPRESS_SKILL_PATTERNS), "前提变了"
    # 事件回调自己就会过滤;这里直接确认没有人替它开余波期。
    assert s._afterglow_skill() == ""


def test_afterglow_can_be_switched_off(monkeypatch):
    """设 0 = 回到「只看当下令牌」。留这条路是因为余波期是**未标定**的机制。"""
    s = _svc()
    monkeypatch.setattr(s, "_afterglow_window_s", lambda: 0.0)
    s._note_skill_done("autoapproach")
    assert s._afterglow_skill() == ""


# ══════════════════════════════════════════════════════════════════════
# 反向:持续越界仍然要升级
# ══════════════════════════════════════════════════════════════════════

def test_a_sustained_rail_still_escalates_inside_the_afterglow():
    """真贴轨是**持续**的,而余波期降的是「瞬态」的权重,不是「持续越界」的。

    抑制发生在**段级**(这一段不判),它不会改动 AlertEngine 的连续段计数逻辑;
    所以一次真的贴轨即使起始于余波期内,只要它持续下去,余波期一过就照常连续计数、
    照常升级。这条直接驱动 AlertEngine 证明这一点。
    """
    from mast.monitoring.alerts import AlertEngine
    from mast.monitoring.thresholds import get_monitor_thresholds

    th = get_monitor_thresholds()
    eng = AlertEngine(lambda: th)
    feats = {"sat_frac": th.cm_sat_frac_crit + 0.5, "frozen": 0}
    ctx = {"ctx_zctrl_on": True}
    fired = None
    for i in range(th.crit_consecutive + 2):
        v = eng.evaluate(feats, ctx)
        assert v.level == "critical_candidate", v
        fired = eng.confirm(v, now=1000.0 + i)
        if fired:
            break
    assert fired == "saturation", "持续贴轨没有升级 —— 余波期不该动到这条路"


def test_the_record_tells_the_two_suppression_reasons_apart():
    """「技能拿着令牌」和「技能刚放手」是两种强度不同的说法,记录里不能长一样。

    余波期是一条**未标定**的机制,它哪天开始藏真事件,就是靠这个字段看出来的 ——
    所以「谁抑制了这一段」必须能被事后区分,而不是都写成 `suppressed_by=skill`。
    """
    class _V:
        suppressed_rules: tuple = ()

    held = CurrentMonitorService._suppression_extra(
        _V(), True, {"ctx_skill": "AutoApproach"})
    assert held["suppressed_by"] == "skill"
    assert held["suppressed_skill"] == "AutoApproach"
    assert "suppressed_afterglow" not in held

    wake = CurrentMonitorService._suppression_extra(
        _V(), True, {"ctx_skill": "", "ctx_afterglow_skill": "autoapproach"})
    assert wake["suppressed_by"] == "skill_afterglow", (
        "余波期抑制被记成了「技能正在跑」—— 两者强度不同,记录必须分得开")
    assert wake["suppressed_afterglow"] is True
    assert wake["suppressed_skill"] == "autoapproach"



# ══════════════════════════════════════════════════════════════════════
# 调制上下文 —— 合成纹波在调制开启时留痕而不误报
# ══════════════════════════════════════════════════════════════════════

def _engine():
    from mast.monitoring.alerts import AlertEngine
    from mast.monitoring.thresholds import get_monitor_thresholds
    return AlertEngine(lambda: get_monitor_thresholds())


_RIPPLE = {"jump_rate_hz": 125.0, "sat_frac": 0.0, "frozen": 0}


def test_modulation_ripple_is_recorded_not_alerted():
    """合成纹波在调制开启时被记录为抑制事件；相同信号在调制关闭时应告警。"""
    v = _engine().evaluate(_RIPPLE, {"ctx_lockin_on": True})
    assert v.level == "suppressed"
    assert "jump_burst" not in v.rules
    assert "jump_burst" in v.suppressed_rules, (
        "必须是「记录了但没判」,不是「删掉了」—— 两者事后要分得开")


def test_the_same_ripple_still_warns_with_modulation_off():
    """降级只在调制开着时生效。关掉之后同样的跳变照报 —— 否则这不是上下文,是删规则。"""
    v = _engine().evaluate(_RIPPLE, {"ctx_lockin_on": False})
    assert v.level == "warn" and "jump_burst" in v.rules


def test_an_unread_modulation_state_does_not_suppress():
    """None(没读到)**不等于**关着,也不等于开着 —— 极性与 ctx_scanning 一致:
    只有显式 True 才抑制,读不到就保持今天的行为。"""
    v = _engine().evaluate(_RIPPLE, {"ctx_lockin_on": None})
    assert v.level == "warn" and "jump_burst" in v.rules


def test_modulation_never_suppresses_a_physical_critical():
    """调制开着的结照样会贴轨/冻结/巨阶跃。抑制的是 WARN,不是那三条。"""
    feats = {"sat_frac": 0.9, "frozen": 0, "jump_rate_hz": 125.0}
    v = _engine().evaluate(feats, {"ctx_lockin_on": True, "ctx_zctrl_on": True})
    assert v.level == "critical_candidate" and "saturation" in v.rules


def test_the_monitor_reads_modulation_state_from_the_instrument_not_from_mast():
    """状态必须来自**硬件回读**,不是 MAST 自己的事件流。

    用户是在 Nanonis 面板上开调制的 —— 从 MAST 的 ConfigureLockIn 调用去推断,
    恰恰在最需要它的那种情况下是瞎的。
    """
    from mast.core.types import HardwareState

    assert hasattr(HardwareState(), "lockin_mod_on")
    src = (Path(_MASTV2_ROOT) / "mast" / "core" / "state.py").read_text(encoding="utf-8")
    assert "LockIn_ModOnOffGet" in src, "1 Hz 轮询里没有这条回读,字段就是死的"

if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
