"""前置条件须精确区分 z_controller_on 与 z_controller_off；子步骤执行后的状态应回写缓存，供下一步使用。"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.core.preconditions import (  # noqa: E402
    PRECONDITION_CHECKS,
    _SUBSTRING_RULES,
    check_state_preconditions,
)
from mast.core.state import InstrumentState, patch_state_from_result  # noqa: E402
from mast.core.types import HardwareState  # noqa: E402


def _st(**kw) -> HardwareState:
    return HardwareState(**kw)


# ══════════════════════════════════════════════════════════════════════
# 一、极性
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("hardware_on, precondition, should_refuse", [
    # 「必须关着」——关着放行,开着拒绝。
    (False, "z_controller_off", False),
    (True, "z_controller_off", True),
    # 「必须开着」——开着放行,关着拒绝。
    (True, "z_controller_on", False),
    (False, "z_controller_on", True),
])
def test_the_gate_is_not_inverted(hardware_on, precondition, should_refuse):
    st = _st(z_controller_on=hardware_on,
             z_controller_status="On" if hardware_on else "Off")
    v = check_state_preconditions([precondition], st)
    assert bool(v) is should_refuse, (
        f"硬件 {'ON' if hardware_on else 'OFF'} + 前置 {precondition!r} → "
        f"{'拒绝' if v else '放行'}(应当{'拒绝' if should_refuse else '放行'});{v}")


def test_the_dangerous_side_is_the_allow_side():
    """反过来判时,**放行**那一侧才是撞针那一侧。

    `MotorMove` 声明 `z_controller_off`;反馈环闭着跑开环粗进针马达就是撞针
    (`zcontrol.py` 的 TryEngageController 逐字写过这句)。所以这条单独钉:
    控制器**开着**时,要求它关着的前置必须拒绝。
    """
    from mast.skills.builtins.motor import MotorMove

    assert "z_controller_off" in MotorMove().metadata().preconditions
    st = _st(z_controller_on=True, z_controller_status="On")
    assert check_state_preconditions(["z_controller_off"], st), (
        "反馈环闭着,而要求它断开的前置放行了 —— 这是撞针那一侧")


def test_the_message_says_what_was_read_and_what_was_wanted():
    """真机上那句「'z_controller_off' — Z controller is OFF」把**满足的条件**
    印成了失败理由,而读的人只能读成「判据反了」。

    印出「读到什么 / 要什么」之后,同一句话自己就把两种可能分开了。"""
    st = _st(z_controller_on=True, z_controller_status="On")
    msg = check_state_preconditions(["z_controller_off"], st)[0]
    assert "读到" in msg and "要求" in msg, msg
    assert "z_controller_on=True" in msg, msg
    assert "z_controller_status=On" in msg, msg   # 六态里的哪一态


def test_no_substring_rule_can_shadow_an_exact_name():
    """结构闸门:精确表里的每个名字都不许被子串规则**改判**。

    这才是根因的一般形式 —— `z_controller_off` 只是第一个被抓到的。任何新增的
    否定形式前置(`*_off` / `not_*` / `*_stopped`)都可能包含它自己的肯定词。
    """
    probes = [_st(z_controller_on=True, scan_running=True, bias_v=1.0,
                  withdrawn=True),
              _st(z_controller_on=False, scan_running=False, bias_v=0.0,
                  withdrawn=False)]
    mismatches = []
    for name, (attr, expected) in PRECONDITION_CHECKS.items():
        for st in probes:
            actual = getattr(st, attr, None)
            want_refuse = actual is not None and actual != expected
            got_refuse = bool(check_state_preconditions([name], st))
            if want_refuse != got_refuse:
                mismatches.append(
                    f"{name}: state.{attr}={actual!r} 期望{'拒绝' if want_refuse else '放行'}"
                    f",实得{'拒绝' if got_refuse else '放行'}")
    assert not mismatches, "子串规则改判了精确表的判定:\n  " + "\n  ".join(mismatches)
    # 自检:探针真的覆盖了两种取值,否则上面的循环可能一条都没检到冲突。
    assert len(PRECONDITION_CHECKS) >= 5
    assert any("off" in n for n in PRECONDITION_CHECKS)


def test_the_substring_trap_itself_is_pinned():
    """把根因本身钉住:`'on'` 确实是 `'z_controller_off'` 的子串。

    钉它是因为**下一个人会重新想到用子串**(它看起来更宽容、更省事),
    而这条断言把「为什么不能」变成一句可执行的话,不是一段会被跳过的注释。
    """
    assert "on" in "z_controller_off", "c-ON-troller"
    # 而且第一条 z_controller 规则仍然会命中它 —— 所以保护必须来自「精确表优先」,
    # 不能指望规则顺序。
    z_rules = [subs for subs, *_ in _SUBSTRING_RULES if "z_controller" in subs]
    assert any(all(s in "z_controller_off" for s in subs) for subs in z_rules)


# ══════════════════════════════════════════════════════════════════════
# 二、缓存回写
# ══════════════════════════════════════════════════════════════════════

def test_a_successful_write_patches_the_cache():
    """写回用的是真的 `InstrumentState.apply_patch`,不是替身。"""
    state = InstrumentState(object())            # pool 只被存起来,不调用
    assert state.snapshot().z_controller_on is None
    patch_state_from_result(state, {"z_controller_on": True, "verified": True},
                            what="ZControllerOnOff")
    assert state.snapshot().z_controller_on is True


def test_the_write_back_never_raises():
    """缓存一致性不许影响技能结果。"""
    class _Boom:
        def apply_patch(self, **kw):
            raise RuntimeError("nope")

    patch_state_from_result(_Boom(), {"z_controller_on": True})   # 不抛
    patch_state_from_result(None, {"z_controller_on": True})
    patch_state_from_result(InstrumentState(object()), None)


def test_both_paths_share_one_write_back_implementation():
    """结构闸门:两条路径都必须调**同一个**函数。

    在此之前写回只存在于 agent 边界,而 composite 子步骤那条没有 —— 两处各写一遍
    的下一步就是只有一处被修。
    """
    import ast

    root = Path(_MASTV2_ROOT) / "mast"
    callers = {
        "agent 工具边界": root / "agents" / "_shared" / "skill_adapter.py",
        "composite 子步骤": root / "core" / "execution_context.py",
    }
    for label, path in callers.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "patch_state_from_result" in names, (
            f"{label}({path.name})没有调用共用的写回函数")
        # 而且不许再各自直接调 apply_patch(那就是第二份实现)。
        attrs = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert "apply_patch" not in attrs, (
            f"{label} 又直接调了 apply_patch —— 两份实现,改一处漏一处")


def test_the_composite_path_rechecks_after_a_refresh():
    """前置不满足时先问一次硬件再判一次 —— 手动路径从 v1 起就这么做,
    而 agent / composite 那条一直没有。这条不对称本身值得钉。"""
    import ast

    root = Path(_MASTV2_ROOT) / "mast"
    ec = (root / "core" / "execution_context.py").read_text(encoding="utf-8")
    assert "_recheck_after_refresh" in ec
    tree = ast.parse(ec)
    fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_recheck_after_refresh" in fns
    # 手动路径那份仍然在(两条路径做同一件事)。
    ex = (root / "core" / "executor.py").read_text(encoding="utf-8")
    assert "self._state.refresh()" in ex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
