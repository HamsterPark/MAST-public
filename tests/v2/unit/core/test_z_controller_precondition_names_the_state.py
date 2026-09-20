"""前置失败须指出 Z 模块的具体状态。实时反馈开关与模块状态属于不同寄存器，不能互相替代。"""
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
    PRECONDITION_CONTEXT_FIELDS,
    check_state_preconditions,
    context_suffix,
)
from mast.core.types import HardwareState  # noqa: E402

#: `state.py` 里那张码表 —— 这里**照抄一份是有意的**:如果那边改了而这边没改,
#: 下面 `test_every_non_on_status_reads_as_off` 会红,而那正是要提醒的事
#: (新增一个码就多一种「False 其实是什么」)。
_STATUS_CODES = {1: "Off", 2: "On", 3: "Hold", 4: "SwitchingOff",
                 5: "SafeTip", 6: "Withdrawing"}


def _state(**kw) -> HardwareState:
    return HardwareState(**kw)


# ══════════════════════════════════════════════════════════════════════
# 布尔背后那个更精确的字段
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("status", ["Off", "Hold", "SwitchingOff", "SafeTip",
                                    "Withdrawing"])
def test_the_message_names_which_of_the_six_it_is(status):
    """五种「False」各说各的,而不是同一句裸布尔。"""
    from mast.skills.base import BaseSkill
    from mast.core.types import SkillMetadata, SkillCategory, SafetyLevel

    class _Probe(BaseSkill):
        def metadata(self):
            return SkillMetadata(
                name="_Probe", version="1.0.0", category=SkillCategory.WRITE,
                safety_level=SafetyLevel.AUTO, description="probe",
                parameters=[], preconditions=["z_controller_on"])

        def execute(self, context, params):      # pragma: no cover - 不跑
            raise AssertionError

    unmet = _Probe().check_preconditions(
        _state(z_controller_on=False, z_controller_status=status))
    assert len(unmet) == 1, unmet
    assert status in unmet[0], (
        f"消息里没有出现模块状态 {status!r},它仍然是一句裸布尔:{unmet[0]}")


def test_hold_and_safetip_do_not_read_the_same():
    """两种最容易混淆的:「被挂起」与「仪器自己躲开了」。

    它们指向完全不同的下一步 —— 一个是「谁挂起的没放回来」,一个是
    「不是人关的,是 Nanonis 的保护态」。同一句话打发掉两者,就是 2026-08-10。"""
    hold = context_suffix("z_controller_on", _state(z_controller_status="Hold"))
    safe = context_suffix("z_controller_on", _state(z_controller_status="SafeTip"))
    assert hold and safe and hold != safe
    assert "挂起" in hold
    assert "SafeTip" in safe and "保护" in safe


def test_both_precondition_layers_print_it():
    """两层都要印。

    `preconditions.py` 开头写着:这两层曾经各写一份判据而悄悄漂移。只在其中一层
    加补充说明,下一次仍然会有一半现场看到裸布尔。"""
    st = _state(z_controller_on=False, z_controller_status="SafeTip")
    subs = check_state_preconditions(["z_controller_on"], st)
    assert len(subs) == 1 and "SafeTip" in subs[0], subs


def test_an_unreadable_status_does_not_invent_one():
    """读不到就不写 —— 「没读到」不能被印成一个编出来的状态名。"""
    assert context_suffix("z_controller_on", _state(z_controller_status=None)) == ""
    assert context_suffix("scan_running", _state(scan_running=True)) == ""


def test_the_diagnostics_ledger_only_names_real_state_fields():
    """台账里的字段名必须真的存在于 `HardwareState`。

    `base.py` 用 `if hasattr(state, k)` 过滤,于是**一个拼错的名字会被静默跳过** ——
    它不报错,只是那一格永远缺席。`tip_withdrawn` 就是这么活了很久的
    (真名是 `withdrawn`):「针尖是不是退开的」从上线起一次都没进过台账,
    而那份台账存在的全部理由就是事后复盘。
    """
    import ast

    src = Path(_MASTV2_ROOT).parent / "MASTv2" / "mast" / "skills" / "base.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    st = HardwareState()
    checked = 0
    for node in ast.walk(tree):
        # 找 `for k in (...) if hasattr(state, k)` 里那个字面元组。
        if not isinstance(node, ast.comprehension):
            continue
        if not isinstance(node.iter, ast.Tuple):
            continue
        names = [e.value for e in node.iter.elts
                 if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if not names or "z_controller_on" not in names:
            continue
        checked += 1
        missing = [n for n in names if not hasattr(st, n)]
        assert not missing, (
            f"台账列了 HardwareState 上不存在的字段 {missing} —— "
            "hasattr 会静默跳过它们,那几格永远是空的")
    assert checked == 1, f"没找到台账字段列表(找到 {checked} 处),AST 匹配写歪了"


def test_the_companion_field_actually_exists_on_the_state():
    """自检:伴随字段名写错的话,上面每一条都会静默退化成「读不到」而仍然全绿。"""
    assert PRECONDITION_CONTEXT_FIELDS, "映射表是空的,这个文件什么都没测"
    for attr, companion in PRECONDITION_CONTEXT_FIELDS.items():
        assert hasattr(HardwareState(), attr), attr
        assert hasattr(HardwareState(), companion), (
            f"{companion!r} 不是 HardwareState 的字段 —— 补充说明永远是空串")


# ══════════════════════════════════════════════════════════════════════
# 布尔是怎么算出来的 —— 六个码坍缩成两个值
# ══════════════════════════════════════════════════════════════════════

def test_every_non_on_status_reads_as_off():
    """钉住坍缩本身:只有 On(2) 是 True,其余五个码全是 False。

    这不是在提议改判定(对 MoveToXY 来说 Hold 确实不能移动),而是把
    「一个布尔背后有五种 False」这件事写成可执行的断言 —— 它正是消息必须带上
    状态名的**理由**,而理由不该只活在注释里。"""
    from mast.core import state as state_mod

    src = Path(state_mod.__file__).read_text(encoding="utf-8")
    assert "_ZCTRL_STATUS" in src and "(code == 2)" in src, (
        "state.py 里算 z_controller_on 的方式变了,这个文件的前提要重新核对")
    on_codes = [c for c, n in _STATUS_CODES.items() if (c == 2)]
    off_codes = [c for c, n in _STATUS_CODES.items() if (c != 2)]
    assert on_codes == [2]
    assert len(off_codes) == 5, off_codes


def test_the_two_registers_are_not_the_same_question():
    """`ZControllerOnOff` 校验的寄存器与这条前置读的**不是同一个**。

    `verify_z_controller` → `ZCtrl_OnOffGet`(实时控制器);
    `state.refresh` → `ZCtrl_StatusGet`(Z-Controller 模块)。
    Nanonis 手册明说两者可以不一致(`mast/skills/verify.py` 的模块注释逐字引了)。

    所以「打开反馈这一步成功了」和「下一步的 z_controller_on 前置仍然失败」
    **可以同时为真**。钉住它,是因为下一个人会把这两句话当成矛盾,然后去找一个
    根本不存在的 bug —— 2026-08-10 就是这么过去的。"""
    from mast.core import state as state_mod
    from mast.skills import verify as verify_mod

    verify_src = Path(verify_mod.__file__).read_text(encoding="utf-8")
    state_src = Path(state_mod.__file__).read_text(encoding="utf-8")
    assert 'safe_call("ZCtrl_OnOffGet")' in verify_src, (
        "verify_z_controller 不再读实时控制器了?这条测试的前提变了")
    assert 'safe_call("ZCtrl_StatusGet"' in state_src, (
        "state.refresh 不再读模块状态了?这条测试的前提变了")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
