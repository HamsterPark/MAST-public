"""可选的反馈恢复步骤失败时也须报告尝试与原因，让后续 MoveToXY 前置失败有完整上下文。"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
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

# ⚠️ ``_fast_and_offline`` 是 **autouse fixture,必须跟着一起导进来**(2026-08-15)。
#
# autouse 只对**定义它的那个模块**里的测试生效。拿到了 ``_run`` 而没拿到它,
# 意味着这里跑的是**没打替身**的 ``_qplus_blocked`` / ``_tip_shaper_preflight``
# —— 两个都去读进程里的针尖/仪器状态。单跑这个文件时那是「读不到 ⇒ 放行」,
# 于是一路绿;而只要别的测试在同一进程里留下一个 qPlus 针尖态,``ForgeAuTip``
# 就在第一步 abort,连 ``ZControllerOnOff`` 都不会被调到,本文件的断言全部落空。
#
# 实测坐实:把 ``_tip_policy.qplus_gate`` 打成「拦」,同一条脚本的 ScanAt 调用数
# 从 **15 掉到 0**、outcome 从 ready 变 aborted。
# 完整论证在 ``test_forge_scan_working_point.py`` 同一处导入的注释里。
# 导入而不是抄一份:替身只有一份。
from tests.v2.unit.skills.composite.test_forge_au_tip import (  # noqa: E402
    FAIL,
    _base_script,
    _fast_and_offline,  # noqa: F401 —— autouse fixture,导入即生效,别删
    _run,
)


def _notes(data: dict) -> list[str]:
    return [str(v) for k, v in data.items()
            if k.endswith(":feedback_restore_failed")]


def test_a_failed_feedback_restore_says_so():
    """恢复反馈失败 → 报告里有它自己的一句话。"""
    res, ctx = _run(_base_script(ZControllerOnOff=FAIL), max_sites=1)
    assert ctx.count("ZControllerOnOff") >= 1, "这条流程根本没试过开反馈?"
    notes = _notes(res.data)
    assert notes, (
        "恢复反馈失败了,而报告里一个字都没提 —— 现场只会看到 MoveToXY 的前置失败,"
        f"然后去找一个关它的凶手。data 的键:{sorted(res.data)}")
    joined = " ".join(notes)
    assert "ZControllerOnOff" in joined
    # 必须把人从「谁关了它」引开 —— 那正是两次真机被读错的方向。
    assert "这一步本身失败了" in joined


def test_a_working_restore_stays_quiet():
    """成功时不许留噪声 —— 一条永远出现的「警告」等于没有警告。"""
    res, _ctx = _run(_base_script(), max_sites=1)
    assert _notes(res.data) == []


def test_the_step_is_still_optional():
    """钉住被否掉的那个「顺手」改法:把它改成必需步。

    改成 `optional=False` 看起来更严格,实际是把 `MoveToXY` 那句**精确**的前置失败
    换成 `ZControllerOnOff` 一句更早、更含糊的失败,而且流程会在一个本来可能能走
    下去的地方中止(反馈可能在下一步之前就自己回来了)。要的是「说出来」,
    不是「停下来」。"""
    import ast

    src = Path(_MASTV2_ROOT) / "mast" / "skills" / "composite" / "_tip_phases.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    found = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "CompositeStep"):
            continue
        kws = {k.arg: k.value for k in node.keywords}
        sid = kws.get("step_id")
        text = ast.unparse(sid) if sid is not None else ""
        if "pre_move_feedback" not in text:
            continue
        found += 1
        opt = kws.get("optional")
        assert isinstance(opt, ast.Constant) and opt.value is True, (
            "移动前的恢复反馈步被改成了必需步 —— 它会盖掉 MoveToXY 那句精确的"
            "前置失败,而那句话是现场唯一的线索")
    # 2026-08-17 起有**两处**:``_relocate`` 一处、扎针前的 ``_move_to`` 一处。
    # 后者是那天补的 —— 在此之前扎针**根本不移动**,针扎在上一次停下的地方,
    # 而簇图扫在台面落点上(要求:「那我们找台面是在做什么?」)。
    # 断言改成「至少一处,而且每一处都是 optional」—— 上面的循环已经逐处查过了。
    assert found >= 1, "一处 pre_move_feedback 都没找到 —— AST 匹配写歪了"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
