"""每一张图都必须说清楚**它的坐标是哪来的**。

## 起因

问题:扫图前是否都会检查地图可用区域,没有就换;这应是自动流程,而非需要特别强调的步骤。

查下来**不自动**:``_tip_phases.py`` 里 6 处 ``_relocate`` 和 5 处扫图靠人肉配对。
风险很实在 —— 下一个人加一张图、忘了先找干净地方,就会**在刚炸出来的坑上判针尖**,
而那正是 6.2.19 修过的那个 bug。

## 但也不能改成「自动找干净地方」

一半的扫图是**故意**指定坐标的:

  · 200 nm 回退图 —— 要的就是**同一片**换个视野(换地方就换了问题)
  · 看簇的小图   —— 要看**刚扎出来的那个簇**(换地方等于换对象)
  · 验收图       —— 要回到**台阶处**(换到干净地方就没有台阶可量)

自动找会把这三种全毁掉,而且毁得很安静。

## 所以强制的是「表态」,不是「去找」

``scan_at_params(..., origin=...)`` 必填,取值限于 :data:`SCAN_ORIGINS` 受控词表。
忘了找 → 没有合法的 origin 可填;故意指定 → 填对应那一个,理由留在调用点。

**这条测试防的是遗忘,不是防错**:填错值靠人读代码,而忘了填在 Python 层面就
过不去(必填的 keyword-only 参数)。
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of

#: 仓库根 = 本文件往上 5 级(composite/skills/unit/v2/tests → 根)。
#: 算错一级会让 ast 扫描找不到文件而**整条测试 error** —— 那看起来像基建坏了,
#: 不像闸门在报警,所以下面额外断言目录真的存在。
_COMPOSITE = (pathlib.Path(__file__).parents[5] / "MASTv2" / "mast"
              / "skills" / "composite")
assert _COMPOSITE.is_dir(), f"找不到 composite 目录:{_COMPOSITE}"
#: 会扫图的 composite 文件。新增一个就加进来 —— 漏掉的那个文件不会被这道闸门看住。
_FILES = ("_tip_phases.py", "forge_au_tip.py", "prepare_noble_tip.py")


def _composite_steps(path: pathlib.Path):
    """→ [(skill_name, keywords dict, lineno)],只取 ``CompositeStep(...)`` 调用。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if getattr(fn, "id", None) != "CompositeStep":
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        sk = kw.get("skill_name")
        if isinstance(sk, ast.Constant) and isinstance(sk.value, str):
            yield sk.value, kw, node.lineno


def test_origin_is_required_and_vocabulary_is_closed():
    """必填 + 受控词表。拼错的 origin 和没想过的 origin 长得一模一样。"""
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as wf
    from mast.skills.composite._tip_phases import SCAN_ORIGINS, scan_at_params

    sig = inspect.signature(scan_at_params)
    p = sig.parameters["origin"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty, (
        "origin 有了默认值 ⇒「忘了表态」会静默变成「表了某个态」—— "
        "这道闸门的全部作用就没了")

    # 忘了传 → TypeError(不用等测试跑到那条路径)
    with pytest.raises(TypeError):
        scan_at_params(wf, 0.0, 0.0, size_nm=100.0, pixels=256, line_time_s=0.15)

    # 传了词表外的值 → ValueError
    with pytest.raises(ValueError, match="受控词表"):
        scan_at_params(wf, 0.0, 0.0, size_nm=100.0, pixels=256,
                       line_time_s=0.15, origin="wherever")

    assert set(SCAN_ORIGINS) == {"clean_spot", "same_frame", "analysis",
                                 "just_poked"}, (
        "词表变了 —— 新增取值要同时写清楚它是哪种情况,并在这里更新")
    for key, why in SCAN_ORIGINS.items():
        assert why.strip(), f"{key} 没有说明它是什么情况"

    # origin 不进 params(ScanAt 不认识这个键)
    got = scan_at_params(wf, 0.0, 0.0, size_nm=100.0, pixels=256,
                         line_time_s=0.15, origin="clean_spot")
    assert "origin" not in got


def test_every_scan_at_step_declares_its_origin():
    """⭐ **所有** ``skill_name="ScanAt"`` 的步骤都走 ``scan_at_params`` 且带 origin。

    这条是给**下一个加扫图的人**准备的:必填参数挡住了忘记,而这条 AST 断言
    挡住「绕过 scan_at_params 自己手写一份 params dict」——
    那正是 2026-08-10 之前的样子(四个调用点各写一份,没有一处传工作点)。
    """
    seen = 0
    for name in _FILES:
        path = _COMPOSITE / name
        for skill, kw, lineno in _composite_steps(path):
            if skill != "ScanAt":
                continue
            seen += 1
            params = kw.get("params")
            assert isinstance(params, ast.Call) and \
                getattr(params.func, "id", None) == "scan_at_params", (
                f"{name}:{lineno} 的 ScanAt 没有走 scan_at_params —— "
                "工作点和坐标来源都会绕过唯一的组装点")
            origins = [k for k in params.keywords if k.arg == "origin"]
            assert origins, f"{name}:{lineno} 的 ScanAt 没有声明 origin"
            val = origins[0].value
            assert isinstance(val, ast.Constant) and isinstance(val.value, str), (
                f"{name}:{lineno} 的 origin 不是字面量 —— "
                "受控词表要在静态就看得出来,不能算出来")
    assert seen >= 5, f"只找到 {seen} 处 ScanAt,是不是有文件没进 _FILES?"


def test_the_verify_frame_still_gets_its_spot_from_the_map():
    """verify 走 ``PreScanCheck``(不经 scan_at_params),单独钉它的坐标来源。

    它是唯一一张不由 ``scan_at_params`` 组装的图,所以上面那条 AST 闸门看不住它。
    传空排除表就是「在刚炸出来的坑上判针尖」的那半个原因(6.2.19 修的)。
    """
    from mast.skills.composite import _tip_phases

    src = source_of(_tip_phases.verify_phase)
    i = src.index('skill_name="PreScanCheck"')
    head = src[:i]
    assert "_relocate(" in head, (
        "verify 在扫图前不再找干净地方 —— 它会在刚打过脉冲的坑上判针尖")
    assert "_dirty(executor)" in head, (
        "verify 的 _relocate 没有传全流程脏点表 —— "
        "传空表等于没排除,那正是 6.2.19 修的那半个原因")


def test_scan_origins_are_actually_used():
    """词表里的每一条都得有人用 —— 没人用的取值是**死词表**,读起来像在治理。"""
    from mast.skills.composite._tip_phases import SCAN_ORIGINS

    used: set[str] = set()
    for name in _FILES:
        for skill, kw, _ln in _composite_steps(_COMPOSITE / name):
            if skill != "ScanAt":
                continue
            params = kw.get("params")
            if not isinstance(params, ast.Call):
                continue
            for k in params.keywords:
                if k.arg == "origin" and isinstance(k.value, ast.Constant):
                    used.add(k.value.value)

    unused = set(SCAN_ORIGINS) - used
    assert not unused, (
        f"这些 origin 没有任何调用点在用:{sorted(unused)} —— "
        "要么接上,要么从词表里删掉。一个没人用的取值会让下一个人以为"
        "「有这种情况被处理过」。")
