"""未标定的锐度阈值应保持未设。

边缘宽度会随采样密度、插值和平滑核改变；稳定读数本身不证明具有独立物理含义。
只有消除算法尺度依赖并完成独立标定后，才能将 measured 变为 sharp/blunt 判决。"""
from __future__ import annotations

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of


def test_factory_default_is_unset():
    """出厂留空。见模块 docstring —— 这是 2026-08-14 的结论,不是待办。"""
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE, NobleTipWorkflow

    assert NobleTipWorkflow().accept_sharp_edge_nm is None
    assert NOBLE_METAL_BASELINE.accept_sharp_edge_nm is None


def test_unset_means_the_acceptance_step_does_not_block():
    """留空 ⇒ ``measured`` ⇒ ``undecidable`` ⇒ **不拦流程**。

    这条连着上一条一起看才有意义:留空之所以是安全的默认,正因为它不会把
    修针卡死 —— 否则"别填"就成了一句会造成损害的建议。
    """
    from mast.skills.composite.forge_au_tip import sharpness_verdict_kind

    assert sharpness_verdict_kind("measured") == "undecidable"
    assert sharpness_verdict_kind("no_step") == "undecidable"
    assert sharpness_verdict_kind("unresolved") == "undecidable"
    # 只有真的量到"不够尖"才拦
    assert sharpness_verdict_kind("blunt") == "fail"
    assert sharpness_verdict_kind("sharp") == "pass"


def test_threshold_is_only_sent_when_set():
    """``_accept`` 只在字段非空时才把 ``sharp_edge_nm`` 传下去。

    钉这一条是因为「留空」的语义**完全依赖于这个 if**:哪天有人给它补一个
    默认值(哪怕是在 ParameterSpec 那一侧),「没设」就会变成「设了某个数」——
    ``prescan_check.py:96-102`` 记着这个坑的另一次发作。
    """
    import inspect

    from mast.skills.composite.forge_au_tip import ForgeAuTip

    src = source_of(ForgeAuTip._accept)
    assert "wf.accept_sharp_edge_nm is not None" in src, (
        "_accept 不再用 `is not None` 守住阈值下发 —— "
        "确认「未设」没有被某个默认值悄悄变成「设了」")
