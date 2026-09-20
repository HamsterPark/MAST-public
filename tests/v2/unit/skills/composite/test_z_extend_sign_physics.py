"""使用合成负极性仪器验证退针方向。

模拟原始 Z 读数独立于被测判决生成，防止用结论验证配置。未配置极性必须拒绝判定。
"""
from __future__ import annotations

# ── path bootstrap ──
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

import pytest

from mast.core import instrument_profile as ip
from mast.skills.composite._z_settle import ZSettle
from mast.skills.composite.relocate_coarse_xy import RelocateCoarseXY
from mast.skills.composite.retract_for_sample_change import RetractForSampleChange

# ── 合成仪器的两端轨值──────────────────────────
#: 压电**收缩**到底 = 针尖离样品最远。``ZCtrl_Withdraw`` 停在这里。
Z_RAIL_WITHDRAWN_M = +160e-9
#: 压电**伸长**到底 = 针尖离样品最近。反馈找不到表面时停在这里。
Z_RAIL_EXTENDED_M = -160e-9

#: 模拟仪器的符号，由上面两个数唯一确定：伸长端比收缩端**小** ⇒ -1。
rig_SIGN = "-1" if Z_RAIL_EXTENDED_M < Z_RAIL_WITHDRAWN_M else "+1"

#: 合成退针移动距离。
TRAVEL_M = 40e-9
#: 隧穿态时压电落在量程中段的某处。取哪个值都行，下面只用**差**。
Z_JUNCTION_M = -20.0e-9


def _extended_by(z0: float, d: float) -> float:
    """从 ``z0`` **伸长** ``d`` 之后的 Z 读数。

    这一行就是那句物理事实本身：伸长 ⇒ 朝伸长轨走 ⇒ 在模拟仪器上 Z 变**小**。
    刻意不写成 ``z0 - d``：那样的话「为什么是减」就只剩注释在解释，
    而这里让它由**两个合成轨值**决定，换一台机器改常数即可。
    """
    return z0 + d * (1.0 if Z_RAIL_EXTENDED_M > Z_RAIL_WITHDRAWN_M else -1.0)


def _contracted_by(z0: float, d: float) -> float:
    """从 ``z0`` **收缩** ``d`` 之后的 Z 读数（伸长的反面）。"""
    return _extended_by(z0, -d)


def test_the_physical_fact_this_file_is_built_on():
    """先把上面那句话本身钉住 —— 后面每一条都从它推出来。

    没有这一条的话，``_extended_by`` 写反了会让所有断言一起反过来、
    然后整个文件依然全绿：**一个会跟着被测对象一起错的判据**。
    """
    assert Z_RAIL_EXTENDED_M < Z_RAIL_WITHDRAWN_M, "模拟仪器伸长端在收缩端之下"
    assert rig_SIGN == "-1"
    # 伸长 ⇒ Z 变小。这就是那句中文对应的数值方向。
    assert _extended_by(0.0, TRAVEL_M) < 0.0
    assert _contracted_by(0.0, TRAVEL_M) > 0.0


def _settle(z_m: float, *, state: str = "tracking") -> ZSettle:
    return ZSettle(z_m=z_m, current_a=1e-14, setpoint_a=100e-12,
                   settled=True, state=state)


def _verdicts(baseline: ZSettle, after: ZSettle) -> tuple[str, str]:
    """同一对读数喂给两个判据，返回 (relocate, retract) 的判决。

    两个 composite 各有一份 ``_judge_recede``，而它们必须给出**同一个**答案：
    一个机器不能既在远离又在靠近。分开写是历史，不是设计。
    """
    reloc = RelocateCoarseXY()._judge_recede(baseline, after)[0]
    t = RetractForSampleChange()
    t._baseline = baseline
    t._setpoint_a = baseline.setpoint_a
    retr = t._judge_recede(after)[0]
    return reloc, retr


@pytest.fixture
def _rig():
    """一台**如实声明了自己符号**的模拟仪器。"""
    before = ip.get_profile()
    ip.set_profile({"z_extend_sign": rig_SIGN, "z_recede_min_nm": 1.0})
    yield
    ip.set_profile(before)


# ── ① 正确的退针必须判 receding ──────────────────────────────────────────────
def test_a_real_retract_reads_as_receding(_rig):
    """粗动把针尖拉远 ⇒ 反馈**伸长**去追 ⇒ 模拟仪器 Z 变小 ⇒ 必须是 ``receding``。

    钉的是「伸长」这个物理动作，不是「Z 减小」这个数值 —— 数值由
    ``_extended_by`` 从两个轨推出来。
    """
    baseline = _settle(Z_JUNCTION_M)
    after = _settle(_extended_by(Z_JUNCTION_M, TRAVEL_M))
    assert after.z_m < baseline.z_m          # 模拟仪器上「伸长」长这样
    assert _verdicts(baseline, after) == ("receding", "receding")


# ── ② 针尖在逼近必须判 approaching ───────────────────────────────────────────
def test_a_tip_closing_in_reads_as_approaching(_rig):
    """粗动方向配反 ⇒ 样品变近 ⇒ 反馈**收缩**保住结 ⇒ 模拟仪器 Z 变大 ⇒ ``approaching``。

    **这一条是这条自检存在的全部理由。** 它误判成 ``receding`` 的代价不是
    「烦」：梯子会接着爬到 10 步、89 步，全部朝样品里送。
    """
    baseline = _settle(Z_JUNCTION_M)
    after = _settle(_contracted_by(Z_JUNCTION_M, TRAVEL_M))
    assert after.z_m > baseline.z_m          # 模拟仪器上「收缩」长这样
    assert _verdicts(baseline, after) == ("approaching", "approaching")


# ── ③ 四种组合摊开：配错符号会把两个判决**整个对调** ────────────────────────
@pytest.mark.parametrize("declared,motion,expected", [
    # 声明对了（-1）：物理怎么走，判决就怎么说。
    ("-1", "extend", "receding"),
    ("-1", "contract", "approaching"),
    # 声明反了（+1）：两个判决**互换**。第二行是致命的那一半 ——
    # 针尖在靠近，而自检说它在远离，于是放行接下来的几百步粗动。
    ("+1", "extend", "approaching"),
    ("+1", "contract", "receding"),
])
def test_four_combinations_of_sign_and_motion(declared, motion, expected):
    before = ip.get_profile()
    try:
        ip.set_profile({"z_extend_sign": declared, "z_recede_min_nm": 1.0})
        baseline = _settle(Z_JUNCTION_M)
        move = _extended_by if motion == "extend" else _contracted_by
        after = _settle(move(Z_JUNCTION_M, TRAVEL_M))
        assert _verdicts(baseline, after) == (expected, expected)
    finally:
        ip.set_profile(before)


def test_the_deadly_half_is_named(_rig):
    """把**被否掉的方案**钉成测试:出厂 ``+1`` 曾经是这条链上活着的默认值。

    (既有教训：否掉的东西要有名字，否则它会被
    当成冗余删回去。)

    失败方向不对称，两半各写一行：
      · 误判 approaching ⇒ 对一次正确的退针 panic 撤针（烦，安全）；
      · 误判 receding    ⇒ **针尖在靠近却说在远离**，然后放行几百步粗动。
    """
    baseline = _settle(Z_JUNCTION_M)
    closing_in = _settle(_contracted_by(Z_JUNCTION_M, TRAVEL_M))

    # 如实声明 ⇒ 抓住。
    assert _verdicts(baseline, closing_in) == ("approaching", "approaching")

    # 声明成出厂那个值 ⇒ 同一组读数被说成「在远离」。这就是被否掉的那个默认值。
    before = ip.get_profile()
    try:
        ip.set_profile({"z_extend_sign": "+1", "z_recede_min_nm": 1.0})
        assert _verdicts(baseline, closing_in) == ("receding", "receding"), (
            "符号反了却没有把判决对调 —— 那说明这个测试没有在测符号")
    finally:
        ip.set_profile(before)


# ── ④ 没声明过 ⇒ 两个判据都必须拒判，而不是吃出厂值 ─────────────────────────
def test_undeclared_sign_refuses_to_judge():
    """从没配过的机器上，``get_z_extend_sign()`` 会给出 ``+1`` —— 而 ``+1`` 在模拟仪器
    是**反的**。所以判据必须能说「不知道」。

    这一条同时是**变异钉**：把两个 ``_judge_recede`` 里的
    ``ip.z_extend_sign_or_none()`` 写回 ``ip.get_z_extend_sign()``（= 吃出厂 ``+1``）
    ⇒ 这里会拿到 receding/approaching 而不是 no_sign ⇒ 红。
    """
    before = ip.get_profile()
    try:
        ip.set_profile({"z_recede_min_nm": 1.0})       # 刻意不写 z_extend_sign
        assert ip.z_extend_sign_or_none() is None
        assert ip.get_z_extend_sign() == 1             # 出厂值仍在，只是没人再吃它

        baseline = _settle(Z_JUNCTION_M)
        for motion in (_extended_by, _contracted_by):
            after = _settle(motion(Z_JUNCTION_M, TRAVEL_M))
            assert _verdicts(baseline, after) == ("no_sign", "no_sign")
    finally:
        ip.set_profile(before)


def test_undeclared_sign_says_which_setting_to_fill():
    """拒判的同时要给处方，而且**不能是 unsettled 那张方子**。

    ``unsettled`` 的处方是「调大 z_settle_timeout_s」。对着一个没填的符号开那张
    方子，就是本仓反复栽的「把人指向一个没坏的东西」。
    """
    before = ip.get_profile()
    try:
        ip.set_profile({"z_recede_min_nm": 1.0})
        baseline = _settle(Z_JUNCTION_M)
        after = _settle(_extended_by(Z_JUNCTION_M, TRAVEL_M))
        _, why = RelocateCoarseXY()._judge_recede(baseline, after)
        assert "z_extend_sign" in why
        assert "z_settle_timeout_s" not in why
    finally:
        ip.set_profile(before)


# 第四个消费者(``_phase_clear`` 台账里的 ``dz_m``)要跑整条 composite 才够得着,
# 钉在有假机器的那个文件里:``test_relocate_coarse_xy.py::
# test_undeclared_sign_stops_the_ladder_and_leaves_no_dz_in_the_ledger``。
