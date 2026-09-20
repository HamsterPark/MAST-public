"""电流跳闸报告阈值来源、设定点是否可读以及读数是否收敛。"""
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

from mast.skills.composite._z_settle import ZSettle  # noqa: E402
from mast.skills.composite.relocate_coarse_xy import RelocateCoarseXY  # noqa: E402

# 独立构造的电流与设定点输入。
SYNTHETIC_CURRENT = 1.4e-9
VERIFY_SETPOINT = 1e-10          # verify 相设的 100 pA —— 换位前最后一次设的
JUNCTION_SETPOINT = 1e-9         # pulse 相的结条件 1000 pA


def _settle(*, current, setpoint, z_m=1.0e-7, converged=True) -> ZSettle:
    """一个 ZSettle。``usable`` 由 ``settled`` + ``state`` 共同决定 ——
    只设 ``state`` 不设 ``settled`` 的话它永远是 False,而那会让
    「收敛/未收敛」两条断言都走到同一支(第一版就是这么写的,自检当场逮到)。"""
    s = ZSettle()
    s.settled = bool(converged)
    s.state = "tracking" if converged else "moving"
    s.z_m = z_m
    s.current_a = current
    s.setpoint_a = setpoint
    return s


def _judge(current, setpoint, *, settled=True):
    after = _settle(current=current, setpoint=setpoint, converged=settled)
    base = _settle(current=0.0, setpoint=setpoint, z_m=0.0)
    return RelocateCoarseXY()._judge_recede(base, after)


@pytest.fixture(autouse=True)
def _declare_the_z_sign():
    """本文件测的是**电流**那一支,但 Z 那一支要先能跑到。

    当前实现中 ``_judge_recede`` 不再吃 ``z_extend_sign`` 的出厂默认:
    没声明过的机器一律 ``no_sign``。这里的读数 (``z_m`` 从 0 升到 2e-7) 是
    「伸长时 Z 增大」那种机器,所以如实声明 ``+1``。
    """
    from mast.core import instrument_profile as ip

    before = ip.get_profile()
    ip.set_profile(dict(before, z_extend_sign="+1"))
    yield
    ip.set_profile(before)


# ══════════════════════════════════════════════════════════════════════
# 一、说出哪一项赢了
# ══════════════════════════════════════════════════════════════════════

def test_the_message_names_which_term_set_the_bar():
    """合成输入使绝对地板决定阈值，报告应明确说明。"""
    verdict, why = _judge(SYNTHETIC_CURRENT, VERIFY_SETPOINT)
    assert verdict == "approaching", why
    # ⚠️ 断言必须钉**那个论断本身**,不是「地板」两个字。
    # 第一版写的是 `"地板" in why` —— 而消息里还有一句「相对项 X,地板 Y」在列数,
    # 所以把论断整个换成「预期」之后它照样绿。变异当场逮到:
    # **一个在字符串别处也成立的断言,等于没有断言。**
    assert "由**绝对地板**决定" in why, why
    # setpoint 读到了多少,也要说 —— 它是「相对判据」这四个字的全部内容。
    assert "1e-10" in why or "1.000e-10" in why or "1e-010" in why, why
    assert "3e-10" in why or "3.000e-10" in why or "3e-010" in why, why


def test_the_relative_term_wins_when_the_setpoint_is_large():
    """结条件 1 nA 下:相对项 3e-9 > 地板,而 1.4 nA **不该**触发。

    这条同时是探针:如果哪天它开始触发,说明相对判据被什么东西架空了。
    """
    verdict, why = _judge(SYNTHETIC_CURRENT, JUNCTION_SETPOINT)
    assert verdict != "approaching", (verdict, why)


def test_a_big_current_at_a_big_setpoint_still_trips_and_says_relative():
    verdict, why = _judge(5e-9, JUNCTION_SETPOINT)
    assert verdict == "approaching"
    assert "由**相对项 3×setpoint**决定" in why, why


# ══════════════════════════════════════════════════════════════════════
# 二、读不到 setpoint = 判不了,**不出方向判定**
# ══════════════════════════════════════════════════════════════════════

def test_no_setpoint_falls_through_to_the_z_evidence(monkeypatch):
    """相对判据没有相对量时:**跳过电流这条,落到 Z 方向这条主证据**。

    ⚠️ 我的第一版返回 `unsettled` —— **那是错的,而且不解决问题**:
    * 语义错:`unsettled` 对上层的意思是「Z 读数取不到」,它的处方是
      「调大 z_settle_timeout_s」;对着一个 setpoint 读不到的毛病开那张方子,
      又是一次把人指向没坏的东西。
    * 而且它**照样中止外环**(`_phase_clear` 对 unsettled 也 panic + fail),
      所以只是把错话换成对话,6.2.7 仍然过不去。

    正确的是:电流只在反方向上当危险跳闸用(本函数开头写着),**主证据是 Z 方向**。
    没有相对量就不用电流那条,让 Z 说话 —— 针尖真在逼近时压电会缩回,那一支照判。

    **要推翻需要回答**:有没有一种情况,压电方向说远离而针尖其实在逼近?
    """
    import mast.core.instrument_profile as ip
    monkeypatch.setattr(ip, "get_config",
                        lambda k, d=None: None if k == "preamp_full_scale_a"
                        else (1.0 if k == "z_recede_min_nm" else d),
                        raising=False)
    # Z 伸长 = 在追远离的样品 ⇒ receding,即使电流有 1.4 nA 且 setpoint 读不到。
    after = _settle(current=SYNTHETIC_CURRENT, setpoint=None, z_m=2e-7)
    base = _settle(current=0.0, setpoint=None, z_m=0.0)
    verdict, why = RelocateCoarseXY()._judge_recede(base, after)
    assert verdict == "receding", (verdict, why)
    # 但**必须说出电流那条这次没判** —— 跳过一条判据不许悄悄跳。
    assert "没判" in why, why
    assert "读不到 setpoint" in why, why


def test_a_railed_preamp_still_trips_without_a_setpoint(monkeypatch):
    """读不到 setpoint 时仍然保留一条**与工作点无关**的界:前置放大器满量程。

    「电流大到放大器已经饱和」与「你打算跑在多少 pA」无关,所以它在任何工作点上
    都成立 —— 而 1e-9 那个地板不是(它是照成像条件定的)。
    """
    import mast.core.instrument_profile as ip
    monkeypatch.setattr(ip, "get_config",
                        lambda k, d=None: 1e-8 if k == "preamp_full_scale_a"
                        else (1.0 if k == "z_recede_min_nm" else d),
                        raising=False)
    verdict, why = _judge(2e-8, None)          # 2e-8 ≥ 满量程 1e-8
    assert verdict == "approaching", (verdict, why)
    assert "满量程" in why and "与工作点无关" in why, why
    # 合成电流低于满量程时不能触发满量程退针条件。
    v2, _ = _judge(SYNTHETIC_CURRENT, None)
    assert v2 != "approaching"


def test_zero_and_none_are_two_different_readings():
    """`setpoint or 0.0` 把 `None` 与 `0.0` 一视同仁 —— 这两者含义不同。

    读出来是 0(用户真把设定点设成 0)是一个**测量**,判据可以用它(地板生效,
    1.4 nA 触发);读不到是**没有测量**,那条判据整个不适用。
    折叠它们正是本仓反复记的那条。
    """
    v_none, _ = _judge(SYNTHETIC_CURRENT, None)
    v_zero, why_zero = _judge(SYNTHETIC_CURRENT, 0.0)
    assert v_zero == "approaching", why_zero
    assert v_none != v_zero


def test_the_reason_for_an_unreadable_setpoint_is_carried():
    """消费方:「读不到」要把**为什么**带进结论。"""
    after = _settle(current=SYNTHETIC_CURRENT, setpoint=None, z_m=2e-7)
    after.setpoint_why = "ZCtrl_SetpntGet 报错:module not running"
    base = _settle(current=0.0, setpoint=None, z_m=0.0)
    _v, why = RelocateCoarseXY()._judge_recede(base, after)
    assert "module not running" in why, why


@pytest.mark.parametrize("mode,marker", [("error", "报错"), ("junk", "读不懂")])
def test_the_producer_actually_records_why_the_setpoint_was_unreadable(mode, marker):
    """**产出方**:`settle_and_read_z` 必须真的把原因写进去。

    ⚠️ 上面那条只是**手动**给 `setpoint_why` 赋了值 —— 它钉的是消费方。
    变异验证当场逮到:把产出方那一行改成写空串,上面那条照样绿。
    「钉到判定层,抓不到落地层」——本仓记过的那条,这次又踩了一遍。

    两种成因分开测:TCP/模块报错 vs 回包读不懂,它们指向不同的下一步。
    """
    from mast.core.types import NanonisCallRecord
    from mast.skills.composite._z_settle import settle_and_read_z

    class _Ctx:
        def safe_call(self, method, *args, role="main"):
            if method == "ZCtrl_SetpntGet":
                if mode == "error":
                    return NanonisCallRecord(method=method, args=args,
                                             error="module not running",
                                             return_value=None)
                # 构造可读但形状错误的回包，验证退化处理。
                return NanonisCallRecord(method=method, args=args, error="",
                                         return_value=("", b"", [(1e-10,), (2,)]))
            if method == "ZCtrl_ZPosGet":
                return NanonisCallRecord(method=method, args=args, error="",
                                         return_value=("", b"", [1e-7]))
            if method == "Current_Get":
                return NanonisCallRecord(method=method, args=args, error="",
                                         return_value=("", b"", [1e-12]))
            return NanonisCallRecord(method=method, args=args, error="",
                                     return_value=("", b"", [0.0]))

        def check_abort(self):
            return False

    out = settle_and_read_z(_Ctx(), withdraw_first=False, timeout_s=0.3,
                            poll_interval_s=0.01, window_n=3)
    assert out.setpoint_a is None
    assert marker in out.setpoint_why, out.setpoint_why
    # 台账里也要有 —— 事后复盘读的是它。
    assert out.as_dict()["setpoint_why"] == out.setpoint_why


# ══════════════════════════════════════════════════════════════════════
# 三、这次读数收没收敛
# ══════════════════════════════════════════════════════════════════════

def test_an_unconverged_reading_is_flagged_in_the_trip():
    """危险判据**有意**跑在「读数收没收敛」之前(它必须在 settle 失败时仍然生效)。

    那就更要说出来:反馈刚被重新打开、Z 还在追的时候,一个大电流很可能是**瞬态**,
    而不是「台子走反了」。不说的话,这条判据会把一次收敛过程报成一个硬件接线错误
    —— 2026-08-10 就是这么读的。
    """
    _v, why_ok = _judge(SYNTHETIC_CURRENT, VERIFY_SETPOINT, settled=True)
    v_bad, why_bad = _judge(SYNTHETIC_CURRENT, VERIFY_SETPOINT, settled=False)
    assert v_bad == "approaching"          # 判据仍然生效
    assert "未收敛" in why_bad, why_bad
    assert "未收敛" not in why_ok, why_ok


def test_the_trip_still_fires_when_the_settle_failed():
    """钉住那条**被否掉的**改法:把危险判据挪到 `after.usable` 之后。

    看起来更严谨(不拿没收敛的读数下结论),实际是把唯一一条「必须在 settle 失败时
    仍然生效」的保护关掉 —— 而那正是它存在的理由(源码注释写着)。
    要的是**说出来**,不是**停下来**。
    """
    verdict, _why = _judge(5e-8, VERIFY_SETPOINT, settled=False)
    assert verdict == "approaching"


def test_the_prescription_matches_which_evidence_decided():
    """Z 方向证据与电流阈值证据应给出不同的处理建议。"""
    pres_z = RelocateCoarseXY._approach_prescription("Z 压电缩回 200.0 nm")
    pres_i = RelocateCoarseXY._approach_prescription(
        "电流 1.4e-09 A 高于阈值 1e-09 A(由**绝对地板**决定…);读数已收敛")
    assert "方向很可能配反" in pres_z, pres_z
    assert "z_settle_timeout_s" not in pres_z, "Z 那条不该开 settle 预算的方子"
    assert "绝对地板" in pres_i and "z_settle_timeout_s" in pres_i, pres_i
    assert pres_z != pres_i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
