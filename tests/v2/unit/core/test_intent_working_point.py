# -*- coding: utf-8 -*-
"""工作点由**意图**决定，不由上一个 skill 决定。

「**每一个 skill 在一开始会制定自己的工作点，
而不是依赖上一个 skill。**」

背景（这条规则的代价）：resolver 原本的 bias 链是 explicit > prefs > keep-current，
理由写在代码里且成立 ——「bias 是物理意图参数，不是尺度的函数；同一个 50 nm 的框，
看形貌用 50 mV、看分子态用 1 V」。但它漏了一种情况：``purpose='atomic'`` **本身
就是那句意图**。缺了这一环，工作点一路 keep-current ⇒ 沿用上一个 skill 留下的值
⇒ 修针流程把偏压留在 1.0 V ⇒ 之后每帧 FFT 周期 0.30–0.36 nm、散布 80%+ ⇒
读成「针尖极其糟糕」⇒ 打脉冲 ⇒ **真把一根好针尖打坏了**。
"""
from __future__ import annotations

import pytest

from mast.core.scan_resolver import (
    SOURCE_EXPLICIT,
    SOURCE_INTENT,
    SOURCE_KEEP,
    ScanIntent,
    resolve_scan,
)
from mast.vision.imaging_window import ATOMIC_WORKING_POINT


def _r(size_m=5e-9, purpose="auto", explicit=None):
    return resolve_scan(ScanIntent(center_x_m=0.0, center_y_m=0.0, size_m=size_m,
                                   purpose=purpose, explicit=dict(explicit or {})))


def _src(res, key):
    return (res.trace.get(key) or {}).get("source")


@pytest.mark.parametrize("purpose", ["atomic", "atomic_verify"])
def test_naming_an_atomic_purpose_sets_its_own_working_point(purpose):
    """显式点名原子档 = 说了「我要看原子」= 该带上自己的工作点。"""
    r = _r(size_m=4e-9, purpose=purpose)
    assert (r.set_bias or {}).get("bias_v") == ATOMIC_WORKING_POINT["bias_v"]
    assert (r.set_setpoint or {}).get("setpoint_a") == ATOMIC_WORKING_POINT["setpoint_a"]
    assert _src(r, "bias_v") == SOURCE_INTENT
    assert _src(r, "setpoint_a") == SOURCE_INTENT


def test_auto_tier_selection_must_not_impose_a_bias():
    """**按尺寸碰巧定到原子档 ≠ 意图。**

    这一条守的是 resolver 原有的设计理由：同一个框可以想看别的东西。
    自动定档只是尺度巧合，不该替调用方决定探测什么电子态。
    """
    r = _r(size_m=4e-9, purpose="auto")
    assert r.tier_name in ("atomic", "atomic_verify"), "这个尺寸本该落在原子档"
    assert r.set_bias is None
    assert _src(r, "bias_v") == SOURCE_KEEP


@pytest.mark.parametrize("purpose,size", [("survey", 5e-7), ("roi", 3e-7),
                                          ("highres", 8e-8)])
def test_non_atomic_purposes_keep_current(purpose, size):
    """非原子意图不受影响 —— 这一环只服务「我要看原子」这一句。"""
    r = _r(size_m=size, purpose=purpose)
    assert r.set_bias is None
    assert _src(r, "bias_v") == SOURCE_KEEP


def test_explicit_beats_intent():
    """用户逐字点名的值永远压过意图默认 —— 优先级是 explicit > intent > keep。"""
    r = _r(size_m=4e-9, purpose="atomic", explicit={"bias_v": 0.5})
    assert (r.set_bias or {}).get("bias_v") == pytest.approx(0.5)
    assert _src(r, "bias_v") == SOURCE_EXPLICIT
    # setpoint 没被点名，仍走意图默认
    assert _src(r, "setpoint_a") == SOURCE_INTENT


def test_prefs_beat_intent():
    """实验默认偏好也压过意图默认（它比档位更贴近这次实验的约定）。"""
    pytest.importorskip("mast.core.scan_resolver")
    r = resolve_scan(ScanIntent(center_x_m=0.0, center_y_m=0.0, size_m=4e-9,
                                purpose="atomic", explicit={}),
                     prefs={"bias_v": 0.03})
    assert (r.set_bias or {}).get("bias_v") == pytest.approx(0.03)
    assert _src(r, "bias_v") == "prefs"


def test_the_intent_default_is_inside_the_atomic_window():
    """意图默认值本身必须能通过成像窗口闸门 —— 否则两处规则会互相打架。"""
    from mast.vision.imaging_window import check_atomic_window

    v = check_atomic_window(ATOMIC_WORKING_POINT["bias_v"],
                            ATOMIC_WORKING_POINT["setpoint_a"], "Au(111)")
    assert v.ok, v.reason


def test_the_incident_value_would_no_longer_survive():
    """事故复现：上一个 skill 留下 1.0 V，下一次显式要原子分辨时必须被改回来。

    这里断言的是 resolver **下发了 SetBias**（而不是 keep-current）——
    有没有下发，就是「自己制定工作点」与「依赖上一个 skill」的分界。
    """
    r = _r(size_m=4e-9, purpose="atomic")
    assert r.set_bias is not None, "必须下发 SetBias，否则就会沿用上一个 skill 的 1.0 V"
    assert abs((r.set_bias or {}).get("bias_v", 9.9)) <= 0.15
