# -*- coding: utf-8 -*-
"""CalibrateCoarseStep —— 用有刻度的压电量没刻度的马达。

三条设计都来自 2026-08-28 的失败：站点检查拒绝小步长、20 步在 800 nm 视野里
看不到位移（每步可能 100 nm 量级）、以及动马达时仪器锁还被 ScanAt 握着。
"""
from __future__ import annotations

import numpy as np
import pytest

from mast.skills.builtins import CalibrateCoarseStep
from mast.skills.builtins.coarse_step_calib import _MIN_CORR_SNR, phase_shift

BUSY = "仪器正被占用：技能直调 API 正在执行 ScanAt，已持有 363s。'MotorMove' 未执行"


def _img(n=256, shift=(0, 0), seed=0):
    """带一个强特征（坑）的合成帧，**整幅内容一起平移**。

    ⚠ 第一版只移动坑、噪声用同一个 seed 保持不动 —— 于是相关峰锁在「噪声对齐」的
    (0,0) 上，测试红了而代码是对的。真实情况是**整个视野的内容一起移动**（是台子在走），
    所以这里用 np.roll 整体平移。又一次「替身与现实不符」。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    cy, cx = n // 2, n // 2
    pit = -3000.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 12.0 ** 2)))
    base = pit + rng.normal(0, 20.0, (n, n))
    return np.roll(np.roll(base, shift[0], axis=1), shift[1], axis=0)


# ── 相位相关 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("dx,dy", [(0, 0), (10, 0), (0, -14), (25, 18), (-31, 7)])
def test_phase_shift_recovers_a_known_translation(dx, dy):
    a = _img()
    b = _img(shift=(dx, dy))
    gx, gy, snr = phase_shift(a, b)
    assert (gx, gy) == (-dx, -dy) or (gx, gy) == (dx, dy)
    assert snr > _MIN_CORR_SNR


def test_featureless_frames_give_a_weak_peak():
    """平坦表面上相位相关立不住 —— 峰锐度必须低到会被拒答。
    这正是 make_pit 存在的理由。"""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 20.0, (256, 256))
    b = rng.normal(0, 20.0, (256, 256))
    _, _, snr = phase_shift(a, b)
    assert snr < _MIN_CORR_SNR


def test_mismatched_shapes_return_none():
    assert phase_shift(_img(256), _img(128)) == (None, None, None)


# ── 参数校验 ───────────────────────────────────────────────────────────
class _Ctx:
    def __init__(self, busy=False):
        self.busy = busy
        self.calls: list[str] = []

    def run(self, name, params=None):
        self.calls.append(name)
        if name == "MotorMove" and self.busy:
            return type("R", (), {"success": False, "error": BUSY, "data": {}})()
        if name == "GetCurrent":
            return type("R", (), {"success": True, "data": {"current_a": 1e-13}})()
        if name == "GetScanFrame":
            return type("R", (), {"success": True,
                                  "data": {"center_x_m": 0.0, "center_y_m": 0.0}})()
        return type("R", (), {"success": True, "data": {}})()


def test_bad_axis_rejected():
    res = CalibrateCoarseStep().execute(_Ctx(), {"axis": "z"})
    assert res.success is False and "axis" in (res.error or "")


def test_single_step_count_is_rejected():
    """单个步数算得出一个数，但证不了线性 —— 这正是「两个步数」的用处。"""
    res = CalibrateCoarseStep().execute(_Ctx(), {"axis": "x", "steps": "4"})
    assert res.success is False
    assert "线性" in (res.error or "")


def test_lock_held_refuses_before_moving_the_motor():
    """08-28 撞针的直接原因：动马达时 ScanAt 还握着锁。"""
    ctx = _Ctx(busy=True)
    res = CalibrateCoarseStep().execute(ctx, {"axis": "x", "steps": "2,4"})
    assert res.success is False
    assert res.data["blocked_by_lock"] is True
    assert "TipShape" not in ctx.calls, "锁被占时不该已经开始扎坑"


# ── 判读 ───────────────────────────────────────────────────────────────
class _FullCtx(_Ctx):
    """够 execute 跑完一轮的桩：ScanAt 给路径，其余都成功。"""

    def run(self, name, params=None):
        r = super().run(name, params)
        if name == "ScanAt":
            return type("R", (), {"success": True,
                                  "data": {"scan_path": "synthetic.sxm"}})()
        return r


def test_low_correlation_refuses_instead_of_reporting_a_number(monkeypatch):
    """峰锐度不够 ⇒ **拒答**，不给 nm/步。凑一个位移出来比不给更糟 —— 它会变成刻度。

    ⚠ 这条原来只断言 `_MIN_CORR_SNR > 1.0`，那是在检查常量不是检查行为：
    把整道闸门删掉测试照样绿（变异验证抓到的）。现在喂互不相关的帧，验它真的拒答。
    """
    rng = np.random.default_rng(7)
    monkeypatch.setattr(CalibrateCoarseStep, "_load",
                        staticmethod(lambda path: rng.normal(0, 20.0, (256, 256))))
    res = CalibrateCoarseStep().execute(
        _FullCtx(), {"axis": "x", "steps": "2,4", "make_pit": False})
    assert res.success is True
    assert res.data["verdict"] == "undetermined", "低相关时不该给出 nm/步"
    assert any("refused" in r for r in res.data["runs"])
    assert all(r.get("nm_per_step_x") is None for r in res.data["runs"])


def test_good_correlation_does_report_a_number(monkeypatch):
    """反面：帧之间真有一致平移时必须给得出数 —— 否则上一条会因为「永远拒答」而空过。"""
    base = _img(seed=3)
    seq = iter([base, _img(shift=(20, 0), seed=3), _img(shift=(60, 0), seed=3)])
    monkeypatch.setattr(CalibrateCoarseStep, "_load",
                        staticmethod(lambda path: next(seq)))
    res = CalibrateCoarseStep().execute(
        _FullCtx(), {"axis": "x", "steps": "2,4", "make_pit": False})
    assert res.data["verdict"] in ("ok", "nonlinear")
    assert res.data.get("nm_per_step") is not None


@pytest.mark.parametrize("vals,want", [
    ([100.0, 102.0], "ok"),        # 两个步数一致 ⇒ 线性
    ([100.0, 200.0], "nonlinear"), # 差一倍 ⇒ 黏滑不均，别当刻度
])
def test_linearity_self_check(vals, want):
    spread = (max(vals) - min(vals)) / (sum(vals) / len(vals))
    got = "ok" if spread <= 0.35 else "nonlinear"
    assert got == want


def test_default_steps_are_small_enough_for_a_100nm_per_step_motor():
    """若每步 100 nm，默认步数 × 100 nm 必须落在默认视野之内 ——
    08-28 第一次用 20 步 / 800 nm 什么都没看到，正是因为 2 µm 早跑出视野。"""
    meta = CalibrateCoarseStep().metadata()
    steps = {p.name: p for p in meta.parameters}
    default_steps = [int(t) for t in str(steps["steps"].default).split(",")]
    default_size_nm = float(steps["size_m"].default) * 1e9
    # 位移必须只占视野的一小部分，否则两帧几乎不重叠、相关立不住。
    # 「小于视野」是不够的（800 nm 视野 + 400 nm 位移照样过，而那正是 08-28
    # 看不到位移的那一组）—— 要求 ≤1/3。
    assert max(default_steps) * 100.0 <= default_size_nm / 3.0
