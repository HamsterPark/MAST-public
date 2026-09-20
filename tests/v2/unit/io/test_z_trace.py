"""z(t) 跳变判据 —— 先证明它在纯噪声上不说谎,再证明它抓得住真跳变。

顺序是有意的。一个只测「注入的阶跃被检出」的判据测试可以在误报率 30% 的情况下
全绿:合成信号里那个阶跃当然找得到,问题从来是它在**没有**跳变的时候报不报。
先检验白噪声上的误报，再检验注入阶跃的检出，避免仅凭检出用例判断可靠性。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mast.io.z_trace import (
    DEFAULT_JUMP_K,
    baseline_sigma,
    detect_jumps,
    step_verdict,
)

#: 合成窗口使用 2 kHz × 0.5 s；误报率随样本数变化，不能直接跨窗口沿用结论。
_FS_HZ = 2000.0
_WINDOW_S = 0.5
_N = int(_FS_HZ * _WINDOW_S)


def _times(n, fs=_FS_HZ):
    return [i / fs for i in range(n)]


# ── 1. 纯噪声:不许报 ────────────────────────────────────────────────────────

def test_pure_white_noise_zero_false_jumps_at_default_k():
    """纯高斯白噪声,32 段独立记录,k=8 下一次误报都不许有。"""
    rng = np.random.default_rng(20260801)
    total = 0
    for _ in range(32):
        z = rng.normal(0.0, 7e-12, _N)          # 独立选择的合成噪声幅度
        total += detect_jumps(z.tolist(), _times(_N), k=DEFAULT_JUMP_K)["count"]
    assert total == 0, f"白噪声上报了 {total} 次假跳变"


def test_lower_k_is_why_the_default_is_eight():
    """k=5(旧默认)在同样的纯噪声上会报 —— 记录这个默认值的来历。

    若哪天这条断言变绿(k=5 也零误报了),说明窗口长度或采样率变了,DEFAULT_JUMP_K
    需要重新标定,而不是把这个测试删掉。"""
    rng = np.random.default_rng(20260801)
    total = 0
    for _ in range(32):
        z = rng.normal(0.0, 5e-12, _N)
        total += detect_jumps(z.tolist(), _times(_N), k=5.0)["count"]
    assert total > 0


def test_noise_only_verdict_is_none():
    """没有事件的一段轨迹,前后稳定值之差必须判「没变」。"""
    rng = np.random.default_rng(7)
    z = rng.normal(0.0, 5e-12, _N).tolist()
    out = step_verdict(z, _times(_N), event_start_t=0.1, post_roll_s=0.1)
    assert out["direction"] == "none"


def test_slow_drift_alone_is_not_a_jump():
    """整段线性热漂移(200 pm/s)不该被当成跳变。"""
    rng = np.random.default_rng(11)
    t = _times(_N)
    z = [200e-12 * ti + n for ti, n in zip(t, rng.normal(0.0, 5e-12, _N))]
    assert detect_jumps(z, t, k=DEFAULT_JUMP_K)["count"] == 0


# ── 2. 真信号:必须抓到 ──────────────────────────────────────────────────────

@pytest.mark.parametrize("step_m", [50e-12, 500e-12, 20e-9])
def test_injected_step_is_found(step_m):
    """注入的阶跃(50 pm / 500 pm / 20 nm)在同一个 k 下全都要抓到。"""
    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 5e-12, _N)
    z[_N // 2:] += step_m
    out = detect_jumps(z.tolist(), _times(_N), k=DEFAULT_JUMP_K)
    assert out["count"] >= 1
    assert out["max_abs_delta"] == pytest.approx(step_m, rel=0.3)


def test_step_verdict_up_and_down():
    rng = np.random.default_rng(5)
    t = _times(_N)
    for sign, expect in ((+1.0, "up"), (-1.0, "down")):
        z = rng.normal(0.0, 5e-12, _N)
        z[_N // 2:] += sign * 400e-12
        out = step_verdict(z.tolist(), t, event_start_t=0.2, post_roll_s=0.15)
        assert out["direction"] == expect
        assert out["delta_m"] == pytest.approx(sign * 400e-12, rel=0.1)


def test_transient_spike_between_the_windows_is_ignored():
    """事件中间的瞬态峰(qPlus 音叉起跳那种)不参与判定 —— 只记进 z_min/z_max。

    这是用户口述的判据判据:「这个跳变中间可能也会产生一个峰,这个峰我们同样不
    管他」。前后都回到同一水平 → none,而峰值仍如实报出来。"""
    t = _times(_N)
    z = np.zeros(_N)
    z[_N // 2 - 40:_N // 2 + 40] = 8e-9        # 8 nm 的瞬态尖峰
    out = step_verdict(z.tolist(), t, event_start_t=0.2, post_roll_s=0.15)
    assert out["direction"] == "none"
    assert out["z_max_m"] == pytest.approx(8e-9)


def test_tuning_fork_ringing_does_not_fake_a_jump():
    """qPlus 音叉的零均值振荡不该被读成跳变。

    取中位数已经把零均值成分抵消掉,振荡撑大的 σ 又让容差自动变保守 ——
    两个效果都在,所以不需要为 qPlus 写特例分支。"""
    rng = np.random.default_rng(13)
    t = _times(_N)
    ring = [300e-12 * math.sin(2 * math.pi * 120.0 * ti) for ti in t]
    z = [r + n for r, n in zip(ring, rng.normal(0.0, 5e-12, _N))]
    out = step_verdict(z, t, event_start_t=0.2, post_roll_s=0.15)
    assert out["direction"] == "none"


def test_real_jump_survives_the_ringing():
    """振荡叠一个真的 20 nm 台阶 —— 判据要么抓到,要么就没用。"""
    rng = np.random.default_rng(17)
    t = _times(_N)
    z = []
    for i, ti in enumerate(t):
        v = 300e-12 * math.sin(2 * math.pi * 120.0 * ti) + rng.normal(0.0, 5e-12)
        z.append(v + (20e-9 if i >= _N // 2 else 0.0))
    out = step_verdict(z, t, event_start_t=0.2, post_roll_s=0.15)
    assert out["direction"] == "up"
    assert out["delta_m"] == pytest.approx(20e-9, rel=0.1)


# ── 3. σ 估计与退化保护 ─────────────────────────────────────────────────────

def test_baseline_sigma_matches_std_on_clean_noise():
    rng = np.random.default_rng(23)
    xs = rng.normal(0.0, 5e-12, 4000).tolist()
    assert baseline_sigma(xs) == pytest.approx(5e-12, rel=0.15)


def test_baseline_sigma_ignores_drift_where_std_would_not():
    """带斜坡的基线:MAD-of-diffs 只看噪声,标准差会把漂移算进去。"""
    rng = np.random.default_rng(29)
    n = 2000
    xs = [50e-12 * i / n + v for i, v in enumerate(rng.normal(0.0, 5e-12, n))]
    plain_std = float(np.std(xs))
    assert baseline_sigma(xs) == pytest.approx(5e-12, rel=0.2)
    assert plain_std > 3.0 * baseline_sigma(xs)


def test_degenerate_window_falls_back_to_std():
    """样本太少 / MAD 恰好为 0 时回退到标准差。

    没有这层保护,σ 会精确地等于 0,容差跟着塌成 0,之后任何一点差异都判「变了」
    —— 而恰恰是量化到同一台阶的安静基线最容易触发它。"""
    assert baseline_sigma([1.0, 1.001]) == pytest.approx(0.0005)
    assert baseline_sigma([2.0, 2.0, 2.0, 2.0]) == 0.0
    assert baseline_sigma([1.0]) == 0.0


# ── 4. 数据不足要说不知道,不能猜 ────────────────────────────────────────────

@pytest.mark.parametrize("z,t,ev", [
    ([1.0, 1.0, 1.0], [0.0, 0.1, 0.2], 0.05),      # 少于 4 个采样
    ([1.0] * 10, [i * 0.1 for i in range(10)], None),   # 不知道事件何时发生
    ([1.0] * 10, [i * 0.1 for i in range(10)], -1.0),   # 事件早于整段采集
])
def test_insufficient_data_is_reported_not_guessed(z, t, ev):
    assert step_verdict(z, t, ev, post_roll_s=0.1)["direction"] == "insufficient_data"


# ── 一次超长往返不许劫持整个判定（2026-08-05, KNOWN_ISSUES）──────────────
#
# 采集循环在调用**返回之后**才打时间戳,所以一次异常长的 TCP 往返会造出一个
# 时间戳远在窗口之外的样本。后窗锚在 ``times[-1]`` 上,于是被这一个离群值挟持,
# 把全部真实样本挡在窗外 → 手里握着几百个好样本却报 ``insufficient_data``。
# 真机上那句话的意思是「打了一发脉冲,而我说不出针尖发生了什么」。
#
# 这几条**不需要时钟** —— 直接把带滞后尾巴的序列喂给判据。


def _capture_with_step(n=500, span=0.3, event_t=0.1, step_m=18e-9):
    """一次健康的采集:事件之前平,之后抬 step_m。"""
    times = [i * (span / n) for i in range(n)]
    k = sum(1 for t in times if t < event_t)
    return [0.0] * k + [step_m] * (n - k), times


def test_one_stalled_sample_does_not_destroy_the_verdict():
    """在合成阶跃序列尾部追加迟到样本，判定仍须保持正确。"""
    z, t = _capture_with_step()
    out = step_verdict(z + [18e-9], t + [0.45], 0.1,
                       post_roll_s=0.08, tol_abs_m=0.5e-9)
    assert out["direction"] == "up"          # 修前:insufficient_data
    assert out["delta_m"] == pytest.approx(18e-9, rel=0.05)


def test_the_stall_is_reported_not_silently_absorbed():
    """扩窗是对**判据**的修复,不是对卡顿的修复。那次异常长的往返本身仍然是一个
    该被上报的事实(本项目有 comms_health 与成文的 Nanonis TCP 脆弱史)——
    判定给出正确答案的**同时**必须说清它是在什么条件下给出的。"""
    z, t = _capture_with_step()
    out = step_verdict(z + [18e-9], t + [0.45], 0.1,
                       post_roll_s=0.08, tol_abs_m=0.5e-9)
    assert out["post_window_starved"] is True
    assert out["max_gap_s"] == pytest.approx(0.45 - t[-1], abs=1e-9)


def test_max_gap_is_computed_even_when_nothing_went_wrong():
    """正常合成采集也要报告最大采样间隔，不能只在退化路径计算。"""
    z, t = _capture_with_step()
    out = step_verdict(z, t, 0.1, post_roll_s=0.08, tol_abs_m=0.5e-9)
    assert out["direction"] == "up"
    assert out["post_window_starved"] is False
    assert out["max_gap_s"] == pytest.approx(0.3 / 500, rel=0.05)


def test_a_healthy_capture_is_judged_exactly_as_before():
    """对照组:没有卡顿时,窗口选择必须一个样本都不变。"""
    z, t = _capture_with_step()
    out = step_verdict(z, t, 0.1, post_roll_s=0.08, tol_abs_m=0.5e-9)
    assert out["n_post"] == sum(1 for x in t if x >= t[-1] - 0.08)


def test_the_pre_window_is_deliberately_NOT_given_the_same_fallback():
    """**钉住一个不对称**,因为注释拦不住:下一个人会自己重新想到「对称一下」
    这个主意(正因为它看起来显然对),而且不会先读注释。

    ``pre`` 由 ``t < event_start_t`` 选出 —— 它**不靠尾锚**,没有被离群值挟持的
    失败模式。前窗点数少,就是基线**真的**不够,那时 ``insufficient_data`` 是实话,
    不是需要被兜底救回来的退化。给它加同样的扩窗,等于把一句实话改成猜测。
    """
    # 事件几乎在采集一开始就发生 → 前面只有 1 个样本,而后面样本充足。
    times = [i * 0.0005 for i in range(400)]
    z = [0.0] + [25e-9] * 399
    out = step_verdict(z, times, 0.0004, post_roll_s=0.08, tol_abs_m=0.5e-9)
    assert out["direction"] == "insufficient_data", (
        "前窗不足被兜底救了回来 —— 那不是这个兜底该管的事")


def test_a_genuinely_short_capture_still_says_insufficient_data():
    """``insufficient_data`` 严格只表示「样本真的太少」。扩窗**不许**把这句实话
    也变成一个猜测 —— 总点数不足 4 的闸门在扩窗之前,原样保留。"""
    out = step_verdict([1.0, 1.0, 1.0], [0.0, 0.1, 0.2], 0.05, post_roll_s=0.1)
    assert out["direction"] == "insufficient_data"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
