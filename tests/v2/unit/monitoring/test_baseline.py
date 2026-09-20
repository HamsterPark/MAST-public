# -*- coding: utf-8 -*-
"""电流噪声基线：模型、存储、判据接线。

本文件通过明确的模型公式生成独立合成数据，验证参数恢复、误报边界和存储行为；
不包含任何仪器的实测表征数据或拟合结果。
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from mast.monitoring import baseline as B
from mast.monitoring.alerts import AlertEngine
from mast.monitoring.features import detrended_rms
from mast.monitoring.store import (
    CurrentMonitorStore,
    _BPOINT_COLS,
    _baseline_point_column_spec,
)
from mast.monitoring.thresholds import MonitorThresholds

#: Independently constructed fixture, not a renamed measurement table.
SYNTHETIC_AMP_A_PER_RTHZ = 12e-15
SYNTHETIC_MULTIPLICATIVE = 200e-6
SYNTHETIC = tuple(
    (i, math.sqrt((0.4e-12) ** 2 + (0.006 * i) ** 2),
     SYNTHETIC_AMP_A_PER_RTHZ ** 2 + 2 * B.Q_E * i
     + (SYNTHETIC_MULTIPLICATIVE * i) ** 2)
    for i in (25e-12, 75e-12, 150e-12, 300e-12, 800e-12, 1500e-12)
)


@pytest.fixture()
def store(tmp_path):
    return CurrentMonitorStore(str(tmp_path / "m.sqlite"), data_dir=str(tmp_path))


# ── 口径对账 ────────────────────────────────────────────────────────────────

def test_width_sigma_is_bit_for_bit_features_detrended_rms():
    """基线的 sigma 必须与实时判据读的那一列是同一个数。

    这不是「差不多」，是**逐位相同**。基线的全部价值在于它可以和 features 表里
    那一列直接比较；两边各算各的，比出来的是两套实现的差，不是仪器的状态。
    """
    rng = np.random.default_rng(35)
    for _ in range(20):
        y = 2e-10 + rng.normal(0, 1.7e-12, 2001) + np.linspace(0, 3e-13, 2001)
        assert B.width_stats(y)["sigma_a"] == detrended_rms(y, 2000.0)["rms_detrended_a"]


def test_width_stats_recovers_a_gaussian():
    """三种宽度估计在高斯上必须一致 —— 它们的比值就是尾巴诊断。"""
    rng = np.random.default_rng(7)
    w = B.width_stats(rng.normal(0, 1.0, 200_000))
    assert w["sigma_a"] == pytest.approx(1.0, rel=0.02)
    assert w["sigma_iqr_a"] == pytest.approx(1.0, rel=0.02)
    assert w["sigma_mad_a"] == pytest.approx(1.0, rel=0.02)
    # 高斯的 FWHM/sigma = 2*sqrt(2 ln2)
    assert w["fwhm_over_sigma"] == pytest.approx(2.3548, rel=0.05)


def test_width_stats_sees_a_flat_top_a_single_sigma_would_hide():
    """纯正弦与高斯可以有相同的 sigma，形状完全不同。"""
    t = np.linspace(0, 1, 20000)
    sine = np.sin(2 * np.pi * 40 * t) * math.sqrt(2)     # sigma = 1
    g = np.random.default_rng(3).normal(0, 1.0, 20000)
    ws, wg = B.width_stats(sine), B.width_stats(g)
    assert ws["sigma_a"] == pytest.approx(wg["sigma_a"], rel=0.05)
    # 同样的 sigma，峰度天差地别：正弦 -1.5，高斯 0
    assert ws["kurtosis"] < -1.0 < wg["kurtosis"] + 0.5


# ── 模型 ────────────────────────────────────────────────────────────────────

def test_sigma_model_recovers_the_synthetic_sweep():
    m = B.fit_sigma_model([i for i, _, _ in SYNTHETIC], [s for _, s, _ in SYNTHETIC])
    assert m is not None and m.r2 > 0.99
    for i, s, _ in SYNTHETIC:
        assert m.expected(i) == pytest.approx(s, rel=0.10)


def test_white_model_recovers_the_declared_synthetic_components():
    """恢复生成公式中独立给定的加性、乘性分量及其交叉点。"""
    m = B.fit_white_model([i for i, _, _ in SYNTHETIC], [w for _, _, w in SYNTHETIC])
    assert m is not None
    assert m.amp_a_per_rthz == pytest.approx(SYNTHETIC_AMP_A_PER_RTHZ, rel=0.05)
    assert math.sqrt(m.b) == pytest.approx(SYNTHETIC_MULTIPLICATIVE, rel=0.05)
    assert m.crossover_a == pytest.approx(
        SYNTHETIC_AMP_A_PER_RTHZ / SYNTHETIC_MULTIPLICATIVE, rel=0.10)


def test_white_model_holds_shot_noise_fixed_rather_than_fitting_it():
    """2eI 是已知量。把它也拟合掉会花掉一个自由度去解释物理已经定死的东西,
    而且会让「这台机器是不是散粒噪声极限」这个问题失去答案。"""
    I = np.array([i for i, _, _ in SYNTHETIC])
    m = B.fit_white_model(I, [w for _, _, w in SYNTHETIC])
    shot = 2 * B.Q_E * I
    modelled = np.array([m.expected(i) for i in I])
    # 合成模型仍须保持固定的散粒噪声项，而不是另拟合其系数。
    assert np.allclose(modelled, m.amp_a2_per_hz + shot + m.b * I ** 2, rtol=1e-9)


def test_a_linear_space_fit_would_have_got_the_additive_term_wrong():
    """跨量级合成数据的朴素线性拟合会丢失加性分量；对数拟合应恢复生成值。"""
    I = np.array([i for i, _, _ in SYNTHETIC])
    W = np.array([w for _, _, w in SYNTHETIC])
    X = np.vstack([np.ones_like(I), I ** 2]).T
    coef, *_ = np.linalg.lstsq(X, W, rcond=None)
    lin_amp = math.sqrt(coef[0]) if coef[0] > 0 else float("inf")
    log_amp = B.fit_white_model(I, W).amp_a_per_rthz
    assert log_amp == pytest.approx(SYNTHETIC_AMP_A_PER_RTHZ, rel=0.05)
    assert lin_amp > 5 * log_amp, (
        "线性拟合这次没有失准 —— 这条测试的前提要重看，别直接放宽断言")


def test_model_refuses_to_extrapolate_far_outside_its_range():
    m = B.fit_sigma_model([i for i, _, _ in SYNTHETIC], [s for _, s, _ in SYNTHETIC])
    v = B.compare(0.3e-12, 3e-12, m)          # 3 pA，在合成标定范围之外
    assert not v.judged and "不外推" in v.reason
    assert v.ratio is None, "判不了的时候不能给出一个看起来正常的比值"


def test_compare_states_a_reason_for_every_refusal():
    m = B.fit_sigma_model([i for i, _, _ in SYNTHETIC], [s for _, s, _ in SYNTHETIC])
    for sigma, cur in ((None, 2e-10), (1.6e-12, None), (1.6e-12, 0.0),
                       (float("nan"), 2e-10)):
        v = B.compare(sigma, cur, m)
        assert not v.judged and v.reason, "拒绝判断时必须说得出为什么"
    assert not B.compare(1.6e-12, 2e-10, None).judged


# ── 机制归因 ────────────────────────────────────────────────────────────────

def test_mechanism_needs_both_exponents():
    """距离调制与电压耦合的 a_I 都是 +1 —— 只有 a_V 分得开它们。

    这是「为什么要扫两组」的可执行版本。
    """
    assert B.classify_mechanism(1.0, 0.0)[0] == "distance_modulation"
    assert B.classify_mechanism(1.0, -1.0)[0] == "voltage_pickup"
    assert B.classify_mechanism(0.0, 0.0)[0] == "additive_preamp"
    # 独立选取的合成扰动指数，验证机制分类在理想点附近仍成立。
    assert B.classify_mechanism(0.97, 0.04)[0] == "distance_modulation"
    assert B.classify_mechanism(1.06, -0.75)[0] == "voltage_pickup"


def test_mechanism_declines_to_label_what_matches_nothing():
    """远离已知机制的指数不得给标签；位于机制之间时要保留距离作为歧义证据。"""
    assert B.classify_mechanism(2.0, 1.0)[0] is None      # 超线性且随偏压上升
    assert B.classify_mechanism(float("nan"), 0.0)[0] is None
    # 被吸附的中间地带：标签给了，但距离大到足以自我否认
    mech, dist = B.classify_mechanism(0.5, -0.5)
    assert mech is not None and dist >= 0.5


def test_scaling_exponent_on_a_clean_power_law():
    x = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    a, r2 = B.scaling_exponent(x, 3.0 * x ** 1.0)
    assert a == pytest.approx(1.0, abs=1e-9) and r2 > 0.999


# ── 条件快照 ────────────────────────────────────────────────────────────────

def test_condition_diff_reports_unknown_as_unknown_not_as_match():
    """任一侧未知就报未知。把 None==None 当成「相同」会让一份完全没记录条件的
    基线看起来「条件完全匹配」。"""
    assert B.condition_diff({"tip_id": "A"}, {"tip_id": "A"}) == []
    assert B.condition_diff({"tip_id": "A"}, {"tip_id": "B"})
    assert B.condition_diff({"tip_id": "A"}, {})          # 一侧缺
    assert B.condition_diff({}, {}) == []                  # 两侧都没有 → 不报


# ── 存储 ────────────────────────────────────────────────────────────────────

def test_migration_spec_covers_every_column_the_writer_binds():
    """迁移清单必须从写入方的列常量生成。

    手抄的清单会漂开，而漂开的表现是「插入静默失败、表悄悄不再增长」——
    本仓已为 hand-maintained 清单付过三次学费。
    """
    spec = {name for name, _decl in _baseline_point_column_spec()}
    assert set(_BPOINT_COLS) <= spec
    for extra in ("ordinal", "tag", "psd_path", "hist_path", "extra_json"):
        assert extra in spec


def test_baseline_roundtrip(store):
    bid = store.create_baseline(label="乙", conditions={"tip_id": "T1"}, fs_hz=2000.0)
    assert bid
    pid = store.add_baseline_point(
        bid, tag="setpoint", ordinal=0,
        metrics={"i_measured_a": 2e-10, "sigma_a": 1.6e-12, "n_segments": 70,
                 "seg_id_lo": 10, "seg_id_hi": 79, "white_a2hz": 8e-28},
        psd=(np.arange(5.0), np.ones(5)), hist=(np.arange(4.0), np.ones(4)))
    assert pid
    assert store.finish_baseline(bid, sigma_model={
        "c_a2": 7e-26, "d": 6.6e-5, "i_lo_a": 1.7e-11, "i_hi_a": 1e-9,
        "r2": 0.999, "n_points": 6})
    d = store.baseline(bid)
    assert d["n_points"] == 1 and d["status"] == "complete"
    assert d["sigma_model"]["d"] == pytest.approx(6.6e-5)
    assert d["points"][0]["i_measured_a"] == pytest.approx(2e-10)
    c = store.baseline_point_curve(pid, "psd")
    assert c["y"] == [1.0] * 5


def test_only_a_complete_baseline_can_be_activated(store):
    """一份没跑完的表征没有 sigma 曲线;激活它等于把分母换成 None ——
    与「没有基线」是同一件事,却会在界面上显示成「有基线」。"""
    bid = store.create_baseline(label="半截")
    assert store.activate_baseline(bid) is False
    assert store.active_baseline() is None
    store.finish_baseline(bid, sigma_model={"c_a2": 1e-26, "d": 1e-5,
                                            "i_lo_a": 1e-11, "i_hi_a": 1e-9})
    assert store.activate_baseline(bid) is True
    assert store.active_baseline()["id"] == bid


def test_active_baseline_cannot_be_deleted(store):
    """删掉正在被判据使用的分母会让判据在下一段静默换回固定阈值。"""
    bid = store.create_baseline()
    store.finish_baseline(bid, sigma_model={"c_a2": 1e-26, "d": 1e-5,
                                            "i_lo_a": 1e-11, "i_hi_a": 1e-9})
    store.activate_baseline(bid)
    assert store.delete_baseline(bid) is False
    store.activate_baseline(None)
    assert store.delete_baseline(bid) is True


def test_delete_removes_the_curve_files(store, tmp_path):
    bid = store.create_baseline()
    store.add_baseline_point(bid, ordinal=0, metrics={"sigma_a": 1e-12},
                             psd=(np.arange(3.0), np.ones(3)))
    files = list((tmp_path / "baseline").glob("*.npz"))
    assert files
    store.delete_baseline(bid)
    assert not list((tmp_path / "baseline").glob("*.npz"))


def test_non_finite_metrics_are_stored_as_null_not_as_a_number(store):
    """NaN 落进 REAL 列会变成一个能参与比较的值。"""
    bid = store.create_baseline()
    pid = store.add_baseline_point(bid, ordinal=0, metrics={
        "sigma_a": float("nan"), "white_a2hz": float("inf"), "i_measured_a": 2e-10})
    row = store.baseline(bid)["points"][0]
    assert row["sigma_a"] is None and row["white_a2hz"] is None
    assert row["i_measured_a"] == pytest.approx(2e-10)


# ── 判据接线（最关键的一组） ────────────────────────────────────────────────

def _engine(th=None):
    th = th or MonitorThresholds()
    return AlertEngine(lambda: th), th


def _model():
    return B.fit_sigma_model([i for i, _, _ in SYNTHETIC], [s for _, s, _ in SYNTHETIC])


def _feats(rms_a, i_a):
    return {"rms_detrended_a": rms_a, "mean_a": i_a}


@pytest.mark.parametrize("rms_pa,i_pa", [(2.0, 300.0), (9.0, 1500.0),
                                         (25.0, 300.0), (0.4, 25.0), (60.0, 75.0)])
def test_without_a_baseline_the_verdict_is_unchanged(rms_pa, i_pa):
    """**硬约束**：没有基线的机器上，这次改动必须一个字节都不改变行为。

    否则这个功能会在每一台还没做过表征的机器上变成静默的判据漂移。
    """
    eng, th = _engine()
    v = eng.evaluate(_feats(rms_pa * 1e-12, i_pa * 1e-12))
    assert ("rms_high" in v.rules) == (rms_pa * 1e-12 > th.cm_rms_warn_a)


def test_with_a_baseline_the_same_ratio_holds_across_60x_in_current():
    """相同相对噪声偏离，在合成电流范围的各点应产生相同判决。"""
    eng, _ = _engine()
    m = _model()
    for i, s, _ in SYNTHETIC:
        v = eng.evaluate(_feats(s, i), baseline=m)
        assert "rms_high" not in v.rules, f"{i:.3g} A 上的健康值被判成了告警"
        assert v.detail["rms_ratio"] == pytest.approx(1.0, abs=0.12)
        # 同一个坏法（3.2 倍）在每个电流上都必须被抓到
        bad = eng.evaluate(_feats(s * 3.2, i), baseline=m)
        assert "rms_high" in bad.rules, f"{i:.3g} A 上坏了 3.2 倍却没报"


def test_the_fixed_threshold_misses_what_the_baseline_catches():
    """合成低电流点的噪声相对增加可低于绝对阈值，但仍应被基线判据捕获。"""
    eng, th = _engine()
    m = _model()
    cur, healthy_sigma, _ = SYNTHETIC[0]
    rms = healthy_sigma * 3.2
    assert rms < th.cm_rms_warn_a, "前提变了：这个值本该低于固定阈值"
    assert "rms_high" not in eng.evaluate(_feats(rms, cur)).rules
    assert "rms_high" in eng.evaluate(_feats(rms, cur), baseline=m).rules


def test_the_fixed_threshold_would_cry_wolf_where_the_baseline_stays_quiet():
    """合成高电流点的健康噪声可超过偏紧的绝对阈值，但基线判据应保持安静。"""
    eng, _ = _engine(MonitorThresholds(cm_rms_warn_a=4e-12))
    m = _model()
    cur, rms, _ = SYNTHETIC[-1]
    assert "rms_high" in eng.evaluate(_feats(rms, cur)).rules      # 固定阈值：误报
    assert "rms_high" not in eng.evaluate(_feats(rms, cur), baseline=m).rules


def test_scanning_falls_back_to_the_fixed_threshold():
    """扫描时 rms_detrended 携带形貌的交流成分，比值必然超标。

    ``ctx_scanning`` 从 SQLite 回来是 0/1 而不是 True/False，所以这里传 1 ——
    ``1 is True`` 在 Python 里是 False，这个坑本仓踩过（commission._tri_bool）。
    """
    eng, _ = _engine()
    m = _model()
    cur, healthy_sigma, _ = SYNTHETIC[3]
    f = _feats(healthy_sigma * 3.2, cur)  # 相对阈值以上、绝对阈值以下
    assert "rms_high" in eng.evaluate(f, baseline=m).rules
    v = eng.evaluate(f, ctx={"ctx_scanning": 1}, baseline=m)
    assert "rms_high" not in v.rules and "rms_ratio" not in v.detail


def test_unjudgeable_baseline_falls_back_and_says_why():
    """「判不了」既不能变成「没问题」，也不能让这一段完全失去判据。"""
    eng, _ = _engine()
    v = eng.evaluate(_feats(30e-12, 3e-12), baseline=_model())
    assert "rms_high" in v.rules                     # 30 pA > 20 pA 固定阈值
    assert "不外推" in v.detail["rms_baseline_unjudged"]
    assert "rms_ratio" not in v.detail


# ── Z 通道与 Z-电流联合量 ───────────────────────────────────────────────────

def _synthetic_junction(kappa_per_m=12e9, i_mean=2e-10, n_pairs=20,
                        fs=2000.0, n=2001, seed=11):
    """合成一个隧道结：Z 上有振动，电流 = 2*kappa*I*z + 独立的电学噪声。

    合成的**物理必须成立**，否则测出来的是合成器的性质 —— 本包为此付过学费
    （project_current_monitor 的四个判据）。这里成立的部分是 dI = 2*kappa*I*dz，
    以及「电流侧另有一份与 Z 无关的噪声」，后者正是相干性要能识别出来的东西。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    zs, iss = [], []
    for _ in range(n_pairs):
        z = (0.3e-12 * math.sqrt(2)
             * np.sin(2 * np.pi * 625.0 * t + rng.uniform(0, 2 * np.pi))
             + rng.normal(0, 5e-14, n))
        i = 2 * kappa_per_m * i_mean * z + rng.normal(0, 3e-13, n)
        zs.append(z)
        iss.append(i)
    return zs, iss, fs, i_mean


def test_coherence_refuses_a_single_segment():
    """**单段的相干性在数学上恒等于 1** —— 那是代数恒等式，不是测量结果，
    而且它长得跟「完美耦合」一模一样。"""
    z, i, fs, im = _synthetic_junction(n_pairs=1)
    r = B.z_current_coupling(z, i, fs, im)
    assert not r["available"]
    r4 = B.z_current_coupling(*_synthetic_junction(n_pairs=4)[:3],
                              i_mean_a=2e-10)
    assert not r4["available"] and "偏向 1" in r4["detail"]


def test_coherence_separates_the_shaken_line_from_the_electrical_floor():
    z, i, fs, im = _synthetic_junction()
    r = B.z_current_coupling(z, i, fs, im)
    assert r["available"] and r["n_pairs"] >= 8
    f = np.asarray(r["freqs_hz"])
    g = np.asarray(r["coherence"])
    at_line = g[int(np.argmin(np.abs(f - 625.0)))]
    assert at_line > 0.95, "注入振动的那条线上相干性应接近 1"
    assert 0.0 <= g.min() and g.max() <= 1.0, "相干性必须落在 [0,1]"


@pytest.mark.parametrize("kappa_per_nm", [10.0, 12.0, 14.0])
def test_kappa_falls_out_of_the_noise_without_an_iz_curve(kappa_per_nm):
    """从本来就要采的 burst 里免费得到 kappa —— 不必跑 I-z 谱。"""
    z, i, fs, im = _synthetic_junction(kappa_per_m=kappa_per_nm * 1e9)
    r = B.z_current_coupling(z, i, fs, im)
    assert r["kappa_per_nm"] == pytest.approx(kappa_per_nm, rel=0.05)
    assert 1.0 < r["apparent_barrier_ev"] < 20.0


def test_kappa_is_refused_when_the_channels_are_unrelated():
    """电流噪声与 Z 无关时不能给传递函数 —— 那个比值算得出来，但没有意义。"""
    rng = np.random.default_rng(5)
    z = [rng.normal(0, 5e-14, 2001) for _ in range(20)]
    i = [rng.normal(0, 3e-13, 2001) for _ in range(20)]
    r = B.z_current_coupling(z, i, 2000.0, 2e-10)
    assert r["kappa_per_m"] is None
    assert r["coherent_fraction"] < 0.05


def test_kappa_band_is_stricter_than_the_coherent_band():
    """电流侧独立噪声会抬高传递幅值，因此 kappa 使用比普通相干判据更严格的门槛。"""
    assert B._KAPPA_MIN_COHERENCE > 0.5


def test_z_stats_reports_the_first_difference_not_the_span():
    """白噪声不是 Z 的正确零假设：压电蠕变与热漂是随机游走，极差按 sqrt(N)
    增长，任何固定阈值终将被纯漂移触发。一阶差分才是 iid 的那个量。"""
    rng = np.random.default_rng(9)
    walk = [np.cumsum(rng.normal(0, 1e-13, 2001)) for _ in range(5)]
    out, freqs, psd = B.z_stats(walk, 2000.0)
    assert out["available"] and out["step_rms_m"] == pytest.approx(1e-13, rel=0.15)
    assert freqs is not None and len(freqs) == len(psd)


def test_z_stats_is_honest_about_having_nothing():
    out, f, p = B.z_stats([], 2000.0)
    assert not out["available"] and f is None and p is None


# ── 跨点汇总 ────────────────────────────────────────────────────────────────

def _sweep_points():
    """Synthetic sweeps: both lines scale with I; only the pickup line scales as 1/|V|."""
    pts = [{"sweep": "setpoint", "i_measured_a": i, "sigma_a": s,
            "white_a2hz": w, "bias_v": 0.12,
            "lines": {"l_example_800hz": {"a_per_rthz": 2e-3 * i},
                      "l_50hz": {"a_per_rthz": 6e-4 * i}}}
           for i, s, w in SYNTHETIC]
    reference_i, reference_sigma, _ = SYNTHETIC[3]
    pts += [{"sweep": "bias", "i_measured_a": reference_i, "sigma_a": reference_sigma,
             "bias_v": v, "lines": {"l_example_800hz": {"a_per_rthz": 500e-15},
                                    "l_50hz": {"a_per_rthz": 9e-15 / abs(v)}}}
            for v in (0.06, -0.06, 0.25, -0.25, 0.60, -0.60)]
    pts.append({"sweep": "repeat", "i_measured_a": reference_i,
                "sigma_a": reference_sigma * 1.03, "bias_v": 0.12})
    return pts


def test_build_models_is_the_single_implementation_both_callers_use():
    m = B.build_models(_sweep_points())
    assert m["sigma_model"] is not None and m["white_model"] is not None
    assert m["n_setpoint"] == 6 and m["n_bias"] == 6
    # 合成示例线：正比于电流、不随偏压 → 距离调制
    assert m["lines"]["l_example_800hz"]["mechanism"] == "distance_modulation"
    # 50 线：构造成 1/|V| → 电压耦合
    assert m["lines"]["l_50hz"]["a_v"] == pytest.approx(-1.0, abs=0.05)
    assert m["lines"]["l_50hz"]["mechanism"] == "voltage_pickup"


def test_repeat_point_is_what_gives_a_repeatability_scale():
    """没有重复点就没有重复性 —— 而没有重复性，「差了 3%」无从判断。"""
    pts = _sweep_points()
    assert B.build_models(pts)["repeatability"]["n"] >= 2
    without = [p for p in pts if p.get("sweep") != "repeat"]
    rep = B.build_models(without)["repeatability"] or {}
    # 没有重复点时 rep 里只可能剩极性那一块，绝不该有 n —— 有 n 就意味着
    # 某处把「两个不同工作点」当成了同一个点的两次测量。
    assert rep.get("n") is None or rep["n"] < 2


def test_a_setpoint_only_sweep_gives_a_curve_but_no_mechanism():
    """只扫一组仍然得到可用曲线，但机制留空 —— a_I 单独分不开振动与电压拾取。"""
    only = [p for p in _sweep_points() if p["sweep"] == "setpoint"]
    m = B.build_models(only)
    assert m["sigma_model"] is not None
    assert m["lines"]["l_example_800hz"]["a_i"] is not None
    assert m["lines"]["l_example_800hz"]["a_v"] is None
    assert m["lines"]["l_example_800hz"]["mechanism"] is None


def test_polarity_pairs_only_matched_working_points():
    from mast.skills.builtins.characterise_noise import CharacteriseCurrentNoise
    pol = CharacteriseCurrentNoise._polarity_check(_sweep_points())
    assert pol["n_pairs"] == 3
    assert pol["ratio_mean"] == pytest.approx(1.0, abs=0.01)
    # |V| 配不上就不配对：拿两个不同工作点相除得到的比值看起来完全正常
    odd = [{"sweep": "bias", "i_measured_a": 2e-10, "sigma_a": 1.6e-12, "bias_v": 0.08},
           {"sweep": "bias", "i_measured_a": 2e-10, "sigma_a": 1.7e-12, "bias_v": -0.20}]
    assert CharacteriseCurrentNoise._polarity_check(odd) is None


def test_polarity_pairs_the_nearest_in_time_not_everything():
    """不同间隔可能混入不同程度的漂移；合成序列验证只配对最近点并报告间隔。"""
    far_pos = {"sweep": "bias", "i_measured_a": 2e-10, "sigma_a": 1.90e-12,
               "bias_v": 0.20}                      # 一小时前的那次
    near_pos = {"sweep": "bias", "i_measured_a": 2e-10, "sigma_a": 1.70e-12,
                "bias_v": 0.20}                     # 与负点相邻
    neg = {"sweep": "bias", "i_measured_a": -2e-10, "sigma_a": 1.60e-12,
           "bias_v": -0.20}
    pol = B.polarity_check([far_pos, {"sweep": "setpoint", "i_measured_a": 1e-10,
                                      "sigma_a": 1e-12}, near_pos, neg])
    assert pol["n_pairs"] == 1
    assert pol["pairs"][0]["sigma_pos_a"] == pytest.approx(1.70e-12)
    assert pol["pairs"][0]["points_apart"] == 1


def test_polarity_significance_is_computed_not_left_to_the_reader():
    """显著性必须除以配对平均值的标准误：rsd·√2/√n。合成重复组提供散布尺度。"""
    pts = _sweep_points()
    # 造出一个可比的重复组，让 relative_sd 有值
    reference_i, reference_sigma, _ = SYNTHETIC[3]
    pts += [{"sweep": "repeat", "i_measured_a": reference_i,
             "sigma_a": reference_sigma * scale, "bias_v": 0.12} for scale in (0.98, 1.05)]
    m = B.build_models(pts)
    pol = m["repeatability"]["polarity"]
    rsd = m["repeatability"]["relative_sd"]
    assert pol["std_error"] == pytest.approx(rsd * math.sqrt(2) / math.sqrt(pol["n_pairs"]))
    assert pol["sigma"] == pytest.approx(abs(1 - pol["ratio_mean"]) / pol["std_error"])
    assert isinstance(pol["significant"], bool)
    # 保守性必须写在数据里，否则读的人不知道这个 sigma 是下界还是上界
    assert "保守" in pol["scale_note"]


def test_every_stored_baseline_column_survives_the_api_schema():
    """写入 store 的每个基线字段都应被 API schema 保留，避免静默丢列。"""
    from mast.api.schemas_monitoring import BaselinePoint
    from mast.monitoring.store import _BPOINT_COLS

    declared = set(BaselinePoint.model_fields)
    missing = [c for c in _BPOINT_COLS if c not in declared]
    assert not missing, (
        "这些列 store 会写、API 却读不到（加 store 列时必须同时加 BaselinePoint）："
        f"{missing}")
    # 反方向：schema 声明的测量字段必须真的有列，否则那是个永远为 null 的承诺
    ignore = {"id", "ordinal", "tag", "ts", "extra_json"}
    orphan = [f for f in declared - ignore
              if f not in set(_BPOINT_COLS)]
    assert not orphan, f"schema 声明了 store 没有的字段：{orphan}"


# ── polarity 必须落库并读得回来（2026-08-19）──────────────────────────────

def test_polarity_survives_finish_and_read_back(tmp_path):
    """极性结果必须能落库并从活动基线读回；仅验证写入无异常不足以证明没有丢字段。"""
    from mast.monitoring.store import CurrentMonitorStore

    store = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    bid = store.create_baseline(label="t", note="", conditions={})
    assert bid

    pol = {"pairs": [{"v": 0.6, "rsd": 0.2, "points_apart": 1}],
           "asymmetric": True, "significance": 3.2}
    ok = store.finish_baseline(int(bid), status="complete",
                               sigma_model={"c_a2": 1.0, "d": 0.0},
                               white_model=None, lines=None,
                               repeatability=None, polarity=pol)
    assert ok

    store.activate_baseline(int(bid))
    row = store.active_baseline()
    assert row is not None
    got = row.get("polarity")
    if isinstance(got, str):
        import json as _json
        got = _json.loads(got)
    assert got is not None, "polarity 没有按预期持久化"
    assert got.get("asymmetric") is True
    assert got["pairs"][0]["rsd"] == 0.2


# JSON 列的写入、读取、反序列化与 API schema 必须同步。

def test_every_json_column_is_deserialised_on_read(tmp_path):
    """写进去的每一个 JSON 列，读回来必须是 **dict 而不是字符串**。

    ═══════════════════════════════════════════════════════════════════════
    加一个 JSON 列是**四处**动作，漏一处整个端点就降级
    ═══════════════════════════════════════════════════════════════════════

    2026-08-20：加 `polarity` / `bias_magnitude` 时改了建表、写入、service、
    API schema —— **唯独漏了 `_baseline_row` 里的反序列化白名单**。于是它们
    以 JSON 字符串的形态撞上 `BaselineRow` 的 dict 声明::

        2 validation errors for BaselineRow
        polarity  Input should be a valid dictionary, input_type=str

    后果不是"这两个字段是 null"，而是 **`/monitoring/baselines` 整个端点
    降级成 `baselines: []`** —— 所有基线都读不出来了，而写入端一切正常、
    数据也确实在库里。一个字段的疏漏让一整个列表消失。

    这条测试遍历 `_BASELINE_JSON_COLS`，所以下次再加列时**只要忘了反序列化
    就会当场变红**，不必等它在真机上把端点打掉。
    """
    import json as _json

    from mast.monitoring.store import _BASELINE_JSON_COLS, CurrentMonitorStore

    store = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    bid = store.create_baseline(label="t", note="", conditions={"tip_id": "x"})
    assert bid
    store.finish_baseline(
        int(bid), status="complete",
        sigma_model={"c_a2": 1.0}, white_model={"amp_a2_per_hz": 2.0},
        lines={"l_50hz": {"f_peak_hz": 50.0}},
        repeatability={"relative_sd": 0.03},
        polarity={"n_pairs": 3, "ratio_mean": 0.49},
        bias_magnitude={"positive": {"slope_log10": 0.12}})

    rows = store.baselines(limit=5)
    assert rows, "写完之后应当读得到"
    row = rows[0]
    for col in _BASELINE_JSON_COLS:
        v = row.get(col)
        assert not isinstance(v, str), (
            "%s 读回来是字符串 —— 它没走 json.loads，会让 API schema 的 dict "
            "声明失败并把整个列表端点打掉" % col)
        if v is not None:
            assert isinstance(v, dict), "%s 应当是 dict，实际 %r" % (col, type(v))


def test_the_json_column_list_matches_the_api_schema(tmp_path):
    """反序列化白名单里的列，API schema 必须都声明 —— 否则读出来是 null。

    这是同一件事的另一半（2026-08-18 已经踩过一次：`BaselinePoint` 漏了 Z 那
    批列，FastAPI 按 model_fields 静默过滤，读出来与"根本没采到"长得一模一样）。
    """
    from mast.api.schemas_monitoring import BaselineRow
    from mast.monitoring.store import _BASELINE_JSON_COLS

    missing = [c for c in _BASELINE_JSON_COLS if c not in BaselineRow.model_fields]
    assert not missing, "这些 JSON 列 API schema 里没有，会被静默过滤成 null: %s" % missing
