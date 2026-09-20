"""``AssessSpectrum`` 的技能外壳。

判据本体在 ``tests/v2/unit/vision/test_spectrum_quality.py``;这里测的是**外壳才
管得到的四件事**:

1. **合成 .dat 读得动**,列名与坐标解析正确 —— 独立生成 Nanonis 文本格式
   文件，通过文件解析链路检查结果，不伪造技能回包;
2. **正/反扫显式区分** —— 反扫列排在前面时,正扫列不能被误选;
3. **kind 用内容判据** —— 头里的 ``Experiment`` 字段说谎时以数据为准并点名;
4. **三态** —— ``success=True`` 只要文件读得动,判决在 ``data.verdict``。
"""

from __future__ import annotations

import numpy as np
import pytest

from mast.core.types import SafetyLevel, SkillCategory
from mast.skills.builtins.spectrum_assess import AssessSpectrum

from mast.vision.spectroscopy import assess_iv, spectrum_snr


@pytest.fixture()
def synthetic_columns():
    """公式生成的双向谱：一个正扫尖峰，以及三个独立的反扫离群点。"""
    cols = iv_columns(n=257, seed=14)
    forward = cols["Current (A)"]
    forward[73] += 0.8e-9
    backward = forward.copy()
    backward[[41, 129, 211]] += 0.5e-9
    cols["Current [bwd] (A)"] = backward
    # 与写入文件的小数精度一致，使透传比较包含文件序列化的影响。
    return {
        name: np.array([float(f"{value:.7E}") for value in values])
        for name, values in cols.items()
    }


@pytest.fixture()
def synthetic_dat(tmp_path, synthetic_columns) -> str:
    """临时目录中的独立合成文件；坐标只是解析测试的任意输入。"""
    return write_dat(
        tmp_path / "synthetic_iv.dat",
        {"Experiment": "bias spectroscopy", "X (m)": 1.25e-7, "Y (m)": -7.5e-8},
        synthetic_columns,
    )


def _run(path, **params):
    return AssessSpectrum().execute(None, {"dat_path": str(path), **params})


def write_dat(path, header: dict, columns: dict) -> str:
    """写一个最小的合成 Nanonis .dat(``key\\tvalue`` 头 + [DATA] + 列)。"""
    names = list(columns)
    rows = zip(*[columns[n] for n in names])
    lines = [f"{k}\t{v}\t" for k, v in header.items()]
    lines += ["", "[DATA]", "\t".join(names)]
    lines += ["\t".join(f"{v:.7E}" for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def iv_columns(n=200, *, bwd_gain=1.0, seed=0):
    V = np.linspace(-1.0, 1.0, n)
    I = np.sinh(2.0 * V) * 1e-9
    rng = np.random.default_rng(seed)
    amp = 1e-12
    return {
        "Bias calc (V)": V,
        "Current (A)": I + amp * rng.normal(size=n),
        "Current [bwd] (A)": I * bwd_gain + amp * rng.normal(size=n),
    }


# ── 1. 独立合成文件 ────────────────────────────────────────────────────


def test_synthetic_fixture_is_read_and_every_column_is_named(synthetic_dat):
    res = _run(synthetic_dat)
    assert res.success is True
    d = res.data
    assert d["bias_column"] == "Bias calc (V)"
    assert d["current_column"] == "Current (A)"
    assert d["current_bwd_column"] == "Current [bwd] (A)"
    assert d["n_points"] == 257
    assert d["kind_resolved"] == "iv"
    assert d["header_experiment"] == "bias spectroscopy"
    # 任意合成坐标必须从头部原样读出，不能用默认位置代替。
    assert d["dat_x_m"] == pytest.approx(1.25e-7, rel=0, abs=1e-20)
    assert d["dat_y_m"] == pytest.approx(-7.5e-8, rel=0, abs=1e-20)


def test_synthetic_fixture_default_verdict_is_unrated_because_nothing_is_calibrated(
        synthetic_dat):
    """所有阈值未标定时必须返回 unrated，不能产生 keep 或 discard 判定。"""
    res = _run(synthetic_dat)
    assert res.data["verdict"] == "unrated"
    assert res.data["reasons"] == ["all_criteria_uncalibrated"]
    assert res.data["gated_criteria"] == []


def test_synthetic_fixture_reports_every_number_even_when_it_rates_nothing(
        synthetic_dat, synthetic_columns):
    """未标定仍报告指标；人工注入的三个离群点给出独立计数锚点。"""
    forward = synthetic_columns["Current (A)"]
    backward = synthetic_columns["Current [bwd] (A)"]
    span = float(np.percentile(forward, 95) - np.percentile(forward, 5))
    expected_max = float(np.max(np.abs(forward - backward))) / span
    d = _run(synthetic_dat).data
    assert d["spectrum_snr"] == pytest.approx(spectrum_snr(forward))
    assert d["spectrum_snr"] > 10
    assert d["saturation_frac"] == 0.0
    assert d["hysteresis_median_frac"] == 0.0
    assert expected_max > 0
    assert d["hysteresis_max_frac"] == pytest.approx(expected_max)
    assert d["hysteresis_outlier_points"] == 3
    assert d["hysteresis_outlier_frac"] == pytest.approx(3 / 257)
    assert d["verdict"] == "unrated"


def test_synthetic_fixture_iv_metrics_are_passed_through_not_judged(
        synthetic_dat, synthetic_columns):
    """所有 I(V) 指标原样透传；人工尖峰也不能让未标定的谱产生判定。"""
    expected = assess_iv(synthetic_columns["Bias calc (V)"],
                         synthetic_columns["Current (A)"])
    assert expected.n_spikes == 2
    assert expected.is_stable is False
    assert expected.gap_ev is not None and expected.gap_ev > 0
    d = _run(synthetic_dat).data
    assert d["iv_n_spikes"] == expected.n_spikes
    assert d["iv_is_stable"] is expected.is_stable
    assert d["iv_smoothness"] == pytest.approx(expected.smoothness)
    assert d["iv_symmetry"] == pytest.approx(expected.symmetry)
    assert d["iv_gap_ev"] == pytest.approx(expected.gap_ev)
    assert "gap_ev_is_not_a_gap_measurement" in d["warnings"]
    assert d["verdict"] == "unrated"


def test_synthetic_fixture_family_unknown_never_gates_the_family_dependent_ones(
        synthetic_dat):
    """``spectral_family`` 留空 = unknown,**不是** metallic。"""
    d = _run(synthetic_dat, min_snr=10.0, max_saturation_frac=0.1,
             max_hysteresis_outlier_frac=0.05).data
    assert d["spectral_family"] == "unknown"
    assert d["spectral_family_source"] == "default_unknown"
    assert set(d["gated_criteria"]) == {"saturation", "snr", "hysteresis"}
    assert "iv_stability" in d["ungated_criteria"]
    assert "symmetry" in d["ungated_criteria"]
    assert d["verdict"] in ("keep", "keep_flagged")


def test_synthetic_fixture_calibrated_threshold_can_discard(synthetic_dat):
    """标定过的阈值当然能判掉 —— 未标定不产 bad 说的是**未标定**那一半。"""
    d = _run(synthetic_dat, max_hysteresis_outlier_frac=0.0).data
    assert d["gated_criteria"] == ["hysteresis"]
    assert d["verdict"] == "discard"
    assert d["reasons"] == ["hysteresis_exceeded"]


# ── 2. 正/反扫列不能靠字典顺序 ──────────────────────────────────────────


def test_backward_column_listed_first_does_not_steal_the_forward_column(tmp_path):
    """反扫列排在前面时,正扫列仍要选对。

    正扫与反扫都含 Current，不能简单按第一个包含子串的列来选取。
    交换列序后仍必须按方向选择；两列电流的量级本身不能代替方向信息。
    """
    cols = iv_columns()
    fwd, bwd = cols["Current (A)"], cols["Current [bwd] (A)"]
    reordered = {
        "Bias calc (V)": cols["Bias calc (V)"],
        "Current [bwd] (A)": bwd,     # ← 反扫排在前面
        "Current (A)": fwd,
    }
    p = write_dat(tmp_path / "reordered.dat",
                  {"Experiment": "bias spectroscopy"}, reordered)
    d = _run(p).data
    assert d["current_column"] == "Current (A)"
    assert d["current_bwd_column"] == "Current [bwd] (A)"


def test_missing_backward_column_reports_all_three_as_none(tmp_path):
    """缺反扫列 ⇒ 迟滞三个数**全 None** + ``no_backward_column``。

    而 ``verdict != "keep"`` 只在 ``require_backward=True`` 时成立 —— 少一列不等于
    数据不好。
    """
    cols = iv_columns()
    cols.pop("Current [bwd] (A)")
    p = write_dat(tmp_path / "no_bwd.dat",
                  {"Experiment": "bias spectroscopy"}, cols)

    d = _run(p, max_saturation_frac=0.5).data
    assert d["current_bwd_column"] is None
    assert d["hysteresis_median_frac"] is None
    assert d["hysteresis_max_frac"] is None
    assert d["hysteresis_outlier_points"] is None
    assert "no_backward_column" in d["reasons"]
    assert d["verdict"] in ("keep", "keep_flagged")

    strict = _run(p, max_saturation_frac=0.5, require_backward=True).data
    assert strict["verdict"] == "unrated"
    assert "no_backward_column" in strict["reasons"]


# ── 3. kind 用内容判据，头只交叉检验 ────────────────────────────────────


def test_kind_auto_follows_the_data_when_the_header_lies(tmp_path):
    """头里写着 bias spectroscopy,数据实际在扫 Z ⇒ ``iz`` + 点名不一致。

    字段标签会说谎(它是仪器软件上一次的设置留下的)。以数据为准,并**说出来**。
    """
    n = 120
    z = np.linspace(0.0, 5e-10, n)
    cols = {
        "Bias calc (V)": np.full(n, 0.2),        # 恒定 —— 没在扫
        "Z rel (m)": z,                          # 在扫
        "Current (A)": 1e-9 * np.exp(-21.7 * z * 1e9),
    }
    p = write_dat(tmp_path / "z_but_says_bias.dat",
                  {"Experiment": "bias spectroscopy"}, cols)
    d = _run(p).data
    assert d["kind_resolved"] == "iz"
    assert "kind_disagrees_with_header" in d["warnings"]
    assert "以数据为准" in d["kind_evidence"]
    assert d["iz_fit_r2"] is not None
    assert d["iv_n_spikes"] is None


def test_kind_auto_refuses_when_nothing_is_sweeping(tmp_path):
    """两条都不在扫 ⇒ ``unrated`` + ``kind_undetermined``,而不是挑一个。"""
    n = 40
    cols = {
        "Bias calc (V)": np.full(n, 0.2),
        "Current (A)": np.full(n, 1e-9) + 1e-13 * np.arange(n),
    }
    p = write_dat(tmp_path / "flat.dat", {"Experiment": ""}, cols)
    d = _run(p).data
    assert d["kind_resolved"] == ""
    assert d["verdict"] == "unrated"
    assert d["reasons"] == ["kind_undetermined"]


def test_explicit_kind_wins_and_says_so(tmp_path):
    cols = iv_columns()
    p = write_dat(tmp_path / "iv.dat", {"Experiment": "bias spectroscopy"}, cols)
    d = _run(p, kind="iv").data
    assert d["kind_resolved"] == "iv"
    assert "显式指定" in d["kind_evidence"]


# ── 4. 三态：success 是「读得动」，不是「合格」 ─────────────────────────


def test_success_is_about_the_file_not_the_verdict(tmp_path, synthetic_dat):
    """把「这条谱不合格」表达成技能失败,会让 composite 的必做步骤直接中止整条
    流程,而「这条谱不好」恰恰是流程要处理的正常情况。"""
    bad = _run(synthetic_dat, max_hysteresis_outlier_frac=0.0)
    assert bad.success is True and bad.data["verdict"] == "discard"

    missing = _run(tmp_path / "nope.dat")
    assert missing.success is False and "不存在" in missing.error


def test_unreadable_and_empty_files_fail_but_a_missing_column_does_not(tmp_path):
    """分界:读不出来 ⇒ ``success=False``;读得出来但缺列 ⇒ ``unrated``。"""
    empty = tmp_path / "empty.dat"
    empty.write_text("Experiment\tbias spectroscopy\t\n\n[DATA]\n", encoding="utf-8")
    res = _run(empty)
    assert res.success is False and "没有数据列" in res.error

    n = 60
    p = write_dat(tmp_path / "no_current.dat",
                  {"Experiment": "bias spectroscopy"},
                  {"Bias calc (V)": np.linspace(-1, 1, n),
                   "Something else (V)": np.zeros(n)})
    res = _run(p)
    assert res.success is True
    assert res.data["verdict"] == "unrated"
    assert res.data["reasons"] == ["no_current_column"]
    assert res.data["current_column"] is None


def test_summary_says_which_criteria_actually_decided(synthetic_dat):
    """``verdict`` 没有 ``gated_criteria`` 就是一句没有依据的话 —— 摘要里必须有。"""
    assert "当闸的判据：无" in _run(synthetic_dat).summary
    s = _run(synthetic_dat, max_hysteresis_outlier_frac=0.5).summary
    assert "hysteresis" in s


# ── 5. 元数据 / 记录层 / 通用性 ─────────────────────────────────────────


def test_metadata_is_analysis_and_auto():
    m = AssessSpectrum().metadata()
    assert m.category is SkillCategory.ANALYSIS
    assert m.safety_level is SafetyLevel.AUTO
    assert m.preconditions == []          # 只读文件,不需要仪器处于任何状态
    names = {p.name for p in m.parameters}
    assert names == {"dat_path", "kind", "spectral_family", "min_snr",
                     "max_saturation_frac", "max_hysteresis_outlier_frac",
                     "min_smoothness", "require_backward"}


def test_every_threshold_parameter_defaults_to_uncalibrated():
    """阈值参数**不能**有出厂默认值。

    ``pydantic`` 会把 default 物化,写一个数就分不出「没传」与「显式传了这个数」;
    而这里「没传」的含义是**未标定**,一个兜底的默认值会让它看起来像标定过了。
    """
    m = AssessSpectrum().metadata()
    for name in ("min_snr", "max_saturation_frac", "max_hysteresis_outlier_frac",
                 "min_smoothness"):
        spec = next(p for p in m.parameters if p.name == name)
        assert spec.required is False
        assert spec.default is None, f"{name} 不该有出厂阈值"


def test_analysis_skill_must_not_leave_a_spectroscopy_footprint_on_the_map():
    """只读分析不得新增采集位置；分类元数据和名称回退规则都要排除它。"""
    from mast.io.exp_map import classify_skill

    meta = AssessSpectrum().metadata()
    assert classify_skill(meta.name, meta.category) is None
    # 调用方忘了传 category 时也要挡住(名字规则那一层)。
    assert classify_skill(meta.name) is None


def test_no_sample_names_anywhere_in_the_shell():
    """通用层不带样品名:MAST 是 STM 实验系统,不是某一块样品的系统。"""
    import inspect

    from mast.skills.builtins import spectrum_assess

    src = inspect.getsource(spectrum_assess).lower()
    for banned in ("wo2i2", "wo₂i₂", "woi", "au(111)", "ag(111)", "cu(111)"):
        assert banned not in src, f"外壳里出现了样品名: {banned}"
