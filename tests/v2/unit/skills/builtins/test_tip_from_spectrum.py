"""从谱判针尖：未标定的判据只能说「判不了」。

这一组的重心不在「好谱判 ok、坏谱判 bad」（那部分判据本体在
`tests/v2/unit/vision/` 里已经有合成数据的验证），而在**外壳这一层特有的三种
说谎方式**：

1. 把「所有阈值都没填」和「每条都通过了」说成同一句话 —— 两者在数值上都是
   「零个 flag」，而它们是完全相反的意思；
2. 把「判不了」折进「针坏了」—— 本仓在这个形状上一天栽过四次；
3. 把「这条谱不好」说成「这根针不行」—— 那会让流程去反复修一根没问题的针。
"""

from __future__ import annotations

import numpy as np
import pytest

from mast.skills.builtins.tip_from_spectrum import (
    DEFAULT_BARRIER_EV_MAX,
    DEFAULT_BARRIER_EV_MIN,
    TIP_VERDICTS,
    AssessTipFromSpectrum,
)


def _write_dat(path, columns: dict, *, experiment: str = "") -> str:
    """写一份最小可读的 Nanonis .dat。"""
    names = list(columns)
    n = len(next(iter(columns.values())))
    lines = []
    if experiment:
        lines.append(f"Experiment\t{experiment}")
    lines.append("")
    lines.append("[DATA]")
    lines.append("\t".join(names))
    for i in range(n):
        lines.append("\t".join(f"{columns[c][i]:.9g}" for c in names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _iz(path, *, kappa_per_nm=10.0, n=60, noise=0.0, jump_at=None):
    z_nm = np.linspace(0.0, 0.5, n)
    I = 1e-9 * np.exp(-2 * kappa_per_nm * z_nm)
    if jump_at is not None:
        I[jump_at:] *= 3.0
    if noise:
        rng = np.random.default_rng(0)
        I = I * (1 + noise * rng.standard_normal(n))
    return _write_dat(path, {"Z (m)": z_nm * 1e-9, "Current (A)": I},
                      experiment="Z spectroscopy")


def _iv(path, *, n=101, spikes=0):
    V = np.linspace(-0.5, 0.5, n)
    I = 1e-9 * np.sinh(4 * V)
    if spikes:
        rng = np.random.default_rng(1)
        for i in rng.choice(np.arange(5, n - 5), size=spikes, replace=False):
            I[i] *= 6.0
    return _write_dat(path, {"Bias (V)": V, "Current (A)": I},
                      experiment="bias spectroscopy")


def _run(path, **params):
    skill = AssessTipFromSpectrum()
    return skill.execute(None, {"dat_path": str(path), **params})


# ── 三态纪律 ──────────────────────────────────────────────────────

def test_with_nothing_calibrated_the_verdict_is_unrated_not_ok(tmp_path):
    """**这是这一组最重要的一条。**

    I(V) 的两个形状分没有任何标定值，所以不填阈值时它一条判据都没参与过。
    那时的正确答案是「判不了」，不是「都通过了」——后者会让一道空闸门看起来
    像一道通过了的闸门。
    """
    r = _run(_iv(tmp_path / "a.dat"))
    assert r.success is True
    assert r.data["verdict"] == "unrated"
    assert r.data["gated_criteria"] == []
    assert set(r.data["ungated_criteria"]) == {"n_spikes", "smoothness", "symmetry"}


def test_the_iz_barrier_default_is_physical_so_it_does_gate(tmp_path):
    """I(z) 不同：表观势垒有物理量纲，所以它带一个**有依据的**默认窗口。

    这条与上一条并列存在，是为了让「什么时候可以有默认值」这个区别留在测试里：
    能写下依据的（金属功函数 4-5 eV）可以有，写不下的（无量纲形状分）不许有。
    """
    r = _run(_iz(tmp_path / "b.dat", kappa_per_nm=10.0))
    assert "barrier_ev" in r.data["gated_criteria"]
    assert r.data["verdict"] in ("tip_ok", "tip_suspect", "tip_bad")


def test_a_barrier_below_the_window_is_flagged(tmp_path):
    """κ=5 ⇒ 势垒 ≈0.95 eV：仍是一条干净指数（判据本体认），但远低于金属功函数。

    这正是势垒窗口该抓的东西：曲线本身没毛病，可它在说「这不是一个金属针尖的
    真空隧道结」。选 κ=5 而不是更小的值是有意的 —— κ=1 时判据本体自己就不认它
    是指数了，那时报「势垒不对」不如报「这压根不是隧道曲线」准确。
    """
    r = _run(_iz(tmp_path / "c.dat", kappa_per_nm=5.0))
    assert "barrier_ev" in r.data["gated_criteria"]
    assert "barrier_outside_window" in r.data["reasons"]


def test_a_curve_that_is_not_an_exponential_at_all_says_that_instead(tmp_path):
    """κ=1 ⇒ 判据本体自己不认它是单指数。

    那时势垒值是拟合的副产物、不是读数，所以它**不参与判决** —— 拿它去判针尖，
    就是在自己刚造出来的数上做判决。曲线的问题由 clean_exponential 说。
    """
    r = _run(_iz(tmp_path / "c2.dat", kappa_per_nm=1.0))
    assert "not_clean_exponential" in r.data["reasons"]
    assert "barrier_ev" not in r.data["gated_criteria"]
    assert any("物理意义" in n for n in r.data["notes"])


def test_switching_the_barrier_window_off_leaves_only_the_shape_criterion(tmp_path):
    """关掉势垒窗口之后，"这是不是一条干净指数" 仍然在判 —— 它不需要标定。"""
    r = _run(_iz(tmp_path / "d.dat"), barrier_ev_min=0.0, barrier_ev_max=0.0)
    assert "barrier_ev" not in r.data["gated_criteria"]
    assert r.data["gated_criteria"] == ["clean_exponential"]


def test_every_verdict_is_in_the_closed_set(tmp_path):
    for f in (_iz(tmp_path / "e.dat"), _iv(tmp_path / "f.dat")):
        assert _run(f).data["verdict"] in TIP_VERDICTS


# ── 「判不了」永远不折进「针坏了」 ────────────────────────────────

def test_a_file_with_no_current_column_is_unrated_not_bad(tmp_path):
    p = tmp_path / "g.dat"
    _write_dat(p, {"Z (m)": np.linspace(0, 1e-9, 10),
                   "Amplitude (m)": np.ones(10)})
    r = _run(p)
    assert r.success is True, "文件读得动就不是「没做成」"
    assert r.data["verdict"] == "unrated"
    assert r.data["reasons"] == ["no_current_column"]


def test_a_missing_file_is_a_failure_not_a_bad_tip(tmp_path):
    r = _run(tmp_path / "nope.dat")
    assert r.success is False
    assert "verdict" not in (r.data or {}), (
        "文件不存在时不该给出任何针尖裁决 —— 那是「这件事没做成」")


def test_a_barrier_that_cannot_be_fitted_does_not_count_as_out_of_window(tmp_path):
    """拟合不出势垒 ⇒ 这一条没参与判决，**不是**「势垒不对」。"""
    p = tmp_path / "h.dat"
    # 全是噪声，压根不是指数
    rng = np.random.default_rng(3)
    _write_dat(p, {"Z (m)": np.linspace(0, 5e-10, 40),
                   "Current (A)": np.abs(rng.standard_normal(40)) * 1e-12})
    r = _run(p)
    assert "barrier_outside_window" not in r.data["reasons"]


# ── 与谱质量那一层是两件事 ────────────────────────────────────────

def test_this_skill_is_read_only_and_analysis_class(tmp_path):
    m = AssessTipFromSpectrum().metadata()
    assert m.category.name == "ANALYSIS"
    assert m.safety_level.name == "AUTO"
    # ANALYSIS 不取仪器令牌 —— 一个只读文件的判据不该排在一次长扫描后面。
    from mast.core.instrument_lock import needs_token

    assert needs_token(m) is False


def test_it_does_not_claim_to_judge_data_quality():
    """名字与描述都要说清它判的是针，不是数据。

    两件事混起来的后果很具体：一条 discard 的谱可能只是窗口开错了，而下游会
    据此去修一根没问题的针。
    """
    d = AssessTipFromSpectrum().metadata().description
    # 描述已中文化（2026-08-24）；钉的还是同一句话，只是换了语言。
    assert "不是数据质量闸门" in d
    assert "AssessSpectrum" in d, "描述里要指出那件事归谁管"


def test_symmetry_gating_warns_about_the_substrate(tmp_path):
    """带隙衬底会让好针看起来不对称 —— 用这条判据时必须说出来。"""
    r = _run(_iv(tmp_path / "i.dat"), min_symmetry=0.5)
    assert "symmetry" in r.data["gated_criteria"]
    assert any("衬底" in n for n in r.data["notes"])


def test_the_header_is_only_cross_checked_never_trusted(tmp_path):
    """字段标签会说谎（它是软件上一次设置留下的）。数据说了算，但不一致要说。"""
    p = tmp_path / "j.dat"
    z = np.linspace(0, 0.5, 40)
    _write_dat(p, {"Z (m)": z * 1e-9,
                   "Current (A)": 1e-9 * np.exp(-20 * z)},
               experiment="bias spectroscopy")     # 头说 I(V)，数据是 I(z)
    r = _run(p)
    assert r.data["kind"] == "iz", "跟着头走了 —— 应该跟着数据走"
    assert any("header_says" in n for n in r.data["notes"])
