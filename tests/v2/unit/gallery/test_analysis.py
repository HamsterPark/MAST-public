# -*- coding: utf-8 -*-
"""自动判据、重复保存与片段（设计文档 D6 / T15）。"""
from __future__ import annotations

import numpy as np

from mast.gallery import analysis, build, index
from mast.gallery import config as gcfg


def _lattice(n=256, width_nm=5.0, a_nm=0.25, amp=1e-11, noise=1e-12, seed=0, half=0.0):
    rng = np.random.default_rng(seed)
    x = np.arange(n) * width_nm / n
    X, Y = np.meshgrid(x, x)
    k = 2 * np.pi / a_nm
    z = amp * (np.cos(k * X) + np.cos(k * Y))
    if half:
        z = z + half * amp * np.cos(0.5 * k * X)
    return z + rng.normal(0.0, noise, (n, n))


def _configure(root, name="SPM", workers=2):
    roots, errors = gcfg.normalise_roots([{"name": name, "path": str(root)}])
    assert errors == []
    gcfg.save_config(gcfg.GalleryConfig(roots=roots, workers=workers))


# ── 原子相（严格口径进索引，放宽口径只进缓存）──────────────────────────────


def test_an_atomic_lattice_at_full_scale_passes(tmp_path, synth):
    n = 256
    p = synth.sxm(tmp_path / "atoms.sxm", _lattice(n), range_nm=(5.0, 5.0))
    a = analysis.analyse_frame(str(p), 5.0 / n, 0, n)
    assert a["scale"] == "full" and a["at"] >= 20 and a["ar"] == "", a


def test_noise_is_absent_not_undetermined(tmp_path, synth):
    """「没有」与「判不了」必须分开：纯噪声是前者。"""
    n = 256
    p = synth.sxm(tmp_path / "noise.sxm", np.random.default_rng(1).normal(0, 1e-11, (n, n)),
                  range_nm=(5.0, 5.0))
    a = analysis.analyse_frame(str(p), 5.0 / n, 0, n)
    assert a["at"] == 0 and a["ar"] == "", a


def test_reduced_scale_is_undetermined_in_the_index_but_the_relaxed_verdict_is_kept(tmp_path, synth):
    n = 256
    p = synth.sxm(tmp_path / "coarse.sxm", _lattice(n, width_nm=10.0, a_nm=0.5),
                  range_nm=(10.0, 10.0))
    a = analysis.analyse_frame(str(p), 10.0 / n, 0, n)
    assert a["scale"] == "reduced"
    assert a["at"] == 0 and a["ar"] == "scale_reduced"
    assert a["at_reduced"] >= 20


def test_the_scale_gate_answers_without_reading_the_file(tmp_path):
    a = analysis.analyse_frame(str(tmp_path / "does-not-exist.sxm"), 0.1, 0, 256)
    assert a["ar"] == "scale_gate" and a["at"] == 0


def test_too_few_rows_is_insufficient_data(tmp_path):
    a = analysis.analyse_frame(str(tmp_path / "x.sxm"), 0.01, 0, 40)
    assert a["ar"] == "insufficient_data"


def test_the_undetermined_reasons_still_exist_in_the_judge():
    """那边改了名字，这里不能静默地把「判不了」读成「没有」。"""
    from mast.vision.atomic_phase import ALL_REASONS

    assert set(analysis.UNDETERMINED) <= set(ALL_REASONS)


def test_a_half_order_modulation_is_reported_as_superstructure(tmp_path, synth):
    n = 256
    p = synth.sxm(tmp_path / "ss.sxm", _lattice(n, half=0.3), range_nm=(5.0, 5.0))
    a = analysis.analyse_frame(str(p), 5.0 / n, 0, n)
    assert a["at"] > 0, a
    assert a["hf"] > 1.5 and "0.5" in a["hl"], a


def test_a_plain_lattice_has_no_superstructure(tmp_path, synth):
    n = 256
    p = synth.sxm(tmp_path / "plain.sxm", _lattice(n, seed=3), range_nm=(5.0, 5.0))
    a = analysis.analyse_frame(str(p), 5.0 / n, 0, n)
    assert a["at"] > 0 and a["hf"] == 0 and a["hl"] == "", a


# ── 重复保存 vs 同一次采集的片段（T15）─────────────────────────────────


def test_a_repeat_save_is_dup_and_a_shorter_save_of_the_same_scan_is_seg(
        tmp_path, synth, gallery_dir, t0_ns):
    d = tmp_path / "SPM" / "20010910"
    z = np.random.default_rng(4).normal(0, 1e-11, (64, 64))
    synth.sxm(d / "s_0048.sxm", z, rec_time="15:07:12", mtime_ns=t0_ns)
    synth.sxm(d / "s_0049.sxm", z, rec_time="15:07:12", mtime_ns=t0_ns + 10**9)      # 同一块 Z 又存了一次
    part = z.copy()
    part[40:] = np.nan
    synth.sxm(d / "s_0050.sxm", part, rec_time="15:07:12", mtime_ns=t0_ns + 2 * 10**9)  # 中途停下的那份
    synth.sxm(d / "s_0051.sxm", z, rec_time="15:20:00", mtime_ns=t0_ns + 3 * 10**9)  # 另一次采集
    _configure(tmp_path / "SPM")
    res = build.run_build(state=gallery_dir)
    assert res["phase"] == "done", res
    items = {it["fn"]: it for it in index.read_index()["items"]}
    base = "SPM/20010910/"
    assert items["s_0049.sxm"]["dup"] == base + "s_0048.sxm"
    assert all("dup" not in items[f] for f in ("s_0048.sxm", "s_0050.sxm", "s_0051.sxm"))
    assert items["s_0048.sxm"]["seg"] == 2 and items["s_0050.sxm"]["seg"] == 2
    assert "seg" not in items["s_0049.sxm"] and "seg" not in items["s_0051.sxm"]


def test_the_dup_verdict_is_recomputed_when_a_file_changes(tmp_path, synth, gallery_dir, t0_ns):
    """比对缓存按两边的 (size, mtime_ns) 失效（T1）。"""
    d = tmp_path / "SPM" / "d"
    z = np.random.default_rng(5).normal(0, 1e-11, (32, 32))
    synth.sxm(d / "a.sxm", z, mtime_ns=t0_ns)
    b = synth.sxm(d / "b.sxm", z, mtime_ns=t0_ns + 10**9)
    _configure(tmp_path / "SPM")
    build.run_build(state=gallery_dir)
    assert {it["fn"]: it for it in index.read_index()["items"]}["b.sxm"]["dup"] == "SPM/d/a.sxm"
    z2 = z.copy()
    z2[0, 0] += 1e-12
    synth.sxm(b, z2, mtime_ns=t0_ns + 5 * 10**9)
    build.run_build(state=gallery_dir)
    assert "dup" not in {it["fn"]: it for it in index.read_index()["items"]}["b.sxm"]


def test_a_harmonic_cell_may_not_report_superstructure(monkeypatch):
    """合成回归：谐波原胞必须跳过超结构检验，合理参考周期则继续检验。"""
    import mast.vision.lattice_cell as lc

    half_cell = lc.CellResult(ok=True, a1_nm=0.25, a2_nm=0.2, gamma_deg=90.0, a1_angle_deg=0.0)
    fake = (lc.SuperstructureResult(label="(0.5,0)", period_nm=0.5, ratio_to_control=4.0,
                                    verdict="present"),)
    monkeypatch.setattr(lc, "measure_cell", lambda *a, **k: half_cell)
    monkeypatch.setattr(lc, "superstructure_test", lambda *a, **k: fake)
    rows = np.zeros((64, 64))
    assert analysis.superstructure(rows, 0.02, ref_period_nm=0.5) == (0.0, "", "cell_is_a_harmonic")
    assert analysis.superstructure(rows, 0.02, ref_period_nm=0.25) == (4.0, "(0.5,0)", "")
