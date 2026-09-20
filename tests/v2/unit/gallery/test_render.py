# -*- coding: utf-8 -*-
"""缩略图（设计文档 T4 / T5 / T7 / T16 / T17 / D7）。"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from mast.gallery import grid3ds, inventory, render


def _gray(path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=float)


def _size(path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


# ── 帧：方向与几何 ─────────────────────────────────────────────────────


@pytest.mark.parametrize("scan_dir", ["down", "up"])
def test_the_top_edge_of_the_frame_is_the_top_of_the_thumbnail(tmp_path, synth, scan_dir):
    """T4：翻转只有 ``sxm_oriented_frames`` 一个落点。上下不对称的图案才判得出翻转。"""
    n = 64
    z = np.full((n, n), 1e-11)
    z[: n // 6, :] = 9e-11                               # 上沿亮带
    z += np.linspace(0.0, 4e-12, n)[None, :]             # 左右也不对称
    # 视野 > 30 nm：只减平面。≤ 30 nm 时逐行去中位偏置会把亮带整条减掉。
    p = synth.sxm(tmp_path / f"{scan_dir}.sxm", z, scan_dir=scan_dir, range_nm=(50.0, 50.0))
    thumbs = tmp_path / "thumbs"
    res = render.render_frame(str(p), f"R/{scan_dir}.sxm", inventory.sxm_summary(str(p)), thumbs)
    assert res["ok"] and (res["r0"], res["r1"]) == (0, n)
    img = _gray(thumbs / "R" / f"{scan_dir}.sxm.jpg")
    hh = img.shape[0] - render.STRIP_PX
    assert img.shape[1] == render.THUMB_W and hh == render.THUMB_W
    top = img[: hh // 10].mean()
    bottom = img[hh - hh // 10 - 30: hh - 30].mean()    # 避开右下角比例尺
    assert top > bottom + 40, (scan_dir, top, bottom)


@pytest.mark.parametrize("scan_dir, expect", [("down", (0, 40)), ("up", (24, 64))])
def test_an_unfinished_frame_reports_the_rows_it_drew(tmp_path, synth, scan_dir, expect):
    """T5：前端用 r0/r1 把谱的位置画到缩略图上。没扫到的行按采集顺序在后面：
    向下扫缺下沿，向上扫缺上沿。"""
    n = 64
    z = np.random.default_rng(0).normal(0.0, 1e-11, (n, n))
    if scan_dir == "down":
        z[40:, :] = np.nan
    else:
        z[:24, :] = np.nan
    p = synth.sxm(tmp_path / "p.sxm", z, scan_dir=scan_dir)
    thumbs = tmp_path / "thumbs"
    res = render.render_frame(str(p), "R/p.sxm", inventory.sxm_summary(str(p)), thumbs)
    assert res["ok"]
    assert (res["r0"], res["r1"]) == expect
    assert (res["rows"], res["rows_all"]) == (40, 64)
    w, h = _size(thumbs / "R" / "p.sxm.jpg")
    assert (w, h - render.STRIP_PX) == (render.THUMB_W, max(int(render.THUMB_W * 40 / n), 8))


def test_a_frame_with_fewer_than_eight_rows_is_not_rendered(tmp_path, synth):
    z = np.full((64, 64), np.nan)
    z[:7] = 1e-11
    p = synth.sxm(tmp_path / "e.sxm", z)
    thumbs = tmp_path / "thumbs"
    res = render.render_frame(str(p), "R/e.sxm", inventory.sxm_summary(str(p)), thumbs)
    assert res == {"ok": False, "why": "empty", "rows": 7, "rows_all": 64, "v": render.VER_FRAME}
    assert not (thumbs / "R" / "e.sxm.jpg").exists()


def test_the_lockin_frame_uses_the_channel_that_has_in_row_contrast(tmp_path, synth):
    n = 64
    rng = np.random.default_rng(3)
    contrast = np.sin(np.linspace(0, 12 * np.pi, n))[None, :] * np.ones((n, 1)) * 1e-12
    dead = np.full((n, n), 5e-13) + rng.normal(0.0, 1e-17, (n, n))     # 只剩直流偏置
    p = synth.sxm(tmp_path / "li.sxm", rng.normal(0, 1e-11, (n, n)),
                  extra={"LI_Demod_1_X": dead, "LI_Demod_1_Y": contrast})
    thumbs = tmp_path / "thumbs"
    res = render.render_frame(str(p), "R/li.sxm", inventory.sxm_summary(str(p)), thumbs)
    assert res["li_ch"] == "LI_Demod_1_Y"
    assert res["li"] == "R/li.sxm.li.jpg" and (thumbs / "R" / "li.sxm.li.jpg").exists()


def test_a_frame_without_lockin_says_so(tmp_path, synth):
    p = synth.sxm(tmp_path / "z.sxm", np.random.default_rng(1).normal(0, 1e-11, (32, 32)))
    res = render.render_frame(str(p), "R/z.sxm", inventory.sxm_summary(str(p)), tmp_path / "t")
    assert res["li"] == "" and "li_ch" not in res and res["vli"] == render.VER_LI


# ── 谱：dI/dV 三态（T16）──────────────────────────────────────────────


def _spectrum(tmp_path, synth, name, *, li_x=None, li_y=None, noise_seed=0):
    V = np.linspace(-2.0, 2.0, 201)
    I = np.tanh(2 * V) * 1e-10
    cols, data = ["Bias calc (V)", "Current (A)"], [V, I]
    if li_x is not None:
        cols.append("LI Demod 1 X (A)")
        data.append(li_x)
    if li_y is not None:
        cols.append("LI Demod 1 Y (A)")
        data.append(li_y)
    p = synth.dat(tmp_path / name, cols, np.column_stack(data))
    return p, V, I


def test_lockin_signal_on_the_Y_channel_is_the_one_drawn(tmp_path, synth):
    """合成 Y 通道与数值导数相关而 X 为噪声时，应选择 Y，不能仅凭列名优先取 X。"""
    rng = np.random.default_rng(7)
    V = np.linspace(-2.0, 2.0, 201)
    didv = np.gradient(np.tanh(2 * V) * 1e-10, V)
    p, _V, _I = _spectrum(tmp_path, synth, "y.dat", li_x=rng.normal(0, 1e-13, V.size),
                          li_y=didv * 0.02 + rng.normal(0, 1e-15, V.size))
    res = render.render_spectrum(str(p), "R/y.dat", inventory.dat_summary(str(p)), tmp_path / "t")
    assert res["ok"] and res["didv"] == "li" and res["li"] is True
    assert res["li_ch"] == "LI Demod 1 Y" and res["li_r"] > 0.9


def test_uncorrelated_lockin_falls_back_to_the_numerical_derivative(tmp_path, synth):
    rng = np.random.default_rng(8)
    n = 201
    p, _V, _I = _spectrum(tmp_path, synth, "n.dat", li_x=rng.normal(0, 1e-13, n),
                          li_y=rng.normal(0, 1e-13, n))
    res = render.render_spectrum(str(p), "R/n.dat", inventory.dat_summary(str(p)), tmp_path / "t")
    assert res["ok"] and res["didv"] == "num" and res["li"] is False
    assert res["sg_window"] == 9


def test_a_file_without_a_current_column_is_not_a_bias_spectrum(tmp_path, synth):
    f = np.logspace(0, 3, 50)
    p = synth.dat(tmp_path / "psd.dat", ["Frequency (Hz)", "Z PSD (m/sqrt(Hz))"],
                  np.column_stack([f, 1e-12 / f]), header={"Experiment": "Spectrum"})
    res = render.render_spectrum(str(p), "R/psd.dat", inventory.dat_summary(str(p)), tmp_path / "t")
    assert res["ok"] and res["didv"] == "" and res["li"] is False
    assert (tmp_path / "t" / "R" / "psd.dat.png").exists()


def test_spectra_render_correctly_when_many_threads_draw_at_once(tmp_path, synth):
    """并发生成合成谱图时，共享 mathtext 解析器由绘图锁保护。"""
    paths = [_spectrum(tmp_path, synth, f"c{i}.dat")[0] for i in range(12)]
    summ = inventory.dat_summary(str(paths[0]))
    results: list[dict] = []
    errors: list[BaseException] = []

    def work(p):
        try:
            results.append(render.render_spectrum(str(p), f"R/{p.name}", summ, tmp_path / "t"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(p,)) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert errors == [] and len(results) == 12 and all(r["ok"] for r in results)


# ── 数值导数（T17）────────────────────────────────────────────────────


def test_savitzky_golay_is_exact_on_a_cubic_including_both_ends():
    V = np.linspace(-2.0, 2.0, 251)
    d, w = render.num_deriv(V, V ** 3 - V)
    assert w == 11
    np.testing.assert_allclose(d, 3 * V ** 2 - 1, atol=1e-6)


def test_a_straight_line_has_no_edge_spikes():
    V = np.linspace(-1.0, 1.0, 101)
    d, w = render.num_deriv(V, 2.0 * V + 0.3)
    assert w == 5
    assert np.max(np.abs(d - 2.0)) < 1e-9


def test_uneven_bias_falls_back_to_gradient():
    V = np.r_[np.linspace(-1.0, 0.0, 50), np.linspace(0.05, 1.0, 20)]
    d, w = render.num_deriv(V, V ** 2)
    assert w == 0
    np.testing.assert_allclose(d[1:-1], 2 * V[1:-1], atol=0.06)


def test_too_few_points_do_not_smooth():
    V = np.array([0.0, 1.0, 2.0, 3.0])
    d, w = render.num_deriv(V, V * 2)
    assert w == 0 and np.allclose(d, 2.0)


# ── 网格 ───────────────────────────────────────────────────────────────


def _grid_channels(nx, ny, npts):
    V = np.linspace(2.0, -2.5, npts)
    I = np.tanh(V)[None, None, :] * 1e-10 * np.ones((ny, nx, 1))
    return {
        "Current [AVG] (A)": I,
        "Bias [AVG] (V)": np.broadcast_to(V, (ny, nx, npts)).copy(),
        "LI Demod 1 X [AVG] (A)": np.gradient(I, V, axis=2) * 0.01,
        "LI Demod 1 Y [AVG] (A)": np.random.default_rng(1).normal(0, 1e-15, (ny, nx, npts)),
    }


def test_grid_reader_keeps_row_zero_at_the_bottom_and_counts_have(tmp_path, synth):
    nx, ny, npts = 4, 3, 5
    ch = {"Current [AVG] (A)": np.arange(ny * nx * npts, dtype=float).reshape(ny, nx, npts)}
    p = synth.grid(tmp_path / "g.3ds", nx=nx, ny=ny, npts=npts, channels=ch, have=6)
    G = grid3ds.read_3ds(str(p))
    assert (G["nx"], G["ny"], G["npts"], G["have"]) == (4, 3, 5, 6)
    D = G["D"]
    np.testing.assert_allclose(D[0, 0, 0, :], ch["Current [AVG] (A)"][0, 0, :])   # 第一个点 = 行 0
    np.testing.assert_allclose(D[1, 1, 0, :], ch["Current [AVG] (A)"][1, 1, :])
    assert np.isnan(D[2, 3, 0, :]).all()                                           # 没做完的点
    assert G["pars"] == ["Sweep Start", "Sweep End", "X (m)", "Y (m)", "Z (m)"]


def test_grid_renders_the_full_panel_and_the_partial_panel(tmp_path, synth):
    nx = ny = 8
    npts = 11
    ch = _grid_channels(nx, ny, npts)
    full = synth.grid(tmp_path / "full.3ds", nx=nx, ny=ny, npts=npts, channels=ch,
                      z_param=np.random.default_rng(2).normal(0, 1e-11, (ny, nx)))
    part = synth.grid(tmp_path / "part.3ds", nx=nx, ny=ny, npts=npts, channels=ch, have=5)
    thumbs = tmp_path / "t"
    for p, size in ((full, (840, 430)), (part, (840, 320))):
        res = render.render_grid(str(p), f"R/{p.name}", inventory.grid_summary(str(p)), thumbs)
        assert res["ok"], res
        assert _size(thumbs / "R" / f"{p.name}.png") == size


def test_an_empty_grid_is_not_rendered(tmp_path, synth):
    ch = _grid_channels(4, 4, 5)
    p = synth.grid(tmp_path / "e.3ds", nx=4, ny=4, npts=5, channels=ch, have=0)
    res = render.render_grid(str(p), "R/e.3ds", inventory.grid_summary(str(p)), tmp_path / "t")
    assert res == {"ok": False, "why": "empty", "v": render.VER_GRID}
