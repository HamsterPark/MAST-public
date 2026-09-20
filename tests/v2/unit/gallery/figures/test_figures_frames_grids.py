# -*- coding: utf-8 -*-
"""标记帧对比页与网格谱逐层页（设计文档 D19，T29）。"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

from mast.gallery.figures import service

DAY = "2001/200109/20010910"


def _lum(img, box) -> float:
    return float(np.asarray(img.convert("L").crop(box), dtype=float).mean())


def test_marked_frames_crop_unscanned_rows_keep_orientation_and_are_named_by_start_time(lab):
    rng = np.random.default_rng(0)
    z = rng.normal(0, 5e-12, (64, 64))
    z[:16] = np.nan                                  # 从下往上扫、没扫完：上面 16 行是空的
    z[16:24] += 5e-11                                # 扫到的最上面 8 行是亮带
    fid = lab.frame(DAY, "topo_0001.sxm", z, t=lab.epoch("10.09.2001 15:07:12"), range_nm=(5.0, 5.0), scan_dir="up")
    other = lab.frame(DAY, "topo_0002.sxm", rng.normal(0, 5e-12, (32, 32)), t=lab.epoch("10.09.2001 15:20:00"),
                      range_nm=(5.0, 5.0), angle=12.0)
    big = [lab.frame(DAY, "rot_%04d.sxm" % i, rng.normal(0, 5e-12, (16, 16)),
                     t=lab.epoch("10.09.2001 16:00:00") + 100 * i, range_nm=(5.0, 5.0)) for i in range(3)]
    lab.build()
    lab.mark(items={fid: {"r": 2, "ts": 1}},
             series={"S1": {"name": "小系列", "ids": [other], "ts": 1}, "S2": {"name": "大系列", "ids": big, "ts": 1}})

    res = service.run_job("marked_frames", options={"max_series_frames": 2, "include_grids": False})
    assert res["phase"] == "done" and res["errors"] == [], res
    assert res["made"] == ["frames/20010910-150712_20010910_0001", "frames/20010910-152000_20010910_0002"]
    meta = lab.figure_meta(res["made"][0])
    assert meta["files"] == ["20010910-150712_20010910_0001.png"] and meta["ids"] == [fid]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", meta["created"])
    d = meta["detail"]
    assert (d["i0"], d["i1"], d["rows_ok"], d["rall"]) == (16, 64, 48, 64)
    assert meta["summary"]["有效行"] == "48/64" and meta["summary"]["评级"] == "重点"
    with Image.open(lab.state / "figures" / "frames" / meta["files"][0]) as im:
        assert im.width == 2 * 384 + 4
        # 左面板（原版）：亮带在上沿 —— .sxm 按采集顺序存盘、从下往上扫，读回来要正过来
        assert _lum(im, (220, 24, 380, 44)) > _lum(im, (220, 150, 380, 280)) + 40


def test_grid_sheet_picks_the_correlated_lock_in_channel_and_draws_scanned_rows_top_up(lab, synth):
    nx = ny = 6
    npts = 9
    V = np.linspace(2.0, -2.0, npts)
    amp = (1 + 0.2 * np.arange(ny))[:, None, None]  # 行号越大（越靠视野上沿）电流越大
    rng = np.random.default_rng(1)
    ch = {"Bias [AVG] (V)": np.broadcast_to(V, (ny, nx, npts)).copy(),
          "Current [AVG] (A)": np.tanh(V)[None, None, :] * amp * 1e-10 * np.ones((ny, nx, 1)),
          "LI Demod 1 X [AVG] (A)": rng.normal(0, 1e-15, (ny, nx, npts)),
          "LI Demod 1 Y [AVG] (A)": (1 / np.cosh(V) ** 2)[None, None, :] * amp * 1e-13 * np.ones((ny, nx, 1))}
    synth.grid(lab.root / DAY / "Grid Spectroscopy001.3ds", nx=nx, ny=ny, npts=npts, channels=ch, have=30,
               z_param=np.outer(np.arange(ny), np.ones(nx)) * 1e-11)
    lab.build()
    res = service.run_job("grid_sheets")
    assert res["phase"] == "done" and res["errors"] == [], res
    meta = lab.figure_meta(res["made"][0])
    s, d = meta["summary"], meta["detail"]
    assert meta["base"] == "20010911-153126_Grid_Spectroscopy001"
    assert s["dI/dV"] == "LI Demod 1 Y" and s["r"] >= 0.9 and s["完成"] == "30/36" and s["层数"] == npts
    assert d["rows"] == 5
    with Image.open(lab.state / "figures" / "grids" / meta["files"][0]) as im:
        # 电流块第一格（+2 V 层）：k=24 px/点，5 行；.3ds 第 0 行在下沿，翻过来以后电流最大的那一行在最上面
        x0, y0 = 144 + 10, 158 + 20 + 15
        assert _lum(im, (x0, y0, x0 + 144, y0 + 24)) > _lum(im, (x0, y0 + 96, x0 + 144, y0 + 120)) + 40
