# -*- coding: utf-8 -*-
"""旋转系列：方向约定（T29）、首帧种子（D18 / T26）、一轮的划分（T34）、叠加与配准表。"""
from __future__ import annotations

import csv
import io
import math

import numpy as np
import pytest
from PIL import Image

from mast.gallery.figures import series, service


def _corr(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def test_sampling_turns_the_same_way_as_the_scan_angle(lab):
    """SCAN_ANGLE 为正 = 扫描框顺时针转：把 30° 那帧按约定采回 0° 朝向，要与 0° 那帧重合。"""
    z0 = lab.scene(0.0)
    fr = {"it": {"w": 5.0, "hn": 5.0}, "z": lab.scene(30.0), "th": 30.0}
    G = series.grid(5.0, 128)
    assert _corr(series.sample(fr, G, 0.0), z0) > 0.95
    assert _corr(series.sample(dict(fr, th=-30.0), G, 0.0), z0) < 0.5    # 转反了对不上 —— 判据本身有分辨力


def test_the_seed_is_read_in_x_right_y_up_coordinates(lab):
    th = 20.0
    k = series.seed_basis(lab.scene(th, px=256, w=10.0, defect=(50.0, 50.0)), 10.0 / 256)
    assert k is not None
    want = [series.rotvec(g, th) for g in lab.G]

    def ang(v):
        return math.degrees(math.atan2(v[1], v[0])) % 180

    assert sorted(ang(v) for v in k) == pytest.approx(sorted(ang(v) for v in want), abs=4)
    assert sorted(float(np.linalg.norm(v)) for v in k) == pytest.approx(
        sorted(float(np.linalg.norm(v)) for v in want), rel=0.06)


def test_a_round_ends_when_the_column_does_not_advance():
    cols, rows, keys = series.column_layout([102, 124.5, 147, 102, 124.5, 147, 124.5, 147, 0], 102)
    assert cols == [102, 124.5, 147]
    assert rows == [{0: 0, 1: 1, 2: 2}, {0: 3, 1: 4, 2: 5}, {1: 6, 2: 7}, {0: 8}]
    assert min(rows[2]) != 0                         # 不从第 0 列开始 ⇒ 图上标「补扫」
    assert keys[8] == 0 and keys[8] not in cols      # 只出现一次的转角进角度最近的列（102°）并标注


def test_rotation_series_slides_and_stack(lab):
    ds = lab.rotation_dataset()
    lab.build()
    lab.mark(series={"SR": {"name": "转角系列", "ids": ds["ids"], "ts": 1}})

    res = service.run_job("series_slides", series=["SR"], options={"page_w": 960, "page_h": 540})
    assert res["phase"] == "done" and res["errors"] == [], res
    meta = lab.figure_meta(res["made"][0])
    b = meta["base"]
    assert b == "转角系列_16比9"
    assert meta["files"] == [b + "_第1页.png", b + "_第1页.jpg", b + "_第2页.png", b + "_第2页.jpg"]
    assert meta["detail"]["cols"] == [20.0, 50.0, 80.0] and len(meta["detail"]["rounds"]) == 2
    with Image.open(lab.state / "figures" / "series" / meta["files"][0]) as im:
        assert im.size == (960, 540)

    res = service.run_job("series_stack", series=["SR"], options={"fov_nm": 4.0})
    assert res["phase"] == "done" and res["errors"] == [], res
    meta = lab.figure_meta(res["made"][0])
    b = meta["base"]
    out = lab.state / "figures" / "series"
    assert meta["files"][0] == b + ".png" and (out / (b + "_主图_2x.png")).is_file()
    rows = list(csv.DictReader(io.StringIO((out / (b + "_配准表.csv")).read_text("utf-8-sig"))))
    assert [r["文件"] for r in rows] == ["rot_%04d.sxm" % (k + 1) for k in range(6)]
    for r, d in zip(rows, ds["defect"]):
        # 锚点的压电坐标 = 缺陷的真实位置：转角、帧心、SCAN_ANGLE 符号、行 0 在上 —— 这一串约定任何一处翻错都会漂走
        assert float(r["锚点压电 X nm"]) == pytest.approx(d[0], abs=0.03)
        assert float(r["锚点压电 Y nm"]) == pytest.approx(d[1], abs=0.03)
        assert r["仿射已用"] == "1"
        assert 0.98 < float(r["仿射奇异值小"]) <= float(r["仿射奇异值大"]) < 1.02
    n = round(4.0 / (ds["w"] / ds["px"]))
    for suffix in ("_晶格校正平均.npy", "_刚性平均.npy", "_覆盖帧数.npy"):
        assert np.load(out / (b + suffix)).shape == (n, n)
    s = meta["summary"]
    assert s["帧数"] == 6 and s["仿射可用帧"] == 6 and s["主图"] in ("晶格校正", "刚性")
    want = [series.rotvec(g, ds["angles"][0]) for g in lab.G]
    for gv in np.array(meta["detail"]["gref"]):
        err = min(min(np.linalg.norm(gv - w), np.linalg.norm(gv + w)) for w in want)
        assert err < 0.02 * np.linalg.norm(gv)
