# -*- coding: utf-8 -*-
"""单根谱拼接（设计文档 §10.1，T30 / T31）。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mast.gallery import marks as gmarks
from mast.gallery.figures import service, stitch


def test_three_inside_points_use_the_mean_ratio_not_the_extrapolation():
    """T30：门槛是代码里的 ≥ 3 点。这组数据刚好 3 点重叠，接缝外侧又弯了 —— 按外推会得到另一个数。"""
    V = np.round(np.linspace(-1.0, 1.0, 41), 10)
    core = {"V": V, "D": 1 + V}
    Vp = np.round(np.linspace(0.9, 1.5, 13), 10)
    piece = {"V": Vp, "D": (1 + Vp) / 3 + np.where(Vp > 1.0 + 1e-9, 20 * (Vp - 1.0) ** 2, 0.0)}
    fac, how = stitch.factor_to_core(core, piece, "D")
    assert fac == pytest.approx(3.0, rel=1e-9)
    assert "内侧 3 点均值比" in how
    extrapolated = stitch.edge_val(core["V"], core["D"], 1.0, -1) / stitch.edge_val(piece["V"], piece["D"], 1.0, 1)
    assert abs(extrapolated - 3.0) > 0.3


def test_segments_that_only_touch_are_extrapolated_five_points_each_side():
    V = np.linspace(-1.0, 1.0, 41)
    Vp = np.linspace(-2.0, -1.0, 21)
    fac, how = stitch.factor_to_core({"V": V, "D": 2 + V}, {"V": Vp, "D": (2 + Vp) / 4}, "D")
    assert fac == pytest.approx(4.0, rel=1e-6) and "两侧各 5 点外推到接缝" in how


def _mark_singles(lab, ds) -> None:
    items = {i: {"r": 2, "ts": 1} for i in ds["core"] + [ds["up"], ds["down"], ds["psd"], ds["member"]]}
    items[ds["core"][0]] = {"r": 2, "ts": 1, "anchor": {"id": ds["anchor"]}}
    lab.mark(items=items, series={"SM": {"name": "别的系列", "ids": [ds["member"]], "ts": 1}})


def test_stitch_by_directory_skips_non_bias_spectra_and_reports_them(lab):
    ds = lab.stitch_dataset()
    doc = lab.build()
    _mark_singles(lab, ds)
    groups = stitch.single_groups(doc, gmarks.load_marks())
    assert groups == [("SPM/2001/200109/20010911", ds["core"] + [ds["up"], ds["down"], ds["psd"]])]

    res = service.run_job("sts_stitch", options={})
    assert res["phase"] == "done", res
    assert res["made"] == ["sts_stitch/单根谱拼接_20010911"]
    assert [e["id"] for e in res["errors"]] == [ds["psd"]] and "不是偏压谱" in res["errors"][0]["why"]
    meta = lab.figure_meta(res["made"][0])
    d = meta["detail"]
    assert d["core"] == ds["core"] and d["li"] is True
    seg = {s["id"]: s for s in d["segments"]}
    assert seg[ds["core"][1]]["role"] == "core_member"
    assert seg[ds["up"]]["factor"] == pytest.approx(3.0, rel=1e-3) and "内侧 3 点" in seg[ds["up"]]["how"]
    assert seg[ds["up"]]["factor_I"] == pytest.approx(3.0, rel=1e-3)
    assert seg[ds["down"]]["factor"] == pytest.approx(2.0, rel=0.05) and "外推" in seg[ds["down"]]["how"]
    assert d["kappa"] is None
    assert seg[ds["up"]]["theory"] is None
    assert [s["id"] for s in d["skipped"]] == [ds["psd"]]
    s = meta["summary"]
    assert (s["段数"], s["跳过"], s["核心"]) == (4, 1, "00001+00002")
    assert s["×k 00005"] == pytest.approx(3.0, rel=1e-2)
    assert ds["member"] not in meta["ids"] and ds["psd"] not in meta["ids"]


def test_kappa_is_a_parameter_and_explicit_ids_name_their_range(lab):
    ds = lab.stitch_dataset()
    lab.build()
    _mark_singles(lab, ds)
    res = service.run_job("sts_stitch", ids=ds["core"] + [ds["up"], ds["psd"]], options={"kappa_per_nm": 5.0})
    assert res["phase"] == "done", res
    assert res["made"] == ["sts_stitch/单根谱拼接_20010911_00001-00005"]
    assert [e["id"] for e in res["errors"]] == [ds["psd"]]
    seg = {s["id"]: s for s in lab.figure_meta(res["made"][0])["detail"]["segments"]}
    assert seg[ds["up"]]["theory"] == pytest.approx(math.exp(2 * 5.0 * 0.05), rel=1e-6)
