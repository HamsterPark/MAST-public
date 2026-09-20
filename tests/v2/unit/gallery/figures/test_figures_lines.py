# -*- coding: utf-8 -*-
"""拉线谱：站位推断（设计文档 D17，纯函数）与四张图（T25 / T35）。"""
from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from mast.gallery.figures import lines, service


def _blocks(*, drift=0.08, step=0.2, n=6, angle=30.0, reverse_ref=False, repeat_ref=False, seed=1):
    """纯输入：A（参照）、B（逆序、整体漂移 drift）、C（三个站位 + 一条落在 1.5 个步长处）。"""
    rng = np.random.default_rng(seed)
    u = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    perp = np.array([-u[1], u[0]])
    p0 = np.array([5.0, 3.0])
    truth: dict[str, int] = {}

    def spec(sid, station, t, along=0.0, keep=True):
        p = p0 + u * (station * step + along + rng.normal(0, 0.003)) + perp * rng.normal(0, 0.003)
        if keep:
            truth[sid] = station
        return {"id": sid, "x": float(p[0]), "y": float(p[1]), "t": float(t)}

    ref = list(range(n))[::-1] if reverse_ref else list(range(n))
    if repeat_ref:
        ref = ref[:3] + [ref[2]] + ref[3:]                       # 同一站位重测一次
    A = [spec("a%d" % i, st, 100 + i) for i, st in enumerate(ref)]
    B = [spec("b%d" % i, st, 1000 + i, along=drift) for i, st in enumerate(reversed(range(n)))]
    C = [spec("c0", 0, 2000), spec("c1", 1, 2001), spec("c_out", 1.5, 2002, keep=False), spec("c3", 2, 2003)]
    blocks = [{"sid": sid, "label": sid, "spectra": [dict(s, order=o) for o, s in enumerate(sp)]}
              for sid, sp in (("SA", A), ("SB", B), ("SC", C))]
    return blocks, truth, u


def _stations(plan) -> dict:
    return {a["id"]: a["station"] for b in plan["blocks"] for a in b["assigned"]}


def _block(plan, sid) -> dict:
    return next(b for b in plan["blocks"] if b["series"] == sid)


def test_stations_follow_the_geometry_through_drift_and_reversed_blocks():
    blocks, truth, u = _blocks()
    plan = lines.infer_stations(blocks)
    assert plan["ok"] and plan["n_stations"] == 6 and plan["reference"] == "SA"
    assert plan["step_nm"] == pytest.approx(0.2, rel=0.02)
    assert float(np.dot(plan["direction"], u)) == pytest.approx(1.0, abs=1e-3)
    # B 整体漂了 0.4 个步长（> 对不上门槛 0.35）：只有做了漂移对齐才归得上
    assert 0.08 > lines.MATCH_FRAC * 0.2
    assert _stations(plan) == truth
    assert _block(plan, "SB")["offset_nm"] == pytest.approx(0.08, abs=0.01)
    assert _block(plan, "SC")["unmatched"] == ["c_out"]
    assert [a["order"] for a in _block(plan, "SC")["assigned"]] == [0, 1, 3]
    assert any("没有归入任何站位" in w for w in plan["warnings"])


def test_a_full_block_drifted_by_more_than_one_step_is_placed_by_its_two_ends():
    """合成区组漂移超过一个步长时，只在 ±半个步长里搜索会造成整组错位。"""
    blocks, truth, _u = _blocks(drift=0.28)                    # 1.4 个步长
    plan = lines.infer_stations(blocks)
    assert _stations(plan) == truth
    assert _block(plan, "SB")["offset_nm"] == pytest.approx(0.28, abs=0.01)
    assert not any("「SB」" in w for w in plan["warnings"])


def test_a_partial_block_takes_the_smallest_drift_warns_and_can_be_shifted():
    blocks, truth, _u = _blocks()
    plan = lines.infer_stations(blocks)
    # C 只覆盖站位 0–2：整步偏移靠位置定不下来 ⇒ 取漂移最小的并提示 station_shift
    assert any("「SC」" in w and "station_shift" in w for w in plan["warnings"])
    got = _stations(lines.infer_stations(blocks, shifts={"SC": 2}))
    assert [got[i] for i in ("c0", "c1", "c3")] == [2, 3, 4]
    assert {k: v for k, v in got.items() if k[0] != "c"} == {k: v for k, v in truth.items() if k[0] != "c"}
    assert lines.station_shifts({"station_shift": {"SC": "2", "SB": 0, "SA": "x"}}) == {"SC": 2}


def test_station_zero_is_where_the_reference_series_started():
    blocks, truth, u = _blocks(reverse_ref=True)
    plan = lines.infer_stations(blocks)
    n = plan["n_stations"]
    assert plan["reference"] == "SA" and n == 6
    assert _stations(plan) == {k: n - 1 - v for k, v in truth.items()}
    assert float(np.dot(plan["direction"], u)) == pytest.approx(-1.0, abs=1e-3)


def test_a_repeat_at_one_station_does_not_collapse_the_step():
    blocks, truth, _u = _blocks(repeat_ref=True)
    plan = lines.infer_stations(blocks)
    assert plan["reference"] == "SA" and plan["step_nm"] == pytest.approx(0.2, rel=0.02)
    assert _stations(plan) == truth


def test_degenerate_inputs_do_not_raise():
    assert lines.infer_stations([])["ok"] is False
    empty = lines.infer_stations([{"sid": "S", "label": "", "spectra": []}])
    assert empty["ok"] is False and empty["warnings"]
    same = [{"id": i, "x": 1.0, "y": 2.0, "t": float(o), "order": o} for o, i in enumerate("ab")]
    one = lines.infer_stations([{"sid": "S", "label": "", "spectra": same}])
    assert one["ok"] and one["n_stations"] == 1
    assert [a["station"] for a in one["blocks"][0]["assigned"]] == [0, 0]


def test_block_labels_are_the_series_names_minus_the_common_part():
    line, labels = lines.names_common(["测试线 · 区组0 · 主线", "测试线 · 区组1 · 主线", "测试线 · 区组2 · 主线（中断）"])
    assert line == "测试线 · 主线"
    assert labels == ["区组0", "区组1", "区组2 （中断）"]


def test_sts_lines_job_writes_four_figures_and_the_inferred_assignment(lab):
    ds = lab.line_dataset()
    lab.build()
    lab.mark(series=ds["series"])
    res = service.run_job("sts_lines", series=list(ds["series"]),
                          options={"station_marks": {"3": "D"}, "bad_from": {"SC": 3}})
    assert res["phase"] == "done" and res["errors"] == [], res
    [key] = res["made"]
    meta = lab.figure_meta(key)
    base = meta["base"]
    assert key == "sts_lines/" + base and base == "测试线_L9 过缺陷"
    assert [f[len(base):] for f in meta["files"]] == ["_比较_热图.png", "_比较_瀑布.png", "_均值_热图.png", "_均值_瀑布.png"]
    with Image.open(lab.state / "figures" / "sts_lines" / meta["files"][0]) as im:
        assert im.size == (3200, 1800)

    rows = {r["id"]: r for r in meta["detail"]["rows"]}
    assert {i: r["idx"] for i, r in rows.items()} == ds["truth"]
    plan = meta["detail"]["plan"]
    assert {b["series"]: b["unmatched"] for b in plan["blocks"]}["SC"] == [ds["outlier"]]
    # 剔除 = C 从第 4 条（order 3）起针尖在变 + D 整组评为排除；删去不画、不进均值
    assert {i for i, r in rows.items() if r["excluded"]} == {ds["C"][3]} | set(ds["D"])
    assert rows[ds["C"][3]]["bad"] and not rows[ds["D"][0]]["bad"]
    assert meta["detail"]["per_station_n"] == [3, 3, 2, 2, 2, 2]
    assert "#2（针尖在变）" in meta["detail"]["drop_note"]
    assert "区组3 整组 2 条（系列评为排除）" in meta["detail"]["drop_note"]
    s = meta["summary"]
    assert (s["区组"], s["站位数"], s["谱"], s["剔除"], s["对不上"], s["dI/dV"]) == (3, 6, 14, 3, 1, "LI Demod 1 Y")
    assert meta["series"] == ["SA", "SB", "SC", "SD"]
    assert set(meta["ids"]) == set(ds["truth"]) - {ds["C"][3]} - set(ds["D"])

    again = lines.plan_from_store(list(ds["series"]))
    assert again["line_name"] == "测试线 · L9 过缺陷"
    skip = ("line_name", "degraded", "detail")
    assert {k: v for k, v in again.items() if k not in skip} == {k: v for k, v in plan.items() if k not in skip}
