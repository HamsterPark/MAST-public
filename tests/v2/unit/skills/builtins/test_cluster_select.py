"""SelectPokedCluster 的合成测试：显式阈值、合取筛选、空间锚点与弃权。
不使用样品标定数据，缺参数时仍拒绝且不返回猜测工作点。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.skills.builtins.cluster_select import SelectPokedCluster  # noqa: E402

PX = 64
RANGE_M = 3.2e-8
OFFSET = (5.0e-8, 7.0e-8)
THRESH = dict(min_aspect=0.4, min_area_px=32, min_peak_height_m=200e-12,
              anchor_tolerance_m=3e-9)


def _write_sxm(path, arr, angle="0.000E+0"):
    ny, nx = arr.shape
    hdr = (":NANONIS_VERSION:\n2\n:SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
           ":REC_DATE:\n 01.01.2099\n:REC_TIME:\n12:00:00\n:BIAS:\n\t1.0E+0\n"
           f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
           f":SCAN_RANGE:\n{RANGE_M:>19.6E}{RANGE_M:>19.6E}\n"
           f":SCAN_OFFSET:\n{OFFSET[0]:>19.6E}{OFFSET[1]:>19.6E}\n"
           f":SCAN_ANGLE:\n{angle}\n:SCAN_DIR:\ndown\n"
           ":Z-CONTROLLER>SETPOINT:\n20.0000E-12\n:DATA_INFO:\n"
           "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
           "\t14\tZ\tm\tfwd\t9.000E-9\t0.000E+0\n:SCANIT_END:\n\n")
    f = arr.astype(np.float32).ravel()
    path.write_bytes(hdr.encode() + b"\x1a\x04" + struct.pack(">%df" % f.size, *f.tolist()))
    return path


def _px_to_m(px_x, px_y):
    dx = (px_x + 0.5) / PX * RANGE_M - RANGE_M * 0.5
    dy = RANGE_M * 0.5 - (px_y + 0.5) / PX * RANGE_M
    return OFFSET[0] + dx, OFFSET[1] + dy


def _disc(img, cy, cx, r, h):
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    img[(y - cy) ** 2 + (x - cx) ** 2 <= r * r] += h


@pytest.fixture()
def two_pokes(tmp_path):
    """两个真团簇(高)+ 一个吸附分子(圆但矮)+ 一条划痕(不圆)。"""
    rng = np.random.default_rng(0)
    img = 1e-9 + rng.normal(0, 3e-12, (PX, PX))
    _disc(img, 18, 18, 5, 8e-10)          # A
    _disc(img, 44, 46, 5, 9e-10)          # B
    _disc(img, 18, 46, 3, 6e-11)          # 吸附分子:圆但矮
    img[56, 4:60] += 5e-10                # 划痕:高但不圆
    return _write_sxm(tmp_path / "two.sxm", img), (18, 18), (44, 46)


def _run(path, **kw):
    return SelectPokedCluster().execute(None, {"scan_path": str(path),
                                               "polarity": "bright", **kw})


# ── 必填阈值:拒绝时把处方一起给 ────────────────────────────────────────

def test_missing_thresholds_are_refused_with_a_prescription(two_pokes):
    """只说「缺参数」是把问题丢回去 —— 拒绝的同时要说清**该怎么填**。"""
    path, _, _ = two_pokes
    res = _run(path)
    assert not res.success
    for k in ("min_aspect", "min_area_px", "min_peak_height_m", "anchor_tolerance_m"):
        assert k in res.data["missing_parameters"]
    assert res.data["suggested_operating_point"] == {}
    assert res.data["calibration_required"] is True
    assert "measured_box_raw" not in res.data
    assert "box_provenance" not in res.data
    assert "自己的完整数据集" in res.error


# ── 按锚点挑 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("which", [0, 1])
def test_it_picks_the_cluster_nearest_the_poke(two_pokes, which):
    """一帧里有两个真团簇 —— 挑哪个由**扎针坐标**决定,不是由大小决定。"""
    path, a, b = two_pokes
    cy, cx = (a, b)[which]
    ax, ay = _px_to_m(cx, cy)
    d = _run(path, near_x_m=ax, near_y_m=ay, **THRESH).data
    assert d["selected"] is not None, d["selected_reason"]
    assert d["distance_to_anchor_m"] < 3e-9
    assert d["anchor"]["source"] == "explicit"


def test_no_anchor_falls_back_to_frame_centre_and_says_so(two_pokes):
    path, _, _ = two_pokes
    d = _run(path, **THRESH).data
    assert d["anchor"]["source"] == "frame_centre"


# ── 弃权 ────────────────────────────────────────────────────────────────

def test_it_abstains_when_nothing_is_within_tolerance(two_pokes):
    """**「我找到的最近的东西」不是答案。**

    这是三态里最常被跳过的一步:够不着的时候把最近的交出去,看起来永远像成功。
    """
    path, _, _ = two_pokes
    far_x, far_y = _px_to_m(2, 2)
    d = _run(path, near_x_m=far_x, near_y_m=far_y, **THRESH).data
    assert d["selected"] is None
    assert "超过容差" in d["selected_reason"]
    # 但**距离仍然要报出来** —— 它是下一版容差的数据
    assert d["nearest_passing"]["distance_to_anchor_m"] > 3e-9


def test_it_abstains_when_the_conjunction_rejects_everything(two_pokes):
    """弃权 ≠「没有团簇」,而且要说清每个候选是卡在哪一条。"""
    path, a, _ = two_pokes
    ax, ay = _px_to_m(a[1], a[0])
    d = _run(path, near_x_m=ax, near_y_m=ay,
             **{**THRESH, "min_peak_height_m": 5e-9}).data   # 高得没人过得了
    assert d["selected"] is None
    assert "弃权" in d["selected_reason"]
    assert d["n_passed_conjunction"] == 0
    assert any(c["failed_on"] for c in d["candidates"]), "没说清是卡在哪一条"


def test_the_adsorbate_and_the_streak_are_both_rejected(two_pokes):
    """吸附分子(圆但矮)和划痕(高但不圆)—— 合取的两条边各挡一个。"""
    path, _, _ = two_pokes
    d = _run(path, **THRESH).data
    reasons = [f for c in d["candidates"] if c["failed_on"] for f in c["failed_on"]]
    assert any("peak" in r for r in reasons), "没有任何候选是因为矮被挡的"
    assert any("aspect" in r for r in reasons), "没有任何候选是因为不圆被挡的"


def test_unknown_angle_marks_the_coordinates_untrustworthy(tmp_path):
    """角度不可知 = 像素→米没验证过,而调用方要拿它去移动针尖。"""
    rng = np.random.default_rng(1)
    img = 1e-9 + rng.normal(0, 3e-12, (PX, PX))
    _disc(img, 30, 30, 5, 9e-10)
    p = _write_sxm(tmp_path / "bad.sxm", img, angle="nope")
    d = _run(p, **THRESH).data
    assert d["coords_trustworthy"] is False
    assert "没有验证过" in d["coords_warning"]


def test_the_prescription_is_actually_fillable(two_pokes):
    """调用方明确提供的 SI 参数必须能解析并执行；拒绝响应不得替调用方编造参数。"""
    from mast.agents._shared.skill_adapter import _coerce_si_params
    from mast.skills.builtins.cluster_select import SelectPokedCluster

    path, _, _ = two_pokes
    refused = _run(path)                       # 不给阈值 → 拒绝 + 处方
    assert not refused.success
    assert refused.data["suggested_operating_point"] == {}
    assert refused.data["calibration_required"] is True
    presc = {**THRESH, "min_peak_height_m": "200p", "anchor_tolerance_m": "3n"}

    meta = SelectPokedCluster().metadata()
    _out, errors = _coerce_si_params(meta, dict(presc))
    assert not errors, (
        f"拒绝报文开的处方,这个技能自己收不下:{errors}\n"
        f"处方是 {presc} —— 模型会原样照抄,然后被拒。")

    # 字符串直调与工具适配后的浮点值必须保持相同的选择结果。
    res = _run(path, **presc)
    adapted = _run(path, **_out)
    assert res.success, f"显式参数能解析却跑不通:{res.error}"
    assert adapted.success, adapted.error
    assert res.data["selected"] == adapted.data["selected"]
    assert res.data["n_passed_conjunction"] == adapted.data["n_passed_conjunction"]
