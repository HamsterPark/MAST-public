"""ExtractClusters：分割提取与品质判定分离，支持部分扫描帧。

用独立构造的斑点、噪声和未采集行验证分割、极性、预处理与统计输出。
所有阈值输入属于测试场景，不提供样品的已标定选择规则。
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

from mast.skills.builtins.cluster_extract import ExtractClusters  # noqa: E402

PX = 64
RANGE_M = 3e-8          # 30 nm 帧
OFFSET = (4.0e-8, 6.0e-8)


def _write_sxm(path, arr, *, offset=OFFSET, rng_m=RANGE_M, angle="0.000E+0"):
    ny, nx = arr.shape
    header = (
        ":NANONIS_VERSION:\n2\n"
        ":SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
        ":REC_DATE:\n 10.08.2026\n:REC_TIME:\n12:00:00\n"
        ":BIAS:\n\t1.000000E+0\n"
        f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
        f":SCAN_RANGE:\n{rng_m:>19.6E}{rng_m:>19.6E}\n"
        f":SCAN_OFFSET:\n{offset[0]:>19.6E}{offset[1]:>19.6E}\n"
        f":SCAN_ANGLE:\n{angle}\n"
        ":SCAN_DIR:\ndown\n"
        ":Z-CONTROLLER>SETPOINT:\n20.0000E-12\n"
        ":DATA_INFO:\n\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tfwd\t9.000E-9\t0.000E+0\n"
        ":SCANIT_END:\n\n"
    )
    # **NaN 必须原样写进去。** 第一版这里写的是 ``np.nan_to_num(arr, nan=0.0)``,
    # 于是「未扫区域」在文件里变成了 0.0,读回来处处有限 —— 三条「半张图」的测试
    # 会**全绿而什么也没测到**。真机上未扫行读回来就是 NaN,这里必须一致。
    flat = arr.astype(np.float32).ravel()
    blob = struct.pack(">%df" % flat.size, *flat.tolist())
    path.write_bytes(header.encode("utf-8") + b"\x1a\x04" + blob)
    return path


def _disc(img, cy, cx, r, height):
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    img[(y - cy) ** 2 + (x - cx) ** 2 <= r * r] += height


def _frame(*, seed=0):
    rng = np.random.default_rng(seed)
    return 1e-9 + rng.normal(0, 3e-12, (PX, PX))


def _run(path, **kw):
    return ExtractClusters().execute(None, {"scan_path": str(path), **kw})


# ── 一帧多个团簇 → 返回列表 ─────────────────────────────────────────────

@pytest.fixture()
def three_clusters(tmp_path):
    """三个团簇 + 一条线状伪影 —— 「一个图里扎好几次」的样子。"""
    img = _frame()
    _disc(img, 16, 16, 4, 6e-10)
    _disc(img, 16, 46, 4, 5e-10)
    _disc(img, 44, 30, 5, 7e-10)
    img[54, 5:60] += 4e-10          # 扫描线扰动:很长很细
    return _write_sxm(tmp_path / "multi.sxm", img)


def test_it_returns_a_list_not_a_verdict(three_clusters):
    """提取层不挑、不判 —— 一帧扎几次就返回几个。"""
    d = _run(three_clusters).data
    assert d["n_clusters"] >= 4
    assert isinstance(d["clusters"], list)
    assert "roundness_score" not in d and "is_round" not in d
    for c in d["clusters"]:
        for key in ("aspect", "area_px", "peak_height_pm", "x_m", "y_m"):
            assert key in c, f"判定要用的量 {key} 没报出来"


def test_the_streak_artefact_is_returned_not_filtered(three_clusters):
    """线状伪影**照样返回**,由判定层筛。

    提取层丢掉的东西,上层永远看不见 —— 而我们已经因为「判据自己吃掉了信息」
    栽过好几次。这里要求那条又长又细的东西**在列表里**,并且它的长宽比如实很低。
    """
    d = _run(three_clusters).data
    streaks = [c for c in d["clusters"] if c["aspect"] < 0.2]
    assert streaks, "线状伪影被提取层吃掉了"


def test_the_conjunction_separates_what_no_single_axis_does(three_clusters):
    """圆 且 大 且 高 —— 判定层的形状(在提取层的输出上跑得通)。"""
    cl = _run(three_clusters).data["clusters"]
    picked = [c for c in cl
              if c["aspect"] >= 0.35 and c["area_px"] >= 40
              and c["peak_height_pm"] >= 150]
    assert len(picked) == 3, f"应挑出 3 个圆团簇,得到 {len(picked)}"
    assert all(c["aspect"] > 0.5 for c in picked)


# ── 半张图是输入,不是错误 ───────────────────────────────────────────────

def test_a_half_scanned_frame_is_analysed_not_rejected(tmp_path):
    """未采集行不能进入平面拟合：只处理已完整采集的行，保留有效区域中的特征。"""
    img = _frame(seed=3)
    _disc(img, 12, 30, 5, 8e-10)
    img[26:, :] = np.nan                      # 扫到 26 行就停了
    p = _write_sxm(tmp_path / "half.sxm", img)
    res = _run(p)
    assert res.success, f"半张图被拒了:{res.error}"
    d = res.data
    assert d["frame"]["rows_scanned"] == 26, "已扫行数算错了"
    assert d["frame"]["rows_total"] == PX
    # 已扫区里那个团簇必须真的被找出来 —— 否则「没报错」也只是另一种失败
    assert any(c["aspect"] > 0.5 and c["peak_height_pm"] > 150
               for c in d["clusters"]), "半张图上的团簇没被提取出来"


def test_the_in_progress_row_is_excluded(tmp_path):
    """只有整行有效才计入已采集区域；混有 NaN 的在采行不能污染全帧拟合。"""
    img = _frame(seed=4)
    _disc(img, 10, 32, 5, 9e-10)
    img[20, 30:] = np.nan                     # 第 20 行扫了一半
    img[21:, :] = np.nan
    assert int(np.isfinite(img).all(axis=1).sum()) == 20, "构造失效"
    assert int(np.isfinite(img).any(axis=1).sum()) == 21, "构造失效"

    p = _write_sxm(tmp_path / "partial_row.sxm", img)
    res = _run(p)
    assert res.success, f"带半行的帧被拒了:{res.error}"
    # 关键:算的是 20 不是 21。多算那一行 → plane_subtract 整幅 NaN → 整个技能失败。
    assert res.data["frame"]["rows_scanned"] == 20, (
        "把那条正在扫的半行也算成已扫了 —— 一个 NaN 就会让平场整幅返回 NaN")


def test_a_cluster_on_the_scan_front_is_flagged_not_dropped(tmp_path):
    """贴着扫描前沿的团簇被切了一半,几何量是错的 —— **标记,不过滤**。"""
    img = _frame(seed=5)
    _disc(img, 24, 30, 6, 9e-10)              # 圆心正好在扫描停止处附近
    img[26:, :] = np.nan
    p = _write_sxm(tmp_path / "front.sxm", img)
    d = _run(p).data
    assert d["frame"]["rows_scanned"] < d["frame"]["rows_total"]
    clipped = [c for c in d["clusters"] if c["touches_unscanned"]]
    assert clipped, "跨在扫描前沿上的团簇没有被标出来"


# ── 坐标 ────────────────────────────────────────────────────────────────

def test_a_centred_cluster_lands_on_the_scan_offset(tmp_path):
    """帧正中的团簇,真实坐标就是 scan_offset —— 换算的定标点。"""
    img = _frame(seed=6)
    _disc(img, PX // 2, PX // 2, 5, 8e-10)
    p = _write_sxm(tmp_path / "centre.sxm", img)
    d = _run(p).data
    big = max(d["clusters"], key=lambda c: c["area_px"])
    assert big["x_m"] == pytest.approx(OFFSET[0], abs=RANGE_M / PX)
    assert big["y_m"] == pytest.approx(OFFSET[1], abs=RANGE_M / PX)


def test_an_unparseable_angle_is_reported_as_unknown(tmp_path):
    """``parse_xy_meta`` 在角度解析失败时仍然给 0.0,而「是不是转过」的判据是
    ``abs(angle) > 1`` —— 0.0 恰好让它不触发,兜底值落在「没什么可担心的」那侧。

    所以 ``angle_known`` 必须透出来:一帧真的转过的图不能被当成轴对齐,
    因为调用方要拿这个坐标去移动针尖。
    """
    img = _frame(seed=7)
    _disc(img, 30, 30, 5, 8e-10)
    p = _write_sxm(tmp_path / "badangle.sxm", img, angle="not-a-number")
    d = _run(p).data
    assert d["frame"]["angle_known"] is False


# ── 过滤要可见 ──────────────────────────────────────────────────────────

def test_min_area_only_shortens_the_list_and_says_so(three_clusters):
    """min_area_px 控制输出列表长度，不代表分辨率；同时报告过滤前后的数量。"""
    d = _run(three_clusters, min_area_px=1).data
    d2 = _run(three_clusters, min_area_px=50).data
    assert d["n_raw_components"] == d2["n_raw_components"]
    assert d2["n_clusters"] <= d["n_clusters"]
    assert d2["min_area_px"] == 50


def test_polarity_auto_reports_the_evidence_for_both_sides(three_clusters):
    """亮暗极性不能写死；自动选择时应提供双方证据，调用方也可显式指定。"""
    d = _run(three_clusters).data
    assert d["polarity_used"] in ("bright", "dark")
    assert set(d["polarity_evidence"]) == {"bright", "dark"}
    for side in ("bright", "dark"):
        assert "largest_aspect" in d["polarity_evidence"][side]


def test_the_oral_size_bound_flags_but_never_rejects(tmp_path):
    """「500 pm 扎出的团簇一般不超过 3 nm」是**经验范围不是硬上界** ——
    真值里 `_0053`#1 等效直径 4.13 nm、#2 3.17 nm,两个都超了。
    所以它只打 ``size_plausible=False``,绝不把团簇拿掉。"""
    img = _frame(seed=8)
    _disc(img, 32, 32, 14, 9e-10)             # 大到超过 3 nm 等效直径
    p = _write_sxm(tmp_path / "big.sxm", img)
    d = _run(p).data
    big = max(d["clusters"], key=lambda c: c["area_px"])
    assert big["equiv_diameter_nm"] > 3.0
    assert big["size_plausible"] is False
    assert big in d["clusters"], "超出经验范围的团簇被拿掉了"


def test_a_missing_channel_fails_with_the_candidate_list(tmp_path):
    img = _frame(seed=9)
    _disc(img, 30, 30, 5, 8e-10)
    p = _write_sxm(tmp_path / "ch.sxm", img)
    res = _run(p, channel="Current")
    assert not res.success and "Current" in res.error
    assert res.data["available_channels"] == ["Z"]


# ── RAW 是默认;逐行平场被拒绝 ──────────────────────────────────────────

def test_raw_is_the_default(three_clusters):
    """默认采用 RAW 数据，避免未经请求的预处理改变分割结果；全局调平可显式选择。"""
    d = _run(three_clusters).data
    assert d["leveling_used"] == "none"


def test_per_row_leveling_is_refused_with_the_measurement(three_clusters):
    """逐行拟合会被该行的宽特征抬高，减除拟合后可能削弱特征并产生伪影。
    因此预处理选项不提供逐行多项式操作。"""
    for bad in ("poly1", "poly2", "median", "line_by_line"):
        res = _run(three_clusters, level=bad)
        assert not res.success, f"level={bad!r} 被接受了"
        assert "逐行" in res.error and "伪影" in res.error
        assert "level='none'" in res.error and "'plane'" in res.error


def test_the_global_plane_is_offered_and_reported(three_clusters):
    """全局平面减除作为显式选项，输出必须记录是否使用，以便比较处理口径一致的数据。"""
    d = _run(three_clusters, level="plane").data
    assert d["leveling_used"] == "plane"
    assert d["n_clusters"] >= 3


def test_the_tilt_is_measured_and_reported(three_clusters):
    """报告残余斜面用于诊断；没有给定标定门槛时，不应仅凭该数值否决提取结果。"""
    d = _run(three_clusters).data
    assert d["tilt_pp_pm"] >= 0.0
    assert d["tilt_over_sigma"] is not None
    assert d["tilt_warning"] is None          # 这帧是平的
    hot = _run(three_clusters, tilt_warn_ratio=0.01).data
    assert hot["tilt_warning"] and "没调平" in hot["tilt_warning"]


def test_the_discredited_circularity_field_is_gone_not_renamed(three_clusters):
    """旧 circularity 字段应从输出中消失，防止旧阈值被误用于新的指标。
    边计数周长对网格方向敏感，不能把该定义直接当作连续几何中的圆形度。"""
    d = _run(three_clusters, min_area_px=1).data
    for c in d["clusters"]:
        assert "circularity" not in c, (
            "circularity 还在输出里 —— 它与新判据不可比、不可换算,"
            "旧阈值 0.65 会被照着用")
        assert "perimeter_px" not in c


def test_every_cluster_carries_a_readable_roundness_or_a_reason(three_clusters):
    """新判据要么给一个**有物理含义**的数,要么说判不了 —— 不给凑出来的数。"""
    d = _run(three_clusters, min_area_px=1).data
    # ⚠️ 只按面积挑会把那条 55 px 的**线状伪影**也算成「圆盘」——
    # 第一版就是这么红的(它被读成 0.211,读得完全正确)。
    big = [c for c in d["clusters"] if c["area_px"] >= 40 and c["aspect"] > 0.8]
    assert len(big) >= 3, f"这一帧的三个合成圆盘没被挑齐:{[c['area_px'] for c in big]}"
    for c in big:
        q = c["equivalent_axis_ratio"]
        assert q is not None and c["roundness_undecidable"] is None
        assert 0.0 < q <= 1.0
        # 合成的是干净圆盘 —— 必须被读成圆的(旧判据在这里读 0.53–0.57 < 0.65)。
        # 0.80 而不是 0.90:这几个盘只有 50–80 px,完美圆盘在**最差亚像素相位**
        # 下也就读到 0.844(实测表见 tests/v2/vision/test_roundness.py)。
        # 卡到 0.90 就是在钉运气。
        assert q >= 0.80, f"合成圆盘({c['area_px']} px)被读成轴比 {q:.3f}"
    tiny = [c for c in d["clusters"] if c["area_px"] < 20]
    for c in tiny:
        assert c["equivalent_axis_ratio"] is None
        assert c["roundness_undecidable"], "太小的团簇必须带上「判不了」的理由"


def test_the_streak_is_read_as_far_less_round_than_the_discs(three_clusters):
    """线状伪影 vs 圆盘:新判据必须把它们**明显**分开。

    钉的是方向和量级,不是某个具体读数。
    """
    d = _run(three_clusters, min_area_px=1).data
    qs = {c["rank"]: c["equivalent_axis_ratio"] for c in d["clusters"]}
    streak = min((c for c in d["clusters"] if c["aspect"] < 0.2),
                 key=lambda c: c["aspect"], default=None)
    assert streak is not None, "这一帧里没找到线状伪影"
    discs = [c["equivalent_axis_ratio"] for c in d["clusters"]
             if c["area_px"] >= 40 and c["aspect"] > 0.8]
    assert discs and streak["equivalent_axis_ratio"] is not None
    assert streak["equivalent_axis_ratio"] < min(discs) - 0.3, (
        f"线状伪影读 {streak['equivalent_axis_ratio']:.3f},圆盘读 {discs} —— "
        "两者没有拉开距离")
