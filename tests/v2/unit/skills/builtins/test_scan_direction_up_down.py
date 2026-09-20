"""扫描方向参数必须同时传递到仪器调用和下游坐标解释。

测试核验 StartScan 下发明确方向，并且同一合成物理场景采用 up/down
两种存储方向时，提取与平区定位输出相同的物理坐标。
"""
from pathlib import Path

import numpy as np
import pytest

from mast.skills.builtins.cluster_extract import ExtractClusters
from mast.skills.builtins.flat_region import FindFlatRegion
from mast.skills.builtins.imaging import StartScan


# ── 替身 ────────────────────────────────────────────────────────────────────
class _Rec:
    def __init__(self, error="", return_value=None):
        self.error = error
        self.return_value = return_value


class _Ctx:
    """够 StartScan 走完的最小替身；只记下发过哪些调用。"""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def safe_call(self, method, *args, role="main", allow_on_abort=False):
        self.calls.append((method, args))
        if method == "Scan_PropsGet":
            # [continuous, bouncy, autosave, series, comment, [modules]]
            #
            # ⚠️ continuous 这一位原来写的是 **2**,那是 **SET** 表的「关」——
            # 而这是一条 **GET** 回包,GET 表里只有 0=关 / 1=开,2 不是任何东西。
            # 两张表不共用编码,`imaging.py` 的注释和
            # `test_imaging.test_set_and_get_do_not_share_one_encoding_table`
            # 都写着这件事,而这个替身还是把它们混了 —— 正说明混起来有多顺手。
            #
            # 当时没人发现,是因为下游那句判断是 `continuous == _GET_ON`(1),
            # 一个 2 于是安静地变成「不是开着 ⇒ 关着」。所以这个改动不只是修
            # 替身:`_continuous_state` 现在把表外的值也判成**读不到**(None),
            # 而不是判成「关着」。
            return _Rec(return_value=[0, b"", [0, 1, 0, "t", "", ["Z-Controller"]]])
        return _Rec()

    def args_for(self, method):
        return [a for m, a in self.calls if m == method]


# ── 1. 方向真的下发了，而且不猜 ──────────────────────────────────────────────
@pytest.mark.parametrize("given, expect_dir", [
    ({}, 0),                     # 不给 = down = 出厂行为，一个字节都不该变
    ({"direction": "down"}, 0),
    ({"direction": "up"}, 1),
    ({"direction": "UP"}, 1),    # 大小写不该改变物理动作
])
def test_direction_reaches_scan_action(given, expect_dir):
    ctx = _Ctx()
    StartScan().execute(ctx, dict(given))
    acts = ctx.args_for("Scan_Action")
    assert acts, "StartScan 根本没发 Scan_Action"
    assert acts[0] == (0, expect_dir), (
        f"参数 {given} 应当下发 Scan_Action(0, {expect_dir})，实得 {acts[0]}")


def test_a_bad_direction_is_refused_not_guessed():
    """猜错的代价是一整帧扫在错方向上，而文件头会**如实**记下它 ——
    于是后面每一处按方向归位的分析都跟着错，且没有任何一处会报错。"""
    ctx = _Ctx()
    r = StartScan().execute(ctx, {"direction": "sideways"})
    assert not r.success
    assert not ctx.args_for("Scan_Action"), "拒绝了却还是把扫描起了起来"


# ── 2. 同一场景、两种存储方向 ⇒ 同一个坐标 ──────────────────────────────────
def _make_sxm(path: Path, frame: np.ndarray, *, scan_dir: str,
              size_m=(6e-8, 6e-8), offset_m=(1e-7, -4e-8)) -> Path:
    """写一个最小可读 .sxm。

    ``frame`` 按**几何**朝向给（row 0 = 帧顶 = 最大 y）。本函数按 ``scan_dir``
    把它转成 Nanonis 的**采集顺序**再落盘 —— `up` 扫描第一条采到的线是帧底，
    所以文件里的 row 0 是帧底。这正是被测的那件事。
    """
    stored = np.flipud(frame) if scan_dir == "up" else frame
    ny, nx = stored.shape
    header = (
        ":SCAN_PIXELS:\n" f"{nx} {ny}\n"
        ":SCAN_OFFSET:\n" f"{offset_m[0]:.6E} {offset_m[1]:.6E}\n"
        ":SCAN_RANGE:\n" f"{size_m[0]:.6E} {size_m[1]:.6E}\n"
        ":SCAN_ANGLE:\n" "0.000000E+0\n"
        ":SCAN_DIR:\n" f"{scan_dir}\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.0\t0.0\n"
        "\n:SCANIT_END:\n"
    )
    d = np.asarray(stored, dtype=">f4")
    # 反扫块按**采集方向**存 = 左右镜像（.sxm 的反扫是从右往左采的）
    path.write_bytes(header.encode() + b"\x1a\x04"
                     + d.tobytes() + np.ascontiguousarray(d[:, ::-1]).tobytes())
    return path


def _blob_frame(n: int = 128) -> np.ndarray:
    """在非对称位置构造单个斑点，使任意上下或左右翻转都能被断言发现。"""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:n, 0:n]
    r0, c0 = n // 4, n // 3                      # row 32, col 42 —— 偏上偏左
    bump = 5e-10 * np.exp(-(((y - r0) ** 2 + (x - c0) ** 2) / (2 * 4.0 ** 2)))
    return bump + rng.normal(0.0, 4e-12, (n, n))


def test_up_and_down_report_the_same_cluster_position(tmp_path):
    frame = _blob_frame()
    p_down = _make_sxm(tmp_path / "d.sxm", frame, scan_dir="down")
    p_up = _make_sxm(tmp_path / "u.sxm", frame, scan_dir="up")

    got = {}
    for tag, p in (("down", p_down), ("up", p_up)):
        res = ExtractClusters().execute(None, {"scan_path": str(p)})
        assert res.success, f"{tag}: {res.error}"
        cl = (res.data or {}).get("clusters") or []
        assert cl, f"{tag}: 一个团簇都没找到，测试前提没成立"
        big = max(cl, key=lambda c: c.get("area_px") or 0)
        got[tag] = (big["x_m"], big["y_m"])

    dx = abs(got["up"][0] - got["down"][0]) * 1e9
    dy = abs(got["up"][1] - got["down"][1]) * 1e9
    assert dx < 1.0 and dy < 1.0, (
        f"同一个团簇，换个存储方向就跑了 ({dx:.2f}, {dy:.2f}) nm。\n"
        f"  down={got['down']}\n  up  ={got['up']}\n"
        f"y 差约半帧 ⇒ `up` 帧没有 flipud（row 0 是帧底，而 px_to_m 假定 row 0 是帧顶）。")


def test_up_and_down_report_the_same_flat_region(tmp_path):
    """同一件事在 FindFlatRegion 上再钉一次 —— 它有自己的一份取帧代码。

    「一处修好、同形状的其余处仍然裸着」是本仓反复发生的事，所以两处各钉各的，
    不共用一条测试。
    """
    frame = _blob_frame()
    p_down = _make_sxm(tmp_path / "d2.sxm", frame, scan_dir="down")
    p_up = _make_sxm(tmp_path / "u2.sxm", frame, scan_dir="up")

    out = {}
    for tag, p in (("down", p_down), ("up", p_up)):
        res = FindFlatRegion().execute(None, {"scan_path": str(p)})
        assert res.success, f"{tag}: {res.error}"
        out[tag] = (res.data["center_x_m"], res.data["center_y_m"])

    dx = abs(out["up"][0] - out["down"][0]) * 1e9
    dy = abs(out["up"][1] - out["down"][1]) * 1e9
    assert dx < 1.0 and dy < 1.0, (
        f"同一张图，换个存储方向挑出的平区差了 ({dx:.2f}, {dy:.2f}) nm。\n"
        f"  down={out['down']}\n  up  ={out['up']}")
