"""同次采集的空文件不能冒充完整帧；取文件层需识别空数据并回退到可用缓冲。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.skills.composite.prescan_check import PreScanCheck  # noqa: E402

_HEADER = (
    ":SCAN_PIXELS:\n{nx} {ny}\n"
    ":SCAN_OFFSET:\n0.000000E+0 0.000000E+0\n"
    ":SCAN_RANGE:\n{w:.6E} {h:.6E}\n"
    ":SCAN_ANGLE:\n0.000000E+0\n"
    ":SCAN_DIR:\ndown\n"
    ":DATA_INFO:\n"
    "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
    "\t14\tZ\tm\tboth\t1.0\t0.0\n"
    "\n:SCANIT_END:\n"
)


def _write_sxm(path: Path, frame: np.ndarray, *, size_m=(6e-8, 6e-8)) -> Path:
    ny, nx = frame.shape
    head = _HEADER.format(nx=nx, ny=ny, w=size_m[0], h=size_m[1])
    d = np.asarray(frame, dtype=">f4")
    path.write_bytes(head.encode() + b"\x1a\x04"
                     + d.tobytes() + np.ascontiguousarray(d[:, ::-1]).tobytes())
    return path


class _Skill(PreScanCheck):
    """只借 `_quality_from_saved_frame`；它靠 `self._executor` 拿文件路径。"""

    def __init__(self, path: str):
        super().__init__()

        class _Res:
            data = {"path": path}

        class _Ex:
            sub_results = {"latest_file": _Res()}

        self._executor = _Ex()


def test_an_empty_early_save_is_not_taken_as_the_frame(tmp_path):
    """整帧 NaN（还没扫到任何一行）⇒ **返回 None 回落**，不是当成一帧去判。

    没有这一条，调用方会拿到一个「0/256 行、判不了」的结论，
    而真正的那一帧其实就在隔壁文件里。
    """
    empty = np.full((256, 256), np.nan, dtype=np.float32)
    p = _write_sxm(tmp_path / "empty.sxm", empty)
    out = _Skill(str(p))._quality_from_saved_frame(6e-8)
    assert out is None, (
        f"空存盘被当成了「这一帧」，返回 {out!r} —— "
        "调用方会据此弃权，而这一帧其实量得出来（数据在同一次扫描的另一份存盘里）。")


def test_a_real_frame_is_still_used(tmp_path):
    """对照：有形貌的帧照常走判决路径，**这道拦截不许顺手把好帧也挡掉**。

    只有这一条在，上一条才不是「把判据关掉了」。
    """
    rng = np.random.default_rng(0)
    _y, x = np.mgrid[0:256, 0:256]
    frame = (2.36e-10 * (x > 128) + rng.normal(0, 5e-12, (256, 256))).astype(np.float32)
    p = _write_sxm(tmp_path / "real.sxm", frame)
    out = _Skill(str(p))._quality_from_saved_frame(6e-8)
    assert out is not None, "有形貌的正常帧被这道拦截误伤了"
    assert isinstance(out, dict)
