# -*- coding: utf-8 -*-
"""从一个 .sxm 路径取出**一帧两个方向 + 工作点**的共享入口。

已有的 ``atomic_multiframe._load_one`` 只给正扫 Z 与像素标度，够 ``AnalyseSlowDrift``
与 ``AssessAtomicConsistency`` 用。本模块的三个使用者还需要反扫、电流通道、
偏压/设定点、线速度与 Z 增益，所以另开一个而不是去改那一个 ——
改它会挪动两个已验证技能的取数路径。

IO 一律走 :func:`mast.io.nanonis_files.sxm_oriented_frames`：反扫是镜像存储的、
``:SCAN_DIR: up`` 的第一行是帧底，这两件事只在那一处修正，别在判据层重做。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["LoadedFrame", "load_frame", "split_paths"]


@dataclass(frozen=True)
class LoadedFrame:
    """一帧的图像与工作点。``error`` 非空时其余字段都别用。"""

    error: str = ""
    forward: object = None            # ndarray | None（米）
    backward: object = None           # ndarray | None（米）
    current_forward: object = None     # ndarray | None（安培）
    current_backward: object = None
    nm_per_px: Optional[float] = None
    width_nm: Optional[float] = None
    bias_v: Optional[float] = None
    setpoint_a: Optional[float] = None
    scan_dir: str = ""
    scan_angle_deg: Optional[float] = None
    #: 每行**单程**时间（秒），文件头 ``scan_time`` 的第一个数。
    line_time_s: Optional[float] = None
    #: 正扫线速度（m/s）。文件头有 ``scan>speed_forw._(m/s)`` 就用它，
    #: 否则用 ``width / line_time`` 推 —— 两者都拿不到就是 ``None``，**不猜**。
    speed_m_s: Optional[float] = None
    i_gain_m_s: Optional[float] = None
    p_gain_m: Optional[float] = None
    rec_time: str = ""
    name: str = ""


def split_paths(raw: str) -> list[str]:
    """逗号 / 换行分隔的一串路径。与 ``atomic_multiframe._split_paths`` 同义。"""
    if not raw:
        return []
    parts: list[str] = []
    for chunk in str(raw).replace("\r", "\n").split("\n"):
        parts.extend(p.strip() for p in chunk.split(","))
    return [p for p in parts if p]


def _f(raw) -> Optional[float]:
    """文件头里的一个数（可能带单位后缀）。读不出就是 ``None``，不是 0.0。"""
    if raw is None:
        return None
    try:
        return float(str(raw).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def load_frame(path: str, channel: str = "Z") -> LoadedFrame:
    """读一个 .sxm，返回定向后的两个方向 + 工作点。**永不抛异常。**"""
    from pathlib import Path

    if not path:
        return LoadedFrame(error="没有给 scan_path")
    if not Path(path).exists():
        return LoadedFrame(error=f"文件不存在: {path}")
    try:
        from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    except ImportError as exc:  # noqa: BLE001 — 独立 API 模式下 io 可能没装
        return LoadedFrame(error=f"缺依赖: {exc}")
    try:
        scan = read_sxm(path)
    except Exception as exc:  # noqa: BLE001
        return LoadedFrame(error=f".sxm 读取失败: {exc}")

    topo = sxm_oriented_frames(scan, channel)
    if topo.get("forward") is None:
        return LoadedFrame(error=f"没有通道 {channel!r} 的正扫数据")
    nmpp = topo.get("nm_per_px")
    if not nmpp:
        return LoadedFrame(error="文件头里没有像素标度")
    cur = sxm_oriented_frames(scan, "Current")
    header = scan.get("header") or {}

    line_t = _f(header.get("scan_time"))
    width_nm = topo.get("width_nm")
    speed = _f(header.get("scan>speed_forw._(m/s)"))
    if speed is None and line_t and width_nm:
        speed = float(width_nm) * 1e-9 / float(line_t)

    return LoadedFrame(
        forward=topo.get("forward"), backward=topo.get("backward"),
        current_forward=cur.get("forward"), current_backward=cur.get("backward"),
        nm_per_px=float(nmpp), width_nm=width_nm,
        bias_v=topo.get("bias_v"), setpoint_a=topo.get("setpoint_a"),
        scan_dir=str(topo.get("scan_dir") or ""),
        scan_angle_deg=_f(header.get("scan_angle")),
        line_time_s=line_t, speed_m_s=speed,
        i_gain_m_s=_f(header.get("z-controller>i_gain")),
        p_gain_m=_f(header.get("z-controller>p_gain")),
        rec_time=str(topo.get("rec_time") or ""),
        name=Path(path).name,
    )
