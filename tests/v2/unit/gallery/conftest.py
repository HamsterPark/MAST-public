# -*- coding: utf-8 -*-
"""数据图库测试的共用件。

合成文件按**真文件格式**写（头部键名、表格、结束标记、大端 float32、采集顺序），不借用别的
测试文件里的私有函数 —— 那些函数改了形状，这里的测试会安静地测到别的东西。

方向约定：本文件的所有「图」参数都是**定向后**的（行 0 = 帧的上沿）。写 .sxm 时按采集顺序
落盘：``scan_dir="up"`` 的第一行是下沿，反扫块左右镜像存储 —— 读回来要经过
``sxm_oriented_frames`` 才正过来。只有这样，「上沿亮带出现在缩略图顶部」才是一个能抓住
翻转缺陷的判据。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _no_gallery_build_left_running():
    """后台构建线程不许活过测试：env 复位之后它若还在跑，就可能写到别的地方。"""
    yield
    from mast.gallery import service

    service._reset_for_tests()


@pytest.fixture()
def gallery_dir(tmp_path, monkeypatch) -> Path:
    """本测试的图库状态目录（不预先创建）。与 tests/v2/conftest.py 的 autouse 同一个位置。"""
    d = tmp_path / "gallery"
    monkeypatch.setenv("MAST_GALLERY_DIR", str(d))
    return d


def write_sxm(path, z_top, *, scan_dir: str = "down", extra: dict | None = None,
              rec_date: str = "10.09.2001", rec_time: str = "15:07:12", acq_s: float = 60.0,
              range_nm=(5.0, 5.0), offset_nm=(0.0, 0.0), angle: float = 0.0, bias: float = 1.0,
              setpoint_a: float = 3e-10, both: bool = True, mtime_ns: int | None = None) -> Path:
    """写一个 .sxm。``z_top`` 与 ``extra`` 里的数组都是定向后（行 0 = 上沿）的，单位 SI。"""
    chans = [("Z", "m", np.asarray(z_top, dtype=float))]
    chans += [(name, "A", np.asarray(a, dtype=float)) for name, a in (extra or {}).items()]
    ny, nx = chans[0][2].shape
    lines = [
        ":NANONIS_VERSION:", "2",
        ":SCANIT_TYPE:", "              FLOAT            MSBFIRST",
        ":REC_DATE:", f" {rec_date}",
        ":REC_TIME:", rec_time,
        ":REC_TEMP:", "      290.0000000000",
        ":ACQ_TIME:", f"       {acq_s:.1f}",
        ":SCAN_PIXELS:", f"{nx:>10d}{ny:>10d}",
        ":SCAN_TIME:", "             6.400E-1             6.400E-1",
        ":SCAN_RANGE:", f"{range_nm[0] * 1e-9:>22.6E}{range_nm[1] * 1e-9:>22.6E}",
        ":SCAN_OFFSET:", f"{offset_nm[0] * 1e-9:>22.6E}{offset_nm[1] * 1e-9:>22.6E}",
        ":SCAN_ANGLE:", f"{angle:>20.3E}",
        ":SCAN_DIR:", scan_dir,
        ":BIAS:", f"{bias:.6E}",
        ":Z-CONTROLLER:", "\tName\ton\tSetpoint\tP-gain\tI-gain\tT-const",
        f"\tlog Current\t1\t{setpoint_a:.3E} A\t3.000E-12 m\t5.000E-8 m/s\t6.000E-5 s",
        ":COMMENT:", "",
        ":Z-Controller>Setpoint:", f"{setpoint_a:g}",
        ":DATA_INFO:", "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset",
    ]
    for i, (name, unit, _a) in enumerate(chans):
        lines.append(f"\t{i}\t{name}\t{unit}\t{'both' if both else 'fwd'}\t1.000E+0\t0.000E+0")
    lines += [":SCANIT_END:", "", ""]
    blob = bytearray()
    for _name, _unit, a in chans:
        stored = a[::-1] if scan_dir == "up" else a          # 采集顺序：up 的第一行是下沿
        blob += np.ascontiguousarray(stored, dtype=">f4").tobytes()
        if both:
            blob += np.ascontiguousarray(stored[:, ::-1], dtype=">f4").tobytes()  # 反扫块镜像存储
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes("\r\n".join(lines).encode("latin-1") + b"\x1a\x04" + bytes(blob))
    if mtime_ns is not None:
        os.utime(p, ns=(mtime_ns, mtime_ns))
    return p


def write_dat(path, columns: list[str], data, *, header: dict | None = None,
              mtime_ns: int | None = None) -> Path:
    hdr = {"Experiment": "bias spectroscopy", "Saved Date": "10.09.2001 19:06:08",
           "X (m)": "-66.9234E-9", "Y (m)": "208.799E-9", "Z (m)": "-81.9519E-9",
           "Z offset (m)": "150E-12", "Start time": "10.09.2001 19:04:22"}
    hdr.update(header or {})
    lines = [f"{k}\t{v}\t" for k, v in hdr.items()] + ["", "[DATA]", "\t".join(columns)]
    for row in np.asarray(data, dtype=float):
        lines.append("\t".join(f"{x:.7E}" for x in row))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # write_bytes，不是 write_text：Windows 上文本模式会把 "\r\n" 再翻成 "\r\r\n"，
    # 读回来每行之间多出空行 —— [DATA] 后面那一行（列名）就成了空行。
    p.write_bytes(("\r\n".join(lines) + "\r\n").encode("latin-1"))
    if mtime_ns is not None:
        os.utime(p, ns=(mtime_ns, mtime_ns))
    return p


def write_3ds(path, *, nx: int, ny: int, npts: int, channels: dict, have: int | None = None,
              v0: float = 2.0, v1: float = -2.5, w_nm: float = 3.0, h_nm: float = 3.0,
              z_param=None, start: str = "11.09.2001 15:31:26",
              end: str = "11.09.2001 19:09:34") -> Path:
    """``channels[name]`` 形状 (ny, nx, npts)，**行 0 = 视野下沿**（与 .3ds 的点序一致）。"""
    names = list(channels)
    fixed = ["Sweep Start", "Sweep End"]
    exp = ["X (m)", "Y (m)", "Z (m)"]
    head = [
        f'Grid dim="{nx} x {ny}"',
        f"Grid settings=0.000000E+0;0.000000E+0;{w_nm * 1e-9:E};{h_nm * 1e-9:E};0.000000E+0",
        "Filetype=Linear", 'Sweep Signal="Bias (V)"',
        f'Fixed parameters="{";".join(fixed)}"',
        f'Experiment parameters="{";".join(exp)}"',
        f"# Parameters (4 byte)={len(fixed) + len(exp)}",
        f"Experiment size (bytes)={len(names) * npts * 4}",
        f"Points={npts}",
        f'Channels="{";".join(names)}"',
        'Experiment="Grid Spectroscopy"',
        f'Start time="{start}"', f'End time="{end}"',
        "Bias>Bias (V)=2E+0", "Z-Controller>Setpoint=300E-12",
        f"Bias Spectroscopy>Sweep Start (V)={v0}", f"Bias Spectroscopy>Sweep End (V)={v1}",
        "Bias Spectroscopy>Z offset (m)=0E+0", "Bias Spectroscopy>Number of sweeps=2",
        "Lock-in>Lock-in status=ON",
    ]
    have = nx * ny if have is None else have
    body = bytearray()
    for p in range(have):
        iy, ix = divmod(p, nx)
        z = 0.0 if z_param is None else float(z_param[iy, ix])
        vals = [v0, v1, (ix - nx / 2) * 1e-10, (iy - ny / 2) * 1e-10, z]
        for name in names:
            vals.extend(np.asarray(channels[name][iy, ix, :], dtype=float).tolist())
        body += np.asarray(vals, dtype=">f4").tobytes()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes("\r\n".join(head).encode("latin-1") + b"\r\n:HEADER_END:\r\n" + bytes(body))
    return out


class Synth:
    """合成文件写入器。以 fixture 暴露，测试不必 import conftest。"""

    sxm = staticmethod(write_sxm)
    dat = staticmethod(write_dat)
    grid = staticmethod(write_3ds)


@pytest.fixture()
def synth() -> type[Synth]:
    return Synth


#: 2001 年量级的纳秒时间戳（合成文件的 mtime 用它，免得几条测试撞在同一个 mtime 上）。
T0_NS = 1_000_000_000_000_000_000


@pytest.fixture()
def t0_ns() -> int:
    return T0_NS
