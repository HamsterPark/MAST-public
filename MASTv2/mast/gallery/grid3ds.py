# -*- coding: utf-8 -*-
"""Nanonis ``.3ds`` 网格谱读取（图库专用）。

**为什么不用** ``mast.io.nanonis_files.read_3ds``：它只返回**第一个**通道、逐像素 Python
双循环、也不给「已经完成了几个点」。网格缩略图要 Current、LI Demod X/Y、Bias [AVG] 各通道，
还要知道一张扫到一半的网格完成了多少 —— 那正是它回答不了的。这里移植旧版兼容格式的向量化读取器。

约定：**第 0 行 = 视野下沿**（点坐标 Y 最小；旧版兼容格式按每个点的 X/Y 参数核对过）。画图时用
``imshow(origin="lower")``，别再翻一次。

未完成的网格：文件里只有前 ``have`` 个点，其余位置填 NaN。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

HEADER_TAG = b":HEADER_END:"
_FIRST_READ = 65536
_MAX_HEADER = 1 << 20


def _parse_header(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in re.findall(r"^([^=\r\n]+)=(.*)$", text, re.M):
        out[k.strip()] = v.strip().strip('"')
    return out


def _split(s: str) -> list[str]:
    return [p for p in (s or "").split(";") if p != ""]


def read_3ds(path: str | Path, data: bool = True) -> dict:
    """读一个 .3ds。``data=False`` 只读头（清点用）。

    返回 ``hdr nx ny npts chans have npar pars cx cy w_nm h_nm angle``（坐标 nm、角度度），
    ``data=True`` 时另有 ``P``（ny, nx, npar）与 ``D``（ny, nx, 通道数, npts），float32，
    未完成的点为 NaN。头里的数不自洽（通道数 × 点数 ≠ Experiment size）时抛 ``ValueError``。
    """
    path = str(path)
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(min(size, _FIRST_READ))
        idx = head.find(HEADER_TAG)
        if idx < 0 and size > len(head):
            head += fh.read(min(size, _MAX_HEADER) - len(head))
            idx = head.find(HEADER_TAG)
        if idx < 0:
            raise ValueError("找不到 :HEADER_END:")
        end = idx + len(HEADER_TAG)
        if head[end:end + 2] == b"\r\n":
            end += 2
        elif head[end:end + 1] == b"\n":
            end += 1
        hdr = _parse_header(head[:idx].decode("latin-1"))

        dims = re.split(r"\s*x\s*", hdr.get("Grid dim", "").strip())
        if len(dims) != 2:
            raise ValueError(f"Grid dim 读不懂：{hdr.get('Grid dim')!r}")
        nx, ny = int(dims[0]), int(dims[1])
        npar = int(hdr["# Parameters (4 byte)"])
        npts = int(hdr["Points"])
        chans = _split(hdr.get("Channels", ""))
        exp_size = int(hdr.get("Experiment size (bytes)", len(chans) * npts * 4))
        if nx <= 0 or ny <= 0 or npts <= 0 or npar < 0:
            raise ValueError(f"网格尺寸不合法：{nx}×{ny}×{npts}，参数 {npar}")
        per = npar + exp_size // 4
        have = int(min((size - end) // (4 * per), nx * ny)) if per > 0 else 0
        gs = []
        for v in _split(hdr.get("Grid settings", "")):
            try:
                gs.append(float(v))
            except ValueError:
                gs.append(0.0)
        cx, cy, w, h, ang = (gs + [0.0] * 5)[:5]
        pars = _split(hdr.get("Fixed parameters", "")) + _split(hdr.get("Experiment parameters", ""))
        G: dict = dict(hdr=hdr, nx=nx, ny=ny, npts=npts, chans=chans, have=have, npar=npar,
                       pars=pars, cx=cx * 1e9, cy=cy * 1e9, w_nm=w * 1e9, h_nm=h * 1e9,
                       angle=ang, header_end=end)
        if data:
            if exp_size // 4 != len(chans) * npts:
                raise ValueError(
                    f"Experiment size {exp_size} 与 {len(chans)} 通道 × {npts} 点对不上")
            fh.seek(end)
            raw = fh.read(have * per * 4)
            body = np.frombuffer(raw, dtype=">f4", count=have * per).reshape(have, per)
            P = np.full((nx * ny, npar), np.nan, dtype=np.float32)
            D = np.full((nx * ny, len(chans), npts), np.nan, dtype=np.float32)
            P[:have] = body[:, :npar]
            D[:have] = body[:, npar:].reshape(have, len(chans), npts)
            G["P"] = P.reshape(ny, nx, npar)
            G["D"] = D.reshape(ny, nx, len(chans), npts)
    return G
