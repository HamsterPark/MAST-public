"""把一份测量数据放进会话的 ``inputs/`` —— 连同**算好的物理尺度**。

这是「便利入口」，不是唯一入口
============================
脚本完全可以自己 ``from mast.io.nanonis_files import read_sxm`` 直接读（那份代码
就在会话的 ``mast/`` 里）；批量分析 200 张图就该那么写，不是调 200 次 tool。

那这个入口多给了什么？—— **物理尺度**。``nm_per_px`` / ``bias_V`` /
``scan_range`` 由主进程用 ``sxm_frame_meta`` 从表头算好写进 manifest，脚本读一个
数就行。这一步省掉的不是几行代码，是一类静默错误：模型自己换算像素→纳米时，
一个 ``3.2e-12`` 被复述成 ``3.2`` 就是三个数量级，而结果看起来完全合理。

只有一套 parser
===============
这里 import 的是 ``mast.io.nanonis_files`` —— 全系统唯一那套字节 parser
（2026-07-20 删掉 ``data/formats.py`` 之后就只剩它）。
``test_staging_imports_only_the_canonical_parser`` 用 AST 守着这一条。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXT = {".sxm"}
SPECTRUM_EXT = {".dat", ".txt"}
GRID_EXT = {".3ds"}


@dataclass(frozen=True)
class StagedInput:
    name: str
    rel: str
    kind: str
    arrays: dict           # {通道: {shape, dtype, unit}}
    meta: dict             # 写进 manifest 的其余字段


def resolve(path: str = "", scan_id: str = "") -> Path:
    """按 path 或 scan_id 找到文件。找不到就抛一个**说得出下一步**的错。"""
    if path:
        p = Path(path).expanduser()
        if p.is_file():
            return p
        # 模型常把路径记岔一个目录 —— 按 stem 在已知扫描目录里找回来，
        # 同 DP 的 _resolve_missing（tools.py:57-104）那条容错。
        try:
            from mast.core.scan_registry import known_scan_dirs
            for d in known_scan_dirs():
                cand = Path(d) / p.name
                if cand.is_file():
                    return cand
        except Exception:  # noqa: BLE001
            pass
        raise FileNotFoundError(f"找不到文件：{path}")
    if scan_id:
        from mast.core.scan_registry import resolve_scan_id
        got = resolve_scan_id(scan_id)
        if got:
            return Path(got)
        raise FileNotFoundError(f"scan_id 解析不到：{scan_id}")
    raise ValueError("要么给 path，要么给 scan_id")


def _floats(v) -> list[float]:
    """把表头字段变成一串数。

    ``sxm_frame_meta`` 的返回**混着两种形态**：``scan_pixels`` 已经是
    ``[64, 64]``，而 ``scan_range`` / ``scan_offset`` 仍是原始字符串
    ``'2.000000E-8           2.000000E-8'``（它的 docstring 写的就是
    "plain lists/strings"）。只按 list 判断会静默拿不到尺度 —— 第一版就是这么
    让 ``nm_per_px`` 变成 None 的，而 None 一路传下去，脚本就得自己去 split
    那个字符串，正好是我们想替它省掉的那一步。
    """
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                return []
        return out
    try:
        return [float(x) for x in str(v).split()]
    except (TypeError, ValueError):
        return []


def _normalise_frame(frame: dict) -> dict:
    """把 frame 里的数值字段统一成 list[float]，脚本拿到的就是数。"""
    out = dict(frame)
    for k in ("scan_range", "scan_offset", "scan_pixels"):
        if k in out:
            vals = _floats(out[k])
            if vals:
                out[k] = vals
    return out


def _nm_per_px(frame: dict) -> list[float] | None:
    rng = _floats(frame.get("scan_range"))
    px = _floats(frame.get("scan_pixels"))
    if len(rng) < 2 or len(px) < 2:
        return None
    try:
        return [rng[i] / px[i] * 1e9 for i in range(2) if px[i]]
    except (ZeroDivisionError, TypeError):
        return None


def _stage_sxm(src: Path, name: str, dst_dir: Path) -> StagedInput:
    from mast.io.nanonis_files import read_sxm, sxm_frame_meta

    d = read_sxm(str(src))
    header = d.get("header") or {}
    channels = d.get("channels") or {}
    frame = _normalise_frame(sxm_frame_meta(header))

    arrays: dict[str, np.ndarray] = {}
    described: dict[str, dict] = {}
    for chan, fb in channels.items():
        for direction in ("forward", "backward"):
            img = fb.get(direction) if isinstance(fb, dict) else None
            if img is None:
                continue
            key = f"{str(chan).strip().lower().replace(' ', '_')}_{direction[:3]}"
            a = np.asarray(img)
            arrays[key] = a
            described[key] = {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "unit": _unit_hint(str(chan)),
                "channel": str(chan),
                "direction": direction,
            }
    if not arrays:
        raise ValueError(f"{src.name} 里没有可用的通道数据")

    np.savez_compressed(dst_dir / f"{name}.npz", **arrays)
    meta = {
        "kind": "image",
        "frame": frame,
        "nm_per_px": _nm_per_px(frame),
        "bias_V": _first_float(header.get("bias")),
        "channels": list(frame.get("channels") or channels.keys()),
    }
    return StagedInput(name=name, rel=f"inputs/{name}.npz", kind="image",
                       arrays=described, meta=meta)


def _unit_hint(channel: str) -> str:
    try:
        from mast.io.nanonis_files import _CHANNEL_UNIT_HINT
        return _CHANNEL_UNIT_HINT.get(channel.strip().lower(), "")
    except Exception:  # noqa: BLE001
        return ""


def _first_float(v) -> float | None:
    try:
        return float(str(v).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _stage_spectrum(src: Path, name: str, dst_dir: Path) -> StagedInput:
    """``.dat`` 是二义的：Nanonis 点谱有 ``[DATA]`` 块，裸两列导出没有。

    先试 canonical 的谱 reader，失败再退回通用文本 —— 顺序反了会让裸 .dat 报
    「no columns」（DP 的 ``_load_array`` 踩过同一个坑，``tools.py:196-218``）。
    """
    from mast.io.nanonis_files import read_dat, read_txt

    cols: dict[str, np.ndarray] = {}
    try:
        d = read_dat(str(src))
        data = d.get("data")
        names = d.get("columns") or []
        if data is not None:
            a = np.asarray(data)
            for i, cname in enumerate(names[:a.shape[1] if a.ndim > 1 else 1]):
                cols[str(cname).strip().replace(" ", "_") or f"col{i}"] = a[:, i]
    except Exception:  # noqa: BLE001
        cols = {}
    if not cols:
        m = np.asarray(read_txt(str(src)).get("matrix"))
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        for i in range(m.shape[1]):
            cols[f"col{i}"] = m[:, i]
    if not cols:
        raise ValueError(f"{src.name} 里读不出任何列")

    np.savez_compressed(dst_dir / f"{name}.npz", **cols)
    described = {k: {"shape": list(np.asarray(v).shape),
                     "dtype": str(np.asarray(v).dtype), "unit": ""}
                 for k, v in cols.items()}
    return StagedInput(name=name, rel=f"inputs/{name}.npz", kind="spectrum",
                       arrays=described, meta={"kind": "spectrum"})


def stage(session, *, path: str = "", scan_id: str = "",
          name: str = "") -> StagedInput:
    """解析一份数据、写成 ``inputs/<name>.npz``、更新 manifest。"""
    src = resolve(path=path, scan_id=scan_id)
    name = (name or src.stem).strip().replace(" ", "_") or "input"
    dst_dir = session.inputs_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    ext = src.suffix.lower()
    if ext in IMAGE_EXT:
        staged = _stage_sxm(src, name, dst_dir)
        parser = "mast.io.nanonis_files.read_sxm"
    elif ext in SPECTRUM_EXT:
        staged = _stage_spectrum(src, name, dst_dir)
        parser = "mast.io.nanonis_files.read_dat"
    elif ext == ".npy":
        a = np.load(src, allow_pickle=False)
        np.savez_compressed(dst_dir / f"{name}.npz", data=a)
        staged = StagedInput(
            name=name, rel=f"inputs/{name}.npz", kind="array",
            arrays={"data": {"shape": list(a.shape), "dtype": str(a.dtype),
                             "unit": ""}},
            meta={"kind": "array"})
        parser = "numpy.load"
    else:
        raise ValueError(
            f"不认识的后缀 {ext}。支持 .sxm / .dat / .txt / .npy；"
            f"其它格式请在 py_run 里直接用 mastkit 或 numpy 读。")

    _update_manifest(session, staged, src, parser, scan_id)
    return staged


def _update_manifest(session, staged: StagedInput, src: Path,
                     parser: str, scan_id: str) -> None:
    """写 ``inputs/manifest.json``（原子替换）。同名条目就地替换，不追加重复。"""
    p = session.inputs_dir / "manifest.json"
    doc = {"schema": 1, "session_id": session.sid, "inputs": []}
    if p.is_file():
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    entry = {
        "name": staged.name,
        "file": staged.rel,
        "source_path": str(src),
        "scan_id": scan_id or src.stem,
        "parser": parser,              # 唯一真源的凭证，写进 manifest 以便追溯
        "arrays": staged.arrays,
        **staged.meta,
    }
    items = [e for e in doc.get("inputs", []) if e.get("name") != staged.name]
    items.append(entry)
    doc["inputs"] = items
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    os.replace(tmp, p)


def scan_dirs_for_env() -> str:
    """给子进程 env 的 ``MAST_PYEXEC_SCAN_DIRS`` —— 主动告诉脚本数据在哪。

    不设读白名单，但 agent 不知道路径就等于没有权限。
    """
    dirs: list[str] = []
    try:
        from mast.core.scan_registry import known_scan_dirs
        dirs += [str(d) for d in known_scan_dirs()]
    except Exception:  # noqa: BLE001
        pass
    try:
        from mast._runtime_paths import project_root
        root = project_root()
        for sub in ("experiments", "data", "stm-datasets"):
            d = root / sub
            if d.is_dir():
                dirs.append(str(d))
    except Exception:  # noqa: BLE001
        pass
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return os.pathsep.join(out)


__all__ = ["StagedInput", "resolve", "scan_dirs_for_env", "stage"]
