# -*- coding: utf-8 -*-
"""自动判据、重复保存与同一次采集的片段。

原子分辨诊断调用 vision.atomic_phase；满权重尺度才报告肯定结果，
尺度降低或数据不足时明确标为无法判定。超结构调用 vision.lattice_cell 的通用
分数阶检验，以空白对照抑制背景，并先核对所测原胞是否为谐波。
这些判据从输入数据推断，不预置材料名称、晶格常数或研究结论。
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from mast.gallery.inventory import dir_key

#: 2：超结构检验前先核对原胞是不是谐波（见 :func:`superstructure`）。
VER_ANALYSIS = 2
#: 有效行少于这个数不跑原子判据（逐行 1D 谱与二维谱都需要足够的行）。
MIN_ROWS = 64

#: ``atomic_phase.ALL_REASONS`` 里「判不了」的那一半。测试钉住它们都还在那张表里 ——
#: 那边改了名字，这里不能静默地把「判不了」读成「没有」。
UNDETERMINED = ("too_few_periods", "scale_gate", "scale_reduced", "unknown_pixel_size",
                "insufficient_data", "dead_flat", "dependency_unavailable")


def _oriented_z(path: str, scan: dict | None):
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames

    if scan is None:
        scan = read_sxm(path)
    z = sxm_oriented_frames(scan, "Z").get("forward")
    return None if z is None else np.asarray(z, dtype=np.float64)


#: 原胞最长基矢短于原子判据周期的这个比例 ⇒ 原胞是谐波，不做超结构检验。
CELL_PERIOD_MIN_RATIO = 0.75


def superstructure(z_rows: np.ndarray, nm_per_px: float, scan_dir: str = "",
                   ref_period_nm: float | None = None) -> tuple[float, str, str]:
    """返回 (最大候选/对照比, 标签, 未检验原因)。

    当所测原胞最长基矢短于参考周期的 CELL_PERIOD_MIN_RATIO 倍时，
    该原胞可能对应高次谐波。此时跳过分数阶检验，避免将基频峰误报为超结构。
    """
    from mast.vision.lattice_cell import measure_cell, superstructure_test

    cell = measure_cell(z_rows, float(nm_per_px), scan_dir=scan_dir or "")
    if not cell.ok or not cell.a1_nm:
        return 0.0, "", "cell:" + (cell.reason or "not_ok")
    longest = max(float(cell.a1_nm), float(cell.a2_nm or 0.0))
    if ref_period_nm and longest < CELL_PERIOD_MIN_RATIO * float(ref_period_nm):
        return 0.0, "", "cell_is_a_harmonic"
    tests = superstructure_test(z_rows, float(nm_per_px), cell)
    present = [t for t in tests if t.verdict == "present" and t.ratio_to_control]
    if not present:
        return 0.0, "", ""
    best = max(present, key=lambda t: float(t.ratio_to_control))
    return round(float(best.ratio_to_control), 2), str(best.label), ""


def analyse_frame(path: str, nm_per_px: float | None, r0: int | None, r1: int | None, *,
                  scan: dict | None = None, scan_dir: str = "") -> dict:
    """一帧的自动判据。永不抛：判据自己的异常记成 ``ar="error: …"``。"""
    out: dict = {"v": VER_ANALYSIS, "at": 0.0, "ar": "", "hf": 0.0, "hl": "",
                 "at_reduced": 0.0, "scale": None}
    try:
        from mast.vision.atomic_phase import SCALE_OFF_NMPP, assess_atomic_phase
    except Exception as exc:  # noqa: BLE001
        out["ar"] = "dependency_unavailable"
        out["err"] = str(exc)[:200]
        return out
    if not nm_per_px or not math.isfinite(nm_per_px) or nm_per_px <= 0:
        out["ar"] = "unknown_pixel_size"
        return out
    if nm_per_px > SCALE_OFF_NMPP:
        out["ar"] = "scale_gate"            # 物理上分辨不了原子，不跑
        return out
    if r0 is None or r1 is None or (r1 - r0) < MIN_ROWS:
        out["ar"] = "insufficient_data"
        return out
    try:
        z = _oriented_z(path, scan)
        if z is None:
            out["ar"] = "insufficient_data"
            return out
        # 一次调用拿两种口径：放宽尺度门时 passed 且 scale=="full" ⇔ 严格口径 passed。
        res = assess_atomic_phase(z, nm_per_px=float(nm_per_px), allow_reduced_scale=True)
        out["scale"] = res.scale
        conc = float(res.angular_concentration)
        if res.passed:
            out["at_reduced"] = float(round(conc))
            if res.scale == "full":
                out["at"] = float(round(conc))
            else:
                out["ar"] = "scale_reduced"
        elif res.scale == "reduced":
            # 严格口径下这一档一律「判不了」，不论判据本身过没过。
            out["ar"] = "scale_reduced"
        else:
            undet = [r for r in res.reasons if r in UNDETERMINED]
            out["ar"] = undet[0] if undet else ""
        if out["at"] > 0:
            refs = [p for p in (res.period_nm, res.period_fast_axis_nm) if p]
            hf, hl, skipped = superstructure(z[r0:r1], float(nm_per_px), scan_dir,
                                             ref_period_nm=min(refs) if refs else None)
            out["hf"], out["hl"] = hf, hl
            if skipped:
                out["hs"] = skipped              # 只进缓存：为什么没做超结构检验
    except Exception as exc:  # noqa: BLE001 — 一帧判据出错不能拖垮构建
        out.update(at=0.0, hf=0.0, hl="", ar="error")
        out["err"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


# ── 重复保存 ───────────────────────────────────────────────────────────


def _z_equal(path_a: str, path_b: str) -> bool:
    a = _oriented_z(path_a, None)
    b = _oriented_z(path_b, None)
    return (a is not None and b is not None and a.shape == b.shape
            and bool(np.array_equal(np.nan_to_num(a), np.nan_to_num(b))))


def _frame_ok(inv_entry: dict, render: dict | None) -> bool:
    return (inv_entry.get("kind") == "f" and not inv_entry.get("err")
            and render is not None and bool(render.get("ok"))
            and render.get("src") == [inv_entry.get("size"), inv_entry.get("mtime_ns")]
            and bool(inv_entry.get("t0")))


def find_repeat_saves(inv: dict, render: dict, cache: dict) -> tuple[dict, dict]:
    """``({重复帧 id: 原帧 id}, 新比对缓存)``。

    旧版兼容格式语义（陷阱 T15）：同目录、同 REC 时刻、同像素、同视野、同中心的帧里，按保存时刻
    排序，后面的每一张与**第一张**比 Z 块；逐字节相同（NaN 视为 0）就是重复保存。
    比对结果按两边的 ``(size, mtime_ns)`` 缓存 —— 任何一边内容变了就重比（T1）。"""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for i, e in inv.items():
        if not _frame_ok(e, render.get(i)):
            continue
        key = (dir_key(i), e["t0"], e.get("nx"), e.get("ny"), round(e.get("w_nm") or 0, 4),
               round(e.get("cx_nm") or 0, 4), round(e.get("cy_nm") or 0, 4))
        groups[key].append(i)
    dup: dict[str, str] = {}
    new_cache: dict[str, dict] = {}
    for ids in groups.values():
        if len(ids) < 2:
            continue
        ids.sort(key=lambda i: (inv[i].get("mtime_ns") or 0, i))
        first = ids[0]
        for other in ids[1:]:
            ck = f"{first}|{other}"
            src = [inv[first]["size"], inv[first]["mtime_ns"], inv[other]["size"], inv[other]["mtime_ns"]]
            c = cache.get(ck)
            if not isinstance(c, dict) or c.get("src") != src:
                try:
                    eq = _z_equal(inv[first]["path"], inv[other]["path"])
                except Exception:  # noqa: BLE001
                    eq = False
                c = {"src": src, "eq": bool(eq)}
            new_cache[ck] = c
            if c["eq"]:
                dup[other] = first
    return dup, new_cache


def acquisition_fragments(inv: dict, render: dict, dup: dict) -> dict[str, int]:
    """同一次采集落成的**不同**文件（自动保存 + 手工保存 + 中途停，长度不同、字节不同）。

    判据 ``(目录, REC 时刻, 像素, 视野, 中心, 扫描角)``；重复保存不算（它们另有 ``dup``）。
    只在 ≥ 2 个不同文件时给每个成员记数。"""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for i, e in inv.items():
        if not _frame_ok(e, render.get(i)):
            continue
        key = (dir_key(i), e["t0"], e.get("nx"), e.get("ny"), round(e.get("w_nm") or 0, 4),
               round(e.get("cx_nm") or 0, 4), round(e.get("cy_nm") or 0, 4),
               round(e.get("angle") or 0, 2))
        groups[key].append(i)
    seg: dict[str, int] = {}
    for ids in groups.values():
        distinct = [i for i in ids if i not in dup]
        if len(distinct) >= 2:
            for i in distinct:
                seg[i] = len(distinct)
    return seg
