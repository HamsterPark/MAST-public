# -*- coding: utf-8 -*-
"""清点：数据根里有哪些 .sxm / .dat / .3ds，各自头信息的摘要。

**身份是 ``(size, mtime_ns)``，不是只看 size**（陷阱 T1）。旧版兼容格式 ``inventory()`` 在尺寸
相同时就跳过重读；Nanonis 会重用文件编号，于是同名同尺寸的一次新测量会被旧记录静默
顶替 —— 顶替进来的是一份合法、能解析、看着完全正常的旧数据，没有任何东西会报错。
``mtime_ns`` 用整数纳秒：浮点秒在 2001 年的 epoch 上已经用掉 10 位有效数字，比不出相等
（``webui/scan_preview.py`` 的 ``ScanStat`` 为同一件事写过说明）。

**字节级副本**（MAST 自动拷贝进实验文件夹、原件留在原处）用现成的
``webui.scan_preview.collapse_scan_groups`` 折叠成一条，不另写一份判据（陷阱 T14）。
**同名 ≠ 同内容**：跨会话目录重名的不同测量 mtime 不同，不会被折叠。

读头信息只读头：.sxm 用 ``read_sxm_header``（不读数据块），.3ds 用 ``grid3ds.read_3ds(data=False)``；
.dat 本身很小，整份读。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VER_INVENTORY = 1
KIND_BY_EXT = {".sxm": "f", ".dat": "s", ".3ds": "g"}

#: 永远不往里走的目录（小写）。``_NON_SCAN_DIR_NAMES``（env/signals/monitors）从
#: ``webui.scan_preview`` 取，这里只补图库自己的状态目录名与几个肯定不是数据的目录。
_EXTRA_SKIP_DIRS = frozenset({"_gallery", ".mast", "__pycache__", ".git"})


@dataclass(frozen=True)
class FileRef:
    id: str          # <根名>/<相对路径>，正斜杠
    root: str
    rel: str
    path: str        # 绝对路径（本机分隔符）
    size: int
    mtime_ns: int


# ── id 小工具（render / index / marks 共用）─────────────────────────────

_DATE8 = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def dir_key(item_id: str) -> str:
    """条目所在目录的键：``<根名>/<相对父目录>``；文件直接在根下时就是根名。"""
    return item_id.rsplit("/", 1)[0] if "/" in item_id else item_id


def dir_label(d: str) -> str:
    """目录键的显示名：最后一段；``YYYYMMDD`` 显示成 ``YYYY-MM-DD``。"""
    last = d.rsplit("/", 1)[-1]
    m = _DATE8.match(last)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else last


def dir_short(d: str) -> str:
    """缩略图信息条上的目录小标签：日期目录取 ``MMDD``，其余取前 12 个字符。"""
    last = d.rsplit("/", 1)[-1]
    return last[4:] if _DATE8.match(last) else last[:12]


def file_prefix(fn: str) -> str:
    """文件名前缀（设计文档 D15）：去扩展名后去掉末尾的「分隔符 + 数字」。

    ``WO..._0010.sxm`` 这类 Nanonis 系列名去掉计数器之后，剩下的就是操作员给这一串
    测量起的名字 —— 通用层不认识样品，但这个前缀是操作员自己写下的分组。"""
    stem = fn.rsplit(".", 1)[0] if "." in fn else fn
    p = re.sub(r"[\s_\-.]*\d+$", "", stem)
    return p or stem


def num(s, default=None):
    """头信息里的数：取第一个空白分隔的 token；读不出或非有限值 ⇒ ``default``。"""
    if s is None:
        return default
    try:
        v = float(str(s).split()[0])
    except (ValueError, IndexError):
        return default
    return v if np.isfinite(v) else default


def _floats(s) -> list:
    return [num(x) for x in str(s or "").split()]


# ── 头信息摘要 ─────────────────────────────────────────────────────────


def sxm_summary(path: str) -> dict:
    from mast.io.nanonis_files import read_sxm_header

    h = read_sxm_header(path)
    if not h:
        raise ValueError("读不到 .sxm 头（没有结束标记）")
    rng = _floats(h.get("scan_range"))
    off = _floats(h.get("scan_offset"))
    pix = h.get("scan_pixels") or []
    # :Z-CONTROLLER: 是两行的表，头解析保留最后一行（值那一行，首尾空白已去掉）：
    # "log Current\t1\t3.000E-10 A\t..." ⇒ 第 3 列是设定点。旧版兼容格式取的就是这张表。
    zc = str(h.get("z-controller") or "").split("\t")
    setp = num(zc[2]) if len(zc) > 2 else None
    if setp is None:
        setp = num(h.get("z-controller>setpoint"))
    rec_d = str(h.get("rec_date") or "").strip()
    rec_t = str(h.get("rec_time") or "").strip()
    t0 = None
    if rec_d and rec_t:
        try:
            t0 = time.mktime(time.strptime(f"{rec_d} {rec_t}", "%d.%m.%Y %H:%M:%S"))
        except ValueError:
            t0 = None
    acq = num(h.get("acq_time"))
    if t0 is None:
        t0 = os.path.getmtime(path) - (acq or 0.0)
    return {
        "t0": t0,
        "rec": f"{rec_d} {rec_t}".strip(),
        "acq_s": acq,
        "w_nm": rng[0] * 1e9 if rng and rng[0] else None,
        "h_nm": rng[1] * 1e9 if len(rng) > 1 and rng[1] else None,
        "cx_nm": off[0] * 1e9 if off and off[0] is not None else None,
        "cy_nm": off[1] * 1e9 if len(off) > 1 and off[1] is not None else None,
        "angle": num(h.get("scan_angle")),
        "nx": int(pix[0]) if len(pix) > 0 else None,
        "ny": int(pix[1]) if len(pix) > 1 else None,
        "scan_dir": str(h.get("scan_dir") or "").strip().lower(),
        "bias_V": num(h.get("bias")),
        "setpoint_pA": setp * 1e12 if setp is not None else None,
        "channels": [str(c) for c in (h.get("channel_names") or [])],
        "lockin_status": str(h.get("lock-in>lock-in_status") or ""),
        "series": str(h.get("scan>series_name") or ""),
    }


def dat_time(hdr: dict, path: str) -> float:
    for k in ("Start time", "Saved Date"):
        v = hdr.get(k)
        if v:
            try:
                return time.mktime(time.strptime(str(v).strip(), "%d.%m.%Y %H:%M:%S"))
            except ValueError:
                pass
    return os.path.getmtime(path)


def dat_summary(path: str) -> dict:
    from mast.io.nanonis_files import read_dat

    res = read_dat(path)
    h = res.get("header") or {}
    cols = res.get("columns") or {}
    names = list(cols)
    V = np.asarray(cols[names[0]], dtype=float) if names else np.zeros(0)
    finite = V[np.isfinite(V)]
    sweeps = len(set(re.findall(r"\[(\d{5})\]", ";".join(names)))) or 1
    return {
        "exp": str(h.get("Experiment") or ""),
        "t0": dat_time(h, path),
        "x_nm": (num(h.get("X (m)")) or 0.0) * 1e9,
        "y_nm": (num(h.get("Y (m)")) or 0.0) * 1e9,
        "zoff_pm": (num(h.get("Z offset (m)")) or 0.0) * 1e12,
        "bias_V": num(h.get("Bias>Bias (V)")),
        "setpoint_pA": (num(h.get("Z-Controller>Setpoint")) or 0.0) * 1e12,
        "n": int(V.size),
        "Vmin": float(finite.min()) if finite.size else None,
        "Vmax": float(finite.max()) if finite.size else None,
        "has_LI": any(c.startswith("LI Demod 1") for c in names),
        "has_current": any(c.startswith("Current") for c in names),
        "sweeps": sweeps,
    }


def grid_summary(path: str) -> dict:
    from mast.gallery.grid3ds import read_3ds

    G = read_3ds(path, data=False)
    g = G["hdr"]

    def tparse(key):
        try:
            return time.mktime(time.strptime(str(g.get(key, "")).strip(), "%d.%m.%Y %H:%M:%S"))
        except ValueError:
            return None

    sw = str(g.get("Bias Spectroscopy>Number of sweeps", "")).strip()
    return {
        "t0": tparse("Start time") or os.path.getmtime(path),
        "t1": tparse("End time"),
        "nx": G["nx"], "ny": G["ny"], "npts": G["npts"], "have": G["have"],
        "w_nm": G["w_nm"], "h_nm": G["h_nm"], "cx_nm": G["cx"], "cy_nm": G["cy"],
        "angle": G["angle"],
        "bias_V": num(g.get("Bias>Bias (V)")),
        "setpoint_pA": (num(g.get("Z-Controller>Setpoint")) or 0.0) * 1e12,
        "v0": num(g.get("Bias Spectroscopy>Sweep Start (V)")),
        "v1": num(g.get("Bias Spectroscopy>Sweep End (V)")),
        "zoff_pm": (num(g.get("Bias Spectroscopy>Z offset (m)")) or 0.0) * 1e12,
        "sweeps": int(sw) if sw.isdigit() else None,
        "lockin": str(g.get("Lock-in>Lock-in status", "")),
    }


SUMMARISERS = {".sxm": sxm_summary, ".dat": dat_summary, ".3ds": grid_summary}


# ── 遍历 ───────────────────────────────────────────────────────────────


def _skip_names() -> set[str]:
    try:
        from mast.webui.scan_preview import _NON_SCAN_DIR_NAMES

        base = {n.lower() for n in _NON_SCAN_DIR_NAMES}
    except Exception:  # noqa: BLE001
        base = {"env", "signals", "monitors"}
    return base | set(_EXTRA_SKIP_DIRS)


def _norm(p: str | Path) -> str:
    try:
        return os.path.normcase(os.path.realpath(str(p)))
    except OSError:
        return os.path.normcase(os.path.abspath(str(p)))


def normalise_only(only: str | None) -> str:
    return (only or "").strip().replace("\\", "/").strip("/")


def in_scope(item_id: str, prefix: str) -> bool:
    """按路径段判前缀：``SPM/2001/0910`` 不匹配 ``SPM/2001/09101``。"""
    if not prefix:
        return True
    return item_id == prefix or item_id.startswith(prefix + "/")


def walk_roots(cfg, *, only: str | None = None, state: Path | str | None = None
               ) -> tuple[list[FileRef], list[str]]:
    """``(文件, 走过的范围)``。

    「走过的范围」是 id 前缀列表，只有**这次真的遍历到**的根才在里面：清点缓存里
    「文件消失了就删条目」只在这些范围里做 —— 一块暂时拔掉的盘不会让它的全部条目被清空。
    """
    skip = _skip_names()
    state_key = _norm(state) if state is not None else None
    only = normalise_only(only)
    only_root, _, only_sub = only.partition("/")
    files: list[FileRef] = []
    scopes: list[str] = []
    seen: set[str] = set()

    def add(root, base: str, full: str) -> None:
        ext = os.path.splitext(full)[1].lower()
        if ext not in KIND_BY_EXT:
            return
        key = os.path.normcase(full)
        if key in seen:
            return
        try:
            st = os.stat(full)
        except OSError:
            return          # 在 glob 与 stat 之间消失了（仪器正在写）
        seen.add(key)
        rel = os.path.relpath(full, base).replace("\\", "/")
        files.append(FileRef(id=f"{root.name}/{rel}", root=root.name, rel=rel, path=full,
                             size=int(st.st_size), mtime_ns=int(st.st_mtime_ns)))

    for root in cfg.roots:
        if not root.enabled:
            continue
        if only and root.name != only_root:
            continue
        base = str(Path(root.path))
        if not os.path.isdir(base):
            continue
        start = str(Path(base, *only_sub.split("/"))) if only_sub else base
        scopes.append(f"{root.name}/{only_sub}" if only_sub else root.name)
        if os.path.isfile(start):
            add(root, base, start)
            continue
        if not os.path.isdir(start):
            continue
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(
                d for d in dirnames
                if d.lower() not in skip
                and (state_key is None or _norm(os.path.join(dirpath, d)) != state_key))
            for fn in sorted(filenames):
                add(root, base, str(Path(dirpath, fn)))
    return files, scopes


def collapse(files: list[FileRef]) -> list[tuple[FileRef, list[FileRef]]]:
    """字节级副本折叠：``[(代表, 全部成员)]``。代表优先取实验根之外的原件。"""
    from mast.webui.scan_preview import ScanStat, _experiment_root_key, collapse_scan_groups

    by = {os.path.normcase(f.path): f for f in files}
    stats = [ScanStat(path=Path(f.path), mtime_ns=f.mtime_ns, size_bytes=f.size) for f in files]
    out = []
    for g in collapse_scan_groups(stats, _experiment_root_key()):
        rep = by[os.path.normcase(str(g.rep.path))]
        members = [by[os.path.normcase(str(m.path))] for m in g.members]
        out.append((rep, members))
    return out


def same_identity(entry: dict | None, ref: FileRef) -> bool:
    """这条缓存记录是不是描述的**这一份**字节。size **与** mtime_ns 都要相等（T1）。"""
    return (entry is not None
            and entry.get("size") == ref.size
            and entry.get("mtime_ns") == ref.mtime_ns)


def update_inventory(groups: list[tuple[FileRef, list[FileRef]]], inv: dict, stamp: str,
                     *, force: bool = False) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """就地更新清点缓存。``(新 id, 变了的 id, 读失败 [(id, 原因)])``。

    内容变了（size 或 mtime_ns 不同）按**新文件**对待：``added`` 换成本批次 ——
    编号重用的那次新测量应当出现在「最新一批」里。只是解析版本号变了则保留原批次。
    """
    new_ids: list[str] = []
    changed: list[str] = []
    failed: list[tuple[str, str]] = []
    for rep, members in groups:
        old = inv.get(rep.id)
        copies = [m.path for m in members] if len(members) > 1 else []
        if (same_identity(old, rep) and old.get("v") == VER_INVENTORY
                and not (force and old.get("err"))):
            old["path"], old["root"], old["rel"] = rep.path, rep.root, rep.rel
            if copies:
                old["copies"] = copies
            else:
                old.pop("copies", None)
            continue
        ext = os.path.splitext(rep.path)[1].lower()
        content_same = same_identity(old, rep)
        entry = {
            "kind": KIND_BY_EXT[ext], "root": rep.root, "rel": rep.rel, "path": rep.path,
            "size": rep.size, "mtime_ns": rep.mtime_ns, "v": VER_INVENTORY,
            "added": (old.get("added") if (old and content_same) else None) or stamp,
        }
        if copies:
            entry["copies"] = copies
        try:
            entry.update(SUMMARISERS[ext](rep.path))
        except Exception as exc:  # noqa: BLE001 — 一个坏文件不能拖垮整次清点
            entry["err"] = f"{type(exc).__name__}: {exc}"[:200]
            failed.append((rep.id, entry["err"]))
        inv[rep.id] = entry
        (new_ids if old is None else changed).append(rep.id)
    return new_ids, changed, failed


def purge(inv: dict, scopes: list[str], seen_ids: set[str]) -> list[str]:
    """删掉「在这次走过的范围里、却没再出现」的条目。返回被删的 id。"""
    removed = [i for i in inv if i not in seen_ids and any(in_scope(i, s) for s in scopes)]
    for i in removed:
        inv.pop(i, None)
    return removed
