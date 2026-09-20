# -*- coding: utf-8 -*-
"""``index.json``：前端一次拉取的全量索引（设计文档 §4.1、D9、D10）。

条目用短键名，与旧版兼容格式 ``data.js`` 的键一致（外加 ``p pf cp cpl r0 r1 ar hl seg``），
便于逐字段对账。值为 None 的可选键省略；帧始终带 ``at ar hf hl li rows rall r0 r1``。
浮点四舍五入到 4 位。NaN / inf 一律变 None（JSON 里没有 NaN）。

同时写 ``index.json.gz``。读的时候 gz 不比 json 新就不用它（两次写之间被读到的那一瞬，
不能拿旧的 gz 冒充新的索引）。
"""
from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from mast.gallery import paths as _paths
from mast.gallery.inventory import dir_key, file_prefix

INDEX_VERSION = 1
THUMB_URL_PREFIX = "/api/gallery/thumb/"

#: 帧条目始终出现的键（其余可选键为 None 时省略）。
FRAME_ALWAYS = ("at", "ar", "hf", "hl", "li", "rows", "rall", "r0", "r1")


def thumb_url(thumbs: Path, rel: str | None) -> str:
    """``/api/gallery/thumb/<URL 编码的 rel>?v=<缩略图文件整数 mtime>``；文件不在 ⇒ ""。"""
    if not rel:
        return ""
    try:
        v = int(os.stat(Path(thumbs, *rel.split("/"))).st_mtime)
    except OSError:
        return ""
    return THUMB_URL_PREFIX + quote(rel, safe="/") + f"?v={v}"


def _clean(v):
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return round(v, 4)
    return v


def _put(item: dict, key: str, value, *, always: bool = False) -> None:
    value = _clean(value)
    if value is None and not always:
        return
    item[key] = value


def build_items(inv: dict, render: dict, analysis: dict, dup: dict, seg: dict,
                enabled_roots: set[str], thumbs: Path) -> list[dict]:
    """从三层缓存拼出索引条目。

    只收「渲染记录与当前文件是同一份字节、渲染成功、缩略图文件还在」的条目：一个内容
    变了但还没重出图的文件，**不能**拿旧缩略图配新元数据出现在索引里（T1）。"""
    from mast.gallery.analysis import VER_ANALYSIS
    from mast.gallery.render import VER_LI

    items: list[dict] = []
    for iid, e in inv.items():
        if iid.split("/", 1)[0] not in enabled_roots or e.get("err"):
            continue
        r = render.get(iid)
        src = [e.get("size"), e.get("mtime_ns")]
        if not r or not r.get("ok") or r.get("src") != src:
            continue
        th = thumb_url(thumbs, r.get("th"))
        if not th:
            continue
        k = e.get("kind")
        fn = iid.rsplit("/", 1)[-1]
        it: dict = {"id": iid, "k": k, "d": dir_key(iid), "fn": fn, "p": e.get("path") or "",
                    "pf": file_prefix(fn)}
        _put(it, "t", e.get("t0"))
        _put(it, "mt", (e["mtime_ns"] / 1e9) if e.get("mtime_ns") else None)
        it["ad"] = e.get("added") or ""
        it["th"] = th
        copies = e.get("copies") or []
        if len(copies) > 1:
            it["cp"] = len(copies)
            it["cpl"] = list(copies)

        if k == "f":
            for key, val in (("w", e.get("w_nm")), ("hn", e.get("h_nm")), ("b", e.get("bias_V")),
                             ("sp", e.get("setpoint_pA")), ("nx", e.get("nx")), ("ny", e.get("ny")),
                             ("ang", e.get("angle")), ("cx", e.get("cx_nm")), ("cy", e.get("cy_nm")),
                             ("acq", e.get("acq_s"))):
                _put(it, key, val)
            it["sd"] = "u" if e.get("scan_dir") == "up" else "d"
            it["rows"] = int(r.get("rows") or 0)
            it["rall"] = int(r.get("rows_all") or 0)
            it["r0"] = int(r.get("r0") or 0)
            it["r1"] = int(r.get("r1") or 0)
            li = ""
            if r.get("li") and r.get("vli") == VER_LI:
                li = thumb_url(thumbs, r.get("li"))
            it["li"] = li
            a = analysis.get(iid)
            if a and a.get("src") == src and a.get("v") == VER_ANALYSIS:
                it["at"] = _clean(float(a.get("at") or 0.0))
                it["ar"] = str(a.get("ar") or "")
                it["hf"] = _clean(float(a.get("hf") or 0.0))
                it["hl"] = str(a.get("hl") or "")
            else:
                it["at"], it["ar"], it["hf"], it["hl"] = 0.0, "", 0.0, ""
            if iid in dup:
                it["dup"] = dup[iid]
            if seg.get(iid, 0) >= 2:
                it["seg"] = int(seg[iid])
        elif k == "s":
            for key, val in (("n", e.get("n")), ("v0", e.get("Vmin")), ("v1", e.get("Vmax")),
                             ("zo", e.get("zoff_pm")), ("x", e.get("x_nm")), ("y", e.get("y_nm")),
                             ("b", e.get("bias_V")), ("sp", e.get("setpoint_pA")),
                             ("sw", e.get("sweeps"))):
                _put(it, key, val)
            it["lic"] = 1 if r.get("li") else 0
            it["dn"] = 1 if r.get("didv") == "num" else 0
            if not r.get("didv") and not r.get("li"):
                # 没有电流列（Z 噪声谱等）或电流读不出：页面不显示偏压范围，也不做前后帧对照。
                it["ex"] = (str(e.get("exp") or "") or "非偏压谱").strip()
        elif k == "g":
            for key, val in (("t1", e.get("t1")), ("gx", e.get("nx")), ("gy", e.get("ny")),
                             ("n", e.get("npts")), ("w", e.get("w_nm")), ("hn", e.get("h_nm")),
                             ("v0", e.get("v0")), ("v1", e.get("v1")), ("zo", e.get("zoff_pm")),
                             ("b", e.get("bias_V")), ("sp", e.get("setpoint_pA")),
                             ("sw", e.get("sweeps")), ("have", e.get("have")),
                             ("cx", e.get("cx_nm")), ("cy", e.get("cy_nm")), ("ang", e.get("angle"))):
                _put(it, key, val)
        else:
            continue
        items.append(it)
    items.sort(key=lambda it: (it["d"], it.get("t") or 0, it["fn"]))
    return items


def make_doc(items: list[dict], roots: list[dict], generated: str) -> dict:
    return {
        "version": INDEX_VERSION,
        "generated": generated,
        "built": True,
        "roots": roots,
        "n_frames": sum(1 for i in items if i["k"] == "f"),
        "n_spectra": sum(1 for i in items if i["k"] == "s"),
        "n_grids": sum(1 for i in items if i["k"] == "g"),
        "last_batch": max((i.get("ad") or "" for i in items), default=""),
        "items": items,
        "degraded": False,
        "detail": None,
    }


def write_index(doc: dict, lay: _paths.Layout) -> None:
    data = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _paths.ensure_state_dir(lay)
    # 先写 json 再写 gz；读的一侧只在 gz 不比 json 旧时才用 gz。
    _paths.atomic_write_bytes(lay.index, data)
    _paths.atomic_write_bytes(lay.index_gz, gzip.compress(data, compresslevel=6, mtime=0))


def read_index_bytes(prefer_gzip: bool, lay: _paths.Layout | None = None) -> tuple[bytes, bool] | None:
    """``(数据, 是否 gzip)``；没构建过 ⇒ None。只读。"""
    lay = lay or _paths.layout()
    try:
        jst = os.stat(lay.index)
    except OSError:
        return None
    if prefer_gzip:
        try:
            gst = os.stat(lay.index_gz)
            if gst.st_mtime_ns >= jst.st_mtime_ns:
                return lay.index_gz.read_bytes(), True
        except OSError:
            pass
    try:
        return lay.index.read_bytes(), False
    except OSError:
        return None


def read_index(lay: _paths.Layout | None = None) -> dict | None:
    got = read_index_bytes(False, lay)
    if got is None:
        return None
    try:
        doc = json.loads(got[0].decode("utf-8"))
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None
