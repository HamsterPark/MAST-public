# -*- coding: utf-8 -*-
"""标记：``marks.json`` 是唯一真源（设计文档 D11；语义逐字照旧版兼容格式的存盘服务）。

结构::

    items   {id: {r 评级 2/1/-1, tags, note, t 标记时刻, ts 毫秒戳, tt, k, meta, anchor?}}
            anchor（谱 / 网格）= {id 帧 id, fn, rel prev|next, dt 秒, desc, u, v, inside}
    series  {系列号: {name, k f|s|g|mix, ids [id…], r, tags, note, t, ts, anchor?}}
    days    {目录键: {done, note, ts}}      —— 名字沿用旧版兼容格式（那边的目录是日期），导入导出可往返
    tags    标签表
    tomb / stomb   删除时刻（毫秒戳），挡住迟到的旧改动

并发（陷阱 T23）：客户端每次改动带毫秒戳 ``ts``；服务端丢弃比「现存 ts 与墓碑 ts」都旧的改动。
每次 patch 在进程内锁里「读 → 改 → 原子写」，再顺手重写 ``marks.md`` / ``marks.csv`` /
``marks_series.csv``（csv 被 Excel 锁着时跳过，下次再写 —— 陷阱 T24），并按 30 分钟节流留快照。

**损坏的 ``marks.json`` 绝不当成空文档**：那样下一次存盘就会用空文档覆盖掉操作员的全部标记。
读失败直接抛，API 层给 degraded。
"""
from __future__ import annotations

import copy
import csv
import io
import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path

from mast.gallery import paths as _paths
from mast.gallery.inventory import dir_key, dir_label

logger = logging.getLogger(__name__)

DEFAULT_TAGS = ["原子分辨佳", "超结构", "空位/缺陷", "台阶/边界", "偏压系列", "针尖变化",
                "dI/dV", "作图候选", "漂移"]
RATING = {2: "★ 重点", 1: "✓ 可用", -1: "✗ 排除", 0: ""}
KIND = {"f": "帧", "s": "谱", "g": "网格", "mix": "混合"}
BACKUP_EVERY_S = 1800

_LOCK = threading.RLock()
_TS_LOCK = threading.Lock()
_LAST_TS = [0]
_IDX_LOCK = threading.Lock()
_IDX_CACHE: dict = {"key": None, "by": {}, "dirs": set()}


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def next_ts() -> int:
    with _TS_LOCK:
        _LAST_TS[0] = max(int(time.time() * 1000), _LAST_TS[0] + 1)
        return _LAST_TS[0]


# ── 读 ─────────────────────────────────────────────────────────────────


def load_marks(lay: _paths.Layout | None = None) -> dict:
    """读标记文档；没有文件时回默认。**只读**；文件损坏时抛（见模块说明）。"""
    lay = lay or _paths.layout()
    try:
        with open(lay.marks, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        doc = {}
    if not isinstance(doc, dict):
        raise ValueError(f"{lay.marks} 不是一个 JSON 对象 —— 拒绝当成空标记（否则下一次存盘会覆盖它）")
    for key, default in (("version", 1), ("rev", 0), ("updated", ""), ("tags", list(DEFAULT_TAGS)),
                         ("items", {}), ("series", {}), ("days", {}), ("tomb", {}), ("stomb", {})):
        doc.setdefault(key, default)
    return doc


def is_empty(m) -> bool:
    return (not m or bool(m.get("del"))
            or (not m.get("r") and not m.get("tags") and not (m.get("note") or "").strip()
                and not m.get("anchor")))


def apply_patch(doc: dict, p: dict) -> None:
    """旧版兼容格式 ``apply_patch`` 的逐字移植。"""
    p = p or {}
    tomb, stomb = doc.setdefault("tomb", {}), doc.setdefault("stomb", {})
    items, series, days = doc.setdefault("items", {}), doc.setdefault("series", {}), doc.setdefault("days", {})
    for k, m in (p.get("items") or {}).items():
        ts = (m or {}).get("ts") or 0
        if ts and ts < max((items.get(k) or {}).get("ts") or 0, tomb.get(k, 0)):
            continue                                  # 迟到的旧改动
        if is_empty(m):
            items.pop(k, None)
            tomb[k] = ts
        else:
            items[k] = m
            tomb.pop(k, None)
    for sid, s in (p.get("series") or {}).items():
        ts = (s or {}).get("ts") or 0
        if ts and ts < max((series.get(sid) or {}).get("ts") or 0, stomb.get(sid, 0)):
            continue
        if not s or s.get("del") or not s.get("ids"):
            series.pop(sid, None)
            stomb[sid] = ts
        else:
            series[sid] = s
            stomb.pop(sid, None)
    for d, v in (p.get("days") or {}).items():
        ts = (v or {}).get("ts") or 0
        if ts and ts < ((days.get(d) or {}).get("ts") or 0):
            continue
        days[d] = v or {}
    if isinstance(p.get("tags"), list):
        doc["tags"] = [str(t) for t in p["tags"] if str(t).strip()]


# ── 写 ─────────────────────────────────────────────────────────────────


def patch_marks(patch: dict, lay: _paths.Layout | None = None) -> tuple[int, str]:
    lay = lay or _paths.layout()
    with _LOCK:
        doc = load_marks(lay)
        apply_patch(doc, patch or {})
        doc["rev"] = int(doc.get("rev") or 0) + 1
        doc["updated"] = now_str()
        write_all(doc, lay)
        return doc["rev"], doc["updated"]


def write_all(doc: dict, lay: _paths.Layout) -> None:
    _paths.ensure_state_dir(lay)
    _paths.atomic_write_text(lay.marks, json.dumps(doc, ensure_ascii=False, indent=1))
    try:
        _backup_if_due(lay)
    except OSError as exc:
        logger.warning("gallery marks backup failed: %s", exc)
    try:
        by, _dirs = items_meta(lay)
        roots = _root_paths(lay)
        _paths.atomic_write_text(lay.marks_md, render_md(doc, by, roots), tolerate_locked=True)
        items_csv, series_csv = render_csv(doc, by, roots)
        _paths.atomic_write_bytes(lay.marks_csv, items_csv, tolerate_locked=True)
        _paths.atomic_write_bytes(lay.marks_series_csv, series_csv, tolerate_locked=True)
    except Exception:  # noqa: BLE001 — 派生文件写不成不能让这次存盘失败
        logger.exception("gallery marks: derived md/csv not written")


def _backup_if_due(lay: _paths.Layout) -> bool:
    bk = lay.marks_backup
    bk.mkdir(parents=True, exist_ok=True)
    snaps = sorted(f for f in os.listdir(bk) if f.startswith("marks_") and f.endswith(".json"))
    if snaps and time.time() - os.path.getmtime(bk / snaps[-1]) <= BACKUP_EVERY_S:
        return False
    shutil.copy2(lay.marks, bk / f"marks_{time.strftime('%Y%m%d_%H%M%S')}.json")
    return True


# ── 索引元数据（给 md / csv 配参数与路径）─────────────────────────────────


def items_meta(lay: _paths.Layout | None = None) -> tuple[dict, set]:
    """``({id: 索引条目}, {目录键})``；按索引文件的路径 + mtime_ns + size 缓存。只读。"""
    lay = lay or _paths.layout()
    p = lay.index
    try:
        st = os.stat(p)
    except OSError:
        return {}, set()
    key = (str(p), st.st_mtime_ns, st.st_size)
    with _IDX_LOCK:
        if _IDX_CACHE["key"] == key:
            return _IDX_CACHE["by"], _IDX_CACHE["dirs"]
    try:
        doc = json.loads(Path(p).read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return {}, set()
    by = {it["id"]: it for it in (doc.get("items") or []) if isinstance(it, dict) and "id" in it}
    dirs = {it["d"] for it in by.values() if it.get("d")}
    with _IDX_LOCK:
        _IDX_CACHE.update(key=key, by=by, dirs=dirs)
    return by, dirs


def _root_paths(lay: _paths.Layout) -> dict[str, str]:
    try:
        from mast.gallery.config import load_config

        return {r.name: r.path for r in load_config(lay).roots}
    except Exception:  # noqa: BLE001
        return {}


def full_path(k: str, by: dict, roots: dict) -> str | None:
    it = by.get(k)
    if it and it.get("p"):
        return str(it["p"])
    name, _sep, rest = str(k).partition("/")
    base = roots.get(name)
    if base and rest:
        return str(Path(base, *rest.split("/")))
    return None


def fmt_t(t) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(t)) if t else "?"


def meta_of(it: dict) -> str:
    k = it.get("k")
    if k == "f":
        return "%s · %g nm · %+.2f V · %.0f pA" % (fmt_t(it.get("t")), it.get("w") or 0,
                                                  it.get("b") or 0, it.get("sp") or 0)
    if k == "s":
        return "%s · %d 点 %.2f…%.2f V · (%.2f, %.2f) nm" % (
            fmt_t(it.get("t")), it.get("n") or 0, it.get("v0") or 0, it.get("v1") or 0,
            it.get("x") or 0, it.get("y") or 0)
    return "%s · 网格 %s×%s · %g nm" % (fmt_t(it.get("t")), it.get("gx"), it.get("gy"), it.get("w") or 0)


def short(fn: str) -> str:
    """清单里的短编号：文件名末尾 ≥3 位的计数器取最后 4 位，否则整个 stem。"""
    stem = str(fn or "").rsplit(".", 1)[0]
    m = re.search(r"(\d{3,})$", stem)
    if m and len(stem) > len(m.group(1)):
        return m.group(1)[-4:]
    return stem


def _rows(doc: dict) -> list[tuple[str, str, dict]]:
    out = [(dir_key(k), k, m) for k, m in (doc.get("items") or {}).items() if not is_empty(m)]
    out.sort(key=lambda x: (x[0], x[2].get("tt") or 0, x[1]))
    return out


def _series_of(doc: dict) -> dict[str, list[str]]:
    mem: dict[str, list[str]] = {}
    for sid, s in (doc.get("series") or {}).items():
        for k in s.get("ids", []):
            mem.setdefault(k, []).append(s.get("name") or sid)
    return mem


def _series_sorted(doc: dict, by: dict):
    def first_t(s):
        ts = [by[k]["t"] for k in s.get("ids", []) if k in by and by[k].get("t")]
        return min(ts) if ts else 0
    return sorted((doc.get("series") or {}).items(), key=lambda kv: first_t(kv[1]))


def _anchor_text(a: dict | None) -> str:
    if not a:
        return ""
    return "%s（%s）" % (short(a.get("fn", "")), a.get("desc", ""))


def render_md(doc: dict, by: dict, roots: dict) -> str:
    rs = _rows(doc)
    mem = _series_of(doc)
    n = {r: sum(1 for _d, _k, m in rs if m.get("r") == r) for r in (2, 1, -1)}
    L = ["# 数据标记清单", "",
         "更新于 %s · 单条 %d（★ 重点 %d · ✓ 可用 %d · ✗ 排除 %d）· 系列 %d。由数据图库自动生成，别手改。"
         % (doc.get("updated") or "", len(rs), n[2], n[1], n[-1], len(doc.get("series") or {})), ""]
    if doc.get("series"):
        L += ["## 系列", ""]
        for sid, s in _series_sorted(doc, by):
            ids = s.get("ids", [])
            its = sorted((by[k] for k in ids if k in by), key=lambda it: it.get("t") or 0)
            rng = "%s → %s" % (short(its[0]["fn"]), short(its[-1]["fn"])) if its else ""
            tr = ("%s → %s" % (fmt_t(its[0].get("t")), fmt_t(its[-1].get("mt") or its[-1].get("t")))
                  if its else "")
            L.append("### %s %s" % (RATING.get(s.get("r", 0), "") or "·", s.get("name") or sid))
            L.append("- %d 个%s · %s · %s · 系列号 `%s`" % (len(ids), KIND.get(s.get("k"), ""), rng, tr, sid))
            if s.get("tags"):
                L.append("- 标签：" + " ".join("`%s`" % t for t in s["tags"]))
            if s.get("anchor"):
                L.append("- 位置系于 " + _anchor_text(s["anchor"]))
            if (s.get("note") or "").strip():
                L.append("- 备注：" + s["note"].strip().replace("\n", "\n  "))
            L.append("- 成员：")
            for it in its:
                L.append("  - %s　%s　`%s`" % (it["fn"], meta_of(it), full_path(it["id"], by, roots) or it["id"]))
            for k in ids:
                if k not in by:
                    L.append("  - `%s`（图库里没有）" % (full_path(k, by, roots) or k))
            L.append("")
    days = doc.get("days") or {}
    dirs = sorted({d for d, _k, _m in rs}
                  | {d for d, v in days.items() if (v or {}).get("note") or (v or {}).get("done")})
    for d in dirs:
        dv = days.get(d) or {}
        label = dir_label(d)
        head = label if label == d else f"{label}　`{d}`"
        L.append("## %s%s" % (head, "　（已过完）" if dv.get("done") else ""))
        if (dv.get("note") or "").strip():
            L += ["", "> " + dv["note"].strip().replace("\n", "\n> ")]
        L.append("")
        for dd, k, m in rs:
            if dd != d:
                continue
            extra = "".join("　▤ %s" % nm for nm in mem.get(k, []))
            if m.get("anchor"):
                extra += "　⌖ 位置系于 " + _anchor_text(m["anchor"])
            tags = " ".join("`%s`" % t for t in m.get("tags", []))
            L.append("- **%s** %s　%s　%s%s" % (RATING.get(m.get("r", 0), "") or "·", k.split("/")[-1],
                                             m.get("meta", ""), tags, extra))
            if (m.get("note") or "").strip():
                L.append("  - " + m["note"].strip().replace("\n", "\n    "))
            fp = full_path(k, by, roots)
            L.append("  - `%s`" % fp if fp else "  - `%s`（图库里没有）" % k)
        L.append("")
    return "\n".join(L)


def render_csv(doc: dict, by: dict, roots: dict) -> tuple[bytes, bytes]:
    """``(单条 csv, 系列 csv)``，utf-8-sig（Excel 认得中文）。"""
    mem = _series_of(doc)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["目录", "文件", "评级", "标签", "备注", "系列", "位置系于", "参数", "标记时间", "路径"])
    for d, k, m in _rows(doc):
        w.writerow([d, k.split("/")[-1], RATING.get(m.get("r", 0), ""), ";".join(m.get("tags", [])),
                    m.get("note", ""), ";".join(mem.get(k, [])),
                    ("%s（%s）" % ((m.get("anchor") or {}).get("fn", ""), (m.get("anchor") or {}).get("desc", "")))
                    if m.get("anchor") else "",
                    m.get("meta", ""), m.get("t", ""), full_path(k, by, roots) or k])
    items_csv = buf.getvalue().encode("utf-8-sig")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["系列", "评级", "标签", "备注", "类型", "成员数", "首", "末", "位置系于", "系列号", "成员路径"])
    for sid, s in _series_sorted(doc, by):
        its = sorted((by[k] for k in s.get("ids", []) if k in by), key=lambda it: it.get("t") or 0)
        a = s.get("anchor") or {}
        w.writerow([s.get("name", ""), RATING.get(s.get("r", 0), ""), ";".join(s.get("tags", [])),
                    s.get("note", ""), KIND.get(s.get("k"), ""), len(s.get("ids", [])),
                    its[0]["fn"] if its else "", its[-1]["fn"] if its else "",
                    ("%s（%s）" % (a.get("fn", ""), a.get("desc", ""))) if a else "", sid,
                    ";".join(full_path(k, by, roots) or k for k in s.get("ids", []))])
    return items_csv, buf.getvalue().encode("utf-8-sig")


# ── 导入 / 导出 ────────────────────────────────────────────────────────


def _prefix(p: str) -> str:
    p = str(p or "").strip().replace("\\", "/").lstrip("/")
    return p if (not p or p.endswith("/")) else p + "/"


def _map_dir_key(d: str, pre: str, dirs: set) -> str:
    """旧版兼容格式的目录键是最后一段（日期）；按最后一段在索引的目录键里找唯一匹配，找不到加前缀。"""
    if not pre and d in dirs:
        return d
    cands = sorted(x for x in dirs if x.rsplit("/", 1)[-1] == d and (not pre or x.startswith(pre)))
    if len(cands) == 1:
        return cands[0]
    return pre + d if pre else d


def import_marks(doc_in: dict, key_prefix: str = "", lay: _paths.Layout | None = None) -> dict:
    """合并一份 marks.json（设计文档 D11）。

    单条 / 系列：本地没有，或导入方的 ``t`` 更新时写入（新的毫秒戳，压过旧墓碑）；
    目录：本地没有时写入；标签：并集。``key_prefix`` 作用于单条键、系列成员与系定帧 id。
    在当前索引里找不到的键**照样导入**，只计数。"""
    if not isinstance(doc_in, dict):
        raise ValueError("导入的不是一个 marks.json 对象")
    lay = lay or _paths.layout()
    pre = _prefix(key_prefix)
    with _LOCK:
        doc = load_marks(lay)
        by, dirs = items_meta(lay)
        touched: set[str] = set()
        n_items = n_series = n_days = 0
        for k, m in (doc_in.get("items") or {}).items():
            if not isinstance(m, dict) or is_empty(m):
                continue
            nk = pre + str(k)
            m2 = copy.deepcopy(m)
            m2.pop("del", None)
            anc = m2.get("anchor")
            if isinstance(anc, dict) and anc.get("id"):
                anc["id"] = pre + str(anc["id"])
                touched.add(anc["id"])
            m0 = (doc.get("items") or {}).get(nk)
            if m0 is None or str(m2.get("t") or "") > str(m0.get("t") or ""):
                m2["ts"] = next_ts()
                apply_patch(doc, {"items": {nk: m2}})
                n_items += 1
                touched.add(nk)
        for sid, s in (doc_in.get("series") or {}).items():
            if not isinstance(s, dict) or s.get("del") or not s.get("ids"):
                continue
            s2 = copy.deepcopy(s)
            s2["ids"] = [pre + str(i) for i in s2.get("ids") or []]
            anc = s2.get("anchor")
            if isinstance(anc, dict) and anc.get("id"):
                anc["id"] = pre + str(anc["id"])
                touched.add(anc["id"])
            s0 = (doc.get("series") or {}).get(sid)
            if s0 is None or str(s2.get("t") or "") > str(s0.get("t") or ""):
                s2["t"] = now_str()
                s2["ts"] = next_ts()
                apply_patch(doc, {"series": {sid: s2}})
                n_series += 1
                touched.update(s2["ids"])
        for d, v in (doc_in.get("days") or {}).items():
            if not isinstance(v, dict) or (not v.get("done") and not str(v.get("note") or "").strip()):
                continue
            nd = _map_dir_key(str(d), pre, dirs)
            if nd not in doc["days"]:
                doc["days"][nd] = dict(v, ts=next_ts())
                n_days += 1
        tags = list(doc.get("tags") or [])
        added = 0
        for t in doc_in.get("tags") or []:
            if str(t).strip() and t not in tags:
                tags.append(str(t))
                added += 1
        if added:
            doc["tags"] = tags
        if n_items or n_series or n_days or added:
            doc["rev"] = int(doc.get("rev") or 0) + 1
            doc["updated"] = now_str()
            write_all(doc, lay)
        return {"items": n_items, "series": n_series, "days": n_days, "tags_added": added,
                "unmatched": sum(1 for i in touched if i not in by),
                "rev": int(doc.get("rev") or 0), "updated": str(doc.get("updated") or "")}


def export_file(fmt: str, lay: _paths.Layout | None = None) -> tuple[bytes, str, str]:
    """``(内容, media_type, 文件名)``。现场从 ``marks.json`` 生成，**只读**。"""
    lay = lay or _paths.layout()
    doc = load_marks(lay)
    if fmt == "json":
        return (json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8"),
                "application/json", "marks.json")
    by, _dirs = items_meta(lay)
    roots = _root_paths(lay)
    if fmt == "md":
        return render_md(doc, by, roots).encode("utf-8"), "text/markdown; charset=utf-8", "marks.md"
    if fmt in ("csv", "series_csv"):
        items_csv, series_csv = render_csv(doc, by, roots)
        if fmt == "csv":
            return items_csv, "text/csv; charset=utf-8", "marks.csv"
        return series_csv, "text/csv; charset=utf-8", "marks_series.csv"
    raise ValueError(f"不认识的导出格式：{fmt}")
