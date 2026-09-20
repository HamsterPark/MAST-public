# -*- coding: utf-8 -*-
"""增量构建：清点 → 渲染与判据 → 重复保存 / 片段 → ``index.json``（设计文档 D8）。

* **只用线程**（陷阱 T8）：打包版的启动器没有 ``multiprocessing.freeze_support()``，Windows
  spawn 一个进程池会把整个 MAST.exe 再拉起来。numpy 的 FFT / 矩阵乘、Pillow 编码会放 GIL。
* **状态目录只在开头解析一次**，之后显式传下去（见 :mod:`mast.gallery.paths`）：构建跑在
  后台线程里，半路重新读 env 的话，测试结束、env 复位之后它就写进真实数据根了。
* 缓存三层都以 ``(size, mtime_ns)`` 为身份（陷阱 T1），每 :data:`SAVE_EVERY` 个任务落一次盘；
  取消在每个任务开始前检查，正在跑的几个会跑完。
* ``only``（id 前缀）只清点 / 渲染 / 判据这一段，但**索引按全部缓存重写** —— 一次局部构建
  不会把索引缩成一天。副本折叠只看得见这一段里的文件（段外的副本下次全量构建再折叠）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from mast.gallery import config as _config
from mast.gallery import inventory as _inv
from mast.gallery import paths as _paths

logger = logging.getLogger(__name__)

SAVE_EVERY = 200
LOG_KEEP = 50
ERRORS_KEEP = 200

#: ``GalleryBuildStatus`` 的字段，一个不多一个不少。
STATUS_KEYS = ("running", "phase", "started", "finished", "done", "total", "n_files", "n_new",
               "n_changed", "n_render", "n_analysis", "n_failed", "errors", "log", "message",
               "degraded", "detail")

TIMING_LABEL = {"frame": "帧", "spectrum": "谱", "grid": "网格", "analysis": "判据"}


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Progress:
    """构建状态（线程安全）。``snapshot()`` 的键正好是 :data:`STATUS_KEYS`。"""

    def __init__(self, echo=None) -> None:
        self._lock = threading.Lock()
        self._echo = echo
        self._s: dict = {
            "running": False, "phase": "idle", "started": None, "finished": None,
            "done": 0, "total": 0, "n_files": 0, "n_new": 0, "n_changed": 0,
            "n_render": 0, "n_analysis": 0, "n_failed": 0, "errors": [], "log": [],
            "message": "", "degraded": False, "detail": None,
        }

    def update(self, **kw) -> None:
        with self._lock:
            self._s.update(kw)

    def add(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._s[key] = int(self._s.get(key) or 0) + n

    def log(self, line: str) -> None:
        with self._lock:
            self._s["log"] = (self._s["log"] + [f"{time.strftime('%H:%M:%S')} {line}"])[-LOG_KEEP:]
        if self._echo is not None:
            self._echo(line)

    def error(self, item_id: str, why) -> None:
        with self._lock:
            self._s["n_failed"] += 1
            if len(self._s["errors"]) < ERRORS_KEEP:
                self._s["errors"].append({"id": str(item_id), "why": str(why)[:300]})

    def snapshot(self) -> dict:
        with self._lock:
            s = {k: self._s.get(k) for k in STATUS_KEYS}
            s["errors"] = [dict(e) for e in self._s["errors"]]
            s["log"] = list(self._s["log"])
        return s


# ── 缓存 ───────────────────────────────────────────────────────────────


def _read_cache(lay: _paths.Layout, name: str) -> dict:
    d = _paths.read_json(lay.cache_file(name), {})
    return d if isinstance(d, dict) else {}


def _save_cache(lay: _paths.Layout, name: str, obj: dict) -> None:
    _paths.atomic_write_json(lay.cache_file(name), obj)


def _remove_thumbs(lay: _paths.Layout, item_id: str) -> None:
    for suffix in (".jpg", ".li.jpg", ".png"):
        try:
            Path(lay.thumbs, *(item_id + suffix).split("/")).unlink()
        except OSError:
            pass


# ── 任务 ───────────────────────────────────────────────────────────────


@dataclass
class Task:
    id: str
    kind: str
    path: str
    summ: dict
    src: list
    prev_render: dict | None = None
    need_z: bool = False
    need_li: bool = False
    need_render: bool = False
    need_analysis: bool = False


def plan_tasks(inv: dict, render: dict, analysis: dict, enabled: set[str], thumbs: Path,
               only: str = "", force: bool = False) -> list[Task]:
    from mast.gallery import render as R
    from mast.gallery.analysis import VER_ANALYSIS

    tasks: list[Task] = []
    for iid, e in inv.items():
        if iid.split("/", 1)[0] not in enabled or e.get("err"):
            continue
        if only and not _inv.in_scope(iid, only):
            continue
        src = [e.get("size"), e.get("mtime_ns")]
        r = render.get(iid)
        same = r is not None and r.get("src") == src
        ok_cached = same and bool(r.get("ok"))
        k = e.get("kind")
        t = Task(id=iid, kind=k, path=e["path"], summ=e, src=src, prev_render=r if same else None)
        if k == "f":
            # 渲染抛过异常（不是「有效行太少」这类数据本身的结论）的，下次构建再试 ——
            # 一次线程竞态或文件正被占用，不该变成永久的「出不了图」。
            retry = same and not r.get("ok") and bool(r.get("error"))
            stale_z = bool(force or not same or r.get("v") != R.VER_FRAME or retry
                           or (ok_cached and not R.thumb_exists(thumbs, r.get("th"))))
            if not stale_z and not ok_cached:
                continue                    # 已知出不了图（有效行太少等），内容没变就不再试
            has_li = any(str(c).upper().startswith("LI_DEMOD_1") for c in (e.get("channels") or []))
            stale_li = has_li and bool(stale_z or r.get("vli") != R.VER_LI
                                       or (bool(r.get("li")) and not R.thumb_exists(thumbs, r.get("li"))))
            a = analysis.get(iid)
            need_an = bool(force or a is None or a.get("src") != src or a.get("v") != VER_ANALYSIS
                           or a.get("ar") == "error")
            if not (stale_z or stale_li or need_an):
                continue
            t.need_z, t.need_li, t.need_analysis = stale_z, stale_li, need_an
        elif k in ("s", "g"):
            ver = R.VER_STS if k == "s" else R.VER_GRID
            stale = bool(force or not same or r.get("v") != ver
                         or (same and not r.get("ok") and bool(r.get("error")))
                         or (ok_cached and not R.thumb_exists(thumbs, r.get("th"))))
            if not stale:
                continue
            t.need_render = True
        else:
            continue
        tasks.append(t)
    return tasks


def _run_task(t: Task, thumbs: Path, cancel: threading.Event) -> dict:
    out: dict = {"id": t.id, "kind": t.kind, "timing": {}}
    if cancel.is_set():
        out["cancelled"] = True
        return out
    from mast.gallery import analysis as A
    from mast.gallery import render as R

    try:
        if t.kind == "f":
            scan = None
            rr = t.prev_render
            if t.need_z or t.need_li:
                from mast.io.nanonis_files import read_sxm

                t0 = time.perf_counter()
                try:
                    scan = read_sxm(t.path)
                    res = R.render_frame(t.path, t.id, t.summ, thumbs, want_z=t.need_z,
                                         want_li=t.need_li, scan=scan)
                except Exception as exc:  # noqa: BLE001
                    why = f"{type(exc).__name__}: {exc}"[:300]
                    out["error"] = why
                    if not t.need_z and rr is not None:
                        # 只是重出 LI 失败：保留原来的 Z 图，记一次「LI 已处理过」免得每次重试。
                        res = dict(rr, vli=R.VER_LI, li="")
                        res.pop("li_ch", None)
                    else:
                        res = {"ok": False, "why": why, "v": R.VER_FRAME, "error": True}
                if not t.need_z and rr is not None and res.get("ok") and "error" not in out:
                    merged = dict(rr)
                    for key in ("li", "li_ch", "vli"):
                        if key in res:
                            merged[key] = res[key]
                    if not res.get("li"):
                        merged.pop("li_ch", None)
                    res = merged
                res["src"] = t.src
                out["render"] = res
                out["timing"]["frame"] = time.perf_counter() - t0
                rr = res
            if t.need_analysis and rr and rr.get("ok"):
                t1 = time.perf_counter()
                w, nx = t.summ.get("w_nm"), t.summ.get("nx")
                nmpp = (float(w) / float(nx)) if (w and nx) else None
                a = A.analyse_frame(t.path, nmpp, rr.get("r0"), rr.get("r1"), scan=scan,
                                    scan_dir=t.summ.get("scan_dir") or "")
                a["src"] = t.src
                out["analysis"] = a
                out["timing"]["analysis"] = time.perf_counter() - t1
                if a.get("err"):
                    out.setdefault("error", "判据：" + str(a["err"]))
        elif t.kind == "s":
            t0 = time.perf_counter()
            try:
                res = R.render_spectrum(t.path, t.id, t.summ, thumbs)
            except Exception as exc:  # noqa: BLE001
                out["error"] = f"{type(exc).__name__}: {exc}"[:300]
                res = {"ok": False, "why": out["error"], "v": R.VER_STS, "error": True}
            res["src"] = t.src
            out["render"] = res
            out["timing"]["spectrum"] = time.perf_counter() - t0
        elif t.kind == "g":
            t0 = time.perf_counter()
            try:
                res = R.render_grid(t.path, t.id, t.summ, thumbs)
            except Exception as exc:  # noqa: BLE001
                out["error"] = f"{type(exc).__name__}: {exc}"[:300]
                res = {"ok": False, "why": out["error"], "v": R.VER_GRID, "error": True}
            res["src"] = t.src
            out["render"] = res
            out["timing"]["grid"] = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 — 任务永不抛到池外
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


# ── 主流程 ─────────────────────────────────────────────────────────────


def run_build(force: bool = False, workers: int | None = None, only: str | None = None,
              progress: Progress | None = None, cancel: threading.Event | None = None,
              state: Path | str | None = None) -> dict:
    """跑一次构建。永不抛；结果在返回的状态快照里（外加 ``timing`` / ``elapsed_s``）。"""
    from mast.gallery.analysis import acquisition_fragments, find_repeat_saves
    from mast.gallery.index import build_items, make_doc, write_index

    prog = progress or Progress()
    cancel = cancel or threading.Event()
    lay = _paths.layout(state)
    only = _inv.normalise_only(only)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    timing: dict[str, list] = defaultdict(lambda: [0.0, 0])
    t_start = time.perf_counter()
    prog.update(running=True, phase="inventory", started=prog.snapshot()["started"] or now_str(),
                finished=None, done=0, total=0, message="清点文件…", detail=None)
    inv: dict = {}
    render: dict = {}
    analysis: dict = {}
    try:
        cfg = _config.load_config(lay)
        nworkers = max(1, min(_config.MAX_WORKERS, int(workers or cfg.workers or _config.default_workers())))
        enabled = {r.name for r in cfg.roots if r.enabled}
        _paths.ensure_state_dir(lay)
        inv = _read_cache(lay, "inventory.json")
        render = _read_cache(lay, "render.json")
        analysis = _read_cache(lay, "analysis.json")
        dups_cache = _read_cache(lay, "dups.json")
        if not cfg.roots:
            prog.log("还没有配置数据根（设置里加一个再构建）")
        missing = [r.name for r in cfg.roots if r.enabled and not os.path.isdir(r.path)]
        if missing:
            prog.log("这些数据根现在不存在，保留它们已有的条目：" + "、".join(missing))

        files, scopes = _inv.walk_roots(cfg, only=only or None, state=lay.state)
        groups = _inv.collapse(files)
        new_ids, changed_ids, failed = _inv.update_inventory(groups, inv, stamp, force=force)
        removed = _inv.purge(inv, scopes, {rep.id for rep, _m in groups})
        for rid in removed:
            render.pop(rid, None)
            analysis.pop(rid, None)
            _remove_thumbs(lay, rid)
        for iid, why in failed:
            prog.error(iid, "读头信息失败：" + why)
        prog.update(n_new=len(new_ids), n_changed=len(changed_ids))
        prog.log("清点%s：%d 个文件 → %d 条（副本折叠后）；新 %d · 变动 %d · 消失 %d · 读失败 %d"
                 % (f"（{only}）" if only else "", len(files), len(groups), len(new_ids),
                    len(changed_ids), len(removed), len(failed)))
        _save_cache(lay, "inventory.json", inv)

        tasks = [] if cancel.is_set() else plan_tasks(inv, render, analysis, enabled, lay.thumbs,
                                                      only, force)
        kinds = defaultdict(int)
        for t in tasks:
            kinds[t.kind] += 1
        prog.update(phase="render", total=len(tasks), done=0,
                    message=f"渲染与判据：{len(tasks)} 个任务")
        prog.log("任务 %d 个（帧 %d · 谱 %d · 网格 %d），workers %d"
                 % (len(tasks), kinds["f"], kinds["s"], kinds["g"], nworkers))
        try:
            if tasks:
                with ThreadPoolExecutor(max_workers=nworkers, thread_name_prefix="gallery") as ex:
                    futs = [ex.submit(_run_task, t, lay.thumbs, cancel) for t in tasks]
                    done = 0
                    for fu in as_completed(futs):
                        res = fu.result()
                        done += 1
                        if not res.get("cancelled"):
                            if "render" in res:
                                render[res["id"]] = res["render"]
                                prog.add("n_render")
                                why = res["render"].get("why")
                                if not res["render"].get("ok") and why not in ("empty", None, "") \
                                        and not res.get("error"):
                                    prog.error(res["id"], why)
                            if "analysis" in res:
                                analysis[res["id"]] = res["analysis"]
                                prog.add("n_analysis")
                            if res.get("error"):
                                prog.error(res["id"], res["error"])
                            for key, secs in res["timing"].items():
                                timing[key][0] += secs
                                timing[key][1] += 1
                        prog.update(done=done)
                        if done % SAVE_EVERY == 0:
                            _save_cache(lay, "render.json", render)
                            _save_cache(lay, "analysis.json", analysis)
                            prog.log("%d/%d  %.0f s" % (done, len(tasks), time.perf_counter() - t_start))
        finally:
            _save_cache(lay, "render.json", render)
            _save_cache(lay, "analysis.json", analysis)

        prog.update(phase="dups", message="识别重复保存与同一次采集的片段…")
        dup, dcache = find_repeat_saves(inv, render, dups_cache)
        seg = acquisition_fragments(inv, render, dup)
        _save_cache(lay, "dups.json", dcache)

        prog.update(phase="index", message="写索引…")
        items = build_items(inv, render, analysis, dup, seg, enabled, lay.thumbs)
        roots = [{"name": r.name, "path": r.path, "enabled": r.enabled,
                  "exists": os.path.isdir(r.path)} for r in cfg.roots]
        write_index(make_doc(items, roots, stamp), lay)

        for key in ("frame", "spectrum", "grid", "analysis"):
            secs, n = timing.get(key, [0.0, 0])
            if n:
                prog.log("%s %d 个，平均 %.3f s/个（workers %d）" % (TIMING_LABEL[key], n, secs / n, nworkers))
        n_f = sum(1 for i in items if i["k"] == "f")
        n_s = sum(1 for i in items if i["k"] == "s")
        n_g = sum(1 for i in items if i["k"] == "g")
        final = "cancelled" if cancel.is_set() else "done"
        msg = ("已取消；" if final == "cancelled" else "") + \
            "索引 %d 条（帧 %d · 谱 %d · 网格 %d），重复保存 %d，用时 %.0f s" % (
                len(items), n_f, n_s, n_g, len(dup), time.perf_counter() - t_start)
        prog.log(msg)
        prog.update(running=False, phase=final, finished=now_str(), n_files=len(items), message=msg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("gallery build failed")
        prog.log(f"构建失败：{type(exc).__name__}: {exc}")
        prog.update(running=False, phase="error", finished=now_str(),
                    message=f"构建失败：{exc}", detail=f"{type(exc).__name__}: {exc}")
    out = prog.snapshot()
    out["timing"] = {k: {"n": n, "total_s": round(s, 2), "avg_s": round(s / n, 3) if n else 0.0}
                     for k, (s, n) in timing.items()}
    out["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    return out


# ── CLI（python -m mast.gallery）─────────────────────────────────────────


def _print(line: str) -> None:
    print(line, flush=True)


def _cmd_build(a) -> int:
    res = run_build(force=a.force, workers=a.workers, only=a.only, progress=Progress(echo=_print))
    _print("阶段 %s · 索引 %d 条 · 新 %d · 变动 %d · 渲染 %d · 判据 %d · 失败 %d · %.1f s"
           % (res["phase"], res["n_files"], res["n_new"], res["n_changed"], res["n_render"],
              res["n_analysis"], res["n_failed"], res["elapsed_s"]))
    for e in res["errors"][:20]:
        _print(f"  失败 {e['id']}：{e['why']}")
    return 0 if res["phase"] in ("done", "cancelled") else 1


def _cmd_config(a) -> int:
    cfg = _config.load_config()
    changed = False
    if a.add_root or a.remove_root:
        remove = {n.strip() for n in a.remove_root}
        inputs = [{"name": r.name, "path": r.path, "enabled": r.enabled}
                  for r in cfg.roots if r.name not in remove]
        for spec in a.add_root:
            name, sep, path = spec.partition("=")
            if not sep:
                name, path = "", spec
            inputs.append({"name": name.strip(), "path": path.strip(), "enabled": True})
        roots, errors = _config.normalise_roots(inputs)
        if errors:
            for e in errors:
                _print("  ✗ " + e)
            return 2
        cfg.roots = roots
        changed = True
    if a.workers:
        cfg.workers = a.workers
        changed = True
    if changed:
        _config.save_config(cfg)
    _print(f"状态目录：{_paths.state_dir()}")
    for r in cfg.roots:
        _print("  %s %s → %s%s" % ("●" if r.enabled else "○", r.name, r.path,
                                   "" if os.path.isdir(r.path) else "（不存在）"))
    if not cfg.roots:
        _print("  （没有数据根）")
    _print(f"workers：{cfg.workers}")
    return 0


def _cmd_import(a) -> int:
    from mast.gallery import marks

    with open(a.path, encoding="utf-8") as fh:
        doc = json.load(fh)
    res = marks.import_marks(doc, key_prefix=a.prefix)
    _print("导入：单条 %d · 系列 %d · 目录 %d · 新标签 %d · 索引里找不到 %d · rev %d"
           % (res["items"], res["series"], res["days"], res["tags_added"], res["unmatched"], res["rev"]))
    return 0


def _cmd_status(a) -> int:
    from mast.gallery import index, marks

    lay = _paths.layout()
    _print(f"状态目录：{lay.state}{'' if lay.state.is_dir() else '（还不存在）'}")
    cfg = _config.load_config(lay)
    for r in cfg.roots:
        _print("  %s %s → %s%s" % ("●" if r.enabled else "○", r.name, r.path,
                                   "" if os.path.isdir(r.path) else "（不存在）"))
    doc = index.read_index(lay)
    if doc:
        _print("索引：%s · 帧 %d · 谱 %d · 网格 %d · 最新一批 %s" % (
            doc.get("generated"), doc.get("n_frames", 0), doc.get("n_spectra", 0),
            doc.get("n_grids", 0), doc.get("last_batch")))
    else:
        _print("索引：还没有构建过")
    try:
        m = marks.load_marks(lay)
        _print("标记：rev %d · 单条 %d · 系列 %d · 更新于 %s" % (
            m.get("rev", 0), len(m.get("items") or {}), len(m.get("series") or {}), m.get("updated") or "—"))
    except Exception as exc:  # noqa: BLE001
        _print(f"标记：读不了（{exc}）")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(prog="python -m mast.gallery",
                                 description="MAST 数据图库：增量预处理 / 数据根 / 导入标记")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="增量构建（只处理新增与变动的文件）")
    b.add_argument("--force", action="store_true", help="忽略渲染/判据缓存，全部重出")
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--only", default=None, help="只处理这个 id 前缀，如 SPM/2001/200109/20010910")
    c = sub.add_parser("config", help="查看/修改数据根")
    c.add_argument("--add-root", action="append", default=[], metavar="NAME=PATH")
    c.add_argument("--remove-root", action="append", default=[], metavar="NAME")
    c.add_argument("--workers", type=int, default=None)
    c.add_argument("--list", action="store_true")
    m = sub.add_parser("import-marks", help="把一份 marks.json 合并进来")
    m.add_argument("path")
    m.add_argument("--prefix", default="", help="键前缀，如旧版兼容格式导入时的 SPM/")
    sub.add_parser("status", help="状态目录、数据根、索引与标记概况")
    a = ap.parse_args(argv)
    return {"build": _cmd_build, "config": _cmd_config, "import-marks": _cmd_import,
            "status": _cmd_status}[a.cmd](a)
