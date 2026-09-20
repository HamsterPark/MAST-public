# -*- coding: utf-8 -*-
"""出图任务：进程内唯一的出图槽（设计文档 D21，T33）+ 同步的 :func:`run_job`。

* 全进程同一时刻最多一个出图任务；已经在跑时 :func:`start` 原样返回**那个任务**的状态（带 ``kind``）。
* 请求线程只启动、不等待（「UI 绝不冻结」）；取消在每个产物开始前检查。
* **状态目录在启动时解析一次**，显式传给后台线程（与本体 ``gallery.service`` 同一条纪律）。
* 与构建可以同时跑：出图读的是已经写好的 ``index.json`` / ``marks.json`` 与原始文件；
  matplotlib 那一段与构建共用本体的绘图锁，自然串行。
* :func:`_reset_for_tests` 只清内存状态、等后台线程退出，不缓存、不回落到任何路径。
"""
from __future__ import annotations

import threading
import time

from mast.gallery import paths as _paths
from mast.gallery.figures import common as C

LOG_KEEP = 50
ERRORS_KEEP = 200

#: ``GalleryFigureJobStatus`` 的字段，一个不多一个不少。
STATUS_KEYS = ("running", "phase", "kind", "done", "total", "made", "errors", "log", "started",
               "finished", "message", "degraded", "detail")

KINDS = ("marked_frames", "frame_sheet", "grid_sheets", "sts_lines", "sts_stitch",
         "series_slides", "series_stack")


class JobProgress:
    def __init__(self, echo=None) -> None:
        self._lock = threading.Lock()
        self._echo = echo
        self._s: dict = {"running": False, "phase": "idle", "kind": None, "done": 0, "total": 0,
                         "made": [], "errors": [], "log": [], "started": None, "finished": None,
                         "message": "", "degraded": False, "detail": None}

    def update(self, **kw) -> None:
        with self._lock:
            self._s.update(kw)

    def log(self, line: str) -> None:
        with self._lock:
            self._s["log"] = (self._s["log"] + [f"{time.strftime('%H:%M:%S')} {line}"])[-LOG_KEEP:]
        if self._echo is not None:
            self._echo(line)

    def made(self, key: str) -> None:
        with self._lock:
            self._s["made"] = self._s["made"] + [key]

    def error(self, item: str, why) -> None:
        with self._lock:
            if len(self._s["errors"]) < ERRORS_KEEP:
                self._s["errors"] = self._s["errors"] + [{"id": str(item), "why": str(why)[:300]}]
        if self._echo is not None:
            self._echo(f"  ✗ {item}：{why}")

    def step(self) -> None:
        with self._lock:
            self._s["done"] = int(self._s["done"]) + 1

    def snapshot(self) -> dict:
        with self._lock:
            s = {k: self._s.get(k) for k in STATUS_KEYS}
            s["made"] = list(self._s["made"])
            s["errors"] = [dict(e) for e in self._s["errors"]]
            s["log"] = list(self._s["log"])
        return s


# ── 同步执行（CLI、测试、后台线程都走这里）─────────────────────────────


def _plan_units(kind: str, ids: list[str], series: list[str], options: dict, lay: _paths.Layout,
                doc: dict, marks: dict, cache: dict, cancel: threading.Event,
                report=None) -> list[tuple[str, object]]:
    """``[(标签, 无参函数 → 条目 key)]``。标签用于进度日志与 errors；``report(id, why)`` 记「跳过了哪一条」。"""
    by = C.by_id(doc)
    items = marks.get("items") or {}

    def need(i: str, k: str) -> dict:
        it = by.get(i)
        if it is None:
            raise C.JobError("索引里没有这一项（数据根变了，或者还没构建）")
        if it.get("k") != k:
            raise C.JobError("类型不对：要%s，这是%s" % ({"f": "帧", "s": "谱", "g": "网格谱"}[k],
                                                  {"f": "帧", "s": "谱", "g": "网格谱"}.get(it.get("k"), "?")))
        return it

    units: list[tuple[str, object]] = []
    if kind in ("marked_frames", "frame_sheet"):
        from mast.gallery.figures import frames

        if kind == "marked_frames":
            fids = frames.marked_frame_ids(marks, by, int(options.get("max_series_frames") or 100))
        else:
            fids = list(ids)
        for fid in fids:
            units.append((fid, lambda fid=fid: frames.make_frame_sheet(lay, need(fid, "f"), items.get(fid), options)))
        if kind == "marked_frames" and options.get("include_grids", True):
            from mast.gallery.figures import grids

            for it in (x for x in doc.get("items") or [] if x.get("k") == "g"):
                units.append((it["id"], lambda gid=it["id"]: grids.make_grid_sheet(lay, need(gid, "g"), options)))
    elif kind == "grid_sheets":
        from mast.gallery.figures import grids

        gids = list(ids) or [x["id"] for x in doc.get("items") or [] if x.get("k") == "g"]
        for gid in gids:
            units.append((gid, lambda gid=gid: grids.make_grid_sheet(lay, need(gid, "g"), options)))
    elif kind == "sts_lines":
        from mast.gallery.figures import lines

        if not series:
            raise C.JobError("拉线谱出图要至少一个谱系列")
        units.append(("拉线谱 " + "、".join(series),
                      lambda: lines.make_lines(lay, doc, marks, list(series), options, cache)))
    elif kind == "sts_stitch":
        from mast.gallery.figures import stitch

        if ids:
            units.append(("拼接 " + "、".join(ids[:3]) + ("…" if len(ids) > 3 else ""),
                          lambda: stitch.make_stitch(lay, doc, marks, list(ids), options, cache, report)))
        elif options.get("group_by_dir", True):
            groups = stitch.single_groups(doc, marks)
            if not groups:
                raise C.JobError("没有「带标记、是谱、不在任何系列里」的单根谱")
            for d, gids in groups:
                units.append(("拼接 " + d, lambda gids=gids, d=d: stitch.make_stitch(
                    lay, doc, marks, gids, dict(options, group_dir=d), cache, report)))
        else:
            raise C.JobError("拼接出图要给谱的 id，或者 group_by_dir")
    elif kind in ("series_slides", "series_stack"):
        from mast.gallery.figures import series as S

        if len(series) != 1:
            raise C.JobError("旋转系列出图一次一个帧系列")
        sid = series[0]
        if kind == "series_slides":
            units.append(("16:9 拼图 " + sid, lambda: S.make_slides(lay, doc, marks, sid, options)))
        else:
            units.append(("叠加 " + sid, lambda: S.make_stack(lay, doc, marks, sid, options, cancel)))
    else:
        raise C.JobError(f"不认识的出图类型：{kind}")
    return units


def run_job(kind: str, ids=(), series=(), options=None, *, state=None,
            progress: JobProgress | None = None, cancel: threading.Event | None = None) -> dict:
    """跑一个出图任务。永不抛；结果在返回的状态快照里。"""
    from mast.gallery import index as _index
    from mast.gallery import marks as _marks

    prog = progress or JobProgress()
    cancel = cancel or threading.Event()
    lay = _paths.layout(state)
    options = dict(options or {})
    prog.update(running=True, phase="running", kind=kind, done=0, total=0, made=[], errors=[],
                started=prog.snapshot()["started"] or C.now_str(), finished=None,
                message="准备中…", detail=None)
    try:
        if kind not in KINDS:
            raise C.JobError(f"不认识的出图类型：{kind}")
        doc = _index.read_index(lay)
        if not doc or not doc.get("items"):
            raise C.JobError("图库还没有索引 —— 先在「数据根与构建」里构建一次")
        marks = _marks.load_marks(lay)
        cache: dict = {}
        units = _plan_units(kind, list(ids), list(series), options, lay, doc, marks, cache, cancel, prog.error)
        prog.update(total=len(units), message=f"出图：{len(units)} 项")
        prog.log(f"{kind}：{len(units)} 项")
        for label, fn in units:
            if cancel.is_set():
                break
            try:
                key = fn()
                if isinstance(key, (list, tuple)):
                    for k in key:
                        prog.made(k)
                elif key:
                    prog.made(key)
                prog.log(f"✓ {label}")
            except C.JobError as exc:
                prog.error(label, exc)
            except Exception as exc:  # noqa: BLE001 — 一项坏了不拖垮整个任务
                prog.error(label, f"{type(exc).__name__}: {exc}")
            prog.step()
        snap = prog.snapshot()
        final = "cancelled" if cancel.is_set() else "done"
        msg = ("已取消；" if final == "cancelled" else "") + "出了 %d 项，失败 %d 项" % (
            len(snap["made"]), len(snap["errors"]))
        prog.log(msg)
        prog.update(running=False, phase=final, finished=C.now_str(), message=msg)
    except C.JobError as exc:
        prog.log(f"出图失败：{exc}")
        prog.update(running=False, phase="error", finished=C.now_str(), message=str(exc), detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        prog.log(f"出图失败：{type(exc).__name__}: {exc}")
        prog.update(running=False, phase="error", finished=C.now_str(),
                    message=f"出图失败：{exc}", detail=f"{type(exc).__name__}: {exc}")
    return prog.snapshot()


# ── 后台槽（API 用）────────────────────────────────────────────────────

_LOCK = threading.Lock()

#: 当前（或上一个）任务：``(线程, 取消事件, 进度)``。**先启动线程、再整体替换，读的时候一次取出。**
#: ``get_status`` 不拿锁：若先发布进度再 ``start()``，或把线程和进度放在两个全局里分开读，状态请求会读到
#: 「新任务的进度（phase=running）+ 还没启动的线程（running=False）」，轮询方据此当成任务已经结束。
#: 端到端撞上过：刚增量构建完、机器正忙时点「出对比图」，查到的第一次状态就是这个组合。
_SLOT: tuple[threading.Thread | None, threading.Event | None, JobProgress] = (None, None, JobProgress())


def _alive(th: threading.Thread | None) -> bool:
    return th is not None and th.is_alive()


def get_status() -> dict:
    """键正好是 ``GalleryFigureJobStatus`` 的字段。只读。"""
    th, _ev, prog = _SLOT
    snap = prog.snapshot()
    snap["running"] = _alive(th)
    return snap


def start(kind: str, ids=(), series=(), options=None) -> dict:
    global _SLOT
    with _LOCK:
        if _alive(_SLOT[0]):
            return get_status()
        lay = _paths.layout()
        cancel_ev = threading.Event()
        prog = JobProgress()
        prog.update(running=True, phase="running", kind=kind, started=C.now_str(), message="准备中…")
        th = threading.Thread(target=run_job, args=(kind, list(ids), list(series), dict(options or {})),
                              kwargs={"state": lay.state, "progress": prog, "cancel": cancel_ev},
                              name="gallery-figures", daemon=True)
        th.start()
        _SLOT = (th, cancel_ev, prog)
    return get_status()


def cancel() -> dict:
    with _LOCK:
        th, ev, prog = _SLOT
        if _alive(th) and ev is not None:
            ev.set()
            prog.update(message="正在取消…（正在出的这一张会出完）")
    return get_status()


def wait(timeout: float | None = None) -> bool:
    th = _SLOT[0]
    if th is not None:
        th.join(timeout)
    return not _alive(_SLOT[0])


def _reset_for_tests(timeout: float = 120.0) -> None:
    global _SLOT
    with _LOCK:
        th, ev, _prog = _SLOT
    if _alive(th):
        if ev is not None:
            ev.set()
        th.join(timeout)
    with _LOCK:
        _SLOT = (None, None, JobProgress())
