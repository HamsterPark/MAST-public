# -*- coding: utf-8 -*-
"""进程内唯一的后台构建（API 用）。

* 全进程同一时刻最多一个构建；已经在跑时 :func:`start_build` 原样返回当前状态。
* 请求线程只启动、不等待（「UI 绝不冻结」）。
* **状态目录在启动时解析一次**，交给后台线程全程使用 —— 见 :mod:`mast.gallery.paths`。
* :func:`_reset_for_tests` 只清内存里的状态、等后台线程退出，不重新解析或缓存路径。
"""
from __future__ import annotations

import threading

from mast.gallery import build as _build
from mast.gallery import paths as _paths

_LOCK = threading.Lock()

#: 当前（或上一次）构建：``(线程, 取消事件, 进度)``。**先启动线程、再整体替换，读的时候一次取出。**
#: ``get_status`` 不拿锁：若先发布进度再 ``start()``，或把线程和进度放在两个全局里分开读，状态请求会读到
#: 「新构建的进度（phase=inventory）+ 还没启动的线程（running=False）」，轮询方据此当成构建已经结束。
#: 出图任务槽（:mod:`mast.gallery.figures.service`）在端到端里撞上过同一个窗口。
_SLOT: tuple[threading.Thread | None, threading.Event | None, _build.Progress] = (None, None, _build.Progress())


def _alive(th: threading.Thread | None) -> bool:
    return th is not None and th.is_alive()


def get_status() -> dict:
    """键正好是 ``GalleryBuildStatus`` 的字段。只读。"""
    th, _ev, prog = _SLOT
    snap = prog.snapshot()
    snap["running"] = _alive(th)
    return {k: snap.get(k) for k in _build.STATUS_KEYS}


def _run(lay: _paths.Layout, force: bool, prog: _build.Progress, cancel: threading.Event) -> None:
    try:
        _build.run_build(force=force, progress=prog, cancel=cancel, state=lay.state)
    except Exception as exc:  # noqa: BLE001 — run_build 自己不抛，这里只是兜底
        prog.update(running=False, phase="error", finished=_build.now_str(),
                    message=f"构建失败：{exc}", detail=str(exc))


def start_build(force: bool = False) -> dict:
    global _SLOT
    with _LOCK:
        if _alive(_SLOT[0]):
            return get_status()
        lay = _paths.layout()
        cancel = threading.Event()
        prog = _build.Progress()
        prog.update(running=True, phase="inventory", started=_build.now_str(), message="准备中…")
        th = threading.Thread(target=_run, args=(lay, bool(force), prog, cancel),
                              name="gallery-build", daemon=True)
        th.start()
        _SLOT = (th, cancel, prog)
    return get_status()


def cancel_build() -> dict:
    with _LOCK:
        th, ev, prog = _SLOT
        if _alive(th) and ev is not None:
            ev.set()
            prog.update(message="正在取消…（已经在跑的几个任务会跑完）")
    return get_status()


def wait(timeout: float | None = None) -> bool:
    """等后台构建结束（测试与 CLI 用）。返回是否已经没有在跑的构建。"""
    th = _SLOT[0]
    if th is not None:
        th.join(timeout)
    return not _alive(_SLOT[0])


def _reset_for_tests(timeout: float = 60.0) -> None:
    global _SLOT
    with _LOCK:
        th, ev, _prog = _SLOT
    if _alive(th):
        if ev is not None:
            ev.set()
        th.join(timeout)
    with _LOCK:
        _SLOT = (None, None, _build.Progress())
