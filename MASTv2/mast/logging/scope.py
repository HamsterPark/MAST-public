"""作用域切换 —— 「当前实验 / 当前样品」是一个指针，不是行的状态。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §10

一句话模型
----------

    实验和样品是**永久存在**的记录行；「当前实验/当前样品」是一个与它们正交的、
    全局唯一的、持久化的指针。**切换 = 只改指针，不碰任何行。**

设计放弃了「归档」这个概念：实验可能十年后重启，写总结或报告也不必以归档为前提。

在此之前，「哪个是当前实验」被编码在 ``experiments.status='running'`` 里，于是
切换必须"顺手关掉上一个"（``start_experiment`` 切走时 supersede 旧的）。这**一个**
设计缺陷同时造成了三件事：

1. 陈旧的 active 行 —— ``RightPanel.tsx`` 的注释就在抱怨 ``find(status==="active")``
   会命中过期的行；
2. 顶栏与右栏各自推断"当前实验"，结果可能显示不同的实验；
3. v2 库的 ``ended_at`` 只能写一次（append-only 触发器），一旦 end 就复活不了。

改成单一指针后三个一起消失，而且**不需要为"复活"写任何绕行代码** —— 因为从来
就没有"结束"这个动作。
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScopeChange:
    """一次切换的结果。**永远返回它，不抛异常** —— 失败必须能被说出来。"""

    ok: bool
    changed: bool = False               # False = 已经在那儿了（幂等空转）
    experiment_id: str | None = None
    sample_id: str | None = None
    experiment_name: str = ""
    sample_name: str = ""
    error: str = ""                     # 中文原因：给人看的和给模型看的是同一句
    block_code: str = ""                # "" | unknown_experiment | unknown_sample
                                        # | sample_mismatch | instrument_busy | no_experiment
    can_force: bool = False
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "changed": self.changed,
            "experiment_id": self.experiment_id, "sample_id": self.sample_id,
            "experiment_name": self.experiment_name, "sample_name": self.sample_name,
            "error": self.error, "block_code": self.block_code,
            "can_force": self.can_force, "warnings": list(self.warnings),
        }


_lock = threading.RLock()
_subs: list[Callable[[ScopeChange, ScopeChange], None]] = []
_pool: ThreadPoolExecutor | None = None


def subscribe(fn: Callable[[ScopeChange, ScopeChange], None]) -> Callable[[], None]:
    """注册切换后的回调 ``(old, new)``。返回取消订阅的函数。

    回调是 **post-commit、离线程**的：指针在回调之前就已经持久化了，回调抛异常
    不会让切换失败。实验目录创建、环境 CSV 换句柄、对话导出都挂这里。
    """
    with _lock:
        _subs.append(fn)

    def _unsub() -> None:
        with _lock:
            try:
                _subs.remove(fn)
            except ValueError:
                pass
    return _unsub


def _executor() -> ThreadPoolExecutor:
    """单槽线程池 —— 保证副作用按切换顺序执行。

    单槽（而不是多线程）是必须的：两次快速切换的副作用如果乱序，实验目录和
    CSV 句柄就会指向错的样品。
    """
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mast-scope")
        return _pool


def notify(old: ScopeChange, new: ScopeChange) -> None:
    """把切换事件派发给订阅者。永不抛，永不阻塞调用方。"""
    with _lock:
        subs = list(_subs)
    if not subs:
        return

    def _run() -> None:
        for fn in subs:
            try:
                fn(old, new)
            except Exception as exc:  # noqa: BLE001 — 订阅者失败绝不能影响切换
                logger.warning("scope subscriber %r failed: %r",
                               getattr(fn, "__name__", fn), exc, exc_info=True)
    try:
        _executor().submit(_run)
    except RuntimeError:
        # 解释器关闭中 —— 同步跑一遍，实在不行就算了
        try:
            _run()
        except Exception:  # noqa: BLE001
            pass


def shutdown(wait: bool = False) -> None:
    """关掉线程池（进程退出时用）。"""
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=wait)


def reset_subscribers() -> None:
    """测试用：清空订阅者。"""
    with _lock:
        _subs.clear()
