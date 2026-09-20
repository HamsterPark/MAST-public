"""``mast.pyexec`` —— 在一个**独立解释器进程**里跑数据分析代码。

这个包给 data_processing agent 一个真正的 Python 环境：完整的科学计算栈、
会话式工作目录、可以读遍测量数据、可以并行、可以跑很久。

**不叫 sandbox。** 这个仓库里已经有一个东西叫「沙箱」——
``agents/data_processing/tools.py`` 的 ``run_numpy_snippet``，一个
RestrictedPython 的名字白名单。那个限制的是**名字**（进程内），这个隔离的是
**进程**。两个子系统共用一个名字，是本仓反复被咬的形状。

隔离与能力
==========

只有三条底线，每条都能回答「它在保护什么」，而且都不挡数据处理的正当做法：

============  ====================================  ==========================
底线           保护什么                                机制
============  ====================================  ==========================
B1            **仪器**                               运行时里没有 ``nanonis_spm``、
                                                     没有 ``mast.core.connection``；
                                                     env 从零构造；``-I``
B2            **原始测量数据**（不可重来）             审计钩子只拦一件事：覆盖或
                                                     删除**已存在的**
                                                     ``.sxm/.dat/.3ds/.h5``
B3            **正在采数据的机器**                     Job Object：可杀 + 内存有界。
                                                     上限给得很松 —— 关键不是
                                                     「限制小」，是「能停下来」
============  ====================================  ==========================

B1 同时是「DP 很安全」这个论断的前提：DP 没有 ``SafetyGateMiddleware``、没有
HITL（``agents/data_processing/graph.py:12-13`` 明写它是只读后处理 agent）。
拆掉 B1 等于开一条没有闸门的硬件通道 —— 那要连带给 DP 补上 IC 那整套闸门，
是另一件工程，不是一个开关。

除此之外能力面是宽的：读任何路径、写任何路径（除 B2 那一类）、
``import mastkit`` 直接解析 Nanonis 文件、整个 ``mast/`` 源码只读可查、
``multiprocessing`` 并行、六库科学栈、联网（只拦仪器端口）。

**每加一条限制都要能回答两句：它挡住的合法用途是什么？它保护的东西能不能挽回？**
答不上第二句的一律不加。
"""

from __future__ import annotations

from mast.pyexec.execute import ExecResult, build_env, run
from mast.pyexec.harvest import HarvestResult, harvest, images_for_toolmessage, snapshot
from mast.pyexec.runtime import (
    LEAK_MODULES,
    OPTIONAL_PACKAGES,
    REQUIRED_PACKAGES,
    RuntimeInfo,
    find_runtime,
    probe_runtime,
    reset_runtime_cache,
)
from mast.pyexec.session import PySession, get_session, sessions_root
from mast.pyexec.staging import stage

__all__ = [
    "LEAK_MODULES",
    "OPTIONAL_PACKAGES",
    "REQUIRED_PACKAGES",
    "ExecResult",
    "build_env",
    "HarvestResult",
    "PySession",
    "RuntimeInfo",
    "find_runtime",
    "get_session",
    "harvest",
    "images_for_toolmessage",
    "probe_runtime",
    "reset_runtime_cache",
    "run",
    "sessions_root",
    "snapshot",
    "stage",
]
