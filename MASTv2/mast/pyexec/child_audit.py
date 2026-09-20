"""**跑在子进程里的**审计钩子 —— 保护原始测量数据与仪器端口。

这个文件住在源码树里（可 lint、可 review、可 diff），但每次运行是被**拷贝**进
会话目录、由 ``sitecustomize`` 在解释器启动时 import 的。它必须：

* 不 import 任何 ``mast`` 的东西（子进程里根本没有那个包）；
* 只用标准库；
* 在用户代码之前把钩子装好。

为什么走 ``sitecustomize`` 而不是一个 exec 引导器
================================================
第一版是「引导器 exec 用户脚本」。实测（2026-08-19）撞出两个问题，第二个更严重：

1. ``sys.modules["__main__"]`` 指向引导器而不是脚本，于是 ``multiprocessing``
   的 spawn 子进程按 ``__main__.<fn>`` 找不到函数（``PicklingError``）——
   并行处理一批图这个真需求当场用不了。
2. **spawn 出来的子进程根本不经过引导器，压根不装钩子** —— 保护面漏掉了所有
   并行进程，而那正是最容易批量误删文件的地方。

改成 ``sitecustomize`` 之后：``__main__`` 就是用户脚本本身（multiprocessing 正常），
而 ``sitecustomize`` 会被**每一个** spawn 子进程加载，保护面反而更完整。

顺带纠正此前写下的一条：排除 sitecustomize 的理由是「monkeypatch ``socket``
能被 ``import _socket`` 绕过」——那针对的是 **monkeypatch 这个手法**，不是
sitecustomize 这个**加载点**。在里面装 ``sys.addaudithook`` 一样删不掉。

它拦什么、不拦什么
==================

只有两类，每一类都能回答「它在保护什么」：

1. **覆盖或删除已存在的测量文件** —— 保护原始数据。一次低温 STM 实验几个小时，
   重来不了；而一个 ``open(p,'w')`` 打成 ``'r'``、一个 ``rmtree`` 路径拼错，就是
   这么没的。判据故意收得很窄：**已存在** + **测量后缀** + **不在会话目录里**。

   三个条件各司其职，别混为一谈（变异实测确认过）：

   * **后缀**是挡误伤的那一条。完整 import numpy/scipy/matplotlib/pandas/
     skimage/sklearn 会触发 2000 多次 ``open`` 和几次 ``os.remove``，那些临时
     文件叫 ``h5f7jo5j`` 这种名字，**根本没有后缀** —— 第一个判据就放过去了，
     而且它最便宜，放在最前面。
   * **已存在**是让「新建一个同后缀文件」保持合法的那一条。写
     ``<原名>_leveled.npy`` 或导出一个新 ``.sxm`` 都是正当做法。
   * **不在会话目录**让脚本自己的产物随便改。

   宽一点点就会误伤，而误伤的表现是「运行时坏了」，是最难诊断的一类失败。

2. **连到仪器端口** —— 顺手的一层，不是主保证。真正让子进程碰不到仪器的是它
   **没有 nanonis_spm**：越界得手搓 Nanonis 的二进制协议，那不是犯错能碰到的。
   端口号由父进程传进来（``mast/config.py`` 是可配的，不硬编码）。

**不拦**：读任何路径、写任何非测量文件、``subprocess`` / ``multiprocessing``
（并行处理 200 张图是真需求，整棵树由 Job Object 兜着）、联网到别的端口、
``ctypes``、``import``。

``sys.addaudithook`` 而不是 monkeypatch：钩子**装上就删不掉**。monkeypatch
``open`` 用 ``os.open`` 绕过，monkeypatch ``socket.socket`` 用 ``import _socket``
绕过。
"""

from __future__ import annotations

import json
import os
import sys

# ── 配置（父进程经 env 传入；在用户代码跑起来之前就抓进闭包，改 env 无效）──
_SESSION_ROOT = os.path.normcase(os.path.abspath(
    os.environ.get("MAST_PYEXEC_SESSION", "") or os.getcwd()))
_EXTRA_WRITABLE = tuple(
    os.path.normcase(os.path.abspath(p))
    for p in (os.environ.get("MAST_PYEXEC_WRITABLE", "") or "").split(os.pathsep)
    if p.strip()
)
try:
    _BLOCKED_PORTS = frozenset(
        int(x) for x in (os.environ.get("MAST_PYEXEC_BLOCKED_PORTS", "") or "").split(",")
        if x.strip().isdigit())
except Exception:
    _BLOCKED_PORTS = frozenset()

# 覆盖/删除这些后缀的**已存在**文件 = 毁掉不可重来的测量数据。
_MEASUREMENT_SUFFIXES = frozenset({
    ".sxm", ".3ds", ".dat", ".h5", ".hdf5", ".nc", ".mtrx", ".sm4", ".nid",
})

# 写意图：O_WRONLY|O_RDWR|O_APPEND|O_TRUNC|O_CREAT 任一。
# 必须看 flags 而不只是 mode —— 实测 os.open 触发的 `open` 事件 mode 是 None。
_WRITE_FLAGS = 1 | 2 | 8 | 512 | 256
_WRITE_MODE_CHARS = frozenset("wax+")


class MastDataProtection(PermissionError):
    """写到一个已存在的原始测量文件上时抛出。"""


class MastInstrumentPortBlocked(PermissionError):
    """连仪器端口时抛出。"""


def _protected(path) -> bool:
    """这条路径是不是一个**必须原样留着**的测量文件。

    三个条件缺一不可，顺序按代价从小到大排（这个函数每次 open 都跑）。
    """
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    try:
        s = os.fspath(path)
        if isinstance(s, bytes):
            s = s.decode("utf-8", "replace")
    except Exception:
        return False
    if not s:
        return False
    # 1) 后缀（最便宜的判据，绝大多数调用在这里就返回了）
    dot = s.rfind(".")
    if dot < 0 or s[dot:].lower() not in _MEASUREMENT_SUFFIXES:
        return False
    try:
        full = os.path.normcase(os.path.abspath(s))
    except Exception:
        return False
    # 2) 会话目录（含额外可写区）里的东西是脚本自己的产物，随便改
    if full.startswith(_SESSION_ROOT + os.sep) or full == _SESSION_ROOT:
        return False
    for w in _EXTRA_WRITABLE:
        if full.startswith(w + os.sep) or full == w:
            return False
    # 3) 已存在 —— 新建一个同后缀的文件是完全正当的（派生产物就该这么写）
    try:
        return os.path.exists(full)
    except Exception:
        return False


def _is_write(mode, flags) -> bool:
    if isinstance(mode, str) and mode:
        return bool(_WRITE_MODE_CHARS & set(mode))
    try:
        return bool(int(flags) & _WRITE_FLAGS)
    except Exception:
        return True     # 看不懂就当成写 —— 这里保守是便宜的


def _deny_data(path, what: str):
    raise MastDataProtection(
        f"拒绝{what}原始测量文件：{path}\n"
        "分析是非破坏性的 —— 派生结果请写成新文件（例如 "
        "<原名>_leveled.npy），或写进本次会话的 out/ 目录。\n"
        "（只有「已经存在的」测量文件受此保护；新建文件、写别的位置都不受限。）"
    )


def _audit(event: str, args) -> None:
    # 热路径：绝大多数事件在第一个 if 就走掉了。
    if event == "open":
        path = args[0] if args else None
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if _is_write(mode, flags) and _protected(path):
            _deny_data(path, "覆盖")
        return
    if event in ("os.remove", "os.unlink"):
        if args and _protected(args[0]):
            _deny_data(args[0], "删除")
        return
    if event == "os.rename":
        # src 被移走 = 原文件没了，和删除等价；dst 被覆盖也一样。
        for p in args[:2]:
            if _protected(p):
                _deny_data(p, "移动/覆盖")
        return
    if event == "os.truncate":
        # 实测发现的一条：把一个 .sxm 截断到 0 同样是毁掉它。
        if args and _protected(args[0]):
            _deny_data(args[0], "截断")
        return
    if event == "shutil.rmtree":
        if args:
            _deny_tree(args[0])
        return
    if event in ("socket.connect", "socket.bind") and _BLOCKED_PORTS:
        addr = args[1] if len(args) > 1 else None
        port = None
        if isinstance(addr, (tuple, list)) and len(addr) >= 2:
            port = addr[1]
        if isinstance(port, int) and port in _BLOCKED_PORTS:
            raise MastInstrumentPortBlocked(
                f"拒绝连接端口 {port}：那是仪器的 Nanonis TCP 端口。\n"
                "数据处理环境不驱动仪器 —— 需要仪器动作请交给 instrument_control。"
            )
        return


def _deny_tree(root) -> None:
    """rmtree 要看整棵树里有没有测量文件 —— 只看根目录名是看不出来的。"""
    try:
        r = os.fspath(root)
    except Exception:
        return
    if _protected(r):
        _deny_data(r, "删除")
    try:
        full = os.path.normcase(os.path.abspath(r))
    except Exception:
        return
    if full.startswith(_SESSION_ROOT + os.sep) or full == _SESSION_ROOT:
        return
    for w in _EXTRA_WRITABLE:
        if full.startswith(w + os.sep):
            return
    try:
        for dirpath, _dirnames, filenames in os.walk(r):
            for fn in filenames:
                dot = fn.rfind(".")
                if dot >= 0 and fn[dot:].lower() in _MEASUREMENT_SUFFIXES:
                    raise MastDataProtection(
                        f"拒绝递归删除 {r}：里面有原始测量文件（{fn}）。\n"
                        "要清理请逐个删除你自己产生的文件。")
    except MastDataProtection:
        raise
    except Exception:
        return


_INSTALLED = False


def install() -> None:
    """装钩子。幂等 —— 每个 spawn 子进程会各自 import 一次 sitecustomize。

    钩子一旦装上就**无法卸载**（PEP 578 的设计），这正是它比 monkeypatch 强的
    地方：脚本没有任何办法把它摘掉。
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    sys.addaudithook(_audit)


def describe() -> str:
    """当前生效的保护面（给日志/自检看，也让「装没装上」可被观察）。"""
    return json.dumps({
        "installed": _INSTALLED,
        "session_root": _SESSION_ROOT,
        "extra_writable": list(_EXTRA_WRITABLE),
        "blocked_ports": sorted(_BLOCKED_PORTS),
        "protected_suffixes": sorted(_MEASUREMENT_SUFFIXES),
    }, ensure_ascii=False)
