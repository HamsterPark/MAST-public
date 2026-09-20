"""装进**随包运行时**的 ``site-packages/sitecustomize.py``。

⚠️ 这个文件住在源码树里（可 lint、可 review、可 diff），但父进程**永不 import**
它，它也**不 import 任何 mast**。构建时被 :mod:`build_pyruntime` 拷成
``<runtime>/Lib/site-packages/sitecustomize.py``。

为什么需要它 —— 一个陷阱的第二次
================================
随包运行时带 ``python313._pth``，而 ``._pth`` 的存在让解释器**默认 isolated**：
忽略 ``PYTHONPATH`` 和 ``PYTHONHOME``，禁 user site。那本来是好事（B1 的第二道
独立机制），但它同时把 :mod:`mast.pyexec.execute` 用来给子进程送会话目录的那条路
掐断了 —— ``PYTHONPATH=<session>`` 不再生效，于是会话里的 ``sitecustomize.py``
和 ``_mast_audit.py`` **都找不到**，审计钩子静默不装。

这是 B-P1 里 ``-I`` 那个坑的同一个陷阱、第二次、从另一条路进来。而这一次更坏：
``-I`` 那次在开发环境就红了；这一次开发环境走的是非 isolated 的 venv，**测试全绿**，
只有要发给用户的那个运行时上保护会消失。

``._pth``/isolated 忽略的只是 ``PYTHONPATH``/``PYTHONHOME`` 这几个**解释器自己的**
变量，任意自定义环境变量照常可读。所以会话目录改用 ``MAST_PYEXEC_SESSION`` 送。

装不上就退出，不是继续
====================
钩子是 B2（不能覆盖或删除已存在的测量文件）的**全部**机制。装不上而继续跑，就是
一个看起来完全正常、实际毫无保护的分析进程 —— 而用户这时正指望它「碰不坏数据」。
所以这里 fail-closed：打印到 stderr 并以 :data:`EXIT_NO_HOOK` 退出，由父进程翻译成
一句说得清的话。

只在 ``MAST_PYEXEC_SESSION`` 有值时强制 —— 没设就说明这不是 MAST 的分析子进程
（有人手工在跑这个 python），那时不该拦。
"""

import os
import sys

#: 钩子装不上时的退出码。挑一个不会和用户脚本撞的值。
EXIT_NO_HOOK = 97


def _bootstrap() -> None:
    session = os.environ.get("MAST_PYEXEC_SESSION", "").strip()
    if not session:
        return                      # 不是 MAST 的分析子进程，不管
    if not os.path.isdir(session):
        sys.stderr.write(
            "[mast-pyexec] MAST_PYEXEC_SESSION 指向的目录不存在：%s\n" % session)
        sys.stderr.flush()
        os._exit(EXIT_NO_HOOK)
    if session not in sys.path:
        sys.path.insert(0, session)
    try:
        import _mast_audit
        _mast_audit.install()
    except BaseException as exc:                       # noqa: BLE001
        # BaseException 而不是 Exception：SystemExit / KeyboardInterrupt 从这里
        # 逃出去也一样意味着钩子没装上。
        sys.stderr.write(
            "[mast-pyexec] 审计钩子装不上（%s: %s）—— 本次拒绝执行。\n"
            "  没有它就没有「不能覆盖已存在的测量文件」这条保护，而一个看起来"
            "完全正常、实际毫无保护的分析进程比直接失败危险得多。\n"
            "  会话目录：%s\n" % (type(exc).__name__, exc, session))
        sys.stderr.flush()
        os._exit(EXIT_NO_HOOK)


_bootstrap()
