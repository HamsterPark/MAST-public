"""拷进会话目录改名 ``sitecustomize.py`` —— 解释器一启动就装上审计钩子。

薄到只有两行是**故意的**：真正的逻辑在 ``_mast_audit`` 里，那个模块可以被测试
直接 import 来验纯函数，而不会在测试进程里装上一个删不掉的钩子。

``site`` 会在任何用户代码之前 import 本模块，**包括每一个 multiprocessing spawn
出来的子进程** —— 所以并行进程和主进程受同一套保护。
"""

import _mast_audit

_mast_audit.install()
