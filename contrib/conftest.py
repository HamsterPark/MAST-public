"""contrib/ 的 pytest 引导 —— 让投稿自带的测试在 tests/ 之外也能跑。

与 ``tests/conftest.py``、``tests/v2/conftest.py`` 做同样的两件事（那两份在 tests/ 之外
拿不到）：

1. 把 ``MASTv2/`` 放到 ``sys.path`` 最前，``import mast`` 解析到 v2 代码；
2. 往 ``sys.modules`` 放一个 ``nanonis_spm`` 替身 —— 没有仪器、没装厂商库也能 import
   ``mast.core``。技能测试从不连接仪器；命令名存在性由 ``scripts/skill_check.py`` 从
   磁盘上的厂商库源码核对，不经过这个替身。

跑法（仓库根）::

    python -m pytest contrib -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_MASTV2_ROOT = str(Path(__file__).resolve().parents[1] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _file = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _file.replace("\\", "/"):
            del sys.modules[_name]

if "nanonis_spm" not in sys.modules:
    _nanonis_mock = MagicMock()
    sys.modules["nanonis_spm"] = _nanonis_mock
    sys.modules["nanonis_spm.Nanonis"] = _nanonis_mock.Nanonis
