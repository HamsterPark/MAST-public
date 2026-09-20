"""覆盖层加载出来的模块挂在这个包下。

**空包是刻意的**：这里不放任何代码，只提供一个命名空间，让
``mast.skills._overlay.builtins.bias`` 这样的名字能合法地进 ``sys.modules``。

为什么不直接替换 ``sys.modules["mast.skills.builtins.bias"]``：
``SkillRegistry.discover()`` 的冻结兜底扫的就是 ``sys.modules`` 里
``mast.skills.builtins.`` 前缀的东西（``core/registry.py:186-189``），替换之后
**任何新建的 SkillRegistry 都会静默继承覆盖层** —— 一个「行动在远处」的效果。
而且替换并**买不到**「直接 import 也生效」：已经持有旧类引用的地方不会改变。

用独立命名空间还白送一样东西：``skill_adapter.py:1245`` 已经在写
``tool.metadata["skill_source"] = cls.__module__``，于是「这把工具是不是覆盖版」
自动说实话，中间件层一行都不用改。
"""
