"""源码级断言的取源工具 —— **`inspect.getsource` 的安全替代**(2026-08-15)。

放在 ``tests/v2/`` 顶层而不是某个测试目录里,是因为它**没有任何依赖**
(只用 stdlib),而需要它的测试分散在 campaign / skills / api 好几个目录。
与 ``tests/v2/toolcall.py`` 同一层、同一个理由。

⚠️ **不是 conftest**:conftest 是共享文件,多条线同时在动;一个普通模块由需要它的
文件自己 import,谁都不碰谁。

## 为什么不用 ``inspect.getsource``

它按**导入那一刻**记下的 ``co_firstlineno`` 去切**当前磁盘上**的文件。共用工作树上
这两者会脱节:另一个 agent 在跑测试的那几分钟里往文件上方插了几行,``getsource``
返回的就**不再是那个函数的源码**,而是一段错位的切片。
(评审 2026-08-15 用一次性模块做对照实验坐实了机理:import 之后往文件上方插
8 行,``getsource(target)`` 返回的是**另一个函数**的源码。)

**两种坏法,第二种更坏**:

* ``assert "X" in getsource(f)``     → **假红**(吵,但会被查);
* ``assert "Y" not in getsource(f)`` → **假绿** —— 它悄悄不再守任何东西。

而**守得越贵的规矩,它的绊线失效时越没人会发现**,因为没有人会去查一条绿着的测试。
这个形状恰好落在「『读不到』不许被折叠成一个具体的值」这条规矩的绊线上。

## 这里怎么做

**重读一次文件 + ``ast`` 按名字定位**。源码与节点来自**同一次读取**,错位在结构上
不可能。文件被写到一半时抛 ``SyntaxError`` —— 那是一个**看得见**的失败,
比静默错位好得多。

## 什么时候**不**需要它

``inspect.getsource(模块)`` 与 ``Path(mod.__file__).read_text()`` 读的是**整个文件**,
不存在错位,只怕读到中间态(而那会以 `SyntaxError` 或明显的内容缺失暴露)。
那两种写法是**较安全的一档,别顺手一起改** —— 多改一处不如少改一处正确。

## 该用哪一种:按**你要断言的是什么**挑,不是按顺手

(这张表第二行**是实测出来的,不是推的**。)

===========================  ========================  ==========================
要断言的东西                 用什么                     为什么
===========================  ========================  ==========================
访问了哪个属性 /             ``fn.__code__.co_names``  名字访问,取自已加载的
调了哪个模块级函数                                     code object,**完全不碰磁盘**
**字符串 / 数值字面量**      ``fn.__code__.co_consts`` ⚠️ ``co_names``
(SQL 列名、写死的常数)                                 **看不见字符串字面量** ——
                                                       m3a 实测 ``summary`` 的
                                                       ``co_names`` 里**没有**它
                                                       SELECT 的那些列名
表达式**形状**               ``source_of`` + ``ast``   既不是名字也不是常量,
(``float(x.get(k) or 0.0)``)                           是**模式**
控制流**形状**               ``source_of`` + ``ast``   同上
(有没有吞异常的 except)
===========================  ========================  ==========================

⚠️ **`co_*` 这两条各自也要配自检**:断言一个「它确实会命中的东西**在**里面」。
否则「这类访问 ``co_names`` 根本抓不到」时,你那条 ``not in`` 同样通过 ——
换了个地方的同一个假绿。

⚠️ **字符串匹配守的是「一种写法」,不是「那个语义」。** ``'x or 0.0' not in src``
挡不住 ``or 0``、换行、改空格、去掉外层 ``float()``。要守语义就得用 ``ast`` 判结构
(例:``BoolOp(Or)`` 末项是数值 0 字面量)。**如果这一轮不做,就在注释里写明
「这条只守这一个写法,换写法绕得过去」—— 说出来的缺口不是缺口。**
现成范例见 ``test_forge_time_budget_and_merge.py`` 的
``test_no_measurement_is_folded_to_zero_in_the_pulse_path``,连同它踩到的那个坑:
**同一个 `X or 0` 形状,在计数器上是对的、在测量量上是错的** ⇒ 只能按「量」分,
不能按写法分,所以那条结构判据必须配一条「合法写法不许被判红」的反向自检。

## 同一份实现的另一处

``tests/v2/unit/campaign/_harness.py::source_of`` 是这个函数的**出处**(campaign 线
2026-08-15 先写的,论证一模一样)。本模块把它搬到无依赖的位置,好让 skills 那边也能用
—— 那个 harness 在 import 时会做 ``sys.path`` 手术并清 ``sys.modules`` 里的 ``mast.*``,
composite 的测试 import 不起。
**待办(归 campaign 线,不归本文件)**:把 ``_harness.source_of`` 改成从这里 re-export,
两处实现就并成一份。在那之前它是已知的第二份,而不是没人注意到的第二份。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path


def source_of(obj) -> str:
    """``obj``(函数 / 方法 / 类)的**当前**源码。支持 ``Cls.method``。

    先找类再在类体里找方法,所以嵌套名字(``__qualname__``)必须能在 AST 上逐级走通;
    走不通就**抛**,不返回空串 —— 一个空串会让所有 ``not in`` 断言当场变成假绿,
    那正是这个模块要消灭的东西。
    """
    func = getattr(obj, "__func__", obj)
    module = inspect.getmodule(func)
    path = getattr(module, "__file__", "") or ""
    if not path:
        raise ValueError(f"取不到 {obj!r} 所在的文件")
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)          # 半截文件 ⇒ SyntaxError,看得见

    qual = getattr(func, "__qualname__", getattr(func, "__name__", ""))
    parts = [p for p in qual.split(".") if p != "<locals>"]
    if not parts:
        raise ValueError(f"{obj!r} 没有可用的 __qualname__")
    node: "ast.AST | None" = tree
    for name in parts:
        node = _find_named(node, name)
        if node is None:
            raise ValueError(f"在 {path} 里找不到 {qual}(走到 {name!r} 就断了)")
    seg = ast.get_source_segment(text, node)
    if seg is None:
        raise ValueError(f"取不到 {qual} 的源码片段")
    return seg


def _find_named(parent, name: str):
    for child in ast.iter_child_nodes(parent):
        if getattr(child, "name", None) == name:
            return child
    return None
