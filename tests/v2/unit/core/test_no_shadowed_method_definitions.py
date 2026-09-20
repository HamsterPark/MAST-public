"""一个类里同名方法定义两次 —— Python **一声不吭**地取后定义的那份。

## 这条为什么存在(2026-08-06)

给 `AutoApproach` 加「进针前切参数组、完事放回去」时,新加的 `run_composite`
被插在类体前面,而这个类**早就有一个** `run_composite`(它绕开 `_base._graph_execute`,
为了清 sidecar,定义在几百行之后)。Python 保留后定义的那份:

* 没有报错、没有警告;
* `import` 通过、冒烟通过;
* 同一改动的另一半(`ApproachTip`)是绿的,所以「改好了」看上去成立;
* 插进去的那段代码**一次都没有跑过**。

抓住它的是一条断言「调用方看得见的行为」的钉子。这里再加一道结构闸:任何类里出现
第二个同名方法定义就红,理由写在报错里 —— 因为下一个人不会知道那个类在几百行外
还有一个同名方法,而症状是「我的代码没生效」,不是任何一种报错。

`@property` 的 setter/deleter、`@overload`、`@typing.overload` 是**合法**的同名
重定义,放行。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_ROOT / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

_PKG = _ROOT / "MASTv2" / "mast"

#: 合法的同名重定义装饰器(子串匹配)。
_ALLOWED = ("setter", "deleter", "overload")


def _shadowed(path: Path) -> "list[str]":
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover — 语法错另有闸门
        return []
    out: "list[str]" = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        first: "dict[str, int]" = {}
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name in first:
                decs = [ast.unparse(d) for d in item.decorator_list]
                if any(tok in d for d in decs for tok in _ALLOWED):
                    continue
                try:
                    where = path.relative_to(_ROOT)
                except ValueError:      # 探针自检用的临时文件不在仓里
                    where = path
                out.append(
                    f"{where}:{item.lineno} "
                    f"{node.name}.{item.name}() 覆盖了同类第 {first[item.name]} 行的"
                    f"同名定义 —— 前一份**永远不会执行**,而 Python 不报错")
            first.setdefault(item.name, item.lineno)
    return out


def test_no_class_defines_the_same_method_twice():
    hits: "list[str]" = []
    for path in sorted(_PKG.rglob("*.py")):
        hits.extend(_shadowed(path))
    assert not hits, "类里有被静默覆盖的方法定义:\n" + "\n".join(hits)


def test_the_detector_can_actually_fail(tmp_path):
    """探针有效性:这条闸门真的抓得住。

    没有这一条,一个永远返回空列表的探测器和一个干净的仓库长得一模一样。
    """
    bad = tmp_path / "bad.py"
    bad.write_text(
        "class C:\n"
        "    def f(self):\n"
        "        return 1\n"
        "    def f(self):\n"
        "        return 2\n",
        encoding="utf-8")
    assert _shadowed(bad), "探测器抓不住一个明摆着的重复定义"

    ok = tmp_path / "ok.py"
    ok.write_text(
        "class C:\n"
        "    @property\n"
        "    def x(self):\n"
        "        return self._x\n"
        "    @x.setter\n"
        "    def x(self, v):\n"
        "        self._x = v\n",
        encoding="utf-8")
    assert not _shadowed(ok), "property 的 setter 被误报了"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
