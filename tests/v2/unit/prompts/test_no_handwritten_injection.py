"""结构闸门：``_shared/`` 里不许再有第二份「手写拼 system 消息」。

## 为什么要拦

2026-08-24 之前，每个注入中间件各自抄了一份「取出 system 文本 → 拼上自己的块 →
造一个新 SystemMessage → 尽量走 override」。九份副本，彼此有细微差别：

* 有的漏了 ``override``、只做直接赋值，于是每次调用触发一条 DeprecationWarning
  （而且是就地改一个共享对象）；
* 有的把 list 型 content 直接 ``str()`` 掉，多模态块就没了；
* **没有任何一份记下「这段文本是谁塞进来的」** —— 于是抓包里看到的是一整块
  22 k 字符的 system，要回答「其中哪 483 个是针尖块」只能靠人肉正则。

第十份副本会以同样的方式静静地长出来。所以这道闸门不是风格检查，它守的是
「归属账只有一处会记」这个不变式。

Run from repo root::

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/prompts/test_no_handwritten_injection.py -q
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_SHARED = _REPO / "MASTv2" / "mast" / "agents" / "_shared"

#: 允许手写的地方。
#:
#: ``inject.py`` 是那唯一一处实现；``live_state_mw`` 与 ``alert_delivery_mw``
#: 保留 ``SystemMessage`` 的 import 只是为了类型/回退路径的可读性，**它们的
#: ``_apply`` 已经全走 helper** —— 这份豁免只对 import 行成立，构造调用照判。
_ALLOWED_FILES = {"inject.py"}


def _offenders(source: str) -> list[tuple[int, str]]:
    """返回 ``[(行号, 理由)]``。判据走 AST，不逐行 grep。

    逐行 grep 会被两件事打败：注释与 docstring 里**必须**能写清「这里原来有过
    什么」（逐行扫会逼人删掉解释），以及一个构造调用会被换行切开。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return []
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        # SystemMessage(...) 构造
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name == "SystemMessage":
                out.append((node.lineno, "手写构造 SystemMessage"))
        # request.system_message = ...
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == "system_message":
                    out.append((node.lineno, "直接赋值 .system_message"))
        # request.override(system_message=...)
        if isinstance(node, ast.Call):
            fn = node.func
            if getattr(fn, "attr", "") == "override":
                for kw in node.keywords:
                    if kw.arg == "system_message":
                        out.append((node.lineno, "override(system_message=…)"))
    return out


# ── 自检：闸门确实扫到了东西 ────────────────────────────────────────────

def test_the_gate_reads_the_shared_middlewares():
    files = sorted(_SHARED.glob("*.py"))
    assert len(files) >= 15, f"只找到 {len(files)} 个 _shared 模块"
    total = sum(len(p.read_text(encoding="utf-8")) for p in files)
    assert total > 100_000, f"扫到的源码只有 {total} 字符，路径多半错了"


def test_the_gate_catches_a_planted_copy():
    """变异自检：塞一份手写副本进去，闸门必须逮到。"""
    for bad in (
        'new = SystemMessage(content=text + "\\n\\n" + block)',
        "request.system_message = new_sm",
        "return request.override(system_message=new_sm)",
    ):
        assert _offenders(bad), f"闸门漏掉了：{bad!r}"


def test_the_gate_ignores_comments_and_docstrings():
    """历史说明必须能写下来 —— 逼人删掉解释的闸门是负收益。"""
    src = (
        '"""从前这里是 SystemMessage(content=text + block)，2026-08-24 换成 helper。"""\n'
        "# 旧写法：request.system_message = new_sm\n"
        "GOOD = 1\n"
    )
    assert _offenders(src) == []


# ── 主判据 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "path", sorted(p for p in _SHARED.glob("*.py") if p.name not in _ALLOWED_FILES),
    ids=lambda p: p.name)
def test_no_shared_middleware_builds_a_system_message_by_hand(path):
    hits = _offenders(path.read_text(encoding="utf-8"))
    assert not hits, (
        f"{path.name} 里还有 {len(hits)} 处手写注入：\n  "
        + "\n  ".join(f"L{ln}: {why}" for ln, why in hits)
        + "\n改成调用 mast.agents._shared.inject 的 append_system_block / "
        "append_human_block / append_new_human —— 归属账只有那一处会记。")


def test_the_one_allowed_file_is_the_helper_itself():
    """豁免名单必须**窄**，而且里面那一个得真的是实现。"""
    assert _ALLOWED_FILES == {"inject.py"}
    assert _offenders((_SHARED / "inject.py").read_text(encoding="utf-8")), (
        "inject.py 里没有任何 SystemMessage 构造？那实现搬走了，这份豁免该跟着走。")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
