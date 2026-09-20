"""技能必须解包协议信封，不能把原始 bytes 或包装结构当作读数返回。"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

_SRC = _REPO / "MASTv2" / "mast"


def _naked_rv_returns() -> list[str]:
    """找出「函数体只有一句 ``return getattr(x, 'return_value', …)``」的地方。

    用 ast 而不是正则:换行、参数名、有没有默认值都不该影响判定。
    """
    bad: list[str] = []
    for path in _SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            body = [n for n in node.body if not isinstance(n, ast.Expr)]  # 去掉 docstring
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            val = body[0].value
            if not isinstance(val, ast.Call) or not isinstance(val.func, ast.Name):
                continue
            if val.func.id != "getattr" or len(val.args) < 2:
                continue
            arg1 = val.args[1]
            if isinstance(arg1, ast.Constant) and arg1.value == "return_value":
                rel = str(path.relative_to(_REPO)).replace("\\", "/")
                bad.append(f"{rel}:{node.lineno} def {node.name}()")
    return bad


def test_the_gate_can_see_the_tree():
    """先证明这道闸门扫得到东西 —— 扫不到就永远绿。"""
    assert any(_SRC.rglob("*.py")), "源码树扫不到"
    assert (_SRC / "skills" / "builtins" / "readback.py").exists()


def test_no_skill_hands_back_the_raw_envelope():
    """一句话的 ``return getattr(rec, "return_value", …)`` 一个都不许有。"""
    bad = _naked_rv_returns()
    assert not bad, (
        "这些地方把 Nanonis 的整个三段信封当读数交出去了:\n  "
        + "\n  ".join(bad)
        + "\n\n改成 ``from mast.io.nanonis_files import decode_reply`` 再剥。"
        "\n返回字段含有原始信封，而不是已解包的读数；"
        "含 bytes 的还会把 HTTP 层打成 500。")


def test_the_single_source_actually_strips():
    """真源本身要做对 —— 闸门只保证「都用它」,不保证「它对」。"""
    from mast.io.nanonis_files import decode_reply

    assert decode_reply(("", b"\x00", [24])) == 24          # 单值解包
    assert decode_reply(("", b"\x00", [1.0, 2.0])) == [1.0, 2.0]   # 多值保表
    # 扁平的 1-元组串**不是**信封,不许被误剥
    assert decode_reply(((1,), (2,), (3,))) == ((1,), (2,), (3,))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ── 第二种形状:把信封字符串化塞进 data ──────────────────────────────────

def _stringified_envelopes() -> list[str]:
    """找 ``"raw": str(x)`` —— 把整个回包字符串化当数据交出去。"""
    bad: list[str] = []
    # 只扫**技能层** —— ``llm/client.py`` 里也有一句 ``"raw": str(block)``,
    # 那是处理未知 LLM 消息块的兜底,与 Nanonis 回包无关。第一版扫了整个
    # ``mast/`` 就把它误报了进来:**闸门扫太宽,报的就不是它要防的那件事**。
    for path in (_SRC / "skills").rglob("*.py"):
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if '"raw": str(' in line and not line.lstrip().startswith("#"):
                rel = str(path.relative_to(_REPO)).replace("\\", "/")
                bad.append(f"{rel}:{i}")
    return bad


def test_no_skill_stringifies_the_whole_reply():
    """原始回包留在 nanonis_calls，不应整体字符串化后进入业务 data。"""
    bad = _stringified_envelopes()
    assert not bad, (
        "这些地方把整个回包字符串化当数据交出去了:\n  " + "\n  ".join(bad)
        + "\n\n改成 ``decode_reply(parsed)``;要原始字节看 nanonis_calls。")
