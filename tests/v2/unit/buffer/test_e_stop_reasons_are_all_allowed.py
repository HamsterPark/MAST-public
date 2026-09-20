"""急停原因调用点与事件工厂允许集合必须一致。

事件提交层可能吞掉构造异常，因此遗漏原因会让硬件急停与界面通知脱节。
测试逐项检查调用点和工厂白名单，确保每类急停都能产生可见事件。"""
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

from mast.buffer.schemas import E_STOP_REASONS, make_e_stop  # noqa: E402

_SRC = _REPO / "MASTv2" / "mast"


def _call_sites() -> list[tuple[str, int, ast.AST | None]]:
    """全树里每一处 ``make_e_stop(...)`` → (相对路径, 行号, 第一个位置参数)。"""
    out: list[tuple[str, int, ast.AST | None]] = []
    for path in _SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else "")
            if name != "make_e_stop":
                continue
            rel = str(path.relative_to(_REPO)).replace("\\", "/")
            out.append((rel, node.lineno, node.args[0] if node.args else None))
    return out


def test_there_are_call_sites_to_check():
    """先证明这道闸门看得见东西 —— 扫不到调用点的话它永远绿。"""
    sites = _call_sites()
    assert len(sites) >= 3, (
        f"只扫到 {len(sites)} 处 make_e_stop 调用。这道闸门靠 ast 扫源码,"
        "扫不到就等于没在守 —— 先确认导入名/调用形式有没有变。")


def test_every_literal_reason_is_in_the_whitelist():
    """字面量 reason 必须被工厂接受。"""
    bad: list[str] = []
    for rel, lineno, arg in _call_sites():
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value not in E_STOP_REASONS:
                bad.append(f"{rel}:{lineno} reason={arg.value!r}")
    assert not bad, (
        "这些急停发不出去 —— make_e_stop 会抛 ValueError,而调用点全都包在宽 "
        "except 里,所以它们会**静默消失**:\n  " + "\n  ".join(bad)
        + f"\n白名单现在是 {E_STOP_REASONS}。"
        "\n环境告警的急停原因必须属于事件允许的原因集合，"
        "而用户什么提示都没收到。加发起方 = 双边动作,调用点和白名单一起改。")


def test_non_literal_reasons_are_reported_not_assumed_ok():
    """非字面量的 reason 静态判不了 —— 要显式列出来,不许当成通过。

    「判不了」既不是合格也不是不合格。真出现了,这条会红,提醒人去给那个
    调用点补一条运行时测试 —— 而不是让一个静态扫不动的地方悄悄溜过闸门。
    """
    opaque = [f"{rel}:{lineno}" for rel, lineno, arg in _call_sites()
              if arg is not None
              and not (isinstance(arg, ast.Constant) and isinstance(arg.value, str))]
    assert not opaque, (
        "这些调用点的 reason 不是字面量,这道闸门管不到它们:\n  "
        + "\n  ".join(opaque)
        + "\n给它们各补一条运行时断言,或把 reason 收回成字面量。")


@pytest.mark.parametrize("reason", list(E_STOP_REASONS))
def test_the_whitelist_entries_actually_construct(reason):
    """白名单里的每个名字都真的能造出事件 —— 表和工厂不许对不上。"""
    ev = make_e_stop(reason, "gate", seqno=1)
    assert (ev.payload or {}).get("reason") == reason


def test_environment_is_accepted():
    """环境告警急停必须产生 environment 原因事件，供界面与代理获知。"""
    ev = make_e_stop("environment", "tunnel_current error: 0.0 A", seqno=7)
    assert (ev.payload or {}).get("reason") == "environment"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
