"""「等人落在 run 之间」—— 把这条框架原则钉成结构。

## 为什么需要一道结构闸门

这条原则今天写在三处 docstring 里（`conduct/spec.py` 的 WaitSpec、
`conduct/adapters.py` 的通知器、`api/hitl_bridge.py` 的超时常量），
每一处都说得很清楚。但**说清楚不等于拦得住**：hitl_bridge 就在同一个进程里，
`from mast.api.hitl_bridge import ...` 是一行的事，而它带来的三条性质
（900 s fail-closed / 进程本地 / resume 重放整个 tool call）在
conduct 这一层全是错的：

* 900 s ——换样品要几个小时，等降温要一夜；
* 进程本地 ——conduct 存在的理由之一就是熬过重启；
* 重放 ——一次「退针 → 等人 → 进针」重放会**重复退针**。

第三条最狠：它不会报错，它会照做。

所以这里不测「文档写没写」，测的是那条 import 路径**在结构上不存在**。
"""

from __future__ import annotations

import ast
from pathlib import Path

CONDUCT = Path(__file__).resolve().parents[4] / "MASTv2" / "mast" / "conduct"

#: 这些模块带着「等待只能活在一次请求里」的假设，conduct 一律不许碰。
FORBIDDEN_MODULES = (
    "mast.api.hitl_bridge",
    "mast.agents._shared.ask_tools",   # 阻塞式 ask_user：同一个重放问题
)


def _conduct_py():
    for p in CONDUCT.rglob("*.py"):
        if "__pycache__" not in p.parts:
            yield p


def _imported_modules(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
    return out


def test_the_conduct_layer_never_reaches_for_the_in_run_wait():
    offenders: list[str] = []
    for p in _conduct_py():
        mods = _imported_modules(p)
        for bad in FORBIDDEN_MODULES:
            if any(m == bad or m.startswith(bad + ".") for m in mods):
                offenders.append(f"{p.relative_to(CONDUCT.parent).as_posix()} → {bad}")
    assert not offenders, (
        "conduct 层伸手去拿了对话轮内的等待机制：\n  " + "\n  ".join(offenders) + "\n"
        "这条路会把三条性质一起带进来：900 s fail-closed、进程本地的中断表、"
        "resume 重放整个 tool call。第三条不会报错，它会**重复执行上一步的仪器动作**。\n"
        "等人做事 → 异步心愿单 + WaitSpec（状态在 conducts 行里，判定在 tick 之间）。"
    )


def test_the_wait_spec_still_says_why_it_cannot_use_hitl():
    """这条裁决的理由必须留在代码里。

    理由消失之后，「为什么不用现成的 HITL」就会变成一个没人答得上来的问题，
    而答不上来的约束迟早会被当成历史包袱删掉。
    """
    src = (CONDUCT / "spec.py").read_text(encoding="utf-8")
    i = src.find("class WaitSpec")
    assert i > 0, "WaitSpec 不见了？"
    doc = src[i: i + 2500]
    assert "900" in doc, "WaitSpec 的 docstring 里不再提 HITL 那条 900 s 的上限"


def test_waiting_never_gives_up_on_a_human():
    """`max_wait_s` 到点只升级通知，不放弃 —— 等人没有 fail-closed。

    这是与 hitl_bridge 最本质的一处差别：那边超时是一个**终态**（放弃并继续），
    这边超时只是「该更大声地喊了」。把它测出来，是为了让「照抄 HITL 的超时语义」
    这个念头当场变红。
    """
    from mast.conduct.spec import WaitSpec

    doc = (WaitSpec.__doc__ or "")
    assert "不放弃" in doc or "升级" in doc, (
        "WaitSpec 的 docstring 不再说明 max_wait_s 到点之后发生什么。"
    )
