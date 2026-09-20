"""一台仪器有几个入口 —— 用清单钉住，不用一个会漂的数字。

## 为什么是结构测试

2026-07-28 的调度审计发现「三个入口共用一个 ConnectionPool 而全树没有仲裁」，
`instrument_lock` 就是那次的产物。可**入口数量本身从此没人维护得住**：

* `instrument_lock` 的 docstring 写 "Three independent entry points"，
* `core/executor.py` 写自己是 "the THIRD driver"，
* `api/routes/skill_exec.py` 写自己是「第四个入口」，
* conduct 的设计文档写它是「第 5 个」，

而真实的 owner 字符串到 2026-08 已经至少八个。四份文档四种数法，没有一份是当时
的错——它们只是各自停在了自己被写下的那一天。**人肉维护的计数必漂**，本仓在
「每页各自记得」上已经付过一次这个学费，那次的结论是：先写结构闸门。

所以这里钉的不是数量，是两份**可枚举**的东西：

1. **取令牌的机制**（`hold_for_skill` 的调用点）——只有三个模块，新增一个就是
   在开第四条取令牌的路，那需要被看见；
2. **`ExecutionContext` 的构造点**——它们全部经 `execution_context` 那一条路取
   令牌，所以真正要盯的是**有没有显式传 owner**：不传的话拒绝消息里会说
   「未知入口」，而那是一句对用户毫无用处的话。

两份清单都允许增长，但**必须在这里被承认**。改这个文件是有意的动作，加一个
入口而不改这里则会当场变红。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

MAST = Path(__file__).resolve().parents[4] / "MASTv2" / "mast"


def _py_files():
    for p in MAST.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def _rel(p: Path) -> str:
    return p.relative_to(MAST).as_posix()


# ── 1. 取令牌的机制 ──────────────────────────────────────────────────
#
# 每一条都在自己的模块里说明了它服务的是哪一类调用方（见各自的注释）。
EXPECTED_TOKEN_TAKERS = {
    "agents/_shared/skill_adapter.py",   # agent 的工具边界（群聊 IC / 私聊 IC）
    "core/execution_context.py",         # 一切 ExecutionContext 持有者
    "core/executor.py",                  # 手动 / GUI 按钮
}


def test_the_ways_to_take_the_instrument_token_are_the_ones_we_know_about():
    found = set()
    for p in _py_files():
        src = p.read_text(encoding="utf-8", errors="ignore")
        for line in src.splitlines():
            if "hold_for_skill(" in line and "def hold_for_skill" not in line:
                # import 行不算调用点
                if line.lstrip().startswith(("from ", "import ")):
                    continue
                found.add(_rel(p))
    found.discard("core/instrument_lock.py")  # 定义本身

    new = found - EXPECTED_TOKEN_TAKERS
    assert not new, (
        f"发现新的取令牌路径：{sorted(new)}。\n"
        "这不是坏事，但它必须被承认：把它加进 EXPECTED_TOKEN_TAKERS，"
        "并在 instrument_lock 的 docstring 清单里写清楚它服务的是哪一类调用方。\n"
        "（2026-07-28 那次事故的形状正是「多了一条没人知道的路」。）"
    )
    gone = EXPECTED_TOKEN_TAKERS - found
    assert not gone, (
        f"这些路径不再取令牌了：{sorted(gone)}。\n"
        "如果是有意移除，请同时更新 instrument_lock 的 docstring —— "
        "一份说着不存在的东西的清单，比没有清单更容易骗到人。"
    )


# ── 2. ExecutionContext 的构造点必须显式说明自己是谁 ──────────────────
#
# 不传 owner 会落到 execution_context 里的 "未知入口" 默认值上。那不是 bug，
# 是一个**诚实的兜底**——但它出现在拒绝消息里的时候，用户读到的是
# 「未知入口 正在执行 XXX」，而他需要知道的恰恰是那个入口是谁。
EXPECTED_CONTEXT_BUILDERS_WITH_OWNER = {
    # 直调内核：技能直调 API（owner「技能直调 API」）与外部 agent 网关
    # （owner「外部 agent ext:<名>」）共用这一个构造点 —— 2026-09-18 起
    # api/routes/skill_exec.py 不再自己建 ExecutionContext。
    "api/direct_exec.py",
    "api/routes/instrument_init.py",
    "api/routes/scope.py",
    "api/routes/signals.py",
    "conduct/adapters.py",
    "core/runtime.py",
}

#: 已知**不传** owner 的构造点。留着是因为它们确实说不出自己是谁：
#: `pipeline/main.py` 是 Phase 7 的最小 CLI（一次性进程、跑完就退，
#: 且它压根没有第二个入口去和它抢）。新增条目要写清楚理由。
KNOWN_ANONYMOUS_CONTEXT_BUILDERS = {
    "pipeline/main.py",
}

def test_every_execution_context_says_who_it_is():
    # 用 AST 而不是正则：docstring 里写着 "returns ExecutionContext(pool, state,
    # registry)" 的示意文字不是调用点，把它算进来就是一条永远在响的假警报，
    # 而假警报的下场是被人加进白名单了事。
    with_owner: set[str] = set()
    without_owner: set[str] = set()
    for p in _py_files():
        rel = _rel(p)
        if rel in {"core/execution_context.py", "core/executor.py"}:
            continue  # 定义与 legacy 兼容层
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name != "ExecutionContext":
                continue
            kwargs = {kw.arg for kw in node.keywords if kw.arg}
            (with_owner if "owner" in kwargs else without_owner).add(rel)

    unexpected_anon = without_owner - KNOWN_ANONYMOUS_CONTEXT_BUILDERS
    assert not unexpected_anon, (
        f"这些地方建了 ExecutionContext 却没说自己是谁：{sorted(unexpected_anon)}。\n"
        "传 owner=\"...\"。仪器被占用时的拒绝消息会把它念给用户听，"
        "而「未知入口」这四个字回答不了「现在开车的是谁」。"
    )

    new_named = with_owner - EXPECTED_CONTEXT_BUILDERS_WITH_OWNER
    assert not new_named, (
        f"新的 ExecutionContext 构造点：{sorted(new_named)}。\n"
        "加进 EXPECTED_CONTEXT_BUILDERS_WITH_OWNER 即可 —— 这条断言只是要求"
        "「多一个驱动仪器的入口」这件事被人看一眼，不是要拦住它。"
    )


def test_conduct_owner_prefix_is_the_single_source():
    """conduct 的 owner 前缀只有一份定义，别处不许再拼一遍。"""
    from mast.conduct.adapters import OWNER_PREFIX

    assert OWNER_PREFIX == "conduct:"

    # 全仓不许出现第二个硬写的 "conduct:" 前缀拼接（adapters 自己除外）。
    offenders = []
    for p in _py_files():
        rel = _rel(p)
        if rel == "conduct/adapters.py":
            continue
        src = p.read_text(encoding="utf-8", errors="ignore")
        if 'f"conduct:{' in src or "'conduct:' +" in src or '"conduct:" +' in src:
            offenders.append(rel)
    assert not offenders, (
        f"这些地方自己拼了 conduct owner 前缀：{offenders}。"
        "改成 from mast.conduct.adapters import OWNER_PREFIX —— "
        "两份实现里总有一份会先改。"
    )
