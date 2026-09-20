"""服务启动不得向仪器写入扫描或控制参数。

使用源码结构检查 setup 中的仪器命令，避免为了构造运行时而启动外部依赖。
连接成功与参数初始化是不同操作，启动过程不能改动正在执行的扫描。"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_RUNTIME = (Path(__file__).resolve().parents[4]
            / "MASTv2" / "mast" / "core" / "runtime.py")

#: Nanonis 命令的形状：``Scan_SpeedSet`` / ``ZCtrl_Withdraw`` / ``Util_RTFreqGet``。
_VERB = re.compile(r"^[A-Z][A-Za-z0-9]*_[A-Za-z0-9_]+$")

#: ``runtime.py`` 里允许出现的仪器命令，**全部只在急停或环境告警路径上**：
#:
#: - 只读三条：``Current_Get`` / ``Signals_ValGet`` / ``Util_SessionPathGet``
#: - 急停序列：``AutoApproach_OnOffSet`` / ``Motor_StopMove`` / ``Scan_Action``
#:   / ``ZCtrl_Withdraw``（``emergency_stop()``，人为触发）
#: - ``ZCtrl_Withdraw`` 另在环境告警回调里（读数越限才触发）
#:
#: 往这张表里加名字**不是**改测试就完事：先回答「它凭什么在 runtime 里，而不是在
#: 一个带安全等级、能被 SafetyGuard 和 HITL 门控的 skill 里」。
_ALLOWED_VERBS = {
    "Current_Get",
    "Signals_ValGet",
    "Util_SessionPathGet",
    "AutoApproach_OnOffSet",
    "Motor_StopMove",
    "Scan_Action",
    "ZCtrl_Withdraw",
}


def _verb_literals(node: ast.AST) -> dict[str, list[int]]:
    """节点子树里，所有**作为调用首个位置参数**出现的仪器命令字面量。

    只取首参是刻意的：``safe_call("Scan_SpeedSet", …)`` 与
    ``_urgent("Scan_Action", …)`` 都把命令名放在那里，而日志文案（``"Scan speed
    set: …"``）带空格、注释根本不是字面量，两者都不会被误收。
    """
    found: dict[str, list[int]] = {}
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call) or not sub.args:
            continue
        first = sub.args[0]
        if (isinstance(first, ast.Constant) and isinstance(first.value, str)
                and _VERB.match(first.value)):
            found.setdefault(first.value, []).append(first.lineno)
    return found


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(_RUNTIME.read_text(encoding="utf-8"))


def _setup_node(tree: ast.Module) -> ast.FunctionDef:
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "CoreRuntime":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "setup":
                    return fn
    pytest.fail("找不到 CoreRuntime.setup() —— 启动序列被改名或搬走了，"
                "本测试守的不变式需要跟着搬，不要直接删掉它")


def test_setup_issues_no_instrument_commands(tree):
    """**核心不变式：``CoreRuntime.setup()`` 一条仪器命令都不许发。**

    连上 TCP 不等于可以往仪器上写东西。启动时 MAST 并不知道仪器正在做什么 ——
    可能有人正在扫图、正在测谱、正在退针。启动不能假定当前没有仪器任务。

    注意这里查的是 ``setup()`` **函数体**。启动期**注册**、条件满足才触发的回调
    （环境告警 → ``ZCtrl_Withdraw``）不在函数体里，也不该在：它们是对危险读数的
    反应，不是启动的副作用。
    """
    found = _verb_literals(_setup_node(tree))
    assert not found, (
        "CoreRuntime.setup() 里出现了仪器命令，启动会对仪器产生副作用：\n"
        + "\n".join(f"  {v}  行 {lines}" for v, lines in sorted(found.items()))
        + "\n\n需要默认参数请放到 MAST 自己发起动作的路径上"
          "（core/scan_policy + ConfigureScan），不要放在启动里。"
    )


def test_scan_speed_set_is_gone_from_runtime(tree):
    """回归钉：``Scan_SpeedSet`` 不应再出现在 ``runtime.py`` 的任何地方。

    全仓唯一该发它的地方是 ``skills/builtins/imaging.py`` 的 ``ConfigureScan``
    —— 那里是人/agent 显式要求改扫描参数，且用 ``keep_const=2``（保持每行时间），
    与事故那处传 ``0``（速度和每行时间**都**写）的语义也不一样。
    """
    hits = [v for v in _verb_literals(tree) if v == "Scan_SpeedSet"]
    assert not hits, (
        "runtime.py 里又出现了 Scan_SpeedSet。改扫描速度属于 ConfigureScan 技能，"
        "它有安全等级、走 SafetyGuard，也留得下记录；runtime 里的裸调用三者都没有。"
    )


def test_runtime_instrument_command_census(tree):
    """``runtime.py`` 的仪器命令普查：只允许急停 / 告警路径上的那几条。

    这是一道**会挡路的**闸门，不是统计。runtime 是唯一一个不受技能治理（无安全
    等级、无 SafetyGuard、无 HITL、无记录）却能直接摸到连接池的地方，所以每加一条
    命令都应该被看见一次。
    """
    found = _verb_literals(tree)
    unexpected = {v: ls for v, ls in found.items() if v not in _ALLOWED_VERBS}
    assert not unexpected, (
        "runtime.py 里出现了未登记的仪器命令：\n"
        + "\n".join(f"  {v}  行 {ls}" for v, ls in sorted(unexpected.items()))
        + "\n\n先回答：它凭什么绕过技能层的安全等级 / SafetyGuard / HITL / 记录？"
          "答得上来再把它加进 _ALLOWED_VERBS，并在那里写清理由。"
    )
