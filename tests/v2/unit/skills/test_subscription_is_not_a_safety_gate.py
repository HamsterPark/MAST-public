"""订阅**不是**安全机制 —— 把这条钉死。

这是一条 [[pin_rejected_designs_as_tests]] 式的测试：它钉住的不是一个功能，而是一个
**被否掉的设计**。把安全逻辑往订阅门上挂是这个功能最自然、也最错的下一步演化，而
它一旦发生不会有任何报错 —— 只会有一天某条执行路径开始拒绝一个用户明明能手动
跑的技能，而拒绝理由是「你没订阅」。

被钉住的四条：

1. **执行路径不认识订阅**。手动执行 / composite 子步 / conduct 的 ``ctx.run`` 三条
   路都经 ``ExecutionContext.run`` 收口，那里没有名单式判断。这里用 AST 扫 import
   钉住「它们连订阅模块都不 import」—— 结构断言，比行为断言更早发现意图漂移
   （同 ``tests/v2/unit/conduct/test_autonomy.py`` 钉「autonomy 看不见 validator」）。
2. **未订阅的技能照常能执行**（上面那条的行为半边：形状对了不代表说得出话）。
3. **agent 写不进订阅面**，只能写 pending 推荐。
4. **conduct 的工具面全开不受订阅影响**（2026-08-20 拍板）。

每条结构断言都配一个**正例自检**：扫描器必须能在真的 import 了订阅的文件里看见它。
否则「四个文件都没 import」可能只是因为扫描器坏了。
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from mast.core.execution_context import ExecutionContext
from mast.core.registry import SkillRegistry
from mast.core.types import (
    NanonisCallRecord,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills import subscription as sub
from mast.skills.base import BaseSkill

_MASTV2 = Path(__file__).resolve().parents[4] / "MASTv2"

#: 每一次技能调用真正的收口点，加上三个把技能送进去的入口。
#: 它们**都不该**认识订阅模块。
_EXECUTION_PATH_FILES = (
    "mast/core/execution_context.py",
    "mast/core/executor.py",
    "mast/conduct/adapters.py",
    "mast/api/routes/skill_exec.py",
)

#: 正例自检：这个文件**必须** import 得到订阅（它是装配层，那里正是订阅该在的地方）。
_MUST_SEE_IT = "mast/skills/tool_face.py"

_SUBSCRIPTION_MODULE = "mast.skills.subscription"


def _imported_modules(rel: str) -> set[str]:
    """一个文件 import 了哪些模块（含函数体内的惰性 import）。"""
    src = (_MASTV2 / rel).read_text(encoding="utf-8")
    out: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. 结构：执行路径不认识订阅
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", _EXECUTION_PATH_FILES)
def test_execution_paths_do_not_import_subscription(rel):
    mods = _imported_modules(rel)
    offenders = sorted(m for m in mods if m.startswith(_SUBSCRIPTION_MODULE))
    assert not offenders, (
        f"{rel} import 了订阅模块（{offenders}）。\n"
        "订阅是**装载面**偏好，不是权限：手动执行、composite 子步、conduct 三条路\n"
        "都必须能调到未订阅的技能。要在这里加名单式拒绝，先回答：用户在界面上\n"
        "点「执行」的那个技能，凭什么因为他没订阅就不许跑？")


def test_the_import_scanner_can_actually_see_imports():
    """闸门自检：扫描器必须在真的 import 了订阅的文件里看见它。

    没有这一条，上面四条可能只是因为 ``_imported_modules`` 扫错了文件/坏掉了 ——
    「闸门全在检查形状，没一道问它到底说不说得出话」的同一族。
    """
    mods = _imported_modules(_MUST_SEE_IT)
    assert any(m.startswith(_SUBSCRIPTION_MODULE) for m in mods), (
        f"扫描器在 {_MUST_SEE_IT} 里没看见订阅 import —— 它坏了，"
        "于是上面四条「都没 import」是假绿。")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 行为：未订阅的技能照常能执行
# ─────────────────────────────────────────────────────────────────────────────

class _ProbeSkill(BaseSkill):
    """一个纯读、零参数的探针技能。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SubscriptionProbe",
            description="测试用探针（只读，不碰硬件）",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            parameters=[],
        )

    def execute(self, context, params) -> SkillResult:
        return SkillResult(skill_name="SubscriptionProbe", success=True,
                           data={"ran": True})


class _Pool:
    def safe_call(self, method, *args, role="main"):
        return NanonisCallRecord(method=method, args=args,
                                 return_value=("", b"", [0]))


@pytest.fixture
def probe_ctx(subscription_store):
    reg = SkillRegistry()
    reg.register(_ProbeSkill)
    ctx = ExecutionContext(pool=_Pool(), state=None, registry=reg,
                           abort_event=threading.Event())
    return ctx, reg


def test_execution_context_runs_an_unsubscribed_skill(probe_ctx):
    """composite 子步与 conduct 都从这扇门进 —— 它必须放行未订阅的技能。"""
    ctx, _reg = probe_ctx
    sub.set_subscribed(set())          # 退订一切
    assert sub.is_subscribed("SubscriptionProbe") is False, "前提没立住"

    res = ctx.run("SubscriptionProbe", {})
    assert res.success, f"未订阅的技能被执行路径拒了：{res.error}"
    assert res.data.get("ran") is True


def test_a_skill_that_does_not_exist_is_still_refused(probe_ctx):
    """负例先自证会红：这条路**会**拒绝东西，只是不按订阅拒。"""
    ctx, _reg = probe_ctx
    res = ctx.run("NoSuchSkillAtAll", {})
    assert not res.success, "执行路径对不存在的技能也放行 —— 那上一条测试证明不了什么"


def test_unsubscribing_does_not_change_what_the_registry_holds(probe_ctx):
    """未订阅 ≠ 不存在。手动/GUI 执行器路径靠的就是这一点。"""
    _ctx, reg = probe_ctx
    sub.set_subscribed(set())
    assert reg.get("SubscriptionProbe") is not None
    assert any(m.name == "SubscriptionProbe" for m in reg.list_skills())


# ─────────────────────────────────────────────────────────────────────────────
# 3. agent 写不进订阅面
# ─────────────────────────────────────────────────────────────────────────────

def test_recommending_alone_never_changes_the_face(subscription_store):
    sub.set_subscribed({"A"})
    before = sub.subscribed_names()
    sub.add_recommendation("B", by_agent="instrument_control", reason="需要它")
    assert sub.subscribed_names() == before, "一条推荐就把订阅面改了 —— 那用户确认是摆设"
    assert [r["skill"] for r in sub.pending_recommendations()] == ["B"]


def test_duplicate_recommendation_is_idempotent(subscription_store):
    sub.set_subscribed({"A"})
    first = sub.add_recommendation("B", reason="一次")
    second = sub.add_recommendation("B", reason="又一次")
    assert second["duplicate"] is True
    assert second["recommendation"]["id"] == first["recommendation"]["id"]
    assert len(sub.pending_recommendations()) == 1


def test_accepting_is_what_changes_the_face(subscription_store):
    sub.set_subscribed({"A"})
    rec = sub.add_recommendation("B")["recommendation"]
    res = sub.resolve_recommendation(rec["id"], accept=True)
    assert res["ok"] is True
    assert sub.is_subscribed("B") is True
    assert sub.pending_recommendations() == []


def test_rejecting_leaves_a_trace_and_changes_nothing(subscription_store):
    sub.set_subscribed({"A"})
    rec = sub.add_recommendation("B")["recommendation"]
    sub.resolve_recommendation(rec["id"], accept=False)
    assert sub.is_subscribed("B") is False
    resolved = sub.resolved_recommendations()
    assert [r["status"] for r in resolved] == [sub.REC_REJECTED], (
        "拒绝把记录删掉了 —— 留痕的意思是能看见「他拒过」，不是「什么都没发生过」")
    assert resolved[0]["resolved_at"]


def test_accepting_while_uncustomised_does_not_materialise(subscription_store):
    """默认全订阅时接受一条推荐，不该把用户悄悄转成明确名单。

    否则 audit 里看起来像是他主动定制过，而他只是点了「接受」。
    """
    assert sub.is_customised() is False
    rec = sub.add_recommendation("B")["recommendation"]
    res = sub.resolve_recommendation(rec["id"], accept=True)
    assert res["already_subscribed_by_default"] is True
    assert sub.is_customised() is False
    assert sub.is_subscribed("B") is True
