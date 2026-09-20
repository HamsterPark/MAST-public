"""目标终止判据在 supervisor 上的两道闸。

## 它治的病

2026-08-27 之前，「这次任务做完了没有」全部写在路由提示词里由模型自己判 ——
图里没有任何一条边在检查它，代码层的终止只有跳数/预算/递归三个熔断。同一个
缺口生出两个方向相反的老病：

* **太早停**：模型说「完了」，而分析还没写、图还没扫（fail_silent 那 621 条）；
* **停不下来**：每次唤醒是新 run，per-run 的熔断全归零，每一步都「成功」。

## 这里钉的顺序（就是它们的重要性）

1. **判据满足 ⇒ 不问模型就结束**；**判据没满足 ⇒ 不许静默结束**。
2. **变异配对**：拆掉判据，拦截必须消失 —— 否则上面那条可能绿在别的原因上。
3. **零回归**：没给 ``done_when`` 的 state（今天所有调用方）逐键等于从前。
4. **熔断永远优先**，**用户的 @agent 永远优先**。
5. **判不了 ≠ 没满足**：前者保持今天行为（模型说了算）+ 留痕，后者才拦。
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.graph import END  # noqa: E402

import mast.agents.orchestrator.graph as og  # noqa: E402
from mast.agents.orchestrator.graph import (  # noqa: E402
    _GOAL_ASKED_MARKER,
    _GOAL_MARKER,
    _supervisor_node_factory,
)

# 判据里用到的两个产物；``analysis`` 由 data_processing 产出（选项文案要用到）。
_DONE_WHEN = [{"kind": "artifact_present", "field": "analysis"}]


# ── 夹具 ────────────────────────────────────────────────────────────────

def _empty_disk() -> dict:
    """空盘 —— **每个字段都在，计数为 0**。

    这个形状不是随手写的：``_artifact_versions`` 成功读取时一定给出每个可等字段
    的条目，所以**空 dict 表示「读不到」而不是「什么都没有」**。第一版夹具用
    ``{}`` 冒充空盘，于是每条判据都变成 unknown，五条测试绿/红在了错的理由上。
    """
    from mast.agents._shared.artifact_channel import WAITABLE_FIELDS

    return {f: (0, 0.0) for f in WAITABLE_FIELDS}


@pytest.fixture(autouse=True)
def _pin_versions(monkeypatch):
    """把「磁盘上有什么」钉住。

    不钉的话这些断言会依赖开发机 artifacts 目录里恰好有什么 —— 那种测试会在
    别人机器上莫名其妙地红，然后被加上 skip。默认：**空盘**。
    """
    monkeypatch.setattr("mast.goals.sources.artifact_versions_from_disk",
                        _empty_disk)


def _disk(monkeypatch, **fields):
    monkeypatch.setattr("mast.goals.sources.artifact_versions_from_disk",
                        lambda: {**_empty_disk(), **fields})


class _Router:
    """只会说一句话的路由替身。``calls`` 是「模型被问了几次」。"""

    def __init__(self, targets):
        self.targets = list(targets)
        self.calls = 0

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, _m):
                outer.calls += 1
                if "action" in getattr(schema, "__annotations__", {}):
                    return {"action": "start", "waiting_for": [], "reason": "ok"}
                return {"next_agents": list(outer.targets), "reason": "该收尾了"}

        return _S()

    def invoke(self, _m):
        return AIMessage(content="ok")


def _state(goal=None, **kw) -> dict:
    s = {
        "messages": [HumanMessage(content="扫一张图并把分析写出来")],
        "visit_count": {}, "executed_skills": [], "scan_paths": [],
        "scan_metadata": {}, "error_log": [], "event_refs": [],
        "pending_approvals": {},
    }
    if goal is not None:
        s["goal"] = goal
    s.update(kw)
    return s


def _goal(done_when=_DONE_WHEN, **kw) -> dict:
    """一份**已经抓过基线**的目标。

    第一版默认写 ``baseline={}``。那不是「空盘的基线」，那是「没有基线」的
    伪装 —— 而当时的 ``_cmp_versions`` 把「没有基线」读成「当时是空的」，
    于是任何一份历史产物都算「新的」。半数用例是靠这个形状变绿的。
    """
    g = {"text": "扫一张图并把分析写出来", "done_when": done_when,
         "baseline": _empty_disk()}
    g.update(kw)
    return g


def _node(router, **kw):
    kw.setdefault("wired_agents", ("data_processing", "instrument_control"))
    return _supervisor_node_factory(router, **kw)


def _texts(cmd) -> str:
    return "\n".join(str(getattr(m, "content", ""))
                     for m in (cmd.update.get("messages") or []))


# ── 1. 两道闸的方向 ────────────────────────────────────────────────────

def test_router_says_end_but_goal_unmet_holds(monkeypatch):
    """模型判断「做完了」，代码知道还没有 ⇒ 转人问，不静默结束。"""
    r = _Router(["__end__"])
    cmd = _node(r)(_state(_goal()))
    assert cmd.goto == "ask_operator", f"竟然结束了：{cmd.goto}"
    q = cmd.update["pending_user_question"]
    assert q["kind"] == "goal_hold"
    assert "analysis" in q["question"] or "分析" in q["question"]
    assert _GOAL_MARKER in _texts(cmd) and _GOAL_ASKED_MARKER in _texts(cmd)


def test_hold_does_not_re_ask_the_model(monkeypatch):
    """**不 nudge。**

    模型是在看过判据块之后才说的 __end__；拿同样的证据再问一次只是多花一跳一次
    调用，而且那正是「用提示词说服模型」——本仓记过四次的反面教材。
    """
    r = _Router(["__end__"])
    _node(r)(_state(_goal()))
    assert r.calls == 1, f"路由被问了 {r.calls} 次 —— 有人在 hold 里又 nudge 了一遍"


def test_goal_done_ends_without_asking_the_model(monkeypatch):
    """判据满足 ⇒ 确定性结束。**一次模型调用都不花。**"""
    _disk(monkeypatch, analysis=(3, 123.0))
    r = _Router(["data_processing"])
    cmd = _node(r)(_state(_goal()))
    assert cmd.goto == END
    assert r.calls == 0, "判据已满足却还去问模型该派谁"
    assert _GOAL_MARKER in _texts(cmd)
    assert cmd.update["active_agent"] == "__end__"


def test_unknown_falls_through_to_the_model(monkeypatch):
    """判不了 ⇒ 保持今天的行为（模型说了算），但把「读不到」说出来。

    「读不到」被当成答案是本仓一天犯四次的错；这里两个方向都不许：既不许当成
    「满足了」（那会提前结束），也不许当成「没满足」（那会拦住一次合法的结束）。
    """
    monkeypatch.setattr("mast.goals.sources.artifact_versions_from_disk",
                        lambda: None)          # 产物清单读不到
    r = _Router(["__end__"])
    cmd = _node(r)(_state(_goal()))
    assert cmd.goto == END, "判不了却把结束拦住了"
    assert "读不到" in _texts(cmd) and _GOAL_MARKER in _texts(cmd)


def test_a_waiting_park_lets_the_run_end(monkeypatch):
    """有东西在等 ⇒ 允许结束，并说明卡在哪。

    被搁置的 agent 既不是完成也不是失败；判据没满足而它在等，这时逼着继续派发
    只会得到同一个结果 —— 那是把一次诚实的等待做成一个死循环。
    """
    r = _Router(["__end__"])
    st = _state(_goal(), pending_activations={
        "data_processing": {"status": "waiting", "waiting_for": ["last_scan"]}})
    cmd = _node(r)(st)
    assert cmd.goto == END
    assert "在等" in _texts(cmd)


# ── 2. 优先级：熔断与用户永远在前 ──────────────────────────────────

def test_the_hard_cap_wins_over_the_hold(monkeypatch):
    """熔断在函数最顶上，先于一切判据求值。目标闸门不许变成第二个死循环源。"""
    r = _Router(["__end__"])
    st = _state(_goal(), visit_count={"data_processing": 999})
    cmd = _node(r)(st)
    assert cmd.goto == END
    assert "Loop guard tripped" in _texts(cmd)


def test_an_operator_at_agent_beats_a_met_goal(monkeypatch):
    """@agent 是显式指令 —— 「@agent 就要到达那个 agent」是既有规则。"""
    _disk(monkeypatch, analysis=(3, 123.0))

    # control_provider 是一个**零参可调用**，回 {"interjections", "directed_targets"}
    # —— 替身照生产契约写，不照我想当然的那个形状写。
    def _ctl():
        return {"interjections": ["再扫一张"],
                "directed_targets": ["instrument_control"]}

    r = _Router(["data_processing"])
    cmd = _node(r, control_provider=_ctl)(_state(_goal()))
    assert cmd.goto != END, "要求的 agent 被目标判据挡掉了"
    assert "instrument_control" in str(cmd.goto)


# ── 3. 每 run 只问一次 ──────────────────────────────────────────────

def test_after_asking_once_the_run_may_end_loudly(monkeypatch):
    """问过一次之后放行 —— 但**大声**放行。

    上限的代价是「有时会在判据没满足时结束」；换来的是这道闸不会和一个坚持要
    结束的模型互相顶到熔断。放行必须留下一句话和一条台账，否则就又变成静默。
    """
    r = _Router(["__end__"])
    st = _state(_goal(asked=True))
    cmd = _node(r)(st)
    assert cmd.goto == END
    assert "判据未满足仍结束" in _texts(cmd)


def test_a_stale_marker_from_an_earlier_task_does_not_count_as_asked(monkeypatch):
    """**不许**靠扫消息流判断「问过了」。

    ``messages`` 用 ``add_messages``、跨任务留在同一个 checkpoint 线程里。
    第一版还扫 ``[SUPERVISOR:GOAL_ASKED]``，于是同一个群聊里的**第二个任务**
    一开始就是「已经问过了」—— Gate 2 对它从来没生效过。现在只认
    ``goal.asked``，而那个字段随新任务的显式清除一起消失。
    """
    r = _Router(["__end__"])
    st = _state(_goal(), messages=[
        HumanMessage(content="扫一张图"),
        AIMessage(content=f"{_GOAL_ASKED_MARKER} 那是上一个任务问的"),
    ])
    cmd = _node(r)(st)
    assert cmd.goto == "ask_operator", (
        "上一个任务留下的标记被当成了「这个任务问过了」")


def test_asked_in_the_goal_channel_does_count(monkeypatch):
    """而 ``goal.asked`` 算数 —— 同一 run 内 resume 回来照样读得到。"""
    assert _node(_Router(["__end__"]))(_state(_goal(asked=True))).goto == END


def test_without_a_checkpointer_it_cannot_ask_and_says_so(monkeypatch):
    """问不出去 ⇒ 不 hold（interrupt 无处可停），但要说明为什么没问。"""
    r = _Router(["__end__"])
    cmd = _node(r, ask_operator_enabled=False)(_state(_goal()))
    assert cmd.goto == END
    assert "没法向用户提问" in _texts(cmd)


# ── 4. 基线 ────────────────────────────────────────────────────────────

def test_a_preexisting_artifact_does_not_satisfy_a_fresh_goal(monkeypatch):
    """续接线程里躺着上一个任务的产物、磁盘上躺着上周的草稿。

    不抓基线的话目标会在第一跳就「达成」—— 那是「太早停」换了个方式复现。
    """
    _disk(monkeypatch, analysis=(3, 123.0))
    r = _Router(["__end__"])
    cmd = _node(r)(_state(_goal(baseline=None)))   # 首访：现场抓基线
    assert cmd.goto == "ask_operator", "上周的分析把这次的目标判成了「已完成」"
    assert cmd.update["goal"]["baseline"]["analysis"] == (3, 123.0), (
        "基线没把「目标设定那一刻已经有 3 份分析」记下来")


def test_allow_preexisting_opts_back_in(monkeypatch):
    """判据明说「有就算」时才不看基线。默认相反，因为默认要防的是太早停。"""
    _disk(monkeypatch, analysis=(3, 123.0))
    g = _goal([{"kind": "artifact_present", "field": "analysis",
                "allow_preexisting": True}], baseline=None)
    cmd = _node(_Router(["__end__"]))(_state(g))
    assert cmd.goto == END and _GOAL_MARKER in _texts(cmd)


def test_a_new_artifact_after_the_baseline_satisfies_it(monkeypatch):
    _disk(monkeypatch, analysis=(4, 999.0))
    g = _goal(baseline={"analysis": (3, 123.0)})
    cmd = _node(_Router(["data_processing"]))(_state(g))
    assert cmd.goto == END


# ── 5. 变异：拆掉判据，拦截必须消失 ──────────────────────────────────

def test_mutation_removing_the_gate_turns_the_hold_back_into_an_end(monkeypatch):
    """把 ``_goal_verdict`` 打成恒 None（= 这道闸不存在）⇒ 同一份 state 直接结束。

    与第一条测试是一对。没有它，那条测试可能绿在别的原因上（比如 state 根本
    没被认出来），而不是「这道闸在起作用」。
    """
    monkeypatch.setattr(og, "_goal_verdict", lambda *a, **k: None)
    r = _Router(["__end__"])
    cmd = _node(r)(_state(_goal()))
    assert cmd.goto == END, "拆掉判据之后仍然被拦住 —— 拦住它的是别的东西"
    assert _GOAL_ASKED_MARKER not in _texts(cmd)


def test_mutation_a_done_verdict_is_what_ends_the_run(monkeypatch):
    """反向变异：把判据打成恒 done ⇒ 本该 hold 的 state 直接结束。"""
    from mast.goals import GoalVerdict

    monkeypatch.setattr(og, "_goal_verdict",
                        lambda *a, **k: GoalVerdict(verdict="done", satisfied=1,
                                                    total=1, reason="装的"))
    r = _Router(["__end__"])
    cmd = _node(r)(_state(_goal()))
    assert cmd.goto == END and r.calls == 0


# ── 6. 零回归 ──────────────────────────────────────────────────────────

class TestAbsentGoalIsUnchanged:
    """没给 ``done_when`` 的 state —— 今天所有的调用方 —— 必须逐键等于从前。

    做法：同一份 state 跑两次，一次把 ``_goal_verdict`` 打成恒 None（模拟这道闸
    不存在），一次不打；比较 ``goto`` 与 ``update`` 的每一个键。
    """

    def _pair(self, monkeypatch, router_targets, st):
        node = _node(_Router(router_targets))
        real = node(dict(st))
        monkeypatch.setattr(og, "_goal_verdict", lambda *a, **k: None)
        node2 = _node(_Router(router_targets))
        without = node2(dict(st))
        return real, without

    @staticmethod
    def _same(a, b):
        assert a.goto == b.goto, f"goto 不同：{a.goto} vs {b.goto}"
        ka, kb = set(a.update or {}), set(b.update or {})
        assert ka == kb, f"update 的键不同：{ka ^ kb}"
        for k in ka:
            if k == "messages":
                ta = [str(getattr(m, "content", "")) for m in a.update[k]]
                tb = [str(getattr(m, "content", "")) for m in b.update[k]]
                assert ta == tb, f"messages 不同：{ta} vs {tb}"
            else:
                assert a.update[k] == b.update[k], f"{k} 不同"

    def test_ending_branch(self, monkeypatch):
        self._same(*self._pair(monkeypatch, ["__end__"], _state()))

    def test_dispatch_branch(self, monkeypatch):
        self._same(*self._pair(monkeypatch, ["data_processing"], _state()))

    def test_hint_dispatch_branch(self, monkeypatch):
        st = _state(routing_hints=["data_processing"])
        self._same(*self._pair(monkeypatch, ["__end__"], st))

    def test_no_goal_key_is_written(self, monkeypatch):
        cmd = _node(_Router(["__end__"]))(_state())
        assert "goal" not in (cmd.update or {}), (
            "没有目标的 run，supervisor 也往通道里写了东西 —— "
            "那会让「没给」和「给了个空的」分不开")


# ── 7. 结构：goal 是控制面，不是产物 ────────────────────────────────

def test_goal_is_control_plane_not_a_product():
    from mast.agents._shared.artifact_channel import CARRIED_FIELDS
    from mast.agents.state import AgentSubState, CampaignRef, MASTState

    assert "goal" in MASTState.__annotations__
    assert "goal" not in AgentSubState.__annotations__, (
        "goal 进了子图 schema —— agent 就能改写「自己何时该停」")
    assert "goal" not in CARRIED_FIELDS, "goal 进了产物运输表"
    assert "done_when" not in CampaignRef.model_fields, (
        "终止判据被塞进了 CampaignRef —— 那是跨 run 的 DB 指针，由 RD 的工具写")


def test_the_router_prompt_quotes_the_real_hop_cap():
    """提示词里的跳数上限从常量派生。

    「改了常量而行为不跟着变」本仓抓过；这是它的孪生 —— 改了常量而**说给模型
    听的那句话**不跟着变（提示词写 40，代码是 60，岔了将近一个月）。
    """
    from mast.agents.orchestrator.graph import _HOP_HARD_CAP, _ROUTER_PROMPT

    assert f"超过 {_HOP_HARD_CAP} 即 END" in _ROUTER_PROMPT
    assert "超过 40 即 END" not in _ROUTER_PROMPT or _HOP_HARD_CAP == 40


# ── 8. 台账真的收到了（生产方接了，消费方也在） ──────────────────

def test_every_decisive_hop_reaches_the_diagnostics_ledger(monkeypatch):
    """用户问「它为什么停了 / 为什么没停」时看的就是这本账。

    只有 transcript 一条不够：那是对话流，会被压缩、会被翻页。台账是
    `kinds=("goal_verdict",)` 一句话查得到的东西 —— 与 fail_silent 那 621 条
    同一本，而那正是这道闸要治的病。
    """
    from mast.core import diagnostics

    seen: list = []
    monkeypatch.setattr(diagnostics, "record",
                        lambda kind, subject, reason, **f: seen.append((kind, f)))

    _node(_Router(["__end__"]))(_state(_goal()))          # hold
    _disk(monkeypatch, analysis=(3, 123.0))
    _node(_Router(["data_processing"]))(_state(_goal()))  # done

    kinds = [k for k, _ in seen]
    assert kinds.count("goal_verdict") >= 2, f"台账只收到 {kinds}"
    decisions = {f.get("decision") for _, f in seen}
    assert {"hold", "done"} <= decisions, decisions
    # 字段是摊平的 —— 嵌套一层会让「查所有 decision=hold 的行」要写脚本。
    assert all("satisfied" in f and "total" in f for _, f in seen)


# ── 9. 自审抓出来的四个缺陷，各钉一条 ────────────────────────────

def test_a_stale_goal_from_a_previous_task_cannot_end_this_one(monkeypatch):
    """**最严重的那一条。**

    ``run_task(conversation_id=C)`` 复用同一个 ``thread_id``，而 ``goal`` 是裸
    LastValue。任务 #1 带 done_when 跑完并产出 analysis；任务 #2 **不带**
    done_when —— 如果 #1 的 goal 还在通道里，#2 第一跳 Gate 1 就命中「目标已
    全部满足」直接 END：一个 agent 都不派，用户的第二个问题永远没被回答。

    修在 ``initial_state``（写 ``goal: None`` 显式清除，与 ``visit_count: None``
    同一个理由）。这里钉的是**清除之后**的行为：没有 goal 就没有 Gate 1。
    """
    _disk(monkeypatch, analysis=(9, 999.0))          # 上一个任务产出的
    r = _Router(["data_processing"])
    cmd = _node(r)(_state())                          # 没有 goal
    assert cmd.goto != END, "没有目标的 run 竟然被目标闸门结束了"


def test_an_artifact_already_in_state_at_baseline_time_is_not_new(monkeypatch):
    """state 快路不许绕开基线。

    产物通道是 ``last_wins`` LastValue，跨任务留在同一线程里：续接会话时上一个
    任务产出的 ``analysis`` 就在 state 里躺着。第一版的快路直接 ``return True``，
    一眼都不看基线 —— ``allow_preexisting=False`` 在这条路上完全失效。
    """
    from mast.agents._shared.artifact_types import AnalysisResult

    base = {**_empty_disk(), "_state_present": ("analysis",)}
    st = _state(_goal(baseline=base),
                analysis=AnalysisResult(summary="上一个任务留下的"))
    cmd = _node(_Router(["__end__"]))(st)
    assert cmd.goto == "ask_operator", (
        "基线时就在 state 里的产物被当成了「这一 run 刚产出的」")


def test_a_missing_baseline_is_snapshotted_on_the_first_visit(monkeypatch):
    """run 路径上「没有基线」活不过第一跳 —— 它当场被补上。

    补上的那一份包含**当时磁盘上已有的**产物，所以那些历史产物不算「新的」：
    目标落在 not_done，而不是被它们假满足。（campaign 路径没有这一步，那里
    「没有基线」会一路走到 ``_cmp_versions`` —— 见
    ``tests/v2/unit/goals/`` 里那条。）
    """
    _disk(monkeypatch, analysis=(3, 123.0))
    g = _goal()
    g.pop("baseline")
    cmd = _node(_Router(["__end__"]))(_state({**g, "baseline": None}))
    assert cmd.goto == "ask_operator", "上周的 analysis 把这次的目标判成了达成"
    assert (cmd.update["goal"]["baseline"]["analysis"] == (3, 123.0))


def test_a_failed_baseline_snapshot_is_retried_not_frozen(monkeypatch):
    """抓不到基线就**留 None，下一跳再抓** —— 不写一个 `{}` 假装世界是空的。"""
    monkeypatch.setattr("mast.goals.sources.artifact_versions_from_disk",
                        lambda: None)
    g = _goal()
    g.pop("baseline")
    cmd = _node(_Router(["__end__"]))(_state({**g, "baseline": None}))
    written = (cmd.update or {}).get("goal") or {}
    assert written.get("baseline") is None, (
        f"抓不到基线却写了 {written.get('baseline')!r} —— "
        "下一跳不会再抓，而这个值让任何历史产物都算「新的」")


def test_the_blocked_branch_does_not_say_the_criteria_are_unreadable(monkeypatch):
    """有 park 卡着 ≠ 判据读不到。

    用 unknown 那句话，用户和下一跳的路由模型都会读成「目标系统坏了」，
    而不是「有东西在等」。
    """
    st = _state(_goal(), pending_activations={
        "data_processing": {"status": "waiting", "waiting_for": ["last_scan"]}})
    txt = _texts(_node(_Router(["__end__"]))(st))
    assert "在等" in txt
    assert "读不到" not in txt, txt
