"""composite 每一步的旁白 —— 接线、对账，以及「它弄不坏正在跑的实验」。

设计文档：``docs/v2/design/chat_narration_sidechannel.md`` 阶段 2 + §Q8

三组断言：

**① 对账，不是「有就行」。**
   旁白里的 ``facts`` 必须**逐值等于** ``step.params`` 里那一份。这是整套设计要
   保证的那件事（「10 V / 500 ms 必须是真的下发值」），而它成立的原因不是模板
   写得好，是 ``GraphExecutor`` 把送进 ``ctx.run`` 的**同一个 dict** 原样交出去。
   所以这里同时断言 ``ctx.run`` 收到的参数和旁白记下的参数是同一份。

**② 弄不坏正在跑的实验（§Q8 要求「证明」而不是「跑一遍没崩」）。**
   让存储的 ``append_message`` **每次都抛**，跑一个三步计划，断言三步全部执行、
   全部成功、``run_plan`` 返回 True。旁白链路整条烂掉，实验照跑。

**③ 不认识的技能不发。** 一次 ForgeAuTip 有几百个子步骤；给每一步都配一句
   「正在执行某个步骤」等于把真正有信息量的那几条淹掉。

替身纪律：``FakeCtx.run`` 的签名对着**真的** ``ExecutionContext.run`` 核，返回的是
**真的** ``SkillResult``；``params`` 用的是真技能元数据里真有的 key（那一条由
``tests/v2/unit/chat/test_narration_templates_gate.py`` 拿 SkillMetadata 核）。
存储不是替身，是真的 ``ConversationStore``，建在 ``tmp_path`` 上。

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_graph_executor_narration.py -q
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import inspect  # noqa: E402
import json  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import pytest  # noqa: E402

from mast.chat import narration  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402
from mast.core.execution_context import ExecutionContext  # noqa: E402
from mast.core.turn_context import turn_scope  # noqa: E402
from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite.graph_executor import (  # noqa: E402
    CompositeStep,
    GraphExecutor,
)

CID = "conv-executor-narration"


# ── 替身（只有 ctx 是替身，存储不是） ──────────────────────────────────


@dataclass
class FakeCtx:
    """跑一个计划所需的最小 context。

    自校验在 :func:`test_the_double_speaks_the_real_contexts_language` 里：
    ``run`` 的签名对着真的 ``ExecutionContext.run`` 核。一个签名已经漂了的替身
    会让整份文件「证明」一条真机上到不了的路径 —— 本仓今晚刚被这件事咬过。
    """

    run_log: list[tuple[str, dict]] = field(default_factory=list)
    fail_on: str = ""

    def run(self, skill_name: str, params: dict, version: str | None = None) -> SkillResult:
        # dict(params) 而不是 params：记下的是**这一刻**的那一份，
        # 后面谁改了都不影响对账（这正是旁白侧 _snapshot 在做的同一件事）。
        self.run_log.append((skill_name, dict(params)))
        if skill_name == self.fail_on:
            return SkillResult(skill_name=skill_name, success=False,
                               error="仪器没有响应")
        return SkillResult(skill_name=skill_name, success=True, data={})


class _ExplodingStore(ConversationStore):
    """真 store 的子类，只让写**每次都抛**。

    继承而不是另写一个类：签名、seq 分配、裁剪规则全部原样继承，所以这个替身
    只可能在「抛异常」这一件事上和真的不同。
    """

    def append_message(self, *a, **kw):  # type: ignore[override]
        raise RuntimeError("旁白存储炸了")


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path):
    narration.reset_for_tests()
    st = ConversationStore(tmp_path / "chat" / "c.db")
    narration.set_store(st)
    yield st
    narration.flush(3.0)
    narration.reset_for_tests()


def _lines(store: ConversationStore) -> list[dict]:
    assert narration.flush(3.0), "旁白写线程没排空"
    return [r for r in store.messages_since(CID, 0)
            if r["kind"] == narration.NARRATION_KIND]


# ForgeAuTip 里真实出现的那种三步：扫一张图 → 打一发脉冲 → 扎一下。
# key 全部取自真技能的 SkillMetadata（见 test_narration_templates_gate）。
PLAN = [
    CompositeStep(step_id="scan", skill_name="ScanAt",
                  params={"center_x_m": 1e-7, "center_y_m": -2e-7, "size_m": 5e-8}),
    CompositeStep(step_id="pulse", skill_name="TipPulse",
                  params={"pulse_v": 10.0, "duration_s": 0.5, "count": 2}),
    CompositeStep(step_id="poke", skill_name="TipShapeWithReadback",
                  params={"tip_lift_m": -3e-10, "bias_lift_v": 0.05}),
]


def _run(ctx, plan=None) -> bool:
    ex = GraphExecutor("FakeForge", ctx)
    with turn_scope(conversation_id=CID, run_id="run-exec"):
        return ex.run_plan(iter(list(plan if plan is not None else PLAN)))


# ── ⓪ 替身自校验 ────────────────────────────────────────────────────────


def test_the_double_speaks_the_real_contexts_language():
    """``FakeCtx.run`` 的签名必须和真的 ``ExecutionContext.run`` 对得上。

    对不上的话，这份文件测的是一个真机上不存在的调用形状 —— 那正是
    「替身说了真组件永远不会说的话」的那一类，本仓刚为它付过几个月的学费。
    """
    real = inspect.signature(ExecutionContext.run)
    fake = inspect.signature(FakeCtx.run)
    assert list(fake.parameters) == list(real.parameters), (
        f"替身 run{fake} 与真的 run{real} 参数不一致")
    assert real.return_annotation == "SkillResult"


# ── ① 对账 ──────────────────────────────────────────────────────────────


def test_every_known_step_narrates_before_it_runs(store):
    ctx = FakeCtx()
    assert _run(ctx) is True
    rows = _lines(store)
    kinds = [json.loads(r["meta"])["nk"] for r in rows]
    assert kinds == ["scan_at", "tip_pulse", "poke"]


def test_facts_equal_the_params_that_actually_reached_the_skill(store):
    """旁白记下的值 == ``ctx.run`` 真正收到的值。**逐值对账，不是「有就行」。**

    这条断言是整套设计的收口：句子里的 10 V 之所以是真的，不是因为模板写得对，
    而是因为交给旁白的 ``params`` 和交给 ``skill.execute`` 的是同一份。
    """
    ctx = FakeCtx()
    _run(ctx)
    rows = _lines(store)
    delivered = {name: params for name, params in ctx.run_log}

    for row in rows:
        meta = json.loads(row["meta"])
        facts = meta["facts"]
        for path, value in facts.items():
            assert path.startswith("params."), path
            key = path.split(".", 1)[1]
            skill = {"scan_at": "ScanAt", "tip_pulse": "TipPulse",
                     "poke": "TipShapeWithReadback"}[meta["nk"]]
            assert delivered[skill][key] == value, (
                f"{skill}.{key}: 旁白说 {value!r}，实际下发 {delivered[skill][key]!r}")


def test_the_sentence_carries_the_real_numbers(store):
    ctx = FakeCtx()
    _run(ctx)
    texts = [r["text"] for r in _lines(store)]
    assert "50 nm" in texts[0], texts[0]          # size_m = 5e-8
    assert "10 V" in texts[1] and "500 ms" in texts[1], texts[1]
    assert "2 发" in texts[1], texts[1]           # count = 2
    assert "300 pm" in texts[2], texts[2]         # tip_lift_m = -3e-10
    # 2026-08-18 术语化:「往表面里扎」→「压入表面」。钉的始终是**方向**
    # (负的 tip_lift_m 是往里压,不是往上抬),不是某一个字。
    assert "压入" in texts[2], "负的 tip_lift_m 是往里压，不是往上抬"


def test_a_missing_number_degrades_to_a_sentence_without_numbers(store):
    """缺必需字段 → fallback，**不是**编一个。

    ``TipPulse`` 的 ``pulse_v``/``duration_s`` 都是可选参数（留空时由针尖策略表
    在下发时决定），所以「没给」是真实会发生的常态，不是异常。
    """
    ctx = FakeCtx()
    _run(ctx, [CompositeStep(step_id="p", skill_name="TipPulse", params={})])
    row = _lines(store)[0]
    assert not any(ch.isdigit() for ch in row["text"]), row["text"]
    assert json.loads(row["meta"])["degraded"] is True


def test_unknown_skills_say_nothing(store):
    """查不到模板就**不发** —— 不是发一句通用的「正在执行某个步骤」。"""
    ctx = FakeCtx()
    _run(ctx, [CompositeStep(step_id="s", skill_name="SetBias",
                             params={"bias_v": 0.1})])
    assert _lines(store) == []


# ── ② 失败也要说，而且要说清楚跑还在不在跑 ──────────────────────────────


def test_a_mandatory_failure_says_the_task_stopped(store):
    ctx = FakeCtx(fail_on="TipPulse")
    assert _run(ctx) is False
    rows = _lines(store)
    last = rows[-1]
    meta = json.loads(last["meta"])
    assert meta["nk"] == "step_failed"
    assert meta["facts"]["continued"] is False
    assert "仪器没有响应" in last["text"]
    assert "到此停下" in last["text"]


def test_an_optional_failure_says_we_carry_on(store):
    """`optional` 步骤失败 ⇒「继续往下走」。

    「没成」和「没成、而且整件事停了」是两句话。把它们说成同一句，用户就得
    自己去猜现在还在不在跑 —— 而「看起来完成了其实没有」正是本仓的常年投诉。
    """
    plan = [CompositeStep(step_id="p", skill_name="TipPulse", optional=True,
                          params={"pulse_v": 1.0, "duration_s": 0.01}),
            CompositeStep(step_id="s", skill_name="StartScan", params={})]
    ctx = FakeCtx(fail_on="TipPulse")
    assert _run(ctx, plan) is True
    metas = [json.loads(r["meta"]) for r in _lines(store)]
    failed = [m for m in metas if m["nk"] == "step_failed"][0]
    assert failed["facts"]["continued"] is True
    assert "step_failed" in [m["nk"] for m in metas]
    assert "scan_start" in [m["nk"] for m in metas], "失败之后那一步照样要说"


# ── ③ §Q8：旁白链路整条烂掉，实验照跑 ──────────────────────────────────


def test_a_broken_narration_link_cannot_break_a_running_experiment(tmp_path):
    """存储每次写都抛 ⇒ 三步**全部执行完且全部成功**。

    这是设计文档 §Q8 要求「证明」的那一条。判据不是「没崩」——
    是三个子技能都真的被调用了，且 ``run_plan`` 返回 True。
    """
    narration.reset_for_tests()
    narration.set_store(_ExplodingStore(tmp_path / "boom" / "c.db"))
    try:
        ctx = FakeCtx()
        ok = _run(ctx)
        narration.flush(3.0)
        assert ok is True
        assert [name for name, _ in ctx.run_log] == [
            "ScanAt", "TipPulse", "TipShapeWithReadback"]
        assert narration.stats()["errors"] >= 3, \
            "写线程没吞到异常 —— 那说明它根本没跑，这条断言就是假绿的"
    finally:
        narration.reset_for_tests()


def test_with_narration_switched_off_the_plan_is_untouched(tmp_path, monkeypatch):
    """``MAST_CHAT_NARRATION=0`` ⇒ 一行都不写，计划照跑。"""
    monkeypatch.setenv("MAST_CHAT_NARRATION", "0")
    narration.reset_for_tests()
    st = ConversationStore(tmp_path / "off" / "c.db")
    narration.set_store(st)
    try:
        ctx = FakeCtx()
        assert _run(ctx) is True
        assert len(ctx.run_log) == 3
        narration.flush(1.0)
        assert st.messages_since(CID, 0) == []
    finally:
        narration.reset_for_tests()


def test_no_conversation_means_no_narration_but_the_plan_still_runs(store):
    """后台唤醒跑（没有会话 id）：旁白一条没有，计划照跑完。

    这是设计文档 §8 开放问题 3 明写的**已知功能缺口**，不是 bug ——
    正确的补法是让唤醒调度自己建/绑一个会话，而不是让旁白去猜。
    钉在这里，是为了让「猜一个会话」这个改动一旦有人写就会红。
    """
    ctx = FakeCtx()
    ex = GraphExecutor("FakeForge", ctx)
    with turn_scope(conversation_id=""):
        assert ex.run_plan(iter(list(PLAN))) is True
    assert len(ctx.run_log) == 3
    assert _lines(store) == []
