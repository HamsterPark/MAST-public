"""BufferHITL 中间件 + ``dedupe_event_refs`` reducer。

## 这份文件在 ⑰(2026-08-08)之前钉的是什么,以及为什么全部重写

它原来钉的是一套**确认框机制**:关键事件 → ``interrupt()`` → 用户在审批面板
批准/拒绝 → 工具闸门开/关,外加「拒绝不是单向门」「审批通道死掉要 fail closed」
「一次事件只再问一遍」等一整套状态机。那套机制及其全部测试是三次真实事故的产物,
每一条都对:

* 2026-07-27 —— 五个 CRITICAL interrupt 全部发出,run 却照扫不误(审批通道死了,
  ``interrupt()`` 直接返回)⇒ 有了工具闸门;
* 2026-07-28 —— approve / reject / 900s 超时三种结局一模一样 ⇒ 有了裁决读取;
* 2026-08-04 —— 私聊里按一次「拒绝」把仪器卡到进程重启 ⇒ 有了再问一次 + 带外入口。

**现在这套机制整个不存在了。** 判定是:这类缓冲区关键事件属于过度设计,应当移除。

实证是:这条链路弹出的确认框**零真阳性**,全是操作瞬态误报;而真正拦下危险的是
**拒绝型防护**(不弹框、不等人、越界直接说不),它们一条都没动。

所以上面那三条历史 bug 的测试不是「删掉」而是**语义换了**:
「该弹的没弹」→「都不弹」。它们各自的教训在下面被改写成新语义下**仍然成立**的
断言(见 ``TestTheOldInterruptMachineIsGone`` 一节),而不是消失 —— 一条被删掉的
测试和一条从没写过的测试长得一模一样。

## 现在钉的是两侧

* **不再打断**:任何 kind、任何 severity 都不产生 ``interrupt()``、不关工具闸门;
  而且是**结构性**的(模块里没有 interrupt 调用点、``wrap_tool_call`` 无条件透传),
  不是靠「某个集合恒为空」间接失效;
* **照样记录**:事件照进 ``event_refs``、照留在 buffer 里、照进诊断台账。
  这一半钉得和上一半一样硬 —— 把事件**删掉**从用户的椅子上看和「不打断」
  一模一样,直到他去找证据的那一天。
"""
from __future__ import annotations

# parents: 0=test file, 1=_shared, 2=agents, 3=v2, 4=tests, 5=repo root
import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import asyncio

import pytest
import pytest_asyncio

from mast.agents._shared.buffer_hitl import (
    DEFAULT_HITL_EVENT_KINDS,
    BufferHITLMiddleware,
    make_buffer_hitl_middleware,
)
from mast.agents.state import dedupe_event_refs
from mast.buffer.schemas import (
    Severity,
    TipQuality,
    TipStatus,
    VisionEvent,
    VisionEventType,
    make_e_stop,
)
from mast.buffer.service import BufferService


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def buffer(tmp_path):
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        yield buf
    finally:
        await buf.stop()


def _make_drop_event(*, seqno: int, critical: bool) -> VisionEvent:
    """视觉边沿检测器产生的**形态类** TIP_QUALITY_DROP。"""
    return VisionEvent(
        seqno=seqno,
        kind=VisionEventType.TIP_QUALITY_DROP,
        severity=Severity.CRITICAL if critical else Severity.WARN,
        payload={
            "quality": "bad" if critical else "degraded",
            "confidence": 0.9,
            "scan_id": "s",
            "frame_idx": seqno,
        },
        cause_ref=f"tip_status#{seqno}",
    )


def _make_physical_event(*, seqno: int, signal: str = "current_saturation",
                         critical: bool = True) -> VisionEvent:
    """电流监控的**物理类** CRITICAL:前放到轨 / 链死 / 巨幅台阶。

    与上面那条是**同一个 kind** —— 这正是分类器必须读 payload 而不是读 kind 的
    原因,⑰ 之后这个区分仍然存在,只是它决定的是通知怎么写,不再决定拦不拦。"""
    return VisionEvent(
        seqno=seqno,
        kind=VisionEventType.TIP_QUALITY_DROP,
        severity=Severity.CRITICAL if critical else Severity.WARN,
        payload={"signal": signal, "source": "current_monitor",
                 "scan_id": "s", "summary_zh": "前放到轨"},
        cause_ref=f"current_monitor#{seqno}",
    )


class _FakeMeta:
    def __init__(self, *, category="write", tags=(), capabilities=frozenset()):
        self.category = type("C", (), {"value": category})()
        self.tags = list(tags)
        self.capabilities = capabilities


class _FakeTool:
    def __init__(self, name, meta=None):
        self.name = name
        self.metadata = {"skill_metadata": meta} if meta is not None else {}


class _FakeRequest:
    def __init__(self, tool):
        self.tool = tool
        self.tool_call = {"id": "call_1"}


def _blocked(mw, tool) -> bool:
    """闸门此刻拒不拒绝这个工具。⑰ 之后答案恒为 ``False``。"""
    sentinel = object()
    out = mw.wrap_tool_call(_FakeRequest(tool), lambda _r: sentinel)
    return out is not sentinel


def _forward_tool():
    """一个「推进实验」的工具:WRITE,不在解药名单里,也没有解药标签/能力。"""
    return _FakeTool("ScanArea", _FakeMeta())


# ─────────────────────────────────────────────────────────────────────
# dedupe_event_refs — reducer properties(⑰ 未触及)
# ─────────────────────────────────────────────────────────────────────

class TestDedupeEventRefsReducer:
    def test_identity_right_empty(self):
        x = ["e1", "e2", "e3"]
        assert dedupe_event_refs(x, []) == x

    def test_identity_left_empty(self):
        y = ["e1", "e2"]
        assert dedupe_event_refs([], y) == y

    def test_idempotent(self):
        x = ["e1", "e2"]
        assert dedupe_event_refs(x, x) == x

    def test_dedupes_while_preserving_first_seen_order(self):
        assert dedupe_event_refs(["e1", "e2"], ["e2", "e3"]) == ["e1", "e2", "e3"]

    def test_associative(self):
        a, b, c = ["e1"], ["e2", "e1"], ["e3", "e2"]
        left = dedupe_event_refs(dedupe_event_refs(a, b), c)
        right = dedupe_event_refs(a, dedupe_event_refs(b, c))
        assert left == right


# ─────────────────────────────────────────────────────────────────────
# 记录侧 —— 这一半必须一个字节不少
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_middleware_subscribes_on_init(buffer):
    """构造时每个订阅的 kind 注册一个队列。"""
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        for kind in DEFAULT_HITL_EVENT_KINDS:
            assert kind in mw._subs
            assert mw._subs[kind] in buffer._subs[kind]
    finally:
        mw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("signal", ["current_saturation", "current_freeze",
                                    "current_giant_spike"])
async def test_every_physical_critical_is_recorded_and_stops_nothing(buffer, signal):
    """**⑰ 的主判据(物理类)。** 连「前放到轨」都只记录、不打断。

    ⑭ 时这三条里只有 giant_spike 在修针期间豁免;⑰ 之后三条一视同仁 —— 因为拦住
    「针已经压进表面」的是 ``current_monitor`` 自己的 run 级 halt 与撞针状态机,
    不是这个确认框。这个确认框做的只是在事后要求一个人按按钮。
    """
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        ev = _make_physical_event(seqno=buffer.next_seq(), signal=signal)
        buffer.emit_event(ev)
        await asyncio.sleep(0)

        update = mw.before_model(state={}, runtime=None)
        # 记录:事件 id 进了 state。
        assert update is not None and update["event_refs"] == [ev.event_id]
        # 不打断:闸门开着,推进类工具照跑。
        assert mw.gate_state()["closed"] is False
        assert not _blocked(mw, _forward_tool())
        # …而且这条事件确实被**看见并计数**了(「没弹窗」≠「没发生」)。
        assert mw.gate_state()["recorded_not_escalated"] == 1
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_a_morphology_drop_is_recorded_but_never_interrupts(buffer):
    """GOOD → BAD 的视觉边沿:记录,不停任何东西。

    2026-08-05 起就是这个语义(⑬),⑰ 只是把它扩到了全部事件。「记录」这一半钉得
    和「不打断」一样硬:把事件删掉从外面看是一样的,直到有人去找证据。
    """
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        buffer.put_tip_status(TipStatus(
            seqno=buffer.next_seq(), quality=TipQuality.GOOD, confidence=0.9,
            scan_id="s", frame_idx=0,
        ))
        buffer.put_tip_status(TipStatus(
            seqno=buffer.next_seq(), quality=TipQuality.BAD, confidence=0.9,
            scan_id="s", frame_idx=1,
        ))
        await asyncio.sleep(0)

        update = mw.before_model(state={}, runtime=None)
        assert update is not None and len(update["event_refs"]) == 1
        assert mw.gate_state()["closed"] is False
        assert mw.gate_state()["recorded_not_escalated"] == 1
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_e_stop_is_recorded_and_does_not_open_a_dialog(buffer):
    """E_STOP 也不再弹框 —— 而**停止权一点没少**。

    停下来的是 abort 闩锁:``BufferService.register_critical_hook`` →
    ``runtime._estop_sets_abort`` → ``skill_adapter`` 的 0 号闸门拒绝一切新仪器
    动作。这个中间件从来不是 E_STOP 的执行路径,它只是在上面又叠了一个
    「请确认你刚才按的急停」的对话框。

    这一条同时是**保留清单**那条要求的落点:「E_STOP / 用户主动中止不许动」——
    动的是那个多余的对话框,不是通道。通道由
    ``tests/v2/unit/core/test_forensics_20260727_tip_halt.py`` 与 runtime 的钩子
    测试各自钉着。
    """
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        ev = make_e_stop(reason="user", detail="kill switch hit",
                         seqno=buffer.next_seq())
        buffer.emit_event(ev)
        await asyncio.sleep(0)

        update = mw.before_model(state={}, runtime=None)
        assert update is not None and update["event_refs"] == [ev.event_id]
        assert mw.gate_state()["closed"] is False
        assert not _blocked(mw, _forward_tool())
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_the_estop_abort_latch_is_still_wired(buffer):
    """接着上一条:证明「停止权没少」不是靠推断。

    直接走 ``register_critical_hook``(E_STOP 真正的那条路),断言一个 E_STOP 事件
    确实到达了 threading 级消费者。这是 ``pipeline/main`` 与 ``runtime`` 用来 set
    abort Event 的同一个入口。
    """
    seen: list = []
    buffer.register_critical_hook(
        lambda ev: seen.append(ev) if ev.kind is VisionEventType.E_STOP else None)
    buffer.emit_event(make_e_stop(reason="user", detail="x",
                                  seqno=buffer.next_seq()))
    await asyncio.sleep(0)
    assert seen, "E_STOP 没有到达 critical hook —— 那才是真正的停止通道"


@pytest.mark.asyncio
async def test_non_critical_warn_is_recorded_but_not_counted_as_notable(buffer):
    """WARN 级事件进 ``event_refs``,但不单独占一行台账。

    2553 条 warn 刷屏是这条门槛存在的理由 —— 台账被刷满就等于没有台账。
    """
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        warn_ev = _make_drop_event(seqno=buffer.next_seq(), critical=False)
        buffer.emit_event(warn_ev)
        await asyncio.sleep(0)

        update = mw.before_model(state={}, runtime=None)
        assert update is not None and update["event_refs"] == [warn_ev.event_id]
        assert mw.gate_state()["recorded_not_escalated"] == 0
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_empty_queues_return_none(buffer):
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        assert mw.before_model(state={}, runtime=None) is None
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_batched_events_all_reach_event_refs(buffer):
    """两帧之间来的两条事件都要进 ``event_refs`` —— 突发不该吃掉记录。"""
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        e1 = make_e_stop(reason="watchdog", detail="x", seqno=buffer.next_seq())
        buffer.emit_event(e1)
        e2 = _make_physical_event(seqno=buffer.next_seq())
        buffer.emit_event(e2)
        await asyncio.sleep(0)

        update = mw.before_model(state={}, runtime=None)
        assert update is not None
        assert set(update["event_refs"]) == {e1.event_id, e2.event_id}
        assert mw.gate_state()["recorded_not_escalated"] == 2
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_the_event_stays_on_the_bus_for_other_readers(buffer):
    """中间件消费事件**不会**把它从别人眼前拿走。

    ``ReadHardwareEvents`` / 面板 / 监控各自订阅同一条总线。如果本中间件的
    「记录」其实是「独占消费」,那么「面板照看得到」就是一句空话。
    """
    witness = buffer.subscribe(VisionEventType.TIP_QUALITY_DROP)
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        ev = _make_physical_event(seqno=buffer.next_seq())
        buffer.emit_event(ev)
        await asyncio.sleep(0)
        mw.before_model(state={}, runtime=None)

        seen = witness.get_nowait()
        assert seen.event_id == ev.event_id
        assert seen.kind is VisionEventType.TIP_QUALITY_DROP
    finally:
        buffer.unsubscribe(VisionEventType.TIP_QUALITY_DROP, witness)
        mw.close()


@pytest.mark.asyncio
async def test_a_notable_event_writes_a_diagnostics_line(buffer):
    """**通知**那一半。台账里那一行就是「已通知」的可查落点。

    这条改动要能回答「本来会拦我几次」—— 那是一个
    ``kind=notice_only`` 的查询,不是一句「相信我,它记了」。
    """
    from mast.core import diagnostics as diag

    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        before = len(diag.recent(500, kinds=("notice_only",)))
        buffer.emit_event(_make_physical_event(seqno=buffer.next_seq()))
        await asyncio.sleep(0)
        mw.before_model(state={}, runtime=None)

        rows = diag.recent(500, kinds=("notice_only",))
        assert len(rows) == before + 1
        row = rows[0]
        assert row["subject"].startswith("buffer:")
        assert row["event_class"] == "physical_sustained"
        assert "只通知不打断" in row["reason"]
    finally:
        mw.close()


# ─────────────────────────────────────────────────────────────────────
# 不打断侧 —— 结构性,不是「某个集合恰好为空」
# ─────────────────────────────────────────────────────────────────────

class TestTheOldInterruptMachineIsGone:
    """三条历史 bug 的教训,在新语义下改写成仍然成立的断言。

    删掉它们会让「这些事故的结论现在还算不算数」没有落点 —— 而一条被删掉的测试
    和一条从没写过的测试长得一模一样。
    """

    def test_the_module_has_no_interrupt_call_site_at_all(self):
        """**最强的那颗钉子。** 不是「不会走到 interrupt」,是「没有 interrupt」。

        ⑰ 之前这里可以靠让某个白名单恒为空来「关掉」打断 —— 那种关法只要有人把
        一条判据接回去就复活了,而接回去的人不会先读注释。所以打断链是**删掉**
        的:模块不 import ``langgraph``,源码里没有 ``interrupt(`` 这个调用。
        """
        import ast

        import mast.agents._shared.buffer_hitl as mod

        # 走 AST 而不是子串匹配:注释和 docstring 里**必须**能写清楚这里曾经有过
        # 什么(否则下一个人只会看到一段没有来由的代码),而子串匹配会把那些说明
        # 一起判成违规 —— 那样的闸门会逼人删掉解释,正好删掉最该留的东西。
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        calls = {
            n.func.id if isinstance(n.func, ast.Name)
            else getattr(n.func, "attr", "")
            for n in ast.walk(tree) if isinstance(n, ast.Call)
        }
        assert "interrupt" not in calls, "buffer_hitl 里又出现了 interrupt 调用点"

        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                imported.add(n.module)
                imported.update(f"{n.module}.{a.name}" for a in n.names)
        assert not any(m.startswith("langgraph.types") for m in imported), (
            f"buffer_hitl 又 import 了 langgraph.types:{sorted(imported)}")

        assert not hasattr(mod, "_verdict_is_approval"), (
            "审批裁决解析又回来了 —— 没有裁决可读,因为没有裁决")

    def test_escalates_to_operator_is_false_for_everything(self):
        """任何 kind、任何 severity、任何 payload,答案都是 no。"""
        from mast.agents._shared.buffer_hitl import escalates_to_operator

        for ev in (
            make_e_stop(reason="user", detail="x", seqno=1),
            _make_physical_event(seqno=2, signal="current_saturation"),
            _make_physical_event(seqno=3, signal="current_freeze"),
            _make_physical_event(seqno=4, signal="current_giant_spike"),
            _make_physical_event(seqno=5, signal="some_rule_from_tomorrow"),
            _make_drop_event(seqno=6, critical=True),
            _make_drop_event(seqno=7, critical=False),
        ):
            assert escalates_to_operator(ev) is False

    @pytest.mark.asyncio
    async def test_the_tool_gate_passes_through_even_if_someone_forces_it_shut(
            self, buffer):
        """**变异测试。** 强行给实例塞一个未解决事件,闸门仍然透传。

        2026-07-27 的教训是「``interrupt()`` 只是通告,真正停住 run 的是闸门」——
        所以割掉打断链时,闸门必须**自己**被删掉,而不是靠「``_unresolved`` 永远为
        空」间接失效。这条断言把那个区分变成可证的:即使有人把状态塞回去,
        ``wrap_tool_call`` 也不会拦。
        """
        mw = make_buffer_hitl_middleware(buffer=buffer)
        try:
            mw._unresolved = {"e1": _make_physical_event(seqno=1)}  # 人为塞回去
            mw._degraded = True
            assert not _blocked(mw, _forward_tool())
            assert not _blocked(mw, _FakeTool("SetBias", _FakeMeta()))
            assert mw.gate_state()["closed"] is False
        finally:
            mw.close()

    @pytest.mark.asyncio
    async def test_the_async_tool_path_passes_through_too(self, buffer):
        """异步孪生也得钉 —— 生产路径走的是 ``awrap_tool_call``。

        2026-07-28 的取证里,同步覆盖了而异步没覆盖,正是这种形状。
        """
        mw = make_buffer_hitl_middleware(buffer=buffer)
        try:
            mw._unresolved = {"e1": _make_physical_event(seqno=1)}
            sentinel = object()

            async def handler(_r):
                return sentinel

            out = await mw.awrap_tool_call(_FakeRequest(_forward_tool()), handler)
            assert out is sentinel
        finally:
            mw.close()

    @pytest.mark.asyncio
    async def test_a_rejection_can_no_longer_wedge_anything(self, buffer):
        """2026-08-04(私聊里一次「拒绝」把仪器卡到进程重启)的**新语义版本**。

        原来的修法是给拒绝配「再问一次」+ 带外清除入口。现在没有拒绝可按 ——
        所以那个死锁在结构上不可能再发生。钉的是结论而不是当年的机制:反复调用
        ``before_model``,闸门永远开着,推进类工具永远能跑。
        """
        mw = make_buffer_hitl_middleware(buffer=buffer)
        try:
            buffer.emit_event(_make_physical_event(seqno=buffer.next_seq()))
            await asyncio.sleep(0)
            for _ in range(5):
                mw.before_model(state={}, runtime=None)
                assert mw.gate_state()["closed"] is False
                assert not _blocked(mw, _forward_tool())
        finally:
            mw.close()

    @pytest.mark.asyncio
    async def test_a_dead_approval_channel_is_no_longer_a_failure_mode(self, buffer):
        """2026-07-27(审批通道死了,``interrupt()`` 直接返回)的**新语义版本**。

        那次事故的根因是「有一个必须有人回答的问题,而没人能回答」。现在没有问题
        要问 ⇒ 通道死不死都不影响推进。``degraded`` 恒为 False,不是因为通道健康,
        而是因为**没有通道**。
        """
        mw = make_buffer_hitl_middleware(buffer=buffer)
        try:
            buffer.emit_event(make_e_stop(reason="user", detail="x",
                                          seqno=buffer.next_seq()))
            await asyncio.sleep(0)
            mw.before_model(state={}, runtime=None)
            assert mw.gate_state()["degraded"] is False
            assert mw.gate_state()["awaiting"] == 0
        finally:
            mw.close()

    @pytest.mark.asyncio
    async def test_gate_compat_shims_answer_honestly(self, buffer):
        """``reset_all_gates`` / ``resolve_all_gates`` 现在如实回答「0 个闸门」。

        它们**不能删**:``api/routes/orchestrator`` 每个新任务开头调 reset、
        ``api/routes/agents_control`` 的清除端点调 resolve、
        ``skills/builtins/hardware_events`` 调 ``gate_states``。返回 0 是回答,
        不是失败 —— 而 ``gate_states`` 仍然要报出真实的「记录了多少条」。
        """
        from mast.agents._shared.buffer_hitl import (
            gate_states, reset_all_gates, resolve_all_gates,
        )

        mw = make_buffer_hitl_middleware(buffer=buffer)
        try:
            buffer.emit_event(_make_physical_event(seqno=buffer.next_seq()))
            await asyncio.sleep(0)
            mw.before_model(state={}, runtime=None)

            assert reset_all_gates("new run") == 0
            assert resolve_all_gates("operator cleared") == 0
            states = [s for s in gate_states()
                      if s["recorded_not_escalated"] >= 1]
            assert states, "gate_states 必须还看得见这条中间件记了东西"
            assert all(s["closed"] is False for s in gate_states())
        finally:
            mw.close()


# ─────────────────────────────────────────────────────────────────────
# 分类器 —— 判据没删,只是下游从「拦不拦」变成了「通知怎么写」
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def in_tip_work(monkeypatch):
    """假装此刻正在跑 ForgeAuTip(令牌持有者是修针技能)。"""
    from mast.core import tip_intent
    monkeypatch.setattr(tip_intent, "active_tip_work", lambda: "ForgeAuTip",
                        raising=True)


@pytest.mark.parametrize("signal,expect", [
    ("current_saturation", "physical_sustained"),
    ("current_freeze", "physical_sustained"),
    ("current_giant_spike", "physical_transient"),
    ("some_rule_from_tomorrow", "morphology"),
])
def test_classify_event_reads_the_payload_not_the_kind(signal, expect):
    """物理类与形态类是**同一个 kind**,区别只在 payload 里。

    这个区分现在只影响通知措辞,但它必须活着:把「前放到轨」和「视觉觉得针尖变差
    了」印成同一句话,等于没通知。
    """
    from mast.agents._shared.buffer_hitl import classify_event

    assert classify_event(_make_physical_event(seqno=1, signal=signal)) == expect


def test_classify_event_names_the_tip_work_transient(in_tip_work):
    """修针期间的瞬变有自己的名字 —— 「本职动作的签名」,不是「巨幅瞬变」。"""
    from mast.agents._shared.buffer_hitl import classify_event

    ev = _make_physical_event(seqno=1, signal="current_giant_spike")
    assert classify_event(ev) == "tip_work_transient"


def test_classify_event_never_raises_on_garbage():
    """分类器在事件发布线程上跑,永远不能抛 —— 给它一个不是事件的东西也一样。"""
    from mast.agents._shared.buffer_hitl import classify_event

    assert classify_event(object()) == "other"  # type: ignore[arg-type]


def test_classify_event_falls_back_when_the_payload_read_explodes(monkeypatch):
    """payload 读炸了 ⇒ ``morphology``(最安静的一类),而不是抛。"""
    from mast.agents._shared import buffer_hitl

    def _boom():
        raise RuntimeError("alerts down")

    monkeypatch.setattr(buffer_hitl, "physical_current_signals", _boom,
                        raising=True)
    assert buffer_hitl.classify_event(
        _make_physical_event(seqno=1)) == "morphology"


def test_e_stop_and_crash_do_not_look_at_the_payload():
    from mast.agents._shared.buffer_hitl import classify_event

    assert classify_event(make_e_stop(reason="user", detail="x", seqno=1)) == "e_stop"


def test_the_two_buckets_still_cover_every_physical_signal():
    """每个物理信号都必须被归类 —— 「忘了归」不该悄悄变成某一类。"""
    from mast.agents._shared.buffer_hitl import (
        SUSTAINED_PHYSICAL_SIGNALS,
        TRANSIENT_PHYSICAL_SIGNALS,
        _PHYSICAL_SIGNAL_FALLBACK,
    )

    assert not (TRANSIENT_PHYSICAL_SIGNALS & SUSTAINED_PHYSICAL_SIGNALS)
    unclassified = _PHYSICAL_SIGNAL_FALLBACK - (
        TRANSIENT_PHYSICAL_SIGNALS | SUSTAINED_PHYSICAL_SIGNALS)
    assert not unclassified, f"没归类的物理信号:{sorted(unclassified)}"


def test_physical_signals_are_derived_from_the_alert_engine():
    """派生集只增不减:上游加第四条 CRIT_RULE 不需要在这里再写一遍。"""
    from mast.agents._shared.buffer_hitl import physical_current_signals
    from mast.monitoring.alerts import critical_signals

    assert set(critical_signals()) <= set(physical_current_signals())


def test_an_unreadable_tip_intent_does_not_claim_tip_work(monkeypatch):
    """读不到令牌 ⇒ 不算修针期间。"""
    from mast.agents._shared import buffer_hitl
    from mast.core import tip_intent

    def _boom():
        raise RuntimeError("lock unavailable")

    monkeypatch.setattr(tip_intent, "active_tip_work", _boom, raising=True)
    assert buffer_hitl.exempt_during_tip_work("current_giant_spike") is False


# ─────────────────────────────────────────────────────────────────────
# Lifecycle — unsubscribe on close prevents queue retention
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_unsubscribes_from_buffer(buffer):
    mw = BufferHITLMiddleware(buffer=buffer)
    for kind in DEFAULT_HITL_EVENT_KINDS:
        assert any(q is mw._subs[kind] for q in buffer._subs[kind])

    queues_before_close = dict(mw._subs)
    mw.close()

    for kind, q in queues_before_close.items():
        assert q not in buffer._subs.get(kind, []), (
            f"Queue for {kind.value} still registered after close()")
    assert mw._subs == {}
    assert mw._closed is True


@pytest.mark.asyncio
async def test_close_is_idempotent(buffer):
    mw = BufferHITLMiddleware(buffer=buffer)
    mw.close()
    mw.close()


@pytest.mark.asyncio
async def test_before_model_after_close_returns_none(buffer):
    mw = BufferHITLMiddleware(buffer=buffer)
    buffer.emit_event(make_e_stop(reason="user", detail="x",
                                  seqno=buffer.next_seq()))
    await asyncio.sleep(0)
    mw.close()
    assert mw.before_model(state={}, runtime=None) is None


# ─────────────────────────────────────────────────────────────────────
# Remedy 词汇 —— 闸门没了,定义还在(instrument_lock 抄的是这份)
# ─────────────────────────────────────────────────────────────────────

def test_every_retract_skill_in_the_tree_is_recognised_as_a_remedy():
    """反向守卫:树里每个退针技能都在表里。

    ``EmergencyRetract`` 当初就是这么漏掉的(WRITE + AUTO + 无标签,每条分支都掉
    下去)。⑰ 之后本模块不再消费这个判据,但 ``core/instrument_lock`` 的
    ``BYPASS_NAMES``/``BYPASS_TAGS`` 是它的姊妹表,漂移仍然会伤人。
    """
    from mast.agents._shared.buffer_hitl import BufferHITLMiddleware
    from mast.skills.builtins import tip as tip_skills

    for cls_name in ("SafeRetract", "EmergencyRetract"):
        meta = getattr(tip_skills, cls_name)().metadata()
        tool = _FakeTool(meta.name, meta)
        assert BufferHITLMiddleware._is_remedy(tool)


def test_remedy_vocabulary_matches_instrument_lock():
    """两张姊妹表必须说同一件事 —— 它们的注释互相引用,漂了没人会发现。"""
    from mast.agents._shared.buffer_hitl import REMEDY_TAGS, REMEDY_TOOL_NAMES
    from mast.core.instrument_lock import BYPASS_NAMES, BYPASS_TAGS

    assert BYPASS_TAGS == REMEDY_TAGS
    assert BYPASS_NAMES <= REMEDY_TOOL_NAMES


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
