"""Synthetic coverage for envelope diagnosis and refused-approach sequence guards.

In-scale over-limit values require an envelope diagnosis, while magnitude slips
need unit guidance. A refused escalation must outlive the call that made it.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from typing import Any  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from mast.agents._shared.buffer_hitl import (  # noqa: E402
    REMEDY_TOOL_NAMES,
    make_buffer_hitl_middleware,
)
from mast.agents._shared.safety_mw import SafetyGate, SafetyGateMiddleware  # noqa: E402
from mast.agents._shared.skill_adapter import wrap_skill  # noqa: E402
from mast.buffer.schemas import Severity, VisionEvent, VisionEventType  # noqa: E402
from mast.buffer.service import BufferService  # noqa: E402
from mast.config import SafetyLimits  # noqa: E402
from mast.core import safety_escalation as esc  # noqa: E402
from mast.core.types import SkillResult  # noqa: E402
from mast.skills.builtins.approach import ApproachTip, AutoApproach, WithdrawTip  # noqa: E402
from mast.skills.builtins.bias import GetBias, SetBias  # noqa: E402
from mast.skills.composite.full_scan import FullScan  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────

def _make_request(tool, args: dict, call_id: str = "t-1") -> Any:
    """A ToolCallRequest-like object (LangChain accepts a dict tool_call)."""
    req = MagicMock()
    req.tool = tool
    req.tool_call = {"name": tool.name, "args": args, "id": call_id,
                     "type": "tool_call"}
    req.state = {}
    req.runtime = MagicMock()
    return req


def _tool(skill_cls):
    return wrap_skill(skill_cls, lambda: MagicMock(safe_call=lambda *a, **k: None))


@pytest.fixture(autouse=True)
def _clean_escalation_ledger():
    """The refusal ledger is process-global by design — isolate every test."""
    esc.clear_approach_refusal()
    yield
    esc.clear_approach_refusal()


# ─────────────────────────────────────────────────────────────────────
# (10) in-scale over-cap must NOT be diagnosed as a dropped exponent
# ─────────────────────────────────────────────────────────────────────

#: Independent synthetic coordinate, 1.2 times the configured default cap.
SYNTHETIC_CENTER_Y_M = 1.8e-06


class TestOverCapDiagnosis:
    def setup_method(self):
        self.gate = SafetyGate(SafetyLimits())
        self.meta = FullScan().metadata()

    def test_synthetic_value_is_not_called_an_exponent_slip(self):
        v = self.gate.check_global_bounds(
            self.meta, {"center_x_m": 0.0, "center_y_m": SYNTHETIC_CENTER_Y_M,
                        "width_m": 5e-7, "height_m": 5e-7})
        assert len(v) == 1, v
        msg = v[0]
        assert "above global safety maximum" in msg
        # The whole point: the old text told the operator's CORRECT value that
        # it "almost certainly meant nanometers and dropped the exponent".
        assert "dropped the exponent" not in msg
        assert "almost certainly meant nanometers" not in msg
        assert "NOT the classic dropped-exponent" in msg

    def test_synthetic_value_message_names_the_knob_and_where_to_turn_it(self):
        v = self.gate.check_global_bounds(
            self.meta, {"center_x_m": 0.0, "center_y_m": SYNTHETIC_CENTER_Y_M,
                        "width_m": 5e-7, "height_m": 5e-7})
        msg = v[0]
        assert "xy_max_m" in msg                 # WHICH limit
        assert "全局安全限制" in msg               # WHERE the operator changes it
        # And it must forbid the dangerous reflex the old text invited.
        assert "Do NOT silently substitute" in msg

    def test_the_reported_factor_is_the_real_factor(self):
        # Report the ratio computed from this synthetic input and its limit.
        v = self.gate.check_global_bounds(
            self.meta, {"center_x_m": 0.0, "center_y_m": SYNTHETIC_CENTER_Y_M,
                        "width_m": 5e-7, "height_m": 5e-7})
        assert "1.2×" in v[0]

    def test_real_exponent_slip_still_gets_the_metres_teaching(self):
        # 1.5 m written for a 15 nm scan — 1e6× the cap. The #118/#145 teaching
        # text must survive untouched, or we trade one dead loop for another.
        v = self.gate.check_global_bounds(
            self.meta, {"center_x_m": 0.0, "center_y_m": 1.5,
                        "width_m": 5e-7, "height_m": 5e-7})
        assert len(v) == 1, v
        msg = v[0]
        # 2026-08-04：四件实质一件不少（点明单位 / 说出是量级丢了 / 给出一个能照抄
        # 的正确值 / 别重发），换的只是那个正确值的形式 —— 长度类参数现在强制
        # SI 前缀，教 '1.5e-8' 会让它在下一步被 parse_si 再拒一次。
        assert "METRE" in msg or "METER" in msg
        assert "lost the magnitude" in msg or "dropped the exponent" in msg
        assert "'15n'" in msg
        assert "do not resend" in msg.lower()

    def test_below_min_in_scale_also_reports_the_envelope(self):
        # Negative coordinates need the same diagnosis on the below-min branch.
        v = self.gate.check_global_bounds(
            self.meta, {"center_x_m": 0.0, "center_y_m": -SYNTHETIC_CENTER_Y_M,
                        "width_m": 5e-7, "height_m": 5e-7})
        assert len(v) == 1, v
        msg = v[0]
        assert "below global safety minimum" in msg
        assert "xy_min_m" in msg
        assert "NOT the classic dropped-exponent" in msg

    def test_inside_the_envelope_still_passes_clean(self):
        v = self.gate.check_global_bounds(
            self.meta, {"center_x_m": 1.0e-06, "center_y_m": 1.2e-06,
                        "width_m": 5e-7, "height_m": 5e-7})
        assert v == []


class TestEnvelopeDefaultIsDeliberate:
    """±1.5 µm is NOT widened here, and this test says why.

    The value has been the shipped default since v0.1.0; it is
    a conservative design-time choice, not a measured property of any rig. The
    target instrument requires independent envelope configuration; this test
    exercises the declared software default without asserting hardware suitability.
    """

    def test_shipped_default_is_unchanged(self):
        lim = SafetyLimits()
        assert lim.xy_max_m == 1.5e-6
        assert lim.xy_min_m == -1.5e-6

    def test_admin_override_actually_unblocks_the_synthetic_position(self, tmp_path):
        # The remedy the error message now points the operator at must work.
        from mast.admin.override_store import ConfigOverrideRegistry

        try:
            ovr_dir = tmp_path / "config" / "overrides"
            ovr_dir.mkdir(parents=True)
            (ovr_dir / "safety_limits.json").write_text(
                '{"xy_min_m": -3e-6, "xy_max_m": 3e-6}', encoding="utf-8")
            ConfigOverrideRegistry.reset()
            reg = ConfigOverrideRegistry(overrides_dir=ovr_dir)
            gate = SafetyGate(SafetyLimits(), registry=reg)
            meta = FullScan().metadata()
            v = gate.check_global_bounds(
                meta, {"center_x_m": 0.0, "center_y_m": SYNTHETIC_CENTER_Y_M,
                       "width_m": 5e-7, "height_m": 5e-7})
            assert v == [], v
        finally:
            ConfigOverrideRegistry.reset()


# ─────────────────────────────────────────────────────────────────────
# (11) a refused escalation must outlive the tool call
# ─────────────────────────────────────────────────────────────────────

class _FakeRunCtx:
    """Context whose .run(name, params) returns canned SkillResults."""

    def __init__(self, canned: dict):
        self.canned = canned
        self.runs: list = []

    def run(self, skill_name, params):
        self.runs.append((skill_name, params))
        r = self.canned.get(skill_name)
        if r is None:
            return SkillResult(skill_name=skill_name, success=False,
                               error="unmocked")
        return r


def _eng(engaged: bool, needs_auto: bool) -> SkillResult:
    return SkillResult(skill_name="TryEngageController", success=True,
                       data={"engaged": engaged, "needs_auto_approach": needs_auto,
                             "peak_current_a": 1e-9, "setpoint_a": 1e-9})


#: Synthetic refusal text with no copied measurement values.
SYNTHETIC_ENGAGE_ERROR = "未能建立隧穿；控制器状态不足以支持粗进针"


class TestApproachTipRecordsItsRefusal:
    def test_engage_failure_records_a_refusal(self):
        ctx = _FakeRunCtx({
            "TryEngageController": SkillResult(
                skill_name="TryEngageController", success=False,
                error=SYNTHETIC_ENGAGE_ERROR),
        })
        res = ApproachTip().execute(ctx, {})
        assert res.success is False
        r = esc.active_approach_refusal()
        assert r is not None
        assert "未能建立隧穿" in r.reason
        assert r.source == "ApproachTip"

    def test_ambiguous_outcome_records_a_refusal(self):
        # Not engaged AND not asking for an approach — "don't approach on a
        # maybe" is just as much a refusal as an outright engage failure.
        ctx = _FakeRunCtx({"TryEngageController": _eng(False, False)})
        res = ApproachTip().execute(ctx, {})
        assert res.success is False
        assert esc.active_approach_refusal() is not None

    def test_successful_engage_clears_any_refusal(self):
        esc.record_approach_refusal("stale")
        ctx = _FakeRunCtx({"TryEngageController": _eng(True, False)})
        res = ApproachTip().execute(ctx, {})
        assert res.success is True
        assert esc.active_approach_refusal() is None

    def test_escalating_clears_the_refusal(self):
        # Re-running the safe front door is the ESCAPE from the gate: once
        # ApproachTip itself decides a coarse approach is warranted, the earlier
        # refusal is superseded.
        esc.record_approach_refusal("stale")
        ctx = _FakeRunCtx({
            "TryEngageController": _eng(False, True),
            "AutoApproach": SkillResult(skill_name="AutoApproach", success=True,
                                        data={"ok": True}),
            "GetCurrent": SkillResult(skill_name="GetCurrent", success=True,
                                      data={"current_a": 5e-10}),
            "GetSetpoint": SkillResult(skill_name="GetSetpoint", success=True,
                                       data={"setpoint_a": 5e-10}),
        })
        res = ApproachTip().execute(ctx, {})
        assert res.success is True
        assert esc.active_approach_refusal() is None
        assert "AutoApproach" in [n for n, _ in ctx.runs]

    def test_refusal_expires(self):
        esc.record_approach_refusal("x", ttl_s=0.0)
        assert esc.active_approach_refusal() is None


class TestSideDoorIsGated:
    def setup_method(self):
        self.mw = SafetyGateMiddleware(SafetyLimits())

    def test_direct_autoapproach_blocked_while_refusal_is_live(self):
        """A direct call cannot bypass an active approach refusal."""
        esc.record_approach_refusal(SYNTHETIC_ENGAGE_ERROR)
        req = _make_request(_tool(AutoApproach), {"wait_timeout_s": 1800})
        handler = MagicMock()
        result = self.mw.wrap_tool_call(req, handler)
        handler.assert_not_called()
        assert result.status == "error"
        assert "approach_escalation_refused" in result.content
        # It must quote the refusal, not just say "blocked".
        assert "未能建立隧穿" in result.content
        # And it must not be a dead end.
        assert "ApproachTip" in result.content

    def test_autoapproach_runs_normally_with_no_refusal(self):
        req = _make_request(_tool(AutoApproach), {"wait_timeout_s": 1800})
        handler = MagicMock(return_value="ok")
        assert self.mw.wrap_tool_call(req, handler) == "ok"
        handler.assert_called_once()

    def test_autoapproach_runs_again_once_the_refusal_is_cleared(self):
        esc.record_approach_refusal(SYNTHETIC_ENGAGE_ERROR)
        esc.clear_approach_refusal("operator resolved it")
        req = _make_request(_tool(AutoApproach), {"wait_timeout_s": 1800})
        handler = MagicMock(return_value="ok")
        assert self.mw.wrap_tool_call(req, handler) == "ok"

    def test_the_safe_front_door_is_never_gated(self):
        # Blocking ApproachTip would remove the escape and wedge 进针 entirely.
        esc.record_approach_refusal(SYNTHETIC_ENGAGE_ERROR)
        req = _make_request(_tool(ApproachTip), {})
        handler = MagicMock(return_value="ok")
        assert self.mw.wrap_tool_call(req, handler) == "ok"

    def test_unrelated_skills_are_not_gated(self):
        esc.record_approach_refusal(SYNTHETIC_ENGAGE_ERROR)
        req = _make_request(_tool(SetBias), {"bias_v": 1.0})
        handler = MagicMock(return_value="ok")
        assert self.mw.wrap_tool_call(req, handler) == "ok"

    def test_autoapproach_stays_auto(self):
        # The gate is on the SEQUENCE. Re-gating the skill itself would re-break
        # 进针 for every ordinary run (v0.3.21: "确认框又弹不出来").
        from mast.core.types import SafetyLevel

        assert AutoApproach().metadata().safety_level is SafetyLevel.AUTO

    @pytest.mark.asyncio
    async def test_async_path_is_gated_too(self):
        esc.record_approach_refusal(SYNTHETIC_ENGAGE_ERROR)
        req = _make_request(_tool(AutoApproach), {"wait_timeout_s": 1800})

        async def handler(_r):
            raise AssertionError("handler must not run")

        result = await self.mw.awrap_tool_call(req, handler)
        assert "approach_escalation_refused" in result.content


# ─────────────────────────────────────────────────────────────────────
# (9) the tip_quality_drop gate must also STOP something
# ─────────────────────────────────────────────────────────────────────

# ⑷(2026-08-08)：这里原来是 ``_Capture`` —— 一个「``interrupt()`` 直接返回而不是
# 抛出」的替身，复刻 2026-07-27 审批通道死掉时的现场形状。打断链割掉之后没有
# interrupt 可替，连注入口（``interrupt_fn``）都从工厂函数上删掉了。


@pytest_asyncio.fixture
async def buffer(tmp_path):
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        yield buf
    finally:
        await buf.stop()


# ── ⑰(2026-08-08):这一节的**判定语义整个反过来了** ────────────────────────
#
# (9) 当时的结论是:「``interrupt()`` 只是通告,真正停住 run 的必须是工具闸门」
# —— 2026-07-27 五个 CRITICAL interrupt 全部发出而 run 照扫不误,就是因为只有通告
# 没有闸门。那个结论**在当时完全正确**,闸门也确实修好了它。
#
# 2026-08-05(⑬)先把「形态类判定」踢出打断链;2026-08-06(⑭)再把修针期间的瞬变
# 踢出去;2026-08-08(⑰)把**整条打断链**割掉——理由见下:
#
# 依据是两天实机:这条链路弹出的确认框**零真阳性**,全是操作瞬态;而同期真正拦下
# 危险的是拒绝型防护(本文件上半部分那些 SafetyGate 层,一条都没动)。
#
# 所以下面每一条测试都**保留并改写**,不是删除:
#
#   旧断言                                    → 新断言(同一件事的现在时)
#   ─────────────────────────────────────────────────────────────────────
#   闸门必须拦住下一次 SetBias                → 闸门不存在了,SetBias 照跑
#   approve 开、reject 关                     → 没有裁决可读,因为没有问题在问
#   非关键事件不得被误认成「人答过了」        → 事件照进 event_refs,不丢
#   解药类工具必须放行                        → 所有工具都放行(解药词汇搬去别处)
#
# 「都不弹」和「该弹的没弹」是两件事,而一条被删掉的测试和一条从没写过的测试长得
# 一模一样 —— 所以这一节留在原地,记着 2026-07-27 的事故和它的结论后来怎么了。
#
# **要把闸门加回来,需要观测到什么**:举出一次实例,这条链路的确认框拦下了一次真实
# 的损害,而且拒绝型防护接不住它。截至 2026-08-08:零例。


def _drop(seqno: int, *, critical: bool = True, morphology: bool = False
          ) -> VisionEvent:
    """一条 CRITICAL ``tip_quality_drop``。

    默认是**物理类**(``signal=current_saturation``:前放到轨,针被压进表面);
    ``morphology=True`` 给视觉边沿检测器那种形状。⑰ 之后两者都不打断,这个区分
    只影响通知怎么写 —— 但区分本身必须活着(见 ``classify_event``)。"""
    if morphology:
        payload = {"quality": "bad" if critical else "degraded",
                   "confidence": 0.9, "scan_id": "s", "frame_idx": seqno}
        cause = f"tip_status#{seqno}"
    else:
        payload = {"signal": "current_saturation", "source": "current_monitor",
                   "scan_id": "s", "summary_zh": "前放到轨"}
        cause = f"current_monitor#{seqno}"
    return VisionEvent(
        seqno=seqno,
        kind=VisionEventType.TIP_QUALITY_DROP,
        severity=Severity.CRITICAL if critical else Severity.WARN,
        payload=payload,
        cause_ref=cause,
    )


@pytest.mark.asyncio
async def test_a_vision_verdict_no_longer_stops_the_run(buffer):
    """(9) 里 2026-08-05 就已经反转掉的那一半(形态类)。

    事件照发、照落库、照进 ``event_refs``;它不再做的事是把实验挂起。"""
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        buffer.emit_event(_drop(1, morphology=True))
        await asyncio_sleep0()
        update = mw.before_model({}, MagicMock())
        assert update is not None and len(update["event_refs"]) == 1, (
            "…but it must still be on the record"
        )
        handler = MagicMock(return_value="ok")
        req = _make_request(_tool(SetBias), {"bias_v": 1.0})
        assert mw.wrap_tool_call(req, handler) == "ok"
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_the_next_scan_is_no_longer_refused_after_a_critical_drop(buffer):
    """**语义反转的正主。** 这条测试原名
    ``test_next_scan_is_refused_after_an_unresolved_critical_drop``,断言的是
    「关键事件未解决时 SetBias 必须被拒」—— 那正是 2026-07-27 事故的修复。

    ⑰ 之后它断言相反的事:连「前放到轨」这种**物理类** CRITICAL 都不再拦住下一次
    仪器动作。这不是判据放宽了,是这条链路整个不再承担「拦」的职责:针被压进表面
    要靠 ``current_monitor`` 自己的 run 级 halt 和撞针状态机停下来,而不是靠一个
    事后要求人按按钮的对话框。

    记录**照旧**,而且这里一并钉住 —— 否则「不拦」和「不记」会一起悄悄发生。"""
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        buffer.emit_event(_drop(1))
        await asyncio_sleep0()
        update = mw.before_model({}, MagicMock())
        assert update is not None and len(update["event_refs"]) == 1

        req = _make_request(_tool(SetBias), {"bias_v": 1.0})
        handler = MagicMock(return_value="ok")
        assert mw.wrap_tool_call(req, handler) == "ok"
        handler.assert_called_once()
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_every_tool_stays_open_not_just_the_remedies(buffer):
    """旧名 ``test_remedy_and_read_tools_stay_open``。

    当时的道理是「闸门若挡住它自己建议的动作就是死锁」,所以解药/读取类必须放行。
    现在**全部**放行,所以这条要连推进类一起断言 —— 只测解药会绿得毫无信息量。

    解药词汇本身没作废:``core/instrument_lock`` 的 ``BYPASS_NAMES``/``BYPASS_TAGS``
    还在用它,由 ``test_remedy_names_are_real_skills``(本文件下方)和
    ``test_buffer_hitl.test_remedy_vocabulary_matches_instrument_lock`` 钉着。"""
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        buffer.emit_event(_drop(1))
        await asyncio_sleep0()
        mw.before_model({}, MagicMock())

        for skill_cls in (WithdrawTip, GetBias, SetBias):
            handler = MagicMock(return_value="ok")
            req = _make_request(_tool(skill_cls), {})
            assert mw.wrap_tool_call(req, handler) == "ok", skill_cls.__name__
            handler.assert_called_once()

        # Non-skill tools (hand-off, buffer reads) carry no skill_metadata.
        from mast.agents._shared.handoff import make_handoff

        handler = MagicMock(return_value="handed_off")
        req = _make_request(make_handoff("data_processing", "test"),
                            {"reason": "tip bad"})
        assert mw.wrap_tool_call(req, handler) == "handed_off"
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_there_is_no_verdict_left_to_read(buffer):
    """2026-07-28 那条修复(approve/reject/超时三种结局必须不同)的**现在时**。

    那次的根因是「裁决被读丢了,三种答案殊途同归」。⑰ 之后没有问题在问 ⇒ 没有
    裁决可读 ⇒ 那个 bug 在结构上不可能重现。钉的是结构:模块里没有裁决解析函数,
    快照里的三个裁决相关字段恒为「无」。

    这条替换了原来的 ``test_gate_reopens_only_when_the_operator_APPROVED`` 与
    ``test_gate_stays_shut_when_the_operator_rejected``。"""
    from mast.agents._shared import buffer_hitl as bh

    assert not hasattr(bh, "_verdict_is_approval")
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        buffer.emit_event(_drop(1))
        await asyncio_sleep0()
        mw.before_model({}, MagicMock())
        st = mw.gate_state()
        assert st["awaiting"] == 0 and st["unresolved"] == 0
        assert st["degraded"] is False
        assert st["closed"] is False
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_a_later_event_does_not_lose_the_earlier_one(buffer):
    """旧名 ``test_a_later_non_critical_event_does_not_reopen_the_gate``。

    当时防的是「后来的非关键事件被误认成人答过了」。现在没有闸门可重开,但同一个
    序列还剩一半必须成立:两条事件的 id **都要**到达 ``event_refs``,后来的一条
    不能把先来的挤掉。"""
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        e1 = _drop(1)
        buffer.emit_event(e1)
        await asyncio_sleep0()
        u1 = mw.before_model({}, MagicMock())
        e2 = _drop(2, critical=False)
        buffer.emit_event(e2)
        await asyncio_sleep0()
        u2 = mw.before_model({}, MagicMock())

        assert u1["event_refs"] == [e1.event_id]
        assert u2["event_refs"] == [e2.event_id]

        handler = MagicMock(return_value="ok")
        req = _make_request(_tool(SetBias), {"bias_v": 1.0})
        assert mw.wrap_tool_call(req, handler) == "ok"
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_clean_run_never_blocks_anything(buffer):
    """⑰ 之后这条依然成立,而且**只剩下它这一种情形** —— 干净不干净都不拦。"""
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        mw.before_model({}, MagicMock())
        handler = MagicMock(return_value="ok")
        req = _make_request(_tool(SetBias), {"bias_v": 1.0})
        assert mw.wrap_tool_call(req, handler) == "ok"
    finally:
        mw.close()


@pytest.mark.asyncio
async def test_async_tool_path_is_not_gated_either(buffer):
    """旧名 ``test_async_tool_path_is_gated_too``。

    异步孪生必须和同步一致 —— 当年是「同步拦了异步没拦」的对称性要求,现在是
    「同步放行异步也放行」。生产路径走的正是这一条。"""
    mw = make_buffer_hitl_middleware(buffer=buffer)
    try:
        buffer.emit_event(_drop(1))
        await asyncio_sleep0()
        mw.before_model({}, MagicMock())

        called = []

        async def handler(_r):
            called.append(1)
            return "ok"

        req = _make_request(_tool(SetBias), {"bias_v": 1.0})
        assert await mw.awrap_tool_call(req, handler) == "ok"
        assert called
    finally:
        mw.close()


def test_remedy_names_are_real_skills():
    """A name list is only a gate if the names exist — a typo silently blocks
    the remedy it was written to protect."""
    from mast.agents.instrument_control.tools import discover_instrument_skills

    known = {m.name for m in discover_instrument_skills().list_skills()}
    missing = REMEDY_TOOL_NAMES - known
    assert not missing, f"REMEDY_TOOL_NAMES drifted: {sorted(missing)}"


async def asyncio_sleep0() -> None:
    """Let the buffer's ``call_soon_threadsafe`` fanout reach the queues."""
    import asyncio

    await asyncio.sleep(0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
