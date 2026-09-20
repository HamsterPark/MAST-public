"""Regression pins for the 2026-08-01 SAFE-mode audit findings.

Each test here corresponds to a hole the audit found AFTER the first
implementation pass. They are grouped in one file because they share a theme:
**every one of them was a way for SAFE mode to look right and behave wrong.**

The two that matter most:

  - `test_declarative_composite_inherits_pulse_capability` — SAFE's hard block is
    a pure capability lookup, and declarative workflows declared no capabilities
    at all. A workflow containing TipPulse was therefore let through in SAFE and
    would really have fired a pulse.
  - `test_current_monitor_halt_survives_mode_switch` — the mirror: SAFE must not
    become a way to switch off physical protection.
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

from mast.core.operating_mode import bind_mode_source  # noqa: E402


@pytest.fixture(autouse=True)
def _unbind_after():
    yield
    bind_mode_source(None)


def _safe():
    bind_mode_source(lambda: "safe")


# ── P0: declarative composites were invisible to the SAFE hard block ────────

@pytest.fixture(scope="module")
def registry():
    from mast.core.registry import SkillRegistry
    r = SkillRegistry()
    r.discover()
    return r


def _spec_skill(registry, step_skill: str, name: str):
    from mast.skills.composite.interpreter import make_spec_skill
    from mast.skills.composite.spec import CompositeSpec
    spec = CompositeSpec(
        name=name, description="d", safety_level="auto", params=[],
        nodes=[{"type": "step", "id": "s", "skill": step_skill, "params": {}}],
        outputs=[])
    return make_spec_skill(spec, registry=registry)().metadata()


@pytest.mark.parametrize("step,cap", [
    ("TipPulse", "bias_pulse"),
    ("BiasPulse", "bias_pulse"),
    ("ConditionTip", "bias_pulse"),
    ("TipShape", "tip_shaping"),
])
def test_declarative_composite_inherits_pulse_capability(registry, step, cap):
    """A declarative workflow wrapping a tip skill must carry that skill's
    capability tag, or SafetyGate cannot see the tip processing at all."""
    meta = _spec_skill(registry, step, f"AuditSpec_{step}")
    assert cap in meta.capabilities


def test_declarative_composite_is_blocked_by_safe_gate(registry):
    """The end of that chain: the SAFE gate's own predicate must now say yes."""
    from mast.core.safety import is_electrical_pulse

    meta = _spec_skill(registry, "TipPulse", "AuditSpec_GateCheck")
    assert is_electrical_pulse(meta.name, {}, meta.capabilities) is True


def test_non_tip_composite_gains_no_capability(registry):
    """Inheritance must not over-claim: a scan-only workflow stays unrestricted,
    otherwise SAFE would block ordinary scanning."""
    meta = _spec_skill(registry, "StartScan", "AuditSpec_ScanOnly")
    assert meta.capabilities == frozenset()


# ── the audit trail must not travel back into the model's context ───────────

def test_safe_mode_raw_is_stripped_from_tool_message():
    """`summary = str(data)` is the default for composite results, so an audit
    key sitting in `data` goes straight into the agent's ToolMessage — carrying
    "recommendation": "mild_conditioning" back to the model that SAFE just told
    not to think about repairs."""
    from mast.agents._shared.skill_adapter import _strip_audit_only

    data = {"tip_ready": True, "similarity": 0.42,
            "safe_mode_raw": {"tip_ready": False,
                              "recommendation": "mild_conditioning"}}
    stripped = _strip_audit_only(data)
    assert "safe_mode_raw" not in stripped
    assert "mild_conditioning" not in str(stripped)
    assert stripped["similarity"] == 0.42        # everything else untouched
    assert "safe_mode_raw" in data               # the record side keeps it


def test_strip_is_a_no_op_without_audit_keys():
    from mast.agents._shared.skill_adapter import _strip_audit_only
    d = {"a": 1}
    assert _strip_audit_only(d) is d             # same object — no copy churn


# ── the halt latch remembers WHO armed it ──────────────────────────────────
#
# ⑰-C1（2026-08-09）：下面四条钉的是**闩锁的记账**（`raise_tip_halt(source=…)` →
# `consume_tip_halt` 的 SAFE 判别），它们直接调闩锁，不经过 `make_tip_halt_hook`。
# 所以 C1 之后它们仍然全绿，但要知道**现实里已经没有生产者会传 `source="vision"`**
# 了：视觉判定在任何场景下都不再 arm 这个 halt（见
# `tests/v2/unit/core/test_forensics_20260727_tip_halt.py` 的 ⑰-C1 一节）。
#
# 这四条**不删**，因为它们守的是闩锁本身的第二层 fail-safe：任何其它调用方（现在
# 或将来）用 `source="vision"` armed 的 halt，在 SAFE 下仍然会被丢掉。一条被删掉的
# 测试和一条从没写过的测试长得一模一样。

class _App:
    def __init__(self):
        self.halts = []
        self._tip_halt = None
        self._orch_run_id = "r1"

    # borrow the real implementations
    from mast.core.runtime import CoreRuntime
    raise_tip_halt = CoreRuntime.raise_tip_halt
    consume_tip_halt = CoreRuntime.consume_tip_halt
    _tip_halt_remedy = CoreRuntime._tip_halt_remedy


def test_vision_halt_armed_before_switch_is_dropped_in_safe():
    """The operator switches to SAFE *because* the tip went bad — so a vision
    halt is typically already armed at that moment. Consuming it anyway would
    stop the experiment SAFE just promised to keep running."""
    app = _App()
    bind_mode_source(lambda: "auto")
    app.raise_tip_halt("视觉判定针尖状态恶化", run_id="r1", source="vision")

    _safe()
    assert app.consume_tip_halt("r1") == ""
    assert app._tip_halt is None                 # and it is cleared, not left to fire later


def test_current_monitor_halt_survives_mode_switch():
    """The mirror, and the line that must never move: a halt armed by the
    current monitor is physical safety (tip railed into the surface / dead
    measurement chain). SAFE does not switch that off."""
    app = _App()
    bind_mode_source(lambda: "auto")
    app.raise_tip_halt("隧道电流贴轨", run_id="r1", source="current_monitor")

    _safe()
    text = app.consume_tip_halt("r1")
    assert "隧道电流贴轨" in text


def test_vision_halt_still_consumed_outside_safe():
    app = _App()
    bind_mode_source(lambda: "auto")
    app.raise_tip_halt("视觉判定针尖状态恶化", run_id="r1", source="vision")
    assert "视觉判定针尖状态恶化" in app.consume_tip_halt("r1")


def test_halt_source_defaults_to_vision():
    """Callers that predate the `source` argument must not accidentally get the
    current-monitor exemption."""
    app = _App()
    app.raise_tip_halt("x", run_id="r1")
    assert app._tip_halt["source"] == "vision"


# ── persona injection: the one path with no belief-block cover ──────────────

def test_persona_tip_sections_change_in_safe():
    """`llm_node` injects these into composite-internal LLM decisions, which
    never pass through ModeBeliefMiddleware — so before this fix SAFE mode had
    exactly one place still telling the model "quality low → pulse the tip",
    with nothing anywhere to contradict it."""
    from mast.skills.composite.persona import render_sections

    bind_mode_source(lambda: "auto")
    normal = render_sections("instrument_control", ["tip_conditioning", "scanning"])
    assert "TipPulse" in normal and "修针" in normal

    _safe()
    safe_text = render_sections("instrument_control", ["tip_conditioning", "scanning"])
    assert "TipPulse" not in safe_text
    assert "不修针" in safe_text


def test_persona_safety_section_unchanged_in_safe():
    """Sections without a SAFE variant must render identically — the override is
    about tip repair, not about rewriting the safety envelope."""
    from mast.skills.composite.persona import render_sections

    bind_mode_source(lambda: "auto")
    normal = render_sections("instrument_control", ["safety"])
    _safe()
    assert render_sections("instrument_control", ["safety"]) == normal


# ── #43 的落点在 ⑰ 之后换了地方:从「拦截语」搬到「通知语」 ─────────────────
#
# 原来这里钉的是 ``BufferHITLMiddleware._gate_block()`` 渲染出来的**拦截文本**:
# SAFE 模式下它不能建议「修针」,因为 SafetyGate 在 SAFE 下会硬拦修针,而一支已经
# 撞进表面的针也不该去修。那条 bug 是现场 #43(「明明是 safe 模式,还是要修针尖」)。
#
# ⑰(2026-08-08)把整条打断链割掉之后,``_gate_block`` 和它的文本都不存在了 ——
# 没有工具会被拦,自然没有拦截语。**但 #43 那条要求本身没有过期**:同一个
# ``_suggest_action`` 现在给**通知**用,而一条自相矛盾的通知照样会指挥模型去做一件
# 系统自己不肯做的事。
#
# 所以这两条测试改钉那个函数本身,而不是钉一段已经不存在的文本。这样写还有一个好
# 处:原来那个 helper 要手工半构造一个中间件实例(``__new__`` + 手填 ``_unresolved``),
# 它已经因为「闸门长出了新状态」坏过一次;现在钉的是纯函数,没有可以半构造的东西。


def test_the_safe_mode_suggestion_is_not_tip_repair():
    """SAFE 下建议动作必须是「人去看一眼」,不是「去修针」。"""
    from mast.agents._shared.buffer_hitl import _suggest_action
    from mast.buffer.schemas import VisionEventType

    _safe()
    assert _suggest_action(VisionEventType.TIP_QUALITY_DROP,
                           safe_mode=True) == "manual_tip_check"


def test_the_suggestion_outside_safe_is_unchanged():
    """非 SAFE 模式的建议一个字没动 —— #43 改的只有 SAFE 那一个答案。"""
    from mast.agents._shared.buffer_hitl import _suggest_action
    from mast.buffer.schemas import VisionEventType

    bind_mode_source(lambda: "auto")
    assert _suggest_action(VisionEventType.TIP_QUALITY_DROP,
                           safe_mode=False) == "tip_prep"
    # 停止与退针在任何模式下都允许,所以这两个答案与模式无关。
    assert _suggest_action(VisionEventType.E_STOP, safe_mode=True) == "halt"
    assert _suggest_action(VisionEventType.EMERGENCY_RETRACT_NEEDED,
                           safe_mode=True) == "retract"


def test_the_gate_block_renderer_is_really_gone():
    """反向守卫:别让「拦截语」以任何形式回来。

    上面两条只证明建议语对;如果哪天有人把 ``_gate_block`` 接回去,它们照样绿。
    ⑰ 的承诺是「没有任何工具会被拦」,那就得有一条断言直接对着这件事。"""
    from mast.agents._shared.buffer_hitl import BufferHITLMiddleware

    assert not hasattr(BufferHITLMiddleware, "_gate_block"), (
        "buffer_hitl 的工具闸门又回来了")


# ── current-monitor CRITICAL keeps its halt but drops the repair advice ─────

def test_current_monitor_recommend_drops_repair_in_safe(monkeypatch):
    """The alert must still fire (physical safety) but must not ask for an
    action the gate will then refuse — that combination is ."""
    from mast.buffer import active as buf_active
    from mast.monitoring import alerts

    captured = []

    class _Buf:
        def next_seq(self):
            return 1

        def emit_event(self, ev):
            captured.append(ev)

    monkeypatch.setattr(buf_active, "get_active_buffer", lambda: _Buf())

    _safe()
    assert alerts.emit_critical("saturation", "隧道电流贴轨", {}, None, 1) is True
    assert captured, "the CRITICAL itself must still be published in SAFE"
    assert captured[0].payload["recommend"] == ["StopScan"]
    assert captured[0].payload["source"] == "current_monitor"

    captured.clear()
    bind_mode_source(lambda: "auto")
    alerts.emit_critical("saturation", "隧道电流贴轨", {}, None, 1)
    assert captured[0].payload["recommend"] == ["StopScan", "ConditionTip"]
