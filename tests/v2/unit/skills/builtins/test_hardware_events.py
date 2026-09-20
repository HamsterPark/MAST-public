"""ReadHardwareEvents 返回事件本身与可核对字段，不能用其他状态表代替事件证据。"""
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

import pytest  # noqa: E402

from mast.core.types import SafetyLevel, SkillCategory  # noqa: E402
from mast.skills.builtins.hardware_events import ReadHardwareEvents  # noqa: E402


def _ev(seqno=7, kind="tip_quality_drop", severity="critical",
        cause_ref="current_monitor#132024", **payload):
    """一条 current_monitor 发出来的 CRITICAL,payload 形状照 alerts.emit_critical。"""
    from mast.buffer.schemas import Severity, VisionEvent, VisionEventType
    base = {
        "signal": "current_saturation",
        "scan_id": "",
        "summary_zh": "隧道电流持续贴轨饱和(段内 50% 的样本达到满量程)——疑似撞针。",
        "network_free": True,
        "frame_path": "/tmp/ev.png",
        "features": {"sat_frac": 0.5, "rms_detrended_a": 3.2e-12,
                     "spike_max_sigma": 41.0},
        "recommend": ["StopScan", "ConditionTip"],
        "source": "current_monitor",
    }
    base.update(payload)
    return VisionEvent(seqno=seqno, kind=VisionEventType(kind),
                       severity=Severity(severity), payload=base,
                       cause_ref=cause_ref)


class _Buf:
    def __init__(self, events):
        self._events = events

    def get_event_history(self, since_seqno=-1, limit=100):
        return [e for e in self._events if e.seqno > since_seqno][-limit:]


def _run(monkeypatch, buf, params=None, gates=None):
    monkeypatch.setattr("mast.buffer.active.get_active_buffer", lambda: buf)
    if gates is not None:
        monkeypatch.setattr("mast.agents._shared.buffer_hitl.gate_states",
                            lambda: gates)
    return ReadHardwareEvents().execute(None, params or {})


# ── 形状 ────────────────────────────────────────────────────────────────


def test_it_is_an_auto_read_skill():
    """只读、AUTO —— 一个「给已经被拦住的人用」的工具本身绝不能需要审批。"""
    m = ReadHardwareEvents().metadata()
    assert m.safety_level == SafetyLevel.AUTO
    assert m.category == SkillCategory.READ


def test_the_description_points_at_the_blocked_case():
    """技能描述得让被拦的 agent 知道该调它 —— 否则工具在列表里也等于不存在。"""
    d = ReadHardwareEvents().metadata().description
    assert "buffer_hitl" in d
    assert "seqno=-1" in d, "必须点名那个「看起来一切正常」的陷阱"


# ── 核心:证据到得了手上 ────────────────────────────────────────────────


def test_a_current_monitor_critical_is_visible_with_its_numbers(monkeypatch):
    """此前 agent 看不见的那条,现在要看得见 —— **而且带着指标**。
    只给段号还是失明:`current_monitor#132024` 本身不可核实。"""
    res = _run(monkeypatch, _Buf([_ev()]))
    assert res.success
    ev = res.data["recent"]["events"][0]
    assert ev["kind"] == "tip_quality_drop"
    assert ev["severity"] == "critical"
    assert ev["source"] == "current_monitor"
    assert ev["cause_ref"] == "current_monitor#132024"
    assert ev["features"]["sat_frac"] == 0.5
    assert ev["features"]["spike_max_sigma"] == 41.0
    assert "贴轨" in ev["summary_zh"]
    assert ev["recommend"] == ["StopScan", "ConditionTip"]


def test_metrics_survive_without_the_monitoring_store(monkeypatch):
    """指标优先取自事件 payload,**不依赖监控数据库** —— 库没起时仍然可核实。"""
    monkeypatch.setattr("mast.monitoring.store.get_store",
                        lambda: (_ for _ in ()).throw(RuntimeError("no store")))
    res = _run(monkeypatch, _Buf([_ev()]))
    ev = res.data["recent"]["events"][0]
    assert ev["features"]["sat_frac"] == 0.5
    assert "cause_detail" not in ev          # 补充查不到,主答案不受影响


def test_the_summary_names_the_worst_event(monkeypatch):
    """一行摘要要说出最严重的那条 —— agent 常常只读 summary。"""
    res = _run(monkeypatch, _Buf([
        _ev(seqno=1, severity="info", kind="scan_complete", cause_ref=None),
        _ev(seqno=2, severity="critical"),
    ]))
    assert "critical" in res.summary and "tip_quality_drop" in res.summary


# ── 「没有」与「读不到」是两句话 ────────────────────────────────────────


def test_no_events_is_not_the_same_as_cannot_read(monkeypatch):
    """这条正是本 skill 存在的理由:`seqno=-1` 当初就被读成了「一切正常」。"""
    empty = _run(monkeypatch, _Buf([]))
    assert empty.data["recent"]["available"] is True
    assert empty.data["n_events"] == 0
    assert "没有事件" in empty.summary

    blind = _run(monkeypatch, None)
    assert blind.data["recent"]["available"] is False
    assert blind.data["recent"]["why"]
    assert "读不到" in blind.summary
    # ⚠️ 断言的是「没有下那个结论」,不是「没出现这四个字」——
    # 读不到时的文案里**故意**写着「这不代表没有事件」,那正是要的话。
    # 用裸子串判会把一句正确的澄清判成错误。
    assert "缓冲区里没有事件" not in blind.summary, "读不到被说成了没有事件"
    assert empty.summary != blind.summary


def test_a_broken_buffer_does_not_break_the_tool(monkeypatch):
    """给「已经出故障时」用的工具,自己不许炸。"""
    class Exploding:
        def get_event_history(self, **kw):
            raise RuntimeError("boom")

    res = _run(monkeypatch, Exploding())
    assert res.success is True
    assert res.data["recent"]["available"] is False
    assert "boom" in res.data["recent"]["why"]


def test_a_broken_gate_reader_does_not_break_the_events(monkeypatch):
    """两块独立:闸门读不到不该把事件那块也拖下水。"""
    def boom():
        raise RuntimeError("gate down")

    monkeypatch.setattr("mast.agents._shared.buffer_hitl.gate_states", boom)
    res = _run(monkeypatch, _Buf([_ev()]))
    assert res.data["blocking"]["available"] is False
    assert res.data["recent"]["available"] is True
    assert res.data["recent"]["events"]


# ── 拦截侧 ──────────────────────────────────────────────────────────────


def test_it_reports_who_is_blocking(monkeypatch):
    res = _run(monkeypatch, _Buf([_ev()]),
               gates=[{"closed": True, "unresolved": ["tip_quality_drop"],
                       "reask_armed": False}])
    assert res.data["blocking"]["any_blocking"] is True


def test_no_gate_closed_reports_not_blocking(monkeypatch):
    """v6.2 屏蔽之后大多数时候会走这条 —— 而 `recent` 那块**照常有内容**,
    那正是它在屏蔽后的价值:自主判断的读口,不是拦截流程的补丁。"""
    res = _run(monkeypatch, _Buf([_ev(severity="warn")]),
               gates=[{"closed": False, "unresolved": [], "reask_armed": False}])
    assert res.data["blocking"]["any_blocking"] is False
    assert res.data["recent"]["events"], "屏蔽后事件读口不该跟着空掉"


# ── 过滤与上限 ──────────────────────────────────────────────────────────


def test_min_severity_filters(monkeypatch):
    evs = [_ev(seqno=1, severity="info", kind="scan_complete", cause_ref=None),
           _ev(seqno=2, severity="warn"),
           _ev(seqno=3, severity="critical")]
    res = _run(monkeypatch, _Buf(evs), {"min_severity": "warn"})
    got = [e["severity"] for e in res.data["recent"]["events"]]
    assert got == ["warn", "critical"]


def test_limit_is_honoured(monkeypatch):
    evs = [_ev(seqno=i, severity="warn") for i in range(1, 21)]
    res = _run(monkeypatch, _Buf(evs), {"limit": 3})
    assert len(res.data["recent"]["events"]) == 3


def test_it_never_touches_hardware(monkeypatch):
    """纯读。context 传 None 也必须跑得完 —— 它一次 safe_call 都不该发。"""
    res = ReadHardwareEvents().execute(None, {})
    assert res.success
    assert not res.nanonis_calls


# ── 可达性:注册表能发现、agent 工具列表里能出现 ──────────────────────
#
# 「工具存在」和「被拦的 agent 能调到它」是两件事。今晚反复出现的形状就是
# 「原语能用,但没有任何东西断言有人够得着它」—— 所以这两条钉的是**可达性**。


def test_the_registry_discovers_it():
    from mast.core.registry import SkillRegistry
    r = SkillRegistry()
    r.discover()
    cls = r.get("ReadHardwareEvents")
    assert cls is ReadHardwareEvents


def test_it_becomes_a_usable_agent_tool():
    """IC agent 把注册表里每个 skill 都 wrap 成工具(tools.py:63),
    所以只要 wrap 得动、参数进得了 schema,它就在被拦 agent 的工具列表里。"""
    from mast.agents._shared.skill_adapter import wrap_skill
    t = wrap_skill(ReadHardwareEvents, lambda: None)
    assert t.name == "ReadHardwareEvents"
    fields = set(t.args_schema.model_fields.keys())
    assert {"limit", "min_severity"} <= fields
    # 描述会进 LLM 的工具清单 —— 被拦时该调它这件事必须写在里面。
    assert "buffer_hitl" in (t.description or "")
