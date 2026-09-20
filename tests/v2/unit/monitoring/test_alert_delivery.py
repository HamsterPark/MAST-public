"""告警应在下一次模型调用中送达，并遵循优先级、抑制和确认语义。
以独立构造的混合告警序列验证 critical 事件不会被重复低优先级事件淹没。"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
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

import sqlite3
import time

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from mast.agents._shared.alert_delivery_mw import (
    VISION_ID_KEY, AlertDeliveryMiddleware, format_alert_block, select_alerts,
    vision_rows,
)
from mast.monitoring.alert_routing import (
    DEFAULT_MUTED_RULES, AlertRouting, DeliveryClass, classify,
)
from mast.monitoring.store import CurrentMonitorStore

# 独立构造不同严重度的告警文本，验证通知筛选与送达。

SAT_ZH = ("隧道电流持续贴轨饱和(段内 100% 的样本达到满量程)"
          "——疑似撞针或前置放大器过载,建议停止扫描并检查针尖。")
RTN_ZH = "电流在两个电平之间来回跳变(RTN 判分 0.80,速率 8.0 Hz,间距 4000.0 pA)——典型的针尖顶端不稳定。"
RMS_ZH = "电流噪声偏高(去趋势 RMS 500.0 pA)——针尖或环境可能不稳。"
HUM_ZH = "工频干扰明显(50 Hz 峰高出邻频 12 倍)——检查接地与屏蔽。"


@pytest.fixture()
def store(tmp_path):
    s = CurrentMonitorStore(tmp_path / "monitor.sqlite", tmp_path)
    yield s
    s.close()


def _incident(store: CurrentMonitorStore, t0: float) -> dict[str, int]:
    """按独立时间表写入不同严重度的合成告警，返回规则到告警 ID 的映射。"""
    ids = {}
    # 工频背景与高严重度告警交错，用于检验筛选。
    for k in range(8):
        store.add_alert(ts=t0 - 1200 + k * 180, level="warn", rule="line_hum",
                        summary_zh=HUM_ZH)
    ids["rms_high"] = store.add_alert(ts=t0 + 5, level="warn", rule="rms_high",
                                      summary_zh=RMS_ZH)
    ids["rtn_bistable"] = store.add_alert(ts=t0 + 10, level="warn",
                                          rule="rtn_bistable", summary_zh=RTN_ZH)
    ids["saturation"] = store.add_alert(ts=t0 + 15, level="critical",
                                        rule="saturation", summary_zh=SAT_ZH,
                                        emitted_buffer=True)
    return ids


class _Req:
    """最小的 ModelRequest 替身 —— 只需要 messages / system_message 两个属性。"""

    def __init__(self, messages=None, system_message=None):
        self.messages = list(messages or [])
        self.system_message = system_message


# ══════════════════════════════════════════════════════════════════════════
# 1. 事故回放 —— 那天缺的到底是什么
# ══════════════════════════════════════════════════════════════════════════

def test_critical_saturation_reaches_the_model(store):
    """**这条是本组的理由。** CRITICAL 必须出现在送进模型的那段文本里。"""
    t0 = time.time() - 60
    _incident(store, t0)
    mw = AlertDeliveryMiddleware(get_store=lambda: store)

    seen: dict = {}

    def handler(req):
        seen["req"] = req
        return "ok"

    req = _Req([HumanMessage(content="继续扫描第 3 个区域")])
    mw.wrap_model_call(req, handler)

    text = seen["req"].messages[-1].content
    assert "CRITICAL" in text
    assert "saturation" in text
    assert "建议停止扫描并检查针尖" in text


def test_line_hum_does_not_drown_the_critical(store):
    """8 条工频干扰不得出现;而三级恶化的前两级必须出现。"""
    t0 = time.time() - 60
    _incident(store, t0)
    rows = store.undelivered_alerts(time.time() - 3600, limit=50)
    block = format_alert_block(select_alerts(rows, AlertRouting()))

    assert "line_hum" not in block
    assert "检查接地与屏蔽" not in block
    # 恶化的第一级和第二级是 agent 该看到的时间线,不是噪声。
    assert "rms_high" in block
    assert "rtn_bistable" in block
    assert "saturation" in block
    # 静音掉的那些要如实报个数,而不是假装不存在。
    assert "8 条环境类告警" in block


def test_delivered_once_not_re_injected(store):
    """同一条 CRITICAL 不会每一轮都重新注入(确认语义)。"""
    t0 = time.time() - 60
    _incident(store, t0)
    mw = AlertDeliveryMiddleware(get_store=lambda: store)

    texts: list[str] = []

    def handler(req):
        texts.append(req.messages[-1].content)
        return "ok"

    for _ in range(2):
        mw.wrap_model_call(_Req([HumanMessage(content="下一步")]), handler)

    assert "saturation" in texts[0]
    assert "saturation" not in texts[1], "第二轮不该重复注入同一条告警"


# ══════════════════════════════════════════════════════════════════════════
# 2. 优先级与抑制
# ══════════════════════════════════════════════════════════════════════════

def test_critical_can_never_be_muted_by_config(store):
    """把 saturation 写进静音表也静不掉 —— 必达由行的 level 决定,不由名字。

    这是「硬编码清单型陷阱」的反面:不是靠校验名字防住,而是靠那条分支到不了。
    """
    routing = AlertRouting(muted_rules=frozenset({"saturation", "line_hum"}))
    assert classify("saturation", "critical", routing) is DeliveryClass.ALWAYS
    assert classify("line_hum", "warn", routing) is DeliveryClass.MUTE

    t0 = time.time() - 60
    _incident(store, t0)
    rows = store.undelivered_alerts(time.time() - 3600, limit=50)
    block = format_alert_block(select_alerts(rows, routing))
    assert "saturation" in block and "CRITICAL" in block


def test_unclassified_rule_defaults_to_delivered():
    """明天新加的规则,没人分类时默认**送达**(吵),不是静音(哑)。

    极性与 ``alerts.py`` 对 ctx_scanning / ctx_lockin_on 的论证同向:
    噪声可以事后收敛,漏报不能事后补。
    """
    assert classify("a_brand_new_rule_nobody_classified", "warn",
                    AlertRouting()) is DeliveryClass.FOLD


def test_default_muted_rule_is_a_real_rule_name():
    """出厂静音的名字必须是真规则 —— 写错一个字母就等于什么都没静音。

    这不是一份需要同步的注册表(新规则不必登记),只是钉住**我们自己写下的
    那个默认值**不是错别字。``lookup(name) || DEFAULT`` 型的静默失败正是这样来的。
    """
    from mast.monitoring.alerts import CRIT_RULES, WARN_RULES

    known = set(WARN_RULES) | set(CRIT_RULES)
    assert DEFAULT_MUTED_RULES <= known, (
        f"静音表里有不存在的规则名:{DEFAULT_MUTED_RULES - known}")


def test_warns_fold_with_a_count(store):
    """同一 rule 在窗口内折叠成一行 + 计数,保留最新那条的正文。"""
    now = time.time()
    for k in range(7):
        store.add_alert(ts=now - 300 + k * 10, level="warn", rule="rms_high",
                        summary_zh=f"第 {k} 条")
    rows = store.undelivered_alerts(now - 3600, limit=50)
    sel = select_alerts(rows, AlertRouting())

    assert len(sel["warn"]) == 1
    assert sel["warn"][0]["count"] == 7
    block = format_alert_block(sel)
    assert "×7" in block
    assert "第 6 条" in block, "折叠应保留最新一条的正文"


def test_no_row_is_looked_at_and_then_silently_dropped(store):
    """回看窗口内的每一条非静音行,要么被显示、要么被计数 —— 而且都要被标记。

    曾经有两个窗口(回看 900 s + 折叠 600 s),中间那条缝里的行既不显示、也不计数、
    也不标记,每一轮被重新扫出来再重新扔掉。「看了一眼,什么都没决定,然后悄悄
    扔掉」是本仓反复出事的形状,所以这里把「没有缝」钉住。
    """
    now = time.time()
    routing = AlertRouting()
    # 铺满整个回看窗口,同一个 rule —— 正是会落进旧缝里的那种分布。
    n = 10
    for k in range(n):
        store.add_alert(ts=now - (routing.lookback_s - 30) * k / (n - 1),
                        level="warn", rule="rms_high", summary_zh=f"第 {k} 条")

    mw = AlertDeliveryMiddleware(get_store=lambda: store,
                                 get_routing=lambda: routing)
    mw.wrap_model_call(_Req([HumanMessage(content="x")]), lambda r: "ok")

    left = [r for r in store.undelivered_alerts(0.0, limit=100)
            if r["rule"] == "rms_high"]
    assert left == [], f"{len(left)} 条被看过却既没显示也没标记"


def test_string_muted_rules_is_ignored_not_iterated():
    """``muted_rules="line_hum"`` 不得被当成字符集合(会静音掉 l/i/n/e...)。"""
    r = AlertRouting.from_mapping({"muted_rules": "line_hum"})
    assert r.muted_rules == DEFAULT_MUTED_RULES


# ══════════════════════════════════════════════════════════════════════════
# 3. 确认语义 —— acked 与 delivered_agent 是两个主体
# ══════════════════════════════════════════════════════════════════════════

def test_ack_and_delivery_are_independent_subjects(store):
    """人点掉 ≠ agent 看过。两个字段互不影响。"""
    aid = store.add_alert(ts=time.time(), level="critical", rule="saturation",
                          summary_zh=SAT_ZH)

    store.ack_alert(aid)
    rows, _ = store.alerts_query(limit=10)
    row = next(r for r in rows if r["id"] == aid)
    assert row["acked"] == 1
    assert row["delivered_agent"] == 0, "人点掉不该顺手替 agent 确认"

    # 人点掉的告警仍然要送给 agent。
    assert any(r["id"] == aid for r in store.undelivered_alerts(0.0, limit=50))

    store.mark_alerts_delivered([aid])
    rows, _ = store.alerts_query(limit=10)
    row = next(r for r in rows if r["id"] == aid)
    assert row["delivered_agent"] == 1
    assert row["acked"] == 1


def test_muted_rows_are_not_marked_delivered(store):
    """静音行不标已送达 —— agent 确实没看见它们。"""
    now = time.time()
    store.add_alert(ts=now, level="warn", rule="line_hum", summary_zh=HUM_ZH)
    store.add_alert(ts=now, level="critical", rule="saturation", summary_zh=SAT_ZH)

    mw = AlertDeliveryMiddleware(get_store=lambda: store)
    mw.wrap_model_call(_Req([HumanMessage(content="x")]), lambda r: "ok")

    rows, _ = store.alerts_query(limit=10)
    by_rule = {r["rule"]: r for r in rows}
    assert by_rule["saturation"]["delivered_agent"] == 1
    assert by_rule["line_hum"]["delivered_agent"] == 0


def test_model_call_failure_leaves_the_alert_undelivered(store):
    """模型调用抛异常时不得标记已送达 —— 重试时告警必须还在。"""
    store.add_alert(ts=time.time(), level="critical", rule="saturation",
                    summary_zh=SAT_ZH)
    mw = AlertDeliveryMiddleware(get_store=lambda: store)

    def boom(req):
        raise RuntimeError("provider 500")

    with pytest.raises(RuntimeError):
        mw.wrap_model_call(_Req([HumanMessage(content="x")]), boom)

    left = store.undelivered_alerts(0.0, limit=50)
    assert any(r["rule"] == "saturation" for r in left), \
        "调用失败后告警被吞了 —— 重试时就再也看不到它"


# ══════════════════════════════════════════════════════════════════════════
# 4. 不打扰、不阻塞、不污染 state
# ══════════════════════════════════════════════════════════════════════════

def test_no_alerts_means_no_injection(store):
    """没有告警时一个字都不加(不能给每一轮都加噪声)。"""
    mw = AlertDeliveryMiddleware(get_store=lambda: store)
    original = HumanMessage(content="继续")
    seen: dict = {}
    mw.wrap_model_call(_Req([original]), lambda r: seen.setdefault("r", r))
    assert seen["r"].messages[-1].content == "继续"


def test_only_muted_alerts_means_no_injection(store):
    """只有静音项时也不注入 —— 那一行会每轮重复,变成它自己要防的噪声。"""
    now = time.time()
    for k in range(5):
        store.add_alert(ts=now - k * 10, level="warn", rule="line_hum",
                        summary_zh=HUM_ZH)
    rows = store.undelivered_alerts(now - 3600, limit=50)
    assert format_alert_block(select_alerts(rows, AlertRouting())) == ""


def test_original_message_is_not_mutated(store):
    """块挂在**副本**上 —— 原消息是被 checkpoint 的那一份,不能改。"""
    store.add_alert(ts=time.time(), level="critical", rule="saturation",
                    summary_zh=SAT_ZH)
    mw = AlertDeliveryMiddleware(get_store=lambda: store)
    original = HumanMessage(content="继续")
    mw.wrap_model_call(_Req([original]), lambda r: "ok")
    assert original.content == "继续", "原消息被就地改写了,会污染 checkpoint"


def test_alerts_older_than_the_lookback_are_not_injected(store):
    """回看窗口之外的历史不刷屏。"""
    store.add_alert(ts=time.time() - 7200, level="critical", rule="saturation",
                    summary_zh=SAT_ZH)
    mw = AlertDeliveryMiddleware(get_store=lambda: store)
    seen: dict = {}
    mw.wrap_model_call(_Req([HumanMessage(content="继续")]),
                       lambda r: seen.setdefault("r", r))
    assert seen["r"].messages[-1].content == "继续"


def test_falls_back_to_system_message_without_a_human_turn(store):
    """没有 human 消息可挂时退回 system 消息 —— 数字的正确性高于 cache 命中。"""
    store.add_alert(ts=time.time(), level="critical", rule="saturation",
                    summary_zh=SAT_ZH)
    mw = AlertDeliveryMiddleware(get_store=lambda: store)
    seen: dict = {}
    req = _Req([], system_message=SystemMessage(content="你是仪器控制 agent"))
    mw.wrap_model_call(req, lambda r: seen.setdefault("r", r))
    assert "saturation" in seen["r"].system_message.content


def test_the_middleware_never_creates_the_monitoring_db(monkeypatch):
    """投递中间件是**只读**的:默认路径不许调会建库的 ``get_store()``。

    ``get_store()`` 会在 ``project_root()/experiments/current_monitor/`` 下建库
    建目录。对一个只读中间件,空库里没有告警 —— 创建什么都没买到;而在测试里
    它会写进用户真实的 experiments 目录,且 conftest 对这个库**没有**
    autouse 守卫(wishlist / 文献注册表都有)。本仓「测试污染真实数据」已四次。
    """
    import mast.monitoring.store as S

    created = {"n": 0}

    def _boom_get_store():
        created["n"] += 1
        raise AssertionError("投递中间件调用了会建库的 get_store()")

    monkeypatch.setattr(S, "get_store", _boom_get_store, raising=True)
    monkeypatch.setattr(S, "get_store_if_exists", lambda: None, raising=True)

    # get_store=None ⇒ 走默认路径,正是生产上 graph.py 构造它的方式。
    mw = AlertDeliveryMiddleware()
    seen: dict = {}
    mw.wrap_model_call(_Req([HumanMessage(content="继续")]),
                       lambda r: seen.setdefault("r", r))
    assert created["n"] == 0
    assert seen["r"].messages[-1].content == "继续"


def test_broken_store_is_a_noop_not_a_crash():
    """库读不到时这一轮不注入,而不是弄崩对话。"""
    class Boom:
        def undelivered_alerts(self, *a, **k):
            raise RuntimeError("db gone")

    mw = AlertDeliveryMiddleware(get_store=lambda: Boom())
    seen: dict = {}
    mw.wrap_model_call(_Req([HumanMessage(content="继续")]),
                       lambda r: seen.setdefault("r", r))
    assert seen["r"].messages[-1].content == "继续"

# 监控规则与视觉事件是独立通知通道；任一通道失败时另一条仍须正常送达。

VISION_ZH = ("扫描中途针尖状态突变(已采集 80 行中第 ~40 行,校准 z=50/阈值 30)"
             "——50% 处,其后行不可信,建议中止扫描并修针尖")


class _Ev:
    """最小的 VisionEvent 替身。"""

    def __init__(self, *, event_id, severity="critical", payload=None,
                 kind="tip_quality_drop", age_s=5.0):
        self.event_id = event_id
        self.severity = severity
        self.kind = kind
        self.payload = dict(payload or {})
        self.t_mono_ns = time.monotonic_ns() - int(age_s * 1e9)


class _Buf:
    def __init__(self, events):
        self._events = list(events)

    def get_event_history(self, since_seqno=-1, limit=50):
        return self._events[-limit:]


def _vision_ev(**kw):
    return _Ev(payload={"signal": "tip_quality_drop",
                        "summary_zh": VISION_ZH,
                        "source": "vision_scan_monitor"}, **kw)


def test_the_vision_pipe_reaches_the_agent_too(store):
    """视觉 CRITICAL 不在告警表里 —— 只读表会整条漏掉它。"""
    buf = _Buf([_vision_ev(event_id="v1")])
    mw = AlertDeliveryMiddleware(get_store=lambda: store,
                                 get_buffer=lambda: buf)
    seen: dict = {}
    mw.wrap_model_call(_Req([HumanMessage(content="继续")]),
                       lambda r: seen.setdefault("r", r))
    text = seen["r"].messages[-1].content
    assert "扫描中途针尖状态突变" in text
    assert "视觉·扫描中途" in text, "来源没印出来 —— agent 收到一句不知道谁说的话"


def test_both_pipes_appear_together_and_are_told_apart(store):
    """两条同时报时,必须能分开看 —— 处方与可信度不同,不是互相印证。"""
    t0 = time.time() - 30
    store.add_alert(ts=t0, level="critical", rule="saturation",
                    summary_zh=SAT_ZH, emitted_buffer=True)
    rows = store.undelivered_alerts(time.time() - 3600, limit=50)
    rows += vision_rows(_Buf([_vision_ev(event_id="v1", age_s=20.0)]),
                        AlertRouting())
    rows.sort(key=lambda r: float(r.get("ts") or 0.0), reverse=True)
    block = format_alert_block(select_alerts(rows, AlertRouting()))

    assert "电流监控" in block and "视觉·扫描中途" in block
    assert "来自不同的判定方" in block


def test_a_current_monitor_event_is_not_delivered_twice(store):
    """电流监控的 CRITICAL 同时在表里和缓冲区里 —— 只能出现一次。"""
    store.add_alert(ts=time.time() - 10, level="critical", rule="saturation",
                    summary_zh=SAT_ZH, emitted_buffer=True)
    dup = _Ev(event_id="cm1",
              payload={"signal": "current_saturation", "summary_zh": SAT_ZH,
                       "source": "current_monitor"})
    rows = store.undelivered_alerts(time.time() - 3600, limit=50)
    rows += vision_rows(_Buf([dup]), AlertRouting())
    block = format_alert_block(select_alerts(rows, AlertRouting()))
    assert block.count("隧道电流持续贴轨饱和") == 1


def test_an_unlabelled_event_is_treated_as_vision_not_dropped():
    """读不到 ``source`` 一律当视觉 —— 与 ``runtime.tip_halt_source`` 同向:
    宁可多送一条,不可漏一条。"""
    ev = _Ev(event_id="old1", payload={"summary_zh": "老事件,没有 source"})
    got = vision_rows(_Buf([ev]), AlertRouting())
    assert len(got) == 1
    assert got[0]["source"] == "vision"
    assert got[0][VISION_ID_KEY] == "old1"


def test_a_vision_event_is_not_re_injected(store):
    """视觉事件不在表里,没有 delivered_agent 可标 —— 进程内记忆要顶上。"""
    buf = _Buf([_vision_ev(event_id="v1")])
    mw = AlertDeliveryMiddleware(get_store=lambda: store,
                                 get_buffer=lambda: buf)
    texts: list[str] = []
    for _ in range(2):
        mw.wrap_model_call(_Req([HumanMessage(content="下一步")]),
                           lambda r: texts.append(r.messages[-1].content))
    assert "扫描中途针尖状态突变" in texts[0]
    assert "扫描中途针尖状态突变" not in texts[1]


def test_vision_warns_do_not_come_down_this_pipe():
    """这条路只送 CRITICAL —— warn 那一半是告警表的职责。"""
    ev = _Ev(event_id="w1", severity="warn",
             payload={"summary_zh": "轻微变化", "source": "vision_scan_monitor"})
    assert vision_rows(_Buf([ev]), AlertRouting()) == []


def test_an_old_vision_event_is_outside_the_lookback():
    assert vision_rows(_Buf([_vision_ev(event_id="old", age_s=99999.0)]),
                       AlertRouting()) == []


def test_the_vision_producers_label_their_own_source():
    """事件 source 应由产出方记录，读取方无需从 kind 猜测具体判定来源。"""
    # ① 扫描中途的形态判定。
    from mast.vision.scan_monitor import ScanVisionMonitor

    captured: list = []

    class _B:
        def next_seq(self):
            return 1

        def emit_event(self, ev):
            captured.append(ev)

    sm = object.__new__(ScanVisionMonitor)
    sm._buf = lambda: _B()          # type: ignore[method-assign]
    sm._scan_id = "scan-1"          # type: ignore[attr-defined]
    ScanVisionMonitor._emit_alert(sm, "tip_quality_drop", "critical", "突变", {})
    assert captured, "产出方没有发出事件"
    assert captured[0].payload.get("source") == "vision_scan_monitor"

    # ② TipStatus 上升沿。走真的 BufferService,不是替身。
    from mast.buffer.schemas import TipQuality, TipStatus
    from mast.buffer.service import BufferService

    svc = BufferService()
    got: list = []
    svc.emit_event = lambda ev: got.append(ev)   # type: ignore[method-assign]
    svc._last_emitted_quality = TipQuality.GOOD
    svc._maybe_emit_quality_drop(
        TipStatus(seqno=7, quality=TipQuality.BAD, confidence=0.9,
                  scan_id="s", frame_idx=3))
    assert got, "上升沿没有发出事件"
    assert got[0].payload.get("source") == "vision_tip_status"


def test_a_broken_buffer_does_not_lose_the_table_half(store):
    """缓冲区读不到时,告警表那一半必须照常送 —— 两条管道互相独立。"""
    class Boom:
        def get_event_history(self, **kw):
            raise RuntimeError("buffer gone")

    store.add_alert(ts=time.time() - 5, level="critical", rule="saturation",
                    summary_zh=SAT_ZH)
    mw = AlertDeliveryMiddleware(get_store=lambda: store,
                                 get_buffer=lambda: Boom())
    seen: dict = {}
    mw.wrap_model_call(_Req([HumanMessage(content="继续")]),
                       lambda r: seen.setdefault("r", r))
    assert "saturation" in seen["r"].messages[-1].content


# ══════════════════════════════════════════════════════════════════════════
# 6. 升级中的序列 ≠ 反复复发的噪声
# ══════════════════════════════════════════════════════════════════════════

def test_the_run_up_warns_are_labelled_as_a_deterioration_timeline(store):
    """`rms_high` → `rtn_bistable` → `saturation`,59 秒内三级恶化。

    前两级单独看是 warn、容易被当背景噪声,但**它们连成的序列**才是最早的可行动
    信号(比 CRITICAL 早 8 秒)。所以有 CRITICAL 在场时,先于它发生的 warn 要被
    标成恶化时间线,并且把提前量说出来。
    """
    t0 = time.time() - 120
    _incident(store, t0)
    rows = store.undelivered_alerts(time.time() - 3600, limit=50)
    block = format_alert_block(select_alerts(rows, AlertRouting()))

    assert "恶化时间线" in block
    assert "最早的早 10 秒" in block, block
    assert "不是背景噪声" in block
    # 序列的每一级都还在 —— 折叠压的是「同一条反复响」,不是「不同条依次响」。
    assert "rms_high" in block and "rtn_bistable" in block


def test_recurring_noise_alone_is_not_called_a_timeline(store):
    """没有 CRITICAL 时,warn 就是 warn —— 不许把普通 warn 说成恶化时间线。"""
    now = time.time()
    for k in range(3):
        store.add_alert(ts=now - 60 + k * 5, level="warn", rule="rms_high",
                        summary_zh=RMS_ZH)
    rows = store.undelivered_alerts(now - 3600, limit=50)
    block = format_alert_block(select_alerts(rows, AlertRouting()))
    assert "恶化时间线" not in block
    assert "仅供参考" in block


# ══════════════════════════════════════════════════════════════════════════
# 7. 老库迁移
# ══════════════════════════════════════════════════════════════════════════

def test_old_db_without_the_delivery_columns_is_migrated(tmp_path):
    """v7 老库开机即补列 —— 不需要迁移脚本,也不能静默失败。"""
    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(str(db))
    con.executescript(
        "CREATE TABLE alerts ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,"
        " level TEXT NOT NULL, rule TEXT NOT NULL, summary_zh TEXT NOT NULL,"
        " segment_id INTEGER, evidence_png TEXT, features_json TEXT,"
        " emitted_buffer INTEGER NOT NULL DEFAULT 0,"
        " acked INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO alerts (ts, level, rule, summary_zh)"
        " VALUES (1000.0, 'critical', 'saturation', '老库里的一条');"
    )
    con.commit()
    con.close()

    s = CurrentMonitorStore(db, tmp_path)
    try:
        cols = {r["name"] for r in s._conn.execute("PRAGMA table_info(alerts)")}
        assert "delivered_agent" in cols and "delivered_ts" in cols
        # 老行默认「未送达」—— 它们确实没被送给过任何 agent。
        rows = s.undelivered_alerts(0.0, limit=10)
        assert [r["summary_zh"] for r in rows] == ["老库里的一条"]
        assert s.mark_alerts_delivered([rows[0]["id"]]) == 1
        assert s.undelivered_alerts(0.0, limit=10) == []
    finally:
        s.close()
