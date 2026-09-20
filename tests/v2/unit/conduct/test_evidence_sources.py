"""外部证据源 —— ``monitor_events`` / ``frame_metrics``,以及它们共用的那个原语。

M3-b。这一组钉的**不是**「探针能返回数据」,而是三条区分:

1. **「读不到」不是「没有」。** 一个停着的监控守护线程报出来的「零条告警」说的是
   「没人在看」,不是「这段时间很太平」;而对「要不要接着扫一整夜」这个问题,
   那两句话给的是相反的答案。空列表在闸门那里会被读成「查过了,没事」——
   所以读不到一律走 ``missing``,不走 ``values``。
2. **外部证据没有 conduct 的代次章。** ``step_data`` 是 Director 亲手盖的章,
   而监控告警和扫描文件是别的链路产的,只有时间戳。代次归属只能**派生**:
   晚于「当前代次开始时刻」的才算当代。那个时刻答不出来 ⇒ **判不了**,
   而不是「就当它是当代的」。
3. **``verify_verdict`` 仍然没接,而且不是忘了** —— 全仓没有裁决的持久化真源。

全程 ``tmp_path`` + 注入探针:一次真实的监控库、一个真实的 .sxm 都不碰。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from _harness import FakeExecutor, build, outcome, spec, stage, step

from mast.conduct.director import EPOCH_START_KEY, TickReport
from mast.conduct.spec import (
    DetourPolicy,
    EvidenceSpec,
    GateOutcome,
    GateSpec,
    RuleLeaf,
)


def _gate(source: str, *, selector: str = "", max_age_s=None,
          field_name: str = "monitor_alert_count", op: str = "==", value=0,
          gate_id: str = "g") -> GateSpec:
    return GateSpec(
        gate_id=gate_id, kind="rule",
        evidence=(EvidenceSpec(source=source, selector=selector,
                               max_age_s=max_age_s, min_epoch="current"),),
        rule=RuleLeaf(field_name, op, value),
        routes={"pass": GateOutcome("pass"),
                "fail": GateOutcome("wait_operator", "判定不通过")},
        unattended_escape="wait_operator", evidence_missing="wait_operator")


def _rig(tmp_path, gate, **kw):
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),), exit_gate=gate)])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 1})]})
    return build(tmp_path, s, executor=ex, **kw)


def _run_to_gate(rig, n: int = 4):
    for _ in range(n):
        rig.tick()
    return rig.events(kind="gate_evaluated")[-1]["payload"]


# ── 1. 代次原语:当前这一代是什么时候开始的 ──────────────────────────────

def test_being_adopted_stamps_when_the_first_epoch_began(tmp_path):
    """第一代从被采纳那一刻开始。

    不写这个章,第一代就永远答不出开始时刻,于是**每一条**外部证据都被判成
    跨代次而安静地转人 —— 一个看起来接好了、实际一条证据都收不到的源。
    """
    rig = _rig(tmp_path, _gate("monitor_events"), monitor_probe=lambda s, since: {})
    rig.tick()
    p = rig.events(kind="adopted")[-1]["payload"]
    assert p["evidence_epoch"] == 0
    assert p[EPOCH_START_KEY] == pytest.approx(rig.clock.t, abs=1e-6)
    assert rig.director._epoch_started_at(rig.conduct_id, 0) == pytest.approx(
        p[EPOCH_START_KEY], abs=1e-6)


def test_a_detour_moves_the_epoch_start_forward(tmp_path):
    """进绕道 = 换了一代证据,而**新一代从进绕道那一刻开始**。

    这正是「在自己刚炸出来的坑上判针尖」的结构化预防在外部证据上的落点:
    坏针之前那些监控告警、那些帧,时间戳都早于这个时刻,收不进来。
    """
    g = _gate("monitor_events")
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),), exit_gate=g),
              stage("R", steps=(step("R.00", "SafeRetract"),))],
             detour=DetourPolicy(target_stage="R"))
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 1})]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick()
    first = rig.director._epoch_started_at(rig.conduct_id, 0)
    rig.clock.advance(3600.0)
    rig.director._enter_detour(rig.conduct_id, rig.row(), s, "针坏了",
                               TickReport())
    row = rig.row()
    assert int(row["evidence_epoch"]) == 1
    second = rig.director._epoch_started_at(rig.conduct_id, 1)
    assert second is not None and second > first
    assert second == pytest.approx(rig.clock.t, abs=1e-6)


def test_a_finished_step_is_not_an_epoch_boundary(tmp_path):
    """⚠️ **回归**:代次边界的标记必须是**专用**的。

    第一版用「payload 里同时有 ``evidence_epoch`` 和 ``at``」当标记,而
    ``step_finished`` 的 payload 两样都有 —— 于是每跑完一步就算一次代次边界,
    「这一代从什么时候开始」一路跟着最后一步的完成时刻往前爬,证据窗口越缩越窄。
    方向还是**开的**:五分钟前那条 critical 告警,会因为一分钟前有一步跑完了
    而被排除在外 —— 漏掉告警,不是多报。

    这个 bug 是变异测试逼出来的(把 ``max_age`` 那道下界拿掉,窗口测试本该变红
    却没有)。**标记要专用,别拿一个「碰巧只有它有」的组合当标记。**
    """
    rig = _rig(tmp_path, _gate("monitor_events"),
               monitor_probe=lambda s, since: {"monitor_alert_count": 0})
    rig.tick()                                   # adopted
    started = rig.director._epoch_started_at(rig.conduct_id, 0)
    rig.clock.advance(600.0)
    for _ in range(4):                           # 跑完一步 + 判闸
        rig.tick()
    assert any(e["kind"] == "step_finished" for e in rig.events()), "步没跑完"
    after = rig.director._epoch_started_at(rig.conduct_id, 0)
    assert after == pytest.approx(started, abs=1e-6), (
        "一步跑完把代次开始时刻往前推了 —— 证据窗口会一路收窄")


def test_an_epoch_with_no_start_time_makes_external_evidence_undecidable(tmp_path):
    """代次开始时刻答不出来 ⇒ 外部证据的代次归属答不出来 ⇒ **判不了**。

    不是「就当它是当代的」—— 那正是把上一根针留下的告警喂进这一代闸门的那条路。
    """
    rig = _rig(tmp_path, _gate("monitor_events"),
               monitor_probe=lambda s, since: {"monitor_alert_count": 0})
    rig.director._epoch_started_at = lambda cid, epoch: None
    p = _run_to_gate(rig)
    assert p["verdict"] == "wait_operator"
    assert any("什么时候开始" in m for m in p["missing"])
    assert rig.row()["status"] == "waiting_operator"


# ── 2. 读不到 ≠ 没有 ────────────────────────────────────────────────────

@pytest.mark.parametrize("source,field", [
    ("monitor_events", "monitor_alert_count"),
    ("frame_metrics", "frame_count"),
])
def test_an_unwired_probe_is_absent_evidence_not_a_clean_bill(tmp_path, source, field):
    rig = _rig(tmp_path, _gate(source, field_name=field))
    p = _run_to_gate(rig)
    assert p["verdict"] == "wait_operator"
    assert any("未接入探针" in m for m in p["missing"])


@pytest.mark.parametrize("source,field,kw", [
    ("monitor_events", "monitor_alert_count", "monitor_probe"),
    ("frame_metrics", "frame_count", "frame_metrics_probe"),
])
def test_a_probe_that_says_unreadable_never_becomes_zero(tmp_path, source, field, kw):
    """探针回 ``None`` = 读不到。**绝不折成 0 条 / 0 帧。**

    折成 0 的话,判据 ``count == 0`` 会**通过** —— 一次「监控根本没在跑」于是
    长成一句「这段时间很太平」,而闸门放行了一整夜的扫描。
    """
    rig = _rig(tmp_path, _gate(source, field_name=field),
               **{kw: lambda s, since: None})
    p = _run_to_gate(rig)
    assert p["verdict"] == "wait_operator"
    assert any("读不到不等于没有" in m for m in p["missing"])


def test_a_probe_that_explodes_is_unreadable_not_a_crash(tmp_path):
    def _boom(selector, since):
        raise RuntimeError("监控库锁住了")

    rig = _rig(tmp_path, _gate("monitor_events"), monitor_probe=_boom)
    p = _run_to_gate(rig)
    assert p["verdict"] == "wait_operator"
    assert any("探针抛异常" in m for m in p["missing"])


def test_a_probe_returning_the_wrong_shape_is_unreadable(tmp_path):
    rig = _rig(tmp_path, _gate("monitor_events"),
               monitor_probe=lambda s, since: [1, 2, 3])
    p = _run_to_gate(rig)
    assert any("不是证据 dict" in m for m in p["missing"])


# ── 3. 接通之后真的能判 ─────────────────────────────────────────────────

def test_a_wired_monitor_source_lets_a_rule_gate_actually_judge(tmp_path):
    """这一条是整件事的目的:闸门第一次看得见 ``step_data`` 之外的东西。"""
    rig = _rig(tmp_path, _gate("monitor_events", field_name="monitor_crit_count",
                               op="==", value=0),
               monitor_probe=lambda s, since: {
                   "monitor_alert_count": 2, "monitor_crit_count": 0,
                   "monitor_worst_level": "warn"})
    p = _run_to_gate(rig)
    assert p["verdict"] == "pass" and p["rule_state"] == "true"
    assert not p["missing"]


def test_a_critical_alert_in_this_epoch_stops_the_gate(tmp_path):
    rig = _rig(tmp_path, _gate("monitor_events", field_name="monitor_crit_count",
                               op="==", value=0),
               monitor_probe=lambda s, since: {"monitor_crit_count": 1,
                                               "monitor_worst_level": "critical"})
    p = _run_to_gate(rig)
    assert p["verdict"] == "wait_operator" and p["rule_state"] == "false"


def test_the_window_is_the_later_of_epoch_start_and_max_age(tmp_path):
    """两道下界都要,互相替代不了:代次管「针换过没有」,年龄管「这条消息还新鲜吗」。

    只用代次:一段跑了三天的 conduct 会把三天前的告警当成当代证据;
    只用年龄:换过针之后,上一根针留下的告警只要够新就照样进得来。
    """
    seen: list[float] = []

    def _probe(selector, since):
        seen.append(since)
        return {"monitor_alert_count": 0}

    rig = _rig(tmp_path, _gate("monitor_events", max_age_s=300.0),
               monitor_probe=_probe)
    rig.tick()                        # adopted —— 代次从此刻开始
    started = rig.director._epoch_started_at(rig.conduct_id, 0)
    rig.clock.advance(10_000.0)       # 代次开始已经很久 ⇒ 年龄那道更晚
    for _ in range(4):
        rig.tick()
    assert seen, "探针没被调用"
    assert seen[-1] == pytest.approx(rig.clock.t - 300.0, abs=1e-6)
    assert seen[-1] > started

    # 反过来:代次刚开始 ⇒ 代次那道更晚
    seen.clear()
    rig2 = _rig(tmp_path / "b", _gate("monitor_events", max_age_s=10_000.0),
                monitor_probe=_probe)
    for _ in range(4):
        rig2.tick()
    started2 = rig2.director._epoch_started_at(rig2.conduct_id, 0)
    assert seen[-1] == pytest.approx(started2, abs=1e-6)


def test_the_selector_reaches_the_probe(tmp_path):
    got: list[str] = []
    rig = _rig(tmp_path, _gate("frame_metrics", selector="Current",
                               field_name="frame_count", op=">=", value=0),
               frame_metrics_probe=lambda s, since: got.append(s) or {"frame_count": 0})
    _run_to_gate(rig)
    assert got and got[-1] == "Current"


# ── 4. verify_verdict:仍然没接,而且不是忘了 ────────────────────────────

def test_verify_verdict_is_still_absent_and_says_why(tmp_path):
    """⚠️ 这条钉的是**一个还没有真源的源**,不是一个待办。

    裁决今天只有两个去处:步产出(那已经是 ``step_data`` —— 接成第二个源就是同一
    个事实的两个真源),或者 S2 逐偏压账本那样的文件(要先有一份「裁决登记」的
    设计)。在那之前,接它只能是**把 step_data 抄一遍**。

    **这条测试变红 = 有人给了裁决一个持久化真源**,那时把它翻成正向断言。
    """
    rig = _rig(tmp_path, _gate("verify_verdict", field_name="anything",
                               op="==", value="x"))
    p = _run_to_gate(rig)
    assert p["verdict"] == "wait_operator"
    assert any("没有裁决持久化真源" in m for m in p["missing"])


# ── 5. 探针本身(adapters)——三态在这里 ─────────────────────────────────

class _FakeStore:
    def __init__(self, rows=None, boom=False):
        self.rows = list(rows or [])
        self.boom = boom
        self.asked: list[float] = []

    def alerts_query(self, since=None, limit=50, level=None):
        self.asked.append(float(since or 0.0))
        if self.boom:
            raise RuntimeError("库锁住了")
        return [r for r in self.rows if float(r["ts"]) >= float(since or 0.0)], 0


class _FakeSvc:
    def __init__(self, running=True, boom=False):
        self.running = running
        self.boom = boom

    def status(self):
        if self.boom:
            raise RuntimeError("status 读不到")
        return {"running": self.running}


def _monitor(svc, store):
    from mast.conduct.adapters import RuntimeMonitorEvents

    return RuntimeMonitorEvents(service_getter=lambda: svc,
                                store_getter=lambda: store)


def test_a_monitor_that_never_ran_reads_unreadable_not_peaceful():
    """守护线程从没起过 ⇒ ``None``。**不是「零条告警」。**

    ``get_store_if_exists`` 的 docstring 说「没跑过就是没有告警,None 是诚实的
    答案」—— 那是**它那个调用方**的语义(投递中间件:没有就不投)。在闸门这一侧
    同一个 None 意思完全不同:没人在看。同一条降级路径,换个调用方就换了意思。
    """
    assert _monitor(None, None)("", 0.0) is None


def test_a_stopped_monitor_cannot_report_a_quiet_hour():
    assert _monitor(_FakeSvc(running=False), _FakeStore())("", 0.0) is None


def test_a_monitor_whose_status_cannot_be_read_is_unreadable():
    assert _monitor(_FakeSvc(boom=True), _FakeStore())("", 0.0) is None


def test_a_running_monitor_with_no_alerts_reports_a_real_zero():
    """这一支才是「查过了,这段时间很太平」—— 一个答得上来的 0。"""
    got = _monitor(_FakeSvc(), _FakeStore())("", 0.0)
    assert got is not None
    assert got["monitor_alert_count"] == 0
    assert got["monitor_worst_level"] == "none"


def test_a_query_that_explodes_is_unreadable_not_zero():
    assert _monitor(_FakeSvc(), _FakeStore(boom=True))("", 0.0) is None


def test_the_real_store_agrees_with_this_probe_about_what_a_failure_looks_like(tmp_path):
    """⚠️ **两个模块要对同一件事有同一个说法**,而上面那些用的都是替身。

    在 2026-08-15 之前,真的 ``CurrentMonitorStore.alerts_query`` 查询失败时
    **自吞异常回 ``([], 0)``** —— 于是替身钉得再准也没用:这条探针在真机上收到的
    是一个 0 条告警的成功返回,前面两道检查(守护线程在跑 + store 存在)一道都
    拦不住,「监控库锁住了」就长成了「这段时间很太平」。

    所以这一条用**真的 store**(建在 tmp_path、然后关掉 —— 库坏掉在真机上就是
    这个形状),把生产方与消费方接在一起验一次。替身测的是这条探针的逻辑,
    这一条测的是那个逻辑**建立在真的前提上**。
    """
    from mast.monitoring.store import CurrentMonitorStore

    real = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    try:
        healthy = _monitor(_FakeSvc(), real)("", 0.0)
        assert healthy is not None, "健康的空库要答得出一个真的 0"
        assert healthy["monitor_alert_count"] == 0
        assert healthy["monitor_worst_level"] == "none"

        real.close()
        assert _monitor(_FakeSvc(), real)("", 0.0) is None, (
            "库读不到却报成「零条告警」—— 闸门会拿它放行一整夜的扫描")
    finally:
        real.close()


def test_alerts_are_counted_by_level_and_the_worst_one_is_named():
    rows = [{"ts": 10.0, "level": "warn", "rule": "drift"},
            {"ts": 20.0, "level": "critical", "rule": "tip_crash"},
            {"ts": 30.0, "level": "info", "rule": "note"}]
    got = _monitor(_FakeSvc(), _FakeStore(rows))("", 0.0)
    assert got["monitor_alert_count"] == 3
    assert got["monitor_crit_count"] == 1
    assert got["monitor_worst_level"] == "critical"
    assert got["monitor_rules"] == ("drift", "note", "tip_crash")
    assert isinstance(got["monitor_rules"], tuple), "RuleLeaf(op='in') 要的是 tuple"


def test_the_selector_filters_by_level_floor():
    rows = [{"ts": 10.0, "level": "info", "rule": "note"},
            {"ts": 20.0, "level": "critical", "rule": "tip_crash"}]
    got = _monitor(_FakeSvc(), _FakeStore(rows))("critical", 0.0)
    assert got["monitor_alert_count"] == 1 and got["monitor_rules"] == ("tip_crash",)


def test_an_unknown_alert_level_is_treated_as_the_most_severe():
    """新加一个 ``fatal`` 级别时,它**不该**被当成最低级而静默滤掉 ——
    「闸门看不见它本该看见的东西」正是这么来的。"""
    rows = [{"ts": 10.0, "level": "fatal", "rule": "meltdown"}]
    got = _monitor(_FakeSvc(), _FakeStore(rows))("critical", 0.0)
    assert got["monitor_alert_count"] == 1


def test_the_since_watermark_reaches_the_store():
    store = _FakeStore()
    _monitor(_FakeSvc(), store)("", 1234.0)
    assert store.asked == [1234.0]


# ── frame_metrics 探针 ──────────────────────────────────────────────────

class _M:
    rowcorr_median = 0.87
    fb_instability = None
    fine_periodic_snr = float("nan")


def _frames(scans, *, loader=None, measure=None):
    from mast.conduct.adapters import RuntimeFrameMetrics

    return RuntimeFrameMetrics(list_scans=lambda n=20: scans,
                               loader=loader or (lambda p, c: {"forward": [[1.0]],
                                                               "backward": None,
                                                               "nm_per_px": 0.1}),
                               measure=measure or (lambda fr: _M()))


def test_no_frame_in_this_epoch_is_a_real_zero_not_unreadable():
    """登记表读到了,只是这一代还没有帧 —— **答得上来**,所以给 0。

    把它也报成「读不到」,真正的读不到就淹没在噪声里了。
    """
    got = _frames([{"scan_id": "a", "path": "x.sxm", "mtime": 10.0}])("Z", 999.0)
    assert got == {"frame_count": 0, "frame_channel": "Z",
                   "frame_scan_id": "", "frame_age_s": None}


def test_a_registry_record_with_no_timestamp_is_skipped_not_assumed_fresh():
    """时刻答不出来 ⇒ 代次归属答不出来 ⇒ 这一条不能用。
    「那就当它是新的」正是把旧帧喂进闸门的那条路。"""
    got = _frames([{"scan_id": "a", "path": "x.sxm"}])("Z", 0.0)
    assert got["frame_count"] == 0 and got["frame_scan_id"] == ""


def test_a_frame_that_cannot_be_loaded_is_unreadable_not_zero():
    def _boom(path, channel):
        raise OSError("文件没了")

    got = _frames([{"scan_id": "a", "path": "x.sxm", "mtime": 10.0}],
                  loader=_boom)("Z", 0.0)
    assert got is None


def test_a_missing_channel_is_unreadable_not_a_metric_of_zero():
    got = _frames([{"scan_id": "a", "path": "x.sxm", "mtime": 10.0}],
                  loader=lambda p, c: {"forward": None})("Z", 0.0)
    assert got is None


def test_a_measured_frame_reports_its_metrics_and_nan_becomes_none():
    """``measure_frame`` 用 NaN 表示「没算出来」。带到闸门那里,NaN 的比较一律
    为假 —— 于是「没算出来」会长成「不合格」。换成 ``None``,三态求值才report
    得出「判不了」。"""
    got = _frames([{"scan_id": "s001", "path": "x.sxm", "mtime": 10.0}])("Z", 0.0)
    assert got["frame_count"] == 1 and got["frame_scan_id"] == "s001"
    assert got["frame_rowcorr_median"] == pytest.approx(0.87)
    assert got["frame_fb_instability"] is None      # 判据本来就没算成
    assert got["frame_periodic_snr"] is None        # NaN ⇒ None,不是 0.0
