"""runtime 挂点 —— 开关的两侧,以及「一扇门」的外溢。

设计 §3-1(住在哪)、§4.8(实验文件夹)、§7(WS 帧)。四组:

1. **关着 = 什么都没建**:不建线程、不建 store、连库文件都不落。
   「默认关」如果只是「建好了但不跑」,那它就不是「逐字节等于今天」。
2. **开着 = 先清算再起线程**:顺序反了的话,线程会在清算之前先推进一步,
   而那一步的世界状态正是「进程刚死过一次」。
3. **一扇门**:凡是经 ``store.record`` 改了状态的,都必然外溢一行 jsonl + 一帧。
   靠人肉在 N 个调用点各加一行,漏掉的那几个要等真机才发现。
4. **abort 不等 tick**:API 线程立刻置 per-run Event。

全程 ``tmp_path``:store 落在 tmp 的库文件里,实验文件夹用注入的 resolver。
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.conduct import service as svc_mod  # noqa: E402
from mast.conduct.settings import (  # noqa: E402
    ConductKnobs, get_conduct_knobs, knob_catalog, set_conduct_knobs,
)


class _Rt:
    """最小的 runtime 替身:一个 config.db_path 和一个急停闩。"""

    def __init__(self, db_path):
        self.config = type("C", (), {"db_path": db_path})()
        self._orch_abort = threading.Event()
        self._pool = object()
        self._state = object()
        self._registry = None

    def emergency_latch_state(self):
        return {"latched": False, "abort_set": False, "why": ""}

    def latest_temperature(self, channel=None):
        from mast.core.temperature import NO_SOURCE, TempReading
        return TempReading(channel=str(channel or ""), reason=NO_SOURCE)


@pytest.fixture(autouse=True)
def _isolated_service():
    """每条测试独占一份单例 + 旋钮,跑完还回去。"""
    before = get_conduct_knobs()
    svc_mod.set_service_for_test(None)
    yield
    svc_mod.stop_service(drop=True)
    svc_mod.set_service_for_test(None)
    set_conduct_knobs(before)


class _Bus:
    def __init__(self):
        self.frames = []

    def publish_conduct_status(self, conduct_id, **kw):
        self.frames.append(("status", conduct_id, kw))

    def publish_conduct_gate(self, conduct_id, **kw):
        self.frames.append(("gate", conduct_id, kw))

    def publish_conduct_alert(self, conduct_id, **kw):
        self.frames.append(("alert", conduct_id, kw))


def _service(tmp_path, bus=None):
    """开着的服务,库与实验文件夹都在 tmp_path 下。"""
    set_conduct_knobs({"cd_enabled": 1.0})
    exp_root = tmp_path / "exp"
    return svc_mod.ConductService(
        _Rt(tmp_path / "db" / "camp.db"), db_path=tmp_path / "db" / "camp.db",
        folder_resolver=lambda _e: exp_root, bus=bus or _Bus())


# ── 1. 关着 ────────────────────────────────────────────────────────

def test_default_is_off():
    assert ConductKnobs().cd_enabled == 0.0
    assert ConductKnobs().enabled is False


def test_disabled_builds_nothing_at_all(tmp_path):
    set_conduct_knobs({"cd_enabled": 0.0})
    db = tmp_path / "db" / "camp.db"
    assert svc_mod.start_service(_Rt(db), db_path=db) is None
    assert svc_mod.get_service() is None
    assert not db.exists(), (
        "关着却把库建出来了 —— 「默认关」的承诺是逐字节等于这个功能落地之前,"
        "不是「建好了但不跑」")


def test_enabling_then_disabling_actually_stops_the_thread(tmp_path):
    set_conduct_knobs({"cd_enabled": 1.0})
    db = tmp_path / "db" / "camp.db"
    svc = svc_mod.start_service(_Rt(db), db_path=db,
                                folder_resolver=lambda _e: tmp_path / "exp")
    assert svc is not None and svc.is_running
    set_conduct_knobs({"cd_enabled": 0.0})
    svc_mod.apply_settings(None)
    assert not svc.is_running, (
        "开关关了线程还在跑 —— 一个「按了没反应」的开关比没有开关更危险")


def test_knob_catalog_exposes_every_knob_with_bounds():
    keys = {k["key"] for k in knob_catalog()}
    assert keys == {"cd_enabled", "cd_stall_grace_s",
                    "cd_autonomy", "cd_ignition_delay_s"}
    grace = next(k for k in knob_catalog() if k["key"] == "cd_stall_grace_s")
    assert grace["min"] >= 30.0, "余量下界太小,停滞告警会变成必然误报"

    # 自主度的上界必须跟着闭集走。写死一个 2 而闭集加到四档时,新档位会被
    # 静默夹回去,而设置页会显示成「设上了」—— 那是本仓「夹紧让填错看起来像
    # 填对」的同一个形状。
    from mast.conduct.autonomy import AUTONOMY_LEVELS

    aut = next(k for k in knob_catalog() if k["key"] == "cd_autonomy")
    assert aut["max"] == float(len(AUTONOMY_LEVELS) - 1)
    assert aut["min"] == 0.0, "自主度下界不是最严档 —— 那道默认就没了"


def test_knobs_reject_a_string_instead_of_coercing_it():
    """``"1"`` 不是 1.0。强转会让「类型填错了」看起来像「填对了」。"""
    assert ConductKnobs.from_mapping({"cd_enabled": "1"}).enabled is False
    assert ConductKnobs.from_mapping({"cd_enabled": True}).enabled is True


def test_out_of_range_knob_is_clamped_to_the_declared_bounds():
    k = ConductKnobs.from_mapping({"cd_stall_grace_s": 999999.0})
    assert k.cd_stall_grace_s == 7200.0


# ── 2/3. 开着:清算在前,外溢是一扇门 ─────────────────────────────

def test_start_reconciles_before_the_thread_runs(tmp_path):
    svc = _service(tmp_path)
    cid = svc.store.create(experiment_id="e1", spec_id="_smoke_v1",
                           spec_version=1)
    svc.store.record(cid, "approved", changes={"status": "approved"})
    svc.store.record(cid, "adopted", changes={"status": "running"})
    # 进程「重启」:非终态 ⇒ RECOVERY_PENDING。
    rep = svc.director.reconcile_after_restart()
    assert rep.actions == ["restart_to_recovery"]
    assert svc.store.get(cid)["status"] == "recovery_pending"


def test_paused_survives_a_restart(tmp_path):
    """尊重人的暂停 —— 重启不该把它变成「重新开始」。"""
    svc = _service(tmp_path)
    cid = svc.store.create(experiment_id="e1", spec_id="_smoke_v1", spec_version=1)
    svc.store.record(cid, "status_change",
                     changes={"status": "paused", "status_reason": "人按了暂停"})
    svc.director.reconcile_after_restart()
    assert svc.store.get(cid)["status"] == "paused"


def test_every_state_change_spills_one_line_and_one_frame(tmp_path):
    bus = _Bus()
    svc = _service(tmp_path, bus=bus)
    cid = svc.store.create(experiment_id="e1", spec_id="_smoke_v1", spec_version=1)
    svc.store.record(cid, "approved", changes={"status": "approved"})
    svc.store.record(cid, "adopted", changes={"status": "running"})

    st = svc.journal(cid, "e1").status()
    assert st.progress_lines == 3, "created/approved/adopted 三条,一条都不能少"
    kinds = [f[2].get("kind") for f in bus.frames if f[0] == "status"]
    assert kinds == ["created", "approved", "adopted"]
    assert bus.frames[-1][2]["status"] == "running"


def test_a_gate_event_becomes_a_gate_frame_not_a_status_frame(tmp_path):
    bus = _Bus()
    svc = _service(tmp_path, bus=bus)
    cid = svc.store.create(experiment_id="e1", spec_id="_smoke_v1", spec_version=1)
    svc.store.record(cid, "gate_evaluated", stage_id="S1",
                     payload={"gate_id": "g1", "verdict": "pass"})
    gates = [f for f in bus.frames if f[0] == "gate"]
    assert len(gates) == 1 and gates[0][2]["gate_id"] == "g1"


def test_a_broken_journal_never_rolls_back_the_source_of_truth(tmp_path):
    """人读副本写不下去 ⇒ 真源照样提交。反过来就是拿审计副本绑架状态机。"""
    svc = svc_mod.ConductService(
        _Rt(tmp_path / "db" / "camp.db"), db_path=tmp_path / "db" / "camp.db",
        folder_resolver=lambda _e: None, bus=_Bus())
    cid = svc.store.create(experiment_id="ghost", spec_id="_smoke_v1",
                           spec_version=1)
    svc.store.record(cid, "approved", changes={"status": "approved"})
    assert svc.store.get(cid)["status"] == "approved"
    assert svc.journal(cid, "ghost").status().wired is False


def test_an_observer_that_explodes_is_swallowed(tmp_path):
    from mast.conduct.store import ConductStore

    store = ConductStore(tmp_path / "c.db")

    def _boom(_ev):
        raise RuntimeError("外溢炸了")

    store.set_observer(_boom)
    cid = store.create(experiment_id="e", spec_id="s", spec_version=1)
    store.record(cid, "approved", changes={"status": "approved"})
    assert store.get(cid)["status"] == "approved"


# ── 4. abort 不等 tick ────────────────────────────────────────────

def test_abort_signals_the_running_step_without_a_tick(tmp_path):
    svc = _service(tmp_path)
    cid = svc.store.create(experiment_id="e1", spec_id="_smoke_v1", spec_version=1)
    ev = svc.aborts.register("run-xyz")
    svc.store.record(cid, "step_started", run_id="run-xyz",
                     changes={"active_run_id": "run-xyz"})
    assert svc.signal_abort(cid) is True
    assert ev.is_set(), (
        "Director 卡在一次 executor.run 里时 tick 不会来 —— "
        "abort 按钮必须不经 tick 就生效")


def test_abort_with_no_running_step_says_so_rather_than_lying(tmp_path):
    svc = _service(tmp_path)
    cid = svc.store.create(experiment_id="e1", spec_id="_smoke_v1", spec_version=1)
    assert svc.signal_abort(cid) is False


def test_cost_reader_says_unreadable_not_zero(tmp_path):
    """账本读不到 ⇒ **None**,不是 0。

    0 会让预算闸门报「没超」,而真相是「没看」。

    ⚠️ 探针必须注入:``RuntimeConductCost()`` 默认读
    ``experiments/usage_ledger.sqlite`` —— **用户真实的账本**。它只读不写,
    所以不是污染,但会让这条测试的结果取决于这台机器上那个文件里有什么。
    """
    from mast.conduct.adapters import ConductCost

    svc = _service(tmp_path)
    svc._cost_probe = lambda _cid: None                 # 读不到
    assert svc._cost_reader("whatever") is None
    assert svc.cost_detail("whatever") is None


def test_cost_reader_refuses_to_hand_over_a_partial_usd_total(tmp_path):
    """跨币种 ⇒ **None**,不是「USD 那一部分」。

    把 USD 那一笔单独交给预算闸,它会拿去比上限、判「没超」,而大头在另一个币种
    里 —— 那是一次**「检查通过」形态的漏报**,比不检查坏:director 会把它记成
    查过了。合并需要汇率,而本仓不自造汇率。
    """
    from mast.conduct.adapters import ConductCost

    svc = _service(tmp_path)
    svc._cost_probe = lambda _cid: ConductCost(
        {"CNY": 120.0, "USD": 0.4}, count=5, all_priced=True, reason="跨币种")
    assert svc._cost_reader("c-1") is None, "把 USD 那一笔单独交出去了"
    # 但面板那一口要看得见真相
    detail = svc.cost_detail("c-1")
    assert detail.by_currency == {"CNY": 120.0, "USD": 0.4}


def test_a_pure_usd_conduct_can_actually_be_measured(tmp_path):
    """全是 USD ⇒ 交出那个数。这条上限**在这种配置下真的能拦**。"""
    from mast.conduct.adapters import ConductCost

    svc = _service(tmp_path)
    svc._cost_probe = lambda _cid: ConductCost(
        {"USD": 3.25}, count=2, all_priced=True, reason="")
    assert svc._cost_reader("c-1") == 3.25


def test_a_conduct_that_spent_nothing_is_a_real_zero(tmp_path):
    """账本读到了、这份 conduct 一条记录都没有 ⇒ **0.0**,不是「读不到」。

    把答得上来的也报成读不到,真正的读不到就淹没在噪声里了(与帧指标探针
    「这一代还没有帧 ⇒ 0」同一条纪律)。
    """
    from mast.conduct.adapters import ConductCost

    svc = _service(tmp_path)
    svc._cost_probe = lambda _cid: ConductCost({}, count=0, all_priced=True,
                                                reason="还没有任何一条调用")
    assert svc._cost_reader("c-1") == 0.0


# ── 5. 判决席位(M3-a)───────────────────────────────────────────────

_NODE = {"id": "n", "responsibility": "判这一段还值不值得接着测",
         "routes": {"go": "继续", "hold": "停"}, "escape": "hold"}


@pytest.fixture()
def _no_network(monkeypatch):
    """把这条路径上会打网络、会写真实 experiments 的两处都掐掉。

    决策日志落在 ``project_root()/experiments/decision_log.jsonl`` —— 那是**真实
    用户数据**。测试污染真实数据本仓已经发生过五次,每次的入口都是一个善意的
    默认路径,所以这里不是「顺手 patch 一下」,是必须。
    """
    import mast.agents._shared.models as models_mod
    from mast.skills.composite import llm_node

    rows: list[dict] = []
    monkeypatch.setattr(llm_node, "log_decision", rows.append)
    monkeypatch.setattr(models_mod, "make_chat_model",
                        lambda *a, **kw: ("fake-model", kw))
    return rows


def test_the_llm_seat_is_actually_wired_not_just_declared(tmp_path, _no_network):
    """``decide_route`` 真的接上了 —— 而且经过的是**席位**,不是裸的
    ``decide_route``(裸的那个不写决策日志:写日志的是调用方)。"""
    from mast.skills.composite import llm_node

    seen: dict = {}

    def _fake_decide(node, inputs, model=None):
        seen["model"] = model
        return {"route": "go", "reason": "两个偏压都有分辨", "escaped": False,
                "parse_path": "structured", "model": "kimi-k3"}

    svc = _service(tmp_path)
    assert svc.director._decide_route is not None, "判决席位没接上"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(llm_node, "decide_route", _fake_decide)
        out = svc.director._decide_route(_NODE, {"n_resolved": 2})
    assert out["route"] == "go"
    # 席位建的模型带 request_timeout:一条卡住的连接不该把工作线程永久停住
    assert seen["model"][1]["request_timeout"] > 0
    # 留痕真的写了(而且带着 conduct 侧的身份)
    assert len(_no_network) == 1 and _no_network[0]["mechanism"] == "conduct_gate"
    assert "conduct_id" in _no_network[0]


def test_a_seat_that_cannot_be_seated_reports_unjudgeable(tmp_path, monkeypatch):
    """一台没配任何 provider key 的机器上:``make_chat_model`` 抛 ⇒ **判不了**。

    不是 pass,也不是「就当模型选了 escape」—— 后者会让一次配置缺失长得像一次
    判决,而 escape 那条路可能通向 ``detour``。
    """
    import mast.agents._shared.models as models_mod
    from mast.conduct.llm_seat import SeatUnavailable
    from mast.skills.composite import llm_node

    monkeypatch.setattr(llm_node, "log_decision", lambda rec: None)

    def _no_key(*a, **kw):
        raise RuntimeError("no provider has an API key configured")

    monkeypatch.setattr(models_mod, "make_chat_model", _no_key)
    svc = _service(tmp_path)
    with pytest.raises(SeatUnavailable):
        svc.director._decide_route(_NODE, {})


def test_the_decision_context_never_invents_an_id(tmp_path):
    """没有 active conduct 时上下文是空串,不是一个看起来像 id 的东西。"""
    svc = _service(tmp_path)
    ctx = svc._decision_context()
    assert ctx["conduct_id"] == "" and ctx["spec_id"] == ""
