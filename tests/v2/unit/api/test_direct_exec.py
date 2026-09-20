"""直调内核 ``mast.api.direct_exec`` —— 技能直调 API 与外部 agent 网关共用的那一段。

## 这里钉的五件事（每一件都是直调路径曾经缺的）

1. **入账**：直调一次 ⇒ v1 与 v2 ``actions`` 各一行，且带调用方身份
   （v1 ``context`` / v2 ``agent_id`` + ``thread_id``）；失败与门口拒绝同样入账。
2. **不双记、不串账**：带位置的技能恰好一条地图标记；不写群聊 run 的核对台账。
3. **run_id**：每次直调一个新的 —— composite sidecar 的分键靠它。
4. **门口门控**：样品门控在门口判一次；放行后子步继承，不逐步重判。
5. **SI 字符串**：``"5n"`` 与 ``5e-9`` 都通；解析不了的在门口拒并入账。

这里刻意用**真的** ``ExecutionContext`` + 真注册表 + 假连接池 + 临时 v1/v2 存储：
在鸭子替身上断言「走了哪条路」等于什么都没测（本仓记过不止一次）。
"""
from __future__ import annotations

import ast
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.api import direct_exec  # noqa: E402
from mast.core.registry import SkillRegistry  # noqa: E402
from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.core.types import (  # noqa: E402
    HardwareState,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.logging.storage import ExperimentStorage  # noqa: E402
from mast.skills.base import BaseSkill  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# 探针技能
# ─────────────────────────────────────────────────────────────────────

RAN: list[tuple[str, dict]] = []


def _xspec():
    return ParameterSpec(name="x_m", type="float", unit="m", required=False,
                         min_value=-1e-6, max_value=1e-6,
                         description="横向位置,写成 '5n' 这样带前缀的字符串或 float")


class ExtProbeRead(BaseSkill):
    """只读、不取令牌；可选地失败。"""

    def metadata(self):
        return SkillMetadata(
            name="ExtProbeRead", version="1.0.0", category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO, description="probe",
            parameters=[_xspec(),
                        ParameterSpec(name="fail", type="bool", required=False,
                                      default=False, description="让它失败")],
            tags=["read"])

    def execute(self, context, params):
        RAN.append(("ExtProbeRead", dict(params)))
        if params.get("fail"):
            return SkillResult(skill_name="ExtProbeRead", success=False,
                               error="probe asked to fail")
        return SkillResult(skill_name="ExtProbeRead", success=True,
                           data={"x_m": params.get("x_m")}, summary="ok")


class ExtProbeScan(BaseSkill):
    """产数据的写技能（``scan`` 标签）—— 样品门控要管它。"""

    def metadata(self):
        return SkillMetadata(
            name="ExtProbeScan", version="1.0.0", category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO, description="probe scan",
            parameters=[], tags=["scan"])

    def execute(self, context, params):
        RAN.append(("ExtProbeScan", dict(params)))
        return SkillResult(skill_name="ExtProbeScan", success=True)


class ExtProbeParent(BaseSkill):
    """一个 composite 形状的技能：执行途中样品被切走，再调产数据的子步。"""

    def metadata(self):
        return SkillMetadata(
            name="ExtProbeParent", version="1.0.0", category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO, description="probe parent",
            parameters=[], tags=["workflow"])

    def execute(self, context, params):
        RAN.append(("ExtProbeParent", dict(params)))
        _LOG.current_sample_id = None          # 中途样品被切走
        sub = context.run("ExtProbeScan", {})
        return SkillResult(skill_name="ExtProbeParent", success=bool(sub.success),
                           error=str(sub.error or ""))


class ExtProbeHuman(BaseSkill):
    def metadata(self):
        return SkillMetadata(
            name="ExtProbeHuman", version="1.0.0", category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO, description="probe human", tags=["read"])

    def execute(self, context, params):
        from langgraph.errors import GraphInterrupt

        raise GraphInterrupt(())


_LOG = SimpleNamespace(current_experiment_id=None, current_sample_id=None)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    RAN.clear()
    _LOG.current_experiment_id = None
    _LOG.current_sample_id = None
    import mast.logging.experiment_log as el

    monkeypatch.setattr(el, "get_active_log", lambda: _LOG)
    yield
    RAN.clear()


# ─────────────────────────────────────────────────────────────────────
# 一个真 CoreRuntime 方法组成的「runtime」—— 只借入账相关的几个方法
# ─────────────────────────────────────────────────────────────────────

class _RT:
    _record_direct_skill_call = CoreRuntime._record_direct_skill_call
    _record_v1_action = CoreRuntime._record_v1_action
    _record_v2_action = CoreRuntime._record_v2_action
    _note_skill_activity = CoreRuntime._note_skill_activity

    def __init__(self, storage, v2=None):
        self._storage = storage
        self._experiment_log = _LOG
        self._v2_repos, self._v2_eid = v2 if v2 else (None, None)
        self._active_thread_id = "agents-concurrent-chat"   # 一个不相干的会话线程
        self._map_last_skill_ts = 0.0
        self.markers: list[dict] = []
        self.ledger: list[dict] = []

    def _record_map_marker(self, payload):
        self.markers.append(dict(payload))

    def _note_run_skill(self, payload):          # 群聊 run 的核对台账 —— 不许碰
        self.ledger.append(dict(payload))


class _State:
    def snapshot(self):
        return HardwareState(bias_v=0.1)

    def refresh(self):  # pragma: no cover — 直调绝不许调它（真硬件 I/O）
        raise AssertionError("direct-exec 调了 state.refresh()")


def _registry():
    reg = SkillRegistry()
    for cls in (ExtProbeRead, ExtProbeScan, ExtProbeParent, ExtProbeHuman):
        reg.register(cls)
    return reg


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    from mast.logging.v2.live import open_live_v2

    repos, v2eid = open_live_v2()
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    _LOG.current_experiment_id = eid
    rt = _RT(st, (repos, v2eid))
    ctx = SimpleNamespace(connection_pool=SimpleNamespace(name="pool"), state=_State(),
                          skill_registry=_registry(), live_app=rt, app=None)
    return SimpleNamespace(ctx=ctx, rt=rt, st=st, eid=eid, repos=repos, v2eid=v2eid)


def _run(w, skill, params=None, **kw):
    ec, missing = direct_exec.build_context(w.ctx, owner="外部 agent ext:tester")
    assert missing == [] and ec is not None
    kw.setdefault("agent_id", "ext:tester")
    kw.setdefault("thread_id", "ext:tester/s1")
    kw.setdefault("context", "ext:tester")
    kw.setdefault("approval_source", direct_exec.APPROVAL_LLM)
    return direct_exec.run_and_record(ec, skill, params or {}, runtime=w.rt, **kw), ec


# ─────────────────────────────────────────────────────────────────────
# 1. 入账
# ─────────────────────────────────────────────────────────────────────

def test_a_direct_call_lands_in_v1_and_v2_with_the_callers_identity(world):
    run, _ = _run(world, "ExtProbeRead", {"x_m": 1e-9}, tool_call_id="j_test")
    assert run.success and run.recorded.get("v1") is True
    assert run.recorded.get("v2_action_id")

    v1 = world.st.recent_actions(world.eid)
    assert [r["skill_name"] for r in v1] == ["ExtProbeRead"]
    assert v1[0]["context"] == "ext:tester"
    assert v1[0]["approval_source"] == "llm"
    assert v1[0]["success"] is True

    v2 = world.repos.actions.for_experiment(world.v2eid)
    assert len(v2) == 1
    row = v2[0]
    assert row["agent_id"] == "ext:tester"
    assert row["thread_id"] == "ext:tester/s1", (
        "外部动作被挂到了并发的内部会话线程上")
    assert row["tool_call_id"] == "j_test"
    assert row["status"] == "succeeded"


def test_a_failed_skill_is_recorded_as_a_failure(world):
    run, _ = _run(world, "ExtProbeRead", {"fail": True})
    assert run.success is False
    v1 = world.st.recent_actions(world.eid)
    assert v1[0]["success"] is False and "probe asked to fail" in v1[0]["error"]
    v2 = world.repos.actions.for_experiment(world.v2eid)
    assert v2[0]["status"] == "failed"


def test_without_an_active_experiment_the_record_says_so(world):
    _LOG.current_experiment_id = None
    run, _ = _run(world, "ExtProbeRead", {})
    assert run.success
    assert run.recorded.get("v1") is False, (
        "没有活动实验时 v1 行插不进去（外键），回执必须如实说出来")
    assert run.recorded.get("experiment_id") is None


# ─────────────────────────────────────────────────────────────────────
# 2. 不双记、不串账
# ─────────────────────────────────────────────────────────────────────

def test_one_direct_call_leaves_exactly_one_map_marker(world):
    _run(world, "ExtProbeRead", {"x_m": 1e-9})
    assert len(world.rt.markers) == 1, (
        f"地图标记 {len(world.rt.markers)} 条 —— 顶层经 run() 已由 marker_sink 记过，"
        "入账这一侧再记就是双记")


def test_a_direct_call_does_not_touch_the_group_run_ledger(world):
    _run(world, "ExtProbeRead", {})
    assert world.rt.ledger == [], "外部直调被记进了群聊 run 的核对台账"


def test_the_direct_path_never_claims_human_authorisation(world):
    ec, _ = direct_exec.build_context(world.ctx, owner="外部 agent ext:tester")
    assert ec._approval_source != "human"


# ─────────────────────────────────────────────────────────────────────
# 3. run_id
# ─────────────────────────────────────────────────────────────────────

def test_every_direct_call_gets_its_own_run_id(world):
    ec1, _ = direct_exec.build_context(world.ctx, owner="a")
    ec2, _ = direct_exec.build_context(world.ctx, owner="a")
    assert ec1.run_id and ec2.run_id and ec1.run_id != ec2.run_id, (
        "空 / 相同的 run_id ⇒ composite sidecar 按名字续跑（假成功的形状）")


# ─────────────────────────────────────────────────────────────────────
# 4. 门口门控
# ─────────────────────────────────────────────────────────────────────

def test_the_sample_gate_refuses_at_the_door_and_the_refusal_is_recorded(world):
    _LOG.current_sample_id = None
    run, _ = _run(world, "ExtProbeScan", {})
    assert run.refused_by == "sample_gate"
    assert RAN == [], "门口拒了还执行了"
    assert world.st.recent_actions(world.eid)[0]["success"] is False


def test_sub_steps_inherit_the_admission_instead_of_being_regated(world):
    _LOG.current_sample_id = "S1"
    run, _ = _run(world, "ExtProbeParent", {})
    names = [n for n, _ in RAN]
    assert names == ["ExtProbeParent", "ExtProbeScan"], (
        f"子步被逐步重判了门控：{names} / {run.error}")
    assert run.success


def test_a_broken_gate_opens_the_door_but_does_not_admit_the_sub_steps(world, monkeypatch):
    """门控自己抛异常：门口放行（坏掉的门控不许锁死仪器），但**不**置 ``_scope_admitted`` ——
    否则一次异常同时放开门口与 ``ExecutionContext.run`` 里那一层（安全审计第 7 条）。"""
    import mast.core.sample_gate as sg

    def _boom(*a, **k):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(sg, "check_sample_scope", _boom)
    _LOG.current_sample_id = "S1"
    run, ec = _run(world, "ExtProbeRead", {})
    assert run.refused_by != "sample_gate", run.error
    assert getattr(ec, "_scope_admitted", False) is False


def test_a_passed_gate_admits_the_sub_steps(world):
    _LOG.current_sample_id = "S1"
    run, ec = _run(world, "ExtProbeRead", {})
    assert run.success and ec._scope_admitted is True


# ─────────────────────────────────────────────────────────────────────
# 5. SI 字符串
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("given", ["5n", 5e-9])
def test_prefixed_strings_and_floats_both_reach_the_skill_as_floats(world, given):
    run, _ = _run(world, "ExtProbeRead", {"x_m": given})
    assert run.success, run.error
    assert RAN[-1][1]["x_m"] == pytest.approx(5e-9)


def test_an_unparseable_quantity_is_refused_at_the_door(world):
    run, _ = _run(world, "ExtProbeRead", {"x_m": "five nanometres"})
    assert run.refused_by == "si_parse"
    assert RAN == []
    assert world.st.recent_actions(world.eid)[0]["success"] is False


# ─────────────────────────────────────────────────────────────────────
# 其余：GraphInterrupt、仪器占用、载荷键集
# ─────────────────────────────────────────────────────────────────────

def test_a_human_node_becomes_a_readable_failure_not_an_exception(world):
    run, _ = _run(world, "ExtProbeHuman", {})
    assert run.refused_by == "needs_human_node"
    assert run.success is False and "human" in run.error


def test_a_busy_instrument_is_refused_not_queued(world):
    from mast.core.instrument_lock import instrument_lock

    held, release = threading.Event(), threading.Event()

    def _holder():
        with instrument_lock().hold(owner="别的入口", skill="LongScan"):
            held.set()
            release.wait(30)

    _LOG.current_sample_id = "S1"
    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    assert held.wait(5)
    try:
        run, _ = _run(world, "ExtProbeScan", {})
    finally:
        release.set()
        t.join(10)
    assert run.refused_by == "busy"
    assert (run.busy_holder or {}).get("owner") == "别的入口"
    assert ("ExtProbeScan", {}) not in RAN


def test_the_record_payload_is_a_superset_of_the_agent_paths_payload():
    """``wrap_skill`` 交给 recorder 的键，直调载荷一个都不能少 —— 否则那一列在
    直调路径上永远是空的，而没有任何东西会报错（这个仓库记过不止一次）。"""
    src = (Path(_MASTV2_ROOT) / "mast" / "agents" / "_shared" / "skill_adapter.py"
           ).read_text(encoding="utf-8")
    keys: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "recorder" and node.args
                and isinstance(node.args[0], ast.Dict)):
            keys |= {k.value for k in node.args[0].keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    assert len(keys) >= 10, f"没扫到 wrap_skill 的 recorder 载荷（扫描器空转？）：{keys}"
    ours = set(direct_exec.record_payload(
        "X", {}, None, SkillResult(skill_name="X", success=True),
        duration_ms=0).keys())
    missing = keys - ours
    assert not missing, f"直调入账载荷缺这些键：{sorted(missing)}"


# ─────────────────────────────────────────────────────────────────────
# 老端点：接上入账之后，响应形状不变，身份按请求头记
# ─────────────────────────────────────────────────────────────────────

def test_the_old_endpoint_now_records_and_takes_an_optional_actor_header(world):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mast.api.routes.skill_exec import router

    api = FastAPI()
    api.state.ctx = world.ctx
    api.include_router(router, prefix="/api")
    c = TestClient(api)
    body = c.post("/api/skills/ExtProbeRead/execute", json={"params": {}}).json()
    assert body["ok"] is True and body["success"] is True
    body = c.post("/api/skills/ExtProbeRead/execute", json={"params": {}},
                  headers={"X-MAST-Actor": "Night Watch"}).json()
    assert body["success"] is True
    ctxs = [r["context"] for r in world.st.recent_actions(world.eid)]
    assert ctxs == ["ext:night-watch", "技能直调 API"], ctxs
    v2 = {r["agent_id"] for r in world.repos.actions.for_experiment(world.v2eid)}
    assert v2 == {"ext:night-watch", "skill_exec_api"}
