"""Director 测试的共用替身与建造器。

**不是 conftest**:conftest 是共享文件,本批次里别人也在动;一个普通模块由本目录
自己 import,谁都不碰谁。

替身的纪律(§9):
* 时钟注入,拨钟测 hold_s / stale_after / renotify,不用 sleep;
* 假 executor 按脚本回答,``StepOutcome`` 逐字段模拟真实形状;
* 「什么都没有」的替身报「读不到」,**不报一切正常**。
"""
from __future__ import annotations

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

import ast
from dataclasses import dataclass, field

from mast.conduct.director import ConductDirector
from mast.conduct.ports import LatchState, LoggingNotifier, StepOutcome
from mast.conduct.spec import (
    ConductBudget,
    ConductSpec,
    ConditionSpec,
    DetourPolicy,
    EvidenceSpec,
    GateOutcome,
    GateSpec,
    RecoveryPolicy,
    RuleLeaf,
    StageFailPolicy,
    StageSpec,
    StepSpec,
    WaitSpec,
)
from mast.conduct.store import ConductStore

RETRACT = "SafeRetract"


class FakeClock:
    def __init__(self, t0: float = 1_700_000_000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeExecutor:
    """按技能名或调用序号回答。

    ``script`` 是 ``{skill: [outcome, ...]}``,用完最后一个就一直用它;
    没写的技能默认成功。
    """

    def __init__(self, script: "dict | None" = None):
        self.script = dict(script or {})
        self.calls: list[tuple[str, dict, str]] = []

    def run(self, skill: str, params: dict, *, run_id: str) -> StepOutcome:
        self.calls.append((skill, dict(params), run_id))
        seq = self.script.get(skill)
        if not seq:
            return StepOutcome(ok=True, run_id=run_id, data={})
        idx = min(len([c for c in self.calls if c[0] == skill]) - 1, len(seq) - 1)
        out = seq[idx]
        return StepOutcome(ok=out.ok, busy=out.busy, tip_event=out.tip_event,
                           crash=out.crash, error=out.error, data=dict(out.data),
                           run_id=run_id)

    def skills_called(self) -> list[str]:
        return [c[0] for c in self.calls]


class HangingExecutor(FakeExecutor):
    """跑一步就不返回 —— 用来验「Director 不做超时杀步」这条诚实短板。

    真正的挂起没法在单测里等,所以这里只暴露「调用发生了、还没返回」这个事实,
    由测试驱动。
    """

    def __init__(self):
        super().__init__()
        self.entered = 0

    def run(self, skill, params, *, run_id):
        self.entered += 1
        raise TimeoutError("模拟一次卡死的 TCP 事务")


class FakeLatch:
    def __init__(self, latched: bool = False, why: str = ""):
        self.latched = latched
        self.why = why
        self.raises = False

    def state(self):
        if self.raises:
            raise RuntimeError("闩读不到")
        return LatchState(latched=self.latched, abort_set=self.latched,
                          why=self.why)


@dataclass
class FakeReading:
    """模仿 ``core.temperature.TempReading`` 的形状(含三态 freshness)。"""

    value_k: "float | None" = None
    age_s: "float | None" = None
    reason: "str | None" = None
    channel: str = "SPM"
    source: str = "fake"

    def freshness(self, max_age_s: float) -> str:
        if self.value_k is None or self.age_s is None:
            return "unknown"
        return "stale" if float(self.age_s) > float(max_age_s) else "fresh"


class FakeTemperature:
    def __init__(self, reading: "FakeReading | None" = None):
        self.reading = reading or FakeReading(reason="no_sensor")
        self.reads = 0

    def read(self, channel: str = ""):
        self.reads += 1
        return self.reading


class FakeRecoveryProbe:
    def __init__(self, verdicts: "dict | None" = None, default: str = "pass"):
        self.verdicts = dict(verdicts or {})
        self.default = default
        self.asked: list[str] = []

    def __call__(self, item: str) -> str:
        self.asked.append(item)
        return self.verdicts.get(item, self.default)


# ── spec 建造器 ─────────────────────────────────────────────────────

def step(step_id: str, skill: str = "ScanAt", **kw) -> StepSpec:
    kw.setdefault("kind", "skill")
    kw.setdefault("skill", skill)
    return StepSpec(step_id=step_id, **kw)


def retract_step(step_id: str = "R.00") -> StepSpec:
    return StepSpec(step_id=step_id, kind="skill", skill=RETRACT)


def wait_step(step_id: str = "W.01", *, kind: str = "operator",
              condition: "ConditionSpec | None" = None, **kw) -> StepSpec:
    return StepSpec(step_id=step_id, kind="wait",
                    wait=WaitSpec(kind=kind, message="请处理",
                                  condition=condition, **kw))


def rule_gate(gate_id: str = "g", *, field_name: str = "n", op: str = ">=",
              value=1, selector: str = "S.01", max_age_s: float = 3600.0,
              fail_verdict: str = "wait_operator") -> GateSpec:
    return GateSpec(
        gate_id=gate_id, kind="rule",
        evidence=(EvidenceSpec(source="step_data", selector=selector,
                               max_age_s=max_age_s, min_epoch="current"),),
        rule=RuleLeaf(field_name, op, value),
        routes={"pass": GateOutcome("pass"),
                "fail": GateOutcome(fail_verdict, "判定不通过")},
        unattended_escape="wait_operator", evidence_missing="wait_operator")


def stage(stage_id: str = "S", steps=(), **kw) -> StageSpec:
    kw.setdefault("title", stage_id)
    return StageSpec(stage_id=stage_id, steps=tuple(steps), **kw)


def spec(stages, *, detour=None, params=(), spec_id="t_v1", version=1,
         auto_resume=False, budgets=None, recovery=None) -> ConductSpec:
    return ConductSpec(
        spec_id=spec_id, spec_version=version, title="t", stages=tuple(stages),
        detour=detour or DetourPolicy(target_stage="", triggers=frozenset()),
        budgets=budgets or ConductBudget(),
        auto_resume_after_recovery=auto_resume,
        params_schema=tuple(params),
        recovery=recovery or RecoveryPolicy())


def tip_check(skill: str = "PreScanCheck", *, rule=None, retries: int = 1,
              steps=None) -> RecoveryPolicy:
    """一份**声明了**恢复期针尖复验的策略。

    默认一步 ``PreScanCheck`` 产 ``tip_ready``,判据 ``tip_ready == True``
    —— 与 ``synthetic_sample`` 同形状,但技能名与产出都由测试说了算。
    """
    steps = steps if steps is not None else (
        StepSpec(step_id="R.00_check", kind="composite", skill=skill,
                 produces=("tip_ready",)),)
    return RecoveryPolicy(tip_check=tuple(steps),
                          tip_rule=rule or RuleLeaf("tip_ready", "==", True),
                          tip_retries=retries)


# ── 组装 ────────────────────────────────────────────────────────────

@dataclass
class Rig:
    store: ConductStore
    director: ConductDirector
    clock: FakeClock
    executor: FakeExecutor
    latch: FakeLatch
    temperature: FakeTemperature
    notifier: LoggingNotifier
    conduct_id: str
    spec: ConductSpec

    def tick(self, n: int = 1):
        rep = None
        for _ in range(n):
            rep = self.director.step_tick()
        return rep

    def run_until(self, status: str, *, limit: int = 40):
        """一直 tick 到状态到达(或 limit)。返回最后一次报告。"""
        rep = None
        for _ in range(limit):
            rep = self.director.step_tick()
            if self.row()["status"] == status:
                return rep
        return rep

    def row(self) -> dict:
        return self.store.get(self.conduct_id)

    def events(self, kind: "str | None" = None):
        return self.store.events(self.conduct_id, kind=kind, limit=2000)

    def event_kinds(self) -> list[str]:
        return [e["kind"] for e in self.events()]


def build(tmp_path, conduct_spec: ConductSpec, *, params: "dict | None" = None,
          executor: "FakeExecutor | None" = None, attended: bool = True,
          approve: bool = True, temperature=None, recovery_probe=None,
          decide_route=None, skill_meta=None, cost_reader=None,
          lock_probe=None, analyses_get=None, monitor_probe=None,
          frame_metrics_probe=None, link_probe=None,
          escalation_advisor=None) -> Rig:
    clock = FakeClock()
    store = ConductStore(tmp_path / "conduct.db", clock=clock)
    cid = store.create(experiment_id="exp-1", spec_id=conduct_spec.spec_id,
                       spec_version=conduct_spec.spec_version,
                       params=params or {}, attended=attended)
    if approve:
        store.record(cid, "approved", changes={"status": "approved",
                                               "approved_by": "operator"})
    ex = executor or FakeExecutor()
    latch = FakeLatch()
    temp = temperature or FakeTemperature()
    notifier = LoggingNotifier()
    director = ConductDirector(
        store, spec_provider=lambda sid: conduct_spec, executor=ex,
        latch=latch, temperature=temp, notifier=notifier, clock=clock,
        recovery_probe=recovery_probe, decide_route=decide_route,
        skill_meta=skill_meta, cost_reader=cost_reader, lock_probe=lock_probe,
        analyses_get=analyses_get, monitor_probe=monitor_probe,
        frame_metrics_probe=frame_metrics_probe, link_probe=link_probe,
        escalation_advisor=escalation_advisor)
    return Rig(store=store, director=director, clock=clock, executor=ex,
               latch=latch, temperature=temp, notifier=notifier,
               conduct_id=cid, spec=conduct_spec)


def outcome(ok=True, **kw) -> StepOutcome:
    return StepOutcome(ok=ok, **kw)


# ── 源码级断言的取源工具 ──────────────────────────────────────────────

# ``source_of`` **实现不在这里** —— 它搬到了 ``tests/v2/srcref.py``(零依赖,
# 只用 stdlib),因为需要它的测试分散在 conduct / skills 好几个目录,而**别的目录
# 不能 import 本模块**:本文件在模块级做 sys.path 手术,还会把 ``sys.modules`` 里
# 所有 ``mast.*`` 清掉再拉进整个 conduct 引擎 —— 从 composite 的测试 import 它,
# 等于在别人跑到一半时炸掉已加载的 ``mast.*`` 和别人打好的 monkeypatch。
#
# 这里留一个 re-export,好让本目录里既有的 import 不必全改。**两处各留一份实现
# 才是要防的那件事**:同一个工具两份实现,迟早只有一份跟着现实走。
#
# 挑工具的那张表(``co_names`` / ``co_consts`` / ``source_of`` 各管哪一半)在
# ``srcref.py`` 的 docstring 里,单一真源。
from tests.v2.srcref import source_of  # noqa: E402,F401
