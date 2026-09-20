"""冒烟模板：验证批准、执行、等待、过期读数、恢复与中止链路。

只使用已注册的只读或退针技能与纯分析函数。离线结构检查与带注册表的
可执行性检查分别由 ok 和 complete 表达。等待前执行确认式退针；
温度门槛和点位来自调用参数，模板不填入目标仪器的实验值。"""

from mast.conduct.spec import (
    ConductBudget,
    ConductSpec,
    ConditionSpec,
    DetourPolicy,
    EvidenceSpec,
    GateOutcome,
    GateSpec,
    ParamSpec,
    RuleLeaf,
    StageFailPolicy,
    StageSpec,
    StepSpec,
    WaitSpec,
)

PARAMS = (
    ParamSpec(
        name="target_temperature_k", type="float", unit="K",
        min_value=0.0, max_value=400.0,
        help="等到多少开尔文以下。冒烟跑填一个当前肯定满足的值即可。"),
    ParamSpec(
        name="temperature_stale_after_s", type="float", unit="s",
        min_value=1.0, max_value=86400.0,
        help="温度读数多旧算「读不到」。停掉温度采集程序验告警时靠它。"),
    ParamSpec(
        name="probe_positions_m", type="str",
        help="取谱点位,分号分点、逗号分 xy,单位米,如 '1e-8,2e-8; -3e-8,0'。"
             "冒烟不真的去取谱,只验参数一路传到分析步。"),
    ParamSpec(
        name="probe_points_max", type="int", min_value=1, max_value=64,
        help="点位数上限。"),
)

SMOKE = StageSpec(
    stage_id="SMOKE",
    title="链路冒烟(无针尖风险)",
    entry_actions=(
        StepSpec(step_id="SMOKE.00_temp", kind="skill", skill="GetTemperature",
                 title="重申:先看一眼温度口通不通"),
    ),
    steps=(
        # 等待之前先把针拿开 —— 校验器规则①要的就是这一步。
        StepSpec(step_id="SMOKE.01_retract", kind="skill", skill="SafeRetract",
                 title="确认式退针(等待的前提)"),
        StepSpec(step_id="SMOKE.02_points", kind="analysis",
                 analysis_fn="plan_sts_points",
                 title="整理点位(纯函数,不发明坐标)",
                 bindings={"positions_m": "params.probe_positions_m",
                           "n_points": "params.probe_points_max"},
                 produces=("positions_json", "n_points", "truncated")),
        StepSpec(step_id="SMOKE.03_wait", kind="wait",
                 title="等温度到位(验 stale 告警就停这里)",
                 wait=WaitSpec(
                     kind="condition",
                     message="等温度到位。停掉占着温度口的采集程序,"
                             "应当在 stale_after 之后降级为「要人来看」。",
                     condition=ConditionSpec(
                         signal="temperature_k", op="<=",
                         # 阈值与 stale 窗口都来自 params —— 模板不写死物理量。
                         # 下面两个字面值只是**占位**:``*_ref`` 一给,进等待
                         # 那一刻就按用户填的参数解析,取不到直接判步失败
                         # (不兜底 —— 拿占位值去等,等的就不是人填的条件了)。
                         value=400.0, stale_after_s=120.0, hold_s=0.0,
                         value_ref="params.target_temperature_k",
                         stale_after_ref="params.temperature_stale_after_s",
                         desc="温度 ≤ 目标值"),
                     renotify_every_s=3600.0)),
        StepSpec(step_id="SMOKE.04_temp", kind="skill", skill="GetTemperature",
                 title="等待解除后再读一次温度",
                 produces=()),
    ),
    exit_gate=GateSpec(
        gate_id="SMOKE_gate",
        kind="rule",
        evidence=(EvidenceSpec(source="step_data", selector="SMOKE.02_points",
                               max_age_s=86400.0, min_epoch="current"),),
        rule=RuleLeaf("n_points", ">=", 1),
        routes={"pass": GateOutcome("pass", "点位整理出来了,链路通"),
                "fail": GateOutcome("wait_operator",
                                    "一个点位都没整理出来,请人看参数")},
        unattended_escape="wait_operator",
        evidence_missing="wait_operator",
    ),

    # 冒烟模板显式覆盖 escalate 的真实装配路径，而不仅依赖引擎替身测试。
    # 授权范围仅包含重试和请求人工处理；mandatory 阶段不能跳过，也没有修针绕行。
    # 其他模板需按各自授权与验收策略决定是否启用。
    on_fail=StageFailPolicy(max_retries=0, then="escalate"),
    mandatory=True,
    allowed_escalations=frozenset({"continue_retry", "wait_operator"}),
    capabilities=frozenset(),
)

SPEC = ConductSpec(
    spec_id="_smoke_v1",
    spec_version=1,
    title="链路冒烟(无针尖风险)",
    stages=(SMOKE,),
    detour=DetourPolicy(target_stage="", triggers=frozenset()),
    budgets=ConductBudget(usd_max=1.0, llm_wakes_per_stage_max=0,
                           tick_interval_s=5.0, wait_tick_interval_s=60.0),
    attended_default=True,
    auto_resume_after_recovery=False,
    params_schema=PARAMS,
)

__all__ = ["SPEC", "PARAMS", "SMOKE"]
