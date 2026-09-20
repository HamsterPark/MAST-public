"""持续维护最佳发表候选帧的 conduct 模板。

以成像为主线，质量不足时绕道修针，TrackBestFrame 持久化当前最佳结果与
连续未改善次数。针尖指标决定是否值得采帧，图像比较决定是否更新候选。

已知校验限制：TIP 段未先退针，因此 detour_without_retract 与
detour_writes_before_retract 会拒批。原位优化与保护性恢复需要不同前置
条件；在引擎和校验器明确区分之前，本模板保持 approvable=False。
不能添加形式上的退针步骤仅为绕过校验。等待操作员的路径独立负责确认式
退针；普通质量不足进入 detour，持续无改善由绕道次数上限限制。"""

from mast.conduct.spec import (
    ConductBudget,
    ConductSpec,
    DetourPolicy,
    EvidenceSpec,
    GateOutcome,
    GateSpec,
    ParamSpec,
    RuleLeaf,
    StageFailPolicy,
    StageSpec,
    StepSpec,
)

PARAMS = (
    ParamSpec(
        name="hunt_tag", type="str",
        help="这一轮追猎的标识（建议用 conduct_id 或实验 id）。"
             "「目前最好的那张」按它分开记 —— 两轮追猎共用一个记录会互相污染。"),
    ParamSpec(
        name="dry_limit", type="int", min_value=1, max_value=100,
        help="连着几张没能超过现有最好的，就算「差不多了」，收手。"
             "这是「差不多就放弃继续修」的机器表达 —— 它不假装知道「多好才够」。"),
    ParamSpec(
        name="frame_line_time_s", type="float", unit="s",
        min_value=0.05, max_value=120.0,
        help="发表帧每线时间（慢扫）。512 线 × 3 s ≈ 51 分钟，就是那「一小时」。"),
    ParamSpec(
        name="frame_pixels", type="int", min_value=64, max_value=4096,
        help="发表帧像素数。"),
    ParamSpec(
        name="settle_s", type="float", unit="s", min_value=0.0, max_value=3600.0,
        help="在扫描起点静置多久 —— 让压电蠕变衰减在**扫描开始之前**，"
             "而不是被记录进前几行。"),
    ParamSpec(
        name="tip_time_budget_min", type="float", unit="min",
        min_value=1.0, max_value=1440.0,
        help="每一次「修针」阶段的时间上限。修不出来就回来，由外层决定要不要再修。"),
)


# ── 绕道目标：修针段 ────────────────────────────────────────────────
# 正常流程一步都不进 —— 只有出口闸判「还没够」时才绕进来。
TIP = StageSpec(
    stage_id="TIP",
    title="修针：把针尖修到有原子分辨（分钟级，内部自己反复重试）",
    entry_actions=(),
    steps=(
        StepSpec(
            step_id="TIP.00_condition", kind="composite",
            skill="MakeAtomicResolutionTip",
            title="偏压打磨 + 线级判读 + 不成就扎针换地方",
            # 时间预算由参数给 —— 修针这一层的「多久算够」是外层的事，
            # 不该由模板拍一个数。
            bindings={"time_budget_min": "params.tip_time_budget_min"},
            produces=("outcome",),
            timeout_s=7200.0),
    ),
    # 这一段没有出口闸：修得怎么样由**下一轮出图那道闸**去判 ——
    # 让修针段自己判「修好了没有」，等于让做事的人给自己打分。
    # 失败去向是 **abort 而不是 wait_operator**，理由是一个只有读引擎才知道的
    # 事实：``director._confirmed_retract`` 只挂在**中止序列**上（``director.py``
    # 2170 行），而进 ``wait_operator`` **不退针**。这一段失败时人可能几小时后才
    # 来，针不该在隧道结上等他。
    #
    # 注意「没修出原子分辨」**不是**失败：``MakeAtomicResolutionTip`` 那时候仍然
    # ``success=True``（本仓的老规矩：「没锻成」写在 outcome 里，不表达成技能失败）。
    # 所以这条 abort 只在真出错时才触发。
    on_fail=StageFailPolicy(max_retries=0, then="abort"),
    mandatory=False,
    allowed_escalations=frozenset({"continue_retry", "wait_operator"}),
    capabilities=frozenset({"tip_shaping"}),
)


# ── 主线：出图 ──────────────────────────────────────────────────────
IMG = StageSpec(
    stage_id="IMG",
    title="出一张发表级图，并跟目前最好的那张比",
    entry_actions=(
        StepSpec(
            step_id="IMG.00_peek", kind="skill", skill="TrackBestFrame",
            title="现在最好的是哪张、连着几轮没更好、这一轮门槛多少（只读）",
            bindings={"tag": "params.hunt_tag",
                      "dry_limit": "params.dry_limit"},
            params={"peek": True},
            produces=("best_path", "best_quality", "dry_rounds",
                      "good_enough_to_stop", "gate_floor")),
    ),
    steps=(
        StepSpec(
            step_id="IMG.01_frame", kind="composite",
            skill="ScanPublicationFrame",
            title="慢扫 + 精调平 + 加分辨率 + 起点静置 + 扫一张（约一小时）",
            # 门槛从 peek 那一步来 —— **随 incumbent 上移**。
            # 这个技能自己会因为针尖不够好而拒绝（outcome=tip_not_good_enough），
            # 那不是失败，是这条流程的正常分支：回去接着修。
            bindings={"min_angular_concentration": "steps.IMG.00_peek.gate_floor",
                      "line_time_s": "params.frame_line_time_s",
                      "pixels": "params.frame_pixels",
                      "settle_s": "params.settle_s"},
            produces=("outcome", "frame_path", "angular_concentration"),
            timeout_s=10800.0),
        StepSpec(
            step_id="IMG.02_track", kind="skill", skill="TrackBestFrame",
            title="记下这一张：它更好吗？连着几轮没更好了？",
            # ``angular_concentration`` 缺席（针尖没过闸、没出图）时，
            # TrackBestFrame 会把它当成「判不了」而**不推进 dry 计数** ——
            # 「没出成图」和「出了图但不够好」是两件事，只有后者才算一轮徒劳。
            bindings={"tag": "params.hunt_tag",
                      "frame_path": "steps.IMG.01_frame.frame_path",
                      "quality": "steps.IMG.01_frame.angular_concentration",
                      "dry_limit": "params.dry_limit"},
            produces=("is_best", "best_path", "best_quality",
                      "dry_rounds", "good_enough_to_stop")),
    ),
    exit_gate=GateSpec(
        gate_id="IMG_enough_gate",
        kind="rule",
        evidence=(EvidenceSpec(source="step_data", selector="IMG.02_track",
                               max_age_s=86400.0 * 7, min_epoch="current"),),
        rule=RuleLeaf("good_enough_to_stop", "==", True),
        routes={
            "pass": GateOutcome(
                "pass", "连着若干张都没能更好 —— 差不多了，交出最好的那一张。"),
            # **「还没够」是正常状态，不是异常。** 所以去向是绕道回去接着修，
            # 不是 wait_operator。要叫人的是 max_detours_per_conduct 那道熔断。
            "fail": GateOutcome(
                "detour", "还能更好 —— 回去接着修，修完再来一张。"),
        },
        # 无人值守时判不了 ⇒ 停下来问人，**不要**默认「够了」也不要默认「接着烧」。
        unattended_escape="wait_operator",
        evidence_missing="wait_operator",
    ),
    on_fail=StageFailPolicy(max_retries=1, then="escalate"),
    mandatory=True,
    allowed_escalations=frozenset({"continue_retry", "detour", "wait_operator"}),
    capabilities=frozenset(),
)


SPEC = ConductSpec(
    spec_id="paper_frame_v1",
    spec_version=1,
    title="为了一张能放进论文里的图（多天）",
    stages=(IMG, TIP),
    detour=DetourPolicy(
        target_stage="TIP",
        triggers=frozenset({"gate_verdict", "escalation_approved"}),
        # 出厂 3 是给「修针 ping-pong」用的熔断；这条流程**本来就要绕很多次**
        # （每一轮「修 → 出图」就是一次绕道），所以放宽到几天的量级。
        # 真正的收手判据是 dry_limit（连着几张没更好），不是绕了几次 ——
        # 后者只是兜底：dry 计数万一因为「判不了」一直不推进，这道熔断把人叫来。
        max_detours_per_conduct=24,
        on_return="gate_recheck",
    ),
    budgets=ConductBudget(usd_max=20.0, llm_wakes_per_stage_max=2,
                           tick_interval_s=15.0, wait_tick_interval_s=300.0),
    attended_default=True,
    auto_resume_after_recovery=True,
    params_schema=PARAMS,
)

__all__ = ["SPEC", "PARAMS", "IMG", "TIP"]
