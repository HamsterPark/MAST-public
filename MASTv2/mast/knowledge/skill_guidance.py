"""Skill expert guidance for LLM consumption.

Provides per-skill metadata (when to use / when not / related skills),
decision trees, and workflow recipes. Shared between GUI (encyclopedia)
and LLM (planner meta-tool).
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════
#  Per-skill extra metadata (what registry doesn't carry)
# ═══════════════════════════════════════════════════════════════════════

SKILL_EXTRA: dict[str, dict] = {
    # -- Bias & Current --
    "GetBias": {
        "when_use": "设置新偏压前读取当前值以备恢复，或确认偏压状态。",
        "when_not": "不需要知道偏压值时无需调用。",
        "related": "后续 SetBias",
        "nanonis": "Bias_Get",
        "example": '{"name": "GetBias", "input": {}}',
    },
    "SetBias": {
        "when_use": "切换成像条件、准备 STS 起始偏压、恢复偏压。",
        "when_not": "Z 控制器关闭且针尖在表面时勿大幅改变。",
        "related": "前序 GetBias / 被 TipPulse 调用",
        "nanonis": "Bias_Set",
        "example": '{"name": "SetBias", "input": {"bias_v": -0.5}}',
    },
    "GetCurrent": {
        "when_use": "监测隧穿状态、判断针尖是否在表面附近。",
        "when_not": "退针时电流读数无意义。",
        "related": "后续 SetSetpoint",
        "nanonis": "Current_Get",
        "example": '{"name": "GetCurrent", "input": {}}',
    },
    # -- Z Control --
    "GetZPosition": {
        "when_use": "监测 Z 位置判断针尖与表面距离。",
        "when_not": "STS 采谱期间不应频繁读取。",
        "related": "后续 SetSetpoint",
        "nanonis": "ZCtrl_ZPosGet",
        "example": '{"name": "GetZPosition", "input": {}}',
    },
    "SetSetpoint": {
        "when_use": "调整成像条件（更高 setpoint = 更强信号但风险更高）。",
        "when_not": "STS 采谱期间不应改变。",
        "related": "被 OptimizeResolution_BO 调用",
        "nanonis": "ZCtrl_SetpntSet",
        "example": '{"name": "SetSetpoint", "input": {"setpoint_a": 100e-12}}',
    },
    "ZControllerOnOff": {
        "when_use": "STS 前关闭反馈、扫描前确保反馈开启。",
        "when_not": "长时间关闭可能导致漂移碰撞。",
        "related": "被多个技能依赖为前置条件",
        "nanonis": "ZCtrl_OnOffSet",
        "example": '{"name": "ZControllerOnOff", "input": {"enable": true}}',
    },
    # -- Scanning --
    "ConfigureScan": {
        "when_use": "扫描前设定区域和通道。替代: 用 FullScan 一步完成。",
        "when_not": "只需重新扫描当前区域时直接调 StartScan。",
        "related": "后续 SetScanSpeed, StartScan / 替代 FullScan",
        "nanonis": "Scan_FrameSet, Signals_InSlotsGet, Scan_BufferSet",
        "example": '{"name": "ConfigureScan", "input": {"center_x_m": 0, "center_y_m": 0, "width_m": 100e-9, "height_m": 100e-9}}',
    },
    "SetScanSpeed": {
        "when_use": "调整扫描速度以平衡分辨率和采集时间。",
        "when_not": "使用 FullScan 时由其内部调用。",
        "related": "被 FullScan 内部调用",
        "nanonis": "Scan_SpeedSet",
        "example": '{"name": "SetScanSpeed", "input": {"fwd_speed": 200e-9, "bwd_speed": 200e-9, "fwd_line_time": 0.5, "bwd_line_time": 0.5}}',
    },
    "StartScan": {
        "when_use": "ConfigureScan + SetScanSpeed 后启动扫描。",
        "when_not": "使用 FullScan 时由其内部调用。",
        "related": "回滚 StopScan / 替代 FullScan",
        "nanonis": "Scan_PropsGet, Scan_PropsSet, Scan_Action",
        "example": '{"name": "StartScan", "input": {}}',
    },
    "StopScan": {
        "when_use": "紧急停止扫描。",
        "when_not": "扫描未在进行时无需调用。",
        "related": "前序 StartScan",
        "nanonis": "Scan_Action",
        "example": '{"name": "StopScan", "input": {}}',
    },
    "FullScan": {
        "when_use": "一步完成 ConfigureScan + SetScanSpeed + StartScan。",
        "when_not": "只需改速度不改区域时单独调 SetScanSpeed + StartScan。",
        "related": "分解: ConfigureScan + SetScanSpeed + StartScan",
        "nanonis": "(组合技能)",
        "example": '{"name": "FullScan", "input": {"center_x_m": 0, "center_y_m": 0, "width_m": 50e-9, "height_m": 50e-9}}',
    },
    # -- Navigation --
    "MoveToXY": {
        "when_use": "STS 前移到目标位置、网格逐点移动。",
        "when_not": "Z 控制器关闭时不可移动。",
        "related": "被 GridSTS, AdaptiveSTS_GP, AtomManip_SAC 调用",
        "nanonis": "FolMe_XYPosSet",
        "example": '{"name": "MoveToXY", "input": {"x_m": 50e-9, "y_m": -30e-9}}',
    },
    # -- Tip Management --
    "SafeRetract": {
        # 这里只有 when_use/when_not/related/nanonis 会被渲染进知识块
        # (chunk_registry._fmt_skill),别加新键 —— 加了不报错,只是永远不出现。
        "when_use": (
            "正常操作中需要缩回针尖时。返回的 `retracted` 是**三态**:True=已回读确认"
            "到位(Z 反馈断开且 Z 停在收回端)、False=读到了但还没到位、"
            "None=判不了(读不到/本机没声明 z_extend_sign)。"
            "要依赖「针已退开」再做别的动作时,判据是 `retracted is True`,"
            "**不是**「这一步没报错」。"
        ),
        "when_not": "紧急情况下用 EmergencyRetract。",
        "related": "替代 EmergencyRetract / 后续 AutoApproach",
        "nanonis": "ZCtrl_Withdraw + 回读 ZCtrl_OnOffGet/ZCtrl_ZPosGet 确认",
        "example": '{"name": "SafeRetract", "input": {}}',
    },
    "EmergencyRetract": {
        "when_use": "异常（电流突增/振动）时紧急退针。使用 TCP 6504 紧急端口。",
        "when_not": "正常操作用 SafeRetract。",
        "related": "替代 SafeRetract",
        "nanonis": "ZCtrl_OnOffSet, Scan_Action, ZCtrl_Withdraw (emergency)",
        "example": '{"name": "EmergencyRetract", "input": {}}',
    },
    "WithdrawTip": {
        "when_use": "AutoApproach 的回滚操作。Withdraw 是静态状态。",
        "when_not": "快速临时退针用 SafeRetract。",
        "related": "是 AutoApproach 的回滚 / 后续 AutoApproach",
        "nanonis": "ZCtrl_Withdraw",
        "example": '{"name": "WithdrawTip", "input": {}}',
    },
    "AutoApproach": {
        "when_use": "退针后重新逼近表面。最危险的内置操作之一。",
        "when_not": "针尖已在表面时不应调用。",
        "related": "回滚 WithdrawTip",
        "nanonis": "AutoApproach_Open, AutoApproach_OnOffSet",
        "example": '{"name": "AutoApproach", "input": {}}',
    },
    "TipPulse": {
        "when_use": "图像质量下降时通过高压脉冲重整针尖。",
        "when_not": "先用 AssessImageQuality 评估再决定。",
        "related": "升级 ConditionTip / 被 ConditionTip 调用",
        "nanonis": "(组合: Bias_Get + Bias_Set)",
        "example": '{"name": "TipPulse", "input": {"pulse_v": 3.0, "count": 3}}',
    },
    "ConditionTip": {
        "when_use": "单次 TipPulse 不足时自动循环修针。",
        "when_not": "先尝试 TipPulse。考虑升级到 ConditionTip_DQN。",
        "related": "降级 TipPulse / 升级 ConditionTip_DQN",
        "nanonis": "(组合: TipPulse + Scan + AssessQuality)",
        "example": '{"name": "ConditionTip", "input": {"pulse_v": 3.0, "max_attempts": 5}}',
    },
    "ConditionTip_DQN": {
        "when_use": "ConditionTip 固定脉冲无法改善时用 DQN 自适应选择参数。",
        "when_not": "简单情况下 TipPulse 或 ConditionTip 已足够。",
        "related": "降级 ConditionTip / 引用: DeepSPM",
        "nanonis": "(组合: TipPulse + Scan + AssessQuality)",
        "example": '{"name": "ConditionTip_DQN", "input": {"strategy": "escalating"}}',
    },
    # -- Spectroscopy --
    "ConfigureSTS": {
        "when_use": "AcquireSTS 前设定偏压范围和点数。",
        "when_not": "使用当前参数重新采谱时直接调 AcquireSTS。",
        "related": "后续 AcquireSTS / 被 GridSTS 调用",
        "nanonis": "BiasSpectr_Open, BiasSpectr_LimitsSet, BiasSpectr_PropsSet",
        "example": '{"name": "ConfigureSTS", "input": {"start_v": -2.0, "end_v": 2.0, "num_points": 500}}',
    },
    "AcquireSTS": {
        "when_use": "ConfigureSTS 设好参数后采谱。",
        "when_not": "大量谱用 GridSTS 或 AdaptiveSTS_GP。",
        "related": "前序 ConfigureSTS / 被 GridSTS 调用",
        "nanonis": "BiasSpectr_Open, BiasSpectr_PropsSet, BiasSpectr_Start",
        "example": '{"name": "AcquireSTS", "input": {}}',
    },
    "ConfigureZSpectr": {
        "when_use": "AcquireZSpectr 前设定 Z 偏移和扫描距离。",
        "when_not": "使用当前参数时直接调 AcquireZSpectr。",
        "related": "后续 AcquireZSpectr",
        "nanonis": "ZSpectr_Open, ZSpectr_RangeSet, ZSpectr_PropsSet",
        "example": '{"name": "ConfigureZSpectr", "input": {"z_offset_m": 0, "z_sweep_distance_m": 1e-9, "num_points": 200}}',
    },
    "AcquireZSpectr": {
        "when_use": "ConfigureZSpectr 后采集 I(z) 曲线。",
        "when_not": "研究电子结构时用 AcquireSTS。",
        "related": "前序 ConfigureZSpectr",
        "nanonis": "ZSpectr_Open, ZSpectr_PropsSet, ZSpectr_Start",
        "example": '{"name": "AcquireZSpectr", "input": {}}',
    },
    "GridSTS": {
        "when_use": "NxN 网格逐点 STS，需要均匀空间分辨时。",
        "when_not": "稀疏高效采样用 AdaptiveSTS_GP。",
        "related": "替代 AdaptiveSTS_GP（均匀 vs 自适应）",
        "nanonis": "(组合: ConfigureSTS + MoveToXY + AcquireSTS)",
        "example": '{"name": "GridSTS", "input": {"center_x_m": 0, "center_y_m": 0, "nx": 5, "ny": 5, "spacing_m": 2e-9}}',
    },
    "AdaptiveSTS_GP": {
        "when_use": "GP 引导选择信息量最大的位置优先采样，减少总测量次数。",
        "when_not": "需要完整均匀覆盖时用 GridSTS。",
        "related": "替代 GridSTS / 引用: Thomas 2022",
        "nanonis": "(组合: MoveToXY + AcquireSTS)",
        "example": '{"name": "AdaptiveSTS_GP", "input": {"center_x_m": 0, "center_y_m": 0, "max_measurements": 20}}',
    },
    # -- Lock-in & Sweep --
    "ConfigureLockIn": {
        "when_use": "需要 dI/dV 信号时开启 lock-in 调制。",
        "when_not": "纯形貌成像不需要。",
        "related": "后续 ConfigureLockInSweep",
        "nanonis": "LockIn_ModOnOffSet, LockIn_ModPhasFreqSet, LockIn_ModAmpSet",
        "example": '{"name": "ConfigureLockIn", "input": {"mod_on": true, "amplitude_v": 0.01, "frequency_hz": 973}}',
    },
    "ConfigureBiasSweep": {
        "when_use": "通过 Generic Sweeper 做偏压扫描。",
        "when_not": "标准 STS 用 ConfigureSTS + AcquireSTS。",
        "related": "后续 AcquireBiasSweep",
        "nanonis": "GenSwp_Open, GenSwp_SwpSignalSet, GenSwp_LimitsSet, GenSwp_PropsSet",
        "example": '{"name": "ConfigureBiasSweep", "input": {"lower_v": -1.0, "upper_v": 1.0, "num_steps": 200}}',
    },
    "AcquireBiasSweep": {
        "when_use": "ConfigureBiasSweep 后执行扫描。",
        "when_not": "标准 STS 用 AcquireSTS。",
        "related": "前序 ConfigureBiasSweep / 替代 AcquireSTS",
        "nanonis": "GenSwp_Open, GenSwp_SwpSignalSet, GenSwp_PropsSet, GenSwp_Start",
        "example": '{"name": "AcquireBiasSweep", "input": {}}',
    },
    "ConfigureLockInSweep": {
        "when_use": "扫描 lock-in 频率响应（如寻找谐振）。",
        "when_not": "已知频率时直接 ConfigureLockIn。",
        "related": "前序 ConfigureLockIn / 后续 AcquireLockInSweep",
        "nanonis": "LockInFreqSwp_Open, LockInFreqSwp_LimitsSet, LockInFreqSwp_PropsSet",
        "example": '{"name": "ConfigureLockInSweep", "input": {"lower_hz": 100, "upper_hz": 2000, "num_steps": 200}}',
    },
    "AcquireLockInSweep": {
        "when_use": "ConfigureLockInSweep 后执行频率扫描。",
        "when_not": "不需要频率响应分析时。",
        "related": "前序 ConfigureLockInSweep",
        "nanonis": "LockInFreqSwp_Open, LockInFreqSwp_PropsSet, LockInFreqSwp_Start",
        "example": '{"name": "AcquireLockInSweep", "input": {}}',
    },
    # -- Data Processing --
    "SubtractPlane_RANSAC": {
        "when_use": "扫描图像有整体倾斜时。RANSAC 排除离群点。",
        "when_not": "图像已经很平时无需扣除。",
        "related": "后续 LevelLines_Median / 引用: DeepSPM",
        "nanonis": "(纯分析)",
        "example": '{"name": "SubtractPlane_RANSAC", "input": {"image_path": "scan_001.sxm"}}',
    },
    "LevelLines_Median": {
        "when_use": "扫描线间有行间噪声。平面扣除后常需行校正。",
        "when_not": "行间已对齐时无需处理。",
        "related": "前序 SubtractPlane_RANSAC / 引用: spym",
        "nanonis": "(纯分析)",
        "example": '{"name": "LevelLines_Median", "input": {"image_path": "scan_001.sxm"}}',
    },
    "CorrectDrift_XCorr": {
        "when_use": "比较两帧图像间的平移漂移。",
        "when_not": "单帧图像无法使用，需要两帧。",
        "related": "前序 SubtractPlane_RANSAC / 引用: scikit-image",
        "nanonis": "(纯分析)",
        "example": '{"name": "CorrectDrift_XCorr", "input": {"ref_path": "s1.sxm", "target_path": "s2.sxm"}}',
    },
    "FindEmptySpot": {
        "when_use": "STS 前需要找到干净无分子区域。",
        "when_not": "已知位置时直接 MoveToXY。",
        "related": "后续 MoveToXY / 被 FindGoodRegion_Heuristic 使用",
        "nanonis": "(纯分析)",
        "example": '{"name": "FindEmptySpot", "input": {"image_path": "scan_001.sxm"}}',
    },
    # -- Image Analysis --
    "AssessImageQuality": {
        "when_use": "扫描后评估图像质量，决定是否需要修针。",
        "when_not": "不操作硬件，可随时安全调用。",
        "related": "后续 ConditionTip / 被修针循环调用",
        "nanonis": "(纯分析: FFT + RMS + Noise)",
        "example": '{"name": "AssessImageQuality", "input": {}}',
    },
    "AssessTip_VGG": {
        "when_use": "VGG CNN 针尖质量评估。有模型用 CNN，否则启发式。",
        "when_not": "只需整体图像质量用 AssessImageQuality。",
        "related": "替代 AssessTip_ResNet / 引用: DeepSPM",
        "nanonis": "(纯分析)",
        "example": '{"name": "AssessTip_VGG", "input": {"image_path": "scan_001.sxm"}}',
    },
    "AssessTip_ResNet": {
        "when_use": "ResNet18 针尖质量评估。与 VGG 选一即可。",
        "when_not": "两者选一，不需要同时调用。",
        "related": "替代 AssessTip_VGG / 引用: Zhu 2024",
        "nanonis": "(纯分析)",
        "example": '{"name": "AssessTip_ResNet", "input": {"image_path": "scan_001.sxm"}}',
    },
    "SegmentRegion_UNet": {
        "when_use": "将图像分割为干净/分子/缺陷区域。",
        "when_not": "只需找最平坦区域用 FindEmptySpot。",
        "related": "被 FindGoodRegion_UNet 使用 / 引用: Zhu 2024",
        "nanonis": "(纯分析)",
        "example": '{"name": "SegmentRegion_UNet", "input": {"image_path": "scan_001.sxm"}}',
    },
    "DetectAtoms_FCN": {
        "when_use": "定位原子位置，用于计数或操纵前定位。",
        "when_not": "分辨率不足以分辨单原子时无法使用。",
        "related": "后续 AtomManip_SAC / 引用: AtomAI",
        "nanonis": "(纯分析)",
        "example": '{"name": "DetectAtoms_FCN", "input": {"image_path": "scan_001.sxm"}}',
    },
    # -- Molecular Recognition --
    "PredictStructure_ASD": {
        "when_use": "从 STM 图像预测分子化学结构。需要预训练模型。",
        "when_not": "无模型时无法运行（没有回退）。",
        "related": "替代 IdentifyTopology_CARP / 引用: Kurki 2024",
        "nanonis": "(纯分析, 需 PyTorch)",
        "example": '{"name": "PredictStructure_ASD", "input": {"image_path": "s.sxm", "model_path": "m.pt"}}',
    },
    "IdentifyTopology_CARP": {
        "when_use": "识别纳米碳环分子拓扑。需要 Detectron2。",
        "when_not": "无模型和 Detectron2 时无法运行。",
        "related": "替代 PredictStructure_ASD / 引用: Su 2024",
        "nanonis": "(纯分析, 需 Detectron2)",
        "example": '{"name": "IdentifyTopology_CARP", "input": {"image_path": "s.sxm", "model_path": "m.pt"}}',
    },
    # -- Spectral Analysis --
    "FitFano_Kondo": {
        "when_use": "dI/dV 谱在费米面附近出现不对称峰时拟合 Kondo 温度。",
        "when_not": "超导能隙拟合用 FitGap_BCS。",
        "related": "前序 AcquireSTS / 引用: HurwitzFanoFit",
        "nanonis": "(纯分析)",
        "example": '{"name": "FitFano_Kondo", "input": {"spectrum_path": "sts.dat"}}',
    },
    "FitGap_BCS": {
        "when_use": "dI/dV 谱呈现超导能隙特征时拟合 BCS 模型。",
        "when_not": "Kondo 共振用 FitFano_Kondo。",
        "related": "前序 AcquireSTS / 引用: stmpy",
        "nanonis": "(纯分析)",
        "example": '{"name": "FitGap_BCS", "input": {"spectrum_path": "sts.dat", "temperature_k": 4.2}}',
    },
    # -- Optimization --
    "OptimizeResolution_BO": {
        "when_use": "手动调参难以获得原子分辨率时自动搜索最优 bias-setpoint。",
        "when_not": "已知最佳参数时直接设定。",
        "related": "使用 SetBias, SetSetpoint, FullScan, AssessImageQuality",
        "nanonis": "(组合: 多步循环)",
        "example": '{"name": "OptimizeResolution_BO", "input": {"n_iterations": 15}}',
    },
    # -- Autonomous --
    "FindGoodRegion_Heuristic": {
        "when_use": "自动搜索干净平坦区域。",
        "when_not": "已知好区域坐标时直接 MoveToXY。",
        "related": "替代 FindGoodRegion_UNet / 被 ContinuousImaging_Auto 使用",
        "nanonis": "(组合: FullScan + FindEmptySpot + MoveToXY)",
        "example": '{"name": "FindGoodRegion_Heuristic", "input": {"survey_width_m": 200e-9}}',
    },
    "FindGoodRegion_UNet": {
        "when_use": "语义级区域搜索。有 U-Net 模型时更准确。",
        "when_not": "无模型时与 FindGoodRegion_Heuristic 效果相同。",
        "related": "替代 FindGoodRegion_Heuristic / 引用: Zhu 2024",
        "nanonis": "(组合: FullScan + SegmentRegion_UNet + MoveToXY)",
        "example": '{"name": "FindGoodRegion_UNet", "input": {"survey_width_m": 200e-9}}',
    },
    "AutonomousSurvey_Scanbot": {
        "when_use": "系统性巡查大范围样品，生成质量映射。",
        "when_not": "只需单区域扫描时用 FullScan。",
        "related": "使用 MoveToXY, FullScan, AssessImageQuality, TipPulse",
        "nanonis": "(组合: 网格遍历)",
        "example": '{"name": "AutonomousSurvey_Scanbot", "input": {"n_areas_x": 3, "n_areas_y": 3}}',
    },
    "ContinuousImaging_Auto": {
        "when_use": "长时间无人值守自主成像闭环。",
        "when_not": "短时间实验或需要人工干预时。",
        "related": "使用 FindGoodRegion, TipPulse, FullScan, AssessImageQuality",
        "nanonis": "(组合: 全自主循环)",
        "example": '{"name": "ContinuousImaging_Auto", "input": {"max_cycles": 10}}',
    },
    # -- Atom Manipulation --
    "AtomManip_SAC": {
        "when_use": "将单个原子从当前位置移动到目标位置。",
        "when_not": "不需要原子级精度操控时。",
        "related": "前序 DetectAtoms_FCN / 引用: Chen 2022",
        "nanonis": "(组合: MoveToXY + SetBias + SetSetpoint)",
        "example": '{"name": "AtomManip_SAC", "input": {"atom_x_m": 1e-9, "atom_y_m": 2e-9, "target_x_m": 3e-9, "target_y_m": 2e-9}}',
    },
    "AutoOSS_Dehalogenation": {
        "when_use": "针尖诱导脱卤反应。不可逆操作。",
        "when_not": "非脱卤反应不适用。",
        "related": "前序 DetectAtoms_FCN / 引用: Wu 2025",
        "nanonis": "(组合: MoveToXY + SetBias + TipPulse)",
        "example": '{"name": "AutoOSS_Dehalogenation", "input": {"target_x_m": 5e-9, "target_y_m": 3e-9}}',
    },
    # -- New Skills (#4-30) --
    "DetectAtomJump": {
        "when_use": "原子操纵动作后判断原子是否移动。替代重新扫描验证。",
        "when_not": "不在操纵模式时无需使用。",
        "related": "被 AtomManip_SAC 调用 / 引用: Chen 2022",
        "nanonis": "(纯分析)",
        "example": '{"name": "DetectAtomJump", "input": {"current_trace": "trace.npy"}}',
    },
    "CheckLineQuality": {
        "when_use": "扫描中检查行质量，判断针尖是否退化。",
        "when_not": "扫描完成后用 AssessImageQuality 更全面。",
        "related": "被 PreScanCheck 调用",
        "nanonis": "(纯分析)",
        "example": '{"name": "CheckLineQuality", "input": {"fwd_lines": "fwd.npy", "bwd_lines": "bwd.npy"}}',
    },
    "Destripe_MorphOpen": {
        "when_use": "图像有水平条纹伪影时。通常在 LevelLines_Median 之后使用。",
        "when_not": "图像无明显条纹时无需处理。",
        "related": "前序 LevelLines_Median / 引用: spym",
        "nanonis": "(纯分析)",
        "example": '{"name": "Destripe_MorphOpen", "input": {"image_path": "scan.npy"}}',
    },
    "CorrectDrift_BraggPeak": {
        "when_use": "原子分辨图像有明显剪切或拉伸时。单帧内晶格畸变校正。",
        "when_not": "两帧之间的刚性位移用 CorrectDrift_XCorr。",
        "related": "替代 CorrectDrift_XCorr（单帧 vs 双帧） / 引用: stmpy",
        "nanonis": "(纯分析)",
        "example": '{"name": "CorrectDrift_BraggPeak", "input": {"image_path": "scan.npy"}}',
    },
    "SubtractPoly2D": {
        "when_use": "当 SubtractPlane_RANSAC 后图像仍有明显弯曲时使用。",
        "when_not": "平坦衬底用 SubtractPlane_RANSAC 即可。",
        "related": "替代/补充 SubtractPlane_RANSAC（高阶 vs 1阶）",
        "nanonis": "(纯分析)",
        "example": '{"name": "SubtractPoly2D", "input": {"image_path": "scan.npy", "order_x": 2, "order_y": 2}}',
    },
    "DeconvolveTip_RL": {
        "when_use": "图像被针尖卷积模糊，需要恢复锐利特征时。",
        "when_not": "针尖已很尖锐时无需反卷积。",
        "related": "后续 SubtractPlane 系列 / 引用: pySPM",
        "nanonis": "(纯分析)",
        "example": '{"name": "DeconvolveTip_RL", "input": {"image_path": "scan.npy", "psf_sigma": 2.0}}',
    },
    "MotorMove": {
        # 横向移动请用 RelocateCoarseXY —— 自主路径上 MotorMove 的 x±/y± 已被
        # Layer-0 挡住并要求人工审批(见 mast.core.safety.
        # is_unguarded_lateral_coarse_move)。这里留下的用法只剩 z-retract。
        "when_use": "粗动马达单步原语。**横向换区请用 RelocateCoarseXY,不要用它。**"
                    "z-retract(远离样品)可以直接用。",
        "when_not": "x±/y± 横向移动:自主路径会被拒绝,因为它只检查压电是否收到顶"
                    "(约 1 µm 余量,读不到状态还会放行),不做粗动退针清障、不看真空、"
                    "不核对驱动电压、移动中不看电流、也不记进粗动大地图。"
                    "z-approach(朝样品)是全系统唯一必须人工审批的动作。",
        "related": "横向换区 → RelocateCoarseXY(+ get_coarse_map 选方向步数);"
                   "大退针 → RetractForSampleChange;进针 → ApproachTip",
        "nanonis": "Motor_StartMove, Motor_StopMove, Motor_PosGet",
        "example": '{"name": "MotorMove", "input": {"direction": "z-retract", "steps": 100}}',
    },
    "RelocateCoarseXY": {
        "when_use": "当前这片表面用完了(get_map_analysis 建议换区),要横向粗动到新的一片。"
                    "先用 get_coarse_map 拿方向/步数建议 —— 它知道哪些片已经去过。",
        "when_not": "压电量程内还有干净地方时(先用 get_next_scan_position)。"
                    "换样品/关机的大退针用 RetractForSampleChange。",
        "related": "前序 get_coarse_map / get_map_analysis;内部已含退针+重新进针",
        "nanonis": "Motor_StartMove(内部), ZCtrl_Withdraw, AutoApproach",
        "example": '{"name": "RelocateCoarseXY", '
                   '"input": {"axis": "x", "direction": "+", "steps": 300}}',
    },
    "GetChamberPressure": {
        "when_use": "粗动被拒绝时问原因;或换区前先确认真空度够低。",
        "when_not": "与真空无关的操作。",
        "related": "RelocateCoarseXY 会自动检查,不需要每次先手动问",
        "nanonis": "(读环境监控,不走 Nanonis)",
        "example": '{"name": "GetChamberPressure", "input": {}}',
    },
    "PreScanCheck": {
        # PreScanCheck 读取整帧，不能按单条扫描线的代价使用它；速度与预计耗时
        # 应从实际扫描配置计算。判决只从保存的 .sxm 产生。
        # 实时缓冲区可能采用不同的处理口径，也可能对应另一帧，不能替代存盘判据。
        # 缺少可比较的文件时 tip_ready=None 表示判不了，不表示针尖不好。
        # ⚠️ 只有 when_use / when_not / related 会被 `get_skill_guidance` 取出来
        # (`nanonis`/`example` 走别处)—— 想让模型看见的话就得写进这三个键里,
        # 新加一个键会被**静默丢掉**。所以三态那段话塞进了 when_use。
        "when_use": "正式扫描前预检针尖质量(正反扫是否重合)。ContinuousImaging_Auto 中使用。"
                    "**tip_ready 是三态:True / False / None=判不了**。判决只从"
                    "存盘的 .sxm 出;存不下盘或找不到文件会回落到实时缓冲区,而那条路"
                    "(与 .sxm 不可比、且未必装着这一帧)一律给 None。拿到 None 就读"
                    "read_failure / abstain_reason,下一步是换个地方重新量或先把帧存下来"
                    "——**不是修针,更不是打脉冲**。",
        "when_not": "已知针尖良好时直接 FullScan。**它不便宜**:扫整帧,50 nm 约 8.5 分钟。",
        "related": "使用 CheckLineQuality / 后续 ConditionTip",
        "nanonis": "(组合: 整帧扫描 + 正反扫比对)",
        # 每线时间不传 = 按扫描宽度查档位表(50 nm ⇒ 1.0 s/线)。
        # **只有用户明确要求更快/更慢时才传 line_time_s** —— 它直接决定针尖
        # 在表面上的横向速度。
        "example": '{"name": "PreScanCheck", "input": {"center_x_m": 0, "center_y_m": 0, "width_m": 50e-9}}',
    },
    "UnmixSpectra": {
        "when_use": "GridSTS 后分析准粒子干涉图案的空间分布。",
        "when_not": "单点谱无需分解。",
        "related": "前序 GridSTS / 引用: AtomAI",
        "nanonis": "(纯分析)",
        "example": '{"name": "UnmixSpectra", "input": {"data_path": "grid.npy", "n_components": 3}}',
    },
    "Denoise_AE": {
        "when_use": "低电流或高速扫描导致的热噪声较大时。",
        "when_not": "信噪比已足够时无需去噪。",
        "related": "后序 SubtractPlane/LevelLines / 引用: AtomAI",
        "nanonis": "(纯分析)",
        "example": '{"name": "Denoise_AE", "input": {"image_path": "scan.npy"}}',
    },
    "ClusterDefects_rVAE": {
        "when_use": "需要自动分类不同类型的吸附子或缺陷时。",
        "when_not": "无原子分辨率时无法使用。",
        "related": "前序 DetectAtoms_FCN / 引用: AtomAI",
        "nanonis": "(纯分析)",
        "example": '{"name": "ClusterDefects_rVAE", "input": {"image_path": "s.npy", "positions_path": "pos.npy"}}',
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Workflow Recipes (10)
# ═══════════════════════════════════════════════════════════════════════

WORKFLOW_RECIPES: list[dict] = [
    {
        "name": "基础扫描",
        "desc": "标准扫描流程：设定偏压和 setpoint，配置并执行扫描，评估质量。",
        "chain": ["SetBias", "SetSetpoint", "FullScan", "AssessImageQuality"],
        "params": "bias=-0.5V, setpoint=100pA, 50nm, line_time=0.3s",
    },
    {
        "name": "STS 谱学",
        "desc": "在特定位置采集偏压谱。多点采谱在 MoveToXY+AcquireSTS 间循环。",
        "chain": ["ConfigureSTS", "MoveToXY", "AcquireSTS"],
        "params": "start=-2V, end=2V, num_points=500",
    },
    {
        "name": "网格谱映射",
        "desc": "GridSTS 均匀网格 vs AdaptiveSTS_GP 自适应采样。完整映射用前者，高效探索用后者。",
        "chain": ["GridSTS"],
        "params": "nx=5, ny=5, spacing=2nm",
    },
    {
        "name": "针尖修复",
        "desc": "分级策略：TipPulse(1级) -> ConditionTip(2级) -> ConditionTip_DQN(3级)。",
        "chain": ["AssessImageQuality", "TipPulse", "FullScan", "ConditionTip"],
        "params": "pulse_v=3V, target_quality=0.3",
    },
    {
        "name": "全自主成像",
        "desc": "长时间无人值守闭环：找区域 -> 修针 -> 扫描 -> 评估 -> 重复。",
        "chain": ["ContinuousImaging_Auto"],
        "params": "max_cycles=10, target_quality=0.3",
    },
    {
        "name": "多区域巡查",
        "desc": "系统性覆盖大区域。蛇形遍历减少针尖移动距离，输出质量映射。",
        "chain": ["AutonomousSurvey_Scanbot"],
        "params": "3x3 areas, 50nm, spacing=100nm",
    },
    {
        "name": "分辨率优化",
        "desc": "在 bias-setpoint 空间贝叶斯搜索最佳原子分辨率。",
        "chain": ["OptimizeResolution_BO"],
        "params": "bias: -1~-0.01V, setpoint: 10pA~1nA, 15 iterations",
    },
    {
        "name": "Kondo 分析",
        "desc": "完整 Kondo 共振分析：扫描、找空位、采谱、拟合。",
        "chain": ["FullScan", "FindEmptySpot", "MoveToXY", "AcquireSTS", "FitFano_Kondo"],
        "params": "STS: +/-50mV, 500 points",
    },
    {
        "name": "数据后处理",
        "desc": "标准三步处理：先扣平面再行校正，最后漂移估计。",
        "chain": ["SubtractPlane_RANSAC", "LevelLines_Median", "CorrectDrift_XCorr"],
        "params": "处理顺序很重要",
    },
    {
        "name": "安全回退",
        "desc": "按严重程度分级：轻微用 SafeRetract，严重用 EmergencyRetract。",
        "chain": ["SafeRetract", "WithdrawTip", "EmergencyRetract"],
        "params": "EmergencyRetract 使用 TCP 6504 专用端口",
    },
    {
        "name": "渐进式针尖恢复",
        "desc": (
            "三级递进策略：1️⃣ TipPulse(3V,50ms) 温和脉冲 → 2️⃣ ConditionTip(4V,5次) "
            "中等整形 → 3️⃣ ConditionTip_DQN(deepspm) DQN 自适应。"
            "每级之后用 AssessImageQuality 评估，仍差则升级到下一级。"
        ),
        "chain": [
            "TipPulse", "AssessImageQuality",
            "ConditionTip", "AssessImageQuality",
            "ConditionTip_DQN",
        ],
        "params": "1级: 3V/50ms; 2级: 4V/5次; 3级: DQN escalating",
    },
    {
        "name": "数据后处理 (完整)",
        "desc": "完整处理链：平面扣除 → 行校正 → 条纹去除 → 漂移校正 → 去噪。",
        "chain": [
            "SubtractPlane_RANSAC", "LevelLines_Median",
            "Destripe_MorphOpen", "CorrectDrift_XCorr", "Denoise_AE",
        ],
        "params": "处理顺序很重要，依次执行",
    },
    {
        "name": "原子操纵组装",
        "desc": (
            "多原子组装工作流：检测原子 → 匈牙利分配 → RRT路径规划 → SAC操纵。"
            "使用 plan_atom_assembly + AtomManip_SAC 循环。"
        ),
        "chain": ["DetectAtoms_FCN", "AtomManip_SAC"],
        "params": "每步操纵后 DetectAtomJump 验证",
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  Decision Trees (text for display)
# ═══════════════════════════════════════════════════════════════════════

DECISION_TREES: dict[str, dict] = {
    "spectroscopy": {
        "title": "谱学方法选择",
        "nodes": [
            {"q": "需要谱学测量?", "options": [
                ("单点", "ConfigureSTS + AcquireSTS"),
                ("多点均匀", "GridSTS"),
                ("多点高效", "AdaptiveSTS_GP"),
                ("Z方向", "ConfigureZSpectr + AcquireZSpectr"),
                ("自定义通道", "ConfigureBiasSweep + AcquireBiasSweep"),
            ]},
        ],
    },
    "tip_repair": {
        "title": "针尖修复升级",
        "nodes": [
            {"q": "图像质量差?", "options": [
                ("轻微 (0.2-0.3)", "TipPulse(3V, 1次)"),
                ("中等 (0.1-0.2)", "ConditionTip(3V, 5次)"),
                ("严重 (< 0.1)", "ConditionTip_DQN(escalating)"),
                ("极端", "WithdrawTip + 人工干预"),
            ]},
        ],
    },
    "region_search": {
        "title": "区域搜索策略",
        "nodes": [
            {"q": "需要找好区域?", "options": [
                ("快速/无模型", "FindGoodRegion_Heuristic"),
                ("精确/有模型", "FindGoodRegion_UNet"),
                ("系统巡查", "AutonomousSurvey_Scanbot"),
                ("全自主", "ContinuousImaging_Auto"),
            ]},
        ],
    },
    "scan_optimization": {
        "title": "扫描优化策略",
        "nodes": [
            {"q": "扫描图像有什么问题?", "options": [
                ("条纹噪声/线噪声", "CheckLineQuality → Destripe_MorphOpen; 检查接地和电磁屏蔽"),
                ("热漂移/图像扭曲", "TrackDrift_ReferenceScan → CorrectDrift_XCorr; 等待热平衡 (4K: 6-12h)"),
                ("分辨率不足", "AssessImageQuality → 如果 score<0.3: TipPulse/ConditionTip; 如果 score>0.3: 优化偏压和电流"),
                ("扫描速度太慢", "SetScanSpeed 增大 lines/s; 减小像素数; 或缩小扫描范围"),
                ("反馈振荡/条纹", "GetZCtrlGain → SetZCtrlGain 降低增益; 检查 approach 参数"),
            ]},
        ],
    },
    "bias_selection": {
        "title": "偏压选择指南",
        "nodes": [
            {"q": "要成像什么类型的表面?", "options": [
                ("金属清洁面", "±0.1~1.0V 均可; 表面态: Au(111) -0.5V, Cu(111) -0.4V; 100-500pA"),
                ("半导体", "填充态 -1.5~-2.5V (dangling bond) 或空态 +1.5~+2.5V; 50-300pA; 注意 TIBB"),
                ("分子吸附物", "查知识库 phases_override; 通常 HOMO 偏压负侧, LUMO 正侧; 10-200pA"),
                ("超导体", "小偏压 ±5~50mV 看超导 gap; CDW 通常 ±50~200mV; 需低温"),
                ("磁性表面", "选择自旋极化对比最大的偏压; 通常 ±0.3~1.0V; SP-STM 需磁性针尖"),
            ]},
        ],
    },
    "noise_diagnosis": {
        "title": "噪声诊断流程",
        "nodes": [
            {"q": "噪声特征是什么?", "options": [
                ("50/60Hz 及谐波", "接地回路问题; 检查所有设备共地; 断开非必要设备电源; 使用隔离变压器"),
                ("宽频白噪声", "前置放大器噪声或热噪声; 检查增益设置; 降低带宽; 降低温度"),
                ("低频 1/f 漂移", "热漂移或压电蠕变; 等待热平衡; 使用漂移补偿 SetDriftCompensation"),
                ("机械振动 (尖峰)", "建筑/设备振动; 检查减振系统; 关闭涡旋泵; 脉冲管制冷需主动减振"),
                ("随机电报噪声", "针尖不稳定或表面吸附物跳跃; TipPulse 整形; 提高偏压清除吸附物"),
            ]},
        ],
    },
    "sample_preparation": {
        "title": "样品制备策略",
        "nodes": [
            {"q": "样品类型?", "options": [
                ("金属单晶", "Ar+ 溅射 (0.5-1.5keV, 10-30min) → 退火 (查材料知识库温度); 重复 3-5 轮"),
                ("半导体解理面", "UHV 内解理; GaAs/InAs 用(110)面; 解理后立即进 STM"),
                ("薄膜生长", "先制备基底 → 蒸镀 (查材料知识库速率/温度) → RHEED/LEED 确认结构"),
                ("分子蒸镀", "基底先清洁 → 分子 Knudsen cell (查蒸镀温度) → 控制覆盖度 (0.1-1ML)"),
                ("低温解理 (拓扑/超导)", "冷却至 <30K → UHV 内解理 → 直接转入 STM; 避免升温"),
            ]},
        ],
    },
    "lockin_parameter": {
        "title": "Lock-in 参数选择",
        "nodes": [
            {"q": "测量什么信号?", "options": [
                ("超导 gap (meV 级)", "调制 50-200µV rms; 频率 700-1200Hz; 时间常数 30-100ms; 需要 T<<Tc"),
                ("Kondo 共振 (meV 级)", "调制 0.1-2mV rms; 频率 800-4500Hz; 时间常数 10-30ms; 偏压范围 ±20-100mV"),
                ("分子 HOMO-LUMO (eV 级)", "调制 5-20mV rms; 频率 500-1000Hz; 时间常数 10-20ms; 偏压范围 ±2-3V"),
                ("半导体带隙 (eV 级)", "调制 20-50mV rms (RT) / 10mV (LT); 频率 800-1000Hz; 范围 ±2.5V"),
                ("nc-AFM 频率偏移", "PLLOnOff 开启 → ConfigurePLL 设频率范围 → ConfigurePLLExcitation 设振幅"),
            ]},
        ],
    },
    "fault_escalation": {
        "title": "故障升级处理",
        "nodes": [
            {"q": "遇到什么故障?", "options": [
                ("针尖 crash (电流突增)", "EmergencyRetract → 检查针尖 (AssessImageQuality) → TipPulse/ConditionTip → 如果无效: WithdrawTip 换针尖"),
                ("反馈失控 (Z 振荡)", "GetZCtrlGain → SetZCtrlGain 降低 P/I 增益 → 重新 approach; 检查接线"),
                ("TCP 连接丢失", "检查 NI Service Locator 服务 → 重启 Nanonis → reconnect; 不要 force-kill"),
                ("扫描仪压电不响应", "检查高压放大器输出; SaveSettings 保存状态; 尝试小范围 XY 移动测试"),
                ("样品损坏/表面污染", "WithdrawTip → 重新溅射退火 (如果可能) → 或换新位置; 记录损坏区域坐标"),
            ]},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Measurement templates – experiment design patterns (from MAST-reference)
# ═══════════════════════════════════════════════════════════════════════

MEASUREMENT_TEMPLATES: dict = {
    # ------------------------------------------------------------------
    # 1. Kondo 效应 STS
    # ------------------------------------------------------------------
    "kondo_sts": {
        "name": "Kondo 效应 STS",
        "name_en": "Kondo Effect STS",
        "description": "近藤效应的扫描隧道谱学测量，需低温和高能量分辨率",
        "parameters": {
            "bias_range_mv": {"min": -200, "max": 200, "note": "5-10× k_B×T_K, 对称于 E_F"},
            "lockin_vmod_mv": {"min": 0.03, "max": 2.5, "rule": "< HWHM of Kondo feature"},
            "lockin_freq_hz": {"min": 700, "max": 5000, "note": "高于 1/f 噪声拐角"},
            "temperature_k": {"max": 10, "note": "T << T_K 以分辨 Fano 线形"},
            "setpoint_na": {"min": 0.1, "max": 1.0, "note": "避免 Kondo 淬灭"},
            "sts_points": {"min": 256, "max": 1024, "note": "能量分辨率"},
        },
        "success_criteria": "零偏压 Fano dip/peak 可辨; Fano 拟合 R² > 0.98; q/T_K 物理合理",
        "skills_chain": ["ConfigureLockIn", "ConfigureSTS", "MoveToXY", "AcquireSTS", "FitFano_Kondo"],
        "tips": "需对比 Kondo 位点与裸表面参考谱",
    },
    # ------------------------------------------------------------------
    # 2. 准粒子干涉 (QPI) 映射
    # ------------------------------------------------------------------
    "qpi_mapping": {
        "name": "准粒子干涉 (QPI) 映射",
        "name_en": "Quasiparticle Interference Mapping",
        "description": "通过 dI/dV 网格谱的 FFT 提取能带色散关系",
        "parameters": {
            "grid_size": {"min": 128, "max": 256, "note": "像素; 决定 k 空间分辨率"},
            "fov_nm": {"min": 30, "max": 100, "note": "需覆盖足够多散射波长"},
            "energy_slices": {"min": 20, "max": 60, "note": "均匀间隔覆盖感兴趣能量范围"},
            "bias_range_mv": {"min": -500, "max": 500, "note": "覆盖待研究能带"},
            "lockin_vmod_mv": {"min": 0.5, "max": 5.0, "note": "权衡分辨率与信噪比"},
        },
        "success_criteria": "FFT 中散射矢量 q(E) 弧/环清晰可辨; 色散关系与 DFT 一致",
        "skills_chain": ["ConfigureLockIn", "ConfigureSTS", "GridSTS"],
        "tips": "需足够缺陷/台面边缘作为散射源; 数据处理需对称化 FFT",
    },
    # ------------------------------------------------------------------
    # 3. 超导能隙测量
    # ------------------------------------------------------------------
    "sc_gap": {
        "name": "超导能隙谱学",
        "name_en": "Superconducting Gap Spectroscopy",
        "description": "超导能隙的高分辨 dI/dV 测量，要求极低温和极小调制幅度",
        "parameters": {
            "bias_range_mv": {"min": -5, "max": 5, "note": "±3Δ 到 ±5Δ; 随材料调整"},
            "lockin_vmod_mv": {"min": 0.02, "max": 1.0, "rule": "eV_mod << Δ"},
            "lockin_freq_hz": {"min": 833, "max": 927, "note": "标准超导测量频率范围"},
            "temperature_k": {"max": 4.2, "note": "T < T_c/5 以获得充分展开的能隙"},
            "setpoint_pa": {"min": 50, "max": 200, "note": "稳定化电流; 典型 100-200 pA"},
            "sts_points": {"min": 400, "max": 1000, "note": "高点密度覆盖能隙区间"},
        },
        "success_criteria": "dI/dV 谱呈清晰 U 形/V 形能隙; BCS 拟合 R² > 0.99; 相干峰明显",
        "skills_chain": ["ConfigureLockIn", "ConfigureSTS", "AcquireSTS", "FitGap_BCS"],
        "tips": "超导针尖 (Nb/Pb) 可突破 Fermi-Dirac 热展宽极限; 涡旋成像需外加磁场",
    },
    # ------------------------------------------------------------------
    # 4. 自旋极化 STM
    # ------------------------------------------------------------------
    "sp_stm": {
        "name": "自旋极化 STM",
        "name_en": "Spin-Polarized STM",
        "description": "利用磁性针尖的自旋依赖隧穿测量表面磁结构",
        "parameters": {
            "bias_imaging_mv": {"min": -300, "max": -100, "note": "成像用负偏压"},
            "current_imaging_na": {"min": 0.3, "max": 3.0, "note": "成像电流"},
            "current_spectroscopy_pa": {"min": 10, "max": 100, "note": "谱学用低电流"},
            "lockin_vmod_imaging_mv": {"min": 10, "max": 30, "note": "成像调制幅度"},
            "lockin_freq_hz": {"min": 893, "max": 1777, "note": "成像 ~1777 Hz; 谱学 ~893 Hz"},
            "magnetic_field_t": {"min": -3, "max": 3, "note": "外场扫描范围 ±3 T"},
        },
        "success_criteria": "磁性不对称度 A = 5-50%; 已知磁结构的对比度与预期一致; 非磁性区域无对比",
        "skills_chain": ["SetBias", "SetSetpoint", "ConfigureLockIn", "FullScan"],
        "tips": "Fe/W 针尖偏振率 ~40% 但杂散场大; Cr 针尖杂散场小但偏振率 ~10%",
    },
    # ------------------------------------------------------------------
    # 5. 带隙 / 半导体 STS
    # ------------------------------------------------------------------
    "band_gap": {
        "name": "带隙 / 半导体 STS",
        "name_en": "Band Gap / Semiconductor STS",
        "description": "半导体或绝缘体的带隙测量，需要宽偏压范围和 Feenstra 归一化",
        "parameters": {
            "bias_range_mv": {"min": -2000, "max": 2000, "note": "需覆盖价带和导带"},
            "lockin_vmod_mv": {"min": 5, "max": 20, "note": "较大调制以改善信噪比"},
            "lockin_freq_hz": {"min": 700, "max": 1500, "note": "标准 lock-in 频率"},
            "setpoint_pa": {"min": 50, "max": 200, "note": "低电流减少针尖诱导带弯"},
            "sts_points": {"min": 200, "max": 500, "note": "兼顾分辨率与采集速度"},
            "n_spectra_average": {"min": 16, "max": 80, "note": "带隙边缘需足够平均"},
        },
        "success_criteria": "dI/dV 谱在带隙内接近零; 带边陡峭; Feenstra 归一化后带隙值与文献一致",
        "skills_chain": ["ConfigureLockIn", "ConfigureSTS", "MoveToXY", "AcquireSTS"],
        "tips": "使用 Feenstra (dI/dV)/(I/V) 归一化消除指数背景; 多点统计报告带隙值 ± 误差",
    },
    # ------------------------------------------------------------------
    # 6. 分子轨道成像 (HOMO/LUMO)
    # ------------------------------------------------------------------
    "molecular_orbital": {
        "name": "分子轨道成像 (HOMO/LUMO)",
        "name_en": "Molecular Orbital Imaging",
        "description": "通过 dI/dV 映射可视化分子前沿轨道的空间分布",
        "parameters": {
            "bias_homo_mv": {"min": -3000, "max": -500, "note": "HOMO 能量; 分子依赖"},
            "bias_lumo_mv": {"min": 500, "max": 3000, "note": "LUMO 能量; 分子依赖"},
            "lockin_vmod_mv": {"min": 5, "max": 20, "note": "轨道成像需较大调制"},
            "fov_nm": {"min": 2, "max": 10, "note": "覆盖单分子或小区域"},
            "grid_size": {"min": 64, "max": 128, "note": "轨道空间分辨率"},
            "sts_points": {"min": 200, "max": 500, "note": "覆盖 HOMO-LUMO 范围"},
        },
        "success_criteria": "dI/dV 图在 HOMO/LUMO 能量处显示明确的轨道节面和叶瓣; 与 DFT 轨道一致",
        "skills_chain": ["ConfigureLockIn", "ConfigureSTS", "MoveToXY", "AcquireSTS", "GridSTS"],
        "tips": "先做宽范围 STS 定位 HOMO/LUMO 峰位; 再做窄范围高分辨映射",
    },
    # ------------------------------------------------------------------
    # 7. 原子/分子操纵
    # ------------------------------------------------------------------
    "atom_manipulation": {
        "name": "原子/分子操纵",
        "name_en": "Atom / Molecule Manipulation",
        "description": "通过降低隧穿结电阻实现单原子或分子的推、拉、滑动操控",
        "parameters": {
            "manip_bias_mv": {"min": 5, "max": 50, "note": "操纵偏压; 远低于成像偏压"},
            "manip_current_na": {"min": 10, "max": 100, "note": "操纵电流; ~100-1000× 成像"},
            "manip_resistance_kohm": {"min": 20, "max": 200, "note": "结电阻; 远低于成像"},
            "manip_speed_a_per_s": {"min": 1, "max": 5, "note": "横向扫描速度 Å/s"},
            "pulse_v_pickup": {"min": 2, "max": 5, "note": "垂直提取脉冲电压"},
            "temperature_k": {"max": 10, "note": "< 10 K 防止热扩散"},
        },
        "success_criteria": "原子移动到目标位置 ±1 Å 内; Z 信号呈锯齿/跳变确认操纵事件; 成功率 > 90% (金属原子)",
        "skills_chain": ["DetectAtoms_FCN", "MoveToXY", "AtomManip_SAC", "DetectAtomJump"],
        "tips": "拉模式 (attractive): 锯齿 Z 信号; 推模式 (repulsive): 反向锯齿; 每步后验证",
    },
    # ------------------------------------------------------------------
    # 8. 非接触 AFM 键分辨成像
    # ------------------------------------------------------------------
    "nc_afm": {
        "name": "非接触 AFM 键分辨成像",
        "name_en": "Non-Contact AFM Bond-Resolved Imaging",
        "description": "CO 功能化针尖 qPlus AFM，亚埃振幅实现化学键分辨",
        "parameters": {
            "oscillation_amplitude_a": {"min": 0.4, "max": 1.0, "note": "亚埃振幅是键分辨关键"},
            "resonance_freq_khz": {"min": 24, "max": 30, "note": "qPlus 传感器共振频率"},
            "q_factor": {"min": 10000, "max": 50000, "note": "5 K 下典型 Q 值"},
            "tip_height_pm": {"min": 280, "max": 370, "note": "针尖-分子距离; 10 pm 步进逼近"},
            "bias_mv": {"min": -10, "max": 0, "note": "纯 AFM 用 0 mV; 同时 STM 用 -5~-10 mV"},
            "fov_nm": {"min": 2, "max": 5, "note": "单分子尺度 FOV"},
            "temperature_k": {"max": 5, "note": "4-5 K LHe; 漂移 < 1 pm/min"},
        },
        "success_criteria": "恒高 Δf 图像中化学键骨架清晰可辨; 键长/键角与晶体学数据一致",
        "skills_chain": ["ConfigureScan", "SetScanSpeed", "FullScan", "SubtractPlane_RANSAC"],
        "tips": "CO 拾取: 移到 CO 上方瞬间增大电流至 5-20 nA 或脉冲 ~200 mV; 从远处 10 pm 步进逼近",
    },
    "noise_baseline": {
        "name": "噪声基线诊断",
        "name_en": "Noise Baseline Diagnostic",
        "description": "系统噪声基线测量与诊断，识别频率源并迭代消除",
        "parameters": {
            "sample_rate_ks": {"min": 10, "max": 1000, "note": "≥10 kS/s, Nyquist 至 5 kHz"},
            "acquisition_time_s": {"min": 10, "max": 60, "note": "≥10 s 以获得 <0.1 Hz 分辨"},
            "welch_averages": {"min": 10, "max": 50, "note": "方差 vs 频率分辨率权衡"},
            "null_image_pixels": {"value": 128, "note": "128×128, scan_size=0, 1 Hz 行频"},
        },
        "success_criteria": "Z noise RMS 达到系统基准; 无 >3× 局部 PSD 中值的尖峰; 1/f 拐角 < 100 Hz",
        "skills_chain": ["SafeRetract", "GetCurrent", "FullScan"],
        "tips": "三条基线: retracted (电子底噪), engaged FB-off (隧穿噪声), null image FB-on (Z 噪声). Ref: Ge, Ovadia & Hoffman 2019",
    },
}


def get_measurement_template(query: str) -> dict | None:
    """Return measurement template by exact key or fuzzy match on name/name_en."""
    tmpl = None
    if query in MEASUREMENT_TEMPLATES:
        tmpl = MEASUREMENT_TEMPLATES[query]
    else:
        q = query.lower()
        for key, t in MEASUREMENT_TEMPLATES.items():
            if q in key or q in t["name"].lower() or q in t["name_en"].lower():
                tmpl = t
                break
    if tmpl is None:
        return None
    # Apply admin override if available
    try:
        from mast.admin.override_store import ConfigOverrideRegistry, deep_merge
        ovr = ConfigOverrideRegistry.get().get_measurement_template_override(query)
        if ovr:
            return deep_merge(tmpl, ovr)
    except (ImportError, Exception):
        pass
    return tmpl


def format_measurement_template_for_llm(key: str) -> str:
    """Format a measurement template as compact text for LLM context."""
    tmpl = MEASUREMENT_TEMPLATES.get(key)
    if not tmpl:
        return f"Unknown template: {key}"
    lines = [f"## {tmpl['name']} ({tmpl['name_en']})", tmpl["description"], ""]
    lines.append("Parameters:")
    for pname, spec in tmpl["parameters"].items():
        parts = [f"  {pname}:"]
        if "min" in spec:
            parts.append(f"min={spec['min']}")
        if "max" in spec:
            parts.append(f"max={spec['max']}")
        if "note" in spec:
            parts.append(f"({spec['note']})")
        elif "rule" in spec:
            parts.append(f"({spec['rule']})")
        lines.append(" ".join(parts))
    lines.append(f"\nSuccess: {tmpl['success_criteria']}")
    lines.append(f"Skills: {' → '.join(tmpl['skills_chain'])}")
    if tmpl.get("tips"):
        lines.append(f"Tips: {tmpl['tips']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Public API for LLM consumption
# ═══════════════════════════════════════════════════════════════════════

STS_CORRECTION_METHODS: dict = {
    "feenstra_normalization": {
        "name": "Feenstra 归一化",
        "purpose": "消除半导体 STS 中隧穿透射系数的指数依赖, 近似提取 LDOS",
        "formula": "(dI/dV) / (I/V + c)",
        "steps": [
            "1. 获取 I(V) 曲线 (足够细的偏压步进)",
            "2. 数值微分得 dI/dV (或直接用 lock-in 信号)",
            "3. 计算 I/V (V=0 附近需正则化)",
            "4. 选择正则化常数 c (通常取 I/V 在带隙外均值的 1-10%)",
            "5. 归一化: N(V) = (dI/dV) / (I/V + c)",
        ],
        "regularization_c": "c = 0.01-0.1 × mean(|I/V|) outside gap; 过小→噪声放大; 过大→特征被压平",
        "limitations": "高偏压处仍有误差; 不能完全消除 TIBB; 需配合 SEMITIP 做完整校正",
        "applicable_to": ["semiconductor", "wide bandgap oxide", "molecular adsorbate on insulator"],
    },
    "dynes_fitting": {
        "name": "Dynes 展宽拟合",
        "purpose": "从超导 STS 谱中提取能隙 Δ 和准粒子寿命展宽 Γ",
        "formula": "N(E) = Re[(E + iΓ) / sqrt((E + iΓ)² - Δ²)]",
        "fitting_steps": [
            "1. 获取 dI/dV(V) 曲线 (对称偏压范围, 覆盖 >3Δ)",
            "2. 构建 Dynes DOS 函数 N(E, Δ, Γ)",
            "3. 卷积 Fermi 函数导数: dI/dV ∝ ∫ N(E) × (-df/dE)(E-eV) dE",
            "4. 最小二乘拟合提取 Δ 和 Γ (初值: Δ from peak spacing, Γ ~ 0.01Δ)",
            "5. 多 gap 系统: 加权叠加 w₁N(Δ₁,Γ₁) + w₂N(Δ₂,Γ₂)",
        ],
        "parameters": {
            "delta_meV": "超导能隙, 从相干峰间距粗估: Δ ≈ peak_spacing/2 (N-I-S) 或 /4 (S-I-S)",
            "gamma_meV": "准粒子展宽, clean limit Γ/Δ < 0.01, dirty limit Γ/Δ > 0.1",
            "temperature_K": "需已知, 影响 Fermi 展宽 (3.5kBT)",
        },
        "common_pitfalls": [
            "Gap 不对称: 可能是 tip DOS 非平坦或 tip 有 gap (Pb tip S-I-S 需两个 Δ)",
            "背景斜率: 正常态 DOS 非常数, 需减去线性或多项式背景",
            "Γ 过大使 gap 被填充: Γ/Δ > 0.3 时 gap 不再可见, 可能是温度过高或杂质散射",
        ],
        "applicable_to": ["superconductor", "proximity-coupled systems"],
    },
    "semitip_correction": {
        "name": "SEMITIP 带弯曲校正",
        "purpose": "半导体 STS 中建立偏压→能量映射关系, 消除 TIBB",
        "method": "三维泊松方程数值求解 (Feenstra SEMITIP 软件)",
        "required_inputs": {
            "carrier_concentration_cm3": "载流子浓度 (n 或 p 型)",
            "dielectric_constant": "介电常数 ε_r",
            "tip_radius_nm": "tip 曲率半径 (通常 5-100 nm)",
            "tip_work_function_eV": "tip 功函数 (~4.0 eV for W)",
            "contact_potential_V": "接触电位差",
        },
        "output": "偏压-能量映射函数 E(V); 表面带弯曲量 φ_s(V)",
        "simplified_approach": "平板电容近似: φ_s ≈ (V - V_FB) × C_tip / (C_tip + C_SC); 仅适合低偏压",
        "limitations": "需要准确的 tip 形状; 计算量大; 不考虑表面态钉扎",
        "applicable_to": ["semiconductor", "Schottky barrier", "MOS interface"],
    },
}


def get_skill_guidance(skill_name: str) -> dict:
    """Return expert guidance for a skill.

    Returns dict with keys: when_use, when_not, related,
    decision_context (if skill appears in a decision tree),
    workflows (if skill appears in a recipe).
    Returns empty dict if skill is unknown.
    """
    result: dict = {}

    # Skill extra metadata (with admin override merge)
    extra = SKILL_EXTRA.get(skill_name)
    try:
        from mast.admin.override_store import ConfigOverrideRegistry, deep_merge
        ovr = ConfigOverrideRegistry.get().get_guidance_override(skill_name)
        if ovr:
            extra = deep_merge(extra or {}, ovr)
    except (ImportError, Exception):
        pass
    if extra:
        result["when_use"] = extra.get("when_use", "")
        result["when_not"] = extra.get("when_not", "")
        result["related"] = extra.get("related", "")

    # Relevant decision trees
    relevant_trees: list[dict] = []
    for tree_id, tree in DECISION_TREES.items():
        for node in tree.get("nodes", []):
            for option_text, option_skill in node.get("options", []):
                if skill_name in option_skill:
                    relevant_trees.append({
                        "tree": tree["title"],
                        "context": option_text,
                        "recommendation": option_skill,
                    })
    if relevant_trees:
        result["decision_context"] = relevant_trees

    # Relevant workflow recipes
    relevant_recipes: list[dict] = []
    for recipe in WORKFLOW_RECIPES:
        if skill_name in recipe.get("chain", []):
            relevant_recipes.append({
                "name": recipe["name"],
                "desc": recipe["desc"],
                "chain": " -> ".join(recipe["chain"]),
            })
    if relevant_recipes:
        result["workflows"] = relevant_recipes

    return result


def format_skill_guidance_for_llm(skill_name: str) -> str:
    """Format skill guidance as compact text for LLM consumption.

    Returns empty string if skill is unknown.
    """
    info = get_skill_guidance(skill_name)
    if not info:
        return ""

    lines: list[str] = [f"## Skill Guidance: {skill_name}"]

    if "when_use" in info:
        lines.append(f"When to use: {info['when_use']}")
    if "when_not" in info:
        lines.append(f"When NOT to use: {info['when_not']}")
    if "related" in info:
        lines.append(f"Related: {info['related']}")

    if "decision_context" in info:
        lines.append("Decision context:")
        for ctx in info["decision_context"]:
            lines.append(f"  - {ctx['tree']}: {ctx['context']} -> {ctx['recommendation']}")

    if "workflows" in info:
        lines.append("Workflow recipes:")
        for wf in info["workflows"]:
            lines.append(f"  - {wf['name']}: {wf['chain']}")

    return "\n".join(lines)
