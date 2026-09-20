"""Structured data from the MAST Skill Encyclopedia for GUI integration.

Provides domain groupings, workflow recipes, intent mappings, decision trees,
and per-skill metadata extensions beyond what SkillRegistry carries.

LLM-facing data (SKILL_EXTRA, DECISION_TREES, WORKFLOW_RECIPES) lives in
mast.knowledge.skill_guidance and is re-exported here for backward compat.
"""

from __future__ import annotations

from mast.knowledge.skill_guidance import (  # noqa: F401
    SKILL_EXTRA,
    DECISION_TREES,
    WORKFLOW_RECIPES,
    get_skill_guidance,
    format_skill_guidance_for_llm,
)

# ═══════════════════════════════════════════════════════════════════════
#  Domain Groups (18 domains, 249 skills)
# ═══════════════════════════════════════════════════════════════════════

DOMAINS: list[dict] = [
    {
        "id": "bias_current",
        "name": "偏压与电流",
        "desc": "读写偏压电压和隧穿电流，是所有 STM 操作的基础参量。",
        "skills": [
            "GetBias", "SetBias", "GetCurrent", "BiasPulse", "SetCurrentGain",
            "GetBiasCalibration", "SetBiasCalibration", "SetBiasRange",
            "GetCurrentBEEM", "SetCurrentCalibration",
        ],
    },
    {
        "id": "z_control",
        "name": "Z控制",
        "desc": "Z 轴压电位置和反馈控制器管理。",
        "skills": [
            "GetZPosition", "SetSetpoint", "ZControllerOnOff", "GetZCtrlGain", "SetZCtrlGain",
            "SetZPosition", "SetTipLift", "GetTipLift",
            "SetZLimitsEnabled", "GetZLimitsEnabled", "GetZCtrlList",
            "GetHomeProps", "SetHomeProps", "SetSwitchOffDelay", "GetWithdrawRate",
        ],
    },
    {
        "id": "scanning",
        "name": "扫描成像",
        "desc": "配置扫描帧、速度，执行扫描。FullScan 是一步组合。",
        "skills": [
            "ConfigureScan", "SetScanSpeed", "StartScan", "StopScan", "FullScan",
            "GetScanFrame", "GetScanSpeed", "GetScanBuffer", "SaveScan",
            "WaitScanComplete", "SetTipSpeed",
            "GetScanXYPosition", "ScanBackgroundPaste", "ScanBackgroundDelete",
        ],
    },
    {
        "id": "navigation",
        "name": "导航",
        "desc": "使用 Follow Me 功能移动针尖到指定 XY 坐标。",
        "skills": [
            "MoveToXY", "MotorMove", "MotorGetPos", "StopMotor",
            "MotorMoveClosedLoop", "GetMotorStepCounter", "SetMotorFreqAmp",
            "GetTipSpeed", "SetFolMeOversampling", "StopFolMe",
            "GetPointShootOnOff", "SetPointShootOnOff",
            "SetPointShootExperiment", "GetPointShootProps",
        ],
    },
    {
        "id": "tip_management",
        "name": "针尖管理",
        "desc": "针尖安全操作和修针。从简单到复杂：SafeRetract < TipPulse < ConditionTip < ConditionTip_DQN。",
        "skills": [
            "SafeRetract", "EmergencyRetract", "WithdrawTip", "AutoApproach",
            "TipPulse", "ConditionTip", "ConditionTip_DQN",
            "TipShape", "ConfigureAtomTrack",
            "AtomTrackDriftComp", "AtomTrackQuickCompStart", "AtomTrackStatusGet",
        ],
    },
    {
        "id": "spectroscopy",
        "name": "谱学",
        "desc": "STS 偏压谱、Z 谱以及网格/自适应谱采集。",
        "skills": [
            "ConfigureSTS", "AcquireSTS", "ConfigureZSpectr", "AcquireZSpectr",
            "GridSTS", "AdaptiveSTS_GP",
            "ConfigureSTSTiming", "ConfigureSTSChannels", "ConfigureZSpectrTiming",
            "StopSTS", "StopZSpectr", "RunGridExperiment",
            "GetSTSChannels", "SetSTSChannels", "GetSTSLimits",
            "SetSTSAdvancedProps", "GetSTSTiming", "GetSTSAltZCtrl",
            "GetSTSDigSync", "GetSTSTTLSync", "GetSTSPulseSeqSync",
            "GetSTSZOffRevert", "GetSTSMLSLockinPerSeg",
            "SetSTSMLSMode", "SetSTSMLSVals",
            "SetSTSSafeCond1", "GetSTSSafeCond1", "SetSTSSafeCond2",
            "GetZSpectrChannels", "SetZSpectrChannels",
            "GetZSpectrRange", "SetZSpectrRange",
            "GetZSpectrRetract", "SetZSpectrRetract",
            "SetZSpectrAdvProps", "GetZSpectrDigSync",
            "GetZSpectrPulseSeqSync", "GetZSpectrRetract2nd",
            "SetZSpectrRetractDelay", "GetZSpectrTTLSync", "GetZSpectrTiming",
            "GenSwpAcqChsGet", "GenSwpPropsGet", "GenSwpStop", "GenSwpSwpSignalGet",
        ],
    },
    {
        "id": "lockin_sweep",
        "name": "锁相与扫频",
        "desc": "Lock-in 放大器调制配置、偏压扫描和频率扫描。",
        "skills": [
            "ConfigureLockIn", "ConfigureBiasSweep", "AcquireBiasSweep",
            "ConfigureLockInSweep", "AcquireLockInSweep",
            "ConfigurePLL", "ConfigurePLLExcitation", "PLLOnOff",
            "GetPLLStatus", "PLLSignalAnalyzer", "AcquirePLLFreqSweep",
            "ConfigureLockInDemod", "GetLockInConfig",
            "GetDemodHPFilter", "GetDemodHarmonic", "GetDemodLPFilter",
            "GetDemodPhase", "GetDemodPhasReg", "SetDemodRTSignals",
            "GetDemodSignal", "SetDemodSyncFilter",
            "SetModHarmonic", "SetModPhasReg", "SetModSignal",
            "GetPLLAddOnOff", "GetPLLAmpCtrlOnOff",
            "SetPLLAmpCtrlBandwidth", "SetPLLAmpCtrlSetpnt",
            "GetPLLDemodFilter", "SetPLLDemodFilter",
            "GetPLLDemodHarmonic", "GetPLLDemodInput", "SetPLLDemodInput",
            "SetPLLDemodPhasRef", "GetPLLExcRange",
            "SetPLLFreqExcOverwrite", "GetPLLFreqRange", "SetPLLFreqRange",
            "PLLFreqShiftAutoCenter", "GetPLLInpCalibr", "SetPLLInpCalibr",
            "GetPLLInpProps", "SetPLLInpProps", "SetPLLInpRange",
            "PLLPerfectPLLUpdtZTC", "SetPLLPhasCtrlBandwidth",
            "GetPLLPhasCtrlOnOff",
            "GetPLLSignalAnlzrCh", "GetPLLSignalAnlzrFFTProps",
            "GetPLLSignalAnlzrTimebase", "PLLSignalAnlzrTrigAuto",
            "SetPLLSignalAnlzrTrig",
            "GetPLLFreqSwpParams", "StopPLLFreqSwp",
            "GetLockInSweepLimits", "GetLockInSweepProps", "GetLockInSweepSignal",
        ],
    },
    {
        "id": "data_processing",
        "name": "数据处理",
        "desc": "离线图像后处理：平面扣除、行校正、漂移估计、空白区域搜索。",
        "skills": [
            "SubtractPlane_RANSAC", "LevelLines_Median", "CorrectDrift_XCorr", "FindEmptySpot",
            "Destripe_MorphOpen", "SubtractPoly2D", "DeconvolveTip_RL", "Denoise_AE",
            "CorrectDrift_BraggPeak", "CheckLineQuality", "DetectAtomJump",
            "GetSignalRange", "GetSignalValues", "GetSignalsAddRT",
        ],
    },
    {
        "id": "image_analysis",
        "name": "图像分析",
        "desc": "图像质量评估、针尖状态判断、区域分割、原子检测。",
        "skills": [
            "AssessImageQuality", "AssessTip_VGG", "AssessTip_ResNet",
            "SegmentRegion_UNet", "DetectAtoms_FCN",
            "ClusterDefects_rVAE",
        ],
    },
    {
        "id": "molecular_recognition",
        "name": "分子结构识别",
        "desc": "从 STM 图像预测分子结构或拓扑类型。需要预训练模型。",
        "skills": ["PredictStructure_ASD", "IdentifyTopology_CARP"],
    },
    {
        "id": "spectral_analysis",
        "name": "谱分析",
        "desc": "dI/dV 谱线型拟合：Fano/Frota-Fano（Kondo）和 BCS 超导能隙。",
        "skills": ["FitFano_Kondo", "FitGap_BCS", "UnmixSpectra", "PredictSpectrumFromTopo"],
    },
    {
        "id": "optimization",
        "name": "参数优化",
        "desc": "贝叶斯优化自动调参，搜索最佳原子分辨率。",
        "skills": ["OptimizeResolution_BO"],
    },
    {
        "id": "autonomous",
        "name": "自主工作流",
        "desc": "多步自主流程：区域搜索、多区域巡查、持续成像。",
        "skills": [
            "FindGoodRegion_Heuristic", "FindGoodRegion_UNet",
            "AutonomousSurvey_Scanbot", "ContinuousImaging_Auto",
        ],
    },
    {
        "id": "atom_manipulation",
        "name": "原子操纵",
        "desc": "单原子搬运和表面化学反应。高危操作。",
        "skills": ["AtomManip_SAC", "AutoOSS_Dehalogenation"],
    },
    {
        "id": "safety",
        "name": "安全",
        "desc": "安全联锁和针尖保护机制。",
        "skills": [
            "EnableSafeTip", "GetSafeTipStatus",
            "GetAutoApproachStatus", "GetSafeTipProps", "GetSafeTipSignal",
        ],
    },
    {
        "id": "utility",
        "name": "工具",
        "desc": "UI 锁定、设置保存、漂移补偿等通用工具。",
        "skills": [
            "LockNanonisUI", "SaveSettings", "SetDriftCompensation", "GetDriftCompensation",
            "GetAcqPeriod", "LoadLayout", "SaveLayout",
            "GetRTFreq", "SetRTFreq", "GetRTOversample", "SetRTOversample",
            "GetSessionPath", "SetSessionPath", "UnlockNanonisUI",
            "GetPatternCloud", "SetPatternCloud", "GetPatternProps",
            "SetPatternLine", "OpenPatternExperiment", "PausePatternExperiment",
            "SetPiezoTilt", "GetPiezoTilt", "SetPiezoRange",
            "SetPiezoSensitivity", "GetPiezoSensitivity",
            "GetPiezoHVAInfo", "GetPiezoHVAStatusLED", "GetPiezoXYZLimits",
            "SetPiezoHysteresisOnOff", "SetPiezoHysteresisValues",
            "LoadPiezoHysteresisFile",
        ],
    },
    {
        "id": "feedback",
        "name": "反馈",
        "desc": "扫描前检查和漂移追踪等反馈监控功能。",
        "skills": ["PreScanCheck", "TrackDrift_ReferenceScan"],
    },
    {
        "id": "optics_bench",
        "name": "光学平台",
        "desc": "TERS/THz 光学部分：位移台、泵浦-探测延迟线与延迟扫描（非 Nanonis 硬件）。",
        "skills": [
            "ListOpticalDevices", "OpticalStageGetPos", "OpticalStageMove",
            "OpticalStageWiggle", "StopOpticalStage", "HomeOpticalStage",
            "DelayLineGetDelay", "DelayLineMoveTo", "AcquireSignalPoint",
            "PumpProbeScan", "OpticalStageScan",
        ],
    },
]

# Reverse lookup: skill_name -> domain_id
SKILL_DOMAIN: dict[str, str] = {}
for _d in DOMAINS:
    for _s in _d["skills"]:
        SKILL_DOMAIN[_s] = _d["id"]

# domain_id -> domain dict
DOMAIN_BY_ID: dict[str, dict] = {d["id"]: d for d in DOMAINS}

# ═══════════════════════════════════════════════════════════════════════
#  Domain → owning agent (Skills hierarchy v2.1+)
# ═══════════════════════════════════════════════════════════════════════
#
# Each MAST agent owns a subset of skills via the domains it speaks for.
# This map drives the per-agent grouping shown in the Skill Codex tab
# and the QuickAsk "查询范围" selector. Domains not listed default to
# "instrument_control" since the vast majority of read/write skills are
# IC-side Nanonis ops.

AGENT_BY_DOMAIN: dict[str, str] = {
    # Instrument-control agent — Nanonis read/write + autonomous loops
    "bias_current":            "instrument_control",
    "z_control":               "instrument_control",
    "scanning":                "instrument_control",
    "navigation":              "instrument_control",
    "tip_management":          "instrument_control",
    "spectroscopy":            "instrument_control",
    "lockin_sweep":            "instrument_control",
    "atom_manipulation":       "instrument_control",
    "safety":                  "instrument_control",
    "optimization":            "instrument_control",
    "autonomous":              "instrument_control",
    # Data-processing agent — file parsing + image / spectral analysis
    "data_processing":         "data_processing",
    "image_analysis":          "data_processing",
    "molecular_recognition":   "data_processing",
    "spectral_analysis":       "data_processing",
}

# Convenience: skill_name -> owning agent
SKILL_AGENT: dict[str, str] = {
    skill: AGENT_BY_DOMAIN.get(domain, "instrument_control")
    for skill, domain in SKILL_DOMAIN.items()
}

# ═══════════════════════════════════════════════════════════════════════
#  Canonical agent order — SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════════════
# Scientific-workflow order (NOT instrument-control-first): a study flows
# 文献 → 实验设计 → 仪器控制 → 数据处理 → 论文写作 → 论文审稿, with the two
# side-channel agents (视觉摘要, 编排) last. Every per-agent UI list — the
# Skills tab sub-tabs, the Codex「Agent」selector, the admin per-agent tabs —
# must order off THIS list so reordering is a one-line change.
#
# (agent_id, 中文+缩写 label) in workflow order.
AGENT_ORDER: list[tuple[str, str]] = [
    ("literature",          "文献 LIT"),
    ("experiment_design",   "实验设计 XD"),
    ("instrument_control",  "仪器控制 IC"),
    ("data_processing",     "数据处理 DP"),
    ("paper_writing",       "论文写作 PW"),
    ("paper_review",        "论文审稿 PR"),
    ("buffer_summarizer",   "视觉摘要 BUF"),
    ("_supervisor",         "编排 SUP"),
]

# Agents that actually OWN registry skills today (SkillRegistry.list_skills()
# maps only these via SKILL_AGENT — IC ≈ 226 Nanonis ops, DP ≈ 26 analysis
# skills). 2026-06-15: literature/experiment_design/paper_writing/paper_review
# now own BRIDGED @tool skills (tool_skills auto-bridge → registry, tag-resolved
# by build_skill_codex_html), so they are no longer empty in the Codex「Agent」
# filter and are included here. buf / supervisor own no skills → stay out.
_SKILL_OWNING_AGENTS: set[str] = set(AGENT_BY_DOMAIN.values()) | {
    "literature", "experiment_design", "paper_writing", "paper_review",
}

# Display choices for the Codex per-agent selector. "All" (every registry
# skill) + only the agents that own registry skills, in canonical workflow
# order. Filtering to a tool-only agent would always be empty, so those are
# omitted here (the full skill set stays reachable via "All").
# label-first tuples for gr.Dropdown((label, value)).
AGENTS_FOR_SKILLS: list[tuple[str, str]] = [("All", "All")] + [
    (label, aid)
    for (aid, label) in AGENT_ORDER
    if aid in _SKILL_OWNING_AGENTS
]

# ═══════════════════════════════════════════════════════════════════════
#  Source mapping
# ═══════════════════════════════════════════════════════════════════════

_BUILTIN = {
    "GetBias", "SetBias", "GetCurrent", "GetZPosition", "SetSetpoint",
    "ZControllerOnOff", "ConfigureScan", "SetScanSpeed", "StartScan",
    "StopScan", "MoveToXY", "SafeRetract", "EmergencyRetract",
    "WithdrawTip", "AutoApproach", "ConfigureLockIn",
    "ConfigureSTS", "AcquireSTS", "ConfigureZSpectr", "AcquireZSpectr",
    "ConfigureBiasSweep", "AcquireBiasSweep",
    "ConfigureLockInSweep", "AcquireLockInSweep",
}
_COMPOSITE = {
    "TipPulse", "ConditionTip", "FullScan", "GridSTS", "AssessImageQuality",
}
_PAPER = {
    "SubtractPlane_RANSAC", "LevelLines_Median", "CorrectDrift_XCorr",
    "FindEmptySpot", "FitFano_Kondo", "FitGap_BCS",
    "AssessTip_VGG", "AssessTip_ResNet", "SegmentRegion_UNet",
    "DetectAtoms_FCN", "PredictStructure_ASD", "IdentifyTopology_CARP",
    "OptimizeResolution_BO", "AdaptiveSTS_GP",
    "FindGoodRegion_Heuristic", "FindGoodRegion_UNet",
    "ConditionTip_DQN", "AutonomousSurvey_Scanbot",
    "ContinuousImaging_Auto", "AtomManip_SAC", "AutoOSS_Dehalogenation",
}


def get_source(skill_name: str) -> str:
    """Return source label for a skill: '内置' / '组合' / '论文'."""
    if skill_name in _BUILTIN:
        return "内置"
    if skill_name in _COMPOSITE:
        return "组合"
    if skill_name in _PAPER:
        return "论文"
    return "自定义"


# ═══════════════════════════════════════════════════════════════════════
#  Composite Skill Hierarchy
# ═══════════════════════════════════════════════════════════════════════

COMPOSITE_HIERARCHY: dict[str, list[str]] = {
    "FullScan": ["ConfigureScan", "SetScanSpeed", "StartScan"],
    "GridSTS": ["ConfigureSTS", "MoveToXY", "AcquireSTS"],
    "TipPulse": ["GetBias", "SetBias"],
    "ConditionTip": ["TipPulse", "ConfigureScan", "StartScan", "AssessImageQuality"],
    "ConditionTip_DQN": ["TipPulse", "FullScan", "AssessImageQuality"],
    "OptimizeResolution_BO": ["SetBias", "SetSetpoint", "FullScan", "AssessImageQuality"],
    "AdaptiveSTS_GP": ["MoveToXY", "AcquireSTS"],
    "FindGoodRegion_Heuristic": ["FullScan", "FindEmptySpot", "MoveToXY"],
    "FindGoodRegion_UNet": ["FullScan", "SegmentRegion_UNet", "MoveToXY"],
    "AutonomousSurvey_Scanbot": ["MoveToXY", "FullScan", "AssessImageQuality", "TipPulse"],
    "ContinuousImaging_Auto": ["FindGoodRegion_Heuristic", "TipPulse", "FullScan", "AssessImageQuality"],
    "AtomManip_SAC": ["MoveToXY", "SetBias", "SetSetpoint", "ZControllerOnOff"],
    "AutoOSS_Dehalogenation": ["MoveToXY", "SetBias", "TipPulse"],
}


# ═══════════════════════════════════════════════════════════════════════
#  Intent-to-Skill Mapping (for Guide tab)
# ═══════════════════════════════════════════════════════════════════════

INTENT_MAPPING: list[dict] = [
    {"keywords": "扫描 / 拍照 / 成像", "skill": "FullScan", "note": "一步完成"},
    {"keywords": "偏压 / 电压", "skill": "GetBias / SetBias", "note": "读/写"},
    {"keywords": "电流 / 隧穿", "skill": "GetCurrent", "note": "只读"},
    {"keywords": "setpoint / 设定值", "skill": "SetSetpoint", "note": ""},
    {"keywords": "移动 / 去 / 位置", "skill": "MoveToXY", "note": "需 z_controller_on"},
    {"keywords": "STS / 谱 / dI/dV", "skill": "ConfigureSTS + AcquireSTS", "note": ""},
    {"keywords": "网格谱 / 谱映射", "skill": "GridSTS / AdaptiveSTS_GP", "note": "均匀 vs 自适应"},
    {"keywords": "Z谱 / I(z)", "skill": "ConfigureZSpectr + AcquireZSpectr", "note": ""},
    {"keywords": "修针 / 针尖不好", "skill": "TipPulse -> ConditionTip", "note": "分级升级"},
    {"keywords": "退针 / 缩回", "skill": "SafeRetract", "note": ""},
    {"keywords": "紧急 / 停止", "skill": "EmergencyRetract / StopScan", "note": ""},
    {"keywords": "逼近 / 进针", "skill": "AutoApproach", "note": ""},
    {"keywords": "质量 / 图像好不好", "skill": "AssessImageQuality", "note": ""},
    {"keywords": "找区域 / 找好地方", "skill": "FindGoodRegion_Heuristic", "note": ""},
    {"keywords": "巡查 / 大范围", "skill": "AutonomousSurvey_Scanbot", "note": ""},
    {"keywords": "自动跑 / 无人值守", "skill": "ContinuousImaging_Auto", "note": "DANGEROUS"},
    {"keywords": "Kondo / Fano", "skill": "FitFano_Kondo", "note": ""},
    {"keywords": "能隙 / 超导", "skill": "FitGap_BCS", "note": ""},
    {"keywords": "平面校正", "skill": "SubtractPlane_RANSAC", "note": ""},
    {"keywords": "行校正", "skill": "LevelLines_Median", "note": ""},
    {"keywords": "原子操纵 / 搬原子", "skill": "AtomManip_SAC", "note": "DANGEROUS"},
    {"keywords": "合成 / 脱卤", "skill": "AutoOSS_Dehalogenation", "note": "DANGEROUS"},
    {"keywords": "优化分辨率", "skill": "OptimizeResolution_BO", "note": ""},
    {"keywords": "原子检测 / 数原子", "skill": "DetectAtoms_FCN", "note": ""},
    {"keywords": "锁相 / lock-in", "skill": "ConfigureLockIn", "note": ""},
    {"keywords": "频率扫描", "skill": "ConfigureLockInSweep + AcquireLockInSweep", "note": ""},
]


# ═══════════════════════════════════════════════════════════════════════
#  Skill Verification Status
# ═══════════════════════════════════════════════════════════════════════

SKILL_VERIFICATION: dict[str, str] = {
    # -- Bias & Current (verified via test_full_system G1/G3) --
    "GetBias": "模拟器验证",
    "SetBias": "模拟器验证",
    "GetCurrent": "模拟器验证",
    # -- Z Control (verified via test_full_system G1/G2) --
    "GetZPosition": "模拟器验证",
    "SetSetpoint": "模拟器验证",
    "ZControllerOnOff": "模拟器验证",
    # -- Scanning (verified via test_full_system G6) --
    "ConfigureScan": "模拟器验证",
    "SetScanSpeed": "模拟器验证",
    "StartScan": "模拟器验证",
    "StopScan": "模拟器验证",
    "FullScan": "模拟器验证",
    # -- Navigation (verified via test_full_system G5) --
    "MoveToXY": "模拟器验证",
    # -- Tip Management (verified via test_full_system G4/G9/G10) --
    "SafeRetract": "模拟器验证",
    "EmergencyRetract": "模拟器验证",
    "WithdrawTip": "模拟器验证",
    "AutoApproach": "模拟器验证",
    "TipPulse": "模拟器验证",
    "ConditionTip": "模拟器验证",
    "ConditionTip_DQN": "未验证",
    # -- Spectroscopy (verified via test_full_system G7) --
    "ConfigureSTS": "模拟器验证",
    "AcquireSTS": "模拟器验证",
    "ConfigureZSpectr": "模拟器验证",
    "AcquireZSpectr": "模拟器验证",
    "GridSTS": "模拟器验证",
    "AdaptiveSTS_GP": "未验证",
    # -- Lock-in & Sweep (verified via test_full_system G8/G14/G15) --
    "ConfigureLockIn": "模拟器验证",
    "ConfigureBiasSweep": "模拟器验证",
    "AcquireBiasSweep": "模拟器验证",
    "ConfigureLockInSweep": "模拟器验证",
    "AcquireLockInSweep": "模拟器验证",
    # -- Data Processing (verified via test_paper_skills_sim G20) --
    "SubtractPlane_RANSAC": "模拟器验证",
    "LevelLines_Median": "模拟器验证",
    "CorrectDrift_XCorr": "模拟器验证",
    "FindEmptySpot": "模拟器验证",
    # -- Image Analysis (verified via test_full_system + G20) --
    "AssessImageQuality": "模拟器验证",
    "AssessTip_VGG": "模拟器验证",
    "AssessTip_ResNet": "模拟器验证",
    "SegmentRegion_UNet": "模拟器验证",
    "DetectAtoms_FCN": "模拟器验证",
    # -- Molecular Recognition (require trained models) --
    "PredictStructure_ASD": "未验证",
    "IdentifyTopology_CARP": "未验证",
    # -- Spectral Analysis (verified via test_paper_skills_sim G20) --
    "FitFano_Kondo": "模拟器验证",
    "FitGap_BCS": "模拟器验证",
    # -- Optimization (pending G21 Nanonis test) --
    "OptimizeResolution_BO": "未验证",
    # -- Autonomous (pending G22-G23 Nanonis test) --
    "FindGoodRegion_Heuristic": "未验证",
    "FindGoodRegion_UNet": "未验证",
    "AutonomousSurvey_Scanbot": "未验证",
    "ContinuousImaging_Auto": "未验证",
    # -- Atom Manipulation (pending G24 Nanonis test) --
    "AtomManip_SAC": "未验证",
    "AutoOSS_Dehalogenation": "未验证",
}


# ═══════════════════════════════════════════════════════════════════════
#  Skill Usage Frequency (dynamic, based on execution counts)
# ═══════════════════════════════════════════════════════════════════════

# Static hint counts — fallback when no historical data is available
_STATIC_HINT_COUNTS: dict[str, int] = {
    # 原"常用" → 20
    "GetBias": 20, "SetBias": 20, "GetCurrent": 20, "FullScan": 20,
    "SetSetpoint": 20, "MoveToXY": 20, "AssessImageQuality": 20,
    "StartScan": 20, "StopScan": 20, "ConfigureScan": 20,
    # 原"偶用" → 5
    "SetScanSpeed": 5, "GetZPosition": 5, "ZControllerOnOff": 5,
    "TipPulse": 5, "ConditionTip": 5, "AutoApproach": 5,
    "SafeRetract": 5, "ConfigureSTS": 5, "AcquireSTS": 5,
    "ConfigureLockIn": 5, "SubtractPlane_RANSAC": 5,
    "LevelLines_Median": 5, "FindEmptySpot": 5, "GridSTS": 5,
    "FitFano_Kondo": 5, "FitGap_BCS": 5, "OptimizeResolution_BO": 5,
    # 其余默认 0
}

_FREQ_LEVELS: list[tuple[int, str]] = [
    (40, "热门"), (15, "活跃"), (5, "一般"), (1, "冷门"), (0, "未用"),
]


def classify_usage_freq(count: int) -> str:
    """Classify a raw execution count into a frequency label."""
    for threshold, label in _FREQ_LEVELS:
        if count >= threshold:
            return label
    return "未用"


def get_verification(skill_name: str) -> str:
    """Return verification status: '未验证' / '模拟器验证' / '实机验证'."""
    # Check admin override first
    try:
        from mast.admin.override_store import ConfigOverrideRegistry
        ovr = ConfigOverrideRegistry.get().get_encyclopedia_overrides()
        sv = ovr.get("skill_verification", {})
        if skill_name in sv:
            return sv[skill_name]
    except (ImportError, Exception):
        pass
    return SKILL_VERIFICATION.get(skill_name, "未验证")


def get_usage_freq(skill_name: str, counts: dict[str, int] | None = None) -> str:
    """Return usage frequency label based on execution counts.

    If *counts* (from ``get_skill_execution_counts()``) is provided,
    uses the dynamic count; otherwise falls back to static hints.
    """
    if counts is not None:
        n = counts.get(skill_name, 0)
    else:
        n = _STATIC_HINT_COUNTS.get(skill_name, 0)
    return classify_usage_freq(n)
