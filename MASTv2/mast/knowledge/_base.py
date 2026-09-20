"""Generic workflow phases shared across all sample types."""

from __future__ import annotations

# ── Generic workflow phases ────────────────────────────────────────────
# Every sample type file provides its own CATEGORY["phases"]; these generic
# phases serve as the fallback skeleton for stub types that haven't been
# fully elaborated yet.

GENERIC_PHASES: list[dict] = [
    {
        "id": "surface_prep",
        "name": "表面制备",
        "name_en": "Surface Preparation",
        "description": "溅射退火循环获得原子级平整表面",
        "steps": [
            {
                "skill": None,
                "action": "Ar+ 溅射 + 退火",
                "params": {"energy_eV": 1000, "duration_min": 10},
                "notes": "具体温度和时间取决于材料",
            },
        ],
        "success_criteria": "LEED 确认表面重构 / STM 可见宽平台和单原子台阶",
        "on_fail": "增加溅射退火循环次数或调整参数",
    },
    {
        "id": "tip_prep",
        "name": "针尖制备",
        "name_en": "Tip Preparation",
        "description": "制备尖锐、稳定的针尖",
        "steps": [
            {"skill": "TipPulse", "params": {"pulse_v": 5.0, "count": 3}},
            {"skill": "FullScan", "params": {"width_m": 100e-9}},
            {"skill": "AssessImageQuality", "params": {}},
        ],
        "success_criteria": "质量评分 > 0.3, 无双针尖伪影",
        "on_fail": "升级到 ConditionTip",
    },
    {
        "id": "survey",
        "name": "初步巡查",
        "name_en": "Survey",
        "description": "大范围扫描，寻找干净、合适的区域",
        "steps": [
            {"skill": "FullScan", "params": {"width_m": 500e-9}},
            {"skill": "AssessImageQuality", "params": {}},
        ],
        "success_criteria": "找到干净平坦区域",
        "on_fail": "MoveToXY 换区域或 FindGoodRegion_Heuristic",
    },
    {
        "id": "imaging",
        "name": "高分辨成像",
        "name_en": "High-Resolution Imaging",
        "description": "在选定区域进行高分辨率扫描",
        "steps": [
            {"skill": "SetBias", "params": {"bias_v": -1.0}},
            {"skill": "SetSetpoint", "params": {"setpoint_a": 100e-12}},
            {"skill": "FullScan", "params": {"width_m": 20e-9, "line_time_s": 0.5}},
        ],
        "success_criteria": "可分辨目标特征结构",
        "on_fail": "OptimizeResolution_BO",
    },
    {
        "id": "spectroscopy",
        "name": "谱学测量",
        "name_en": "Spectroscopy",
        "description": "在感兴趣位置采集 dI/dV 或 I-V 谱",
        "steps": [
            {"skill": "ConfigureLockIn", "params": {"mod_on": True, "amplitude_v": 0.01}},
            {"skill": "ConfigureSTS", "params": {"start_v": -1.0, "end_v": 1.0, "num_points": 500}},
            {"skill": "AcquireSTS", "params": {}},
        ],
        "success_criteria": "dI/dV 谱特征与预期/文献一致",
        "on_fail": "检查针尖状态，重新制备",
    },
    {
        "id": "analysis",
        "name": "数据分析",
        "name_en": "Data Analysis",
        "description": "对采集的数据进行后处理",
        "steps": [
            {"skill": "SubtractPlane_RANSAC", "params": {}},
            {"skill": "LevelLines_Median", "params": {}},
        ],
        "success_criteria": "数据平整，特征清晰",
        "on_fail": "尝试不同的处理参数",
    },
]


def make_stub_category(
    cat_id: str,
    name: str,
    name_en: str,
    description: str,
    tip_requirements: str = "W tip 或 PtIr",
) -> dict:
    """Create a stub CATEGORY dict using GENERIC_PHASES."""
    return {
        "id": cat_id,
        "name": name,
        "name_en": name_en,
        "description": description,
        "completeness": "stub",
        "tip_requirements": tip_requirements,
        "common_issues": [],
        "phases": GENERIC_PHASES,
    }
