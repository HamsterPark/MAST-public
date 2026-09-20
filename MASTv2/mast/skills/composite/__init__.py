"""Composite skills — multi-step orchestrations built on builtin skills (Phase 4)."""

from mast.skills.composite._base import AbortRequested, CompositeSkill
from mast.skills.composite.tip_pulse import TipPulse
from mast.skills.composite.condition_tip import ConditionTip
from mast.skills.composite.angle_series_calibration import AcquireAngleSeriesForCalibration
from mast.skills.composite.bias_imaging_series import AcquireBiasImagingSeries
from mast.skills.composite.achieve_atomic import AchieveAtomicResolution
from mast.skills.composite.scan_until_atomic import ScanUntilAtomicResolution
from mast.skills.composite.full_scan import FullScan
from mast.skills.composite.grid_sts import GridSTS
from mast.skills.composite.assess_quality import AssessImageQuality
from mast.skills.composite.prescan_check import PreScanCheck
from mast.skills.composite.drift_track import TrackDrift_ReferenceScan
from mast.skills.composite.demo_scan_and_sts import DemoScanAndSTS
from mast.skills.composite.shape_tip_on_surface import ShapeTipOnSurface
from mast.skills.composite.survey_surface import SurveySurface_TileScan
from mast.skills.composite.batch_regions_scan import BatchRegionsScan
from mast.skills.composite.retract_for_sample_change import RetractForSampleChange
from mast.skills.composite.relocate_coarse_xy import RelocateCoarseXY
# 扫图智能脚本化 (2026-07-31, docs/v2/design/scan_intelligence_scripted_rfc.md)
from mast.skills.composite.auto_tilt import AutoTilt, TiltCalibrate
from mast.skills.composite.bias_settle import BiasSettleChange
from mast.skills.composite.execute_scan_plan import ExecuteScanPlan
from mast.skills.composite.scan_at import ScanAt
# 贵金属针尖修整 (2026-08-01, docs/v2/design/tip_conditioning_rfc.md)
from mast.skills.composite.prepare_noble_tip import (
    PokeConditionTip,
    PrepareNobleTip,
    PulseConditionTip,
)
# 特异化针尖锻造 (2026-08-02, docs/v2/design/special_tip_forging.md)
from mast.skills.composite.make_special_tip import (
    MakeAtomicResolutionTip,
    MakeSpectroscopyTip,
)
from mast.skills.composite.publication_frame import ScanPublicationFrame
# Au(111) 全流程修针外环 (2026-08-05,
# docs/v2/design/au111_tip_forge_uninterrupted.md)
from mast.skills.composite.forge_au_tip import ForgeAuTip
# 原子相三态裁决壳 (2026-08-14, S2 偏压序列设计 D1)
from mast.skills.composite.verify_atomic_resolution import VerifyAtomicResolution
# 实验编排的四个执行体 (2026-08-14)。
# 这几行不是「照惯例登记」,是承重的:冻结包里 ``pkgutil.walk_packages``
# 枚举不到任何东西(``core/registry.py`` 有实测记录),discover 只能靠各 skills 包
# ``__init__`` 主动 import 过的模块兜底。漏一行的后果不是报错,是**打包版里那个
# 技能整块消失、agent 得到「技能未找到」,而开发机全绿** —— 2026-07 那次 141 个
# 技能就是这么消失的。
from mast.skills.composite.cross_point_tip_check import CrossPointTipCheck
from mast.skills.composite.spectroscopy_at_positions import SpectroscopyAtPositions
from mast.skills.composite.line_sts_across_wall import LineSTSAcrossWall
from mast.skills.composite.search_domain_boundary import SearchDomainBoundary
from mast.skills.composite.atomic_bias_series import AtomicBiasSeries
from mast.skills.composite.sts_condition_series import STSConditionSeries
# 论文复现:横向操纵 (2026-09)
from mast.skills.composite.move_atom_to import MoveAtomTo

__all__ = [
    "AchieveAtomicResolution",
    "MoveAtomTo",
    "AcquireBiasImagingSeries",
    "AcquireAngleSeriesForCalibration",
    "ScanUntilAtomicResolution",
    "AbortRequested",
    "CompositeSkill",
    "TipPulse",
    "ConditionTip",
    "FullScan",
    "GridSTS",
    "AssessImageQuality",
    "PreScanCheck",
    "TrackDrift_ReferenceScan",
    "DemoScanAndSTS",
    "ShapeTipOnSurface",
    "SurveySurface_TileScan",
    "BatchRegionsScan",
    "RetractForSampleChange",
    "RelocateCoarseXY",
    # 贵金属针尖修整
    "PulseConditionTip",
    "PokeConditionTip",
    "PrepareNobleTip",
    # 特异化针尖锻造
    "MakeSpectroscopyTip",
    "MakeAtomicResolutionTip",
    "ScanPublicationFrame",
    # Au(111) 全流程修针外环
    "ForgeAuTip",
    # 原子相三态裁决
    "VerifyAtomicResolution",
    # 实验编排的四个执行体 (2026-08-14)
    "CrossPointTipCheck",
    "SpectroscopyAtPositions",
    "LineSTSAcrossWall",
    "SearchDomainBoundary",
    # 逐偏压原子序列 + 逐偏压账 (S2-B)
    "AtomicBiasSeries",
    # 同一位置逐条件取谱 (S4-C)
    "STSConditionSeries",
    # 扫图智能脚本化
    "ScanAt",
    "AutoTilt",
    "TiltCalibrate",
    "BiasSettleChange",
    "ExecuteScanPlan",
]
