"""Optional hardware modules — licensed, but not necessarily present.

The operator has every Nanonis licence and only some of the hardware
(2026-07-13). So MAST ships skills for all of it, and turns almost all of it OFF.

WHY OFF BY DEFAULT IS NOT JUST TIDINESS
=======================================
A skill for hardware that is not plugged in is **worse than no skill at all**:

  * The agent sees it in its tool list, calls it, gets a Nanonis error, and — if
    it is stubborn — calls it again. That is a spin, and StallGuard then has to
    clean up a mess that should never have existed.
  * The tool list is a menu the model reads on **every turn**. instrument_control
    already carries ~280 tools; adding 50 more for a KPFM controller nobody owns
    costs routing accuracy on the tools that DO work, on every single call.

So a disabled module's skills are not merely refused — they are **not in the
agent's tool list at all**. It cannot call what it cannot see. The skills stay in
the SkillRegistry (the manual / GUI executor path still reaches them, which is
right: the operator may want to poke a module they just plugged in), but the
LLM's surface only ever contains hardware that is actually there.

Turning a module on is one switch in 设置 → 硬件模块.

WHAT IS **NOT** HERE
====================
Everything the base Nanonis V5e controller always has — bias, scan, Z-control,
spectroscopy, the lock-in, the piezos, the user outputs, the function generators,
the loggers, the markers. Those are never gated: gating them would mean the agent
could not drive the microscope.

WIRING (mirrors the ``mast.vision.thresholds`` live-read holder)
================================================================
A process-level holder keeps the ACTIVE set. The API/runtime writes it (startup
hydration from SettingsStore + each POST /api/settings); the tool-list builder
reads it. One-way: the skills layer never imports ``mast.webui``.

**A toggle is not live until the tool list is rebuilt.** The agent's tool list is
frozen at graph-build time — exactly the trap that made a deleted composite skill
stay callable (runtime._request_composite_rebuild). So the settings route calls
that same rebuild after flipping a switch, and says so in the UI.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The single persisted settings key. NB: a settings key that is not in
# SettingsStore.KNOWN_KEYS is silently dropped on save — it looks like it works
# and does nothing (that bug once made the admin PIN gate accept any text).
SETTINGS_KEY = "hardware_modules"


@dataclass(frozen=True)
class HardwareModule:
    """One optional module: what it is, what it needs, and which skills it owns."""

    id: str
    name: str                 # operator-facing, Chinese
    hardware: str             # what you must physically own for this to work
    description: str          # what it lets the agent do
    prefixes: tuple[str, ...]  # the Nanonis verb prefixes it covers
    skills: tuple[str, ...]    # the skill names it owns — the gate keys on THESE
    default_on: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# The registry. `skills` is the load-bearing field: build_tools() drops exactly
# these names from the agent's tool list when the module is off.
# ─────────────────────────────────────────────────────────────────────────────

MODULES: tuple[HardwareModule, ...] = (
    HardwareModule(
        id="osci_hr",
        name="高分辨示波器 (OsciHR)",
        hardware="Nanonis Oscilloscope High Resolution 模块",
        description="高分辨采集：触发、校准模式、双通道数据、FFT/PSD。",
        prefixes=("OsciHR_",),
        skills=("ConfigureHighResScope", "RunHighResScope", "GetHighResScopeData",
                "GetHighResScopeStatus"),
    ),
    HardwareModule(
        id="osci_2t",
        name="双通道示波器 (Osci2T) + Signal Chart",
        hardware="Nanonis Osci 2-channel / Signal Chart 模块",
        description="双通道时域采集与信号图表。pump-probe 常用。",
        prefixes=("Osci2T_", "SignalChart_"),
        skills=("ConfigureDualScope", "GetDualScopeData", "ConfigureSignalChart"),
    ),
    HardwareModule(
        id="hs_sweeper",
        name="高速扫描器 (HSSwp)",
        hardware="Nanonis High-Speed Sweeper 模块",
        description="高速一维扫描：任意信号、自动反向、多通道同步采集。",
        prefixes=("HSSwp_",),
        skills=("ConfigureHighSpeedSweep", "RunHighSpeedSweep", "StopHighSpeedSweep",
                "GetHighSpeedSweepStatus"),
    ),
    HardwareModule(
        id="aprf_gen",
        name="任意波形 / 射频发生器 (APRFGen)",
        hardware="Nanonis APRF Generator 模块",
        description="任意波形与射频输出、频率扫描。TERS / pump-probe 相关。",
        prefixes=("APRFGen_",),
        skills=("ConfigureRfGenerator", "StartRfGenerator", "StopRfGenerator",
                "RunRfFrequencySweep", "GetRfGeneratorStatus"),
    ),
    HardwareModule(
        id="kelvin",
        name="Kelvin 控制器 (KPFM) + CPD 补偿",
        hardware="Nanonis Kelvin Controller 模块（KPFM）",
        description="接触电位差测量与补偿：Kelvin 反馈环、偏压限值、CPD 补偿。",
        prefixes=("KelvinCtrl_", "CPDComp_"),
        skills=("ConfigureKelvinController", "SetKelvinControllerOnOff",
                "GetKelvinController", "RunCpdCompensation", "GetCpdCompensation"),
    ),
    HardwareModule(
        id="interferometer",
        name="干涉仪 (Interf)",
        hardware="Nanonis Interferometer 模块（干涉式挠度检测）",
        description="干涉式位移检测的控制环与校准。",
        prefixes=("Interf_",),
        skills=("ConfigureInterferometer", "SetInterferometerOnOff",
                "GetInterferometer"),
    ),
    HardwareModule(
        id="beam_deflection",
        name="光束偏转 (BeamDefl)",
        hardware="Nanonis Beam Deflection 模块（AFM 光杠杆）",
        description="光杠杆的水平/垂直/和信号配置与自动调零。",
        prefixes=("BeamDefl_",),
        skills=("ConfigureBeamDeflection", "GetBeamDeflection",
                "AutoZeroBeamDeflection"),
    ),
    HardwareModule(
        id="laser",
        name="激光模块 (Laser)",
        hardware="Nanonis Laser 模块",
        description="激光开关与功率控制。",
        prefixes=("Laser_",),
        skills=("SetLaserOnOff", "SetLaserPower", "GetLaser"),
    ),
    HardwareModule(
        id="multiprobe",
        name="多探针 (MProbe)",
        hardware="多探针系统（4-probe STM 等）",
        description="每根探针独立的 Z 控制、扫描器、偏压、电流。",
        prefixes=("MProbeZCtrl_", "MProbeScanner_", "MProbeBias_", "MProbeCurrent_"),
        skills=("SetProbeZController", "GetProbeZController", "WithdrawProbe",
                "ConfigureProbeScanner", "MoveProbeXY", "StopProbeScanner",
                "SetProbeBias", "PulseProbeBias", "GetProbeBias", "GetProbeCurrent",
                "ConfigureProbeCurrentGain"),
    ),
    HardwareModule(
        id="pi_controller",
        name="通用 PI 控制器 (PICtrl / GenPICtrl)",
        hardware="Nanonis Generic PI Controller 模块",
        description="把任意信号锁到设定值的通用 PI 环（V5e 与 V5 两代）。",
        prefixes=("PICtrl_", "GenPICtrl_"),
        skills=("ConfigurePiController", "SetPiControllerOnOff", "GetPiController",
                "SetGenericPiOutput", "GetGenericPiController"),
    ),
    HardwareModule(
        id="preamp_mcva5",
        name="MCVA5 前置放大器",
        hardware="MCVA5 preamplifier",
        description="前放的增益 / 耦合 / 温度与状态读取。",
        prefixes=("MCVA5_",),
        skills=("ConfigurePreamp", "GetPreamp"),
    ),
    HardwareModule(
        id="pll_analysis",
        name="PLL 分析 (Zoom FFT / 相位扫描 / 信号分析仪)",
        hardware="Nanonis PLL 模块的分析扩展",
        description="PLL 的 Zoom FFT 频谱、相位扫描、信号分析仪。",
        prefixes=("PLLZoomFFT_", "PLLPhasSwp_", "PLLSignalAnlzr_"),
        skills=("RunPllZoomFft", "GetPllZoomFftData", "RunPllPhaseSweep",
                "StopPllPhaseSweep", "ConfigurePllSignalAnalyzer",
                "GetPllSignalAnalyzerData"),
    ),
    HardwareModule(
        id="oc_sync",
        name="OC Sync",
        hardware="Nanonis OC Sync 模块",
        description="振荡控制同步：相位角与链接。",
        prefixes=("OCSync_",),
        skills=("ConfigureOcSync", "GetOcSync"),
    ),
    HardwareModule(
        id="tip_recorder",
        name="针尖移动记录器 (TipRec)",
        hardware="Nanonis Tip Move Recorder 模块",
        description="记录针尖的移动轨迹到缓冲区。",
        prefixes=("TipRec_",),
        skills=("ConfigureTipRecorder", "GetTipRecorderData"),
    ),
)

MODULE_BY_ID: dict[str, HardwareModule] = {m.id: m for m in MODULES}

# skill name → module id. Built once; the gate is a dict lookup, not a scan.
SKILL_OWNER: dict[str, str] = {
    skill: m.id for m in MODULES for skill in m.skills
}


DEFAULT_ENABLED: frozenset[str] = frozenset(m.id for m in MODULES if m.default_on)


# ─────────────────────────────────────────────────────────────────────────────
# The holder + the gate
# ─────────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_enabled: frozenset[str] = DEFAULT_ENABLED   # read lock-free; swapped under _lock


def enabled_ids() -> frozenset[str]:
    """The module ids currently switched ON. Lock-free (a CPython attribute read
    is atomic — a reader never sees a half-updated set; the writer swaps the
    whole frozenset)."""
    return _enabled


def set_enabled(raw) -> frozenset[str]:
    """Install the active set. Called by the API on startup hydration and on each
    settings write. Accepts the persisted ``{id: bool}`` map, or any iterable of
    ids.

    **Fail-closed on garbage.** A value we cannot read means the stored state is
    unknown, and an unknown state is not "everything is plugged in" — it is the
    defaults, which for every optional module is OFF. Unknown ids are dropped
    with a warning rather than silently kept: a typo'd id that lingers in the
    persisted blob would otherwise look enabled forever in the JSON while gating
    nothing.
    """
    global _enabled
    if isinstance(raw, dict):
        wanted = {str(k) for k, v in raw.items() if bool(v)}
    elif isinstance(raw, (list, tuple, set, frozenset)):
        wanted = {str(x) for x in raw}
    else:
        if raw is not None:
            logger.warning("hardware_modules: 无法解析的开关状态 %r；按默认（全关）", type(raw))
        wanted = set(DEFAULT_ENABLED)

    unknown = wanted - set(MODULE_BY_ID)
    if unknown:
        logger.warning("hardware_modules: 忽略未知模块 id %s", sorted(unknown))
    new = frozenset(wanted & set(MODULE_BY_ID))
    with _lock:
        _enabled = new
    logger.info("hardware_modules: 已启用 %s", sorted(new) or "（无——全部硬件模块关闭）")
    return new


def disabled_skill_names() -> frozenset[str]:
    """Skill names that must NOT reach the agent's tool list."""
    on = enabled_ids()
    return frozenset(s for s, mod in SKILL_OWNER.items() if mod not in on)


def module_states() -> list[dict]:
    """Operator-facing view: every optional module, on/off, and what it needs."""
    on = enabled_ids()
    return [
        {
            "id": m.id,
            "name": m.name,
            "hardware": m.hardware,
            "description": m.description,
            "enabled": m.id in on,
            "skill_count": len(m.skills),
            "skills": list(m.skills),
        }
        for m in MODULES
    ]


__all__ = [
    "HardwareModule", "MODULES", "MODULE_BY_ID", "SKILL_OWNER", "SETTINGS_KEY",
    "DEFAULT_ENABLED", "enabled_ids", "set_enabled", "disabled_skill_names",
    "module_states",
]
