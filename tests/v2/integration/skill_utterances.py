"""Natural-language utterances (中文 + English) that drive the MAST orchestrator
to call each agent-reachable skill — the corpus for the 1000-round end-to-end
coverage/stability test (test_orchestrator_skill_coverage.py).

Design
------
Each entry maps a registered skill name → one Chinese + one English utterance,
phrased as an operator would speak to the supervisor so it routes to the right
sub-agent (instrument_control / data_processing / …) and that sub-agent calls the
named skill. Utterances are EXPLICIT (they echo the skill's job) so LLM routing is
reliable, but phrased naturally — not "call SkillX".

Coverage target = AGENT-REACHABLE skills only. The IC agent exposes all builtins +
composites; data_processing exposes its ~10 @tool wrappers; shared meta-tools cover
experiment/sample lifecycle. The 34 paper BaseSkills are mostly NOT individually
agent-callable (only via data_processing wrappers / workflows) — so they are not in
this corpus; the harness reports them as "unreachable by conversation".

`HAND` = hand-authored high-value/action/lifecycle utterances. The deep Nanonis
config getters/setters are filled in by `auto_utterances()` (templated from each
skill's metadata) in skill_utterances_auto.py, merged by the harness, so every
agent-reachable skill has an utterance without 200 hand-typed mechanical lines.
"""
from __future__ import annotations

# ── experiment / sample lifecycle (shared meta-tools) — the test MUST start here ──
LIFECYCLE: dict[str, dict[str, str]] = {
    "start_experiment": {
        "zh": "新建一个实验，名字叫「Si(111) 自治测试」，目标是验证全流程。",
        "en": "Start a new experiment called 'Si(111) autonomy test', goal: validate the full pipeline.",
    },
    "start_sample": {
        "zh": "标记一个新样品，名字叫「Si-111-7x7 #1」，类型是半导体。",
        "en": "Register a new sample named 'Si-111-7x7 #1', type semiconductor.",
    },
    "rename_experiment": {
        "zh": "把当前实验改名为「Si(111) 自治测试 v2」。",
        "en": "Rename the current experiment to 'Si(111) autonomy test v2'.",
    },
    "rename_sample": {
        "zh": "把当前样品改名为「Si-111-7x7 主样」。",
        "en": "Rename the current sample to 'Si-111-7x7 main'.",
    },
    "end_sample": {
        "zh": "结束当前样品。",
        "en": "End the current sample.",
    },
    "end_experiment": {
        "zh": "结束当前实验。",
        "en": "End the current experiment.",
    },
    "report_upgrade_idea": {
        "zh": "记一条升级建议到心愿单：希望有一个自动换样品的 skill。",
        "en": "Log an upgrade idea to the wishlist: we want an auto sample-change skill.",
    },
    "get_latest_scan_info": {
        "zh": "最近一张扫描图的文件路径是什么？",
        "en": "What's the file path of the most recent scan?",
    },
}

# ── instrument_control — read/state ──
IC_READ: dict[str, dict[str, str]] = {
    "GetBias": {"zh": "现在的偏压是多少？", "en": "What's the current bias voltage?"},
    "GetCurrent": {"zh": "读一下当前的隧道电流。", "en": "Read the current tunneling current."},
    "GetSetpoint": {"zh": "当前的电流设定点是多少？", "en": "What's the current setpoint?"},
    "GetZPosition": {"zh": "读一下当前 Z 压电的位置。", "en": "Read the current Z piezo position."},
    "GetScanFrame": {"zh": "现在扫描框的中心、大小和角度是多少？", "en": "Read the current scan frame center, size and angle."},
    "GetScanBuffer": {"zh": "读一下当前扫描缓冲区的通道和像素。", "en": "Read the current scan buffer channels and pixels."},
    "GetScanSpeed": {"zh": "现在的扫描速度参数是多少？", "en": "What are the current scan speed parameters?"},
    "GetScanXYPosition": {"zh": "读一下扫描器当前的 X/Y 位置。", "en": "Read the scanner's current X and Y position."},
    "GetZCtrlGain": {"zh": "读一下 Z 控制器的 P/I 增益和时间常数。", "en": "Read the Z controller P/I gains and time constant."},
    "MotorGetPos": {"zh": "读一下粗动马达的位置。", "en": "Read the coarse motor position."},
    "GetAutoApproachStatus": {"zh": "自动逼近现在在运行吗？", "en": "Is auto-approach currently running?"},
    "GetSafeTipStatus": {"zh": "读一下 SafeTip 保护状态。", "en": "Read the SafeTip protection status."},
    "GetSessionPath": {"zh": "当前 Nanonis 会话文件夹路径是什么？", "en": "What is the current Nanonis session folder path?"},
    "GetLatestScanFile": {"zh": "定位最近写入的 .sxm 扫描文件。", "en": "Locate the most recently written .sxm scan file."},
    "ListSignalChannels": {"zh": "列出 Nanonis 的 128 个可用信号。", "en": "List the 128 available Nanonis signals."},
    "MonitorCurrent": {"zh": "用 0.5 秒高速采一下隧道电流，给我最小/最大/均值。", "en": "Sample the tunneling current at high rate for 0.5 s, report min/max/mean."},
}

# ── instrument_control — bias / current ──
IC_BIAS: dict[str, dict[str, str]] = {
    "SetBias": {"zh": "把偏压设成 0.5 伏。", "en": "Set the bias voltage to 0.5 V."},
    "SetBiasRamp": {"zh": "把偏压从 0.1 伏缓慢斜坡到 1.0 伏。", "en": "Ramp the bias from 0.1 V up to 1.0 V."},
    "BiasPulse": {"zh": "在当前位置打一个 3 伏、100 毫秒的偏压脉冲。", "en": "Apply a 3 V, 100 ms bias pulse at the current position."},
    "SetSetpoint": {"zh": "把电流设定点设成 100 皮安。", "en": "Set the tunneling current setpoint to 100 pA."},
}

# ── instrument_control — Z controller / approach / tip ──
IC_ZTIP: dict[str, dict[str, str]] = {
    "ZControllerOnOff": {"zh": "打开 Z 反馈控制器。", "en": "Turn the Z controller feedback on."},
    "TryEngageController": {"zh": "试着进针：开反馈看看能不能建立隧道电流，不行就退回来。", "en": "Try to engage the tip: turn on feedback and see if tunneling establishes; back off if not."},
    "AutoApproach": {"zh": "开始自动逼近，让针靠近表面。", "en": "Start auto-approach to bring the tip toward the surface."},
    "WithdrawTip": {"zh": "把针完全退出表面。", "en": "Withdraw the tip fully from the surface."},
    "SafeRetract": {"zh": "安全退针。", "en": "Safely retract the tip."},
    "EmergencyRetract": {"zh": "紧急退针！", "en": "Emergency retract the tip now!"},
    "SetZPosition": {"zh": "反馈关掉后，把 Z 压电直接设到 -2 纳米。", "en": "With feedback off, set the Z piezo directly to -2 nm."},
    "TipShape": {"zh": "对针尖做一次硬件整针处理。", "en": "Run the hardware tip-shaper to condition the tip."},
    "MonitorCurrentFFT": {"zh": "对隧道电流做个软件 FFT 看噪声谱。", "en": "Run a software FFT on the tunneling current to see the noise spectrum."},
}

# ── instrument_control — scanning ──
IC_SCAN: dict[str, dict[str, str]] = {
    "ConfigureScan": {"zh": "把扫描框配成中心原点、50 纳米见方、采 Z 和电流通道。", "en": "Configure the scan frame: centered at origin, 50 nm square, channels Z and Current."},
    "SetScanSpeed": {"zh": "把扫描速度设成每行 0.1 秒。", "en": "Set the scan speed to 0.1 s per line."},
    "StartScan": {"zh": "开始扫描。", "en": "Start the scan."},
    "WaitScanComplete": {"zh": "等扫描结束，最多等 60 秒。", "en": "Wait for the scan to finish, up to 60 s."},
    "StopScan": {"zh": "停止当前扫描。", "en": "Stop the current scan."},
    "SaveScan": {"zh": "保存当前扫描数据到文件。", "en": "Save the current scan data to file."},
    "MoveToXY": {"zh": "用 Follow-Me 把针移到 (10 纳米, 5 纳米)。", "en": "Move the tip to (10 nm, 5 nm) using Follow-Me."},
    "GrabScanFrameData": {"zh": "抓取当前扫描第 0 通道的帧数据存成 npy。", "en": "Grab channel 0's scan frame data and save it as .npy."},
    "CheckScanForCrash": {"zh": "检查刚才那张扫描有没有撞针。", "en": "Check the just-acquired scan for a tip crash."},
    "ComputeDriftVector": {"zh": "把当前帧和参考图比对，算一下漂移矢量。", "en": "Cross-correlate the current frame against the reference image and compute the drift vector."},
    "FindFlatRegion": {"zh": "在这张 .sxm 里滑窗找一块最平的区域。", "en": "Slide a window over this .sxm and find the flattest region."},
    "AssessClusterRoundness": {"zh": "评估这张小图里最大亮团的圆度。", "en": "Assess the roundness of the largest bright protrusion in this small scan."},
    "ScanBackgroundPaste": {"zh": "把当前扫描缓冲粘贴成背景。", "en": "Paste the current scan databuffer into the background."},
}

# ── instrument_control — STS / Z-spectroscopy ──
IC_SPEC: dict[str, dict[str, str]] = {
    "ConfigureSTS": {"zh": "配置 STS 偏压扫描：从 -2 伏到 2 伏，200 点。", "en": "Configure STS: sweep bias from -2 V to 2 V, 200 points."},
    "AcquireSTS": {"zh": "在当前位置采一条 STS 谱。", "en": "Acquire one STS spectrum at the current tip position."},
    "StopSTS": {"zh": "停止当前的偏压谱测量。", "en": "Stop the current bias spectroscopy measurement."},
    "ConfigureZSpectr": {"zh": "配置 Z 谱：Z 偏移和扫描距离。", "en": "Configure Z spectroscopy: Z offset and sweep distance."},
    "AcquireZSpectr": {"zh": "在当前位置采一条 Z 谱。", "en": "Acquire a Z spectrum at the current tip position."},
    "StopZSpectr": {"zh": "停止 Z 谱测量。", "en": "Stop the Z spectroscopy measurement."},
    "AcquireBiasSweep": {"zh": "在当前针位做一次偏压扫描。", "en": "Acquire a bias sweep at the current tip position."},
}

# ── instrument_control — motor (coarse) ──
IC_MOTOR: dict[str, dict[str, str]] = {
    "MotorMove": {"zh": "粗动马达向 X 正方向走 100 步。", "en": "Step the coarse motor 100 steps in the x+ direction."},
    "StopMotor": {"zh": "紧急停止马达。", "en": "Emergency-stop the motor."},
    "SetMotorFreqAmp": {"zh": "设置马达的频率和幅度用于粗定位。", "en": "Set the motor frequency and amplitude for coarse positioning."},
}

# ── data_processing — the agent's @tool surface ──
DATA_PROC: dict[str, dict[str, str]] = {
    "load_scan": {"zh": "离线加载这个 .sxm 文件，给我通道和头信息摘要。", "en": "Load this .sxm offline and summarize its channels and header."},
    "fft_2d": {"zh": "对这张扫描图做个二维 FFT。", "en": "Run a 2D FFT on this scan image."},
    "plane_subtract": {"zh": "对这张图做平面扣除。", "en": "Plane-subtract this image."},
    "detect_defects": {"zh": "在这张图里检测亮/暗缺陷。", "en": "Detect bright/dark defects in this image."},
    "fit_sts_peaks": {"zh": "对这条 dI/dV 曲线拟合高斯峰。", "en": "Fit Gaussian peaks to this dI/dV curve."},
    "mosaic_scans": {"zh": "把这几张 .sxm 按真实 xy 拼成一张大图。", "en": "Stitch these .sxm files into one mosaic by their real xy."},
    "auto_crop_scan": {"zh": "把这张图未扫描的纯色边裁掉。", "en": "Crop the unscanned solid-color border off this image."},
    "diff_scans": {"zh": "对比这两张同区域的扫描，给我差分图。", "en": "Diff these two scans of the same area and give me the change map."},
    "compose_montage": {"zh": "把这几张图拼成一张带比例尺的多面板出版图。", "en": "Compose these images into a multi-panel publication figure with scale bars."},
}

# ── composites (IC) ──
COMPOSITES: dict[str, dict[str, str]] = {
    "FullScan": {"zh": "做一次完整扫描：配好区域、设速度、启动、等结束。", "en": "Do a full scan: configure the area, set speed, start, and wait for completion."},
    "TipPulse": {"zh": "用偏压脉冲修一下针。", "en": "Condition the tip with a bias pulse."},
    "GridSTS": {"zh": "在 3×3 的栅格上逐点采 STS。", "en": "Acquire STS on a 3x3 grid, point by point."},
    "ConditionTip": {"zh": "自动修针：脉冲、扫描、FFT 评质，循环到达标。", "en": "Auto-condition the tip: pulse, scan, FFT quality check, repeat until sharp."},
    "PreScanCheck": {"zh": "扫一条单行快速预检一下针况。", "en": "Run a quick single-line pre-scan to check tip quality."},
    "ShapeTipOnSurface": {"zh": "在样品上做整针：找平点、扎针、评圆度、不行换点重试。", "en": "Shape the tip on the surface: find a flat spot, plunge, assess roundness, retry."},
    "SurveySurface_TileScan": {"zh": "用 3×3 瓦片巡扫一遍表面建全局概览。", "en": "Survey the surface with a 3x3 tile scan for a global overview."},
    "TrackDrift_ReferenceScan": {"zh": "按参考图跟踪样品漂移并补偿扫描框。", "en": "Track sample drift against a reference scan and compensate the frame."},
    "BatchRegionsScan": {"zh": "按这个区域清单逐个扫描。", "en": "Scan this list of regions one after another."},
    "DemoScanAndSTS": {"zh": "跑一次演示：扫描加多点 STS。", "en": "Run the demo: a scan plus multi-point STS."},
    "RunGridExperiment": {"zh": "用 Nanonis Pattern 模块跑栅格谱实验。", "en": "Run grid spectroscopy via the Nanonis Pattern module."},
}

# Merged hand-authored corpus (lifecycle first — the test opens exp+sample from here).
HAND: dict[str, dict[str, str]] = {
    **LIFECYCLE, **IC_READ, **IC_BIAS, **IC_ZTIP, **IC_SCAN,
    **IC_SPEC, **IC_MOTOR, **DATA_PROC, **COMPOSITES,
}

__all__ = ["HAND", "LIFECYCLE", "IC_READ", "IC_BIAS", "IC_ZTIP", "IC_SCAN",
           "IC_SPEC", "IC_MOTOR", "DATA_PROC", "COMPOSITES"]
