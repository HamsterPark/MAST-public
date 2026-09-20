"""Instrument profile: configured hardware facts and learned calibration.

The skill and agent layers share this dependency-light data and rendering layer.
Approach/retract direction must be configured for the target instrument; direction
checks use raw Z motion rather than current after the junction has opened.
Contact dI/dV depends on tip state, modulation and bias. Successful approaches
update that calibration with EWMA and an injected persistence sink.
Runtime writers validate incoming values; missing facts remain distinguishable
from configured defaults. No measured instrument profile is bundled here.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Config fields — hard instrument facts the operator sets once ─────────────
# key -> (中文标签, 单位, 强制类型, (lo, hi), 默认值). Unknown keys are DROPPED
# on set. Ranges keep a fat-fingered value from reaching the approach/retract
# logic — but note these are CONFIG (an intent), not the safety net: the
# retract composite's per-step Z-recede self-check is the real anti-crash guard.
_CONFIG_SPEC: dict[str, tuple[str, str, type, tuple[float, float], Any]] = {

    "lockin_signal_index": (
        "dI/dV lock-in 信号索引", "", int, (0, 127), None),
    # 解调 X / Y 各自走哪一路 RT 信号 —— **接线事实,软件观测不到**,只能由用户
    # 填。AutoPhase 在两者缺一时直接拒绝而不是猜:猜错读到的是另一路信号,而由它
    # 算出来的相位角看上去一样合理,然后会被写进硬件。
    "lockin_x_signal_index": (
        "解调 X 的 RT 信号索引(AutoPhase 用)", "", int, (0, 127), None),
    "lockin_y_signal_index": (
        "解调 Y 的 RT 信号索引(AutoPhase 用)", "", int, (0, 127), None),
    "lockin_mod_amp_v": (
        "dI/dV 调制幅度", "V", float, (0.0, 1.0), 0.02),
    "lockin_mod_freq_hz": (
        "dI/dV 调制频率", "Hz", float, (0.0, 1.0e5), 973.0),
    "z_recede_min_nm": (
        "退针方向自检: Z 伸长超过此阈值判定为远离", "nm", float, (0.0, 1.0e4), 1.0),

    "z_settle_timeout_s": (
        "退针自检: 等 Z 反馈稳定的预算上限", "s", float, (0.5, 300.0), 20.0),
    "retract_total_steps": (
        "换样品退针总步数(粗动马达)", "步", int, (1, 1_000_000), 3000),
    "retract_step_max": (
        "单次粗动最大步数(Nanonis 硬件上限 1000)", "步", int, (1, 1000), 1000),
    "approach_didv_engage_frac": (
        "进针 dI/dV 判据: 达到标定值的此比例算接触在望", "", float, (0.0, 2.0), 0.7),

    "approach_p_gain_m": (
        "进针 Z 比例增益 (Nanonis 面板 Proportional)", "m",
        float, (1.0e-15, 1.0e-6), None),
    "approach_i_gain_m_per_s": (
        "进针 Z 积分增益 (Nanonis 面板 Integral)", "m/s",
        float, (1.0e-12, 1.0e-3), None),
    "approach_setpoint_a": (
        "进针电流设定点", "A", float, (1.0e-12, 1.0e-7), None),

    "approach_steps_per_cycle": (
        "Auto Approach 每轮步数 (Number of Pulses)——**镜像值**:抄自 Nanonis 面板,"
        "TCP 读不回也写不了;面板上改了这里不会自动跟,核对靠 "
        "/api/diagnostics/screenshot", "步", int, (1, 1000), None),
    "approach_expected_steps": (
        "进针预计总步数(用户经验值,用于判断是否异常)——**镜像/经验值**,"
        "同样无法回读核对", "步",
        int, (1, 1_000_000), None),
    # qPlus 振荡振幅 —— 独立于电流的撞针判据。
    # 基线是「针尖自由振荡时的振幅」，撞针判据是「当前振幅 / 基线 < 10%」。
    # 上界给到 1e3 是因为不同机器上这条信号的单位不一样（V / nm / 无量纲刻度都见过），
    # 判据用的是比值，所以量纲无关；这里只拦明显不是读数的值。
    "qplus_amplitude_baseline": (
        "qPlus 自由振荡振幅基线(撞针判据的分母)", "", float, (0.0, 1.0e3), None),
    "qplus_amplitude_signal_index": (
        "qPlus 振幅信号索引(-1 = 按通道名自动查找)", "", int, (-1, 127), -1),
    # ── 信号链 (2026-07-31, 针尖登记同批) ─────────────────────────────
    # 前置放大器的跨阻增益:电流 = 读数电压 / 增益。Nanonis 侧
    # ``Current_GainsGet`` 只给得到**索引**和一串标签,索引→实际 V/A 倍数的
    # 映射既不在 Nanonis 也不在本仓任何地方 —— 只能登记。
    # 它错一个量级,MAST 报出去的每一个电流值就整体错一个量级,而且是静默的
    # (readback.GetMiscInstrumentConfig 的 docstring 早就写下了这句警告)。
    "preamp_gain_v_per_a": (
        "前置放大器跨阻增益(电流 = 读数电压 / 此值)", "V/A",
        float, (1.0e3, 1.0e13), None),
    # 前放的**可测最大电流** —— 与上面的增益是同一件事的两种说法
    # (满量程 ≈ ±10 V ÷ 增益)。分开存两个数是刻意的,理由有两条:
    #
    #   1. 用户记得住的是「±10 nA 档」而不是「1e9 V/A」。问他记得住的那个,
    #      答案才可靠。
    #   2. 两个都填就能互相验一次 —— 差一个量级立刻看得出来是哪个填错了
    #      (mast.core.instrument_init.preamp_consistency)。
    #
    # 它是**两条安全线的物理依据**,而在 2026-08-03 之前这两条线都只能靠人手动
    # 从前放量程折算再手动填进另外两个地方:
    #   * SafetyLimits.setpoint_max_a —— 高于前放量程的设定点物理上达不到,
    #     反馈环拿不到目标电流就会一路把 Z 推向样品直到撞针;
    #   * current_monitor.cm_sat_current_a —— 超过满量程的读数不是测量值,是贴轨。
    # 导出**只给建议值**,不自动写入(见 instrument_init.derived_from_preamp:
    # 一条被程序悄悄改过的安全上限,就不再是用户声明过的那条线)。
    "preamp_full_scale_a": (
        "前置放大器满量程电流(可测的最大电流)", "A",
        float, (1.0e-12, 1.0e-2), None),
    # ── 扫描地图避让半径 (2026-07-30) ─────────────────────────────────
    # How much surface each kind of damage/contamination event costs us. These
    # are the operator's own numbers: how far the debris from a tip-forming
    # plunge actually spreads is a property of THIS tip, THIS sample and THIS
    # temperature, and only they have watched it. The defaults are a starting
    # point, not a measurement — the whole point of putting them here is that
    # experience overrides them.
    #
    # Read by mast.io.map_analysis to decide where the tip may still go, so a
    # value that is too small silently walks the tip back into a ruined patch.
    "avoid_radius_tip_shape_nm": (
        "修针尖避让半径(该点周围多大范围内不再扫图)", "nm", float,
        (0.0, 1.0e5), 30.0),
    # 计划扫描范围要生成得更分散。
    #
    # 这个量一直存在（``map_analysis.AnalysisConfig.point_spacing_factor``），
    # 只是从来没有人设过它 —— ``map_scope.build_analysis_config`` 不传，于是永远
    # 是 1.2。1.2 的意思是「相邻帧留一条缝、不重叠」，也就是**尽可能挨着排**，
    # 正是要求的不够分散。调到 3 就是每帧之间留两帧的空。
    #
    # 两个策略（center_first / perimeter_inward）都受它影响，所以这是「分散程度」
    # 这一个旋钮，不是第三种模式。真正的「最大分散」（最远点 / 泊松盘采样）是另
    # 一件事，需要新算法，见报告。
    "scan_spacing_factor": (
        "计划扫描点的间距(相邻帧中心相距几个扫描框;1.2=紧挨着,3=留两帧空)", "",
        float, (1.0, 20.0), 1.2),
    "avoid_radius_pulse_nm": (
        "电脉冲避让半径", "nm", float, (0.0, 1.0e5), 150.0),
    "avoid_radius_crash_nm": (
        "撞针避让半径", "nm", float, (0.0, 1.0e5), 150.0),
    "avoid_radius_approach_nm": (
        "进针扎痕避让半径(仅当「进针会扎表面」时生效)", "nm", float,
        (0.0, 1.0e5), 200.0),
    # Optional, and honestly weak: a coarse piezo actuator has no position
    # feedback and its step size drifts with amplitude/temperature/load. Used
    # ONLY to annotate a coarse-move marker with a rough "we travelled about
    # this far"; nothing navigates by it. Left None → the annotation is omitted
    # rather than invented.
    "xy_motor_step_m": (
        "XY 粗动单步位移标定(可选;开环仅供估算)", "m", float, (0.0, 1.0e-3), None),
    # ── XY 粗动换区 (2026-07-31) ──────────────────────────────────────
    # 横向粗动前必须用**粗动马达**把针尖退开。压电退针只有 1–2 µm 余量,而样品台
    # 侧滑时的垂直跳动、样品倾斜、以及针尖本身的长度都远不止这个数。
    # 100 步是个起点,真值只能实机标(移一次、扫一张图、看有没有刮痕)。
    "xy_prewithdraw_steps": (
        "横向粗动前的粗动退针步数(清障)", "步", int, (0, 100_000), 100),
    # 分块移动:每块之后回读电流(+qPlus 振幅)+重评真空。块越小看护越密,
    # 但每块都有一次 TCP 往返,太小会把一次换区拖成几分钟。
    "xy_move_chunk_steps": (
        "横向粗动分块步数(每块之后做一次看护)", "步", int, (1, 1000), 50),
    # 两个站点之间的最小步距。必须让新区域**完全**跳出压电量程(±1.5 µm),
    # 否则「换了区」其实只是把旧区域挪进视野。同样只能实机标。
    "xy_site_spacing_steps": (
        "粗动站点最小间距", "步", int, (1, 1_000_000), 200),

    "au_step_pm": (
        "本机读出的 Au(111) 单原子台阶高度(多针尖判据用)", "pm",
        float, (50.0, 500.0), 235.5),
    "xy_axis_step_budget": (
        "单轴粗动行程预算(步)", "步", int, (1, 10_000_000), 10_000),
    # 每步位移的相对不确定度。开环步进器的步长随驱动幅度/温度/负载漂移,
    # 里程表因此是个带模糊半径的估计,不是坐标。低温下这个值应当更大。
    "xy_step_uncertainty_frac": (
        "粗动单步位移的相对不确定度(画模糊半径用)", "", float, (0.0, 1.0), 0.3),
    # ── 粗动真空互锁 (2026-07-31) ─────────────────────────────────────
    # 在中间真空区(约 1e-3 … 10 mbar = 0.1 … 1000 Pa,Paschen 极小值附近)给粗动
    # 压电加几百伏会打火,电弧沿绝缘层爬过去就把叠堆废了。抽气和放气途中正好穿过
    # 这个区间 —— 也正是最有人想动东西的时候。
    #
    # 默认上限 1e-2 Pa 比放电区下沿低一个数量级,而且在冷阴极规的量程之内;
    # 任何真实的 UHV STM 都远低于它。
    "coarse_motion_max_pressure_pa": (
        "允许粗动的压强上限", "Pa", float, (0.0, 1.0e5), 1.0e-2),
    # 几分钟前的读数不能给现在授权 —— 抽/放气时压强变化很快。
    "vacuum_reading_max_age_s": (
        "真空计读数最长可用时长(超过即视为不可信)", "s", float, (1.0, 3600.0), 60.0),
    # 规的**可用量程**。判据本身与规的型号无关 —— 型号相关的只有这两个数:
    # 读数只有落在量程内才算证据,出了量程(两端都算)规就不是在测量而是在饱和,
    # 饱和读数不是数据。DL-7 = 5e-8 … 1e-1 Pa。
    #
    # 上端:到满量程附近意味着「超量程、判断不了」而不是「刚好卡在阈值下」——
    # DL-7 的帧解析不校验指数位,超量程可能解出一个看着合理的小数。
    "vacuum_gauge_full_scale_pa": (
        "真空计满量程(到此附近视为超量程)", "Pa", float, (1.0e-9, 1.0e6), 1.0e-1),
    # 下端:规触底时发出的数(下限值/0/噪声)看起来正好像「真空非常好」。
    # 这在 DL-7 上无害(下限 5e-8 Pa 远低于允许上限,触底本来就意味着更安全),
    # 但在**粗糙真空规**(Pirani / 电容薄膜规,下限 ~0.5–1 Pa)上是 fail-open:
    # 触底只说明「低于 1 Pa」,而那包含 0.5 Pa —— 正在放电区里。
    # 所以判据不是「有没有触底」,而是「这只规的下限本身是否已经低于允许上限」。
    "vacuum_gauge_min_pa": (
        "真空计量程下限(触底判据;粗糙真空规填 ~1 Pa)", "Pa", float,
        (1.0e-12, 1.0e6), 5.0e-8),
    # ── 扫描 / 自动调平的仪器事实 (2026-07-30) ─────────────────────────
    # 设计文档 docs/v2/design/scan_intelligence_scripted_rfc.md
    #
    # 这几个是「换一台仪器就要重填」的量,所以在这里,而不是 experiment_prefs
    # (那里装的是「换一个用户就会变」的偏好)。
    #
    # z_range_m 是调平触发判据的分母:判据统一成「这一帧的斜坡吃掉多少 Z 量程」
    # (z_span = L·tanθ),这样同一个角度在 1 µm 帧和 10 nm 帧上自动给出不同的
    # 紧迫程度,不需要为粗扫/精扫各设一个角度阈值。
    "z_range_m": (
        "Z 压电总量程", "m", float, (1.0e-9, 1.0e-4), 1.5e-6),
    # 单轴倾斜补偿的绝对上限。压电倾斜补偿把扫描平面转过来,转过头会吃掉 XY
    # 行程并让 Z 在帧角上打满;5° 对任何 STM 都已经是很大的失配角了。
    "tilt_limit_deg": (
        "压电倾斜补偿绝对上限(单轴)", "°", float, (0.0, 45.0), 5.0),
    # 针尖横向速度上限。像素数、每线时间、帧宽单独看都合法,乘起来才知道针尖
    # 扫得多快 —— 这类组合约束由 scan_resolver 检查。
    "v_tip_max_m_s": (
        "针尖横向扫描速度上限", "m/s", float, (1.0e-12, 1.0e-3), 2.0e-6),
    # Z 读数的噪声底。留空 = 每次从数据现估(RANSAC 内点阈、圆拟合的台阶否决、
    # PI 整定验证都要用它)。填了就当本机的基准值,省掉每次估计。
    "z_noise_floor_m": (
        "Z 噪声底(留空 = 每次从数据估计)", "m", float, (0.0, 1.0e-9), None),
}

# Enumerated config — stored value MUST be one of the choices; else dropped.
# key -> (中文标签, choices, {value: 显示文本}, 默认值).
_CHOICE_SPEC: dict[str, tuple[str, tuple[str, ...], dict[str, str], str]] = {
    "retract_motor_dir": (
        "退针方向(粗动马达 远离样品)", ("z+", "z-"),
        {"z+": "Z+ (Nanonis 标准: 远离样品)", "z-": "Z- (反向装置)"},
        "z+"),
    "z_extend_sign": (
        "压电伸长(趋向样品)对应 Z 读数符号", ("+1", "-1"),
        {"+1": "伸长时 Z 增大", "-1": "伸长时 Z 减小"},
        "+1"),
    # ── 扫描地图策略能力 (2026-07-30) ─────────────────────────────────
    # Whether this rig can move to a fresh patch of surface at all. On a rig
    # without lateral coarse motion the piezo scan range is the ENTIRE sample
    # you will ever see until someone physically pulls the sample out, so
    # spending it well matters far more.
    "xy_coarse_motion": (
        "是否有 XY 粗动马达(可换区)", ("yes", "no"),
        {"yes": "有 — 可粗动换到新区域",
         "no": "无 — 只能用当前压电范围内的表面(换区靠插拔样品)"},
        "yes"),
    # Some approach mechanisms tap the surface on the way in; good ones do not.
    # "unknown" is deliberately treated as "yes" downstream: the cost of
    # avoiding a dimple that was never made is a couple hundred nm of surface,
    # while the cost of scanning one is a wasted image and possibly the tip.
    "approach_damages_surface": (
        "进针是否会在表面留下扎痕", ("yes", "no", "unknown"),
        {"yes": "会 — 避开进针点",
         "no": "不会(好机器) — 进针点只作历史记录",
         "unknown": "未知 — 按保守处理(同「会」)"},
        "unknown"),
    # ── 进针要不要有人看着 (2026-08-04) ────────────────────────────────
    # 与上一条描述两种不同风险：
    #   * approach_damages_surface —— 进去时会不会点一下表面(代价:一片表面);
    #   * 本条 —— 这台机器的 Auto Approach 会不会**扎针**(代价:针,以及之后
    #     几小时的修针)。
    #
    # 后者决定「agent 敢不敢自己发起进针」。⚠️ **本字段目前只注入给模型,不是硬门**
    # —— HITL 门是**构建时**从 safety_level 派生的(`AutoApproach` 现为 AUTO),
    # 运行时字段改不了它。要做成硬门需要在 AutoApproach 的入口加软门 + 重建图,
    # 那是一次会改变硬件行为的改动,应当由用户拍板,不该随初始化页面顺手带上。
    # 在那之前,如实告诉模型「这台机器进针必须有人在场」比什么都不说强,
    # 而这一行本身也是**事实**不是劝说(见 既有教训:
    # 物理安全判据属于要保留的事实那一类)。
    "approach_supervision": (
        "进针是否需要有人在场", ("unattended", "attended", "unknown"),
        {"unattended": "可无人值守 — 本机 Auto Approach 不扎针",
         "attended": "必须有人在场 — 本机进针会撞/扎针,不可自主发起",
         "unknown": "未知 — 按保守处理(同「必须有人在场」)"},
        "unknown"),

    "vacuum_interlock_mode": (
        "粗动真空互锁模式", ("gauge_or_attest", "gauge_only", "off"),
        {"gauge_or_attest": "真空计优先,读不到时可由用户签署(默认)",
         "gauge_only": "只认真空计 — 读不到就绝不粗动",
         "off": "关闭阻断 — 仍记录裁决,风险由人承担"},
        "gauge_or_attest"),
    "scan_path_strategy": (
        "扫描选点策略", ("auto", "center_first", "perimeter_inward"),
        {"auto": "自动 — 按有无 XY 粗动推导",
         "center_first": "中心优先 — 压电蠕变最小",
         "perimeter_inward": "外圈→内圈 — 可用面积利用最大化"},
        "auto"),

    "lockin_readout_form": (
        "lock-in 解调形式(登记的那个信号是什么)", ("xy", "r_phi", "unknown"),
        {"xy": "X / Y（带符号投影 —— 需要先把相位调到信号全落在 X 上）",
         "r_phi": "R / Φ（幅度 —— 恒为正，随接近单调增大）",
         "unknown": "未声明 — 判据只报绝对值，不声称它是幅度"},
        "unknown"),
    "bias_applied_to": (
        "偏压加在哪一侧", ("sample", "tip", "unknown"),
        {"sample": "样品 — 常规约定(正偏压探测样品空态)",
         "tip": "针尖 — 符号与常规整体反号",
         "unknown": "未声明 — 注入块会明确告知模型「不知道」"},
        "unknown"),
}

# ── Free-text fields ─────────────────────────────────────────────────────────
# key -> (中文标签, 最大长度). 型号是自由文本,数值 spec 和枚举 spec 都装不下。
#
# ⚠️ 这一类键**绝不能**混进 _CALIB_KEYS 图省事:sanitize 对 calib 键走
# _coerce_num(float, ...),字符串进去直接变 None 被丢;前端 buildBase() 对
# calib 键也只回显 typeof === "number",字符串会在下一次任意设置保存时被静默
# 抹掉。自由文本需要独立的类型处理和前端回显。
_TEXT_SPEC: dict[str, tuple[str, int]] = {
    "preamp_model": ("前置放大器型号", 120),
}

# Learned calibration — WRITTEN BY THE RUNTIME (set_calibration), not the UI.
# The UI shows them read-only (and may clear them). Bound to the conditions
# they were measured under (bias/setpoint/mod_amp): dI/dV magnitude is not
# comparable across a different bias, so a mismatch replaces rather than EWMAs.
_CALIB_KEYS: tuple[str, ...] = (
    "didv_at_contact_v",     # learned dI/dV R magnitude at tunnelling contact
    "didv_cal_bias_v",       # bias the calibration was taken at
    "didv_cal_setpoint_a",   # setpoint the calibration was taken at
    "didv_cal_mod_amp_v",    # lock-in modulation amplitude at calibration
    "didv_cal_updated_at",   # unix ts of last update
    # ── 倾斜响应矩阵 G (2026-07-30, TiltCalibrate 写入) ────────────────
    # Piezo_TiltSet 的 tilt_x/tilt_y 与图像上测到的斜率之间,轴对应、符号、增益
    # 都取决于仪器接线与 Nanonis 内部约定,**不可先验假定**。TiltCalibrate 用
    # ±0.2° 的小步试探解出一个 2×2 矩阵,把「符号 + 轴交换 + 增益」一次吃掉。
    #
    # 一般情况 G ≈ ±单位阵(轴不交换),但按 2×2 存,免得在「一般情况」不成立的
    # 那台机器上把针尖往错误方向推。
    #
    # **没有这个标定,AutoTilt 一律跳过** —— 绝不带着猜来的符号去动硬件。
    "tilt_cal_g11", "tilt_cal_g12", "tilt_cal_g21", "tilt_cal_g22",
    "tilt_cal_cond",         # 响应矩阵条件数(健全性:> 10 视为标定失败)
    "tilt_cal_updated_at",   # unix ts
    # ── qPlus 共振测量 (AcquirePLLFreqSweep 写入) ──────────
    # 与针尖行上的**标称** qplus_f0_hz / qplus_q 刻意分成两处:标称是这支音叉的
    # 铭牌/型录值,跟着针尖走;实测是当前装机这一支扫出来的真值,换针即失效。
    # 复用同一个键会立刻产生「这个数是谁的、什么时候的」的二义 —— 那正是
    # qplus_amplitude_baseline「注册在 config 却语义是 calib」留下的警告。
    "qplus_f0_measured_hz",  # PLL 频率扫描测得的共振频率
    "qplus_q_measured",      # 同次扫描测得的 Q
    "qplus_fq_updated_at",   # unix ts
)

#: 倾斜响应矩阵的四个元素(按行)。
TILT_CAL_KEYS: tuple[str, ...] = (
    "tilt_cal_g11", "tilt_cal_g12", "tilt_cal_g21", "tilt_cal_g22")

#: 响应矩阵条件数上限 —— 超过说明两个轴的响应几乎共线,解出来的 G 不可靠。
TILT_CAL_MAX_COND = 10.0

# Nanonis Motor_StartMove direction codes: 4 = Z+, 5 = Z-.
_MOTOR_DIR_CODE: dict[str, int] = {"z+": 4, "z-": 5}

# EWMA weight for a compatible re-calibration (same bias/setpoint window).
_CALIB_EWMA_ALPHA = 0.3
# A new bias/setpoint further than this (relative) from the stored calibration
# is treated as a DIFFERENT condition → replace, don't average.
_CALIB_BIAS_TOL_V = 0.01
_CALIB_SETPOINT_REL_TOL = 0.25

# Public list of the operator-editable keys (for the frontend / tests).
EDITABLE_KEYS: tuple[str, ...] = (
    tuple(_CONFIG_SPEC) + tuple(_CHOICE_SPEC) + tuple(_TEXT_SPEC))
# Full set of keys that may appear in a stored profile (config + text + calib).
ALL_KEYS: tuple[str, ...] = EDITABLE_KEYS + _CALIB_KEYS

#: 换针使这些量失效 —— 它们是「这根针」学出来/量出来的,不是仪器属性。
#: 由 :func:`clear_tip_bound_state` 在登记新针尖时清除(清掉的值会先快照进
#: 退役针尖那一行的 ``retire_snapshot``,是归档不是丢弃)。
#:
#: 刻意**不含**:
#:   * ``tilt_cal_*`` —— 倾斜响应是样品/托架的属性,与针无关;
#:   * ``qplus_amplitude_signal_index`` —— Nanonis 信号槽接线,是仪器属性。
#:
#: 注意 ``qplus_amplitude_baseline`` 属于 ``_CONFIG_SPEC`` 而非 ``_CALIB_KEYS``
#: (它注册在 config、语义却是运行时标定),所以清除要走 config 键的 pop 路径。
TIP_BOUND_KEYS: tuple[str, ...] = (
    "didv_at_contact_v",
    "didv_cal_bias_v",
    "didv_cal_setpoint_a",
    "didv_cal_mod_amp_v",
    "didv_cal_updated_at",
    "qplus_amplitude_baseline",   # ← _CONFIG_SPEC 键
    "qplus_f0_measured_hz",
    "qplus_q_measured",
    "qplus_fq_updated_at",
)


def spec_default(key: str) -> Any:
    """这个键的**出厂默认**（没有默认 / 未知键 → ``None``）。

    与 :func:`get_config` 的区别很要紧：``get_config`` 先看当前 profile，返回的是
    「现在生效的值」；本函数只看 spec，返回「用户什么都没填时系统在用什么」。
    新仪器初始化页面要同时显示这两个，才能区分「填过」与「用着默认」——
    而那两者在存储层看起来一模一样。
    """
    if key in _CONFIG_SPEC:
        return _CONFIG_SPEC[key][4]
    if key in _CHOICE_SPEC:
        return _CHOICE_SPEC[key][3]
    return None                     # 文本字段没有出厂默认，未填就是未填


def field_specs() -> list[dict[str, Any]]:
    """每个可编辑字段的规格（标签 / 单位 / 类型 / 范围 / 默认 / 枚举选项）。

    形状照抄 :func:`mast.core.scan_policy.tier_field_specs` —— 让前端能从**同一个
    真源**渲染输入框和做范围校验，而不是在 TSX 里再抄一份区间。抄一份就会漂：
    ``test_instrument_profile_frontend_parity`` 存在的理由正是本仓已经因为
    「后端一张表、前端另一张表」丢过一次 qPlus 基线。
    """
    out: list[dict[str, Any]] = []
    for key, (label, unit, kind, (lo, hi), default) in _CONFIG_SPEC.items():
        out.append({
            "key": key, "label": label, "unit": unit, "type": kind.__name__,
            "min": lo, "max": hi, "default": default, "choices": None,
        })
    for key, (label, choices, disp, default) in _CHOICE_SPEC.items():
        out.append({
            "key": key, "label": label, "unit": "", "type": "choice",
            "min": None, "max": None, "default": default,
            "choices": [{"value": c, "label": disp.get(c, c)} for c in choices],
        })
    for key, (label, maxlen) in _TEXT_SPEC.items():
        out.append({
            "key": key, "label": label, "unit": "", "type": "str",
            "min": None, "max": maxlen, "default": None, "choices": None,
        })
    return out


def _coerce_num(kind: type, value: Any) -> "int | float | None":
    """float()/int() a value, dropping NaN/inf/non-numeric → None."""
    try:
        val = kind(value)
    except (TypeError, ValueError):
        return None
    if isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
        return None
    return val


def sanitize(raw: Any) -> dict[str, Any]:
    """Coerce a raw profile dict into a clean, bounded snapshot.

    Keeps only known keys; coerces + clamps config numbers, validates choice
    enums, and passes calibration numbers through with a light numeric check.
    Returns ``{}`` for a non-dict — never raises.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, (_label, _unit, kind, (lo, hi), _default) in _CONFIG_SPEC.items():
        if key not in raw or raw[key] is None or raw[key] == "":
            continue
        val = _coerce_num(kind, raw[key])
        if val is None:
            continue
        val = max(lo, min(hi, val))
        out[key] = int(val) if kind is int else float(val)
    for key, (_label, choices, _disp, _default) in _CHOICE_SPEC.items():
        v = raw.get(key)
        if isinstance(v, str) and v.strip() in choices:
            out[key] = v.strip()
    # Free text: verbatim + length cap, empty dropped (the shape experiment_prefs
    # uses for its notes field). Non-str is dropped rather than str()-ed — a dict
    # or list here means the caller sent the wrong thing, and "{'a': 1}" as a
    # preamp model is worse than no value at all.
    for key, (_label, maxlen) in _TEXT_SPEC.items():
        v = raw.get(key)
        if not isinstance(v, str):
            continue
        text = v.strip()
        if text:
            out[key] = text[:maxlen]
    # Calibration keys: pass through as floats (updated_at too); drop junk.
    for key in _CALIB_KEYS:
        if key not in raw or raw[key] is None or raw[key] == "":
            continue
        val = _coerce_num(float, raw[key])
        if val is not None:
            out[key] = val
    return out


# ── Process-level holder (live-read) ─────────────────────────────────────────
_lock = threading.RLock()
_profile: dict[str, Any] = {}
_persist_sink: "Callable[[dict[str, Any]], None] | None" = None


def set_persist_sink(fn: "Callable[[dict[str, Any]], None] | None") -> None:
    """Inject the persistence callback (runtime wiring). Called by
    ``set_calibration`` after a learned update so the value survives to the
    next run. A no-op sink (the shipped default) keeps the module standalone."""
    global _persist_sink
    with _lock:
        _persist_sink = fn


def set_profile(raw: Any) -> dict[str, Any]:
    """Replace the active profile snapshot (sanitised). Returns the stored copy.

    Used by startup hydration + POST /api/settings live-apply. Does NOT fire
    the persist sink (the settings write path already persisted)."""
    clean = sanitize(raw)
    with _lock:
        _profile.clear()
        _profile.update(clean)
        return dict(_profile)


def get_profile() -> dict[str, Any]:
    """Return a copy of the active profile snapshot ({} when unset)."""
    with _lock:
        return dict(_profile)


def get_config(key: str, default: Any = None) -> Any:
    """Read one config value, falling back to the spec default then ``default``."""
    with _lock:
        if key in _profile:
            return _profile[key]
    if key in _CONFIG_SPEC:
        spec_default = _CONFIG_SPEC[key][4]
        return spec_default if spec_default is not None else default
    if key in _CHOICE_SPEC:
        return _CHOICE_SPEC[key][3]
    if key in _TEXT_SPEC:
        return default          # 文本字段没有出厂默认,未填就是未填
    return default


def get_retract_dir_code() -> int:
    """Nanonis Motor_StartMove direction code for 'away from sample' on the instrument."""
    return _MOTOR_DIR_CODE.get(str(get_config("retract_motor_dir", "z+")), 4)


def get_z_extend_sign() -> int:
    """+1 if piezo extension (toward sample) shows as INCREASING Z, else -1.

    **未声明时返回出厂 ``+1``** —— 调用方若承受不起「猜错方向」，用
    :func:`z_extend_sign_or_none`。

    ⚠️ **退针/清障的方向自检已经不再用这个读法**（2026-08-11 审计）：它们要的是
    「没声明就说不知道」，见 :func:`z_extend_sign_or_none` 的注释。留着这一个是给
    **显示与诊断**用的 —— 那些地方猜错只是显示不好看，不会驱动粗动马达。
    """
    try:
        return 1 if str(get_config("z_extend_sign", "+1")) == "+1" else -1
    except Exception:  # noqa: BLE001
        return 1


def z_extend_sign_or_none() -> "int | None":
    """读取显式登记的 Z 伸长方向：+1、-1；未声明或无效时返回 None。
    
    方向取决于目标仪器的接线，不能由出厂显示值推断。退针自检与有向余量
    均依赖此符号，因此同一符号的下游解释不能反过来证明它正确。
    核验应使用未经方向翻译的原始 Z 读数与已知动作。
    
    只有实际保存在 _profile 中的声明才算已配置；UI 的默认显示不构成声明。
    """
    with _lock:
        raw = _profile.get("z_extend_sign")
    if raw is None:
        return None
    text = str(raw).strip()
    if text not in ("+1", "-1"):
        return None
    return 1 if text == "+1" else -1


def has_xy_coarse_motion() -> bool:
    """True if this rig can coarse-move laterally to a fresh patch of surface."""
    return str(get_config("xy_coarse_motion", "yes")).strip().lower() != "no"


def approach_damages_surface() -> bool:
    """True if an approach should be treated as leaving a dimple.

    "unknown" resolves to True on purpose. Being wrong in this direction costs a
    couple hundred nanometres of surface; being wrong the other way costs a
    wasted image on a dimple and possibly the tip."""
    return str(get_config("approach_damages_surface", "unknown")).strip().lower() != "no"


def get_scan_path_strategy() -> str:
    """Resolved next-scan-position strategy: ``center_first`` | ``perimeter_inward``.

    Resolves the ``auto`` setting from the rig's capability, which is the whole
    reason ``auto`` is the default: the right route follows from whether you can
    relocate at all. A rig with lateral coarse motion should image at the centre
    of its piezo range, where the scan tube creeps least, and simply move when
    the area is spent. A rig without it can never get fresh surface, so it works
    outside-in to keep the biggest clean region intact for as long as possible."""
    strat = str(get_config("scan_path_strategy", "auto")).strip().lower()
    if strat in ("center_first", "perimeter_inward"):
        return strat
    return "center_first" if has_xy_coarse_motion() else "perimeter_inward"


def get_calibration() -> dict[str, Any]:
    """Return the learned dI/dV-at-contact calibration ({} keys absent when
    never calibrated)."""
    with _lock:
        return {k: _profile[k] for k in _CALIB_KEYS if k in _profile}


def _conditions_compatible(bias_v: float, setpoint_a: float,
                           prev: dict[str, Any]) -> bool:
    """True if a new (bias, setpoint) is close enough to the stored calibration
    that EWMA-averaging is meaningful (same measurement condition)."""
    pb = prev.get("didv_cal_bias_v")
    ps = prev.get("didv_cal_setpoint_a")
    if pb is None or ps is None:
        return False
    if abs(float(bias_v) - float(pb)) > _CALIB_BIAS_TOL_V:
        return False
    denom = max(abs(float(ps)), 1e-15)
    if abs(abs(float(setpoint_a)) - abs(float(ps))) / denom > _CALIB_SETPOINT_REL_TOL:
        return False
    return True


def set_calibration(
    didv_v: float,
    *,
    bias_v: float,
    setpoint_a: float,
    mod_amp_v: "float | None" = None,
) -> dict[str, Any]:
    """Record a verified dI/dV-at-contact from a successful approach (RUNTIME).

    EWMA-updates the stored value when the new reading was taken under a
    compatible bias/setpoint; otherwise replaces it (a different condition is
    not comparable). Updates the binding metadata, then fires the persist sink
    so the learned value survives to the next run. Best-effort + never raises —
    a calibration bookkeeping failure must never break an approach.

    Returns the new calibration dict.
    """
    try:
        new_val = _coerce_num(float, didv_v)
        if new_val is None or new_val <= 0:
            return get_calibration()
        new_val = abs(new_val)
        with _lock:
            prev = {k: _profile[k] for k in _CALIB_KEYS if k in _profile}
            prev_val = prev.get("didv_at_contact_v")
            if (prev_val is not None
                    and _conditions_compatible(bias_v, setpoint_a, prev)):
                merged = (_CALIB_EWMA_ALPHA * new_val
                          + (1.0 - _CALIB_EWMA_ALPHA) * float(prev_val))
            else:
                merged = new_val
            _profile["didv_at_contact_v"] = merged
            _profile["didv_cal_bias_v"] = float(bias_v)
            _profile["didv_cal_setpoint_a"] = float(setpoint_a)
            if mod_amp_v is not None:
                mv = _coerce_num(float, mod_amp_v)
                if mv is not None:
                    _profile["didv_cal_mod_amp_v"] = mv
            _profile["didv_cal_updated_at"] = time.time()
            snapshot = dict(_profile)
            sink = _persist_sink
        # Persist OUTSIDE the lock (sink may touch a store / disk).
        if sink is not None:
            try:
                sink(snapshot)
            except Exception as exc:  # noqa: BLE001
                logger.debug("instrument_profile persist sink failed: %s", exc)
        return {k: snapshot[k] for k in _CALIB_KEYS if k in snapshot}
    except Exception as exc:  # noqa: BLE001 — never break an approach
        logger.debug("set_calibration failed: %s", exc)
        return get_calibration()


def get_tilt_calibration() -> "dict[str, Any] | None":
    """倾斜响应矩阵 G(2×2,按行)+ 条件数 + 时间戳;没标定过返回 None。

    没有它 ``AutoTilt`` 一律跳过 —— ``Piezo_TiltSet`` 的轴对应与符号取决于仪器
    接线,猜错方向就是把倾斜往反方向加倍。
    """
    with _lock:
        if not all(k in _profile for k in TILT_CAL_KEYS):
            return None
        return {
            "g": [[float(_profile["tilt_cal_g11"]), float(_profile["tilt_cal_g12"])],
                  [float(_profile["tilt_cal_g21"]), float(_profile["tilt_cal_g22"])]],
            "cond": _profile.get("tilt_cal_cond"),
            "updated_at": _profile.get("tilt_cal_updated_at"),
        }


def set_tilt_calibration(g: Any, *, cond: float) -> "dict | None":
    """写入倾斜响应矩阵(TiltCalibrate 的产物)。返回存进去的标定或 None。

    条件数超过 :data:`TILT_CAL_MAX_COND` 时**拒绝写入**:两个轴的响应几乎共线
    意味着解出来的 G 不可靠,存进去比不存更危险 —— 之后每次调平都会用它。

    ``cond`` **是必需的,没有默认值**(v6.1.3)。它以前是 ``cond: float | None = None``,
    配一道 ``if cond is not None and cond > MAX`` 的闸门 —— 于是**不传 cond 就等于
    跳过这道闸门**,一次未经验证的标定照样写进去。「算不出条件数」不是「条件数良好」
    的证据,而那个可选性替调用方做了这个决定。

    清点过:生产只有 ``auto_tilt.py`` 一个调用方且总是传 cond,测试也全部显式传 ——
    **``cond=None`` 从来不是正当模式**,那份可选性是一个没人用的灵活性,
    而它恰好凿了个洞。改成必需参数,让「忘了传」变成 **TypeError**(程序员错误,要响)
    而不是一次静默的坏写入。

    两类失败刻意分开:
      * **缺参数** → ``TypeError``(调用方写错了代码);
      * **条件数超限 / 非有限** → 返回 ``None``(数据不合格,这是本函数的既有语义)。
    合并它们会在高一层重建「两个状态共用一个信号」。
    """
    try:
        rows = [[float(g[0][0]), float(g[0][1])], [float(g[1][0]), float(g[1][1])]]
    except (TypeError, ValueError, IndexError):
        logger.warning("tilt 标定矩阵形状非法,拒绝写入: %r", g)
        return None
    if not all(math.isfinite(v) for row in rows for v in row):
        logger.warning("tilt 标定矩阵含非有限值,拒绝写入: %r", rows)
        return None
    # 显式传 None 仍然可能发生(必需参数拦不住 `cond=None`),所以闸门自己也要
    # 挡住「条件数未知」。以前这里写的是 `cond is not None and ...` ——
    # **未知即放行**,也就是「没有证据 = 没有问题」。
    if cond is None or not math.isfinite(float(cond)):
        logger.warning("tilt 标定条件数未知或非有限(%r),拒绝写入 —— "
                       "算不出条件数不是条件数良好的证据", cond)
        return None
    if float(cond) > TILT_CAL_MAX_COND:
        logger.warning(
            "tilt 标定条件数 %.1f 超过上限 %.1f(两轴响应几乎共线),拒绝写入",
            float(cond), TILT_CAL_MAX_COND)
        return None

    with _lock:
        _profile["tilt_cal_g11"] = rows[0][0]
        _profile["tilt_cal_g12"] = rows[0][1]
        _profile["tilt_cal_g21"] = rows[1][0]
        _profile["tilt_cal_g22"] = rows[1][1]
        # 无条件写。以前是 `if cond is not None:` —— 传 None 时这个字段**不更新**,
        # 于是 profile 里留着**上一次标定**的条件数,配着**这一次**的矩阵。
        # 那比没有更坏:一个看起来有依据的数,描述的是另一个已经不在那里的矩阵。
        _profile["tilt_cal_cond"] = float(cond)
        _profile["tilt_cal_updated_at"] = time.time()
        snapshot = dict(_profile)
        sink = _persist_sink
    if sink is not None:
        try:
            sink(snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.debug("instrument_profile persist sink failed: %s", exc)
    return get_tilt_calibration()


def set_qplus_resonance(f0_hz: Any, q: Any) -> dict[str, Any]:
    """记录 PLL 频率扫描测到的 qPlus 共振频率与 Q(RUNTIME 写入器)。

    在此之前 ``AcquirePLLFreqSweep`` 把这两个值提进 SkillResult 就丢掉了 ——
    每次想知道当前音叉的 f₀/Q 都得重扫一次,而它们是判断"这支传感器还好不好"
    的基本量。补上写回,与 ``TiltCalibrate → set_tilt_calibration`` 同一形状。

    这是**实测**值,与针尖行上的**标称** ``qplus_f0_hz`` / ``qplus_q`` 分开存;
    换针尖时由 :func:`clear_tip_bound_state` 清除(旧针的实测共振不属于新针)。

    非有限/非正的值一律拒绝写入(扫描失败时 Nanonis 会回 0 或 NaN,存进去比不存
    更糟 —— 之后每次都会拿它当真值)。Best-effort,永不抛。
    """
    try:
        f0 = _coerce_num(float, f0_hz)
        qq = _coerce_num(float, q)
        if f0 is None or f0 <= 0 or qq is None or qq <= 0:
            logger.debug("qPlus 共振读数非法,拒绝写入: f0=%r q=%r", f0_hz, q)
            return get_calibration()
        with _lock:
            _profile["qplus_f0_measured_hz"] = float(f0)
            _profile["qplus_q_measured"] = float(qq)
            _profile["qplus_fq_updated_at"] = time.time()
            snapshot = dict(_profile)
            sink = _persist_sink
        if sink is not None:
            try:
                sink(snapshot)
            except Exception as exc:  # noqa: BLE001
                logger.debug("instrument_profile persist sink failed: %s", exc)
        return {k: snapshot[k] for k in _CALIB_KEYS if k in snapshot}
    except Exception as exc:  # noqa: BLE001 — 记账失败绝不能弄坏一次扫描
        logger.debug("set_qplus_resonance failed: %s", exc)
        return get_calibration()


def get_tip_bound_state() -> dict[str, Any]:
    """当前 profile 里**绑当前这根针**的量(见 :data:`TIP_BOUND_KEYS`)。

    登记新针尖前调它取快照,存进退役针尖那一行 —— 这些值必须清,但不必丢。
    """
    with _lock:
        return {k: _profile[k] for k in TIP_BOUND_KEYS if k in _profile}


def clear_tip_bound_state() -> dict[str, Any]:
    """换针尖:清掉绑上一根针的学习量。返回被清掉的键值(供归档)。

    **不是** :func:`clear_calibration` —— 那个按 ``_CALIB_KEYS`` 全清,会连
    ``tilt_cal_*`` 一起带走,而倾斜响应是样品/托架的属性,与换针无关;它也漏掉
    ``qplus_amplitude_baseline``(那个键在 ``_CONFIG_SPEC`` 里)。

    为什么必须清:
      * ``didv_at_contact_v`` 是每次成功进针 EWMA 学出来的,依赖针尖态;
      * ``qplus_amplitude_baseline`` 是撞针判据的**分母**(当前振幅/基线 < 10%
        判撞针),而它是"上一根针自由振荡时的振幅" —— 换针后拿旧分母比,
        判据要么永远报撞针,要么永远报不出来。
    """
    with _lock:
        dropped = {k: _profile.pop(k) for k in TIP_BOUND_KEYS if k in _profile}
        snapshot = dict(_profile)
        sink = _persist_sink
    if sink is not None:
        try:
            sink(snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.debug("instrument_profile persist sink failed: %s", exc)
    if dropped:
        logger.info("换针尖:清除绑上一根针的标定 %s", sorted(dropped))
    return dropped


def clear_calibration() -> dict[str, Any]:
    """Drop the learned calibration (UI 'reset'), keeping config. Persists.

    NOTE (2026-07-31): ``_CALIB_KEYS`` 新增了 qPlus 实测共振三键,所以本函数现在
    也会一并清掉它们。本函数全仓无调用方(只有定义与 ``__all__``),行为变化留痕
    在此。换针尖请用 :func:`clear_tip_bound_state` —— 本函数会把 ``tilt_cal_*``
    也清掉,那不是换针该做的事。
    """
    with _lock:
        for k in _CALIB_KEYS:
            _profile.pop(k, None)
        snapshot = dict(_profile)
        sink = _persist_sink
    if sink is not None:
        try:
            sink(snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.debug("instrument_profile persist sink failed: %s", exc)
    return {}


# ── Pure render (unit-testable) ──────────────────────────────────────────────
def _fmt_didv(v: "float | None") -> str:
    """Render the stored dI/dV calibration for the prompt, ALWAYS carrying the SI
    value.

    The stored field is ``didv_at_contact_v`` — volts. This used to return bare
    "3.500 µV" / "1.200 mV" on the readable branches, so the volt value the model
    needs never appeared in the text at all. The same block then tells the agent
    to compare this figure against a live lock-in R reading, which IS in volts —
    a µV string against a V reading is off by 1e6.

    Every human-readable form is now followed by its SI equivalent, the shape
    safety_mw's correction text already uses.
    """
    if v is None:
        return "尚未标定"
    av = abs(float(v))
    if av >= 1e-3:
        return f"{v * 1e3:.3f} mV (= {v:.3e} V)"
    if av >= 1e-6:
        return f"{v * 1e6:.3f} µV (= {v:.3e} V)"
    return f"{v:.3e} V"


def format_profile_block(profile: dict[str, Any] | None) -> str:
    """Render the instrument mechanism + config + learned calibration as a
    system block. Pure function.

    Unlike experiment_prefs, this ALWAYS returns a non-empty block (falling back
    to spec defaults) because the *mechanism* knowledge is a safety-relevant
    instrument fact the agent should always have when reasoning about 进/退针.
    """
    p = dict(profile or {})

    def cfg(key: str) -> Any:
        if key in p:
            return p[key]
        if key in _CONFIG_SPEC:
            return _CONFIG_SPEC[key][4]
        if key in _CHOICE_SPEC:
            return _CHOICE_SPEC[key][3]
        return None

    retract_dir = cfg("retract_motor_dir")
    retract_dir_disp = _CHOICE_SPEC["retract_motor_dir"][2].get(
        str(retract_dir), str(retract_dir))
    total = cfg("retract_total_steps")
    amp = cfg("lockin_mod_amp_v")
    freq = cfg("lockin_mod_freq_hz")
    sig_idx = cfg("lockin_signal_index")

    didv = p.get("didv_at_contact_v")
    cal_bias = p.get("didv_cal_bias_v")
    cal_line = (
        f"到样品(建立隧道)的 dI/dV 标定值 ≈ {_fmt_didv(didv)}"
        + (f"（绑定 bias={cal_bias:.3g} V）" if didv is not None and cal_bias is not None
           else "")
        if didv is not None
        else "到样品的 dI/dV 标定值：尚未标定（首次成功进针后自动记录，供下轮判距离用）")

    # 解调形式决定这句话能不能说「幅度」。X 是带符号投影，只有相位调好时 |X| ≈ R；
    # 把 X 说成 R，就是在教模型相信一个会穿零的量单调递增。
    form = str(cfg("lockin_readout_form") or "unknown").strip().lower()
    sym = {"xy": "|X|", "r_phi": "R"}.get(form, "读数绝对值")
    sig_note = (f"信号索引={sig_idx}" if sig_idx is not None
                else "信号索引：未配置（进针 dI/dV 测距需在设置里填）")
    if form == "xy":
        form_note = (
            f"本机 lock-in 以 **X/Y** 定义（登记的是 X 通道）。X 是对参考相位的"
            f"**带符号投影**，只有相位调到「信号全落在 X 上」时 {sym} 才≈幅度；"
            f"相位没调好、或在接近途中转动（结电容随距离变，相位真的会转），"
            f"{sym} 会在相位扫过 90° 时**穿零** —— 那时「越近越大」会中途掉下去，"
            f"**那是相位问题，不是针尖退开了**。看到非单调下降先怀疑相位。")
    elif form == "r_phi":
        form_note = f"本机 lock-in 以 **R/Φ** 定义，{sym} 是幅度，恒为正。"
    else:
        form_note = (
            "本机 lock-in 的解调形式（X/Y 还是 R/Φ）**未登记** —— "
            "在补上之前只把它当作一个绝对值读数，不要断言它是幅度、也不要因为它"
            "非单调就判断针尖退开了（X 通道穿零看起来一模一样）。")

    return (
        "## 本仪器进/退针机理与装置配置（instrument_profile）\n"
        "以下是本机的装置事实，推理进针/退针时据此判断：\n"
        f"- **换样品/关机退针**：用粗动马达沿「远离样品」方向"
        f"（当前配置：{retract_dir_disp}）分级后退（约 {total} 步）。"
        "每级按 1→10→100→剩余 递增，每级退完开反馈回读 **Z 压电走向**自检——"
        "只有 Z 朝伸长方向（反馈在追已变远的样品=确认在远离）才继续；"
        "一旦回读到在**逼近**（电流暴涨/压电缩回）立即停并撤针。"
        "退针方向配置可能填反，**运行时这一步 Z 自检才是防撞针的最终防线**。\n"
        f"- **进针判距离**：lock-in 加在电流上做 **dI/dV**（调制 {amp} V @ {freq} Hz，"
        f"{sig_note}）。{sym} 随针尖接近样品**指数增大**，能在 DC 电流还看不出时"
        f"判断「离样品还有多远」。{form_note}{cal_line}。\n"
        "- 标定值依赖针尖态/调制幅度/bias，每轮成功进针自动 EWMA 更新——"
        "它是学习量不是常数；换 bias 标定值不通用。\n"
        + _format_approach_supervision_line(cfg)
        + _format_scan_map_block(cfg)
        + _format_coarse_motion_block(cfg)
    )


def _format_approach_supervision_line(cfg) -> str:
    """这台机器的 Auto Approach 会不会扎针 —— 决定 agent 敢不敢自己发起进针。

    这是**装置事实**，不是劝说：有些进针机构在建立隧道那一刻会把针扎进去，有些
    不会，而没有任何读数能告诉模型自己在哪一台上。默认 ``unknown`` 按保守处理，
    与 ``approach_damages_surface`` 同一个理由 —— 猜错方向的代价一边是等用户
    到场，另一边是一根针加几小时修针。

    ⚠️ 这是**注入**不是门。HITL 门在 graph build 时从 ``safety_level`` 派生
    （``AutoApproach`` 现为 AUTO），运行时字段改不了它。把它做成硬门需要在技能
    入口加软门并重建图，那是一次改变硬件行为的改动，该由用户拍板。
    """
    mode = str(cfg("approach_supervision") or "unknown").strip().lower()
    if mode == "unattended":
        return ("- **进针值守**：本机 Auto Approach **不扎针**，可以在无人值守的"
                "自主流程里发起进针。\n")
    if mode == "attended":
        return ("- **进针值守**：⚠ 本机进针**会扎针**，"
                "**不要在自主流程里自己发起 Auto Approach** —— "
                "先用 `request_user_action` 请用户到场，由人确认后再进。\n")
    return ("- **进针值守**：本机进针会不会扎针**未登记**，按保守处理"
            "（＝当作会扎）：不要在无人值守时自己发起 Auto Approach，"
            "先请用户到场或先把这一项登记进仪器档案。\n")


def _format_coarse_motion_block(cfg) -> str:
    """Lateral relocation: the facts, the live gates, and which skill to call.

    Injected unconditionally, next to the retract/approach mechanism, because it
    is the same category of thing — how THIS instrument physically works — and
    because the alternative is the agent discovering each rule by being refused.
    A run that learns "retract before you slide" from a rejection message has
    already spent a turn on it and has learned it as an obstacle rather than as a
    property of the machine.

    The two live gates (pressure, drive declaration) are rendered from their
    CURRENT state, not described in the abstract: "禁止粗动，因为读不到真空计" is
    actionable, "粗动需要真空" is trivia."""
    pre = cfg("xy_prewithdraw_steps")
    spacing = cfg("xy_site_spacing_steps")
    budget = cfg("xy_axis_step_budget")

    lines = [
        "\n## 横向粗动换区（RelocateCoarseXY）\n",
        f"- **换区必须用 `RelocateCoarseXY`**，不要直接调 `MotorMove` 做 x±/y± 横向移动"
        f"（自主路径上后者会被安全门拒绝）。它会：收压电 → 用粗动马达退 {pre} 步清障"
        f"（逐级 1→10→… 自检 Z 压电走向，方向配反只赔 1 步）→ 确认电流归零"
        f"（有 qPlus 时还要振幅恢复）→ 分块横移、每块看电流/振幅/真空 → 可选重新进针。\n",
        "- **为什么必须粗动退针**：压电收到顶只有 1–2 µm 余量，而样品台侧滑时的垂直跳动、"
        "样品倾斜和针尖长度都远不止这个数。这是压电退针替代不了的一步。\n",
        f"- **走多远**：一次位移必须显著超过压电量程（±1.5 µm），否则「新区域」和旧区域重叠 ——"
        f"看着换了地方，其实只是把旧表面挪进视野。本机站点最小间距 {spacing} 步、"
        f"单轴行程预算 {budget} 步。\n",
        "- **往哪走**：调 `get_coarse_map`。那是**样品台尺度**的大地图（单位是**步**，"
        "跨所有坐标代次），记录去过哪些片；`get_map_analysis` 是压电尺度的另一张图。"
        "两者不要混用。粗动是开环的，站点位置带不确定半径、画成模糊斑 —— "
        "**不要把步数换算成米去做几何**，步长随驱动幅度/负载/温度漂移。\n",
        "- **不要回到已经用过的那一片**：修针碎屑、脉冲坑、撞针点会永久消耗表面。"
        "规划器已经按「模糊斑不重叠」帮你排除了，照它给的方向/步数走即可。\n",
    ]

    # Live gate 1 — pressure.
    try:
        from mast.core.vacuum_interlock import format_block as _vac_block
        lines.append("- " + _vac_block() + "\n")
    except Exception:  # noqa: BLE001 — the block must never fail to render
        lines.append("- 【真空互锁】状态不可用 —— 按禁止粗动处理。\n")

    # Live gate 2 — drive declaration.
    try:
        from mast.core.coarse_drive import format_block as _drv_block
        lines.append("- " + _drv_block() + "\n")
    except Exception:  # noqa: BLE001
        lines.append("- 【粗动驱动电压】状态不可用 —— 粗动会被拒绝。\n")

    return "".join(lines)


def _format_scan_map_block(cfg) -> str:
    """The surface-budget facts: can we relocate, what marks the surface, and
    which route the map analyser will therefore recommend.

    Included in the always-on instrument block because it changes what a correct
    plan looks like. On a rig that cannot relocate, "just move somewhere else"
    is not an available recovery, and every tip-forming plunge permanently costs
    part of the only surface there is."""
    can_move = str(cfg("xy_coarse_motion")).strip().lower() != "no"
    dmg = str(cfg("approach_damages_surface")).strip().lower()
    strat = str(cfg("scan_path_strategy")).strip().lower()
    if strat not in ("center_first", "perimeter_inward"):
        strat = "center_first" if can_move else "perimeter_inward"
    strat_disp = ("中心优先（压电蠕变最小）" if strat == "center_first"
                  else "外圈→内圈（可用面积利用最大化）")
    move_line = (
        "- **换区**：本机**有 XY 粗动马达** —— 一片表面用坏了可以粗动到新区域。"
        "注意粗动会让压电坐标系整体失效：地图会自动进入新的坐标代次，"
        "旧标记不再参与分析。\n"
        if can_move else
        "- **换区**：本机**没有 XY 粗动** —— 当前压电范围内的表面就是"
        "「在插拔样品之前你能看到的全部表面」。修针尖的污染会不可逆地吃掉可用面积，"
        "所以选点必须省着来。\n")
    dmg_line = (
        "- **进针扎痕**：本机进针**不扎**表面，进针点只作历史记录、不设避让区。\n"
        if dmg == "no" else
        f"- **进针扎痕**：进针会在表面留扎痕（当前配置：{dmg}"
        + ("，未知按保守处理" if dmg == "unknown" else "")
        + "），进针点会生成避让区。\n")
    return (
        "\n## 表面预算与选点策略（扫描地图）\n"
        + move_line + dmg_line
        + f"- **选点策略**：{strat_disp}。\n"
        "- 「扫了哪里 / 哪里被破坏 / 该不该换区 / 下一个位置」一律调 "
        "`get_map_analysis`、`get_next_scan_position` 查程序算出的结论，"
        "**不要靠看图或凭印象推断**。"
    )


__all__ = [
    "EDITABLE_KEYS",
    "ALL_KEYS",
    "TIP_BOUND_KEYS",
    "spec_default",
    "field_specs",
    "get_tip_bound_state",
    "clear_tip_bound_state",
    "set_qplus_resonance",
    "sanitize",
    "set_profile",
    "get_profile",
    "get_config",
    "get_retract_dir_code",
    "get_z_extend_sign",
    "z_extend_sign_or_none",
    "has_xy_coarse_motion",
    "approach_damages_surface",
    "get_scan_path_strategy",
    "get_calibration",
    "set_calibration",
    "clear_calibration",
    "set_persist_sink",
    "format_profile_block",
]
