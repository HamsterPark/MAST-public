"""PreScanCheck — quick line scan to verify tip condition (Phase 7 graph-shaped composite).

Migration note (Phase 7):
  The original v1 layout (ConfigureScan -> SetScanSpeed -> StartScan ->
  wait_scan_complete -> raw FrameDataGrab) maps cleanly onto the graph
  framework: four sub-skill steps for the main flow, then the post-scan
  quality evaluation stays as raw ``context.safe_call`` inside the
  run_composite override (mirrors how ``full_scan`` keeps its crash-detect
  step inline). The wait step uses the ``WaitScanComplete`` builtin
  (added Phase 4) instead of the legacy ``wait_scan_complete`` helper.
"""

# K (Keep) — migrated to CompositeSkillGraph 2026-05-19

from __future__ import annotations

import logging

import numpy as np

from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.operating_mode import safe_mode_active as _safe_mode_active
from mast.core.state import VERIFIED_STATE_KEY
from mast.core.scan_policy import (
    estimate_scan_seconds,
    get_tier_for_size,
    wait_budget_s,
)

logger = logging.getLogger(__name__)

# 只有无法读取有效扫描宽度时才使用 DEFAULT_LINE_TIME_S。
# 常规预扫描依据 scan_policy 档位表选择速度，避免用单线预算代替整帧工作点。
# Scan_BufferSet 保持现有行数；帧时还取决于行数，不能从帧高推断。
DEFAULT_LINE_TIME_S = 0.1


def resolve_line_time_s(width_m: "object", explicit: "object" = None) -> float:
    """解析预扫描每线时间的唯一入口。
    
    优先级为调用方显式值、按宽度查询档位表、DEFAULT_LINE_TIME_S。
    ParameterSpec 不得为 line_time_s 注入默认值，否则 schema 会把未传入的键变成显式值，
    使按宽度查询的分支不可达。相关测试需要覆盖 schema、coerce、plan 的完整入口。
    """
    val = _num(explicit, None)
    if val is not None and val > 0:
        return float(val)
    w = _num(width_m, None)
    if w is None or w <= 0:
        return DEFAULT_LINE_TIME_S
    try:
        lt = _num(get_tier_for_size(float(w)).get("line_time_s"), None)
    except Exception:  # noqa: BLE001 — 查表失败不该让预扫描本身失败
        logger.debug("PreScanCheck: 档位表查不到(退回出厂每线时间)", exc_info=True)
        return DEFAULT_LINE_TIME_S
    return float(lt) if lt and lt > 0 else DEFAULT_LINE_TIME_S

#: 读不回线数时的**盲估**行数。
#:
#: 512 不是「典型值」,是**最坏的常见值** —— 这是有意的:本技能**从不设线数**
#: (它的 ConfigureScan 走 ``Scan_BufferSet(ch, 0, 0)``,后两位 = pixels/lines,
#: 恒 0 = 保持现值,见 ``scan_buffer.py`` 的模块注释),所以它扫几行由**上一次
#: 扫描**决定。一个不知道自己要扫几行的技能,盲估时只能按它可能遇到的最大值来:
#: 猜小了就是超时,猜大了只是晚一点发现卡住。两种错的代价不对称。
_BLIND_LINES = 512

# 等待常数仅为下限，最终预算根据扫描几何推导。
_V1_FLOOR_S = 15.0

# 默认等待预算使用保守几何估算；有上下文时读取真实行数，提高估算精度。
DEFAULT_WAIT_TIMEOUT_S = wait_budget_s(
    estimate_scan_seconds(_BLIND_LINES, DEFAULT_LINE_TIME_S),
    floor_s=_V1_FLOOR_S)

#: 「这条线判不了」的统一返回。``correlation is None`` 是三态里的那个第三态。
_INCONCLUSIVE: dict = {"correlation": None, "legacy_cosine": None,
                       "unusable_reason": None}

#: 缓冲回落**为什么不出数**(2026-08-14,修复项)。
#:
#: 这句话是**单一真源** —— 实现和钉住它的测试读同一个常量,免得哪天代码改了
#: 而测试里那句复制的中文还在,看着像在守护、其实守护的是一句旧话。
#: 完整论证在 ``_evaluate_line_quality`` 里那段 (a)(b)(c) 注释,那段一个字没删。
_BUFFER_NOT_COMPARABLE = (
    "判不了：缓冲路径相似度与 .sxm 路径不可比,且缓冲内容未必是本帧"
    "，无法确认可比性，因此不作判定。"
    "**这是弃权,不是「针尖不合格」**:下一步是去 .sxm 里取这一帧 / 换个地方"
    "重新量,**不是修针**。")


def _num(value, fallback: float) -> float:
    """A positive float, or the factory default.

    ``None`` (not passed), junk, NaN and non-positive all mean "use the
    default" — a 0 s line time would divide by zero one line below, and a 0 s
    budget would fail the wait before the scan could start.
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return fallback
    if val != val or val <= 0.0 or val == float("inf"):
        return fallback
    return val


class PreScanCheck(CompositeSkillGraph):
    """Scans at the target width and judges the tip from forward/backward overlap.

    抓正扫/反扫,算余弦相似度,判针尖能不能用。

    ⚠️ **它扫多少行不由它决定** —— ``ConfigureScan`` 走
    ``Scan_BufferSet(ch, 0, 0)``(后两位 = pixels/lines,恒 0 = 保持现值),
    所以行数继承自**上一次扫描**。调用方设过 256 行,这里就是 256 行整帧。

    ── 这段自述以前是假的(2026-08-12 改)──────────────────────────────
    原文逐字写着「Performs a **single-line** scan」/「Quick single-line
    pre-scan」。**它从来没扫过一条线。** 那句话描述的是一个「同时还会设线数」
    的版本,而那一半从来不存在 —— 与本仓另外几条「描述了没发生的事」的注释
    同一个形状。

    代价不是判错,是**刮针**:0.1 s/线是照着「一条线 0.2 s」定的速度,拿去扫
    256 行就是 488 nm/s 的光栅接触,足以刮伤针尖。现在每线时间查档位表
    (见 ``resolve_line_time_s``),50 nm ⇒ 1.0 s/线 ⇒ 50 nm/s。

    Reference: DeepSPM pre-scan check pattern.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PreScanCheck",
            version="1.3.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            # 预扫描可能继承整帧行数，不能描述为单线动作。
            # 判决依赖保存的 .sxm；缓冲回落无法确认可比性时返回 tip_ready=None。
            description=(
                "预扫：按正扫/反扫的重合度判针尖好坏。它扫的是**一整帧**，用当前的"
                "线数（它自己从不设线数）—— 默认档速度下一帧 50 nm 就是 256 行、"
                "~8.5 min。**不便宜，也不是只扫一条线。**"
                "返回 tip_ready（True/False/None）+ similarity。判定**只**来自"
                "存盘的 .sxm 帧：帧存不下来或找不到时会回落到实时缓冲区，而那份"
                "**不可比**、甚至可能装的根本不是这一帧，所以它返回 "
                "tip_ready=None（判不了）并给出理由。**判不了的意思是重测 / 换点位 —— 它不是「针尖坏了」，不要去打脉冲。**"
            ),
            parameters=[
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description="扫描中心 X",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description="扫描中心 Y",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="width_m",
                    type="float",
                    description=(
                        "扫描宽度，单位是 METERS（SI），**不是纳米** —— 应当与"
                        "你打算扫的正式图一致。换算：100 nm → 100n，50 nm → "
                        "50n。写成裸的 100（=100 m）就是单位写错了。"
                    ),
                    unit="m",
                    required=True,
                    min_value=1e-10,
                    max_value=1e-5,   # = config scan_size_max_m (10 µm); guards nm→m slips
                ),
                ParameterSpec(
                    name="quality_threshold",
                    type="float",
                    description="判 tip_ready 所需的最低正/反扫相似度",
                    required=False,
                    default=0.8,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ParameterSpec(
                    name="pixels",
                    type="int",
                    description=(
                        "**这一次**预扫的行数（也是每行像素数）。**留空**则继承"
                        "上一次扫描留下的分辨率 —— 那是历史行为。想让预扫"
                        "**便宜**时才传它：耗时 = 行数 × line_time × 2，所以这是"
                        "**唯一**能缩短它的旋钮（把帧改小**不能** —— 那只是让"
                        "针尖温和一些）。"
                    ),
                    unit="px",
                    required=False,
                    # ⚠️ **绝不能给 default** —— 与 line_time_s 同一个理由:
                    # 有了 default,「没传」就再也不是「继承现场」了。
                    min_value=16,
                    max_value=4096,
                ),
                ParameterSpec(
                    name="line_time_s",
                    type="float",
                    description=(
                        "每条扫描线多少秒。**除非现场明确给出过一个值，否则留空** —— "
                        "留空时由这个扫描尺寸对应的扫描策略档决定"
                        "（50 nm → highres 档）。这个数就是针尖的横向速度："
                        "width / line_time。**太快会刮针尖。**"
                    ),
                    unit="s",
                    required=False,
                    # ⚠️ **绝不能给 default。** 见下面那段。
                    min_value=1e-4,
                    max_value=600.0,
                ),
                ParameterSpec(
                    name="wait_timeout_s",
                    type="float",
                    description=(
                        "等预扫结束的上限。**留空**：这时的预算按扫描缓冲区里"
                        "**实际的**行数 × **实际用的**线时间推出来（这个技能**不**设"
                        "行数 —— 它的耗时由上一次扫描留下的分辨率决定）。"
                        "**你传进来的值是 FLOOR（下限），不是上限。** "
                        "只有用户点了名才传它。"
                    ),
                    unit="s",
                    required=False,
                    # ⚠️ **绝不能给 default** —— 与 line_time_s 同一个理由。
                    #
                    # 这条的描述逐字写着「Omit it: the budget is then derived」,
                    # 而 `default=DEFAULT_WAIT_TIMEOUT_S` 让「omit」**不可能发生**:
                    # pydantic 每次都替调用方填上 163.12,于是旧实现里那句
                    # `if params.get("wait_timeout_s") is None:` **永远为假**,
                    # 派生那条路一次都没走过。
                    #
                    # 也就是说 2026-08-10 那次「预算必须从几何派生」的修复,
                    # 从落地起就被这个 default 架空了 —— 而它自己的描述在告诉每
                    # 一个读它的人「omit 就会派生」。**一句描述了一件没发生的事
                    # 的话,会守住一个不该守的设计,而且没人会去核它。**
                    # (与本仓其它几条「描述了没发生的事」的注释同形,
                    #  包括本文件类自述里那条「single-line scan」。)
                    min_value=1.0,
                    max_value=3600.0,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=10.0,
            composition_level=3,
            tags=["composite", "quality", "prescan", "tip"],
        )

    # --- Plan: static (params alone determine the 4-step flow) ---

    def _derived_timeout_s(self, context, line_time: float,
                           lines_override: "int | None" = None) -> float:
        """按实际行数及每线时间推导等待预算；行数不可读时使用公开默认的保守估算。"""
        # 调用方指定了分辨率 ⇒ **我们自己会设它**,那就别再去读继承来的行数:
        # 读回来的是**上一次**扫描的,而这一次马上要改掉。用错的分母算出来的预算
        # 会在两个方向上都错(设得更小 ⇒ 预算过大、卡住晚发现;设得更大 ⇒ 预算
        # 不够、每一帧都超时)。
        if lines_override:
            est = estimate_scan_seconds(int(lines_override), float(line_time))
            return wait_budget_s(est, floor_s=_V1_FLOOR_S)
        lines = None
        try:
            from mast.io.nanonis_files import parse_buffer_get

            rec = context.safe_call("Scan_BufferGet")
            self._all_calls.append(rec)
            if not rec.error:
                parsed = parse_buffer_get(rec.return_value)
                lines = parsed.get("lines") or parsed.get("pixels")
        except Exception:  # noqa: BLE001 — 读不到就用出厂常数,不让预检失败
            logger.debug("PreScanCheck: 读不回缓冲区行数(用出厂等待上限)",
                         exc_info=True)
        if not lines or lines <= 0:
            # 读不回来 → 盲估最坏的常见行数。
            #
            # ⚠️ 这里原来 ``return DEFAULT_WAIT_TIMEOUT_S`` —— 那个模块常数是在
            # **import 时**按 ``_BLIND_LINES × DEFAULT_LINE_TIME_S`` 算死的。
            # 2026-08-12 每线时间改成查档位表(0.1 → 50 nm 的 1.0)之后,它就
            # 立刻低估 10 倍:512 行 × 1.0 × 2 = 1024 s,而那个常数是 163 s。
            # 同一个「限额的计价单位由别处决定就必须派生」的陷阱,只是往下藏了
            # 一层 —— 用**这一次真的会用的** line_time 重算,常数只剩「连宽度都
            # 不知道」那一种退化情形还用得上。
            lines = _BLIND_LINES
        # 公式与 ScanAt 共用同一份(scan_policy.wait_budget_s),下限是这一步自己的
        # v1 常数 —— **只有下限可以各不相同**,公式不行。
        est = estimate_scan_seconds(int(lines), float(line_time))
        return wait_budget_s(est, floor_s=_V1_FLOOR_S)

    def plan(self, params: dict) -> list[CompositeStep]:
        cx = params["center_x_m"]
        cy = params["center_y_m"]
        width = params["width_m"]
        # SetScanSpeed numeric args. The speed used to be the literal
        # ``width * 10``, which is ``width / 0.1`` — i.e. the line time was
        # baked into it twice, once as a constant and once as a factor. Deriving
        # it means the two can no longer disagree; at the default line time the
        # value is bit-identical to the old one.
        line_time = resolve_line_time_s(width, params.get("line_time_s"))
        line_speed = width / line_time
        # 这一次预扫描要扫多少行。``None`` = 不设,继承上一次扫描留下的分辨率
        # (历史行为)。给了就自己设一份 —— 见 ``pixels`` 那条 ParameterSpec。
        want_px = _num(params.get("pixels"), None)
        want_px = int(want_px) if want_px else None
        # 预算 = **max(调用方显式给的, 按真实行数×真实线时派生的)** —— 显式值是
        # 下限不是上限,理由见 run_composite 里那段。
        #
        # 直接调 plan()(测试/离线)时拿不到 context,派生值缺席 —— 那时用**同一个
        # line_time** 现算一个盲估,而不是那个 import 时算死的模块常数:后者永远
        # 按 0.1 s/线,一旦档位表给了别的值它就偏 10 倍(见 _derived_timeout_s)。
        # 盲估的分母:**我们自己要设的行数优先**。只有在「连要扫几行都不知道」
        # (不设分辨率、继承现场)时才退回 `_BLIND_LINES` 的最坏值。
        # 不这么做,同一次调用在 run_composite 路径(有 lines_override)和直接调
        # plan() 路径上会算出两个预算 —— 两条路给出两个数,迟早有人拿错的那个
        # 去解释一次超时。
        blind_s = wait_budget_s(
            estimate_scan_seconds(want_px or _BLIND_LINES, line_time),
            floor_s=_V1_FLOOR_S)
        wait_s = max(
            _num(params.get("wait_timeout_s"), 0.0),
            _num(getattr(self, "_derived_wait_s", None), blind_s),
        )

        return [
            CompositeStep(
                step_id="configure",
                skill_name="ConfigureScan",
                params={
                    "center_x_m": cx,
                    "center_y_m": cy,
                    "width_m": width,
                    # 验证使用方图，以便正反扫判据保留二维形貌信息。
                    # 帧时由行数与每线时间决定；仅缩小高度不会减少本流程保留的扫描行数。
                    "height_m": width,
                },
                optional=False,
                checkpoint_after=False,
                tags=("setup",),
            ),
            CompositeStep(
                step_id="set_speed",
                skill_name="SetScanSpeed",
                params={
                    "fwd_speed": line_speed,
                    "bwd_speed": line_speed,
                    "fwd_line_time": line_time,
                    "bwd_line_time": line_time,
                    "keep_const": 0,
                },
                # v1 used self.step() (not step_or_fail) for SetScanSpeed —
                # speed-tweak failure shouldn't kill the prescan.
                optional=True,
                checkpoint_after=False,
                tags=("setup",),
            ),
            # 仅在调用方明确指定分辨率时插入设置步骤；省略时不发送参数。
            # 帧时由行数和每线时间决定，改变视野高度不能替代调整行数。
            *([CompositeStep(
                step_id="set_buffer",
                skill_name="SetScanBuffer",
                params={"pixels": want_px, "lines": want_px},
                # 与 ScanAt 同序:**必须排在 ConfigureScan 之后** ——
                # ConfigureScan 内部走 Scan_BufferSet(channels, 0, 0)。
                optional=False,
                checkpoint_after=False,
                tags=("setup",),
            )] if want_px else []),
            CompositeStep(
                step_id="start_scan",
                skill_name="StartScan",
                params={},
                optional=False,
                checkpoint_after=False,
                tags=("scan",),
            ),
            # 优先从本次保存的文件读取正反扫数据。
            # 实时缓冲可能包含未完成或其他帧的数据，不能自动视为同源。
            # 文件读取为可选步骤；回落路径须如实说明来源与可判定性。
            CompositeStep(
                step_id="wait_scan",
                skill_name="WaitScanComplete",
                # Default budget is derived from the line count actually in the
                # buffer — see DEFAULT_WAIT_TIMEOUT_S / _derived_timeout_s.
                params={"timeout_ms": int(wait_s * 1000)},
                optional=False,
                checkpoint_after=True,    # checkpoint once the scan finishes
                tags=("wait",),
            ),
            # **存盘必须在等待之后** —— 扫描没停就存,存下来的是半帧。
            CompositeStep(
                step_id="save_scan",
                skill_name="SaveScan",
                params={},
                optional=True,
                checkpoint_after=False,
                tags=("save",),
            ),
            CompositeStep(
                step_id="latest_file",
                skill_name="GetLatestScanFile",
                # 只看**这次扫描期间**写出来的文件。裸取「最新」会在一次扫描落多个
                # 文件时取错帧(2026-08-09 栽过);再加上下面的几何核对,
                # 「这是不是我这一帧」有两个独立答案。
                params={"max_age_s": int(wait_s) + 15},
                optional=True,
                checkpoint_after=False,
                tags=("save",),
            ),
        ]

    # --- Hooks ---

    def on_step_result(self, step: CompositeStep, sub_result) -> None:
        # WaitScanComplete reports success on EVERY way a scan can end — timeout,
        # finished, and stopped part-way. Surface both failure modes so
        # run_composite can fail honestly (KNOWN_ISSUES §2.24).
        if step.step_id == "wait_scan":
            data = getattr(sub_result, "data", {}) or {}
            self._executor.set_partial(
                "wait_timed_out", bool(data.get("timed_out", False)),
            )
            self._executor.set_partial(
                "wait_stopped_early", bool(data.get("stopped_early", False)),
            )
            # 「它从没开始」要单独带出来 —— 报错句子要照它分岔,见下面 fail() 那段。
            self._executor.set_partial(
                "wait_never_started", bool(data.get("never_started", False)),
            )
            self._executor.set_partial("scan_lines_done", data.get("lines_done"))
            self._executor.set_partial("scan_lines_total", data.get("lines_total"))

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        # Quality fields are populated by run_composite *after* the graph
        # finishes; aggregate just echoes whatever partial_data holds.
        out = {
            "tip_ready": progress.partial_data.get("tip_ready"),
            "similarity": progress.partial_data.get("similarity"),
            "recommendation": progress.partial_data.get("recommendation"),
        }
        # 并排上报的两个数 + 「为什么判不了」。缺席的键 = 那件事没发生,
        # 与 safe_mode_raw 同一个惯例。
        for key in ("trace_retrace_correlation", "legacy_cosine_similarity",
                    "unusable_reason", "quality_source", "read_failure",
                    # 「起伏不足,换个地方重量」与「读不到,查通信」都让
                    # correlation 为 None,但指向完全不同的下一步 —— 别再折叠成
                    # 同一句话(这个文件已经因为把三种读失败折叠成一个 None 栽过)。
                    "abstain_reason", "corrugation_rms_m"):
            val = progress.partial_data.get(key)
            if val is not None:
                out[key] = val
        # Only present when SAFE mode rewrote the verdict — an absent key means
        # "this is the real verdict", which is the distinction that matters.
        raw = progress.partial_data.get("safe_mode_raw")
        if raw is not None:
            out["safe_mode_raw"] = raw
        return out

    # --- run_composite override: graph + post-scan quality eval ---

    def run_composite(self, context, params: dict) -> SkillResult:
        """把扫描框借来用一下,**用完放回去**。

        ## 2026-08-12 要求：「那些图被设置成扫描 50nm*2.5nm,不知道为什么」

        2.5 nm 是 ``50 nm × 0.05`` —— 本技能 ``plan()`` 里那个细长条,逐位吻合。
        细长条本身是**有意的**(比对正反扫不需要方图,窄条走得快);
        **错的是它扫完不放回去** —— 一个自称「快速看一眼」的检查,
        永久改掉了用户的扫描几何,下一张图就继承了 20:1 的框。

        ## 顺带解掉一个卡了很久的顾虑

        ``plan()`` 的注释记着:本技能在 2.5 nm 高度里扫 **256 条几乎重合的线**、
        花 51 s 把同一条线量了 256 遍,而「没有顺手把线数设成 1」的理由是
        **「那会改变每一个调用方的硬件行为」**。

        但它**已经在改了** —— 它改了宽高,只是没人去看框被留成了什么样。
        「不敢改分辨率」与「随手改了几何且不还原」不能同时成立。
        **一旦用完放回去,那个顾虑自动消失。** 线数的事仍留给用户定夺
        (见 `docs/v2/design/forge_scan_working_point.md`),但至少几何不再泄漏出去。

        恢复是**尽力而为**:读不到原框就不动(不猜),放不回去只记一句 ——
        它绝不能把一次成功的检查变成失败。
        """
        before = None
        try:
            rec = context.safe_call("Scan_FrameGet")
            if not getattr(rec, "error", ""):
                before = getattr(rec, "return_value", None)
        except Exception:  # noqa: BLE001 — 借不到就不还,后面照跑
            before = None

        try:
            return self._run_composite_inner(context, params)
        finally:
            self._restore_frame(context, before)

    @staticmethod
    def _restore_frame(context, before) -> None:
        """恢复之前读到的扫描框；无法读取原值时保持现状。
        先解开 error/raw/body 信封，再兼容扁平序列，不能把 body 当成单个坐标。"""
        if before is None:
            return
        try:
            body = before
            # ① 剥信封:(error, raw, body) —— 真机走这一支。
            #
            # 判据要看 **body 自己有没有 ≥4 个元素**,不能只看「第三项是不是序列」:
            # 一份「每个数都裹在 1-元组里」的扁平回包
            # ``((cx,),(cy,),(w,),(h,),(ang,))`` 的第三项 ``(w,)`` 也是序列,
            # 只看类型会把它当信封剥掉,只剩一个 w —— 又一次「不还原」。
            if (isinstance(before, (tuple, list)) and len(before) >= 3
                    and isinstance(before[2], (tuple, list))
                    and len(before[2]) >= 4):
                body = before[2]
            # ② 再解 1-元组包裹(Nanonis 的数值数组字段常长这样)。
            vals = [v[0] if isinstance(v, (tuple, list)) and v else v
                    for v in (body if isinstance(body, (tuple, list)) else [])]
            nums = [float(v) for v in vals
                    if isinstance(v, (int, float))]
            if len(nums) < 4:
                logger.debug("PreScanCheck: 原扫描框读回来看不懂(%r),不还原", before)
                return
            cx, cy, w, h = nums[0], nums[1], nums[2], nums[3]
            ang = float(nums[4]) if len(nums) > 4 else 0.0
            rec = context.safe_call("Scan_FrameSet", cx, cy, w, h, ang)
            if getattr(rec, "error", ""):
                logger.info("PreScanCheck: 扫描框没能还原(%s)—— "
                            "当前框仍是预扫描的细长条 %.3g×%.3g m",
                            rec.error, w, h)
        except Exception:  # noqa: BLE001 — 还原失败绝不许把成功的检查变成失败
            logger.debug("PreScanCheck: 扫描框还原失败(已忽略)", exc_info=True)

    def _run_composite_inner(self, context, params: dict) -> SkillResult:
        threshold = params.get("quality_threshold", 0.8)

        # ⑫ tip-crash guard: a spot that has already crashed the tip ≥ threshold
        # times must be escaped (withdraw + coarse move), not re-probed — refusing
        # the pre-scan here is part of breaking the in-place spin (5305868e).
        from mast.core.tip_crash_tracker import crash_guard
        escape = crash_guard(context, params.get("center_x_m"),
                             params.get("center_y_m"))
        if escape:
            return self.fail(escape, repeated_crash=True)

        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        self._executor = executor

        # 按**这一步真的会扫多少行 × 真的会用多慢**派生预算(R7)。
        # 必须在 plan() 之前:计划一旦生成,timeout 就写死在步骤参数里了。
        #
        # ═══════════════════════════════════════════════════════════════════
        # 显式值现在是**下限**,不是上限(2026-08-12)
        # ═══════════════════════════════════════════════════════════════════
        #
        # 原来是 ``if params.get("wait_timeout_s") is None:`` —— 调用方一给值,
        # 派生就整个关掉。而 forge 的 verify **正好给了**(``wf.scan_timeout_s``
        # = 300 s),它给的理由写在 ``_tip_phases.py`` 的注释里:「PreScanCheck 的
        # 出厂预算是 15 s」。**那个理由 2026-08-10 就不成立了**(预算改成派生的
        # 那天),但注释和代码都留着 —— 「修好之后,旧理由会静静变成假话」。
        #
        # 于是这一版把每线时间从 0.1 改成档位表的 1.0 之后,帧时从 52 s 变 512 s,
        # 而 forge 那个 300 s 会让**每一次 verify 都超时**。这将是「限额的计价
        # 单位由别处决定就必须派生」在本仓的**第四次**。
        #
        # 修法不是去 forge 里把 300 改大 —— 那只是把同一个耦合挪个地方。
        # 改成 ``max(显式, 派生)``:
        #   · 显式值仍然有效(用户想等更久,照办);
        #   · 但**没有任何调用方能把预算设成短于扫描本身** —— 那种值不表达
        #     「我只愿意等这么久」,它表达的是「保证失败」。
        _px = _num(params.get("pixels"), None)
        self._derived_wait_s = self._derived_timeout_s(
            context,
            resolve_line_time_s(params.get("width_m"), params.get("line_time_s")),
            lines_override=int(_px) if _px else None)

        all_good = executor.run_plan(iter(self.plan(params)))

        # Mandatory pre-scan steps failed — propagate the abort reason
        if not all_good:
            data = self.aggregate(executor.sub_results, executor.progress)
            data["_progress"] = executor.progress.to_dict()
            return self.fail(
                executor.progress.aborted_reason or "pre-scan aborted",
                **data,
            )

        # A pre-scan line that was cut short measures nothing. Reporting
        # "tip not ready" off a truncated line would be a repair prompt
        # manufactured out of a missing measurement — the same failure this file
        # already guards against by seeding tip_ready=None rather than False.
        #
        # No Scan_Action here, unlike the timeout branch below: this path is only
        # reached AFTER Scan_StatusGet read 0, so the scan is already stopped.
        # Stopping it again would be a hardware write on a read-only conclusion.
        if (executor.progress.partial_data.get("wait_stopped_early")
                or executor.progress.partial_data.get("wait_never_started")):
            data = self.aggregate(executor.sub_results, executor.progress)
            data["_progress"] = executor.progress.to_dict()
            # 预检查离开时须停止扫描，使后续粗动前置状态准确。
            data[VERIFIED_STATE_KEY] = {"scan_running": False}
            done = executor.progress.partial_data.get("scan_lines_done")
            total = executor.progress.partial_data.get("scan_lines_total")
            where = (f" ({done}/{total} 行)"
                     if done is not None and total else "")
            # 等待和采集失败必须由失败原因解释，不能用计划参数描述结果。
            if executor.progress.partial_data.get("wait_never_started"):
                return self.fail(
                    f"预扫描**从未开始**{where} —— `StartScan` 成功返回了,但在"
                    f"整个开扫宽限期里一次都没看到扫描在跑,缓冲区也是空的。"
                    f"这**不是**「被谁停了」:没有人需要去查 Stop 记录。"
                    f"该查的是发起时序(仪器是否还在处理上一条设置)。",
                    **data,
                )
            return self.fail(
                f"预扫描中途停止{where} —— 这条线没扫完,测不出针尖状态。"
                f"不是超时:去查是谁停的(用户 Stop / Nanonis 自停 / 安全停机)。",
                **data,
            )

        # WaitScanComplete returns success even on timeout — handle that
        # as a hard failure (v1 behaviour: stop scan + fail).
        if executor.progress.partial_data.get("wait_timed_out"):
            stopped = False
            try:
                stop = context.safe_call("Scan_Action", 1, 0)  # action=1: stop
                self._all_calls.append(stop)
                stopped = not getattr(stop, "error", "")
            except Exception:
                pass
            data = self.aggregate(executor.sub_results, executor.progress)
            data["_progress"] = executor.progress.to_dict()
            # 同上:这条路**真的下发了停止**,成了就把事实带出去。
            # 没成(safe_call 报错)就**什么都不声明** —— 声明一个没做到的事实,
            # 比不声明更坏:下游会拿它当真的去做粗动。
            if stopped:
                data[VERIFIED_STATE_KEY] = {"scan_running": False}
            return self.fail("Pre-scan timed out", **data)

        # 5. Evaluate line quality (raw safe_call — not a graph step,
        #    mirrors the same pattern in full_scan._check_scan_data).
        #    similarity is None when the line data could not be read
        #    (hardware error / empty / exception) — that is *inconclusive*,
        #    NOT a tip-ready verdict (fail-OPEN here would falsely green-light
        #    a full scan on a tip we never actually measured).
        quality = self._evaluate_line_quality(context, params.get("width_m") or 0.0)
        similarity = quality.get("correlation")

        # ⚠️ 单一收口:**非有限数绝不许走到比较那一行。**
        # ``bool(NaN >= threshold)`` 是 ``False``,于是一个**算不出来**的判据会静默
        # 变成「针尖不合格」,而这里的下游是 forge 的**打脉冲**(不可逆)。
        # 上面 `_frame_gate` 已经在**输入**侧拦了已知的那条路(没扫完的图带 NaN 行);
        # 这一道收口管的是**其余任何来源** —— 它不需要知道原因,只需要保证
        # 「算不出来」永远变成三态里的 ``None``,而不是 ``False``。
        if similarity is not None:
            import math as _math
            try:
                finite = _math.isfinite(float(similarity))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                logger.warning("PreScanCheck: 正反扫相关性不是有限数(%r),"
                               "按判不了处理(绝不当成「针尖不合格」)", similarity)
                quality = dict(quality)
                quality["correlation"] = None
                quality.setdefault("abstain_reason", (
                    f"正反扫一致性算出来不是有限数({similarity!r}),判不了针尖好坏。"
                    "**这是弃权,不是「针尖不合格」** —— 下一步是重新量,不是修针。"))
                quality.setdefault("unusable_reason", quality["abstain_reason"])
                similarity = None

        safe_mode_raw = None
        if similarity is None:
            tip_ready = None  # inconclusive — caller must not assume good tip
            recommendation = "inconclusive"
        else:
            tip_ready = bool(similarity >= threshold)
            recommendation = "none" if tip_ready else "mild_conditioning"
            # SAFE mode: this verdict + its "mild_conditioning" recommendation are
            # a tip-repair prompt, which SAFE suppresses at the source. The
            # measured similarity itself stays truthful — it is a number, not a
            # verdict. `None` (could not read the line) is NOT overridden: that is
            # an instrument/communication fact, not a tip-quality claim.
            if not tip_ready and _safe_mode_active():
                safe_mode_raw = {"tip_ready": tip_ready, "recommendation": recommendation,
                                 "threshold": float(threshold)}
                tip_ready = True
                recommendation = "none"

        # ``similarity`` 这个键名保持不变(下游读它的地方很多),但它现在装的是
        # **去趋势去均值、允许迟滞平移的相关系数**,不再是绝对高度余弦。
        # 旧那个数并排上报,名字不同 —— 两个不是同一个量,不共用一个名字。
        executor.set_partial("similarity", similarity)
        executor.set_partial("trace_retrace_correlation", similarity)
        executor.set_partial("legacy_cosine_similarity", quality.get("legacy_cosine"))
        if quality.get("unusable_reason"):
            executor.set_partial("unusable_reason", quality["unusable_reason"])
        # 数据是从**文件**还是从**实时缓冲区**来的 —— 两条路的可信度不一样,
        # 而报告里从前看不出用了哪条。
        executor.set_partial("quality_source", quality.get("source"))
        if quality.get("read_failure"):
            executor.set_partial("read_failure", quality["read_failure"])
        # 起伏不足的弃权:与「读不到」分开上报,因为下一步不同
        # (那边查通信/去 .sxm 取数,这边**换个有形貌的地方重新量**)。
        if quality.get("abstain_reason"):
            executor.set_partial("abstain_reason", quality["abstain_reason"])
            executor.set_partial("corrugation_rms_m", quality.get("corrugation_rms_m"))
        executor.set_partial("tip_ready", tip_ready)
        executor.set_partial("recommendation", recommendation)
        if safe_mode_raw is not None:
            executor.set_partial("safe_mode_raw", safe_mode_raw)

        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()

        # 离开预扫描检查前停止扫描，成功与弃权分支都需要收尾。
        # 停止已停止的扫描是幂等操作，并刷新状态。
        # 停止失败时不能声明 scan_running=False。
        try:
            stop = context.safe_call("Scan_Action", 1, 0)  # action=1: stop
            self._all_calls.append(stop)
            if not getattr(stop, "error", ""):
                data[VERIFIED_STATE_KEY] = {"scan_running": False}
        except Exception:  # noqa: BLE001 — 收尾绝不许把一个成功的检查变成失败
            logger.debug("PreScanCheck: 收尾 StopScan 失败(已忽略)", exc_info=True)

        return self.ok(**data)

    # --- Quality evaluation: raw safe_call (no sub-skill exists) ---

    def _quality_from_saved_frame(self, width_m: float) -> "dict | None":
        """从**刚存盘的那个 .sxm** 读正反两面。拿不到返回 ``None``(调用方回落)。

        读文件用的是既有的单一入口 :func:`mast.io.nanonis_files.read_sxm` +
        :func:`~mast.io.nanonis_files.sxm_oriented_frames` —— 后者负责把反扫的
        镜像和 ``:SCAN_DIR: up`` 的行序归位,**不在这里再写一份**(那个函数的注释
        记着:从前有三种私人拼写,而缺席的那一份正好是用户看到的那条路)。

        ## 「这是不是我这一帧」有两个独立答案

        1. ``GetLatestScanFile`` 只看这次扫描期间写出来的文件(``max_age_s``);
        2. 文件头里的 ``SCAN_RANGE`` 必须与本次配置的帧宽对得上。

        两条都过才用它。裸取「最新」在一次扫描落多个文件时会取错帧 ——
        2026-08-09 那次「打开另一帧」栽的就是这个,而症状是一张看起来完全正常的图。
        """
        res = (self._executor.sub_results.get("latest_file")
               if getattr(self, "_executor", None) else None)
        path = str(((getattr(res, "data", None) or {}).get("path")) or "")
        if not path:
            return None
        try:
            from mast.io.nanonis_files import read_sxm, sxm_oriented_frames

            scan = read_sxm(path)
            fr = sxm_oriented_frames(scan, "Z")
        except Exception as exc:  # noqa: BLE001 — 读不了就回落,不让预检失败
            logger.debug("PreScanCheck: 读 %s 失败(回落到缓冲区): %s", path, exc)
            return None
        fwd, bwd = fr.get("forward"), fr.get("backward")
        if fwd is None or bwd is None:
            # 单方向文件 —— 正反扫一致性无从谈起,但这不是「读不到」。
            return None
        # 几何核对:这一帧是不是我刚配置的那一帧。
        got_nm = fr.get("width_nm")
        want_nm = float(width_m) * 1e9
        if got_nm and abs(float(got_nm) - want_nm) > max(0.05 * want_nm, 0.1):
            logger.info("PreScanCheck: 最新的 .sxm 帧宽 %.4g nm 与本次配置的 %.4g nm "
                        "对不上,不用它(回落到缓冲区)", float(got_nm), want_nm)
            return None

        # 文件年龄和扫描范围不能区分一次扫描的早期空存盘与完整存盘。
        # 无已采集行的文件返回 None，由调用方尝试缓冲回落并说明数据来源。
        # 不要修改 frame_gate 来放行空帧。
        try:
            import numpy as _np0
            _rows = int(_np0.isfinite(_np0.asarray(fwd, dtype=float)).all(axis=1).sum())
        except Exception:  # noqa: BLE001 — 判不了就别拦,交给下面的 gate
            _rows = -1
        if _rows == 0:
            logger.info(
                "PreScanCheck: 最新的 .sxm(%s)一行都没扫到 —— 一次扫描的多份存盘里"
                "拿到了空的那一份,不用它(回落到缓冲区)。", path)
            return None

        # ⚠️ 用**返回的** fwd/bwd:裁到已扫行发生在 _frame_gate 里面。
        fwd, bwd, gate = self._frame_gate(fwd, bwd, source=f"sxm:{path}",
                                          noun="这一帧")
        if gate is not None:
            return gate

        import numpy as _np

        from mast.vision.tip_metrics import trace_retrace_correlation

        a, b = _np.asarray(fwd, dtype=float), _np.asarray(bwd, dtype=float)
        na, nb = _np.linalg.norm(a), _np.linalg.norm(b)
        legacy = (None if (na < 1e-30 or nb < 1e-30)
                  else float(_np.sum(a * b) / (na * nb)))
        return {"correlation": float(trace_retrace_correlation(a, b)),
                "legacy_cosine": legacy,
                "unusable_reason": None,
                "source": f"sxm:{path}"}

    @staticmethod
    def _frame_gate(fwd, bwd, *, source: str, noun: str) -> "tuple":
        """两条取数路径共用的前置，返回 (fwd, bwd, gate)。
        
        gate 为 None 时使用返回的数组继续计算；它们可能已经裁剪为双方均采集的完整行。
        gate 为 dict 时返回无法判定的原因。先取数据，再检查有效性，最后计算质量。
        死平帧缺少形貌信息；低于起伏下限时相关性不足以支持针尖结论，应弃权。
        起伏阈值复用 line_check._DEFAULT_MIN_CORRUGATION_M，避免两份配置漂移。
        公开默认值没有附带实验标定，不能把弃权解释为针尖合格或不合格。
        """
        import numpy as _np

        from mast.vision.frame_validity import acquired_row_mask, judge_frame

        # 先使用 acquired_row_mask 获取正反扫双方完整行的交集。
        # 未扫描行或半扫描行中的 NaN 不能直接进入相关性计算。
        # 先裁取有效数据，再决定是否弃权；无法判定不能转换成针尖不合格。
        af = _np.asarray(fwd, dtype=float)
        ab = _np.asarray(bwd, dtype=float)
        # ⚠️ 记住原来是不是一维:缓冲区那条路交的是**一条线**(``_extract_1d``),
        # 提升成 (1, N) 之后「只有 1 行」是**正常**,不是「没扫完」。
        # 第一版把下限一律写成 2 行,于是整条缓冲区路径全部弃权(8 条测试当场变红)
        # —— 一道**对一种合法输入永远开火**的门,和没有门一样坏。
        was_1d = af.ndim == 1 or ab.ndim == 1
        if af.ndim == 1:
            af = af.reshape(1, -1)
        if ab.ndim == 1:
            ab = ab.reshape(1, -1)
        rows_total = int(min(af.shape[0], ab.shape[0]))
        mask = acquired_row_mask(af, ab)
        n_rows = int(mask.sum()) if mask.size else 0
        cropped = n_rows != rows_total

        # 裁完剩得太少 ⇒ **这才是弃权**(第 2 段)。
        # 二维帧要 ≥2 行(平面拟合);一条线本来就只有 1 行,要 ≥1。
        # 而「拟合不出来」不是「针尖不好」。
        min_rows = 1 if was_1d else 2
        if n_rows < min_rows:
            reason = (
                f"{noun}只有 {n_rows}/{rows_total} 行是扫完整的 —— "
                "这一帧还没有可比较的区域。**这是弃权,不是「针尖不合格」**:"
                "没扫完的图测不出针尖状态,下一步是把图扫完 / 换个地方重新量,"
                "**不是修针**。")
            return af, ab, {"correlation": None, "legacy_cosine": None,
                            "unusable_reason": reason, "abstain_reason": reason,
                            "rows_total": rows_total, "rows_acquired": n_rows,
                            "source": source}
        if cropped:
            af, ab = af[:rows_total][mask], ab[:rows_total][mask]
            if was_1d:
                # 一维进来就一维出去 —— 判据按维度分派(一维 np.correlate /
                # 二维 FFT),悄悄升维等于**换了一个判据**。
                af, ab = af.ravel(), ab.ravel()
            fwd, bwd = af, ab

        verdicts = {}
        for label, side in (("正扫", fwd), ("反扫", bwd)):
            verdict = judge_frame(side)
            if not verdict.usable:
                return af, ab, {
                    "correlation": None, "legacy_cosine": None,
                    "unusable_reason": (f"{label}{noun}是死平的,正反扫一致性"
                                        f"无从谈起 —— {verdict.reason}"),
                    "source": source}
            verdicts[label] = verdict

        # 公开版：CheckLineQuality（skills/paper）不随仓发布，常量就地保留。
        _DEFAULT_MIN_CORRUGATION_M = 15e-12

        # 取**两侧较小**的那个:一次正反扫比较只和它较弱的一侧一样可靠
        # (与上面「fwd 与 bwd 都要过」同一条论证)。
        corrugation = min(float(v.corrugation_rms_m) for v in verdicts.values())
        if corrugation < _DEFAULT_MIN_CORRUGATION_M:
            reason = (
                f"去趋势起伏只有 {corrugation * 1e12:.1f} pm,低于下限 "
                f"{_DEFAULT_MIN_CORRUGATION_M * 1e12:.1f} pm —— 没有形貌就没有可相关"
                "的信号,这一帧上的正反扫一致性不代表针尖好坏。"
                "**这是弃权,不是「针尖不合格」**:平坦干净的区域正常就长这样。"
                "下一步是**换个有形貌的地方重新量**,不是修针。(下限尚未标定。)")
            return af, ab, {
                "correlation": None, "legacy_cosine": None,
                "unusable_reason": reason,
                # 与「读不到」分开的一个键:两者都让 correlation 为 None,
                # 但指向完全不同的下一步(那边查通信,这边换地方)。
                "abstain_reason": reason,
                "corrugation_rms_m": corrugation,
                "source": source}
        return af, ab, None

    def _evaluate_line_quality(self, context, width_m: float = 0.0) -> "dict[str, Any]":
        """比较正反扫形貌，返回 correlation、legacy_cosine 与 unusable_reason。
        
        correlation 为 None 代表读数缺失或缺少可比较信息，不代表针尖合格。
        去趋势、去均值并允许横向平移的 trace_retrace_correlation 用于当前判据。
        绝对高度余弦容易受直流偏置支配，legacy_cosine 仅为兼容旧报告保留，不参与判定。
        """
        from typing import Any  # noqa: F401 — 仅为签名可读

        # 优先读取本次保存文件，缓冲仅为明确标注来源的回落路径。
        from_file = self._quality_from_saved_frame(width_m)
        if from_file is not None:
            return from_file

        try:
            # Grab forward data (channel 0, direction 1=forward)
            rec_fwd = context.safe_call("Scan_FrameDataGrab", 0, 1)
            self._all_calls.append(rec_fwd)
            # Grab backward data (channel 0, direction 0=backward)
            rec_bwd = context.safe_call("Scan_FrameDataGrab", 0, 0)
            self._all_calls.append(rec_bwd)

            # 不同数据缺失原因分别报告，使下一步处置有依据。
            if rec_fwd.error or rec_bwd.error:
                return dict(_INCONCLUSIVE, source="buffer", read_failure=(
                    f"Scan_FrameDataGrab 报错(正扫: {rec_fwd.error or '正常'};"
                    f"反扫: {rec_bwd.error or '正常'})—— 通信/模块问题,不是针尖问题"))

            fwd = self._extract_1d(rec_fwd.return_value)
            bwd = self._extract_1d(rec_bwd.return_value)

            if fwd is None or bwd is None:
                return dict(_INCONCLUSIVE, source="buffer", read_failure=(
                    "Scan_FrameDataGrab 回包解析不出数组"
                    f"(正扫 {'读不懂' if fwd is None else '正常'}、"
                    f"反扫 {'读不懂' if bwd is None else '正常'})—— "
                    "回包形状与预期不符,不是针尖问题"))
            if len(fwd) == 0 or len(bwd) == 0:
                return dict(_INCONCLUSIVE, source="buffer", read_failure=(
                    f"实时缓冲区里是空的(正扫 {len(fwd)} 点、反扫 {len(bwd)} 点)"
                    " —— 缓冲区未必装着刚存盘的那一帧;这一帧的数据要去 .sxm 里取"))

            # Trim to same length
            n = min(len(fwd), len(bwd))
            fwd, bwd = fwd[:n], bwd[:n]

            # 缓冲回落不能确认与保存帧同源，也不能直接套用文件路径的相关阈值。
            # 展平二维帧会丢失行结构；反扫取向也必须通过同源数据独立验证。
            # 本路径只返回 _INCONCLUSIVE，不输出可用于验收的相似度。
            # 仍复用 _frame_gate 检查完整行、死平和起伏，以返回更具体的无法判定原因。
            # 恢复数值判据前必须确认数据来源、维度和取向，并独立标定。
            _cropped_fwd, _cropped_bwd, gate = self._frame_gate(
                fwd, bwd, source="buffer", noun="这条线")
            if gate is not None:
                return gate

            # 缓冲区路径不返回相关分数：它与文件帧采用不同维度和处理口径，
            # 不应互相套用阈值。保留 source=buffer 并给出 read_failure，
            # 让调用方知道下一步应读取这一帧的 .sxm，而不是另找有形貌的区域。
            return dict(_INCONCLUSIVE, source="buffer",
                        read_failure=_BUFFER_NOT_COMPARABLE)
        except Exception as exc:  # noqa: BLE001
            logger.debug("PreScanCheck 线质量评估异常(按判不了处理)", exc_info=True)
            return dict(_INCONCLUSIVE, source="buffer",
                        read_failure=f"读取路径抛异常:{type(exc).__name__}: {exc}")

    @staticmethod
    def _extract_1d(parsed) -> np.ndarray | None:
        """回包里的一维数组。``None`` = **回包读不懂**;空数组 = 读懂了,里面是空的。

        ⚠️ 这两件事在此之前都返回 ``None``(旧实现要求 ``len(raw) > 0`` 才返回),
        于是「缓冲区是空的」与「回包形状不对」在上层无法区分 —— 而它们指向两个
        完全不同的下一步:前者要去 .sxm 里取数据,后者要去查回包解析。
        写这条测试时才发现:我在上层分了三句话,而这一层只给得出两种。
        """
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            raw = parsed[2]
            if hasattr(raw, '__len__'):
                return np.asarray(raw, dtype=np.float64).ravel()
        return None


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(PreScanCheck, context_provider)
