"""ScanAt —— 扫图的意图级入口:说「扫哪、多大」,参数由策略层定。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``

在它之前,模型扫一张图要自己填 ``width_m`` / ``height_m`` / ``line_time_s``,
而「这个尺度下该扫多快、取多少像素」不是它的知识 —— 那是用户按仪器、按样品
积累出来的东西。结果就是模型每次现编一个数字,编得对不对没人知道。

ScanAt 把这件事倒过来:**「不填」是默认路径**,策略层
(:mod:`mast.core.scan_resolver`)按用户的按-尺度参数表把参数补齐。模型只在
用户**逐字点名过**某个值时才传它,而每个传进来的值都会带上来源记录,显示给
用户 —— 「这个 line_time 标着『用户显式』,但我没说过」是一眼能看出来的。

这不是靠提示词恳求模型别填数字(那条路 2026 年的三模式 assembler 走过,
D-discarded),而是**把发明数字的诱因移除**。

教法上对齐 ``ApproachTip``:一个内部确定性决策的一步到位技能,而不是让模型自己
拆成「先查状态再手动 escalate」。
"""

from __future__ import annotations

import logging

from mast.core.scan_policy import tier_names
from mast.core.scan_resolver import PURPOSE_AUTO, ScanIntent, resolve_scan
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import CompositeStep

logger = logging.getLogger(__name__)

#: 显式覆盖通道:可选、无默认。**每一个的 description 都写死「只有用户逐字
#: 说过才传」** —— 这是转述与发明之间唯一的契约,配合 trace 做事后审计。
_EXPLICIT_ONLY = (
    "**只有现场明确给出过这个值时才传它。** 否则就**留空** —— "
    "由分尺度策略表决定。你在这里传的每一个值都会被记成「用户指定」"
    "并展示给他看。"
)


class ScanAt(CompositeSkillGraph):
    """Scan at a location with parameters resolved from the operator's policy."""

    def metadata(self) -> SkillMetadata:
        tiers = ", ".join(tier_names())
        return SkillMetadata(
            name="ScanAt",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在某个位置扫一帧。你给的是**扫哪**（中心）和**多大**（尺寸）"
                "—— 那才是意图。扫描速度、分辨率、反馈增益和默认 setpoint 由"
                "用户的分尺度策略表**确定性地**选出；**它们不是你的知识，"
                "所以不要传**。只有用户自己点了名的那个值，才把对应的可选"
                "参数传进来。"
                "**这是成像的首选方式**：一次调用就完成配置、设分辨率、启动、"
                "等待。**不要**把它拆成 ConfigureScan + SetScanSpeed + StartScan。"
            ),
            parameters=[
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description=(
                        "扫描中心 X，单位 METERS（SI），**不是纳米**。"
                        "换算：100 nm → 100n。"
                    ),
                    unit="m",
                    required=True,
                    min_value=-1e-3,
                    max_value=1e-3,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description=(
                        "扫描中心 Y，单位 METERS（SI），**不是纳米**。"
                        "换算：100 nm → 100n。"
                    ),
                    unit="m",
                    required=True,
                    min_value=-1e-3,
                    max_value=1e-3,
                ),
                ParameterSpec(
                    name="size_m",
                    type="float",
                    description=(
                        "帧的边长，单位 METERS（SI），**不是纳米**。"
                        "换算：100 nm → 100n，50 nm → 50n，1 µm → 1u。"
                        "写成裸的 100（=100 m）就是单位写错了，**会被拒绝**。"
                        "这个尺寸同时决定用哪一档策略来给速度和分辨率。"
                    ),
                    unit="m",
                    required=True,
                    min_value=1e-10,
                    max_value=1e-5,
                ),
                ParameterSpec(
                    name="purpose",
                    type="str",
                    description=(
                        "意图类别，用来挑策略档。"
                        f"'auto'（默认）按尺寸挑。可选档：{tiers}。"
                        "只有当你**有意**让尺寸与参数不匹配时才点名某一档，"
                        "比如想在一个小帧上快速粗看一眼"
                        "（50 nm 尺寸上用 'survey'）。"
                    ),
                    required=False,
                    default=PURPOSE_AUTO,
                ),
                # ── 显式覆盖(全部可选、无默认;不传 = 走策略层) ──────────
                ParameterSpec(
                    name="bias_v",
                    type="float",
                    description=(
                        "样品偏压，单位 VOLTS。" + _EXPLICIT_ONLY + " "
                        "**偏压从来不由策略表决定** —— 它选的是你要成像哪些"
                        "电子态，那是一个物理判断，不是帧尺寸的函数。"
                        "留空 = 保持当前偏压。"
                    ),
                    unit="V",
                    required=False,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="setpoint_a",
                    type="float",
                    description=(
                        "隧道电流 setpoint，单位 AMPERES（SI），**不是 pA**。"
                        "100p = 100 pA。" + _EXPLICIT_ONLY
                    ),
                    unit="A",
                    required=False,
                    min_value=1e-12,
                    max_value=1e-7,
                ),
                ParameterSpec(
                    name="line_time_s",
                    type="float",
                    description="每条扫描线多少秒。" + _EXPLICIT_ONLY,
                    unit="s",
                    required=False,
                    min_value=1e-4,
                    max_value=600.0,
                ),
                ParameterSpec(
                    name="pixels",
                    type="int",
                    description=(
                        "每行像素数（方形帧）。" + _EXPLICIT_ONLY
                    ),
                    unit="px",
                    required=False,
                    min_value=16,
                    max_value=4096,
                ),
                ParameterSpec(
                    name="angle_deg",
                    type="float",
                    description=(
                        "扫描旋转角，单位度。" + _EXPLICIT_ONLY + " "
                        "留空 = 保持当前扫描角。"
                    ),
                    unit="deg",
                    required=False,
                    min_value=-180.0,
                    max_value=180.0,
                ),
                ParameterSpec(
                    name="channels",
                    type="str",
                    description=(
                        "要采的通道，逗号分隔。默认 'Z,Current'。"
                        "只有要**加**通道时才传，比如做 dI/dV 成像时加 lock-in。"
                    ),
                    required=False,
                ),
                ParameterSpec(
                    name="wait_timeout_s",
                    type="float",
                    description=(
                        "等待完成的上限。**留空**则按解析出来的几何自动估"
                        "（行数 × line time × 2 个方向 + 30% 余量）—— "
                        "手填一个短值会把慢扫或高分辨扫描**截断**。"
                    ),
                    unit="s",
                    required=False,
                    min_value=10.0,
                    max_value=7200.0,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=120.0,
            composition_level=3,
            tags=["scan", "imaging", "composite", "policy"],
        )

    # ── 意图 → 参数 ──────────────────────────────────────────────────────────

    #: 参数名 → resolver 的 explicit 键。只有这些能被显式覆盖;PI 增益刻意不在
    #: 列(「用 P=1e-11 扫」不是人话,要调就去档位表或直接用 SetZCtrlGain)。
    _EXPLICIT_KEYS = (
        "bias_v", "setpoint_a", "line_time_s", "pixels", "angle_deg", "channels",
    )

    def _resolve(self, params: dict):
        explicit = {
            key: params[key]
            for key in self._EXPLICIT_KEYS
            if params.get(key) is not None
        }
        intent = ScanIntent(
            center_x_m=params["center_x_m"],
            center_y_m=params["center_y_m"],
            size_m=params["size_m"],
            purpose=str(params.get("purpose") or PURPOSE_AUTO),
            explicit=explicit,
        )
        return resolve_scan(intent)

    @staticmethod
    def _wait_timeout_s(params: dict, estimated_s: float) -> float:
        """等待预算按最终扫描几何和余量推导，显式值作为下限，防止正常慢扫被提前截断。"""
        # 等待预算取显式下限与几何推导值的较大者。
        # 显式预算不得截短一帧按当前分辨率和线时间所需的采集时间。
        # 保留调用方较长预算，同时避免以等待超时误判已经完成的处理动作。
        derived = None
        explicit = params.get("wait_timeout_s")
        if explicit is not None:
            try:
                explicit_s = float(explicit)
            except (TypeError, ValueError):
                explicit_s = None
            if explicit_s is not None:
                from mast.core.scan_policy import wait_budget_s as _wb
                derived = _wb(estimated_s, floor_s=300.0)
                return max(explicit_s, derived)
        # 等待预算复用 scan_policy.wait_budget_s，避免多个路径各自维护不一致常数。
        from mast.core.scan_policy import wait_budget_s

        return wait_budget_s(estimated_s, floor_s=300.0)

    # ── 计划 ─────────────────────────────────────────────────────────────────

    def plan(self, params: dict) -> list[CompositeStep]:
        resolved = self._resolve(params)
        steps: list[CompositeStep] = []

        # 1. bias —— 排在最前:偏压变了,后面的 setpoint / 反馈才在正确的工作点
        #    上。注意 SetBias 刻意不要求 z_controller_on(STS 常需要关反馈改
        #    bias),这里不改那个既有语义。
        if resolved.set_bias is not None:
            steps.append(CompositeStep(
                step_id="set_bias",
                skill_name="SetBias",
                params=dict(resolved.set_bias),
                optional=False,
                tags=("setup",),
            ))

        # 2. setpoint
        if resolved.set_setpoint is not None:
            steps.append(CompositeStep(
                step_id="set_setpoint",
                skill_name="SetSetpoint",
                params=dict(resolved.set_setpoint),
                optional=False,
                tags=("setup",),
            ))

        # 3. Z 反馈 PI(档位表里配过才有)
        if resolved.set_zctrl_gain is not None:
            steps.append(CompositeStep(
                step_id="set_gain",
                skill_name="SetZCtrlGain",
                params=dict(resolved.set_zctrl_gain),
                optional=False,
                tags=("setup",),
            ))

        # 4. 帧几何 + 通道 + 速度(ConfigureScan 自己会由 line_time 推导速度)
        steps.append(CompositeStep(
            step_id="configure",
            skill_name="ConfigureScan",
            params=dict(resolved.configure_scan),
            optional=False,
            tags=("setup",),
        ))

        # 5. 分辨率 —— **必须排在 ConfigureScan 之后**:ConfigureScan 内部调
        #    ``Scan_BufferSet(channels, 0, 0)``,0/0 的语义(保持还是重置)尚未
        #    在真机上证实。排在后面,无论哪种语义结果都正确。
        steps.append(CompositeStep(
            step_id="set_buffer",
            skill_name="SetScanBuffer",
            params=dict(resolved.set_scan_buffer),
            optional=False,
            tags=("setup",),
        ))

        steps.append(CompositeStep(
            step_id="start_scan",
            skill_name="StartScan",
            params={},
            optional=False,
            tags=("scan",),
        ))
        steps.append(CompositeStep(
            step_id="wait_scan",
            skill_name="WaitScanComplete",
            params={"timeout_ms": int(
                self._wait_timeout_s(params, resolved.estimated_scan_s) * 1000
            )},
            optional=False,
            checkpoint_after=True,
            tags=("wait",),
        ))
        return steps

    # ── 钩子 ─────────────────────────────────────────────────────────────────

    def on_step_result(self, step: CompositeStep, sub_result) -> None:
        data = getattr(sub_result, "data", {}) or {}
        if step.step_id == "wait_scan":
            # WaitScanComplete 在**每一种**扫描结束方式上都报 success —— 超时、
            # 扫完、以及中途被停下。三种都要提上来,让 _decide_outcome 能如实失败。
            # 「硬件停止」不等于「达标」。
            #
            # 这行注释在 v6.1.2 之前就是这么写的,而实现只做了超时那一半:
            # 用户在主路径上按 Stop 打断一帧,ScanAt 照样报成功。
            # KNOWN_ISSUES §2.24 —— 一句写对了的原则和一个只实现了它一半的函数
            # 可以长期共存而不显眼,因为读注释的人会以为它做到了。
            self._executor.set_partial("wait_timed_out",
                                       bool(data.get("timed_out", False)))
            self._executor.set_partial("wait_stopped_early",
                                       bool(data.get("stopped_early", False)))
            self._executor.set_partial("scan_lines_done", data.get("lines_done"))
            self._executor.set_partial("scan_lines_total", data.get("lines_total"))
            # The budget/elapsed/extension evidence the timeout message prints.
            # Lifted unconditionally — a number computed only on the path that
            # needs it is a number that is missing exactly when it is asked for.
            for key in ("budget_s", "elapsed_s", "extensions",
                        "lines_done", "lines_total"):
                if data.get(key) is not None:
                    self._executor.set_partial(key, data[key])
        elif step.step_id == "set_buffer":
            self._executor.set_partial("pixels", data.get("pixels"))
            self._executor.set_partial("lines", data.get("lines"))
            self._executor.set_partial("resolution_verified",
                                       bool(data.get("verified", False)))
        elif step.step_id == "configure":
            self._executor.set_partial("angle_deg", data.get("angle_deg"))
            self._executor.set_partial("linear_speed_m_s",
                                       data.get("linear_speed_m_s"))

    def aggregate(self, sub_results: dict, progress) -> dict:
        data = dict(progress.partial_data)
        stashed = getattr(self, "_resolved_snapshot", None)
        if stashed is not None:
            data.update(stashed)
        return data

    def _decide_outcome(self, all_good: bool, progress, data: dict):
        """帧没扫完 = 失败,哪怕每一步都「成功」了。

        ``WaitScanComplete`` 无论超时、扫完还是中途被停下都返回 success ——
        不在这里拦住,一次被截断的扫描会以「扫完了」的面目出现在结果里。
        这正是本项目记过的假成功模式:硬件「停止」不等于「达标」。

        两种截断分开报,因为**用户要做的事不同**:超时是「参数估短了,调大
        timeout 再来」,中途停止是「去查是谁停的」(用户按了 Stop / Nanonis
        自己停 / 安全停机)。合成一句会把人送去调一个根本没问题的参数。
        """
        if all_good and progress.partial_data.get("wait_stopped_early"):
            done = progress.partial_data.get("scan_lines_done")
            total = progress.partial_data.get("scan_lines_total")
            where = (f"(扫到 {done}/{total} 行)"
                     if done is not None and total else "(行数未知)")
            return False, (
                f"扫描中途停止{where} —— 这一帧没有扫完,不要当作扫好的图使用。"
                f"可能是用户按了 Stop、Nanonis 自行停止,或安全停机;"
                f"**不是**超时,调大 timeout 不解决问题。"
            )
        if all_good and progress.partial_data.get("wait_timed_out"):
            # WHAT THIS MESSAGE MUST CONTAIN (2026-08-05). It used to print the
            # ESTIMATE and nothing else — not the budget actually waited, not the
            # elapsed time, not the line count. A real timeout on the instrument was
            # therefore diagnosed twice over from this sentence alone: once as
            # "the +30 % headroom is not applied on this path" and once as "the
            # explicit wait_timeout_s was ignored". Both were false — the code
            # does apply the headroom and does honour the explicit value — but
            # the message could not have told anyone that, because the only
            # number in it was the one quantity that is NOT the budget.
            pd = progress.partial_data
            est = (getattr(self, "_resolved_snapshot", None) or {}).get(
                "estimated_scan_s")
            bits = []
            if est:
                bits.append(f"按解析参数估计需要 {est:.0f} s")
            budget = pd.get("budget_s")
            if isinstance(budget, (int, float)):
                bits.append(f"等待预算 {budget:.0f} s")
            waited = pd.get("elapsed_s")
            if isinstance(waited, (int, float)):
                bits.append(f"实际等了 {waited:.0f} s")
            ext = pd.get("extensions")
            if ext:
                bits.append(f"其中因扫描仍在推进延长过 {ext} 次")
            done, total = pd.get("lines_done"), pd.get("lines_total")
            if done is not None:
                bits.append(f"放弃时已采 {done}/{total if total else '?'} 行")
            hint = ("(" + ";".join(bits) + ")") if bits else ""
            return False, (
                f"扫描在等待预算内没有完成{hint} —— 帧不完整,不要当作扫好的图使用。"
                "若「已采行数」几乎等于总行数,那是估计帧时偏小而不是扫描卡住:"
                "显式传一个更大的 wait_timeout_s 即可。"
            )
        return super()._decide_outcome(all_good, progress, data)

    # ── 执行 ─────────────────────────────────────────────────────────────────

    def run_composite(self, context, params: dict) -> SkillResult:
        # 先解析一次,把 trace 与档位记下来 —— 无论后面成功还是失败,用户都
        # 要能看到「这次打算用什么参数、每个数字哪来的」。失败时尤其需要。
        try:
            resolved = self._resolve(params)
        except ValueError as exc:
            return self.fail(str(exc))

        self._resolved_snapshot = {
            "tier_name": resolved.tier_name,
            "center_x_m": resolved.configure_scan["center_x_m"],
            "center_y_m": resolved.configure_scan["center_y_m"],
            "size_m": resolved.configure_scan["width_m"],
            "line_time_s": resolved.configure_scan["line_time_s"],
            "requested_pixels": resolved.set_scan_buffer["pixels"],
            "estimated_scan_s": resolved.estimated_scan_s,
            "param_trace": resolved.trace,
            "param_summary": resolved.summary_lines(),
            "policy_warnings": resolved.warnings,
        }
        for warn in resolved.warnings:
            logger.info("ScanAt 参数卫生: %s", warn)

        result = self._graph_execute(context, params)

        # 超时后必须把扫描停掉。一个还在跑的扫描会挡住之后的每一步(改帧、移动、
        # 换参数全要求 scan_not_running),留着它等于让下一个动作莫名其妙失败。
        #
        # ⚠️ **`wait_stopped_early` 刻意不在这里** —— 那条路上扫描**已经停了**
        # (判据就是 `Scan_StatusGet` 读到 0 之后才去数行数的)。再补一发
        # `Scan_Action(1, 0)` 是对着已停的扫描发一条无意义的硬件命令。
        # 「失败了就顺手停一下」看着对称,但它把一个只读的结论变成了一次写操作。
        if (result.data or {}).get("wait_timed_out"):
            try:
                context.safe_call("Scan_Action", 1, 0)   # action=1: stop
            except Exception as exc:  # noqa: BLE001 - 已经在失败路径上了
                logger.warning("ScanAt 超时后停扫失败: %s", exc)

        return result
