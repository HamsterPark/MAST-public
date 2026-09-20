"""STSConditionSeries —— **同一个位置**上，按一张条件表逐条取谱。

设计文档:``docs/v2/design/`` 的「S4 STS」设计,D18/D19/D20(文件名按「S4 STS」
检索;通用层注释不写样品名前缀 —— ``stm_capability_vs_sample_layer.md`` 拍板④)。

## MLS 已经有的,这里不重造(D18)

Nanonis 的 MLS 覆盖「**一条曲线内**多个偏压段」:逐段的起止、点数、整定、积分
时间,以及逐段的 lock-in 开关。「E_F 附近密采、远处稀采」「弱信号段加长积分」
「只在感兴趣的窗口开调制」—— **MLS 全都能做,不要重造**。

MLS **做不到**的是这个流程要的那件事:**多条各自独立的谱,每条有自己的稳定条件**。
一条 MLS 曲线的设定点在开扫之前建立**一次**,Z 全程 hold;而宽偏压测量的标准做法
就是换几个稳定条件各取一条再拼 —— ±3 V 的谱稳在 1 V/100 pA 会在两端把前置放大器
打饱和,稳在 3 V/100 pA 则 E_F 附近的分辨率全丢。那是**不同的针尖-样品间距**,
一条曲线里表达不出来。另外三条 MLS 也给不了:逐条不同的调制幅度、逐窗口的质量
判决、一条坏掉不拖垮其余的部分成功。

⇒ 本流程 = 「一个位置 + 一串条件组名」。每条**自己**要不要用 MLS,是那个条件组
里的一个可选字段,原样透传给既有的 ``SetSTSMLSMode`` / ``SetSTSMLSVals``。

## 采集不自己写,交给逐点引擎

每一条都是 ``CompositeStep(skill_name="SpectroscopyAtPositions", ...)`` 按名字调
一次,位置只有一个。理由与 ``LineSTSAcrossWall`` 逐字相同:.dat 归属那三层
(文件名 / 采前水位 / 位置核对)是这条链上最容易**静默**出错的地方 —— 一个点没落盘,
下一条就会拿到上一条的文件,而两份数据长得一模一样。那套逻辑只该有一份。
按名字走 registry 也意味着版本钉选、安全闸、HITL、进度上报都还是那一套。

## 逐条之间必须重建稳定条件,而且走 BiasSettleChange(D20)

条件表里相邻两条的 ``stab_bias_v`` 完全可能异号(−1 V 综览 → +1 V 综览),**穿零是
必经的**。恒流反馈下偏压穿零会把针尖推进样品 —— 所以每一条的条件建立都走
``BiasSettleChange``(在引擎里),而这个流程只负责**把穿零那一步指出来**,让人事后
能把一次针尖变化与它对上。

## 结果的键叫 spectra,不叫 points

``runtime._marker_subrecords`` 只认 ``regions`` / ``points`` 两个键,并给列表里
**每一项**画一个地图标记。这个流程的 N 条谱在**同一个 xy** 上,叫 ``points`` 会在
一个位置上叠 N 个标记(设计陷阱 6)。叫 ``spectra`` 就退回按技能名的单标记路径 ——
那是对的:针尖只去过一个地方。

⚠️ 逐条调用的**子技能**(``SpectroscopyAtPositions``)各自仍会经由
``ExecutionContext.marker_sink`` 留下自己的记录。那是既有的逐技能记录行为,与本条
无关;这里管的是**本流程自己的产物**别再叠一层。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Iterator

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.sts_workflow import crosses_zero, resolve_series
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
    abort_facts,
)

logger = logging.getLogger(__name__)

#: 逐点取谱引擎的技能名。**按名字引用,不 import 它的类**(见模块注释)。
POINT_ENGINE_SKILL = "SpectroscopyAtPositions"

#: 一次序列最多几条。每条是一次完整的条件建立 + 一条谱,分钟量级 —— 24 条已经是
#: 一两个小时。上限是**预算**不是目标。
MAX_CONDITIONS = 24

#: 每一条的状态(闭集)。``refused`` 与 ``failed`` 分开:前者是这条**根本没发出去**
#: (组名坏了/未标定/覆写越界),后者是发出去了没成 —— 两者的下一步完全不同。
STATUS_OK = "ok"
STATUS_REFUSED = "refused"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
ALL_STATUS: tuple[str, ...] = (STATUS_OK, STATUS_REFUSED, STATUS_FAILED,
                               STATUS_SKIPPED)


def parse_conditions(raw: Any) -> "tuple[list[str], str]":
    """``"a,b,c"`` 或 ``'["a","b"]'`` → 组名列表。``(names, 错误原文)``。

    两种写法都吃:conduct spec 里写 JSON 数组自然,人手打字写逗号分隔自然。
    **空的不当成「用默认」** —— 那会让一次打错的输入变成一条谁也没要的谱。
    """
    if raw is None:
        return [], "没有给条件组名。"
    if isinstance(raw, (list, tuple)):
        names = [str(x).strip() for x in raw]
    else:
        text = str(raw).strip()
        if not text:
            return [], "条件组名是空的 —— 空不当成「用默认」,那会让一次打错的输入变成一条谁也没要的谱。"
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as exc:
                return [], f"条件组名看着像 JSON 数组但解不开:{exc}"
            if not isinstance(loaded, list):
                return [], "条件组名的 JSON 不是一个数组。"
            names = [str(x).strip() for x in loaded]
        else:
            names = [p.strip() for p in text.split(",")]
    names = [n for n in names if n]
    if not names:
        return [], "条件组名列表是空的。"
    if len(names) > MAX_CONDITIONS:
        return [], (f"一次要跑 {len(names)} 条,超过上限 {MAX_CONDITIONS}。"
                    f"每条是一次完整的条件建立加一条谱 —— 这个上限是预算,不是目标。")
    return names, ""


class STSConditionSeries(CompositeSkillGraph):
    """一个位置,一串条件组,逐条取谱。"""

    _records: list[dict]
    _refusal: str

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="STSConditionSeries",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在**同一个**位置上取**若干条**谱，每一条各有自己的稳定条件 —— "
                "走的是一张条件**组名**表。一条曲线表达不了你要的东西时用它："
                "逐条不同的稳定 bias/setpoint（一个宽窗口按它的中心稳下来，"
                "两端会把前置放大器打饱和），或者逐条不同的调制幅度。"
                "**不要**（NOT）拿它去做「一条曲线里的若干个偏压**段**」—— "
                "Nanonis MLS 早就能做（E_F 附近密采、远处稀采、"
                "弱信号段加长积分、只在要紧的窗口开调制），"
                "而一个条件组可以把那些段原样透传下去。数值全都住在条件表里"
                "（core.sts_workflow）；本技能只收**组名**，绝不收数值。"
                "两条谱之间，稳定条件经 BiasSettleChange 重建 —— 因为相邻两条条件"
                "完全可能落在零的两侧，而恒流反馈下偏压穿零会把针尖推进样品。"
            ),
            parameters=[
                ParameterSpec(
                    name="conditions", type="str", required=True,
                    description=(
                        "要逐条走的条件**组名**，按顺序 —— 写成 "
                        "'a,b,c' 或者一个 JSON 数组。**不是数值**：每一个值"
                        "都住在组里（core.sts_workflow）。组名不认识、"
                        "覆写越界、或者这个组未标定，都只拒绝**那一条**，"
                        "并指名是哪里不对；序列其余部分照跑。")),
                ParameterSpec(
                    name="x_m", type="float", unit="m", required=True,
                    min_value=-1e-4, max_value=1e-4,
                    description=(
                        "那**一个**位置的 X，单位是**米**（SI），不是纳米："
                        "20 nm -> 20n。序列里每一条谱都在这里取 —— "
                        "这正是这条序列的意义所在。")),
                ParameterSpec(
                    name="y_m", type="float", unit="m", required=True,
                    min_value=-1e-4, max_value=1e-4,
                    description="那一个位置的 Y，单位是**米**（SI）。"),
                ParameterSpec(
                    name="expected_coord_epoch", type="int", required=True,
                    min_value=0,
                    description=(
                        "这个位置是在哪一代坐标下算出来的。"
                        "对不上**或者**当前代次读不到，都要拒绝；那时候这个"
                        "坐标指向的是另一片表面。由逐点引擎**逐条谱**查一次 —— "
                        "一条序列要跑一个小时，而一次粗动完全可能就落在中间。")),
                ParameterSpec(
                    name="settle_s", type="float", unit="s", required=False,
                    min_value=0.0, max_value=60.0,
                    description=(
                        "**只**覆写整定时间这一项，对序列里每一条条件都生效。"
                        "不填则各条用自己那个组里的值。")),
                ParameterSpec(
                    name="spectral_family", type="str", required=False,
                    default="", allowed_values=["", "metallic", "gapped",
                                                "unknown"],
                    description=(
                        "哪些谱的子判据有资格当闸门。留空 = unknown，"
                        "而 unknown **不是** metallic。")),
                ParameterSpec(
                    name="assess", type="bool", required=False, default=True,
                    description=(
                        "对每一条谱跑一次谱质量闸门。它**只记账** —— "
                        "一个差评绝不会打断序列，除非 "
                        "stop_after_consecutive_discard 说要打断。")),
                ParameterSpec(
                    name="stop_after_consecutive_discard", type="int",
                    required=False, default=2, min_value=0, max_value=24,
                    description=(
                        "连续这么多条判为 'discard' 就停下。"
                        "0 = 永不停。'unrated' **不计**在内 —— 「判不了」"
                        "不是一条差谱。这里不做任何补救；"
                        "序列停下来，并把这件事说出来。")),
                ParameterSpec(
                    name="per_point_timeout_s", type="float", unit="s",
                    required=False, default=600.0,
                    min_value=1.0, max_value=7200.0,
                    description=(
                        "**一条**谱的预算，原样交给逐点引擎。")),
                ParameterSpec(
                    name="run_tag", type="str", required=False, default="",
                    description=(
                        "逐条谱的 .dat 文件名前缀。留空 = 用一个时间戳，"
                        "这样重跑一次的文件仍然分得开。")),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=1800.0,
            composition_level=4,
            tags=["spectroscopy", "sts", "series", "conditions", "composite"],
        )

    # ── 计划 ──────────────────────────────────────────────────────────────

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        names: list[str] = list(executor.progress.partial_data.get("names") or ())
        if not names:
            return
        overrides = ({"settle_s": params["settle_s"]}
                     if params.get("settle_s") is not None else None)
        resolutions = resolve_series(names, [overrides] * len(names)
                                     if overrides else None)
        executor.set_partial(
            "resolved", [{"name": r.name, "usable": r.usable,
                          "problem": r.refusal_detail()} for r in resolutions])

        x, y = float(params["x_m"]), float(params["y_m"])
        positions = json.dumps([{"x_m": x, "y_m": y, "label": "S"}])
        stop_after = int(params.get("stop_after_consecutive_discard", 2) or 0)
        run_tag = str(params.get("run_tag") or "").strip()

        streak = 0
        stopped = ""
        prev_spec = None
        for i, res in enumerate(resolutions):
            rec: dict[str, Any] = {
                "index": i + 1,
                "condition": res.name,
                "status": STATUS_SKIPPED,
                "success": False,
                "verdict": None,
                "dat_path": None,
                "error": None,
                "condition_values": (asdict(res.spec) if res.spec is not None
                                     else {}),
                "condition_summary": (res.spec.describe() if res.spec is not None
                                      else ""),
                # 穿零那一步要**指出来**:恒流反馈下偏压穿零会把针尖推进样品,
                # 而它是宽偏压序列的必经之路。事后要能把一次针尖变化与它对上。
                "crosses_zero": bool(prev_spec is not None and res.spec is not None
                                     and crosses_zero(prev_spec, res.spec)),
            }
            self._records.append(rec)
            if stopped:
                rec["error"] = f"早停后未执行：{stopped}"
                continue
            if not res.usable:
                # **这一条不发出去,其余照跑。** 一个坏组名不该废掉整条序列 ——
                # 而把它记成「跑了但失败」会让人去查仪器，真正要查的是那张表。
                rec["status"] = STATUS_REFUSED
                rec["error"] = res.refusal_detail()
                continue

            prev_spec = res.spec
            step_id = f"c{i:02d}:{res.name}"
            sub = {
                "positions": positions,
                "expected_coord_epoch": int(params["expected_coord_epoch"]),
                "condition": res.name,
                "assess": bool(params.get("assess", True)),
                "spectral_family": str(params.get("spectral_family") or ""),
                "per_point_timeout_s": float(
                    params.get("per_point_timeout_s", 600.0) or 600.0),
                # 逐条一个 tag:同一个位置连着取 N 条,tag 相同会让第 2 条的归属
                # 校验拿第 1 条的文件当自己的(文件名那一层认不出来)。
                "run_tag": f"{run_tag}_{i:02d}" if run_tag else f"c{i:02d}",
            }
            if params.get("settle_s") is not None:
                sub["settle_s"] = float(params["settle_s"])
            yield CompositeStep(
                step_id=step_id, skill_name=POINT_ENGINE_SKILL, params=sub,
                optional=True, checkpoint_after=True,
                tags=("series", f"condition={res.name}"))

            out = executor.sub_results.get(step_id)
            data = dict(getattr(out, "data", None) or {})
            pts = list(data.get("points") or ())
            first = pts[0] if pts else {}
            ok = bool(getattr(out, "success", False)) and bool(first.get("success"))
            rec["status"] = STATUS_OK if ok else STATUS_FAILED
            rec["success"] = ok
            rec["verdict"] = first.get("verdict")
            rec["dat_path"] = first.get("path")
            rec["error"] = (None if ok else
                            (getattr(out, "error", None) or first.get("error")
                             or "取谱引擎没有返回可用的结果"))
            rec["lockin_readback"] = data.get("lockin_readback") or {}

            if stop_after and rec["verdict"] == "discard":
                streak += 1
                if streak >= stop_after:
                    stopped = (f"连续 {streak} 条判为 discard —— 序列停在这里。"
                               f"没有尝试任何补救:接着跑只会得到更多同样的谱。")
            elif rec["verdict"] != "discard":
                # 「判不了」不打断连击,也不重置它:unrated 对「针尖在变坏吗」
                # 这个问题一个字都没说。
                if rec["verdict"] is not None and rec["verdict"] != "unrated":
                    streak = 0

        if stopped:
            executor.set_partial("stopped_early", True)
            executor.set_partial("stopped_reason", stopped)

    # ── 汇总 ──────────────────────────────────────────────────────────────

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        recs = list(self._records)

        def _n(pred) -> int:
            return sum(1 for r in recs if pred(r))

        return {
            # **键名是 spectra,不是 points**(设计陷阱 6):这 N 条谱在同一个 xy 上,
            # 叫 points 会让记录层在一个位置上叠 N 个标记。针尖只去过一个地方。
            "spectra": recs,
            "x_m": pd.get("x_m"),
            "y_m": pd.get("y_m"),
            "n_planned": len(recs),
            "n_acquired": _n(lambda r: r["status"] == STATUS_OK),
            "n_refused": _n(lambda r: r["status"] == STATUS_REFUSED),
            "n_failed": _n(lambda r: r["status"] == STATUS_FAILED),
            "n_skipped": _n(lambda r: r["status"] == STATUS_SKIPPED),
            "n_keep": _n(lambda r: r["verdict"] == "keep"),
            "n_flagged": _n(lambda r: r["verdict"] == "keep_flagged"),
            "n_discard": _n(lambda r: r["verdict"] == "discard"),
            "n_unrated": _n(lambda r: r["verdict"] == "unrated"),
            "n_zero_crossings": _n(lambda r: r["crosses_zero"]),
            # 仅传递本次实际采集成功的谱路径，防止复用上次数据。
            "dat_paths": [r["dat_path"] for r in recs
                          if r["status"] == STATUS_OK and r["dat_path"]],
            "conditions_resolved": pd.get("resolved") or [],
            "stopped_early": bool(pd.get("stopped_early", False)),
            "stopped_reason": str(pd.get("stopped_reason", "") or ""),
            "coord_epoch": pd.get("coord_epoch"),
        }

    def _validate_products(self, data: dict) -> tuple[bool, str]:
        """产物是一批 .dat 与逐条判决,不是一张图 —— 跳过通用产物闸。

        每条谱的质量已经由 ``AssessSpectrum`` 在引擎里判过一次;让通用闸再判一遍
        等于对同一批数据给出第二个来源不同的结论,然后用它盖掉第一个。
        """
        return True, ""

    def _decide_outcome(self, all_good: bool, progress: CompositeProgress,
                        data: dict) -> tuple[bool, str]:
        """一条都没采到 ⇒ 失败;采到一部分 ⇒ 成功,但缺口必须说出来。

        「部分成功」在这条流程上是**正常产物**:条件表里有一条未标定、有一条把前置
        放大器打饱和,都不该让另外几条白跑。但它绝不能长得像全成功 —— 缺口进
        summary,不只是躺在 data 里(本仓「假成功」那一族)。
        """
        n_ok = int(data.get("n_acquired") or 0)
        n = int(data.get("n_planned") or 0)
        if n and n_ok == 0:
            bits = [r["error"] for r in data.get("spectra") or () if r.get("error")]
            return False, (f"{n} 条条件一条都没采成（0/{n}）。"
                           + ("；".join(str(b)[:120] for b in bits[:3]) if bits else ""))
        return super()._decide_outcome(all_good, progress, data)

    # ── 执行 ──────────────────────────────────────────────────────────────

    def run_composite(self, context, params: dict) -> SkillResult:
        self._records = []
        self._refusal = ""
        names, err = parse_conditions(params.get("conditions"))
        if err:
            return self.fail(f"条件序列用不了：{err}")

        executor = GraphExecutor(
            composite_name=self._skill_name(), context=context,
            on_step_result=self.on_step_result, on_step_failed=self.on_step_failed)
        executor.set_partial("names", names)
        executor.set_partial("x_m", float(params["x_m"]))
        executor.set_partial("y_m", float(params["y_m"]))
        executor.set_partial("coord_epoch", int(params["expected_coord_epoch"]))
        executor.set_partial_default("stopped_early", False)
        executor.set_partial_default("stopped_reason", "")
        self._executor = executor

        executor.run_plan(self.plan_dynamic(params, executor))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()
        data.update(abort_facts(executor.progress))

        ok, err = self._decide_outcome(True, executor.progress, data)
        if not ok:
            return SkillResult(skill_name=self._skill_name(), success=False,
                               error=err, data=data)
        return SkillResult(skill_name=self._skill_name(), success=True, data=data,
                           summary=self._summary(data))

    @staticmethod
    def _summary(data: dict) -> str:
        n, ok = data.get("n_planned") or 0, data.get("n_acquired") or 0
        lines = [f"条件序列：{n} 条里采到 {ok} 条"
                 f"（{data.get('x_m', 0.0) * 1e9:.1f}, "
                 f"{data.get('y_m', 0.0) * 1e9:.1f}) nm 同一个位置）。"]
        if ok < n:
            gaps = []
            for key, word in (("n_refused", "条件用不了"), ("n_failed", "采集失败"),
                              ("n_skipped", "早停未执行")):
                if data.get(key):
                    gaps.append(f"{word} {data[key]}")
            if gaps:
                lines.append("缺口：" + "、".join(gaps))
        verdicts = [(k, data.get(f"n_{k}")) for k in
                    ("keep", "flagged", "discard", "unrated")]
        shown = [f"{k} ×{v}" for k, v in verdicts if v]
        if shown:
            lines.append("判决：" + "、".join(shown))
        if data.get("n_zero_crossings"):
            lines.append(
                f"其中 {data['n_zero_crossings']} 次稳定偏压穿零 —— 走的是 "
                f"BiasSettleChange（恒流反馈下穿零会把针尖推进样品）。"
                f"事后若发现针尖变了，先看这几步。")
        for r in (data.get("spectra") or ()):
            if r.get("status") == STATUS_REFUSED:
                lines.append(f"⚠️ 第 {r['index']} 条（{r['condition']}）没发出去："
                             f"{str(r.get('error'))[:160]}")
        if data.get("stopped_early"):
            lines.append(f"早停：{data.get('stopped_reason')}")
        return "\n".join(lines)


def make_tool(context_provider):
    return wrap_skill(STSConditionSeries, context_provider)


__all__ = [
    "ALL_STATUS",
    "MAX_CONDITIONS",
    "POINT_ENGINE_SKILL",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_REFUSED",
    "STATUS_SKIPPED",
    "STSConditionSeries",
    "parse_conditions",
]
