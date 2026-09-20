"""AtomicBiasSeries —— 同一片原子、逐张变偏压，每一张都落账。

设计文档:``docs/v2/design/`` 的「S2 偏压序列」设计,D4/D6/§3.2
(文件名按「S2 偏压序列」检索;通用层注释不写样品名前缀 ——
``stm_capability_vs_sample_layer.md`` 拍板④)。

账的格式与派生逻辑住在 :mod:`mast.core.s2_bias_ledger`(纯数据层),本模块只负责
**跑**和**把每一次尝试写进去**。

## 这个流程要回答的问题

「在哪些偏压上，这片原子有分辨？」——注意是**这片**。换点等于换了被测对象，
所以整条序列在**同一个中心**上跑，而「确实是同一片」这件事由 anchor 帧事后证明:
序列末尾(以及可选的每 N 个偏压)回到起始偏压再扫一张，两张对得上，这段序列的
同一性才是有证据的，否则它只是一个断言。

## 三条不做的事

* **不做在线漂移闭环。** 接受漂移 + 逐帧记录标称中心与 .sxm 头里的真实中心 +
  事后对齐。仓里的 `TrackDrift_ReferenceScan` 用 `Scan_FrameDataGrab` 直读缓冲，
  而缓冲与保存帧的同一性无法保证，因此不把 S2 建在
  它上面。在线补偿的另一条路是每偏压多扫一张参考图，成本翻倍。
* **不靠排序规避高偏压风险。** `order_series_monotonic` 单调走一遍，高 |V| 必然
  落在序列两端之一，排序救不了这件事。靠的是逐帧落账 + 帧间守卫 + 一个**用户
  填的** `elevated_bias_abs_v` 槽位:超过它的偏压扫完**在帧边界**插一次针尖复核。
* **不在运行中打断做质量判定。** 每帧的 verdict 不是闸门，只进账；闸门是 L1 的
  `S2_gate`，序列跑完或预算耗尽时消费一次。

## 高偏压槽位没声明时会怎样

`elevated_bias_abs_v` 是用户填的，代码不发明。**没声明时不拒绝整条序列** ——
那会造出一个谁也解不开的死闸(一个没人填过的旋钮挡住整个 S2)。做法是:槽位不
生效，并且在 summary 与每一条账里**说出来**。「读不到」有自己的分支并且出声，
不冒充「已经检查过了」。
"""
from __future__ import annotations

import math
from typing import Any, Iterator

from mast.core import s2_bias_ledger as ledger
from mast.core.scan_planner import expand_series, order_series_monotonic
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

#: 预算默认值。来自设计 §D4(与 ``MAX_RESCANS_PER_FRAME = 1`` 同量级)。
DEFAULT_MAX_ATTEMPTS_PER_BIAS = 2
DEFAULT_MAX_UNDECIDABLE_RETRIES = 2

#: 这个流程只在原子档上跑。档位表的单一真源在 ``core.scan_policy``。
TIER_NAME = "atomic_verify"

#: 高偏压槽位没声明时写进账里的那句话。**是一个词，不是一个空值** ——
#: 空值会被读成「检查过了，没超」。
SLOT_UNDECLARED = "elevated_bias_abs_v_undeclared"


class AtomicBiasSeries(CompositeSkillGraph):
    """同一位置逐偏压成像 + 逐偏压落账 + anchor 复扫。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AtomicBiasSeries",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            # CONFIRM,与同族的 CrossPointTipCheck / SpectroscopyAtPositions 一致:
            # 它反复改偏压并在同一点上扫,是真动硬件的流程。
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在**同一个**位置上逐个偏压成像，并按偏压逐条记录该偏压下有没有"
                "原子分辨 —— 写进一本**只追加**的账，账的键是**偏压**，不是帧序号"
                "（planner 会重排整条序列，所以 frame 3 不是 bias 3）。"
                "每个偏压各自拿到一个闭集结局：resolved / "
                "absent_confirmed / undecided / blocked / tip_aborted。"
                "'undecided' 与 'absent_confirmed' 绝不（NEVER）合并 —— 前者的意思是"
                "证据不够好，后者是一个真正的科学结论（「在这个偏压上确实没有原子"
                "对比度」），把前者折进后者是一句谎话，而这句谎话的结局是一根好针尖"
                "被扰动。anchor 帧（回到起始偏压再扫一张）才是让「这一整段扫的是"
                "同一片原子」事后可被证明的那个东西。"
                "漂移只记录，绝不在线补偿。"
            ),
            parameters=[
                ParameterSpec(
                    name="bias_values", type="str", required=False,
                    description=(
                        "偏压序列，单位伏特，逗号分隔（例如 "
                        "'-0.4,-0.2,0.2,0.4'）。这个与 "
                        "bias_start/bias_stop/bias_n 三件套二选一。"),
                    default=""),
                ParameterSpec(
                    name="bias_start", type="float", unit="V", required=False,
                    description="序列起点（配合 bias_stop + bias_n 一起用）。",
                    min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="bias_stop", type="float", unit="V", required=False,
                    description="序列终点。", min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="bias_n", type="int", required=False,
                    description="这条序列里有几个偏压。",
                    min_value=1, max_value=64),
                ParameterSpec(
                    name="center_x_m", type="float", unit="m", required=True,
                    description="要成像的位置 —— 每一个偏压都用**同一个**位置。"),
                ParameterSpec(
                    name="center_y_m", type="float", unit="m", required=True,
                    description="要成像的位置。"),
                ParameterSpec(
                    name="size_m", type="float", unit="m", required=False,
                    default=5e-9, min_value=1e-10, max_value=1e-7,
                    description=(
                        "视野大小。原子尺度闸门卡的是 nm/px，所以它和 'pixels' "
                        "合起来决定这一帧到底撑不撑得起一个判定 —— 在扫任何一帧"
                        "**之前**就先查。")),
                ParameterSpec(
                    name="pixels", type="int", required=False,
                    min_value=64, max_value=2048,
                    description=(
                        "每行像素数。留空 = 用 atomic_verify 这一档的值。"
                        "WARNING：只把 pixels 提上去而不把 "
                        "line_time_s 按同样的倍数一起提，买到的是一个更好看的 nm/px，"
                        "而每个采样点携带的信息**更少** —— 那是在骗过尺度闸门，"
                        "不是通过它。")),
                ParameterSpec(
                    name="line_time_s", type="float", unit="s", required=False,
                    min_value=0.01, max_value=60.0,
                    description="留空 = 用 atomic_verify 这一档的值。"),
                ParameterSpec(
                    name="ledger_dir", type="str", required=True,
                    description=(
                        "只追加的那本账放在哪个目录下"
                        "（<dir>/.mast/s2_bias_ledger.jsonl）。账里存的是帧的"
                        "**绝对路径**，所以它不依赖 scan registry"
                        "（后者扫过 50 张之后就开始淘汰）。"),
                    default=""),
                ParameterSpec(
                    name="elevated_bias_abs_v", type="float", unit="V",
                    required=False, min_value=0.0, max_value=10.0,
                    description=(
                        "|V| 高过这个值，扫完那一帧就强制插一次针尖复核。"
                        "**由用户填**（OPERATOR-SUPPLIED）—— 这里不发明"
                        "默认值。留空则这个槽位**不生效**（INACTIVE），而且每一条账"
                        "都会把这件事说出来；它绝不会被悄悄当成"
                        "「查过了，没问题」。")),
                ParameterSpec(
                    name="anchor_every_n", type="int", required=False,
                    default=0, min_value=0, max_value=64,
                    description=(
                        "每 N 个偏压插一张 anchor 复扫（回到起始偏压再扫一张）。"
                        "0 = 只在收尾扫那一张。收尾那张 anchor **永远都会扫** —— "
                        "一张都没有的话，「这一段扫的是同一片原子」无从证明。")),
                ParameterSpec(
                    name="max_attempts_per_bias", type="int", required=False,
                    default=DEFAULT_MAX_ATTEMPTS_PER_BIAS,
                    min_value=1, max_value=5,
                    description="每个偏压在放弃之前最多扫几次。"),
                ParameterSpec(
                    name="max_undecidable_retries_per_bias", type="int",
                    required=False, default=DEFAULT_MAX_UNDECIDABLE_RETRIES,
                    min_value=0, max_value=5,
                    description=(
                        "花在 'undecidable' 裁决上的重试次数，与失败**分开计**"
                        "（SEPARATELY）—— 一张没能得出结论的帧不是一张失败的帧。")),
                ParameterSpec(
                    name="expected_coord_epoch", type="int", required=True,
                    description=(
                        "这些帧中心是在哪一代坐标下规划的。**必填**，对不上要拒绝，"
                        "**而且**当前代次读不到时同样拒绝 —— 一次粗动会让同一组 "
                        "(x, y) 变成另一片表面，而这一步接着会对着它连扫几个小时。"
                        "以前它可以省略，于是那道守卫悄无声息地形同虚设。")),
                ParameterSpec(
                    name="evidence_epoch", type="int", required=False,
                    default=0,
                    description="盖在每一条账上的证据代次。"),
                ParameterSpec(
                    name="conduct_id", type="str", required=False, default="",
                    description="归属的 conduct，每一行都记。"),
                ParameterSpec(
                    name="stage_id", type="str", required=False, default="",
                    description="归属的 stage，每一行都记。"),
                ParameterSpec(
                    name="save_each", type="bool", required=False, default=True,
                    description="每扫完一帧就 SaveScan（判据从头到尾"
                                "只读**已保存**的 .sxm）。"),
            ],
            estimated_duration_s=600.0,
            composition_level=4,
            tags=["atomic", "bias", "series", "ledger", "scan"],
        )

    # ── 输入 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_biases(params: dict) -> "tuple[list[float], str]":
        """展开偏压序列。**展不开就拒绝,绝不发明一个序列。**"""
        raw = str(params.get("bias_values") or "").strip()
        values: "list[float] | None" = None
        if raw:
            try:
                values = [float(tok) for tok in raw.replace(";", ",").split(",")
                          if tok.strip()]
            except ValueError:
                return [], f"bias_values 里有解析不了的数:{raw!r}"
        else:
            spec = {k: params.get(f"bias_{k}") for k in ("start", "stop", "n")}
            if any(v is not None for v in spec.values()):
                values = expand_series(spec)
                if values is None:
                    return [], (f"start/stop/n 展不成一个序列:{spec!r} —— "
                                f"缺一个就展不开,而这里不替你补。")
        if not values:
            return [], ("没有给偏压序列。给 bias_values（逗号分隔）"
                        "或 bias_start/bias_stop/bias_n 三个一起。")
        if any(v != v or math.isinf(v) for v in values):
            return [], "偏压序列里有 NaN/inf。"
        # 去重按**规范化后的键**,不按浮点相等 —— 后者会让 -0.3 与
        # -0.30000000000000004 变成两个偏压。
        seen: dict[str, float] = {}
        for v in values:
            seen.setdefault(ledger.bias_key(v), v)
        return list(seen.values()), ""

    def _scale_precheck(self, size_m: float,
                        pixels: "int | None") -> "tuple[str, str]":
        """下发之前先问「这组帧参数判不判得出原子相」。

        扫完一张判不了的图再说,白花一帧的时间 —— 而且流程会把「判不了」误读成
        「还没弄出原子相」,接着去扰动一根本来好好的针尖。
        """
        from mast.vision.atomic_phase import plan_scale

        if not pixels:
            return "", ""            # 用档位表的值,由 resolver 定,这里不预判
        nm_per_px, scale, problem = plan_scale(size_m, pixels)
        return (str(scale or ""), str(problem or ""))

    # ── 账 ────────────────────────────────────────────────────────────────

    def _ledger_file(self, params: dict):
        return ledger.ledger_path(str(params.get("ledger_dir") or "."))

    # ── 读一步的结果 ──────────────────────────────────────────────────────
    #
    # ⚠️ ``GraphExecutor`` **只把成功的步骤放进 sub_results**(:550);失败的进
    # ``progress.failed_steps`` / ``failed_reasons``。所以 ``sub_results.get(id)
    # is None`` 的意思是「这一步没有成功」,而**不是**「这一步没问题」。
    # 第一版这里写的是 ``if r is not None and not r.success``,于是被安全门拒掉的
    # 偏压和没过的针尖复核**全都悄悄地被当成通过**了 —— 测试当场抓到。
    @staticmethod
    def _ok(executor: GraphExecutor, step_id: str) -> bool:
        return executor.sub_results.get(step_id) is not None

    @staticmethod
    def _why_failed(executor: GraphExecutor, step_id: str) -> str:
        return str(executor.progress.failed_reasons.get(step_id) or "")

    def _write(self, params: dict, executor: GraphExecutor, **fields) -> dict:
        """造一条记录、落盘、并推进 partial_data 里的行数(给续跑看)。"""
        rec = ledger.make_record(
            conduct_id=str(params.get("conduct_id") or ""),
            stage_id=str(params.get("stage_id") or ""),
            evidence_epoch=_int(params.get("evidence_epoch")),
            coord_epoch=_int(params.get("expected_coord_epoch")),
            tier_name=TIER_NAME,
            **fields)
        try:
            ledger.append(self._ledger_file(params), rec)
        except OSError as exc:
            # 账写不进去是一件必须让人知道的事:后面所有的结论都建在它上面。
            executor.set_partial(
                "ledger_write_error", f"{type(exc).__name__}: {exc}")
        rows = list(executor.progress.partial_data.get("rows") or [])
        rows.append(rec)
        executor.set_partial("rows", rows)
        return rec

    # ── 计划 ──────────────────────────────────────────────────────────────

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        biases: list[float] = list(
            executor.progress.partial_data.get("biases") or [])
        cx = float(params["center_x_m"])
        cy = float(params["center_y_m"])
        size_m = float(params.get("size_m") or 5e-9)
        pixels = _int(params.get("pixels"))
        line_time_s = params.get("line_time_s")
        save_each = bool(params.get("save_each", True))
        max_attempts = int(params.get("max_attempts_per_bias")
                           or DEFAULT_MAX_ATTEMPTS_PER_BIAS)
        max_undec = int(params.get("max_undecidable_retries_per_bias")
                        if params.get("max_undecidable_retries_per_bias")
                        is not None else DEFAULT_MAX_UNDECIDABLE_RETRIES)
        anchor_every = int(params.get("anchor_every_n") or 0)
        elevated = _float_or_none(params.get("elevated_bias_abs_v"))

        # 单调走一遍:偏压来回跳会反复激起针尖-样品结的回滞与充放电瞬态。
        order = order_series_monotonic(biases, current=None)
        ordered = [biases[i] for i in order]
        executor.set_partial("bias_order", [ledger.bias_key(b) for b in ordered])
        anchor_bias = ordered[0] if ordered else None

        aborted_tip = False
        for pos, bias_v in enumerate(ordered):
            if executor.progress.aborted:
                break
            outcome = yield from self._one_bias(
                params, executor, bias_v=bias_v, idx=pos,
                role=ledger.ROLE_SERIES, cx=cx, cy=cy, size_m=size_m,
                pixels=pixels, line_time_s=line_time_s, save_each=save_each,
                max_attempts=max_attempts, max_undec=max_undec,
                elevated=elevated)
            if outcome == ledger.OUTCOME_TIP_ABORT:
                # 针尖坏了之后每一帧都是废的。账停在这里,前面的全在。
                aborted_tip = True
                executor.set_partial("stopped_reason", "扫描中检测到针尖事件")
                break
            if (anchor_every and anchor_bias is not None
                    and (pos + 1) % anchor_every == 0 and pos + 1 < len(ordered)):
                yield from self._one_bias(
                    params, executor, bias_v=anchor_bias, idx=1000 + pos,
                    role=ledger.ROLE_ANCHOR, cx=cx, cy=cy, size_m=size_m,
                    pixels=pixels, line_time_s=line_time_s,
                    save_each=save_each, max_attempts=1, max_undec=0,
                    elevated=elevated)

        # 收尾锚点:**无条件**。没有它,「整段扫的是同一片原子」无从证明。
        # 中止时不补 —— 那一刻针尖状态不明,再扫一张既不安全也不构成证据。
        if anchor_bias is not None and not aborted_tip and not executor.progress.aborted:
            yield from self._one_bias(
                params, executor, bias_v=anchor_bias, idx=9999,
                role=ledger.ROLE_ANCHOR, cx=cx, cy=cy, size_m=size_m,
                pixels=pixels, line_time_s=line_time_s, save_each=save_each,
                max_attempts=1, max_undec=0, elevated=elevated)
        executor.set_partial("aborted_tip", aborted_tip)

    def _one_bias(self, params: dict, executor: GraphExecutor, *,
                  bias_v: float, idx: int, role: str, cx: float, cy: float,
                  size_m: float, pixels: "int | None", line_time_s: Any,
                  save_each: bool, max_attempts: int, max_undec: int,
                  elevated: "float | None") -> "Iterator[CompositeStep]":
        """一个偏压的完整尝试循环。返回最后一次尝试的 outcome(或 None)。

        失败预算与 undecidable 预算**分开计**:一张判不了的帧不是一次失败,
        把两者合并会让「证据不足」提前耗光重试机会。
        """
        tag = f"b{idx}"
        attempt = 0
        undec_used = 0
        last_outcome: "str | None" = None

        while attempt < max_attempts:
            attempt += 1
            sid = f"{tag}_a{attempt}"

            # ── 1) 偏压 ──
            yield CompositeStep(
                step_id=f"{sid}_bias", skill_name="BiasSettleChange",
                params={"bias_v": float(bias_v)},
                optional=True, checkpoint_after=False, tags=("bias",))
            if not self._ok(executor, f"{sid}_bias"):
                # 安全门/包络拒了 —— 这是**拒绝**,不是「这里没有原子」。
                self._write(params, executor, bias_v=bias_v, attempt=attempt,
                            role=role, outcome=ledger.OUTCOME_BLOCKED,
                            note=(f"偏压变更被拒:"
                                  f"{self._why_failed(executor, f'{sid}_bias')}"))
                return ledger.OUTCOME_BLOCKED

            # ── 2) 扫 ──
            scan_params: dict[str, Any] = {
                "center_x_m": cx, "center_y_m": cy, "size_m": size_m,
                "purpose": TIER_NAME,
            }
            if pixels:
                scan_params["pixels"] = int(pixels)
            if line_time_s is not None:
                scan_params["line_time_s"] = float(line_time_s)
            yield CompositeStep(
                step_id=f"{sid}_scan", skill_name="ScanAt", params=scan_params,
                optional=True, checkpoint_after=True, tags=("scan",))
            r_scan = executor.sub_results.get(f"{sid}_scan")
            sdata = dict(getattr(r_scan, "data", None) or {})
            tip_event = bool(sdata.get("tip_change_critical")
                             or sdata.get("tip_event"))
            crash = bool(sdata.get("crash_indicator")
                         or sdata.get("crash_check") == "crash")
            resolver_warnings = list(sdata.get("warnings") or ())

            if tip_event or crash:
                self._write(params, executor, bias_v=bias_v, attempt=attempt,
                            role=role, outcome=ledger.OUTCOME_TIP_ABORT,
                            frame_path=sdata.get("saved_path"),
                            nominal_center_m=(cx, cy), size_m=size_m,
                            pixels=pixels, line_time_s=_float_or_none(line_time_s),
                            resolver_warnings=resolver_warnings,
                            note="撞针" if crash else "针尖事件")
                return ledger.OUTCOME_TIP_ABORT

            if not self._ok(executor, f"{sid}_scan"):
                # 扫描整个没成 ⇒ **没有帧结果可读**,所以上面那对 tip_event/crash
                # 读到的是空字典。这一次尝试里针尖状态是**读不到**,不是「没事」——
                # 记进 note,别让它长得像一次「查过了没问题」的重试。
                last_outcome = None
                self._write(params, executor, bias_v=bias_v, attempt=attempt,
                            role=role,
                            verdict=ledger.VERDICT_UNDECIDABLE,
                            remedy="rescan_frame",
                            nominal_center_m=(cx, cy), size_m=size_m,
                            pixels=pixels,
                            resolver_warnings=resolver_warnings,
                            note=(f"扫描没成(本次针尖状态读不到):"
                                  f"{self._why_failed(executor, f'{sid}_scan')}"))
                continue

            if save_each:
                yield CompositeStep(
                    step_id=f"{sid}_save", skill_name="SaveScan", params={},
                    optional=True, checkpoint_after=False, tags=("save",))

            path = str(sdata.get("saved_path") or "")
            if not path:
                # 判据只吃已保存的 .sxm。拿不到路径就是判不了,**不是没有原子**。
                self._write(params, executor, bias_v=bias_v, attempt=attempt,
                            role=role, verdict=ledger.VERDICT_UNDECIDABLE,
                            remedy="rescan_frame",
                            nominal_center_m=(cx, cy), size_m=size_m,
                            pixels=pixels,
                            resolver_warnings=resolver_warnings,
                            note="扫完了但拿不到已保存的 .sxm 路径 —— 判据不吃缓冲")
                continue

            # ── 3) 裁决 ──
            yield CompositeStep(
                step_id=f"{sid}_verify", skill_name="VerifyAtomicResolution",
                params={"scan_path": path},
                optional=True, checkpoint_after=True, tags=("verdict", "read"))
            r_ver = executor.sub_results.get(f"{sid}_verify")
            vdata = dict(getattr(r_ver, "data", None) or {})
            verdict = str(vdata.get("verdict") or "") or ledger.VERDICT_UNDECIDABLE
            admission = vdata.get("frame_admission") or {}

            rec = self._write(
                params, executor, bias_v=bias_v, attempt=attempt, role=role,
                frame_path=path, scan_id=sdata.get("scan_id"),
                nominal_center_m=(cx, cy),
                actual_center_m=_actual_center(sdata),
                size_m=sdata.get("size_m", size_m),
                pixels=sdata.get("pixels", pixels),
                line_time_s=_float_or_none(sdata.get("line_time_s", line_time_s)),
                resolver_warnings=resolver_warnings,
                verdict=verdict, remedy=vdata.get("remedy"),
                metrics=_metrics(vdata),
                frame_admission_passed=(admission.get("passed")
                                        if isinstance(admission, dict) else None),
                profile_name=vdata.get("profile_name"),
                profile_provenance=vdata.get("profile_provenance"),
                note=("" if elevated is not None else SLOT_UNDECLARED))
            executor.set_partial("last_frame_path", path)
            last_outcome = None
            del rec
            # ``drift_shift_px`` / ``drift_flag`` 目前恒为 None:判据是「相邻两帧
            # 互相关平移量 > 帧宽的 25%」,而那个 25% **还没标定**(设计 §7-7)。
            # 与其现在按一个想当然的数去标 drift_excessive,不如把每帧的标称中心
            # 与真实中心都记着 —— 事后对齐是纯离线计算,标定好了随时能回算。
            # 空着的理由写在这里,免得下一个人以为漂移检测接好了。

            # ── 4) 高偏压 ⇒ 帧边界插一次针尖复核 ──
            if elevated is not None and abs(float(bias_v)) > float(elevated):
                yield CompositeStep(
                    step_id=f"{sid}_tipcheck", skill_name="PreScanCheck",
                    params={"center_x_m": cx, "center_y_m": cy},
                    optional=True, checkpoint_after=True, tags=("tip", "check"))
                if not self._ok(executor, f"{sid}_tipcheck"):
                    # 复核不过 ⇒ 走 S1 绕道,**不重扫该偏压**。重扫等于在刚炸
                    # 出来的坑上再判一次。
                    self._write(params, executor, bias_v=bias_v,
                                attempt=attempt + 1, role=role,
                                outcome=ledger.OUTCOME_TIP_ABORT,
                                note="高偏压后针尖复核未通过 —— 转 S1 绕道,不重扫本偏压")
                    return ledger.OUTCOME_TIP_ABORT

            if verdict == ledger.VERDICT_RESOLVED:
                return None
            if verdict == ledger.VERDICT_UNDECIDABLE:
                undec_used += 1
                if undec_used > max_undec:
                    return None
                continue
            # atomic_absent:判据说了话,不重试 —— 重试改变不了一个已经成立的裁决。
            return None

        return last_outcome

    # ── 汇总 ──────────────────────────────────────────────────────────────

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        params = dict(pd.get("params") or {})
        biases = list(pd.get("biases") or [])
        read = ledger.read_ledger(
            ledger.ledger_path(str(params.get("ledger_dir") or ".")),
            current_evidence_epoch=_int(params.get("evidence_epoch")))
        summary = ledger.summarize(read, planned_biases=biases)
        # ``budget_exhausted`` 本流程**自己从不置位**,所以 ``series_aborted_budget``
        # 走不到这里 —— 这是有意的,不是漏了:本流程只有**每偏压**的尝试预算
        # (耗尽 ⇒ 那个偏压 ``bias_undecided`` ⇒ 序列 ``series_partial``),
        # 而「整段跑超了帧数/时间/花销」是 conduct 那一层的账(证据包里的
        # ``budget_used`` 就住在那边)。留着这个入口是给上层置位用的;
        # 若哪天这里也长出全局预算,置位点就在这个 partial 上。
        exit_state = ledger.series_exit(
            summary,
            aborted_tip=bool(pd.get("aborted_tip")),
            budget_exhausted=bool(pd.get("budget_exhausted")))

        unreadable = list(summary.get("unreadable") or [])
        if params.get("elevated_bias_abs_v") is None:
            # 槽位没声明要**出声**。沉默会被读成「查过了,没有超标的偏压」。
            unreadable.append({
                "what": "elevated_bias_abs_v",
                "why": ("用户没有声明高偏压阈值 ⇒ 高偏压后的强制针尖复核"
                        "**整条序列都没有生效**。这不是「没有高偏压」。")})
        summary["unreadable"] = unreadable

        out = dict(summary)
        out.update({
            "series_exit": exit_state,
            "bias_order": list(pd.get("bias_order") or []),
            "ledger_path": read.path,
            "ledger_rows": len(read.rows),
            "elevated_slot_active": params.get("elevated_bias_abs_v") is not None,
            "scale_precheck": pd.get("scale_precheck") or {},
        })
        if pd.get("ledger_write_error"):
            out["ledger_write_error"] = pd["ledger_write_error"]
        return out

    # ── 执行 ──────────────────────────────────────────────────────────────

    def _epoch_gate(self, params: dict) -> "SkillResult | None":
        """坐标代次闸门。放行返回 ``None``,否则返回拒绝结果。

        ## 在这之前它是一道**装得像在拦**的闸

        `expected_coord_epoch` 的参数描述逐字写着「A coarse move invalidates
        them」,而 2026-08-15 之前这个值**只被写进账本**(`_row` 里那一行),
        全文件没有 `from mast.core.coord_epoch`,五个子步骤也都不查。
        也就是说:一个名字、一句描述、一列账,**没有任何东西会因为它停下来**。
        对照 `spectroscopy_at_positions._epoch_gate` —— 那道是真的。

        ## 与它同样严:只有确凿的 MATCH 放行

        `mast.core.coord_epoch` 的默认策略对「查不到当前代次」是放行加告警,
        那是给「省略代次是常态」的调用方定的。这里不是:这批中心坐标是
        **用户在 approve 时填的**,可能是几天前 —— 「查不到」不能当成
        「对得上」。而这一步要拿着它连扫几个小时。
        **加严在这里,模块本身一个字不改** —— 另一个调用方并不欠这份严格。
        """
        from mast.core import coord_epoch as ce

        verdict = ce.verify(params.get("expected_coord_epoch"),
                            what="这一段逐偏压序列的帧中心")
        if verdict.verified:
            return None
        extra = ""
        if verdict.state in (ce.UNVERIFIABLE, ce.UNSTAMPED):
            extra = ("——本技能在这里比通用策略更严:这批中心坐标是**别处、别的"
                     "时刻**定下的,而这一步要拿着它连扫几个小时,"
                     "「查不到当前代次」不能当成「对得上」。")
        return self.fail(
            f"{verdict.message}{extra}",
            refusal_code=verdict.state,
            requested_coord_epoch=verdict.stamped,
            current_coord_epoch=verdict.current,
            series_exit=ledger.EXIT_BLOCKED)

    def run_composite(self, context, params: dict) -> SkillResult:
        # **第一件事**:代次不对就一帧都不扫。放在解析偏压之前,是因为那些解析
        # 错误再具体也不如「你要扫的地方已经不是那片表面了」要紧。
        refused = self._epoch_gate(params)
        if refused is not None:
            return refused
        biases, err = self._parse_biases(params)
        if err:
            return self.fail(err, series_exit=ledger.EXIT_BLOCKED)
        if not str(params.get("ledger_dir") or "").strip():
            return self.fail(
                "没有给 ledger_dir —— 逐偏压账没有落点,这条序列跑完也没有证据。",
                series_exit=ledger.EXIT_BLOCKED)

        size_m = float(params.get("size_m") or 5e-9)
        scale, problem = self._scale_precheck(size_m, _int(params.get("pixels")))
        if problem and scale not in ("full", "reduced"):
            # 下发之前就知道判不出来 —— 别花那一帧。
            return self.fail(
                f"这组帧参数判不出原子相:{problem}",
                series_exit=ledger.EXIT_BLOCKED,
                scale_precheck={"scale": scale, "problem": problem})

        executor = GraphExecutor(
            composite_name=self._skill_name(), context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed)
        executor.set_partial("params", dict(params))
        executor.set_partial("biases", biases)
        executor.set_partial("scale_precheck",
                             {"scale": scale, "problem": problem})
        executor.set_partial_default("rows", [])
        executor.set_partial_default("aborted_tip", False)
        executor.set_partial_default("budget_exhausted", False)
        self._executor = executor

        executor.run_plan(self.plan_dynamic(params, executor))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()
        data.update(abort_facts(executor.progress))
        return self._finalize(executor, data)

    def _finalize(self, executor: GraphExecutor, data: dict) -> SkillResult:
        """成败 + **缺口必须写进 summary**(照 BatchRegionsScan 的既有语义)。"""
        exit_state = str(data.get("series_exit") or "")
        ok = exit_state in (ledger.EXIT_COMPLETE, ledger.EXIT_PARTIAL)
        n = int(data.get("n_biases") or 0)
        parts = [
            f"{n} 个偏压:{data.get('n_resolved', 0)} 个有原子分辨",
            f"{data.get('n_absent_confirmed', 0)} 个确认没有对比度",
            f"{data.get('n_undecided', 0)} 个证据不足",
        ]
        if data.get("n_blocked"):
            parts.append(f"{data['n_blocked']} 个被拒")
        if data.get("n_tip_aborted"):
            parts.append(f"{data['n_tip_aborted']} 个因针尖事件中止")
        summary = "；".join(parts) + f"。出口:{exit_state}"

        anchors = data.get("anchor_consistency") or {}
        if not anchors.get("n_anchors"):
            summary += ("。⚠️ 没有任何 anchor 帧 —— "
                        "「这一段扫的是同一片原子」这句话没有证据支撑。")
        if data.get("unreadable"):
            summary += f"。读不到 {len(data['unreadable'])} 项（未计入任何一栏）"
        if data.get("ledger_write_error"):
            ok = False
            summary += f"。⚠️ 账没写进去:{data['ledger_write_error']}"

        if not ok:
            return self.fail(summary, **data)
        res = self.ok(**data)
        res.summary = summary
        return res


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill

    return wrap_skill(AtomicBiasSeries, context_provider)


# ── 小工具 ────────────────────────────────────────────────────────────────

def _int(v) -> "int | None":
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v) -> "float | None":
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _actual_center(sdata: dict) -> "tuple[float, float] | None":
    """.sxm 头里的真实中心 —— 漂移的证据。读不到就是 None,**不拿标称值顶替**。"""
    for kx, ky in (("actual_center_x_m", "actual_center_y_m"),
                   ("scan_center_x_m", "scan_center_y_m")):
        x, y = _float_or_none(sdata.get(kx)), _float_or_none(sdata.get(ky))
        if x is not None and y is not None:
            return (x, y)
    return None


def _metrics(vdata: dict) -> dict:
    """判据的**数值**段 —— 只留过/不过之外的那些数,好让阈值日后能被重判。"""
    keys = ("angular_concentration", "fft_sharpness", "snr", "order_ratio",
            "nm_per_px", "scale", "period_fast_axis_nm", "period_radial_nm",
            "slow_axis_trusted")
    return {k: vdata.get(k) for k in keys if k in vdata}


__all__ = ["AtomicBiasSeries", "TIER_NAME", "SLOT_UNDECLARED"]
