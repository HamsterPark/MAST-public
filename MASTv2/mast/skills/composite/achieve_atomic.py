# -*- coding: utf-8 -*-
"""自主获得原子分辨的分级流程。

先用 ScanUntilAtomicResolution 检查当前针尖，再尝试 MakeAtomicResolutionTip。
需要继续处理时，按浅扎、复评、脉冲、稳定、复评的顺序推进；ForgeAuTip 默认关闭，
仅用于明确需要重塑针尖的情况。读不到与未通过是不同状态，不因缺少证据升级干预。
针尖问题与表面区域问题分别处理；RelocateCoarseXY 使用粗动地图，不把步数直接当长度。
流程不擅自执行进针。时间预算限制下一轮的开始，不在一个已经开始的物理动作中途截断。
"""
from __future__ import annotations

import logging
import time
from typing import Iterator

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
)
from mast.skills.composite._base import (
    CompositeProgress,
    CompositeSkillGraph,
    CompositeStep,
    GraphExecutor,
)

logger = logging.getLogger(__name__)

#: 「判不了」那一组出局词。与 ``atomic_phase.ALL_REASONS`` 的下半段、
#: ``verify_atomic_resolution.REASON_VERDICT`` 的下半段同源。
#:
#: ⚠️ 加新出局词时三处一起改 —— 这里漏了会让「判不了」被当成「没有」，
#: 后果是去磨一根本来就好的针尖。
UNDECIDABLE_REASONS: frozenset = frozenset({
    "too_few_periods", "scale_gate", "scale_reduced", "unknown_pixel_size",
    "insufficient_data", "dead_flat", "dependency_unavailable",
})

# 默认总时间预算，单位为分钟，可由调用方按任务需要限制。
DEFAULT_BUDGET_MIN = 480.0

#: 记了一条账、但**那一档一次都没跑**的结局码。
#:
#: 它们不该进 ``rungs_used`` —— 那份清单回答的是「代价花在哪」,把没花的代价
#: 算进去,屏幕上就会出现「阶梯上开过 5 档」而其中一档从未打开。
#:
#: ⚠️ 与 ``narration_templates._ATOMIC_OUTCOME_ZH`` 同源(那边给措辞、这边定
#: 计不计账)。加一个「没开成」的结局码时两处一起改。
NEVER_OPENED_OUTCOMES: frozenset = frozenset({
    "skipped_budget",             # 预算用完,这一档没开
    "not_allowed",                # allow_forge 关着,锻造档没开
    "tip_conditioning_disabled",  # 调用方不让动针尖,后面几档都没开
    "not_imaging",                # 反馈没开,一档都没进
})

# 每档之后按配置复评若干帧，并避开已被扎针或脉冲扰动的区域。
_RECHECK = {"max_attempts": 3, "relocate_after_attempts": 2}


def _data(executor: GraphExecutor, step_id: str) -> dict:
    r = executor.sub_results.get(step_id)
    return dict(getattr(r, "data", None) or {})


def _is_no_usable_region(verdict) -> bool:
    """判据说的是「**我找过了，这里没有**」，不是「我没法找」。

    常量从 ``flat_region`` 取，不在这里抄第二份 —— 抄的那份迟早对不上。
    ``flat_region.py:688`` 自己写着这两件事的下一步相反：
    ``no_usable_region`` ⇒ 换一块再扫；别的失败 ⇒ 判不了。
    """
    try:
        from mast.skills.builtins.flat_region import VERDICT_NO_USABLE_REGION as _V
    except Exception:  # noqa: BLE001 — 取不到常量就按「不是」处理（不换区）
        return False
    return str(verdict or "") == _V


def _coarse_suggestion() -> "tuple[dict | None, str]":
    """粗动大地图给的下一站 —— ``(suggestion, note)``。

    ``suggestion`` 为 None 表示**不该走**，理由在 note 里。三种情况：

    * **读不到实验记录** —— 这不是「这片表面很干净」。两者在几何上无法区分，
      而在读不到历史的情况下发起一次粗动，和在确认过的地方发起，是两件事。
    * 地图说这一带已经用完（``suggestion is None``）—— 通常意味着要换样品。
    * suggestion 缺字段 —— 宁可不动。

    ⚠️ **绝不要用步数去算米**：粗动步长随驱动幅度/负载/温度漂移，低温下同样
    步数走的距离可以差几倍（``get_coarse_map`` 自述）。所以这里原样把
    axis/direction/steps 交给 ``RelocateCoarseXY``，一个换算都不做。
    """
    try:
        from mast.core.map_scope import marker_rows
        from mast.io.coarse_map import CoarseMapConfig, build_coarse_map

        rows, ok = marker_rows()
        if not ok:
            return None, "读不到实验记录 —— 不知道去过哪些片，不能凭空换区"
        d = build_coarse_map(rows, CoarseMapConfig()).as_dict()
        sug = d.get("suggestion")
        note = str(d.get("note") or "")
        if not sug:
            return None, note or "粗动地图说这一带已经用完（通常意味着要换样品）"
        if any(sug.get(k) is None for k in ("axis", "direction", "steps")):
            return None, "粗动地图给的 suggestion 缺 axis/direction/steps —— 不动"
        return dict(sug), (str(sug.get("reason") or "") or note)
    except Exception as exc:  # noqa: BLE001
        return None, "粗动地图读不出来: %s" % exc


def _remember_frame(executor: GraphExecutor, data: dict) -> None:
    """记住**最后一张真的存了盘的图** —— 跑完那条旁白要拿它画分析图。

    分析图「**无论成或不成**」都要挂。而没拿到的时候
    ``final_path`` 是空的（那条路径只在验证那一支写），所以每一档都得顺手把
    自己交出来的最后一帧留下来 —— 否则最需要解释的那一次跑,恰好是唯一
    没有证据图的那一次。

    ⚠️ **「真的存了盘」这四个字以前只写在这段注释里,代码从来没查过。**
    后果是它会拿一个**取不到文件**的路径去覆盖前面那个能画图的:
    R1 交出一张真帧、随后 R3a/R3b 的复评各交出一条 history(路径指向一张
    被清掉/没落盘/技能只是回了个名字的帧)⇒ ``last_frame_path`` 被换成后者 ⇒
    ``_render_report`` 提前回 ⇒ **整跑一张分析图都没有**,而这恰恰是要求
    要图的那一支(2026-08-24 由 test_achieve_atomic_narration 当场判红)。
    「注释描述了一道并不存在的检查」是本仓的常客,这里把它做出来:
    **落不到磁盘的候选一律跳过,而且宁可留着上一档那张好的,也不清空。**
    """
    from pathlib import Path as _P

    def _on_disk(p) -> bool:
        try:
            return bool(p) and _P(str(p)).is_file()
        except (OSError, ValueError):   # 非法路径名(Windows 上真的会抛)
            return False

    cands = [data.get("found_path"), data.get("scan_path")]
    cands += [(h or {}).get("path") for h in reversed(list(data.get("history") or []))]
    for c in cands:
        if _on_disk(c):
            executor.set_partial("last_frame_path", str(c))
            return


def _concentrations(history) -> list:
    """一段 ``ScanUntilAtomicResolution`` 历史里**每一帧**的角向集中度。

    只收真的判过的帧(``relocated`` 那种记账行没有读数),而且只收有限数 ——
    一个渲染成 ``None`` 的读数比不说更糟。给旁白用的是
    「每帧的角向集中度」,而在此之前这几个数只存在于回包的 ``history`` 里。
    """
    out: list = []
    for h in list(history or []):
        v = (h or {}).get("concentration")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(round(float(v), 1))
    return out


def _scan_facts(d: dict) -> dict:
    """统一生成阶段复评摘要，区分循环次数与实际判过的帧数。"""
    return {
        "attempts": d.get("attempts"),
        "frames_judged": d.get("frames_judged"),
        "scans_failed": d.get("scans_failed"),
        "repeat_frames": d.get("repeat_frames"),
        "assess_failed": d.get("assess_failed"),
        "n_undetermined": d.get("n_undetermined"),
        "concentrations": _concentrations(d.get("history")),
    }


def _render_report(path: str) -> "tuple[dict | None, dict]":
    """把三联分析图画好落盘,回 ``(image, stats)``。**永不抛。**

    ⚠️ ``project_root`` 住在 ``mast._runtime_paths``，不是 ``mast.core.paths``。
    写错的话这一整段被下面的 except 吞掉 ⇒ 图**静默地永远画不出来**，
    而技能照常报成功。所以配套测试要验「图真的落盘了」，不是只验 import。

    路径不存在时**提前回**:没有文件就没有图,而 ``render_atomic_report`` 会为此
    去 import matplotlib —— 在一条只想说句话的路径上白付一次几百毫秒的 import。
    """
    stats: dict = {}
    try:
        from pathlib import Path as _P

        p = str(path or "")
        if not p or not _P(p).is_file():
            return None, stats
        from mast._runtime_paths import project_root
        from mast.data.atomic_report_plot import render_atomic_report

        out = project_root() / "artifacts" / "atomic_reports"
        png = render_atomic_report(p, out / (_P(p).stem + ".png"), stats=stats)
        if png:
            return {"src": png, "origin": "milestone_png"}, stats
    except Exception as exc:  # noqa: BLE001
        logger.debug("原子分辨三联图没画成: %s", exc)
    return None, stats


def _report_or_nothing(path: str) -> "tuple[dict | None, dict]":
    """``_render_report`` 的**调用点**护栏 —— 画图这件事一次都不许弄坏实验。

    ⚠️ ``_render_report`` 自己写着「永不抛」,而那句话只覆盖它的**函数体**。
    调用点在两条旁白里都写在 ``try`` **外面**,于是只要这条路上任何一环炸了
    (matplotlib 装坏、``project_root()`` 抛、这个函数被替换),异常就直接顺着
    ``aggregate`` 冒到实验线程上 —— 一次八小时的跑被一张配图带走。
    2026-08-24 由 ``test_a_broken_narration_channel_never_breaks_the_run``
    当场判红:``_render_report`` 一抛,整条编排就停了。

    「函数自己保证不抛」和「调用点保证不抛」是**两件事**:前者对不了替身、
    对不了 import 期的错。这里两道都上。
    """
    try:
        return _render_report(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("原子分辨三联图整条路都没走通: %s", exc)
        return None, {}


def _half(stats: dict, i: int):
    """三联图量到的「帧内前后两半」里的第 i 个。**不是两个就一个都不交。**

    生产方是 ``atomic_report_plot._render``:``half_concentrations`` 要么是
    长度 2 的列表,要么是 ``None``。长度不对时摆出来的两个数就不是「两半」——
    而一个说得出口的假读数比缺席更坏。
    """
    h = stats.get("half_concentrations")
    if isinstance(h, (list, tuple)) and len(h) == 2:
        return h[i]
    return None


def _narrate_run_result(out: dict, pd: dict) -> None:
    """整跑的结局 —— **没拿到的时候也要说,而且照样挂分析图**(2026-08-24)。

    要求：要有一段详细介绍分析结果的旁白（含分析图），**无论成或不成**。
    在此之前只有「验证通过」那一支发得出声音,于是最需要解释的那一次跑(没拿到)
    反而是全程最沉默的 —— 屏幕上只剩一串通用的扫描进度。

    失败时挂的那张图恰恰是**「为什么不算」的证据**:功率谱上没有离散峰、
    剖面没有周期,一眼就看得出来,而一句「没拿到」看不出任何东西。

    ⚠️ 拿到并且**已经**由 :func:`_narrate_atomic_result` 说过的那一支不重复发 ——
    同一条结论说两遍会把真正有信息量的那几条淹掉(见
    ``test_atomic_result_narration.py`` 里那条「不进 RESULT_KIND_FOR_SKILL」)。

    ⚠️ 全程吞异常:一条旁白绝不许弄坏一次八小时的实验。
    """
    if pd.get("result_narrated"):
        return
    path = str(pd.get("final_path") or pd.get("last_frame_path") or "")
    image, stats = _report_or_nothing(path)
    try:
        from mast.chat.narration import narrate

        t0 = pd.get("started_monotonic")
        elapsed = (round((time.monotonic() - float(t0)) / 60.0, 1)
                   if isinstance(t0, (int, float)) else None)
        narrate(
            "atomic_run_result", image=image,
            achieved=out.get("achieved"),
            verified=str(out.get("verified") or ""),
            rungs_used=",".join(str(r) for r in (out.get("rungs_used") or [])),
            relocations=pd.get("relocations"),
            elapsed_min=elapsed,
            # 拿不到验证方的读数时退到三联图**本来就量过**的那一份 ——
            # 那是「为什么不算」的证据里唯一的数字。
            concentration=(out.get("angular_concentration")
                           or stats.get("angular_concentration")),
            # 要的是「**详细介绍其分析结果**,无论成或不成」。
            # 这四个数三联图渲染时**本来就算过了**(``stats`` 是它就地填的),
            # 只是从没往这条路上交 —— 于是失败那一次只剩一句「没有拿到」,
            # 而反扫有没有、周期多少、帧内两半差多少,正是「为什么不算」的证据。
            # ⚠️ 键名对着**生产方** ``atomic_report_plot._render`` 里那个
            # ``stats.update({...})`` 读,不对着自己想要的名字读 —— 上一版在
            # ``verify`` 上栽过:六个键里三个生产方根本不产出,句子若无其事地
            # 少说三样,而两边的单测各自都绿。
            concentration_bwd=stats.get("angular_concentration_backward"),
            period_nm=stats.get("period_nm"),
            # 只有恰好两半时才交 —— 长度不是 2 就说明它不是「前后两半」,
            # 而 ``[0]``/``[-1]`` 在长度 1 上会把同一个数摆成「X / X」。
            half_a=_half(stats, 0),
            half_b=_half(stats, 1),
            advice=str(out.get("advice") or ""),
            scan_path=path,
        )
    except Exception:  # noqa: BLE001
        pass


def _narrate_atomic_result(path: str, verify: dict, rungs: list) -> None:
    """将阶段读数映射为旁白字段，缺失值须明确说明，避免错误层级导致统一兜底。"""
    image, stats = _report_or_nothing(path)
    try:
        from mast.chat.narration import narrate

        # ⚠️ **键名要对着生产方读，不是对着自己想要的名字读。**
        # 第一版这里读 ``verify["period_nm"]`` / ``["half_concentrations"]`` /
        # ``["angular_concentration_backward"]`` —— 六个键里三个 VerifyAtomicResolution
        # 根本不产出，于是那三个数**永远是 None**，句子**若无其事地少说三样**。
        # 与 ``auto_tilt_result`` 读四个幽灵键、26/26 走兜底是同一个形状，
        # 只是这次少的是数不是整句话（2026-08-23 由旁白那侧的测试查出来）。
        #
        # 现在：能从验证方拿的就从验证方拿（它是权威），拿不到的退到三联图
        # **本来就量过**的那一份 —— 反扫的角向集中度全仓只有那里有生产方。
        halves = (verify.get("half_concentrations")
                  or stats.get("half_concentrations") or ())
        narrate(
            "atomic_resolution_achieved", image=image,
            verdict=str(verify.get("verdict") or ""),
            concentration=(verify.get("angular_concentration")
                           or stats.get("angular_concentration")),
            concentration_bwd=stats.get("angular_concentration_backward"),
            period_nm=(verify.get("period_radial_nm")
                       or verify.get("period_fast_axis_nm")
                       or stats.get("period_nm")),
            half_a=(halves[0] if len(halves) == 2 else None),
            half_b=(halves[1] if len(halves) == 2 else None),
            rungs_used=",".join(str(r.get("rung") or "") for r in rungs),
            scan_path=str(path),
        )
    except Exception:  # noqa: BLE001
        pass


class AchieveAtomicResolution(CompositeSkillGraph):
    """一句「我想要原子分辨」就能调的顶层技能。零必填参数。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AchieveAtomicResolution",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            # L5 = 通宵 campaign，串起若干 L3/L4 技能（见 SkillMetadata 的分级说明）。
            # 它串的正是 ScanUntilAtomicResolution(L3) / MakeAtomicResolutionTip(L3)
            # / ForgeAuTip(L4) / RelocateCoarseXY，默认预算 8 小时，外面还套站点循环。
            composition_level=5,
            description=(
                "**用户只说「我想要原子分辨」时就调这一个。**"
                "从当前状态出发：先自己量出针尖和表面的现状，再按读数决定从"
                "哪一档起步，逐档升级（重扫 → 偏压抖动 → 浅扎 → 脉冲 → 扎针稳定 → 锻造），"
                "每一档之后都复评，最后做正反扫验证并交出图与数字。"
                "**零必填参数 —— 调用方不需要、也不应该替它填任何数**："
                "它自己会读扫描框、扫描速度、反馈状态。"
                "判据说「判不了」时它改成像条件而**不动针尖**。"
            ),
            parameters=[
                ParameterSpec(
                    name="total_budget_min", type="float", required=False,
                    default=DEFAULT_BUDGET_MIN, unit="min",
                    min_value=5.0, max_value=2880.0,
                    description=(
                        "总预算（分钟）。只用来决定「还要不要开下一档」，"
                        "**绝不会在一档中途因为超时把它判失败**。"
                        "整夜 = 480（默认），一整天 = 1440。"),
                ),
                ParameterSpec(
                    name="allow_tip_conditioning", type="bool", required=False,
                    default=True,
                    description="允许动针尖（脉冲 / 扎针）。关掉就只做前两档。",
                ),
                ParameterSpec(
                    name="allow_forge", type="bool", required=False, default=False,
                    description=(
                        "允许 ForgeAuTip，默认关闭。仅用于明确需要重塑针尖的严重形貌问题；"
                        "常规处理失败本身不是启用理由。表面区域问题使用 RelocateCoarseXY。"),
                ),
                ParameterSpec(
                    name="allow_on_qplus", type="bool", required=False, default=False,
                    description="qPlus 传感器上放行扎针（护栏默认拒绝）。",
                ),
                ParameterSpec(
                    name="allow_relocate", type="bool", required=False, default=True,
                    description=(
                        "表面用完了（判据说 no_usable_region）时**自己粗动换一片新表面**"
                        "并从最便宜那档重来。默认**开** —— 该给的权限"
                        "就应该给它。往哪走、走多少步由粗动大地图的"
                        "suggestion 给，不由这个技能拍；地图说这一带用完了就如实停下"
                        "（通常意味着要换样品）。"
                        "⚠️ 换区会让扫描地图进入**新的坐标代次**，旧坐标全部作废。"),
                ),
                ParameterSpec(
                    name="max_relocations", type="int", required=False, default=3,
                    min_value=0, max_value=20,
                    description=(
                        "最多换几次区。这是**防跑飞的护栏，不是策略** —— 真正的停点是"
                        "粗动地图说没有下一站了，或者总预算用完。"),
                ),
            ],
            tags=["atomic", "resolution", "autonomous", "top-level", "composite"],
        )

    # ── 编排 ────────────────────────────────────────────────────────────────

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        t0 = time.monotonic()
        budget_s = float(params.get("total_budget_min") or DEFAULT_BUDGET_MIN) * 60.0
        allow_tip = bool(params.get("allow_tip_conditioning", True))
        allow_forge = bool(params.get("allow_forge", False))
        allow_qplus = bool(params.get("allow_on_qplus", False))
        allow_reloc = bool(params.get("allow_relocate", True))
        _mr = params.get("max_relocations")
        max_reloc = int(_mr) if _mr is not None else 3

        rungs: list = []
        executor.set_partial("rungs", rungs)
        # 跑完那条旁白要报「用时多少分钟」,而 aggregate 拿不到这个闭包 ——
        # 起跑时刻得留在 partial_data 里。``time.monotonic()`` 只在同一个进程内
        # 可比,而 aggregate 就在同一个进程里(见 ``_base._graph_execute``)。
        executor.set_partial("started_monotonic", t0)

        def _budget_left() -> float:
            return budget_s - (time.monotonic() - t0)

        def _note(rung: str, outcome: str, *, record: bool = True, **kw) -> None:
            """记一笔账,**并且当场把它念给用户听**(2026-08-24)。

            旁白接在这里而不是在每一档各写一句:这是 ``plan_dynamic`` 里唯一的
            记账口,每一档、每一次分诊、每一个停下的理由都从这儿过。于是
            「加了一档忘了发旁白」在结构上不可能发生 —— 而在此之前
            ``AchieveAtomicResolution`` **只在最后成功时**发一条,中间每一步判断
            一个字都没有(而每一步判断都该有旁白)。

            ``record=False`` = **说但不记账**。「我们要开哪一档」是一句话,
            不是一条账目;记进 ``rungs`` 会让 ``rungs_used`` 出现重复,
            而那份清单是「代价花在哪」的答案。
            """
            rec = {"rung": rung, "outcome": outcome,
                   "at_min": round((time.monotonic() - t0) / 60.0, 1)}
            rec.update(kw)
            if record:
                rungs.append(rec)
                executor.set_partial("rungs", list(rungs))
            logger.info("AchieveAtomicResolution [%s] %s %s", rung, outcome, kw)
            try:
                from mast.chat.narration import narrate

                # 旁白字段与生产方结构保持一致，避免错误的嵌套路径使每次调用都走缺失兜底。
                narrate("atomic_rung", **rec)
            except Exception:  # noqa: BLE001 — 一条旁白绝不许弄坏实验
                pass

        # ── Phase 0：现状。不动手，只读 ────────────────────────────────
        #
        # 为什么先读：本技能的全部价值就是「自己把事实量出来」。而**最先要问的
        # 不是针尖好不好，是我们在不在成像状态** —— 反馈没开 / 针不在隧道结时，
        # 后面每一帧都是废的，再多的修针都是在修一个不存在的问题。
        yield CompositeStep(
            step_id="p0:zctrl", skill_name="GetZControllerState", params={},
            optional=True, checkpoint_after=False, tags=("survey",),
        )
        if executor.progress.aborted:
            return
        z = _data(executor, "p0:zctrl")
        executor.set_partial("start_state", {
            "controller_on": z.get("controller_on"),
            "setpoint_a": z.get("setpoint"),
            "z_m": z.get("z_m"),
        })
        # 现状念出来 —— 这是后面每一条旁白的坐标系,也是本技能的**全部价值**
        # (「自己把事实量出来」)在屏幕上唯一看得见的地方。
        # ⚠️ ``controller_on`` 三态:True / False / **读不到**。这里原样交出去,
        # 由模板分三态说;在这里折成 bool 就是把「读不到」念成「没开」。
        try:
            from mast.chat.narration import narrate

            narrate("atomic_begin",
                    controller_on=z.get("controller_on"),
                    setpoint_a=z.get("setpoint"), z_m=z.get("z_m"),
                    budget_min=round(budget_s / 60.0, 1),
                    allow_tip=allow_tip, allow_forge=allow_forge,
                    allow_relocate=allow_reloc)
        except Exception:  # noqa: BLE001
            pass
        if z and z.get("controller_on") is False:
            # **不自己进针。** 进针是另一件事，而且危险 —— 它需要知道样品换没换、
            # 粗动在哪、是不是 4 K，猜错的代价是撞针。如实停下并说清楚，
            # 比替用户猜要好。
            _note("p0", "not_imaging", reason="z_controller_off")
            executor.set_partial("aborted_reason", "z_controller_off")
            return

        # ══ 站点循环 ═══════════════════════════════════════════════════════
        #
        # 一个「站点」= 一片表面。在一个站点上把阶梯爬完还不行，**先问表面**：
        # 如果这儿根本没有干净台面，那是**表面**的问题，不是针尖的 ——
        # 这时候继续修针是在治错的病。
        for site in range(int(max_reloc) + 1):
            # 第一个站点用裸 step_id（绝大多数运行只有这一个站点，报文好读）；
            # 换区之后加 ``#2`` / ``#3`` 后缀保证唯一。冒号仍留给子步骤
            # （``r3a:recheck``），所以 ``split(":")[0]`` 照旧能取到档位名。
            def T(base: str, _s: int = site) -> str:
                return base if _s == 0 else "%s#%d" % (base, _s + 1)

            # ── R1：最便宜的一档 —— 不动针尖、不改参数 ────────────────────
            #
            # 它同时是**基线测量**：跑完之后 history 里就有了 conc 与出局词，
            # 分诊需要的事实全在里面。所以这里不另外扫一帧去「先量一下」——
            # 那会是同一个动作的第二份实现，而两份迟早只有一份对。
            _note(T("r1"), "begin", record=False,
                  note="不动针尖、不改参数,先看它自己会不会出来")
            yield CompositeStep(
                step_id=T("r1"), skill_name="ScanUntilAtomicResolution",
                params=dict(_RECHECK), optional=True, checkpoint_after=True,
                tags=("rung", "r1", "site=%d" % site),
            )
            if executor.progress.aborted:
                return
            r1 = _data(executor, T("r1"))
            _remember_frame(executor, r1)
            if r1.get("found"):
                _note(T("r1"), "found", path=r1.get("found_path"),
                      **_scan_facts(r1))
                yield from self._verify(executor, r1.get("found_path"))
                return
            _note(T("r1"), "no_lattice", **_scan_facts(r1))

            # ── 分诊一：判不了 ≠ 没有 ─────────────────────────────────────
            #
            # 用的是仓里已经标定过的那个区分。每一帧都判不了 ⇒ 证据不足 ⇒
            # **不许动针尖**，也不许换区（换区同样是拿判不了当证据在行动）。
            #
            # ⚠️ 「判过的帧」这三个词从 ``scan_until_atomic`` **import**,不在这里
            # 抄第二份。那边 2026-08-24 加了 ``no_scan`` / ``no_new_frame``
            # 两种记账行(「这一次根本没采到」/「交回来的还是判过的那张」)——
            # 抄一份在这里的话,新出现的记账行会被这道分诊当成一帧真判定,
            # 于是「没采到」被当成关于针尖的证据,而这正是这道分诊要挡的那件事。
            from mast.skills.composite.scan_until_atomic import JUDGED_VERDICTS

            history = list(r1.get("history") or [])
            judged = [h for h in history if h.get("verdict") in JUDGED_VERDICTS]
            n_undet = sum(1 for h in judged if h.get("verdict") == "undetermined")
            if judged and n_undet == len(judged):
                _note("triage", "undecidable_only", n_frames=len(judged))
                executor.set_partial("aborted_reason", "undecidable_frames")
                return

            # ── 分诊二：这是**表面**的问题还是**针尖**的问题 ───────────────
            #
            # `FindFlatRegion` 早就把这两件事分开了（``flat_region.py:688``）：
            #
            #     verdict == "no_usable_region"  ⇒ 「我找过了，这里没有」⇒ 换一块再扫
            #     别的失败                        ⇒ 「我没法找」⇒ 判不了
            #
            # 那句「换一块再扫」是它自己写的下一步，而**没有调用方在这一层接它**。
            # 又是「生产方接好了、消费方缺席」。
            #
            # ⚠️ 只在 R1 判成「**没有**」之后才问 —— 判不了的时候上面已经 return，
            # 不会走到这里。拿判不了去触发一次粗动换区，是把「读不到」当证据行动，
            # 而粗动是本流程里影响面最大的动作。
            last_path = ""
            for h in reversed(history):
                if h.get("path"):
                    last_path = str(h["path"])
                    break
            surface_spent = False
            if allow_reloc and site < int(max_reloc) and not last_path:
                # 缺少表面评估时明确报告未知，不能暗示已经确定是针尖问题。
                _note(T("surface"), "no_frame_to_ask",
                      hint="R1 没交出存盘路径（save_every_frame 关了？）")
            if allow_reloc and last_path and site < int(max_reloc):
                yield CompositeStep(
                    step_id=T("surface"), skill_name="FindFlatRegion",
                    params={"scan_path": last_path},
                    optional=True, checkpoint_after=False,
                    tags=("triage", "surface", "site=%d" % site),
                )
                if executor.progress.aborted:
                    return
                sd = _data(executor, T("surface"))
                surface_spent = _is_no_usable_region(sd.get("verdict"))

                # ⚠️ **结局码写成字面量,不写成三元表达式。** 措辞表那道结构闸门
                # (``test_every_outcome_the_emitter_can_note_has_a_phrase`` /
                # ``…_every_phrase_is_reachable…``)走 AST 读 ``_note`` 的第二个
                # 实参,只认 ``ast.Constant``；写成 ``A if c else B`` 时它两个都
                # 读不到 —— 于是这两条措辞在闸门眼里既「没人发」也「没人要」,
                # 加一档忘了写措辞照样全绿。判定归判定,**发出的码要摆在明面上**。
                if surface_spent:
                    _note(T("surface"), "no_usable_region",
                          verdict=sd.get("verdict"))
                else:
                    _note(T("surface"), "surface_ok", verdict=sd.get("verdict"))

            if surface_spent:
                # 表面用完了 ⇒ 换一片新表面。**往哪走、走多少步由粗动大地图给**，
                # 不由我拍：``build_coarse_map`` 的 suggestion 带 axis/direction/
                # steps 与理由，而 ``suggestion is None`` 的意思是「这一带已经用完」
                # —— 通常意味着要换样品，那不是这个流程能解决的。
                #
                # ⚠️ **绝不要用步数去算米**：粗动步长随驱动幅度/负载/温度漂移，
                # 低温下同样步数走的距离可以差几倍（get_coarse_map 自述）。
                sug, note = _coarse_suggestion()
                if not sug:
                    _note(T("relocate"), "no_site_left", note=note)
                    executor.set_partial("aborted_reason", "coarse_map_exhausted")
                    executor.set_partial("coarse_note", note)
                    return
                # 把地图给的**理由**一起念出来 —— 「沿 x+ 走 300 步」回答的是
                # 「往哪走」,而用户要的是「**为什么是那儿**」(「往 +x 还有没去过
                # 的地方」)。粗动是本流程里影响面最大的动作:它作废整张扫描地图的
                # 坐标代次,一次说不清楚就再也对不回去了。理由本来就在
                # ``_coarse_suggestion`` 的第二个返回值里,只是从没往外交过。
                _note(T("relocate"), "moving", note=note,
                      **{k: sug.get(k) for k in ("axis", "direction", "steps")})
                yield CompositeStep(
                    step_id=T("relocate"), skill_name="RelocateCoarseXY",
                    params={"axis": sug["axis"], "direction": sug["direction"],
                            "steps": int(sug["steps"]), "reapproach": True},
                    optional=True, checkpoint_after=True,
                    tags=("relocate", "site=%d" % site),
                )
                if executor.progress.aborted:
                    return
                rel = _data(executor, T("relocate"))
                executor.set_partial(
                    "relocations",
                    int(executor.progress.partial_data.get("relocations", 0)) + 1)
                _note(T("relocate"), "moved", ok=bool(rel))
                # 换区成功 ⇒ 新站点、**新的坐标代次**（旧坐标全部作废）⇒
                # 回到阶梯最便宜那一档从头来。
                continue

            if not allow_tip:
                _note("triage", "tip_conditioning_disabled")
                return

            # ── R2：偏压抖动 + 阈值浅扎（用户亲手教的手法）──────────────
            if _budget_left() <= 0:
                _note(T("r2"), "skipped_budget")
                return
            _note(T("r2"), "begin", record=False,
                  note="重扫不出来而表面还有台面 ⇒ 升级:偏压抖动 + 阈值浅扎"
                       "(用户亲手教的手法)")
            yield CompositeStep(
                step_id=T("r2"), skill_name="MakeAtomicResolutionTip",
                params={"allow_on_qplus": allow_qplus},
                optional=True, checkpoint_after=True,
                tags=("rung", "r2", "site=%d" % site),
            )
            if executor.progress.aborted:
                return
            r2 = _data(executor, T("r2"))
            _remember_frame(executor, r2)
            r2_path = r2.get("found_path") or r2.get("scan_path")
            if str(r2.get("outcome") or "") == "atomic_tip_ready":
                _note(T("r2"), "found", path=r2_path)
                yield from self._verify(executor, r2_path)
                return
            _note(T("r2"), "no_lattice", skill_outcome=r2.get("outcome"))

            # R3：浅扎 → 复评 → 脉冲 → 稳定后复评。
            # 按形貌选择处理方式；PokeConditionTip 遇到其无法处理的多针尖形貌会拒绝并给出原因。
            # 每次干预后重新取证，不把一次失败当作下一档必然有效的依据。
            for base, skill in (("r3a", "PokeConditionTip"),
                                ("r3b", "PulseConditionTip")):
                if _budget_left() <= 0:
                    _note(T(base), "skipped_budget")
                    return
                # 只传入技能元数据实际声明的参数，参数校验失败必须报告。
                pp = ({"allow_on_qplus": allow_qplus}
                      if skill == "PokeConditionTip" else {})
                _note(T(base), "begin", record=False, skill_outcome=skill,
                      note=("浅扎先来:2 nm 以内的扎针比脉冲温和(判据),"
                            "而且它治不了的情况(多针尖)会自己要求让位给脉冲"
                            if base == "r3a" else
                            "浅扎之后复评仍不达标 ⇒ 才轮到脉冲:它更粗暴,"
                            "但被压垮/分叉的顶端只有它治得了"))
                yield CompositeStep(
                    step_id=T(base), skill_name=skill, params=pp,
                    optional=True, checkpoint_after=True,
                    tags=("rung", base, "site=%d" % site),
                )
                if executor.progress.aborted:
                    return
                _note(T(base), "done",
                      skill_outcome=_data(executor, T(base)).get("outcome"))

                # 脉冲后通过扎针稳定阶段，再进入复评；主流程和独立修针流程应保持此顺序。
                if skill == "PulseConditionTip" and _budget_left() > 0:
                    _note(T(base + ":stabilise"), "begin", record=False,
                          skill_outcome="PokeConditionTip",
                          note="脉冲之后进入扎针稳定步骤，再重新评估针尖状态。")
                    yield CompositeStep(
                        step_id=T(base + ":stabilise"),
                        skill_name="PokeConditionTip",
                        params={"allow_on_qplus": allow_qplus},
                        optional=True, checkpoint_after=True,
                        tags=("rung", base, "stabilise", "site=%d" % site),
                    )
                    if executor.progress.aborted:
                        return
                    _note(T(base + ":stabilise"), "done",
                          skill_outcome=_data(
                              executor, T(base + ":stabilise")).get("outcome"))

                # 复评走最便宜那档 —— 它顺带也可能直接把图扫出来。
                yield CompositeStep(
                    step_id=T(base + ":recheck"),
                    skill_name="ScanUntilAtomicResolution",
                    params=dict(_RECHECK), optional=True, checkpoint_after=True,
                    tags=("recheck", base, "site=%d" % site),
                )
                if executor.progress.aborted:
                    return
                rc = _data(executor, T(base + ":recheck"))
                _remember_frame(executor, rc)
                if rc.get("found"):
                    _note(T(base + ":recheck"), "found",
                          path=rc.get("found_path"), **_scan_facts(rc))
                    yield from self._verify(executor, rc.get("found_path"))
                    return
                _note(T(base + ":recheck"), "no_lattice", **_scan_facts(rc))

            # ── R4：ForgeAuTip ────────────────────────────────────────────
            #
            # ⚠️ 它的适应症是「**针尖已经变得极其糟糕**（顶端毁掉，要重新锻造）」，
            # **不是**「前面几档都失败了」。所以默认关，要用户显式打开。
            # 表面用完了则走上面那条换区分支，不该落到它头上。
            if not allow_forge:
                _note(T("r4"), "not_allowed")
                return
            if _budget_left() <= 0:
                _note(T("r4"), "skipped_budget")
                return
            _note(T("r4"), "begin", record=False,
                  note="用户显式打开了 allow_forge —— 最贵、最不可逆的一档")
            yield CompositeStep(
                step_id=T("r4"), skill_name="ForgeAuTip", params={},
                optional=True, checkpoint_after=True,
                tags=("rung", "r4", "site=%d" % site),
            )
            if executor.progress.aborted:
                return
            _note(T("r4"), "done",
                  skill_outcome=_data(executor, T("r4")).get("outcome"))
            yield CompositeStep(
                step_id=T("r4:recheck"), skill_name="ScanUntilAtomicResolution",
                params=dict(_RECHECK), optional=True, checkpoint_after=True,
                tags=("recheck", "r4", "site=%d" % site),
            )
            if executor.progress.aborted:
                return
            r4c = _data(executor, T("r4:recheck"))
            _remember_frame(executor, r4c)
            if r4c.get("found"):
                _note(T("r4:recheck"), "found", path=r4c.get("found_path"),
                      **_scan_facts(r4c))
                yield from self._verify(executor, r4c.get("found_path"))
                return
            _note(T("r4:recheck"), "no_lattice", **_scan_facts(r4c))
            return

    # ── 验证 ────────────────────────────────────────────────────────────────

    def _verify(self, executor: GraphExecutor, path) -> Iterator[CompositeStep]:
        """拿到图之后再验一道。

        为什么不能省：2026-08-23 独立复核那张成功帧时，是**三重确认**才让人放心
        —— 正反扫都有晶格（只在一个方向出现就是针尖产物）、整帧采满（不是半张帧
        的运气）、帧内两半没突变（认证期间针尖是稳的）。这三条
        ``VerifyAtomicResolution`` 都做，而它 ``AUTO`` 级、只读文件、不碰硬件 ——
        没有不做的理由。
        """
        if not path:
            executor.set_partial("verified", None)
            executor.set_partial("verify_note", "拿到了图但没拿到路径，没法验证")
            return
        yield CompositeStep(
            step_id="verify", skill_name="VerifyAtomicResolution",
            params={"scan_path": str(path)},
            optional=True, checkpoint_after=True, tags=("verify",),
        )
        if executor.progress.aborted:
            return
        v = _data(executor, "verify")
        executor.set_partial("verified", v.get("verdict"))
        executor.set_partial("verify_data", v)
        executor.set_partial("final_path", str(path))
        _narrate_atomic_result(str(path), v,
                               list(executor.progress.partial_data.get("rungs") or []))
        # 这一支已经把结论 + 三联图说过了 ⇒ ``aggregate`` 那条整跑旁白不再重复
        # 发一遍。同一条结论说两遍会把真正有信息量的那几条淹掉。
        executor.set_partial("result_narrated", True)

    # ── 报告 ────────────────────────────────────────────────────────────────

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        rungs = list(pd.get("rungs") or [])
        verified = pd.get("verified")
        got = any(r.get("outcome") == "found" for r in rungs)
        v = dict(pd.get("verify_data") or {})

        out = {
            "achieved": bool(got),
            "verified": verified,
            "final_path": pd.get("final_path"),
            "angular_concentration": v.get("angular_concentration"),
            "rungs": rungs,
            # ``rungs_used`` 回答的是「**代价花在哪**」⇒ 只收真的开过的那些。
            # ⚠️ 判据是 **outcome**,不是「有没有这一条记录」:``not_allowed``
            # (forge 没开)、``skipped_budget``(预算用完没开)这些结局**本身就会
            # 记一条 ``rung="r4"``,而那一档一次都没跑**。以前只滤掉了预算那一种,
            # 于是屏幕上出现「阶梯上开过 5 档」而 r4 从未打开 —— 一句把没花的
            # 代价算进账里的话。
            "rungs_used": [r["rung"] for r in rungs
                           if r.get("outcome") not in NEVER_OPENED_OUTCOMES],
            "start_state": pd.get("start_state"),
        }

        reason = pd.get("aborted_reason")
        if reason == "z_controller_off":
            out["advice"] = (
                "**反馈没开、针不在隧道结上** —— 这不是针尖的问题，是我们根本"
                "不在成像状态。本技能不替你进针（进针需要知道样品换没换、粗动在哪、"
                "是不是 4 K，猜错的代价是撞针）。先把针进上，再叫我。")
        elif reason == "coarse_map_exhausted":
            out["advice"] = (
                "**这一片表面用完了**（判据说没有可用台面），而粗动大地图说"
                "**没有下一站**了：%s。"
                "所以我停在这儿，没有接着去修针 —— 表面没有台面时修针是治错的病，"
                "只会白白扎针。通常这意味着**要换样品**。"
                % (pd.get("coarse_note") or "地图没给理由"))
        elif reason == "undecidable_frames":
            out["advice"] = (
                "扫到的每一帧判据都说**判不了**（这不是「没有原子分辨」）—— "
                "所以我**没有动针尖**：在判不了的证据上修针，是在自己造出来的"
                "空白上判读，会把一根本来就好的针磨掉。"
                "先按判据给的 remedy 改成像条件（视野太小就扩、像素太粗就缩、"
                "残帧就重扫、死平区就换点），再跑一次。")
        elif got and verified and verified != "atomic_resolved":
            out["advice"] = (
                "拿到了看起来像原子分辨的一帧，但**验证没过**（%s）。"
                "验证比初判严：它要求正反扫都有晶格、整帧采满、帧内两半不突变。"
                "这一帧多半是针尖产物，或者只是半张帧的运气。" % verified)
        elif not got:
            tail = ""
            if any(r.get("outcome") == "skipped_budget" for r in rungs):
                tail = ("**预算用完了，还有没开的档** —— 时间不是问题的话，"
                        "把 total_budget_min 调大（整夜 = 480，一整天 = 1440）再跑。")
            # ⚠️ 看 **outcome**，不是「有没有 r4 这一条记录」—— ``not_allowed``
            # 本身就会记一条 ``rung="r4"``，用 rung 名判会把这句话判没。
            elif any(r.get("outcome") == "not_allowed" for r in rungs):
                # ⚠️ **不要**把这句写成「下一档是 forge」。ForgeAuTip 的适应症是
                # 「针尖已经变得极其糟糕」，不是「前面几档都失败了」。
                tail = ("ForgeAuTip 默认关闭。仅在明确需要重塑针尖时考虑启用；"
                        "表面区域不足可使用 RelocateCoarseXY，其他情况可重新评估常规处理。")
            out["advice"] = "走完了允许的档位仍没拿到原子分辨。" + tail

        # 跑完了就把结局念出来 —— **成或不成都念,而且都挂分析图**。
        # 放在 aggregate 而不是 plan_dynamic 里,是因为「停下来的理由」有六种
        # (反馈没开 / 判不了 / 地图没有下一站 / 预算 / 不许动针尖 / 走完了),
        # 而它们只有在这里才汇成同一句话 —— 在六个 return 前各写一遍,
        # 就是六份迟早会漂开的实现。
        _narrate_run_result(out, pd)
        return out
