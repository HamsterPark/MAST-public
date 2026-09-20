"""Meta-tools ported from the legacy MissionPlanner to the agent path.

When the main Chat moved from the single-agent ``MissionPlanner`` onto the IC
agent, the planner's non-hardware *meta-tools* (knowledge queries, experiment /
sample bookkeeping, scan-file ops, surface navigation, plan drafting) had to come
with it or the chat would lose capability. These are ported here as ``@tool``
callables bound to a ``provider()`` closure (the same pattern as
``memory_tools.make_memory_tools``) so they can be attached to the IC agent via
``extra_tools`` without crossing the agent boundary, and so the orchestrator's IC
gains them too (parity bonus).

The handler bodies are lifted near-verbatim from ``mast.llm.planner`` — they call
the same pure ``mast.knowledge`` helpers and return JSON-able dicts. Returns are
``json.dumps``'d because the agent tool path wants a string.

NOTE: the planner's plan-EXECUTION tools (``execute_plan`` / ``pause_plan`` /
``resume_plan`` / ``abort_plan``) were tied to the planner's own mission loop and
do NOT have a clean equivalent on the agent path (the IC agent IS the execution
loop — it runs the plan's steps as ordinary skill tool calls). Only plan
*drafting / approval / listing* is ported; the agent is told to follow an
approved plan directly.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Callable

from langchain_core.tools import InjectedToolCallId, tool
from pydantic import BaseModel, Field

from mast.core.coord_epoch import read_current_epoch
from mast.core.si_quantity import SIParseError, parse_quantity

logger = logging.getLogger(__name__)

# provider() -> {
#   "experiment_log": ExperimentLog|None, "plan_store": PlanStore|None,
#   "storage": ExperimentStorage|None,          # scan-map markers live here
#   "map_analysis_cfg": callable()->AnalysisConfig | None,
#   "coarse_map_cfg": callable()->CoarseMapConfig | None,
#   "experiments_dir": str|None, "session_path": str|None, "session_dir": str|None,
# }
Provider = Callable[[], dict]


def _j(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(obj)


# ── plan-coordinate magnitude check () ───────
#
# `show_plan_on_map` took 17 calls carrying 18 impossible `[xywh]_m` values
# (x_m=6.0 → a scan box six METRES away) and answered success every time, so the
# agent kept re-publishing the same nonsense — five show/clear rounds in two
# minutes — because nothing ever told it the values were wrong.
#
# The units are NOT the problem and the message must not say they are: every
# skill parameter in this repo is SI, the `_m` suffix means metres, and `0.5`
# *is* the correct SI spelling of half a metre. What is wrong is the MAGNITUDE —
# half a metre is 3.3e5 × the STM piezo range. We can state that precisely; we
# cannot know what the caller meant instead, so we offer the in-range readings
# as arithmetic ("6 µm would be 6e-06") and explicitly disclaim intent.
#
# Two tiers, because one threshold cannot serve both jobs:
#   • REJECT beyond `_PLAN_POS_ABS_MAX_M` — no STM reading of the number exists.
#   • WARN between the piezo range and that bound — physically conceivable after
#     coarse motion, but ruinous to the display: ScanMapCanvas auto-fits the
#     viewport over EVERY marker, so one distant step stretches the shared span
#     and collapses the real nanometre-scale markers. Measured against the field
#     payload: span 1.9 µm → 8.8 m, and all real markers land on ONE pixel with
#     footprints of 2e-5 px. The route renders fine; everything else vanishes.
#     (That also disproves 推测-2 — absurd coordinates do not fall off-canvas.)

# Fine-piezo XY range, from config.SafetyLimits.xy_min_m / xy_max_m.
_PLAN_PIEZO_XY_M = 1.5e-6
# Hard ceiling for a plan position. 1 mm is already ~670× the piezo range and
# beyond any coarse-motion travel that shares a scan map; past it there is no
# STM interpretation of the number at all.
_PLAN_POS_ABS_MAX_M = 1e-3
# Extent ceiling, from config.SafetyLimits.scan_size_max_m (10 µm — "way bigger
# than any STM scan").
_PLAN_SIZE_ABS_MAX_M = 1e-5


def _plan_magnitude_hint(value: float) -> str:
    """In-range readings of *value* as arithmetic — NOT a diagnosis of intent.

    Deliberately phrased as "if you meant X, write Y". The tool cannot know what
    the caller intended, and asserting it can is how a sibling guard ended up
    telling operators they had "dropped an exponent" on values they had written
    correctly.
    """
    alts = []
    for unit, factor in (("µm", 1e-6), ("nm", 1e-9)):
        scaled = value * factor
        # Bound by what this tool would ACCEPT, not by the piezo range: a step
        # a few µm out is a legitimate post-coarse-motion plan, so offering it
        # is useful. Readings that would themselves be rejected are not.
        if 0 < abs(scaled) <= _PLAN_POS_ABS_MAX_M:
            alts.append(f"若本意是 {value:g} {unit},应写 {scaled:.3g}")
    if not alts:
        return ""
    return "(" + ";".join(alts) + " —— 仅为换算参考,本工具无法判定你的本意)"


#: Every metre-valued argument name the map tools take, split by what it is
#: judged against. Keyed by EXACT name, not by suffix: ``radius_m`` is a size
#: and ``y_m`` is a position, and no rule over the ``_m`` suffix separates them.
#:
#: A name absent from both lists is not silently skipped — it fails the
#: structural gate in tests/v2/unit/agents/test_map_tool_magnitudes.py, which
#: is the point. Before 2026-08-09 the judgement ran over a FIXED four-name
#: tuple inlined below, so ``radius_m`` and ``frame_size_m`` were not merely
#: unguarded, they were **invisible to the checker even when handed to it**.
_METRE_POSITION_ARGS = ("x_m", "y_m", "z_m")
#: ``min_separation_m`` 加于 2026-08-10:它在 ``data_processing.find_flat_region``
#: 上,而当时的结构闸门只梳 ``meta_tools.py`` 这一个文件 —— 名字没被分类,护栏也
#: 没接。判成「尺寸」是因为它是一帧之内两点的间距(上限 10 µm),不是绝对坐标。
_METRE_SIZE_ARGS = ("w_m", "h_m", "radius_m", "frame_size_m", "size_m",
                    "min_separation_m", "min_window_m")


def _check_metre_magnitudes(tag: str, values: dict) -> tuple[list[str], list[str]]:
    """Judge a set of metre-valued fields → (errors, warnings).

    Split out of :func:`_check_plan_step_magnitudes` on 2026-08-09, after the
    agent handed ``get_next_scan_position`` a ``frame_size_m`` of
    ``5.00000058430487`` — five metres, an exponent lost on the way through a
    number-typed parameter — and the tool **accepted it** and answered
    「当前策略下已无可用位置」. That sentence is a perfectly ordinary thing for
    that tool to say. Nothing looked wrong. Same family as the SI-prefix work on
    the instrument skills, except the map tools never got the defence: a size
    eight orders of magnitude too large simply loses to every candidate.

    ``None`` and unparseable values are skipped (a tool's own signature already
    types them); NaN/inf are errors, because they pass ``float()`` and then
    poison every comparison downstream.
    """
    errors: list[str] = []
    warnings: list[str] = []
    fields = [(n, _PLAN_POS_ABS_MAX_M, "位置") for n in _METRE_POSITION_ARGS]
    fields += [(n, _PLAN_SIZE_ABS_MAX_M, "尺寸") for n in _METRE_SIZE_ARGS]
    for field, ceiling, what in fields:
        raw = values.get(field)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
            errors.append(f"{tag} {field}={raw!r} 不是有限数值。")
            continue
        a = abs(v)
        if a > ceiling:
            ratio = a / _PLAN_PIEZO_XY_M if what == "位置" else a / _PLAN_SIZE_ABS_MAX_M
            errors.append(
                f"{tag} {field}={v:g} 即 {v:g} 米 —— 单位没错(本工具的 _m 后缀就是"
                f"米,SI),错的是量级:STM 压电工作区约 ±{_PLAN_PIEZO_XY_M:g} m"
                f"(±1.5 µm),该{what}超出约 {ratio:.3g} 倍。"
                + _plan_magnitude_hint(v)
            )
        elif what == "位置" and a > _PLAN_PIEZO_XY_M:
            warnings.append(
                f"{tag} {field}={v:g} m 超出压电量程 ±{_PLAN_PIEZO_XY_M:g} m"
                f"(约 {a / _PLAN_PIEZO_XY_M:.3g} 倍) —— 只有粗动过才可能;"
                "地图会自动缩放到这个跨度,已执行的纳米级标记会被压成一个点。"
            )
    return errors, warnings


def _check_plan_step_magnitudes(idx: int, step: dict) -> tuple[list[str], list[str]]:
    """Validate one plan step's metre-valued fields → (errors, warnings).

    A thin shell over :func:`_check_metre_magnitudes` since 2026-08-09 — one
    judgement, two callers, so a name added for the map tools is also judged
    here and vice versa. The step tag (index + label) is this caller's only
    private business.
    """
    label = str(step.get("label") or "").strip()
    tag = f"第 {idx} 步" + (f"({label})" if label else "")
    return _check_metre_magnitudes(tag, step)


# ── 米量纲参数的解析与量级护栏 ─────────────────────────────────
# 数值丢失指数后可能仍能通过类型检查，因此解析和物理量级校验都要存在。
# skill 侧将量纲参数声明为字符串，由 `_coerce_si_params` 解析 SI 前缀，
# 以避免指数丢失后尾数仍被接受为合法数字。@tool 是另一条注册路径，
# 因此必须接入同一套解析机制，不能只覆盖 ParameterSpec。
#
# 这里补的就是那一半,**复用同一个解析器** `mast.core.si_quantity.parse_quantity`
# —— 不是新写一个:同一个物理量有两套解析迟早各自漂移。
#
# 两个刻意的选择:
#
# 1. 类型是 `float | str` 而不是纯 `str`。纯 `str` 会让 pydantic **拒绝**数值形式
#    (实测:`str` 字段收到 5e-08 → ValidationError),而 `5e-8` 今天是能用的正确
#    写法,不能因为加了新写法就把旧写法弄坏。实测 `float | str` 在 payload 里是
#    `anyOf:[number,string]`,两种形式都到得了函数体。
#
# 2. 解析用 `strict=False`(前缀可省),不是 skill 那边米量纲的 `strict=True`。
#    理由不是"松一点方便",而是这条路上**前缀校验和的活已经有人在干了**:
#    strict 的价值是「掉了前缀 = 解析失败」,而这里掉了前缀的 '50n' → '50' → 50 米,
#    正好落进 `_check_metre_magnitudes` 的拒绝区(尺寸 >1e-5 m、位置 >1e-3 m 一律拒)。
#    量级护栏对米量纲比前缀规则更强:它不管你怎么写的,五米就是五米。
#    反过来 strict=True 会**弄坏今天能用的写法** —— 字符串 '5e-8' 现在被 pydantic
#    宽松转成 5e-8 是通的,strict 会把它拒掉。见
#    tests/v2/unit/agents/test_map_tool_magnitudes.py::test_strict_prefix_would_have_broken_a_working_form
#    (这条测试就是为了钉住这个被否掉的方案,否则它会被当成"更安全"重新引入)。

#: 这段话会**进 payload**:实测 `Annotated[..., Field(description=...)]` 能穿过
#: `convert_to_openai_tool`(而 `json_schema_extra` 会被它丢掉,见 skill_adapter 里
#: 同一条教训)。`@tool` 的 docstring 是**函数级**描述,参数级只有这一条通道。
_METRE_HOWTO = (
    "可以写成**带 SI 前缀的字符串**:50 nm 写 '50n',200 nm 写 '200n',1 µm 写 '1u'。"
    "数值形式(5e-8)同样接受。前缀形式更不容易出错 —— '50n' 掉了前缀会当场解析失败,"
    "而 5e-8 掉了指数会变成 5,那是「五米」,一个仍然合法、不会被察觉的数字。"
)

#: 位置 / 尺寸两个别名,只为让模型看到的典型量级贴着它正在填的那个参数。
MetrePos = Annotated[float | str, Field(description="位置坐标,米量纲。" + _METRE_HOWTO)]
MetreSize = Annotated[float | str, Field(description="尺寸/半径,米量纲。" + _METRE_HOWTO)]


def resolve_metre_args(tag: str, values: "dict[str, Any]") -> "tuple[dict, list[str]]":
    """把米量纲参数解析成 float **并**判量级 → ``(parsed, errors)``。

    公开名(没有下划线)是因为 ``agents/data_processing/tools.py`` 也要用 ——
    ``find_flat_region(min_separation_m)`` 是同一条 ``@tool`` 路径上的同一个洞。
    宁可跨模块导入一个公开名,也不要第二份实现。

    **两半必须一次调完,顺序也是承重的**:``_check_metre_magnitudes`` 里读的是
    ``float(raw)``,而 ``float('50n')`` 会抛 —— 它的 except 分支是 ``continue``,
    也就是说一个**没解析过的字符串会被护栏静静跳过**。先解析后判,不然「加上
    字符串写法」这件事本身就会在护栏上开一个洞,而那个洞正是字符串写法要堵的。

    解析失败**只报错,绝不回退**到默认值或 0。回退回来的是一个能跑的错版本:
    ``'50nm'``(多写了个 m)悄悄变成 50e-9 的默认半径,和写对了看起来一模一样。
    """
    parsed: dict = {}
    errors: list[str] = []
    for name, raw in values.items():
        if raw is None:
            parsed[name] = None
            continue
        try:
            parsed[name] = parse_quantity(raw, strict=False, what=name)
        except SIParseError as exc:
            errors.append(str(exc))
    if errors:
        # 解析都没成功就别再判量级 —— 半个字典判出来的错只会盖住真正的原因。
        return parsed, errors
    errs, _warn = _check_metre_magnitudes(tag, parsed)
    return parsed, errs


# — why this is a MODEL and not `steps: list`.
#
# The untyped signature sent the provider `{"steps": {"type": "array"}}` and
# nothing else: no field names, no units, no bounds. The same agent, in the same
# turn, wrote `FullScan(center_x_m=1.4e-06, height_m=5e-08)` correctly — those
# params are typed floats WITH bounds — and `{"x_m": 1.4, "w_m": 5.0}` here. It
# concluded the exponent was being "eaten in transit", retried twice in plain
# decimal, got it wrong again, and the route never reached the map.
#
# Same lesson as the skill parameters: a constraint only reaches the model if it
# is IN THE PAYLOAD.
#
# The bounds are `json_schema_extra`, NOT `ge`/`le`, and the difference matters.
# At the TOP level of a tool signature `convert_to_openai_tool` drops
# `json_schema_extra` (which is why skill parameters must use `ge`/`le`); inside
# a nested model the schema comes from `model_json_schema()`, where it survives
# as `minimum`/`maximum` — verified against the installed langchain_core.
#
# Using `ge`/`le` here would ALSO make pydantic reject the call before the body
# runs, and that is the wrong owner: `_check_plan_step_magnitudes` phrases this
# rejection deliberately — it names the step by index AND label, quantifies how
# far out the value is, offers the µm/nm readings as arithmetic, and explicitly
# disclaims knowing what the caller meant ( pinned by
# test_show_plan_on_map_magnitude.py). A pydantic "Input should be less than or
# equal to 1e-05" would replace all of that with a true, useless sentence.
#
# So: the schema TELLS the model the scale, the checker JUDGES the call.
#
# NB the docstring below is sent to the MODEL (it becomes the item schema's
# `description`), so it says what the model needs, not what a maintainer does.
class PlanStep(BaseModel):
    """计划路线上的一步。坐标一律用米(SI),与 FullScan 的 center_x_m 同一套写法。"""

    kind: str = Field(
        default="plan",
        description="scan|sts|pulse|tip_shape|move —— 该步要做什么")
    x_m: float = Field(
        json_schema_extra={"minimum": -_PLAN_POS_ABS_MAX_M,
                           "maximum": _PLAN_POS_ABS_MAX_M},
        description="X 位置,单位米(SI)。STM 压电工作区约 ±1.5e-6 m,"
                    "所以典型值形如 1.4e-06,不是 1.4")
    y_m: float = Field(
        json_schema_extra={"minimum": -_PLAN_POS_ABS_MAX_M,
                           "maximum": _PLAN_POS_ABS_MAX_M},
        description="Y 位置,单位米(SI)。同 x_m")
    w_m: float | None = Field(
        default=None,
        json_schema_extra={"minimum": -_PLAN_SIZE_ABS_MAX_M,
                           "maximum": _PLAN_SIZE_ABS_MAX_M},
        description="扫描框宽度,单位米。100 nm 写 1e-07,50 nm 写 5e-08")
    h_m: float | None = Field(
        default=None,
        json_schema_extra={"minimum": -_PLAN_SIZE_ABS_MAX_M,
                           "maximum": _PLAN_SIZE_ABS_MAX_M},
        description="扫描框高度,单位米。留空则取 w_m")
    label: str = Field(default="", description="地图上显示的短标签,如 '+1V 50nm'")


def _material_miss(query: str) -> str:
    """The ONE consistent "no such material/sample type" response.

    The three enum paths used to disagree — one rejected free text with no
    candidates, one silently accepted anything, one listed candidates. Every
    material lookup that cannot proceed now returns THIS: a clear miss PLUS the
    controlled-vocabulary candidate list, so the model can re-issue a valid one
    instead of guessing (or the caller can fall back to free text knowingly)."""
    try:
        from mast.knowledge import (
            list_material_candidates,
            material_candidates_hint,
        )
        cands = list_material_candidates()
        hint = material_candidates_hint()
    except Exception:  # noqa: BLE001
        cands, hint = [], ""
    return _j({
        "success": False,
        "error": f"无匹配的材料/样品类型: {query!r}。{hint}",
        "candidates": cands,
    })


def make_meta_tools(provider: Provider) -> list:
    """Build the ported planner meta-tools bound to *provider*."""

    def _ctx() -> dict:
        try:
            return provider() or {}
        except Exception as exc:  # pragma: no cover
            logger.debug("meta-tool provider failed: %s", exc)
            return {}

    # ── experiment / sample bookkeeping ───────────────────────────────
    @tool("start_experiment")
    def start_experiment(name: str, goal: str = "") -> str:
        """开始/继续一个实验会话,后续动作都记录到该实验。name 必填,goal 可选。

        幂等:若已有同名的活动实验(本会话早先建的,或会话恢复后仍在进行的),
        会【复用】它而不是新建重复实验(避免同一研究反复建出多个实验记录)。
        返回里 reused=true 表示复用了既有实验。要开全新实验请换一个名字。"""
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        try:
            exp_id = el.start_experiment(name, goal, reuse_open=True)
            reused = bool(getattr(el, "_last_start_reused", False))
            out = {"success": True, "experiment_id": exp_id, "name": name,
                   "reused": reused}
            if reused:
                out["message"] = ("已存在同名活动实验,已复用该实验(未新建重复实验)。"
                                  "如需全新实验请改用不同名称。")
            return _j(out)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("end_experiment")
    def end_experiment() -> str:
        """取消选中当前实验(不会结束或归档它)。

        实验是永久的,随时可以再切回来继续做——**没有「结束实验」这回事**。
        这个工具只是把「当前实验」置空,通常你不需要调用它:要换个实验做,
        直接 start_experiment("另一个实验名") 即可。
        """
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        try:
            el.end_experiment()
            return _j({"success": True,
                       "message": "已取消选中当前实验(实验本身仍然存在,随时可切回)"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("start_sample")
    def start_sample(name: str, description: str = "", sample_type: str = "",
                     sample_subtype: str = "") -> str:
        """在当前实验下开始一个样品并切过去。给出 sample_type 会注入工作流建议。

        同名样品会被【复用】而不是新建重复的——上一个样品不会被结束,
        它随时可以被切回来继续用(STM 样品经常换回去)。
        """
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        type_inferred = True
        if not sample_type and not sample_subtype:
            try:
                from mast.knowledge import match_material
                m = match_material(name)
                if m:
                    sample_type, sample_subtype = m
                else:
                    type_inferred = False
            except Exception:  # noqa: BLE001
                type_inferred = False
        try:
            sid = el.start_sample(name, description, sample_type=sample_type,
                                  sample_subtype=sample_subtype, reuse_active=True)
            reused = bool(getattr(el, "_last_sample_reused", False))
            out = {"success": True, "sample_id": sid, "name": name, "reused": reused}
            if sample_type:
                out["sample_type"] = sample_type
            if sample_subtype:
                out["sample_subtype"] = sample_subtype
            if reused:
                out["message"] = "已存在同名活动样品,已复用(未新建重复样品)。"
            # Enum consistency: free text is ACCEPTED (the sample is created), but
            # when we could not normalise it to a known sample_type we return the
            # controlled-vocabulary candidates so the model can set a canonical
            # type (via rename/restart) and unlock workflow guidance — same
            # candidate list the material knowledge tools return on a miss.
            if not sample_type and not type_inferred:
                try:
                    from mast.knowledge import list_material_candidates
                    out["sample_type_candidates"] = list_material_candidates()
                    out["note"] = ("未能从名称归一出已知 sample_type(已按自由文本创建"
                                   "样品)。如需注入工作流建议,可从 sample_type_candidates "
                                   "选一个受控类型。")
                except Exception:  # noqa: BLE001
                    pass
            return _j(out)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("end_sample")
    def end_sample() -> str:
        """取消选中当前样品(不会结束它)。样品被物理取下、还没装新的时候用。

        样品是永久的,随时可以切回来继续用——**没有「结束样品」这回事**。
        要换一块样品做,直接 start_sample("另一块样品名")。
        """
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        try:
            el.end_sample()
            return _j({"success": True, "message": "样品已结束"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    # ── 切换（2026-07-28）─────────────────────────────────────────
    #
    # 实验和样品都是永久的、可以来回切的。模型必须能自己切回一个旧实验 ——
    # 常见的场景是:做了一个月这个,又要切回去做那个。
    #
    # 这几个工具**必须接受名字**：模型永远拿不到 UUID（它看不见 DB）。歧义时
    # 返回候选列表让它去问用户，绝不猜。

    def _pick_experiment(el, key: str):
        """按 id 或名字找实验。返回 (row, candidates)。"""
        st = el._storage
        k = (key or "").strip()
        if not k:
            return None, []
        row = st.get_experiment(k)
        if row:
            return row, []
        target = k.casefold()
        rows = st.list_experiments_recent(limit=200)
        exact = [r for r in rows if (r.get("name") or "").strip().casefold() == target]
        if len(exact) == 1:
            return exact[0], []
        if len(exact) > 1:
            return None, exact
        partial = [r for r in rows if target in (r.get("name") or "").casefold()]
        if len(partial) == 1:
            return partial[0], []
        return None, partial[:8]

    @tool("list_experiments")
    def list_experiments(limit: int = 20) -> str:
        """列出实验,按【上次活动时间】倒序(最近动过的在前)。

        用它来找一个要切回去的旧实验 —— 实验永久存在,没有被关闭这回事。
        """
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        try:
            rows = el._storage.list_experiments_recent(limit=max(1, min(int(limit), 100)))
            cur = el.current_experiment_id
            return _j({"experiments": [{
                "name": r.get("name"), "goal": r.get("goal_text") or "",
                "samples": r.get("sample_count"), "actions": r.get("action_count"),
                "last_active_at": r.get("last_active_at") or r.get("start_time"),
                "is_current": r.get("id") == cur,
            } for r in rows]})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("list_samples")
    def list_samples() -> str:
        """列出当前实验下的所有样品(含已经不在用的 —— 样品可以切回来继续用)。"""
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        eid = el.current_experiment_id
        if not eid:
            return _j({"success": False, "error": "当前没有实验,先 start_experiment 或 switch_experiment"})
        try:
            cur = el.current_sample_id
            return _j({"samples": [{
                "name": s.get("name"), "type": s.get("sample_type") or "",
                "description": s.get("description") or "",
                "last_active_at": s.get("last_active_at") or s.get("start_time"),
                "is_current": s.get("id") == cur,
            } for s in el._storage.get_samples(eid)]})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("switch_experiment")
    def switch_experiment(name_or_id: str) -> str:
        """切换到一个【已存在】的实验(按名字或 id)。不会结束任何实验。

        切过去之后会自动落到"上次在那个实验里用的那块样品"。
        名字有歧义时返回候选列表 —— 那时请问用户要哪一个,不要猜。
        """
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        try:
            row, cands = _pick_experiment(el, name_or_id)
            if row is None:
                if cands:
                    return _j({"success": False, "error": "名字有歧义,请让用户确认",
                               "candidates": [c.get("name") for c in cands]})
                return _j({"success": False, "error": f"找不到实验「{name_or_id}」",
                           "hint": "先调 list_experiments 看看有哪些"})
            sc = el.switch_experiment(row["id"], source="agent")
            return _j({"success": sc.ok, "changed": sc.changed,
                       "experiment": sc.experiment_name, "sample": sc.sample_name or None,
                       "error": sc.error or None})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("switch_sample")
    def switch_sample(name_or_id: str) -> str:
        """切换到当前实验下一个【已存在】的样品(按名字或 id)。不会结束任何样品。

        STM 样品经常换回去继续测 —— 用这个,而不是 start_sample 重新建一个同名的。
        """
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        eid = el.current_experiment_id
        if not eid:
            return _j({"success": False, "error": "当前没有实验"})
        try:
            st = el._storage
            k = (name_or_id or "").strip()
            row = st.get_sample(k) or st.find_sample_by_name(eid, k)
            if row is None:
                names = [s.get("name") for s in st.get_samples(eid)]
                return _j({"success": False, "error": f"找不到样品「{k}」",
                           "available": names})
            sc = el.switch_sample(row["id"], source="agent")
            return _j({"success": sc.ok, "changed": sc.changed,
                       "experiment": sc.experiment_name, "sample": sc.sample_name,
                       "error": sc.error or None})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("clear_sample")
    def clear_sample() -> str:
        """取消选中当前样品(样品被物理取下、还没装新的时候用)。

        不会结束样品 —— 它随时可以被切回来。注意:取消之后扫描/谱学会被拦住,
        直到重新选一个样品。
        """
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        try:
            sc = el.clear_sample(source="agent")
            return _j({"success": sc.ok, "changed": sc.changed,
                       "message": "已取消选中样品(产数据的操作在重新选样品前会被拦住)"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("rename_experiment")
    def rename_experiment(name: str) -> str:
        """重命名【当前 MAST 实验】(records 里的实验名,不是 Nanonis 的 Experiment 字段)。name 为新名称。"""
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        try:
            ok = el.rename_experiment(name)
            return _j({"success": bool(ok), "name": name}
                      if ok else {"success": False, "error": "无当前实验或未更新"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("rename_sample")
    def rename_sample(name: str) -> str:
        """重命名【当前 MAST 样品】(records 里的样品名,不是 Nanonis 的 Sample 字段)。name 为新名称。"""
        el = _ctx().get("experiment_log")
        if el is None:
            return "实验记录不可用"
        try:
            ok = el.rename_sample(name)
            return _j({"success": bool(ok), "name": name}
                      if ok else {"success": False, "error": "无当前样品或未更新"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    # ── scan files ─────────────────────────────────────────────────────
    @tool("get_latest_scan_info")
    def get_latest_scan_info() -> str:
        """返回最近一张扫描文件(.sxm)的路径。"""
        ctx = _ctx()
        try:
            from mast.webui.scan_preview import get_latest_scan
            path = get_latest_scan(ctx.get("experiments_dir"),
                                   ctx.get("session_path"), ctx.get("session_dir"))
            if path:
                return _j({"success": True, "path": path})
            return _j({"success": True, "path": None, "message": "无扫描文件"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("load_scan_file")
    def load_scan_file(path: str) -> str:
        """离线加载 Nanonis 扫描文件(.sxm/.3ds/.dat),返回通道与头信息摘要(不返回原始数组)。"""
        try:
            from mast.io.nanonis_files import load_scan_file as _load
            data = _load(path)
            summary: dict = {"success": True, "file": path}
            if "channels" in data:
                summary["channels"] = list(data["channels"].keys())
            if "header" in data:
                summary["header_keys"] = list(data["header"].keys())
            if "columns" in data:
                summary["columns"] = list(data["columns"].keys())
            if "grid" in data and hasattr(data["grid"], "shape"):
                summary["grid_shape"] = list(data["grid"].shape)
            return _j(summary)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    # ── knowledge ──────────────────────────────────────────────────────
    @tool("get_workflow_advice")
    def get_workflow_advice(query: str) -> str:
        """查询某样品类型/材料的推荐实验工作流(阶段、参数、成功/质量标准、常见问题)。"""
        try:
            from mast.knowledge import format_conceptual_for_llm, match_material
            m = match_material(query)
            if m:
                tid, mat = m
                return _j({"success": True, "advice": format_conceptual_for_llm(tid, mat)})
            return _material_miss(query)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("get_skill_guidance")
    def get_skill_guidance(skill_name: str) -> str:
        """获取某技能的专家指导:何时用、何时不用、相关技能、决策上下文、工作流配方。"""
        try:
            from mast.knowledge import format_skill_guidance_for_llm
            g = format_skill_guidance_for_llm(skill_name)
            if g:
                return _j({"success": True, "guidance": g})
            return _j({"success": False, "error": f"无该技能指导: {skill_name}"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("get_literature_parameters")
    def get_literature_parameters(material: str, phase: str = "") -> str:
        """查询文献中该材料某实验阶段的推荐参数(未校准,使用前须向用户确认)。"""
        try:
            from mast.knowledge import format_literature_params, match_material
            m = match_material(material)
            if m:
                tid, mat = m
                return _j({"success": True,
                           "parameters": format_literature_params(tid, mat, phase)})
            return _material_miss(material)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("get_fault_diagnosis")
    def get_fault_diagnosis(symptom: str) -> str:
        """根据症状查询故障诊断库(原因/诊断方法/修复建议)。"""
        try:
            from mast.knowledge.fault_diagnosis import format_fault_for_llm, match_faults
            ms = match_faults(symptom, top_n=5)
            if ms:
                return _j({"success": True, "matches": len(ms),
                           "diagnosis": "\n\n".join(format_fault_for_llm(f) for f in ms)})
            return _j({"success": False, "error": f"未找到匹配故障: {symptom}"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("get_noise_reference")
    def get_noise_reference(frequency_hz: float | None = None,
                            noise_type: str | None = None) -> str:
        """根据频率或噪声类型查询 STM 噪声目录(频率-源映射、耦合路径、关联故障)。"""
        try:
            from mast.knowledge.stm_noise import (
                format_noise_entry_for_llm, lookup_by_frequency, lookup_by_type,
            )
            entries: list = []
            if frequency_hz is not None:
                entries = lookup_by_frequency(float(frequency_hz))
            elif noise_type:
                entries = lookup_by_type(noise_type)
            if entries:
                return _j({"success": True, "matches": len(entries),
                           "reference": "\n\n".join(format_noise_entry_for_llm(e) for e in entries)})
            hint = f"freq={frequency_hz}" if frequency_hz is not None else f"type={noise_type}"
            return _j({"success": False, "error": f"未找到匹配噪声源: {hint}"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("get_measurement_template")
    def get_measurement_template(measurement_type: str) -> str:
        """获取测量模板(推荐参数+成功标准+技能链):kondo_sts/qpi_mapping/sc_gap/sp_stm/band_gap 等。"""
        try:
            from mast.knowledge.skill_guidance import (
                MEASUREMENT_TEMPLATES,
                format_measurement_template_for_llm as _fmt,
                get_measurement_template as _get,
            )
            tmpl = _get(measurement_type)
            if tmpl:
                key = measurement_type
                if key not in MEASUREMENT_TEMPLATES:
                    for k, v in MEASUREMENT_TEMPLATES.items():
                        if v is tmpl:
                            key = k
                            break
                return _j({"success": True, "template": _fmt(key)})
            # Consistent miss: always return the controlled candidate list (here
            # the measurement-template keys, this tool's own vocabulary).
            cands = sorted(MEASUREMENT_TEMPLATES.keys())
            return _j({"success": False,
                       "error": (f"未找到测量模板: {measurement_type!r}。"
                                 f"可用模板: {'、'.join(cands)}"),
                       "candidates": cands})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("search_deep_reference")
    def search_deep_reference(query: str) -> str:
        """搜索 MAST-reference 深度文档的章节索引(返回标题+摘要,不加载全文)。"""
        try:
            from mast.knowledge.reference_index import search_sections
            results = search_sections(query, top_n=5)
            if results:
                lines = [f"[{r['report_id']}] {r['heading']} — {r['summary']} "
                         f"(lines {r['lines'][0]}-{r['lines'][1]})" for r in results]
                return _j({"success": True, "sections": "\n".join(lines)})
            return _j({"success": False, "error": f"未找到匹配章节: {query}"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("read_reference_section")
    def read_reference_section(report_id: str, heading: str) -> str:
        """读取 MAST-reference 某章节原文(先用 search_deep_reference 获取 report_id 和 heading)。"""
        try:
            from mast.knowledge.reference_index import read_section
            text = read_section(report_id, heading)
            if text:
                return _j({"success": True, "content": text})
            return _j({"success": False, "error": f"未找到章节: {report_id}/{heading}"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("query_knowledge")
    def query_knowledge(query: str, detail_level: str = "conceptual") -> str:
        """查询 MAST 知识库(自然语言)。detail_level: conceptual|parameters|full 控制返回深度。"""
        cfg = {"conceptual": (4, 400), "parameters": (8, 1200), "full": (10, None)}
        top_k, cap = cfg.get(detail_level, cfg["conceptual"])
        try:
            from mast.knowledge import get_retriever
            results = get_retriever().query(query, top_k=top_k)
            if results:
                parts = []
                for chunk, _score in results:
                    try:
                        body = chunk.content()
                    except Exception:  # noqa: BLE001
                        continue
                    if cap is not None and len(body) > cap:
                        body = body[:cap].rstrip() + " …[truncated]"
                    parts.append(body)
                return _j({"success": True, "results": len(parts),
                           "detail_level": detail_level, "knowledge": "\n\n".join(parts)})
            return _j({"success": False, "error": f"未找到匹配知识: {query}"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    # ── surface navigation (from the scan map's own record) ────────────
    #
    # These read the recorded map markers and compute the answer. They replaced a
    # separate in-memory grid (SurfaceNavigator) that tracked "used" and
    # "forbidden" areas ONLY when the agent remembered to declare them, held no
    # relation to the footprints actually scanned, and forgot everything on
    # restart. The map is the record; deriving from it means the operator and the
    # agent are looking at the same surface.

    def _map_scope() -> tuple[Any, str | None, str | None]:
        """(storage, experiment_id, sample_id) for the current scope."""
        ctx = _ctx()
        storage = ctx.get("storage")
        el = ctx.get("experiment_log")
        exp_id = getattr(el, "current_experiment_id", None) if el else None
        sample_id = getattr(el, "current_sample_id", None) if el else None
        return storage, exp_id, sample_id

    def _map_cfg(strategy: str = "", frame_size_m: float | None = None):
        """AnalysisConfig from the rig profile, with optional overrides."""
        builder = _ctx().get("map_analysis_cfg")
        cfg = builder() if callable(builder) else None
        if cfg is None:
            from mast.io.map_analysis import AnalysisConfig
            cfg = AnalysisConfig()
        from dataclasses import replace as _replace
        over: dict = {}
        s = (strategy or "").strip().lower()
        # spiral/nearest were the old SurfaceNavigator's strategy names; treat
        # them as "no preference" rather than erroring at an agent that learned
        # them from an older prompt.
        if s in ("center_first", "perimeter_inward"):
            over["strategy"] = s
        if frame_size_m and frame_size_m > 0:
            over["frame_size_m"] = float(frame_size_m)
        return _replace(cfg, **over) if over else cfg

    def _load_map_rows(all_epochs: bool = False):
        """Marker rows for the analysis, plus the live generation."""
        storage, exp_id, sample_id = _map_scope()
        if storage is None:
            return None, 0
        rows = storage.get_markers(exp_id, sample_id)
        epoch = storage.current_epoch(exp_id, sample_id)
        return rows, epoch

    @tool("get_map_analysis")
    def get_map_analysis() -> str:
        """程序化分析扫描地图:扫过多少、哪些区域被破坏、还剩多少可用面积、
        下一个该扫哪里、是否该粗动换区、做了多少谱。

        【这是判断「扫了哪 / 哪不能去 / 下一步去哪 / 该不该换区」的唯一依据。
        不要凭截图、印象或推理去猜这些结论 —— 它们由程序按记录算出。】

        返回的百分比都是相对整个压电可达范围。coverage=已扫面积占比;
        usable=未被破坏的面积占比(注意:已扫过 ≠ 不可用,回到干净的旧区域做谱是正常的);
        usable_unscanned=既没被破坏、也还没扫过的面积占比 —— 这才是「还有多少新地方」。
        coord_epoch 是坐标代次:XY 粗动会让旧坐标全部失效,分析只看当前代次。"""
        try:
            from mast.io.map_analysis import analyze_map
            rows, epoch = _load_map_rows()
            if rows is None:
                return _j({"success": False, "error": "实验记录存储不可用,无法分析扫描地图"})
            # Hand analyze_map the AUTHORITATIVE generation. get_markers() returns
            # at most `limit` newest rows; counting coarse_move inside that window
            # under-reports the generation once the scope outgrows it, and the
            # analysis then silently describes an older patch of surface.
            res = analyze_map(rows, _map_cfg(), plan_markers=_plan_steps_for_map(),
                              current_epoch=epoch)
            out = {
                "success": True,
                "coord_epoch": res.current_epoch,
                "markers_total": res.markers_total,
                "markers_current_epoch": res.markers_current_epoch,
                "coverage_pct": round(res.coverage_frac * 100, 2),
                "usable_pct": round(res.usable_frac * 100, 2),
                "usable_unscanned_pct": round(res.usable_unscanned_frac * 100, 2),
                "strategy": res.strategy,
                "avoid_zones": res.damage_counts,
                "sts_points": res.sts_total,
                "coarse_move": {"suggest": res.coarse_advice.suggest,
                                "reasons": res.coarse_advice.reasons},
                **res.survey,
            }
            if res.next_position is not None:
                out["next_position"] = {
                    "x_m": res.next_position.x_m, "y_m": res.next_position.y_m,
                    "reason": res.next_position.reason,
                    "candidates_left": res.next_position.candidates_left,
                }
            else:
                out["next_position"] = None
                out["next_position_note"] = "当前策略下已无可用位置"
            if res.route_truncated:
                out["route_truncated"] = True
            return _j(out)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("get_next_scan_position")
    def get_next_scan_position(strategy: str = "auto",
                               frame_size_m: MetreSize = 0.0) -> str:
        """从扫描地图记录推导下一个建议扫描位置,自动避开修针尖/电脉冲/撞针/进针
        造成的破坏区,以及已经扫过的区域。

        strategy: auto(按仪器能力自动选) | center_first(中心优先,压电蠕变最小,
        适合能粗动换区的机器) | perimeter_inward(外圈→内圈,适合不能 XY 位移、
        必须省着用面积的机器)。
        frame_size_m: 打算用多大的扫描框,写成带 SI 前缀的字符串,如 '50n'
        (=50 nm)、'200n'、'1u';留 0 用默认。

        位置耗尽时返回 x_m=null 并附上粗动换区建议。"""
        try:
            # 先解析带 SI 前缀的尺寸，再走物理量级护栏。
            # 异常的大尺寸会排除所有候选，不能把输入错误报告为无可用位置。
            _q, errors = resolve_metre_args(
                "get_next_scan_position", {"frame_size_m": frame_size_m})
            if errors:
                return _j({"success": False, "error": " ".join(errors)})
            frame_size_m = _q["frame_size_m"]
            from mast.io.map_analysis import pick_next_position, analyze_map
            rows, epoch = _load_map_rows()
            if rows is None:
                return _j({"success": False, "error": "实验记录存储不可用"})
            cfg = _map_cfg(strategy, frame_size_m or None)
            res = analyze_map(rows, cfg, current_epoch=epoch)
            if res.next_position is None:
                return _j({"success": True, "x_m": None, "y_m": None,
                           "strategy": cfg.strategy,
                           "message": "当前策略下已无可用位置(全被避让区占据或已扫过)。",
                           "coarse_move": {
                               "suggest": res.coarse_advice.suggest,
                               "reasons": res.coarse_advice.reasons}})
            p = res.next_position
            return _j({"success": True, "x_m": p.x_m, "y_m": p.y_m,
                       "strategy": p.strategy, "reason": p.reason,
                       "candidates_left": p.candidates_left,
                       "coord_epoch": res.current_epoch})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("mark_area_used")
    def mark_area_used(x_m: MetrePos, y_m: MetrePos, forbidden: bool = False,
                       radius_m: MetreSize = 50e-9) -> str:
        """在扫描地图上人工标一块区域。

        forbidden=true → 标为**避让区**(污染/损坏,选点永久绕开);
        forbidden=false → 标为**已用**(选点视同已扫过,但不禁止再去)。

        写进实验记录,所以人和 AI 看到的是同一份、跨重启不丢。
        坐标和半径是**米**量纲,写成带 SI 前缀的字符串,STM 尺度典型量级 '100n'。"""
        try:
            from mast.io.map_analysis import META_AVOID_RADIUS, META_USED_RADIUS
            # 参数先判,存储后查 —— 顺序是承重的。
            #
            # 原来是先查 storage:没接上记录库时,一个「半径五米」的调用得到的回答
            # 是「实验记录存储不可用」。那句话是真的,但它回答的不是被问的那个问题
            # ([[evidence_answers_wrong_question]]):模型会去接存储,然后拿同一个
            # 五米重试一次。参数错在有没有存储之前就已经错了。
            #
            # 这是三个地图工具里唯一**写**的那个,所以量级错在这里最贵:一个半径
            # 五米的「避让区」会写进实验记录,从此整张地图永久「已无可用位置」,
            # 而且坏的是数据不是一次调用 —— 读工具错了重来一次就行,这个不行。
            # 另外坐标一旦落库,ScanMapCanvas 会把视口自动撑到覆盖它,把所有真实
            # 纳米级标记压成一个像素。
            # radius_m 也要送:它一直被传进来,但 2026-08-09 之前的检查按固定四个
            # 字段名判,所以它送到了也看不见 —— 一个「有护栏」的调用里的第三个洞。
            _q, errors = resolve_metre_args(
                "人工标记", {"x_m": x_m, "y_m": y_m, "radius_m": radius_m})
            if errors:
                return _j({"success": False, "error": "；".join(errors)})
            x_m, y_m, radius_m = _q["x_m"], _q["y_m"], _q["radius_m"]
            storage, exp_id, sample_id = _map_scope()
            if storage is None:
                return _j({"success": False, "error": "实验记录存储不可用"})
            r = abs(float(radius_m)) or 50e-9
            key = META_AVOID_RADIUS if forbidden else META_USED_RADIUS
            label = "人工标避让区" if forbidden else "人工标已用"
            row_id = storage.log_marker(
                kind="manual", x_m=float(x_m), y_m=float(y_m),
                label=label, skill_name="mark_area_used", source="manual",
                experiment_id=exp_id, sample_id=sample_id,
                meta={key: r, "pos_src": "param"})
            return _j({"success": True,
                       "action": "marked_forbidden" if forbidden else "marked_used",
                       "marker_id": row_id, "radius_m": r,
                       "message": f"已在 ({x_m * 1e9:.0f}, {y_m * 1e9:.0f}) nm "
                                  f"标记{label},半径 {r * 1e9:.0f} nm。"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("record_coarse_move")
    def record_coarse_move(direction: str = "", steps: int = 0,
                           note: str = "") -> str:
        """补记一次【手动在 Nanonis 里做的】XY 粗动马达移动。

        经 MAST 执行的 MotorMove 会自动记录,不需要调这个;**只有你或用户
        直接在 Nanonis 界面上粗动过**才需要补记(MAST 侦测不到那种操作)。

        效果:扫描地图从现在起进入**新的坐标代次** —— 旧标记淡显、不再参与分析,
        因为粗动之后旧坐标指向的已经是另一片表面了。注意这是「从现在分代」,
        不会去改历史记录的归属(我们无法知道那次手动粗动确切发生在哪一刻)。

        direction: x+|x-|y+|y- (可留空);steps: 步数(可留 0)。"""
        try:
            storage, exp_id, sample_id = _map_scope()
            if storage is None:
                return _j({"success": False, "error": "实验记录存储不可用"})
            d = (direction or "").strip()
            meta = {"manual_backfill": True, "direction": d or None,
                    "steps": int(steps) or None, "note": (note or "").strip() or None}
            storage.log_marker(
                kind="coarse_move", x_m=None, y_m=None,
                label=f"补记手动粗动{(' ' + d) if d else ''}",
                skill_name="record_coarse_move", source="manual",
                experiment_id=exp_id, sample_id=sample_id,
                meta={k: v for k, v in meta.items() if v is not None})
            epoch = storage.current_epoch(exp_id, sample_id)
            # Same invalidation the automatic path does. This branch used to skip
            # it, so a hand-made coarse move left the previous region's plan route
            # and crash blocks in force over fresh surface.
            from mast.core.coarse_move_effects import on_coarse_move_recorded
            on_coarse_move_recorded(source="tool:record_coarse_move")
            return _j({"success": True, "new_coord_epoch": epoch,
                       "message": f"已补记手动粗动,扫描地图进入第 {epoch} 代坐标系;"
                                  "此前的标记不再参与选点分析,计划路线与撞针封锁也已清空。"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("get_markers_near")
    def get_markers_near(x_m: MetrePos, y_m: MetrePos,
                         radius_m: MetreSize = 200e-9) -> str:
        """查询扫描地图上某点附近发生过什么(扫图/谱/修针尖/脉冲/撞针/进针)。

        想在某处扫图或做谱之前,先用它问一句「这里之前干过什么」。
        会跨坐标代次查(查历史不受代次限制),但每条会标 coord_epoch;
        stale_coords=true 表示那条记录属于粗动之前的旧坐标系、位置已不可比。
        坐标和半径写成带 SI 前缀的字符串,如 '-132n'、'200n'。"""
        try:
            # 这是只读查询,量级错了不会写坏记录 —— 但会**静静地查不到东西**,
            # 然后「附近没发生过什么」这句话会被当成可以动手的许可。
            _q, errors = resolve_metre_args(
                "get_markers_near",
                {"x_m": x_m, "y_m": y_m, "radius_m": radius_m})
            if errors:
                return _j({"success": False, "error": " ".join(errors)})
            x_m, y_m, radius_m = _q["x_m"], _q["y_m"], _q["radius_m"]
            from mast.io.exp_map import markers_from_rows
            from mast.io.map_analysis import markers_near as _near
            rows, epoch = _load_map_rows()
            if rows is None:
                return _j({"success": False, "error": "实验记录存储不可用"})
            hits = _near(markers_from_rows(rows), float(x_m), float(y_m),
                         abs(float(radius_m)) or 200e-9, current_epoch=epoch)
            return _j({"success": True, "count": len(hits),
                       "coord_epoch": epoch, "markers": hits})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("get_coarse_map")
    def get_coarse_map(steps: int = 0) -> str:
        """粗动【大地图】:整个样品尺度上,我们**去过哪些片**,以及下一步该往哪走。

        这和 get_map_analysis 是**两张不同尺度的地图**,不要混用:
        · get_map_analysis —— 压电尺度(±1.5 µm,单位米,只看当前代次):
          「这一片区域里扫了哪、哪不能去、该不该换区」。
        · get_coarse_map —— 样品台尺度(单位**步**,跨所有代次):
          「样品上我去过哪些片、还能往哪走、走多少步」。

        每一次横向粗动 = 一个新站点 = 一个新坐标代次。站点位置是**开环里程表**
        (步数累加),所以带一个不确定半径 —— 地图上画成模糊斑而不是点。
        **绝不要用步数去算米**:粗动步长随驱动幅度/负载/温度漂移,低温下同样步数
        走的距离可以差几倍。

        suggestion 给出建议的 axis/direction/steps 与理由;拿它去调
        **RelocateCoarseXY**(唯一应该自主使用的换区技能)。
        suggestion=null 表示这一带已经用完 —— 理由在 note 里,通常意味着要换样品。

        steps: 想走多少步(留 0 用本机默认;规划器可能会自动加大以留出不确定量余量)。"""
        try:
            from mast.io.coarse_map import CoarseMapConfig, build_coarse_map

            storage, exp_id, sample_id = _map_scope()
            if storage is None:
                return _j({"success": False, "error": "实验记录存储不可用"})
            rows = storage.get_markers(exp_id, sample_id) or []
            builder = _ctx().get("coarse_map_cfg")
            cfg = builder() if callable(builder) else None
            if cfg is None:
                cfg = CoarseMapConfig()
            cmap = build_coarse_map(rows, cfg, steps=int(steps) or None)
            out = cmap.as_dict()
            out["success"] = True
            out["units"] = "steps (open-loop; 不要换算成米做几何)"
            return _j(out)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    def _plan_steps_for_map():
        """Pending planned route steps, for the survey figure. Never raises."""
        try:
            from mast.io.plan_overlay import get_plan_overlay
            return list(get_plan_overlay().snapshot() or [])
        except Exception:  # noqa: BLE001
            return None

    # ── plan drafting / approval (execution is the agent's own loop) ───
    @tool("create_plan")
    def create_plan(name: str, goal: str, phases: list,
                    tool_call_id: Annotated[str, InjectedToolCallId] = "",
                    ):
        """把实验计划保存为 draft(需用户审批后再执行)。phases 为阶段列表。

        存下的计划同时会登记为上游产物：执行方(instrument_control)在自己的上下文里
        直接看到 plan_id 与阶段数,不必靠交接语复述——一份 28 步的计划以前只以对话
        消息的形式存在,而那条消息会被截断。
        """
        ctx = _ctx()
        ps = ctx.get("plan_store")
        if ps is None:
            return "计划库不可用"
        try:
            from mast.planning.plan_store import ExperimentPlan, PlanPhase, PlanStatus
            el = ctx.get("experiment_log")
            exp_id = (el.current_experiment_id or "") if el else ""
            sample_id = (el.current_sample_id or "") if el else ""
            plan = ExperimentPlan(
                plan_id="", experiment_id=exp_id, sample_id=sample_id,
                name=name, goal=goal,
                phases=[PlanPhase.from_dict(p) for p in (phases or [])],
                status=PlanStatus.DRAFT)
            plan_id = ps.save(plan)
            summary = _j({"success": True, "plan_id": plan_id, "status": "draft",
                          "phases": len(plan.phases), "total_steps": plan.total_steps,
                          "message": "计划已存为 draft,请展示给用户并请求审批。"})
            try:
                from mast.agents._shared.artifact_channel import (
                    ArtifactToolReturn, doc_ref,
                )
                ref = doc_ref(
                    doc_id=str(plan_id), version=0, kind="plan", title=name or "实验计划",
                    produced_by="experiment_design",
                    summary=(f"{goal}｜{len(plan.phases)} 阶段 / "
                             f"{plan.total_steps} 步；状态 draft（需审批后才可执行）"))
                return ArtifactToolReturn(summary, {"experiment_plan": ref},
                                          tool_call_id=tool_call_id,
                                          name="create_plan")
            except Exception as exc:  # noqa: BLE001 — the plan is already saved
                logger.debug("create_plan artifact publish failed: %s", exc)
                return summary
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("approve_plan")
    def approve_plan(plan_id: str) -> str:
        """把 draft 计划标记为 approved(仅在用户明确批准后调用)。"""
        ps = _ctx().get("plan_store")
        if ps is None:
            return "计划库不可用"
        try:
            plan = ps.load(plan_id)
            if not plan:
                return _j({"success": False, "error": f"未找到计划: {plan_id}"})
            if plan.status.value not in ("draft", "approved"):
                return _j({"success": False,
                           "error": f"计划状态为 {plan.status.value},只有 draft 可批准。"})
            from mast.planning.plan_store import PlanStatus
            ps.update_status(plan_id, PlanStatus.APPROVED)
            return _j({"success": True, "plan_id": plan_id, "status": "approved",
                       "message": "计划已批准。用 advance_plan 逐阶段推进,pause_plan/"
                                  "resume_plan 暂停恢复,get_plan_progress 查进度。"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    # ── plan EXECUTION: advance / pause / resume / progress ────────────
    # The DB layer (update_progress / update_status / get_active) always existed
    # but no tool drove it, so multi-phase overnight plans had no execution
    # tracking and couldn't resume after an interruption.
    @tool("get_plan_progress")
    def get_plan_progress(plan_id: str = "") -> str:
        """查看计划进度:当前阶段、已完成阶段、剩余阶段。plan_id 为空则取当前活动
        (running/paused)计划——过夜中断后据此断点续跑。"""
        ps = _ctx().get("plan_store")
        if ps is None:
            return "计划库不可用"
        try:
            plan = ps.load(plan_id) if plan_id else ps.get_active()
            if not plan:
                return _j({"success": False, "error": "无活动计划" if not plan_id
                           else f"未找到计划: {plan_id}"})
            idx = plan.current_phase_idx
            phases = [{"idx": i, "id": p.id, "name": p.name, "status": p.status}
                      for i, p in enumerate(plan.phases)]
            cur = plan.phases[idx] if 0 <= idx < len(plan.phases) else None
            return _j({"success": True, "plan_id": plan.plan_id,
                       "status": plan.status.value, "current_phase_idx": idx,
                       "current_phase": (cur.name if cur else None),
                       "completed_phases": sum(1 for p in plan.phases if p.status == "done"),
                       "total_phases": len(plan.phases), "phases": phases})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("advance_plan")
    def advance_plan(plan_id: str, phase_status: str = "done", note: str = "") -> str:
        """标记当前阶段结果并推进到下一阶段。phase_status ∈ done|failed|skipped。
        完成最后一阶段后计划置为 completed;failed 且该阶段 on_fail=abort 则置 aborted。
        用它把过夜多阶段计划的进度持久化,中断后可 resume_plan 续跑。"""
        ps = _ctx().get("plan_store")
        if ps is None:
            return "计划库不可用"
        try:
            from mast.planning.plan_store import PlanStatus
            plan = ps.load(plan_id)
            if not plan:
                return _j({"success": False, "error": f"未找到计划: {plan_id}"})
            idx = plan.current_phase_idx
            if idx >= len(plan.phases):
                return _j({"success": False, "error": "计划已无剩余阶段"})
            ps.update_progress(plan_id, idx, 0, phase_status=phase_status)
            cur_phase = plan.phases[idx]
            if phase_status == "failed" and cur_phase.on_fail == "abort":
                ps.update_status(plan_id, PlanStatus.ABORTED, notes=note or None)
                return _j({"success": True, "plan_id": plan_id, "status": "aborted",
                           "message": f"阶段 {cur_phase.name} 失败且 on_fail=abort → 计划中止。"})
            next_idx = idx + 1  # done/failed(retry→caller re-runs)/skipped all move on
            if next_idx >= len(plan.phases):
                final = PlanStatus.COMPLETED if phase_status != "failed" else PlanStatus.ABORTED
                ps.update_status(plan_id, final, notes=note or None)
                return _j({"success": True, "plan_id": plan_id, "status": final.value,
                           "message": "所有阶段处理完毕。"})
            ps.update_progress(plan_id, next_idx, 0, phase_status="running")
            ps.update_status(plan_id, PlanStatus.RUNNING, notes=note or None)
            return _j({"success": True, "plan_id": plan_id, "status": "running",
                       "current_phase_idx": next_idx,
                       "current_phase": plan.phases[next_idx].name,
                       "remaining": len(plan.phases) - next_idx})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("pause_plan")
    def pause_plan(plan_id: str, note: str = "") -> str:
        """暂停一个正在执行的计划(保留 current_phase,可 resume_plan 恢复)。"""
        ps = _ctx().get("plan_store")
        if ps is None:
            return "计划库不可用"
        try:
            from mast.planning.plan_store import PlanStatus
            plan = ps.load(plan_id)
            if not plan:
                return _j({"success": False, "error": f"未找到计划: {plan_id}"})
            ps.update_status(plan_id, PlanStatus.PAUSED, notes=note or None)
            return _j({"success": True, "plan_id": plan_id, "status": "paused",
                       "current_phase_idx": plan.current_phase_idx})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("resume_plan")
    def resume_plan(plan_id: str = "") -> str:
        """恢复最近暂停/运行的计划;返回从哪个阶段继续(断点续跑)。plan_id 为空则取
        当前活动计划。"""
        ps = _ctx().get("plan_store")
        if ps is None:
            return "计划库不可用"
        try:
            from mast.planning.plan_store import PlanStatus
            plan = ps.load(plan_id) if plan_id else ps.get_active()
            if not plan:
                return _j({"success": False, "error": "无可恢复的计划"})
            if plan.status.value not in ("paused", "running", "approved"):
                return _j({"success": False,
                           "error": f"计划状态为 {plan.status.value},无法恢复。"})
            ps.update_status(plan.plan_id, PlanStatus.RUNNING)
            idx = plan.current_phase_idx
            cur = plan.phases[idx] if 0 <= idx < len(plan.phases) else None
            return _j({"success": True, "plan_id": plan.plan_id, "status": "running",
                       "resume_phase_idx": idx,
                       "resume_phase": (cur.name if cur else None),
                       "message": "从当前阶段继续执行其技能。"})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    # ── plan PREVIEW on the live scan map (car-navigation style) ───────
    @tool("show_plan_on_map")
    def show_plan_on_map(steps: list[PlanStep], title: str = "") -> str:
        """在扫描地图上预览一条计划路线(像汽车导航的路线)。

        steps: 有序的步骤列表,每步给 {kind, x_m, y_m, [w_m, h_m, label]};
        坐标用**米**量纲(与 mark_area_used / get_next_scan_position 一致),和你调用
        FullScan 时写 center_x_m='1.4u' / height_m='50n' 完全同一套写法。
        量级必须是真实 STM 尺度:压电工作区约 ±'1.5u',100 nm 的扫描框写
        w_m='100n' 而不是 w_m=1.0(那是 1 米)。量级荒谬的步骤会被整条拒绝发布。
        kind ∈ scan|sts|pulse|tip_shape|move。地图会把这些点连成虚线路线并标号;
        之后每执行一个真实操作,路线会自动推进(到达的步骤消失),让用户实时看到进度。
        制定/展示计划时调用此工具,用户就能在地图上看到将要去哪、做什么。"""
        try:
            from mast.io.exp_map import KIND_STYLE, MapMarker
            from mast.io.plan_overlay import get_plan_overlay
            markers: list = []
            errors: list[str] = []
            warnings: list[str] = []
            # The schema hands us PlanStep instances; a hand-built call (tests,
            # the JSON route) may still hand us plain dicts. Normalise once so
            # the body below stays dict-shaped.
            for i, s in enumerate(steps or [], 1):
                if isinstance(s, BaseModel):
                    s = s.model_dump()
                if not isinstance(s, dict):
                    continue
                kind = str(s.get("kind") or "plan").strip().lower()
                if kind not in KIND_STYLE or kind in ("frame", "tip"):
                    kind = "plan"
                x, y = s.get("x_m"), s.get("y_m")
                if x is None or y is None:
                    continue
                errs, warns = _check_plan_step_magnitudes(i, s)
                errors.extend(errs)
                warnings.extend(warns)
                try:
                    markers.append(MapMarker(
                        kind=kind, x_m=float(x), y_m=float(y),
                        w_m=(float(s["w_m"]) if s.get("w_m") is not None else None),
                        h_m=(float(s["h_m"]) if s.get("h_m") is not None else None),
                        label=str(s.get("label") or ""),
                        status="planned", source="plan"))
                except (TypeError, ValueError):
                    continue
            # Reject the WHOLE publish, not the offending steps: a partially
            # published route is a lie about what the operator will see, and
            # dropping steps quietly is exactly how 17 bad calls went unnoticed.
            if errors:
                return _j({
                    "success": False,
                    "error": "计划坐标量级不合法,未发布(整条路线都没有发布)。",
                    "problems": errors,
                    "expected": (f"位置 |x_m|,|y_m| ≤ {_PLAN_POS_ABS_MAX_M:g} m,"
                                 f"尺寸 |w_m|,|h_m| ≤ {_PLAN_SIZE_ABS_MAX_M:g} m;"
                                 f"典型 STM 工作区是 ±{_PLAN_PIEZO_XY_M:g} m"),
                })
            if not markers:
                return _j({"success": False,
                           "error": "无有效步骤(每步至少需要 x_m 和 y_m,单位米)"})
            ov = get_plan_overlay()
            ov.set_plan(markers, title=title)
            # Read the store back instead of assuming the write landed. The old
            # return claimed "计划路线已显示在扫描地图上" purely because nothing
            # threw — and for a long time that claim was false: /api/scan-map
            # never read this overlay, so the route went nowhere while the tool
            # reported success. Report what is actually
            # in the store, and describe only what we can verify.
            published = ov.snapshot()
            if not published:
                return _j({"success": False,
                           "error": "计划路线发布失败:写入后回读为空。"})
            out = {"success": True, "steps": len(published),
                   "message": (f"已发布 {len(published)} 步计划路线到扫描地图;"
                               "每完成一个真实操作,到达的步骤会自动消失。")}
            if warnings:
                out["warnings"] = warnings
            return _j(out)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("clear_plan_on_map")
    def clear_plan_on_map() -> str:
        """清除扫描地图上的计划路线预览。"""
        try:
            from mast.io.plan_overlay import get_plan_overlay
            ov = get_plan_overlay()
            # Say how many were actually removed — "已清除计划路线" read the same
            # whether it cleared 8 steps or nothing at all.
            n = len(ov.snapshot())
            ov.clear()
            return _j({"success": True, "cleared": n,
                       "message": (f"已清除 {n} 步计划路线。" if n
                                   else "地图上本来就没有计划路线。")})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("list_plans")
    def list_plans() -> str:
        """列出已保存的实验计划(id/名称/状态)。"""
        ps = _ctx().get("plan_store")
        if ps is None:
            return "计划库不可用"
        try:
            plans = ps.list_plans() if hasattr(ps, "list_plans") else []
            return _j({"success": True, "plans": [
                {"plan_id": getattr(p, "plan_id", ""), "name": getattr(p, "name", ""),
                 "status": getattr(getattr(p, "status", None), "value", "")}
                for p in plans]})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    # ── true-parallel offload ─────────────────────────────────────────
    @tool("spawn_background_task")
    def spawn_background_task(instruction: str, agent: str = "literature",
                             priority: str = "normal",
                             then_agent: str = "", then_note: str = "") -> str:
        """把一个【长/独立】子任务甩到【后台】异步运行,立即返回,前台(你和仪器操作)不被阻塞。

        用于不依赖当前前台步骤、又比较耗时的独立任务:文献综述、离线分析已保存的扫描、
        起草/评审论文等。典型场景:一边继续调仪器,一边让文献 agent 在后台查资料。
        agent 只能是非仪器 agent:literature / data_processing / experiment_design /
        paper_writing / paper_review;instrument_control 不能后台化(它是前台仪器 agent)。
        后台任务的进度与结果会自动带回【当前群聊】并标注「后台」。

        priority: normal|high —— high 在并发槽紧张时优先占用。
        then_agent / then_note(可选):【显式】声明该后台任务完成后建议的后续步骤
        (如 then_agent="experiment_design")。完成时会把结论+该建议自动带回前台提示编排器;
        【绝不自动执行】,是否继续仍由你/用户决定。留空表示不声明后续。"""
        spawn = _ctx().get("spawn_background")
        if not callable(spawn):
            return _j({"success": False, "error": "后台任务功能不可用"})
        on_done = None
        if then_agent.strip() or then_note.strip():
            on_done = {"next_agent": then_agent.strip(), "note": then_note.strip()}
        try:
            res = spawn(instruction, (agent,), priority=priority, on_done=on_done) or {}
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})
        if not res.get("ok"):
            return _j({"success": False, "error": res.get("error", "无法启动后台任务")})
        return _j({"success": True, "run_id": res.get("run_id"),
                   "agents": res.get("agents"),
                   "message": f"已在后台启动任务(agent={agent},priority={priority})。前台可继续;"
                              "完成后结果会带回本群聊。"})

    # ── 多图规划(要求的第三个场景) ─────────────────────────────
    @tool("plan_scan_batch")
    def plan_scan_batch(kind: str, n_images: int = 0, size_nm: float = 0.0,
                        feature: str = "", final_size_nm: float = 0.0,
                        series_param: str = "", series_values: str = "",
                        max_total_minutes: float = 0.0,
                        publish_to_map: bool = True) -> str:
        """一次要扫若干张图时,由脚本决定扫哪几张、每张什么参数、什么顺序。

        【**你只说要什么,不要说参数**。扫哪几个位置、每张用什么速度/像素、按什么
        顺序走、哪几帧之前要等压电稳定 —— 全部由脚本按记录和用户的按尺度偏好表
        确定性地算出来。这些不是你的知识。】

        kind: survey(在未覆盖表面上铺 N 张) | zoomin(在当前帧里挑特征放大) |
        bias_series(同一位置逐张变偏压) | param_series | repeat(同位置重复)。
        size_nm: 每张图边长,单位**纳米**(人类单位 —— 这个工具替你换算成米)。
        50 nm 的图就写 50。**不要在这里写米**:写 5e-8 会被当成 5e-8 纳米。
        feature: zoomin 的目标类别,只能是 terrace|step_edge|defect|contamination|
        user_marked(分割只有这四类几何/异常语义;「某种具体缺陷」做不到,会被诚实拒绝)。
        series_values: 逗号分隔的数值,例如 "-1.0,-0.5,0.5,1.0"。**不给就拒绝** ——
        规划器不发明序列。

        返回一份逐帧计划(或带 code 的结构化拒绝 + 可行替代)。计划会同时发布到
        扫描地图上让用户先看。拿到计划后用 ExecuteScanPlan 原样执行,不要手改帧。"""
        try:
            import json as _json

            from mast.core.scan_planner import PlanReject, plan_batch
            from mast.io.map_analysis import pick_next_positions

            size_m = float(size_nm) * 1e-9 if size_nm else 0.0
            final_m = float(final_size_nm) * 1e-9 if final_size_nm else 0.0
            # 换算完就送同一道量级护栏。这里**不引入任何新阈值** —— 换算后就是米,
            # 交给判米的那一套(`size_m` 已经在 _METRE_SIZE_ARGS 里)。
            #
            # 为什么这个工具也需要:`plan_batch` 对尺寸只判 `size <= 0`,没有上界。
            # 一个 size_nm=5e7(模型把米当成纳米写)会算出 5 cm 的扫描框,一路排完
            # 计划、返回 plan_json、还叫模型「用 ExecuteScanPlan 原样执行」。
            #
            # 借护栏的**裁决**,不借它的**措辞**。护栏的提示词是按米写的
            # (「若本意是 0.05 µm,应写 5e-08」),而这个参数的单位是纳米 ——
            # 照搬过来会给出一句对这个参数**错误**的建议,那比不提示更糟。
            # 判定仍然只有一处(共用 `_PLAN_SIZE_ABS_MAX_M`),这里只重写给模型看的话。
            for _wrote, _val in (("size_nm", size_nm), ("final_size_nm", final_size_nm)):
                _m = float(_val) * 1e-9 if _val else 0.0
                _q, errors = resolve_metre_args("t", {"size_m": _m})
                if errors:
                    return _j({"success": False, "error": (
                        f"{_wrote}={float(_val):g} 纳米 = {_m:g} 米,超出扫描尺寸上限"
                        f"({_PLAN_SIZE_ABS_MAX_M:g} m = {_PLAN_SIZE_ABS_MAX_M / 1e-9:g} nm)。"
                        f"**本参数的单位是纳米**:50 nm 的图就写 50。"
                        f"如果你刚才写的是米(如 5e-8),那正是这里出错的原因。")})

            intent: dict = {"kind": (kind or "").strip().lower()}
            if n_images:
                intent["n_images"] = int(n_images)
            if size_m:
                intent["size_m"] = size_m
            if feature or final_m:
                intent["target"] = {
                    k: v for k, v in
                    (("feature", feature.strip().lower() or None),
                     ("final_size_m", final_m or None))
                    if v is not None
                }
            if series_param or series_values:
                values = []
                for chunk in (series_values or "").split(","):
                    chunk = chunk.strip()
                    if chunk:
                        try:
                            values.append(float(chunk))
                        except ValueError:
                            return _j({"success": False,
                                       "error": f"序列值 {chunk!r} 不是数值"})
                intent["series"] = {"param": series_param.strip().lower(),
                                    **({"values": values} if values else {})}
            if max_total_minutes:
                intent["constraints"] = {"max_total_minutes": float(max_total_minutes)}

            # 位置来源:survey 用地图分析的候选序列(避开已毁区、已扫区、彼此)。
            survey_positions = None
            if intent["kind"] == "survey" and size_m:
                rows, _ = _load_map_rows()
                if rows is None:
                    return _j({"success": False, "error": "实验记录存储不可用"})
                from mast.io.exp_map import markers_from_rows
                from mast.io.map_analysis import current_epoch_of, filter_epoch
                cfg = _map_cfg(frame_size_m=size_m)
                # 只看当前坐标代次:XY 粗动之后,旧坐标指的是另一片表面。
                rows_now = filter_epoch(rows, current_epoch_of(rows))
                markers = markers_from_rows(rows_now)
                picks = pick_next_positions(markers, cfg, int(n_images or 4))
                survey_positions = [(p.x_m, p.y_m) for p in picks]

            # 起点用于最近邻排序。读不到就用原点 —— 排序只影响走位效率,
            # 不影响任何一帧扫在哪儿,所以这里没必要为读不到而失败。
            tip = (0.0, 0.0)
            try:
                from mast.skills.builtins._tip_xy import read_tip_xy
                got = read_tip_xy(_ctx().get("execution_context"))
                if got and got[0] is not None and got[1] is not None:
                    tip = (float(got[0]), float(got[1]))
            except Exception as exc:  # noqa: BLE001
                logger.debug("规划起点读取失败(用原点排序): %s", exc)

            plan = plan_batch(intent, tip_xy=tip,
                              survey_positions=survey_positions)

            # ── 给整份计划盖坐标代次的章 ────────────────────────────────
            #
            # 计划里每一帧都是一对米坐标,而米坐标只在**一个代次内**有意义:
            # 一次横向粗动之后,同样的 (x, y) 指的是另一片表面。没有这个章,
            # 一份粗动之前排的计划在粗动之后照样能原样执行 —— 每一帧都扫在错的
            # 地方,而且扫得很成功。ExecuteScanPlan 认这个章(开跑时 + 每帧前核对)。
            #
            # 取值走 `core.coord_epoch.read_current_epoch()` —— 消费侧
            # (ExecuteScanPlan / MoveToXY)调的是**同一个函数**,所以一份章不会
            # 因为「两边各查各的」而对不上。它底下是 `storage.current_epoch()`
            # 这条权威查询,不从 get_markers() 的截断窗口里数 coarse_move:那个
            # 窗口只有最新 limit 行,作用域一长就漏报代次,于是陈旧坐标被判成
            # 当前(coarse_motion 已经踩过)。
            # 查不到就**不盖章**(键不出现),不盖 0 —— 0 是「还没粗动过」这个
            # 真实答案,拿它冒充「不知道」会让消费侧误判。
            _epoch = read_current_epoch()

            if isinstance(plan, PlanReject):
                out = {"success": False, "rejected": True, **plan.as_dict()}
                if plan.partial:
                    # 部分计划同样带坐标,同样能被喂给 ExecuteScanPlan ⇒ 同批盖章。
                    _partial: dict = {"frames": [
                        {"index": f.index, "center_x_m": f.center_x_m,
                         "center_y_m": f.center_y_m, "size_m": f.size_m,
                         "label": f.label, "needs_settle": f.needs_settle}
                        for f in plan.partial]}
                    if _epoch is not None:
                        _partial["coord_epoch"] = _epoch
                    out["partial_plan"] = _json.dumps(_partial)
                return _j(out)

            plan_dict = plan.as_dict()
            if _epoch is not None:
                plan_dict["coord_epoch"] = _epoch
            published = 0
            publish_refused = ""
            if publish_to_map:
                # 地图那一侧对每一帧跑的是 `_check_plan_step_magnitudes`。它**已经**
                # 会拒,问题在于这里原本把它的裁决扔了:`published` 归 0,一句 debug
                # 日志,然后照样 success:True + 「用 ExecuteScanPlan 原样执行」。
                # 一道判了但没人听见的护栏,和没有护栏是同一个东西
                # ([[guard_that_isnt]])。
                try:
                    res = show_plan_on_map.invoke({
                        "steps": [f.footprint() for f in plan.frames],
                        "title": f"{plan.kind} × {len(plan.frames)}"})
                    try:
                        parsed = _json.loads(str(res))
                    except (ValueError, TypeError):
                        parsed = {}
                    if parsed.get("success"):
                        published = len(plan.frames)
                    else:
                        publish_refused = str(
                            parsed.get("error") or parsed.get("message") or res)
                except Exception as exc:  # noqa: BLE001
                    publish_refused = str(exc)
                if publish_refused:
                    logger.warning("计划发布到地图被拒: %s", publish_refused)

            message = (f"已排出 {len(plan.frames)} 帧计划,估计 "
                       f"{plan.total_estimated_s / 60.0:.1f} 分钟。")
            out = {
                "success": True,
                "plan": plan_dict,
                "plan_json": _json.dumps(plan_dict),
                "published_to_map": published,
            }
            if publish_refused:
                out["publish_refused"] = publish_refused
                message += ("**但扫描地图拒绝了这份计划,用户看不到它**:"
                            f"{publish_refused} 先按这条改正再重排,不要直接执行。")
            else:
                message += "用 ExecuteScanPlan(plan_json=…) 原样执行。"
            if _epoch is None:
                # 章盖不上就说出来。悄悄少一道保护,比没有保护更糟 ——
                # 下一个人会以为它保护过。
                out["coord_epoch_unavailable"] = True
                message += ("(**这份计划没盖坐标代次的章**:当前代次查不到。"
                            "执行时不会有「粗动之后拒绝旧坐标」的保护 ——"
                            "中间若粗动过,请重排。)")
            out["message"] = message
            return _j(out)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    # ── 针尖登记(仪器级)────────────────────────────────────────────────
    #
    # 术语纪律:本仓「tip change」是 vision 的 mid-scan 针尖态突变检测。这里说的
    # 是**物理更换针尖**,一律用 register / 装入 / 登记。

    def _tip_storage():
        return _ctx().get("storage")

    def _tip_unavailable() -> str:
        return _j({"success": False,
                   "error": "针尖登记不可用(本会话没有接上实验记录库)。"})

    @tool("register_tip")
    def register_tip(
        material: str = "",
        fabrication: str = "",
        form: str = "",
        name: str = "",
        wire_diameter_mm: float | None = None,
        installed_at: str = "",
        qplus_sensor_model: str = "",
        qplus_f0_hz: float | None = None,
        qplus_q: float | None = None,
        note: str = "",
    ) -> str:
        """登记「装入了一根新针尖」。仪器级——换实验/换样品不必重登记。

        material: 钨/W、铂铱/PtIr、铁/Fe… fabrication: 电化学腐蚀/etched、
        钳子剪/cut、打磨/ground、FIB/fib。form: 普通 STM 针尖=stm_wire,
        qPlus 型=qplus。中英文都认。
        installed_at 可回填(ISO 日期,如 2026-07-28),留空=现在。

        **这会清掉上一根针的学习标定**(dI/dV 接触标定、qPlus 自由振幅基线)——
        那些量绑的是上一根针,换针后再用就是错的。旧值会归档进退役的那一行。

        不确定针尖是什么就**别猜**:问用户。登记错的材料会让修针方案表按错的
        针给参数。"""
        st = _tip_storage()
        if st is None:
            return _tip_unavailable()
        try:
            from mast.core import tip_state as _ts
            from mast.logging import tip_registry as _tr
            res = _tr.register_tip(
                st, material=material, fabrication=fabrication, form=form,
                name=name, wire_diameter_mm=wire_diameter_mm,
                installed_at=(installed_at or None),
                qplus_sensor_model=qplus_sensor_model,
                qplus_f0_hz=qplus_f0_hz, qplus_q=qplus_q, note=note)
            if not res.get("ok"):
                return _j({"success": False, "error": res.get("error", "登记失败")})
            out: dict = {"success": True, "tip": res.get("tip"),
                         "cleared_calibration": sorted(res.get("cleared") or {})}
            if res.get("warnings"):
                out["warnings"] = res["warnings"]
            # 词表 miss 时给候选,让模型能改对而不是重试同一个词。
            tip = res.get("tip") or {}
            if material and not _ts.normalize_material(material):
                out["material_candidates"] = _ts.material_candidates()
            if fabrication and tip.get("fabrication") == "unknown":
                out["fabrication_candidates"] = _ts.fabrication_candidates()
            out["message"] = (
                f"已登记针尖「{tip.get('name', '')}」。"
                + ("上一根针的学习标定已清除(旧值已归档)。"
                   if res.get("cleared") else ""))
            return _j(out)
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("get_current_tip")
    def get_current_tip() -> str:
        """当前装在仪器里的针尖是什么(材料/制备/形态/装入日期/qPlus 参数)。

        返回 registered=false 表示没人登记过——不代表仪器里没有针,
        只代表系统不知道它是什么,此时修针方案只能用通用保守参数。"""
        st = _tip_storage()
        if st is None:
            return _tip_unavailable()
        try:
            row = st.get_current_tip()
            if not row:
                return _j({"success": True, "registered": False,
                           "message": "当前没有已登记的针尖。装了新针请调 register_tip 登记。"})
            return _j({"success": True, "registered": True, "tip": row})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("list_tips")
    def list_tips(limit: int = 20) -> str:
        """换针史:用过哪些针尖、各自什么时候装入/取下。最近的在前。"""
        st = _tip_storage()
        if st is None:
            return _tip_unavailable()
        try:
            rows = st.list_tips(limit=max(1, min(int(limit or 20), 200)))
            return _j({"success": True, "count": len(rows), "tips": rows})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("update_tip")
    def update_tip(
        name: str = "",
        material: str = "",
        fabrication: str = "",
        form: str = "",
        wire_diameter_mm: float | None = None,
        qplus_sensor_model: str = "",
        qplus_f0_hz: float | None = None,
        qplus_q: float | None = None,
        note: str = "",
        tip_name: str = "",
    ) -> str:
        """补记/更正针尖属性(线径、音叉型号、备注…)。

        默认改**当前**针尖;要改历史上的某一根,用 tip_name 指名(不是 UUID)。
        补记不会清任何标定——那只在 register_tip(装入新针)时发生。"""
        st = _tip_storage()
        if st is None:
            return _tip_unavailable()
        try:
            from mast.logging import tip_registry as _tr
            if tip_name.strip():
                row = st.find_tip_by_name(tip_name)
                if not row:
                    names = [t.get("name") for t in st.list_tips(limit=50)]
                    return _j({"success": False,
                               "error": f"没有叫「{tip_name}」的针尖。",
                               "known_tips": names})
            else:
                row = st.get_current_tip()
                if not row:
                    return _j({"success": False,
                               "error": "当前没有已登记的针尖;要改历史记录请给 tip_name。"})
            fields = {k: v for k, v in (
                ("name", name), ("material", material),
                ("fabrication", fabrication), ("form", form),
                ("qplus_sensor_model", qplus_sensor_model), ("note", note),
            ) if str(v or "").strip()}
            for k, v in (("wire_diameter_mm", wire_diameter_mm),
                         ("qplus_f0_hz", qplus_f0_hz), ("qplus_q", qplus_q)):
                if v is not None:
                    fields[k] = v
            res = _tr.update_tip(st, row["id"], fields)
            if not res.get("ok"):
                return _j({"success": False, "error": res.get("error"),
                           "warnings": res.get("warnings") or []})
            return _j({"success": True, "tip": res.get("tip"),
                       "updated": res.get("updated"),
                       "warnings": res.get("warnings") or []})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    @tool("remove_current_tip")
    def remove_current_tip(note: str = "") -> str:
        """记录「针尖被取出、还没装新的」。

        通常你不需要这个:装下一根直接 register_tip 即可(它会自动退役上一根)。
        只有针尖已取出而短期不装新针时才用。同样会清掉绑那根针的标定。"""
        st = _tip_storage()
        if st is None:
            return _tip_unavailable()
        try:
            from mast.logging import tip_registry as _tr
            res = _tr.remove_current_tip(st, note=note)
            if not res.get("ok"):
                return _j({"success": False, "error": res.get("error")})
            return _j({"success": True, "changed": res.get("changed", True),
                       "cleared_calibration": sorted(res.get("cleared") or {}),
                       "message": res.get("message", "已记录针尖取出。")})
        except Exception as exc:  # noqa: BLE001
            return _j({"success": False, "error": str(exc)})

    return [
        start_experiment, end_experiment, start_sample, end_sample,
        rename_experiment, rename_sample,
        switch_experiment, switch_sample, clear_sample,
        list_experiments, list_samples,
        register_tip, get_current_tip, list_tips, update_tip, remove_current_tip,
        get_latest_scan_info, load_scan_file,
        get_workflow_advice, get_skill_guidance, get_literature_parameters,
        get_fault_diagnosis, get_noise_reference, get_measurement_template,
        search_deep_reference, read_reference_section, query_knowledge,
        get_map_analysis, get_next_scan_position, mark_area_used,
        record_coarse_move, get_markers_near, get_coarse_map, plan_scan_batch,
        create_plan, approve_plan, list_plans,
        get_plan_progress, advance_plan, pause_plan, resume_plan,
        show_plan_on_map, clear_plan_on_map,
        spawn_background_task,
    ]


# The meta-tool names, as a module-level constant so the artifact data-flow graph
# can be derived WITHOUT constructing the tools (which needs a live provider).
# instrument_control receives this whole set; experiment_design receives
# DESIGN_TOOL_NAMES (below), which is a different — not a smaller — slice.
#
# Kept honest by tests/v2/unit/agents/test_artifact_flow.py, which builds the real
# tool list and asserts these names match it exactly.
LIFECYCLE_TOOL_NAMES: tuple[str, ...] = (
    "start_experiment", "end_experiment", "start_sample", "end_sample",
    "rename_experiment", "rename_sample",
    # 2026-07-28: 实验和样品都是永久的、可来回切换的，模型必须能
    # 自己切回一个旧实验（「做了一个月这个又回去做那个」）。
    "switch_experiment", "switch_sample", "clear_sample",
    "list_experiments", "list_samples",
)

#: 针尖登记的**只读**工具。实验设计 agent 该知道现在装的是什么针(方案要据此
#: 定),但不该能登记换针 —— 那是发生在仪器旁的物理事件,由用户或 IC 记录。
TIP_READ_TOOL_NAMES: tuple[str, ...] = ("get_current_tip", "list_tips")

#: 针尖登记的全部工具。刻意**不并进** LIFECYCLE_TOOL_NAMES:那个子集会整体发给
#: experiment_design,而 register_tip 会清掉学习标定,不是设计 agent 该有的手。
TIP_TOOL_NAMES: tuple[str, ...] = TIP_READ_TOOL_NAMES + (
    "register_tip", "update_tip", "remove_current_tip",
)

META_TOOL_NAMES: tuple[str, ...] = LIFECYCLE_TOOL_NAMES + TIP_TOOL_NAMES + (
    "get_latest_scan_info", "load_scan_file",
    "get_workflow_advice", "get_skill_guidance", "get_literature_parameters",
    "get_fault_diagnosis", "get_noise_reference", "get_measurement_template",
    "search_deep_reference", "read_reference_section", "query_knowledge",
    "get_map_analysis", "get_next_scan_position", "mark_area_used",
    "record_coarse_move", "get_markers_near", "get_coarse_map", "plan_scan_batch",
    "create_plan", "approve_plan", "list_plans",
    "get_plan_progress", "advance_plan", "pause_plan", "resume_plan",
    "show_plan_on_map", "clear_plan_on_map",
    "spawn_background_task",
)

#: **experiment_design 的 meta 工具面 —— 一个集合,两个消费者。**
#:
#: 消费者 1:`core/runtime.py` 建群聊时按这份名单从 IC 的 meta 全集里过滤出 XD 的那份。
#: 消费者 2:`agents/_shared/artifacts.py` 派生 artifact 数据流图上 XD 的读写边。
#:
#: 在此之前这两处各写各的(runtime 给 `lifecycle 六件 | plan 三件 | tip 只读`,
#: artifacts 派生 `meta & LIFECYCLE`),于是**图上少画了 XD → experiment_plan 的写边**,
#: 而校验这条边的测试比对的是 artifacts 自己的派生 —— 派生方定义了自己的输入,
#: 永远自洽。名字进出这份名单会同时改变授予与图,这正是要的。
#:
#: 为什么 `create_plan` 在里面:2026-07-27 取证——XD 产出的 28 步方案**只以对话消息
#: 的形式存在**,第 3 步被 2000 字符截断,两个库的 plans 表都是 0 行。方案要活下来
#: 只有落库这一条路。(当时补了工具却没改提示词,提示词到 2026-08-14 还在教它
#: 「把 JSON 写进对话、supervisor 会解析」——那个解析器从来不存在。)
#:
#: 为什么 `approve_plan` **在**里面(2026-08-20 改):从前的理由是「一个 agent 不能
#: 批准自己写的方案」。那句话防的是「自批自」这件事本身,而它防的方式是**不给工具**
#: —— 于是同一个能力在 IC 手里有、在 XD 手里没有,而两者都是模型。真正该防的东西
#: (没人看过就开跑)由 conduct 那一层的**自主度策略**回答:attended 档下 agent 批
#: 不动 conduct,supervised 档下批了也要过撤销窗。把关移到服务端之后,谁看得见这个
#: 工具就不再是一道安全边界,只是一道给自己找的麻烦。
#: 执行(advance/pause/resume)同样不再按角色裁。
#:
#: 为什么 `get_literature_parameters` **不**在里面:它返回 p25/p50/p75——一个**长得
#: 像可以直接填**的数字,而「这不是权威默认」只写在返回文本里(那是说服,不是结构)。
#: 文献数值要进方案,应经 literature agent 的证据包带着出处进来。
#: 同理不给任何**写**工具(`mark_area_used` 写实验记录、`record_coarse_move` 推进坐标
#: 代次):设计 agent 不碰表面。针尖只给只读两件——`register_tip` 会清掉学习标定,
#: 而换针是发生在仪器旁的物理事件,该由用户或 IC 记录。
DESIGN_TOOL_NAMES: tuple[str, ...] = LIFECYCLE_TOOL_NAMES + TIP_READ_TOOL_NAMES + (
    # 方案:起草 + 批准 + 推进 + 回看。2026-08-20 起 approve/advance/pause/resume
    # 不再按角色裁 —— 见上面那段关于 approve_plan 的注释。
    "create_plan", "approve_plan", "list_plans", "get_plan_progress",
    "advance_plan", "pause_plan", "resume_plan",
    # 知识面(2026-08-14 增授)。在此之前**要出主意的那个 agent 恰好是知识工具最少
    # 的那个**:XD 只有 describe_skills / lookup_sample / query_past_experiments,
    # 而三档知识、工作流建议、测量模板全都只有 IC 拿得到。
    "query_knowledge", "get_workflow_advice", "get_measurement_template",
    "get_skill_guidance", "search_deep_reference", "read_reference_section",
    "get_fault_diagnosis",
)

__all__ = ["make_meta_tools", "META_TOOL_NAMES", "LIFECYCLE_TOOL_NAMES",
           "TIP_TOOL_NAMES", "TIP_READ_TOOL_NAMES", "DESIGN_TOOL_NAMES"]
