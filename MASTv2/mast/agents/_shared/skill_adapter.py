"""Skill → LangChain tool adapter — the SOLE BaseSkill import point in v2.

All 200 builtin + 5 composite + 21 paper skills are surfaced to LangGraph
agents via wrap_skill(). This isolation lets the legacy v1 skill code stay
untouched while v2 agents call them as @tool primitives.

Pattern (Phase 4):
    from mast.skills.builtins.bias import GetBias
    from mast.agents._shared.skill_adapter import wrap_skill

    INSTRUMENT_TOOLS = [
        wrap_skill(GetBias, context_provider=make_ctx),
        wrap_skill(SetBias, context_provider=make_ctx),
        ...
    ]

Design note (Phase 4 revision; corrected 2026-05-30 — ):
  We tried two return types and learned:
    (1) Plain string (or str subclass) return: this is what we shipped, and it
        is BROKEN for state persistence. langgraph 1.1.9 ToolNode calls
        ``tool.invoke(call)`` which converts any non-Command/non-ToolMessage
        return into a bare ``ToolMessage(content=str(ret))`` and DISCARDS any
        extra attributes — so the ``.update`` dict (executed_skills / scan_paths
        / composite_progress / error_log) NEVER reached MASTState in the real
        agent runtime. Only the direct-call unit tests (``tool.func(...)``,
        which bypass ToolNode) ever saw ``.update``. Verified against the
        installed langgraph 1.1.9 / langchain 1.2.15.
    (2) ``Command(update={...})``: ToolNode DOES apply ``command.update`` to
        state (it ``isinstance(response, Command)``-checks and merges). The
        earlier note claimed ``InjectedToolCallId`` "doesn't reach **kwargs in
        langgraph 1.1.x" — the real issue was that the args_schema (built from
        skill ParameterSpecs) had no ``tool_call_id`` field for langgraph to
        inject into. Declaring an ``Annotated[str, InjectedToolCallId]`` field
        in the generated args_schema fixes that: langgraph injects the calling
        tool_call.id into ``kwargs['tool_call_id']`` (hidden from the LLM), and
        the returned ToolMessage carries it so ToolNode's validation passes.
  Fix (this revision): the args_schema includes a hidden InjectedToolCallId
  field, and ``_run`` returns a ``Command(update={...})`` (subclass
  ``_SkillToolReturn`` below) so the state side-effects are really persisted by
  the checkpointer. Backward compatibility with the ~175 direct-call unit tests
  is preserved: ``_SkillToolReturn`` subclasses ``Command`` (so ``.update`` is
  the dict) AND overrides ``__str__`` / ``__contains__`` so ``str(result)`` /
  ``"x" in result`` still see the human summary.

Skills NEVER serialize tensors / arrays into the tool's return value — only
file paths and primitive types. Hook block_scan_tensors_in_checkpointer.py
enforces this at state.py write time.
"""
from __future__ import annotations

import logging
import time
from typing import Annotated, Any, Callable, Literal, TYPE_CHECKING

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool as _tool_decorator
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from mast.core.si_quantity import (
    SIParseError,
    format_si,
    needs_strict_prefix,
    parse_quantity,
)

logger = logging.getLogger(__name__)

_TOOL_RETURN_CAP = 2000

#: Result keys that exist for the RECORD, not for the model. They stay in
#: ``SkillResult.data`` (so records / the GUI keep them) but are stripped from
#: the ToolMessage the agent reads.
#:
#: ``safe_mode_raw`` is the whole reason this exists. In SAFE mode the tip
#: verdicts are rewritten to "good" precisely so the agent stops being handed an
#: argument for repairing the tip — and then the audit trail carried that exact
#: argument ("tip_ready": false, "recommendation": "mild_conditioning",
#: "is_good": false) back into its context through the default
#: ``summary = str(data)``. Keeping the trail out of the ToolMessage is what
#: makes the override actually hold. (2026-08-01 audit.)
_AUDIT_ONLY_KEYS = ("safe_mode_raw",)


def _strip_audit_only(data: "dict | Any") -> "dict | Any":
    """Drop record-only keys from a result dict before it becomes a ToolMessage."""
    if not isinstance(data, dict):
        return data
    if not any(k in data for k in _AUDIT_ONLY_KEYS):
        return data
    return {k: v for k, v in data.items() if k not in _AUDIT_ONLY_KEYS}


def _explanation_suffix(data: "dict | Any", success: bool) -> str:
    """让工具返回带上支撑结论的诊断信息，而不只剩一句摘要。
    
    失败路径附带除审计专用键外的数据；成功路径只附带技能明确给出的 detail。
    不能因技能写了 summary 就丢掉失败诊断，也不能无差别放大每次成功返回。
    超长摘要交给 _offload_long_summary 落盘并提供引用。"""
    if not isinstance(data, dict) or not data:
        return ""
    if not success:
        return "\n" + str(_strip_audit_only(data))
    detail = data.get("detail")
    if isinstance(detail, str) and detail.strip():
        return "\ndetail: " + detail
    return ""


def _offload_long_summary(skill_name: str, full: str, cap: int = _TOOL_RETURN_CAP) -> str:
    """Cap a tool-return summary at ``cap`` chars for the LLM context; when it
    overflows, persist the FULL text under ``artifacts/tool_returns/`` and append
    a reference so nothing is lost. Best-effort: any IO error degrades to a plain
    truncation (the prior behaviour)."""
    if not isinstance(full, str) or len(full) <= cap:
        return full
    try:
        import uuid as _uuid

        from mast._runtime_paths import project_root
        d = project_root() / "artifacts" / "tool_returns"
        d.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() else "_" for ch in (skill_name or "tool"))[:40]
        fp = d / (safe + "_" + _uuid.uuid4().hex[:8] + ".txt")
        fp.write_text(full, encoding="utf-8")
        return full[:cap] + "\n...[truncated " + str(len(full)) + " chars; full result saved: " + str(fp) + "]"
    except Exception as exc:  # noqa: BLE001 -- offload must never break a skill
        logger.debug("tool-return offload failed (plain truncate): %s", exc)
        return full[:cap]


def _auto_approval_reason(meta: Any, args: dict) -> str | None:
    """「这一步在 ⑰ 之前会等人批准吗」——会 → 一句理由;不会 → ``None``。永不抛。

    薄封装,存在的唯一理由是**这里不能有第二份判据**:同一个问题的答案还被
    ``agents/_shared/auto_approval_mw``(发通知)和 ``core/executor`` +
    ``core/execution_context``(原来在那儿拒绝)读。三处各写一遍就是三处各自漂移。
    """
    try:
        from mast.core.auto_approval import would_have_asked

        return would_have_asked(meta, tool_name=getattr(meta, "name", ""),
                                args=args)
    except Exception:  # noqa: BLE001 — 审计标注坏了绝不能让技能失败
        logger.debug("auto_approval 判据不可用(不标注)", exc_info=True)
        return None


def _is_abort_requested(exc: BaseException) -> bool:
    """True iff *exc* is the composite layer's ``AbortRequested`` — the operator
    stopped the run. Control flow, NOT a failure to be rolled back.

    Imported lazily so this module keeps no import-time dependency on the skills
    package (and so an environment without composites still loads)."""
    try:
        from mast.skills.composite._base import AbortRequested
    except Exception:  # noqa: BLE001 — a missing composite layer is not an abort
        return False
    return isinstance(exc, AbortRequested)


class _SkillToolReturn(Command):
    """A ``Command`` whose ``update`` dict carries the skill's state side-effects.

    Returning a real ``Command`` (not a plain string) is what makes langgraph's
    ToolNode apply ``executed_skills`` / ``scan_paths`` / ``composite_progress``
    / ``error_log`` / the ToolMessage into MASTState so the checkpointer
    persists them (审查 — the previous str-subclass silently
    lost every update through ``tool.invoke``).

    Backward-compat shims for the ~175 direct-call unit tests (which call
    ``tool.func(...)`` and inspect the return):
      - ``.update``     → the state-update dict (native Command attribute);
      - ``str(result)`` → the human summary (so ``"Unknown channel" in
                          str(result)`` style assertions keep working);
      - ``x in result`` → substring test against the summary.

    ``summary`` is stashed on the instance so we don't have to dig it back out
    of the ToolMessage every time.
    """

    def __init__(self, summary: str, update_dict: dict | None = None):
        super().__init__(update=update_dict or {})
        # store the plain summary for str()/in checks (not part of Command state)
        object.__setattr__(self, "_summary", summary)

    def __str__(self) -> str:  # so `"X" in str(result)` sees the summary
        return getattr(self, "_summary", "")

    def __contains__(self, item) -> bool:  # so `"X" in result` works
        try:
            return item in self.__str__()
        except TypeError:
            return False

if TYPE_CHECKING:  # avoid hard import to v1 BaseSkill until Phase 4 wires it
    from mast.skills.base import BaseSkill  # type: ignore


# Map ParameterSpec.type strings to Python types for Pydantic field generation.
# v1 uses Python-style type names ("int"/"float"/"str"/"bool"); JSON Schema
# names also accepted for forward compat.
_TYPE_MAP: dict[str, type] = {
    # v1 (mast/core/types.py:ParameterSpec docstring)
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    # JSON Schema aliases
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool,
}


#: Parameter name → the SafetyLimits field pair that bounds it. Only names whose
#: physical meaning is unambiguous appear here; a wrong mapping would silently
#: narrow an unrelated parameter, which is worse than leaving it alone.
_ENVELOPE_FIELDS: dict[str, tuple[str, str]] = {
    "setpoint_a":   ("setpoint_min_a", "setpoint_max_a"),
    "bias_v":       ("bias_min_v", "bias_max_v"),
    "z_pos_m":      ("z_min_m", "z_max_m"),
    "center_x_m":   ("xy_min_m", "xy_max_m"),
    "center_y_m":   ("xy_min_m", "xy_max_m"),
    "x_m":          ("xy_min_m", "xy_max_m"),
    "y_m":          ("xy_min_m", "xy_max_m"),
    "width_m":      ("scan_size_min_m", "scan_size_max_m"),
    "height_m":     ("scan_size_min_m", "scan_size_max_m"),
}


def _envelope_for(spec: Any) -> "tuple[float | None, float | None]":
    """The ACTIVE safety envelope for this parameter, admin overrides included.

    Read at schema-build time from the same source SafetyGate enforces against,
    so the tool schema cannot advertise a range the gate will reject. Returns
    (None, None) for anything unmapped or if config is unavailable — a schema
    must never fail to build over this.

    KNOWN LIMIT: the schema is generated when the agent is built, so an override
    applied afterwards does not reach an already-running agent's tool schema
    (the GATE picks it up immediately either way — enforcement stays correct,
    only the advertised bound goes stale until the next rebuild).
    """
    pair = _ENVELOPE_FIELDS.get(getattr(spec, "name", ""))
    if pair is None:
        return (None, None)
    try:
        from mast.config import SafetyLimits

        lim = SafetyLimits()
        try:                              # admin overrides, same as SafetyGate
            from mast.admin.override_store import ConfigOverrideRegistry

            ovr = ConfigOverrideRegistry().get_safety_limits()
            if ovr:
                lim = lim.model_copy(update=ovr)
        except Exception:  # noqa: BLE001 — no registry / no overrides
            pass
        return (getattr(lim, pair[0], None), getattr(lim, pair[1], None))
    except Exception:  # noqa: BLE001
        return (None, None)


#: 这些量纲上,**裸尾数(0.1 … 1000)在这台仪器上不可能是合法值** —— 所以前缀是强制的,
#: 与调用方有没有想起来声明 min/max 无关。
#:
#: 为什么按量纲判而不是继续靠 min/max（2026-08-10）：``needs_strict_prefix`` 的规则是
#: 「范围整段远离 1 就强制」，而它对**未知范围答 False**（「不知道范围」不是「裸数字
#: 不可能」的证据 —— 那条规则本身是对的）。后果是：范围写得越少，防护越松。实测全仓
#: 有 23 个米制参数因为不写 min/max 而完全没有强制前缀，其中 16 个的描述还逐字向模型
#: 承诺「前缀不可省略」。
#:
#: 量纲这一维不受此影响，因为它不需要知道范围：
#:   * ``m``：STM 的长度全在压电域(≤10 µm)到粗动域(mm)之间。0.1 米没有任何合法含义。
#:   * ``A``：隧道电流 1 pA – 100 nA。0.1 安培会烧掉针尖和前放。
#: 刻意**不含** ``V``(偏压 ±10 V,``-2`` 完全合法)、``s``、``Hz``、``deg`` ——
#: 那几个量纲上 1 附近就是常用值,强制前缀会变成纯噪声。
#:
#: 反例的处理方式是 ``tests/v2/unit/agents/test_dimensioned_param_coverage.py`` 里的
#: ``METRE_PARAM_EXEMPTIONS``（要写理由），不是把这条规则改松。
_STRICT_BY_DIMENSION: frozenset[str] = frozenset({"m", "A"})


def effective_bounds(spec: Any) -> "tuple[float | None, float | None]":
    """The bounds that actually apply to *spec*: its own range ∩ the live envelope.

    THE SINGLE SOURCE for "what range does this parameter really have". Both the
    advertised strictness (the description the model reads) and the ENFORCED
    strictness (what the parser demands) must be computed from this one
    expression — see :func:`_si_params` for what it cost when they were two.

    Intersection only: this can make a bound tighter, never looser.
    """
    lo = getattr(spec, "min_value", None)
    hi = getattr(spec, "max_value", None)
    g_lo, g_hi = _envelope_for(spec)
    if isinstance(g_lo, (int, float)):
        lo = g_lo if lo is None else max(lo, g_lo)
    if isinstance(g_hi, (int, float)):
        hi = g_hi if hi is None else min(hi, g_hi)
    return lo, hi


def _si_params(meta: Any) -> "dict[str, bool]":
    """``{param_name: prefix_is_mandatory}`` for the params the model writes as text.

    A dimensioned float. Counts, indices, pixel numbers, ratios and flags are
    left alone: they have no exponent to lose, and making the model quote them
    would be noise with no protection behind it.

    ⚠️ 2026-08-10 —— **广告与执行必须由同一个表达式算出来。**
    这里原来只看 ``spec.min_value/max_value``，而 ``_schema_from_metadata`` 算的是
    **与安全包络求交之后**的 ``(lo, hi)``。于是像 ``ConfigureScan.center_x_m`` 这种
    「ParameterSpec 不写范围、范围全部来自 ``SafetyLimits.xy_*_m``」的参数出现了
    两个答案：

    * 模型读到的描述逐字写着「**前缀不可省略 —— 裸数字会被拒绝**」（schema 侧
      算出 strict=True，因为 ±1.5 µm 整段远在 1 以下）；
    * 而解析侧算出 strict=False（spec 自己的 min/max 是 None），
      ``parse_quantity(strict=False)`` 于是**欣然接受 ``"2.2"`` 并当成 2.2 米**。

    实测受影响 16 个参数 / 8 个技能，全是米制的中心坐标（ConfigureScan、FullScan、
    GridSTS、PreScanCheck、ConditionTip、AdaptiveSTS_GP、RunGridExperiment、
    DrawScanMarker）。这些参数的 ParameterSpec 恰恰因为「范围由全局包络统一给」
    才不写 min/max —— **写得越规范，防护掉得越干净。**

    修法刻意**不是**去那 8 个文件里各补一遍 ±1.5 µm：同一个物理量有两个来源迟早
    各自漂移（与 ``noble_tip_workflow`` 不收污染半径同一条理由）。
    """
    out: dict[str, bool] = {}
    for spec in getattr(meta, "parameters", None) or []:
        if getattr(spec, "type", "") != "float":
            continue
        unit = (getattr(spec, "unit", "") or "").strip()
        if not unit:
            continue
        if getattr(spec, "allowed_values", None):
            continue                     # an enum is already exact
        out[spec.name] = (needs_strict_prefix(*effective_bounds(spec))
                          or unit in _STRICT_BY_DIMENSION)
    return out


def _coerce_si_params(meta: Any, kwargs: "dict[str, Any]") -> "tuple[dict, list[str]]":
    """String quantities → floats. Returns ``(kwargs, errors)``; never raises."""
    si = _si_params(meta)
    if not si:
        return kwargs, []
    errors: list[str] = []
    out = dict(kwargs)
    for name, strict in si.items():
        if name not in out or out[name] is None:
            continue
        try:
            out[name] = parse_quantity(out[name], strict=strict, what=name)
        except SIParseError as exc:
            errors.append(str(exc))
    return out, errors


def _schema_from_metadata(meta: Any) -> type[BaseModel]:
    """Build a Pydantic args schema from the skill's ParameterSpec list.

    Besides the skill's declared parameters, the schema carries one hidden
    ``tool_call_id`` field annotated with ``InjectedToolCallId``. langgraph
    populates it at tool-call time with the calling AI message's tool_call.id
    and — crucially — keeps it OUT of the tool schema shown to the LLM (it is
    an injected arg, not a model-facing one). ``_run`` reads it back from
    ``kwargs`` so the returned ``Command``'s ToolMessage can carry the matching
    id (required by ToolNode's command validation) — this is what makes the
    state side-effects in ``Command.update`` actually persist ().
    """
    fields: dict[str, Any] = {}
    for spec in meta.parameters:
        py_type = _TYPE_MAP.get(getattr(spec, "type", "string"), str)
        default = ... if getattr(spec, "required", True) else getattr(spec, "default", None)
        description = getattr(spec, "description", "") or ""
        if getattr(spec, "unit", None):
            description = f"{description} (unit: {spec.unit})".strip()

        # Surface the declared range as JSON-Schema minimum/maximum.
        #
        # ParameterSpec has carried min_value/max_value all along and NONE of it
        # reached the model: the generated args_schema had only a description.
        # A range stated in prose is something the model may skip; `minimum` /
        # `maximum` are structural, and both the model and several providers'
        # tool-call validators honour them.
        #
        # 2026-07-27, real end-to-end run: instrument_control called
        # SetSetpoint(setpoint_a=1.5) — 1.5 AMPERES for a 1.5 nA intent. That
        # parameter's description already spelled out the range AND used 1.5
        # itself as the worked counter-example ("A bare value like 1.5 means 1.5
        # AMPERES … pass 1.5e-9, not 1.5"), and min_value=1e-12 /
        # max_value=100e-9 were declared. The model still sent 1.5, eleven times.
        # Prose was not enough; the machine-readable bound was simply absent.
        #
        # json_schema_extra, NOT Field(ge=/le=): ge/le would make Pydantic reject
        # the call first, and the model would then see a generic validation
        # error instead of SafetyGate's message — which is written to teach the
        # correction ("100 pA = 1e-10 … 切勿重试相同数值"). Keep the constraint
        # visible, keep the enforcement (and the wording) where it already works.
        # Tighten against the ACTIVE safety envelope so the schema never
        # advertises a value the gate will refuse. ParameterSpec ships the
        # factory range (setpoint 1 pA–100 nA); an operator whose rig only
        # accepts 0–10 nA narrows SafetyLimits.setpoint_max_a in
        # 高级管理 → 全局安全限制, and without this the model would still be
        # told 100 nA is allowed and would keep proposing rejected values.
        # Intersection only — this can make a bound tighter, never looser.
        #
        # 2026-08-10: 这一段抽成 `effective_bounds()`，因为 `_si_params` 少了它 ——
        # 广告说「前缀不可省略」而解析放行裸数字，两处必须同源。
        lo, hi = effective_bounds(spec)
        # ge/le, NOT json_schema_extra.
        #
        # First attempt used json_schema_extra={"minimum":…, "maximum":…}. It
        # showed up in model_json_schema() — and was then DROPPED by
        # langchain_core.utils.function_calling.convert_to_openai_tool, which is
        # what actually builds the payload sent to the provider. Measured:
        #
        #     json_schema_extra  -> {'type': 'number'}
        #     Field(ge=, le=)    -> {'type':'number','minimum':1e-12,'maximum':1e-07}
        #
        # So the bound reached a local schema dump and never the model. The very
        # thing this change exists to prevent, one layer further out — and it
        # took a live run (instrument_control then sent setpoint_a = 1.0 A) to
        # notice, because a Pydantic-level check looked like it had worked.
        #
        # ge/le do enforce as well as advertise, and a raw ValidationError would
        # replace SafetyGate's teaching text with pydantic boilerplate. That is
        # handled by the handle_validation_error hook wired in wrap_skill below,
        # which renders the same actionable wording.
        # allowed_values → a real enum in the schema.
        #
        # ParameterSpec.allowed_values was in exactly the position min/max were in
        # before the note above: declared on a dozen skills, enforced by
        # validate_params, and INVISIBLE to the model, which saw a bare `string`
        # and had to infer the legal set from prose. Same fix, same reason — a
        # constraint the model can only learn by reading is a constraint it can
        # skip.
        #
        # Literal is how an enum reaches the provider payload (json_schema_extra
        # would be dropped by convert_to_openai_tool, exactly as it was for the
        # numeric bounds). Pydantic will not accept ge/le on a Literal, and does
        # not need to: an explicit set is strictly stronger than the range it
        # sits inside.
        allowed = getattr(spec, "allowed_values", None)
        if allowed and all(
            isinstance(v, (str, int, bool)) and not isinstance(v, float)
            for v in allowed
        ):
            fields[spec.name] = (
                Literal[tuple(allowed)],  # type: ignore[valid-type]
                Field(default, description=description),
            )
            continue

        # A dimensioned quantity is asked for as a STRING, not a number.
        #
        # Measured on this exact path (kimi-k3, tool calling, tool_choice=auto,
        # 2026-08-04, 12 trials each):
        #
        #     number-typed argument   0/12 correct — 3e-12 arrived as 3,
        #                             1.8e-07 as 1.8, 1.5e-10 as 1.5
        #     string-typed argument  12/12 byte-identical
        #
        # The fault is in the constrained-decoding grammar for JSON *numbers* on
        # the provider side (sci-notation-bench reproduced it across every Kimi
        # model and every reasoning level, while DeepSeek and GLM pass the same
        # values through the same response_format unharmed). Nothing on this side
        # can fix that grammar; declaring the field a string routes around it.
        #
        # The cost is that ge/le no longer reach the schema, so the range moves
        # into the description where the model still reads it — and enforcement
        # is unchanged, because _run parses the string back to a float BEFORE
        # validate_params and SafetyGate ever see it. A bound the model can see
        # on a channel that corrupts every value is worth less than a channel
        # that works.
        if py_type is float and getattr(spec, "unit", ""):
            # 与 `_si_params` **同一个表达式**。两处曾各算各的，结果描述里写着
            # 「前缀不可省略」而解析器放行裸数字（2026-08-10，16 个米制参数）。
            strict = _si_params(meta).get(spec.name, needs_strict_prefix(lo, hi))

            def _show(v: float, _strict: bool = strict) -> str:
                """Show the bound in the same notation the model is asked to use.

                A strict parameter is told to write '3p', so its range has to
                read '1f … 1u' — printing "0 … 0.001" there would be telling it
                the answer in the one format the field refuses.

                For a lenient parameter the reverse holds: ``format_si`` never
                emits a bare mantissa (its output must be re-parseable), so a
                ±10 V envelope would render as "-10000m … 10000m" — correct, and
                unreadable. Near unity, plain wins.
                """
                if v == 0:
                    return "0"
                if _strict:
                    return format_si(v)
                return f"{v:g}" if 1e-3 <= abs(v) < 1e4 else format_si(v)

            hint = [description] if description else []
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                hint.append(f"范围 {_show(lo)} … {_show(hi)} {spec.unit}")
            elif isinstance(hi, (int, float)):
                hint.append(f"最大 {_show(hi)} {spec.unit}")
            elif isinstance(lo, (int, float)):
                hint.append(f"最小 {_show(lo)} {spec.unit}")
            if strict:
                hint.append(
                    "**写成带 SI 前缀的字符串**,如 '3p'(=3e-12)、'180n'(=1.8e-7)。"
                    "前缀不可省略 —— 裸数字会被拒绝,因为一个丢了量级的裸数字仍然是"
                    "合法数字,错一万亿倍也无人察觉。"
                )
            else:
                hint.append(
                    "**写成字符串**,如 '0.05'、'5e-3' 或带 SI 前缀的 '5m'"
                    f"(单位 {spec.unit})。"
                )
            # Same shape as every other optional field here: type + a None
            # default. Not Optional[str] — that would let the model pass a bare
            # null where a quantity belongs.
            fields[spec.name] = (str, Field(default, description=" ".join(hint)))
            continue

        constraints: dict = {}
        numeric = py_type in (int, float)
        if numeric and isinstance(lo, (int, float)) and not isinstance(lo, bool):
            constraints["ge"] = lo
        if numeric and isinstance(hi, (int, float)) and not isinstance(hi, bool):
            constraints["le"] = hi

        if constraints:
            fields[spec.name] = (
                Annotated[py_type, Field(**constraints)],
                Field(default, description=description),
            )
        else:
            fields[spec.name] = (py_type, Field(default, description=description))
    # Hidden injected field — never required from the LLM (default "").
    fields["tool_call_id"] = (
        Annotated[str, InjectedToolCallId],
        Field(default=""),
    )
    name = f"{meta.name}Args"
    return create_model(name, **fields)  # type: ignore[arg-type]


def _effective_metadata(meta: Any) -> Any:
    """The skill's metadata with the admin's per-skill override applied.

    WHY THIS EXISTS
    ===============
    ``SkillRegistry._get_metadata`` has applied skill overrides for a long time,
    so ``registry.list_skills()`` — the skill menu, the GUI, and the approval
    decision inside ``ExecutionContext.run`` — all see them. The AGENT tool path
    did not. ``instrument_control/tools.py`` iterates ``registry.list_skills()``
    but keeps only the skill CLASS from each entry, and ``wrap_skill`` then asks
    the freshly-built instance for its metadata again — the un-overridden one.

    Everything the agent path derives from metadata therefore ignored overrides:
    the model-facing JSON schema, the ``skill_metadata`` handed to SafetyGate, and
    ``validate_params``. So an admin who tightened a bound through
    ``POST /api/skills/{name}/override`` got a stored file, a successful response,
    and no enforcement on the model-facing path. The wrapper must apply the same
    override before deriving the schema, safety metadata and validation.

    Applied here rather than at the call sites because ``wrap_skill`` has 30-odd
    of them (the composite and paper tool factories among them) and needs no
    registry reference to do this — the override registry is a singleton and
    ``apply_skill_metadata_override`` is a pure function.

    Fail-open: a broken override file must never stop an agent from building, but
    it must not pass silently either.

    KNOWN LIMIT (shared with ``_envelope_for``): this runs at tool-build time, so
    an override applied to an already-running agent reaches it at the next
    rebuild/restart, not immediately (KNOWN_ISSUES §1.1 — nothing subscribes to
    the reload hook).
    """
    name = getattr(meta, "name", None)
    if not name:
        return meta
    try:
        from mast.admin.override_store import (
            ConfigOverrideRegistry,
            apply_skill_metadata_override,
        )

        # .get() — the singleton, so tests that point the registry at a tmp dir
        # are honoured. Constructing one directly here would silently read the
        # real config directory instead.
        ovr = ConfigOverrideRegistry.get().get_skill_override(name)
        if not ovr:
            return meta
        return apply_skill_metadata_override(meta, ovr)
    except Exception:  # noqa: BLE001 — a bad override must not break tool build
        logger.warning(
            "skill override for %s could not be applied; using the declared "
            "metadata", name, exc_info=True,
        )
        return meta


def wrap_skill(
    skill_cls: type["BaseSkill"],
    context_provider: Callable[[], Any],
    *,
    post_hook: "Callable[[str, dict, Any], dict | None] | None" = None,
    recorder: "Callable[[dict], None] | None" = None,
) -> StructuredTool:
    """Wrap a v1 BaseSkill subclass as a LangGraph-compatible StructuredTool.

    Args:
      skill_cls:        The skill class (uninstantiated) to wrap.
      context_provider: Callable that returns a fresh ExecutionContext per
                        invocation (carries pool / state / approval_source).
      post_hook:        OPTIONAL producer-side glue ``(skill_name, data, ctx) ->
                        dict|None`` run AFTER a successful execute. Used to render
                        a figure + register it in records + emit a vision-buffer
                        event WITHOUT the LLM tool itself touching the buffer
                        (the 'agents never write the buffer' invariant stays —
                        the hook is adapter glue closed over by the GUI, not a
                        model-facing tool). May return ``{"scan_paths": [...]}``
                        to merge into the state update. MUST NOT raise.
    """
    # Defer instantiation: some skills hold no state, others might. Always
    # create one instance for metadata extraction.
    skill = skill_cls()
    _declared = skill.metadata() if callable(skill.metadata) else skill.metadata  # type: ignore
    meta = _effective_metadata(_declared)
    if meta is not _declared:
        # Keep the INSTANCE in step with the schema. ``validate_params`` and
        # ``check_preconditions`` both go through ``self.metadata()``, so without
        # this an override that WIDENS a bound would be advertised and accepted by
        # pydantic, then rejected by validate_params reading the declared spec —
        # the tool would contradict itself. (Tightening alone would have hidden
        # the bug: pydantic rejects first, so validate_params never disagrees.)
        skill.metadata = lambda _m=meta: _m  # type: ignore[method-assign]
    args_schema = _schema_from_metadata(meta)

    safety_level = getattr(meta, "safety_level", None)
    danger_level = getattr(safety_level, "name", "AUTO") if safety_level else "AUTO"

    def _run(**kwargs: Any) -> _SkillToolReturn:
        """Run the skill. Returns ``_SkillToolReturn`` — a ``Command`` subclass.

        Agent-path semantics: because the return is a real ``langgraph.Command``,
        ToolNode applies ``command.update`` (executed_skills / scan_paths /
        composite_progress / error_log / the ToolMessage) into MASTState, so the
        checkpointer actually persists the side-effects (). The
        ToolMessage carries the injected ``tool_call_id`` so ToolNode's command
        validation accepts it.

        Legacy unit-test semantics: result.update is the same update dict, so
        old tests asserting `result.update["executed_skills"][0] == "X"` keep
        working; `str(result)` / `"x" in result` still return the human summary.

        Composite-graph semantics: if this skill subclasses
        :class:`CompositeSkillGraph`, the executor emits per-step
        progress and we lift the final ``CompositeProgress.to_dict()``
        snapshot into ``state.composite_progress[<skill_name>]`` so the
        SqliteSaver checkpointer flushes it.
        """
        # Strip framework-injection kwargs that legacy tests may pass.
        tool_call_id = kwargs.pop("tool_call_id", "") or ""
        prior_state = kwargs.pop("state", None) or {}

        # Dimensioned parameters arrive from the model as STRINGS (see
        # _schema_from_metadata). Turn them back into floats here, before
        # anything else looks at them, so validate_params, the skill's execute,
        # the recorder and every downstream consumer see exactly what they always
        # saw. Internal callers that already hold floats pass through untouched.
        kwargs, si_errors = _coerce_si_params(meta, kwargs)
        if si_errors:
            # `precondition_failed:` on purpose — StallGuard groups repeated
            # failures by that signature, so a model that keeps re-sending the
            # same unparseable magnitude escalates to a forced stop instead of
            # looping.
            summary = f"[{meta.name}] precondition_failed: {'; '.join(si_errors)}"
            return _SkillToolReturn(
                summary,
                _failure_update(meta.name, summary, tool_call_id, also_log=True),
            )

        ctx = context_provider()

        # ── 0. ABORT GATE (2026-07-11) ───────────────────────────────────
        # This wrapper is the ONE door every LLM tool call enters a skill
        # through, so it is where an abort must stop the agent from starting
        # anything new. Before this, abort was honoured only BETWEEN LangGraph
        # super-steps: the agent's ReAct loop happily kept issuing tool calls
        # (SetBias → StartScan → …) after 中止 was pressed, and each one reached
        # the instrument. Refuse to start, and tell the model plainly to stop
        # rather than retry (a retry loop would just burn the step budget).
        _abort_check = getattr(ctx, "check_abort", None)
        if callable(_abort_check):
            try:
                _aborted = bool(_abort_check())
            except Exception:  # noqa: BLE001 — a broken check must not block work
                _aborted = False
            if _aborted:
                # 中止可能来自急停、事件钩子、环境告警或会话停止。
                # 必须读取已记录的原因；未知来源不能被写成用户操作。
                _why = ""
                _reason_fn = getattr(ctx, "abort_reason", None)
                if callable(_reason_fn):
                    try:
                        _why = str(_reason_fn() or "")
                    except Exception:  # noqa: BLE001
                        _why = ""
                _who = f"中止原因:{_why}" if _why else (
                    "**没有留下中止原因** —— 别假定是人停的:急停、E_STOP 事件、"
                    "环境告警都会置这个状态。去看服务日志里最近的 CRITICAL 行。")
                summary = (
                    f"[{meta.name}] aborted: 本次运行处于中止状态——拒绝执行新的"
                    f"仪器动作。{_who}"
                    " STOP now: do not retry, do not call another tool; report that"
                    " the run was aborted."
                )
                logger.warning("abort active — refusing skill %s", meta.name)
                # also_log: the abort belongs in the run's error trail — it is
                # what explains why the agent stopped doing anything.
                return _SkillToolReturn(
                    summary,
                    _failure_update(meta.name, summary, tool_call_id, also_log=True))

        # ── 0b. SAMPLE GATE (2026-07-28) ─────────────────────────────────
        # Data-producing skills need a sample to belong to: a .sxm nobody can
        # attribute to a piece of material is a measurement nobody can use
        # later. Gated here for the same reason the abort gate is — this is the
        # ONE door every LLM tool call enters a skill through, which is harder
        # coverage than SafetyGateMiddleware (that one is installed per-graph
        # and can be absent).
        #
        # Reads, stops, retracts and E-STOP are never gated; see
        # mast.core.sample_gate for why the ambiguous case resolves to ALLOW.
        try:
            from mast.core.sample_gate import check_sample_scope
            from mast.logging.experiment_log import get_active_log
            _gate_msg = check_sample_scope(meta, meta.name, get_active_log())
        except Exception:  # noqa: BLE001 — a broken gate must not block work
            _gate_msg = None
        if _gate_msg and not getattr(ctx, "_scope_admitted", False):
            logger.warning("sample gate — refusing %s (no active sample)", meta.name)
            return _SkillToolReturn(
                _gate_msg,
                _failure_update(meta.name, _gate_msg, tool_call_id, also_log=True))
        # Admitted here → this composite's sub-steps inherit the admission, so a
        # scope change mid-composite cannot strand it half-executed.
        try:
            ctx._scope_admitted = True
        except Exception:  # noqa: BLE001 — some contexts are read-only mocks
            pass

        # Composite-graph resume hook: if the prior state carried a
        # progress snapshot for this skill, expose a get_progress() so
        # GraphExecutor picks up the resume point. We *do not* mutate
        # the supplied state object.
        _wire_progress_bridge(ctx, meta.name, prior_state)

        # Capture state and start the timer before validation and precondition
        # gates so refusals can also be recorded. The snapshot is read-only.
        try:
            current_state = ctx.state.snapshot() if hasattr(ctx, "state") and ctx.state is not None else None
        except Exception:
            current_state = None
        # ── Training/usage recorder (opt-in, fail-safe; RFC P1) ──────────
        # GUI injects ``recorder`` (a fire-and-forget closure over the TraceSink).
        # It captures the skill call for the training-log trajectory. A recorder
        # failure must NEVER affect the skill result → wrapped in try/except.
        _t0 = time.monotonic()

        def _emit_record(*, success: bool, summary: Any, error: Any = "",
                         data: dict | None = None, rolled_back: bool = False,
                         result: Any = None) -> Any:
            """Hand this call to the records layer and return whatever it gives
            back (the v2 action id, or None). The id is what lets a post_hook
            attach its artifacts to THIS action instead of inventing a second
            action row for the same skill call."""
            if recorder is None:
                return None
            try:
                _d = data or {}
                # Approval provenance, not danger level. ExecutionContext carries
                # an explicit source on the manual/HITL paths; on this seam — the
                # LLM tool-call path, which is what wrap_skill IS — the caller is
                # the model. The v1 column used to be filled with the skill's
                # DANGER LEVEL ("CONFIRM"/"AUTO"), which answers a different
                # question ().
                _appr = str(getattr(ctx, "_approval_source", "") or "")
                return recorder({
                    "skill": meta.name,
                    "skill_version": getattr(meta, "version", ""),
                    "danger_level": danger_level,
                    "params": {k: v for k, v in kwargs.items()},
                    "tool_call_id": tool_call_id,
                    "success": bool(success),
                    "error": str(error or "")[:1000],
                    "summary": str(summary or "")[:1000],
                    "duration_ms": int((time.monotonic() - _t0) * 1000),
                    "artifact_path": (_d.get("path") or _d.get("file_path")
                                      or _d.get("sxm_path")),
                    "rolled_back": bool(rolled_back),
                    "state_before": current_state,
                    # Preserve returned data, calls and state in the payload;
                    # read-only skills may carry their entire result here.
                    # The records layer decides which fields it can persist.
                    "data": _d,
                    "state_after": getattr(result, "state_after", None),
                    "nanonis_calls": list(getattr(result, "nanonis_calls", None) or []),
                    "elapsed_s": getattr(result, "elapsed_s", None),
                    "approval_source": _appr if _appr in ("human", "llm") else "llm",
                    # ⑰(2026-08-08):这一步在旧策略下会停下来等人批准吗?
                    # 非 None ⇒ 记录层给这条 action 补一行 ``approvals``,
                    # approver_kind=automated_policy、method=auto_executed_notified。
                    #
                    # 为什么还写这张表:审批链路割掉之后 approvals 会变成一张只有
                    # 历史行的死表,而事后查「这个 DANGEROUS 动作是谁准的」时,
                    # **空表和「没人准过」长得一模一样** —— 2026-07-27 的取证正好
                    # 栽在这个形状上。判据与中间件、executor 共用同一个函数,
                    # 三处不会各自漂移。
                    "auto_approved": _auto_approval_reason(meta, kwargs),
                })
            except Exception:  # logging must never break the skill
                logger.debug("skill recorder failed (swallowed)", exc_info=True)
                return None

        # ── Gates (below the recorder ON PURPOSE) ────────────────────────────
        # These two used to sit ABOVE the recorder's definition and return
        # straight out, so **a call rejected by param validation or by a
        # precondition was written to no records store at all**. The records
        # layer exists to answer "为什么什么都没发生", and a refused call is
        # precisely that question. Nothing else in the system keeps it: the
        # rejection reaches the model as a tool message and is then gone.
        #
        # (2026-07-27 forensics found the record layer empty of failures for a
        # different reason — the post_hook only ran on success — and that half is
        # fixed. This half stayed hidden because the session's five failures were
        # all execution-time, which DO pass through here.)

        # 1. Per-skill param validation
        violations = skill.validate_params(kwargs) if hasattr(skill, "validate_params") else []
        if violations:
            summary = f"[{meta.name}] precondition_failed: {'; '.join(violations)}"
            # Dimensioned parameters no longer carry ge/le (they are strings in
            # the schema now), so an out-of-range value lands HERE instead of in
            # pydantic — and would otherwise arrive as a bare
            # "above maximum 1e-07" with none of the teaching text that made the
            # 2026-07-27 setpoint=1.5 loop stop. Same wording, same place in the
            # sequence; only the layer that caught it moved.
            if any("minimum" in v or "maximum" in v or "above" in v or "below" in v
                   for v in violations):
                summary += " " + _explain_validation(None)
            _emit_record(success=False, summary=summary, error="; ".join(violations))
            return _SkillToolReturn(summary, _failure_update(meta.name, summary, tool_call_id))

        # 2. Per-skill precondition check
        if current_state is not None and hasattr(skill, "check_preconditions"):
            try:
                unmet = skill.check_preconditions(current_state)
            except Exception as _pe:
                # Preconditions are SAFETY-relevant (e.g. z_controller_off must
                # hold before MotorMove — tip must be withdrawn). A crash here
                # must NOT silently let the hardware action through: fail-closed,
                # refuse execution, and surface a clear error.
                summary = (
                    f"[{meta.name}] precondition_failed: precondition check raised "
                    f"{type(_pe).__name__}: {_pe}"
                )
                logger.warning(
                    "check_preconditions for %s raised %s: %s — failing closed (refusing execution)",
                    meta.name, type(_pe).__name__, _pe,
                )
                _emit_record(success=False, summary=summary, error=summary)
                return _SkillToolReturn(
                    summary,
                    _failure_update(meta.name, summary, tool_call_id, also_log=True),
                )
            if unmet:
                summary = f"[{meta.name}] precondition_failed: {'; '.join(unmet)}"
                _emit_record(success=False, summary=summary, error="; ".join(unmet))
                return _SkillToolReturn(summary, _failure_update(meta.name, summary, tool_call_id))

        # 4. Execute
        #
        # INSTRUMENT ARBITRATION (2026-07-28, dispatch audit 致命一): this is the
        # agent-side entry to the one physical instrument, and the group chat's
        # IC agent and the private-chat IC agent both come through here on the
        # SAME pool with no knowledge of each other. Re-entrant per thread (a
        # composite's sub-steps re-take it through ExecutionContext.run);
        # skipped for reads and for stop/retract remedies.
        from mast.core.instrument_lock import InstrumentBusy, hold_for_skill

        try:
            _owner = getattr(ctx, "owner", "") or "agent"
            with hold_for_skill(meta, meta.name, _owner):
                # LOCK-IN 卫生(2026-08-05 定案「用完必须及时关」)。
                #
                # 这里,而不是每个技能里抄一遍:这是 agent 侧唯一的入口,而且已经
                # 在令牌里 —— 一次工具调用检查一次,composite 的子步骤走
                # ExecutionContext.run 不会重复触发。抄进 N 个技能的版本会漂:
                # 下一个新扫图技能不会记得调它。
                #
                # 自动关而不是拒绝(零打断),但**留痕进 data** —— 用户要知道是谁
                # 动了他的调制。判据在 _preflight.wants_modulation_off:写类 + 自己
                # 不用 lock-in + 标签落在物理动作那几族。
                _mod_note = None
                try:
                    from mast.skills.composite._preflight import (
                        ensure_modulation_off, wants_modulation_off,
                    )
                    if wants_modulation_off(meta):
                        _mod_note = ensure_modulation_off(ctx, skill_name=meta.name)
                except Exception:  # noqa: BLE001 — 卫生动作绝不带走技能
                    logger.debug("modulation preflight failed (swallowed)",
                                 exc_info=True)
                result = skill.execute(ctx, kwargs)
            data = getattr(result, "data", {}) or {}
            if _mod_note:
                # 进 data 而不是只进日志:结果里看不到的动作,用户事后查不到。
                data.update(_mod_note)
                try:
                    result.data = data
                except Exception:  # noqa: BLE001
                    pass
            success = getattr(result, "success", True)
            error_msg = getattr(result, "error", "") or ""
            # Write-back: a successful skill whose result carries state fields
            # (ZControllerOnOff→z_controller_on, StartScan/StopScan→scan_running,
            # SetBias→bias_v, …) patches the cached InstrumentState immediately, so
            # the NEXT skill's precondition check sees the new value instead of a
            # stale snapshot. Without this, e.g. StartScan failed its
            # z_controller_on precondition right after ZControllerOnOff(True)
            # because the ~1s monitor refresh hadn't caught up — the agent then
            # retried in a loop. Best-effort: a cache patch must never affect the
            # skill result.
            #
            # 2026-08-10:实现搬进 ``mast.core.state.patch_state_from_result``,
            # 因为 composite 的子步骤走 ``ExecutionContext.run``、**不经过这里**,
            # 于是一个子步骤改了硬件之后下一个子步骤仍读旧值(用真类复现过)。
            # 两条路径现在共用同一份写回 —— 两处各写一遍正是本仓反复栽的形状。
            if success:
                from mast.core.state import patch_state_from_result

                patch_state_from_result(getattr(ctx, "state", None), data,
                                        what=meta.name)
            summary = getattr(result, "summary", None)
            # ``data`` 是否已经原样进了摘要 —— 只有「没设 summary、且不是带错误
            # 信息的失败」那一支会这样。记下来,免得下面再附一遍。
            data_inlined = False
            if summary is None:
                if not success and error_msg:
                    summary = f"[{meta.name}] failed: {error_msg}"
                else:
                    summary = str(_strip_audit_only(data)) if data else f"[{meta.name}] ok"
                    data_inlined = bool(data)
            if not data_inlined:
                summary += _explanation_suffix(data, success)
            summary = _offload_long_summary(meta.name, summary)

            artifact_path = data.get("path") or data.get("file_path") or data.get("sxm_path")

            # ── 图像通道 ───────────────────────────────────────────────
            # 技能渲染出来的图,**路径**挂在 ToolMessage 上。这是全仓唯一的
            # 技能→模型出口,所以接这一处,六个 agent 全都受益 —— 五变一,不是五变六。
            #
            # 这里挂**路径不挂像素**,而且 content 仍然是那个 str:
            #   * messages 会被 SqliteSaver 每轮持久化并重放,base64 停在这里就会
            #     被反复写盘一整个 session(「Checkpoint 不放 tensor / 大对象」);
            #   * 而且这样一来,**降级是默认状态**:``vision_mw`` 没装、模型不支持
            #     视觉、图读不出来 —— 任何一种情况下,发出去的东西与接线前逐字节
            #     相同,不需要谁去「记得关掉」。像素只在出站那一刻由 vision_mw
            #     materialize 成 data URI,且绝不写回 state。
            #
            # 键名用 vision_mw 的常量而不是字面量:生产方在记、消费方读不到,
            # 是这个仓踩过的静默失败形状,而拼错一个字符就是那个形状。
            images = [str(p) for p in (getattr(result, "images", None) or []) if p]
            tool_kwargs: dict[str, Any] = {}
            if images:
                from mast.agents._shared.vision_mw import IMAGES_KEY

                tool_kwargs[IMAGES_KEY] = images

            update_dict: dict[str, Any] = {
                "executed_skills": [meta.name],
                "messages": [
                    ToolMessage(
                        content=summary,
                        tool_call_id=tool_call_id,
                        name=meta.name,
                        status="error" if not success else "success",
                        additional_kwargs=tool_kwargs,
                    )
                ],
            }
            if artifact_path:
                update_dict["scan_paths"] = [str(artifact_path)]
                # Publish the measurement on the inter-agent channel too, so
                # data_processing is TOLD what was just measured instead of
                # having to infer it from the handoff sentence "扫描已完成" and
                # then go hunting through the scan registry. scan_paths is an
                # append-only audit list with no readers; this is the pointer a
                # downstream agent can actually act on.
                if success:
                    try:
                        from mast.agents._shared.artifact_channel import ScanResult
                        update_dict["last_scan"] = ScanResult(
                            handle=str(data.get("handle") or meta.name),
                            status="done",
                            sxm_path=str(artifact_path),
                            duration_s=round(time.monotonic() - _t0, 3),
                            warnings=[str(w) for w in (data.get("warnings") or [])][:5],
                        )
                        _sid = data.get("scan_id")
                        if _sid:
                            update_dict["scan_id"] = str(_sid)
                    except Exception as exc:  # noqa: BLE001 — never fail a skill
                        logger.debug("last_scan publish failed for %s: %s",
                                     meta.name, exc)
            if not success and error_msg:
                update_dict["error_log"] = [f"{meta.name}: {error_msg}"]
            # Composite-graph: lift the final progress snapshot into
            # MASTState.composite_progress so the checkpointer persists it.
            prog_snapshot = data.pop("_progress", None)
            if isinstance(prog_snapshot, dict):
                update_dict["composite_progress"] = {meta.name: prog_snapshot}
            # Record both successful and failed outcomes before the success-only
            # rendering hook. The hook can then attach its figure to the existing
            # action instead of creating a duplicate row.
            _action_id = _emit_record(success=success, summary=summary,
                                      error=error_msg, data=data, result=result)
            # Producer-side post-skill hook (glue, NOT the model tool body):
            # render+record+emit. MUST NOT raise — a hook failure can't turn a
            # successful hardware action into a tool error.
            if post_hook is not None and success:
                try:
                    if _action_id:
                        try:
                            ctx.records_action_id = _action_id  # type: ignore[attr-defined]
                        except Exception:  # noqa: BLE001 — slotted/frozen ctx
                            pass
                    extra = post_hook(meta.name, data, ctx)
                    if isinstance(extra, dict) and extra.get("scan_paths"):
                        update_dict.setdefault("scan_paths", []).extend(
                            str(p) for p in extra["scan_paths"])
                except Exception as _he:  # pragma: no cover - defensive
                    logger.warning("post_hook for %s failed: %s", meta.name, _he)
            return _SkillToolReturn(summary, update_dict)
        except Exception as e:
            # P2-G: GraphInterrupt is CONTROL FLOW — a composite's human node
            # pausing the run for an operator decision. It must bubble so
            # LangGraph pauses the graph, and it must NEVER trigger rollback
            # (rolling back hardware on a pause would undo legitimate steps;
            # the step progress is already persisted in the sidecar).
            try:
                from langgraph.errors import GraphInterrupt
                if isinstance(e, GraphInterrupt):
                    raise
            except ImportError:  # pragma: no cover — langgraph pinned in v2
                pass
            # InstrumentBusy means NOTHING was sent to the instrument — another
            # entry point holds the token. Rolling back here would drive the
            # hardware to undo work that never happened, on a link somebody else
            # is using. Report and stop.
            if isinstance(e, InstrumentBusy):
                summary = f"[{meta.name}] instrument_busy: {e.message()}"
                _emit_record(success=False, summary=summary, error=e.message())
                return _SkillToolReturn(
                    summary,
                    _failure_update(meta.name, summary, tool_call_id, also_log=True),
                )
            # ...and so is AbortRequested (2026-07-11). The operator pressing 中止
            # is not a skill FAILURE, so it must not fire the skill's rollback:
            # rollback undoes completed work the abort never touched — re-driving
            # the very hardware the operator just told us to stop. graph_executor
            # already converts an abort raised INSIDE a plan into a clean
            # aborted-progress return; this is the outer guard for an abort raised
            # anywhere else in a composite (a step helper, a skill's own execute()).
            if _is_abort_requested(e):
                # 同上:不许把「谁停的」写死成用户。见入口那道闸门的注释。
                _why2 = ""
                _rf2 = getattr(ctx, "abort_reason", None)
                if callable(_rf2):
                    try:
                        _why2 = str(_rf2() or "")
                    except Exception:  # noqa: BLE001
                        _why2 = ""
                summary = (
                    f"[{meta.name}] aborted: 本次运行被中止——已停止，未回滚。"
                    + (f"中止原因:{_why2}" if _why2 else "未留下中止原因。")
                    + " STOP now: do not retry, do not issue further instrument tools."
                )
                _emit_record(success=False, summary=summary,
                             error=f"aborted: {_why2}" if _why2 else "aborted")
                return _SkillToolReturn(
                    summary,
                    _failure_update(meta.name, summary, tool_call_id, also_log=True),
                )
            try:
                if hasattr(skill, "rollback"):
                    skill.rollback(ctx, kwargs)
            except Exception:
                pass
            summary = f"[{meta.name}] rolled_back: {type(e).__name__}: {e}"
            _emit_record(success=False, summary=summary,
                         error=f"{type(e).__name__}: {e}", rolled_back=True)
            return _SkillToolReturn(
                summary,
                _failure_update(meta.name, summary, tool_call_id, also_log=True),
            )

    # NOTE: Nanonis manual hints are DELIBERATELY not appended per-skill here.
    # An earlier revision did so, but ~176 IC tools collapse to only 8 distinct
    # module hints (hints resolve at manual-MODULE granularity), so per-skill
    # injection paid ~9k duplicated tokens on EVERY IC turn (no cross-provider
    # prompt caching) for ~470 tokens of real content (review COST-1). The 8
    # deduped module hints now live ONCE in the IC system prompt
    # (mast.knowledge.nanonis_manual.modules_index, wired in instrument_control/
    # prompts.py); per-skill depth is fetched on demand via the `nanonis_manual`
    # tool (Integration B). Keep tool descriptions hint-free.

    # Use @tool decorator factory: it detects Annotated[InjectedToolCallId] and
    # Annotated[InjectedState] correctly during ToolNode invocation. Passing
    # the same _run via StructuredTool.from_function silently dropped those
    # injections, breaking agent-mediated tool calls (test_graph_smoke caught it).
    def _explain_validation(exc: Exception) -> str:
        """Render a bounds rejection the way SafetyGate would.

        The ge/le constraints that put `minimum`/`maximum` into the provider's
        tool schema also make pydantic reject an out-of-range call before _run
        is entered. Left alone, the model would receive pydantic boilerplate
        ("Input should be less than or equal to 1e-07") instead of something it
        can act on. This keeps the schema honest AND the message useful.

        The `precondition_failed:` prefix is deliberate — StallGuard groups
        repeated failures by that signature, so a model that keeps re-sending
        the same bad value still escalates to a forced stop.
        """
        rows = []
        for s in (meta.parameters or []):
            # Enum first: for a parameter with an explicit set, the legal values
            # ARE the answer — printing the numeric range it happens to sit
            # inside would be the less useful half of the truth.
            choices = getattr(s, "allowed_values", None)
            if choices:
                rows.append(f"{s.name} ∈ {{" + ", ".join(repr(c) for c in choices) + "}")
                continue
            lo, hi = getattr(s, "min_value", None), getattr(s, "max_value", None)
            if lo is None and hi is None:
                continue
            u = f" {s.unit}" if getattr(s, "unit", None) else ""
            rows.append(f"{s.name} ∈ [{lo:.3g}, {hi:.3g}]{u}"
                        if (lo is not None and hi is not None)
                        else f"{s.name} " + (f"≥ {lo:.3g}{u}" if lo is not None
                                             else f"≤ {hi:.3g}{u}"))
        allowed = "；".join(rows) if rows else "见参数说明"
        si = _si_params(meta)
        fmt = ""
        if si:
            strict_names = [n for n, s in si.items() if s]
            fmt = (
                "注意这些有量纲的参数要**写成字符串**："
                + "、".join(f"`{n}`" for n in list(si)[:6])
                + "。"
            )
            if strict_names:
                fmt += (
                    "其中 "
                    + "、".join(f"`{n}`" for n in strict_names[:6])
                    + " **必须带 SI 前缀**（如 '3p'、'180n'、'150p'），裸数字会被拒绝。"
                )
        return (
            f"[{meta.name}] precondition_failed: 参数超出允许范围。"
            f"允许范围：{allowed}。{fmt}"
            "这是**数值/单位**问题，不是硬件故障 —— "
            "（100 pA = '100p'，1 nA = '1n'，100 nm = '100n' m）。"
            "**不要重复发送同一个值**；若不确定正确量级，说明情况并结束本回合。"
        )

    tool = _tool_decorator(
        meta.name,
        description=meta.description,
        args_schema=args_schema,
    )(_run)
    # Bounds rejections must read like SafetyGate's, not like pydantic's.
    try:
        tool.handle_validation_error = _explain_validation
    except Exception:  # noqa: BLE001 — older langchain: leave default behaviour
        logger.debug("handle_validation_error not settable on %s", meta.name)
    # Stamp metadata for SafetyGateMiddleware to read (Phase 4)
    tool.metadata = {
        "danger_level": danger_level,
        "skill_source": skill_cls.__module__,
        "skill_version": getattr(meta, "version", "1.0"),
        # Full SkillMetadata reference for middleware introspection (global
        # parameter bounds, state-precondition cross-checks). Not JSON-serial.
        "skill_metadata": meta,
    }
    return tool


def _wire_progress_bridge(ctx: Any, skill_name: str, prior_state: dict) -> None:
    """Wire the ExecutionContext so a CompositeSkillGraph can resume.

    Reads ``prior_state["composite_progress"][skill_name]`` (if present)
    and exposes ``ctx.get_progress(name)`` returning that dict. The
    GraphExecutor in graph_executor.py uses this to skip completed steps
    when the same composite re-runs after a crash / pause / HITL gate.

    Also wires no-op ``emit_progress`` / ``checkpoint_flush`` if the
    context doesn't already define them — those are observed by tests
    via monkey-patch, not by the LangGraph runtime (the runtime sees the
    snapshot lifted from the SkillResult instead).
    """
    if ctx is None:
        return
    prog_map = prior_state.get("composite_progress", {}) if isinstance(prior_state, dict) else {}
    if not isinstance(prog_map, dict):
        prog_map = {}

    def _get_progress(name: str) -> dict | None:
        snap = prog_map.get(name)
        return snap if isinstance(snap, dict) and snap else None

    if not hasattr(ctx, "get_progress"):
        try:
            ctx.get_progress = _get_progress  # type: ignore[attr-defined]
        except Exception:
            pass
    if not hasattr(ctx, "emit_progress"):
        try:
            ctx.emit_progress = lambda progress: None  # type: ignore[attr-defined]
        except Exception:
            pass
    if not hasattr(ctx, "checkpoint_flush"):
        try:
            ctx.checkpoint_flush = lambda: None  # type: ignore[attr-defined]
        except Exception:
            pass


def _failure_update(
    name: str, summary: str, tool_call_id: str, *, also_log: bool = False
) -> dict[str, Any]:
    """Build a `.update`-style dict for failure paths (used by _SkillToolReturn)."""
    out: dict[str, Any] = {
        "executed_skills": [name],
        "messages": [
            ToolMessage(
                content=summary,
                tool_call_id=tool_call_id,
                name=name,
                status="error",
            )
        ],
    }
    if also_log:
        out["error_log"] = [f"{name}: {summary}"]
    return out


__all__ = ["wrap_skill"]
