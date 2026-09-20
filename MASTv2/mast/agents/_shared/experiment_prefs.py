"""User experiment-parameter preferences — default scan / spectroscopy values.

要求：一次实验的喜好——扫多大,扫多快,应该有一个地方可以用户设定。 The operator sets their usual scan size / speed / resolution
/ setpoint / bias once in 设置 → 实验默认参数, and every planning + execution
agent then sees those numbers as **defaults to prefer** — not as hard limits
(SafetyLimits still bound the hardware).

Design mirrors :mod:`mast.vision.thresholds` and the ``orchestrator_recursion_limit``
live-read pattern:

  * A process-level holder (:func:`get_prefs` / :func:`set_prefs`) keeps the
    ACTIVE snapshot. The API/runtime layer WRITES it (startup hydration in
    ``core.runtime.setup`` + each ``POST /api/settings``); the agent middleware
    READS it lock-free before every model call. So a change takes effect on the
    next agent turn with **no graph rebuild**.
  * The agents never import settings — the wiring is strictly one-way
    (settings → holder → middleware). An empty/blank holder makes the middleware
    a no-op, so the shipped default (nothing set) never touches a prompt.

The block is injected as free text; it is a *hint*, never a bound. This module
has no heavy imports and never raises on bad input (it sanitises + drops).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from mast.agents._shared.inject import append_system_block

#: 登记表条目 id。
PROMPT_ID = "mw.experiment_prefs"

#: 哪些 agent 挂它（真源；登记表与共享栈派生）。
AGENTS = ("instrument_control", "experiment_design")

logger = logging.getLogger(__name__)


# ── Field spec — the keys the 设置 UI surfaces + how to render each ───────────
# key -> (中文标签, 单位, 强制类型). Anything not listed here is DROPPED on set,
# so a stray/garbage key from a bad POST can never reach a prompt. Ranges keep a
# fat-fingered value from rendering something absurd (a preference, not a bound).
_FIELD_SPEC: dict[str, tuple[str, str, type, tuple[float, float]]] = {
    "scan_size_nm":    ("扫描尺寸(边长)", "nm", float, (0.0, 1.0e6)),
    "scan_speed_nm_s": ("扫描速度", "nm/s", float, (0.0, 1.0e6)),
    "line_time_s":     ("每线时间", "s", float, (0.0, 1.0e4)),
    "scan_lines":      ("扫描线数(分辨率)", "px", int, (1, 8192)),
    "scan_angle_deg":  ("扫描旋转角", "°", float, (-360.0, 360.0)),
    "setpoint_pa":     ("电流设定点 setpoint", "pA", float, (0.0, 1.0e8)),
    "bias_v":          ("偏压 bias", "V", float, (-10.0, 10.0)),
}

# Enumerated (categorical) preferences — the stored value MUST be one of the
# choices; anything else is dropped. key -> (中文标签, choices, {value: 显示文本}).
_CHOICE_SPEC: dict[str, tuple[str, tuple[str, ...], dict[str, str]]] = {
    "scan_direction": (
        "扫描方向", ("up", "down"),
        {"up": "从下往上 (up)", "down": "从上往下 (down)"},
    ),
}

# Free-text preference (rendered verbatim, length-capped). Kept separate so it
# is coerced as a string rather than a number.
_NOTES_KEY = "notes"
_NOTES_MAX = 500

# Public list of the editable keys (for the frontend / tests).
EDITABLE_KEYS: tuple[str, ...] = (
    tuple(_FIELD_SPEC) + tuple(_CHOICE_SPEC) + (_NOTES_KEY,)
)


def sanitize(raw: Any) -> dict[str, Any]:
    """Coerce a raw settings dict into the clean, bounded prefs snapshot.

    Keeps only known keys; coerces numbers (dropping NaN / non-numeric),
    clamps to the field range, and truncates the free-text note. Returns ``{}``
    for anything that isn't a dict — never raises. An empty result means the
    holder is effectively unset (middleware no-op).
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, (_label, _unit, kind, (lo, hi)) in _FIELD_SPEC.items():
        if key not in raw or raw[key] is None or raw[key] == "":
            continue
        try:
            val = kind(raw[key])
        except (TypeError, ValueError):
            continue
        # drop NaN / inf that survive float()
        if isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
            continue
        val = max(lo, min(hi, val))
        out[key] = int(val) if kind is int else float(val)
    for key, (_label, choices, _disp) in _CHOICE_SPEC.items():
        v = raw.get(key)
        if isinstance(v, str) and v.strip() in choices:
            out[key] = v.strip()
    note = raw.get(_NOTES_KEY)
    if isinstance(note, str) and note.strip():
        out[_NOTES_KEY] = note.strip()[:_NOTES_MAX]
    return out


# ── Process-level holder (live-read) ─────────────────────────────────────────
_lock = threading.RLock()
_prefs: dict[str, Any] = {}


def set_prefs(raw: Any) -> dict[str, Any]:
    """Replace the active preference snapshot (sanitised). Returns the stored
    copy. ``None`` / non-dict / empty clears the holder (middleware no-op)."""
    clean = sanitize(raw)
    with _lock:
        _prefs.clear()
        _prefs.update(clean)
        return dict(_prefs)


def get_prefs() -> dict[str, Any]:
    """Return a copy of the active preference snapshot ({} when unset)."""
    with _lock:
        return dict(_prefs)


# ── Pure render (unit-testable, no middleware plumbing) ──────────────────────
#: pref key -> (SI parameter name the skills actually take, multiplier from the
#: stored human unit to SI). Keys absent here have no SI counterpart (a count, an
#: angle, a duration already in seconds) and render unchanged.
_SI_EQUIV: dict[str, tuple[str, float]] = {
    "scan_size_nm":    ("scan_width_m / scan_height_m", 1e-9),
    "scan_speed_nm_s": ("scan_speed_m_s", 1e-9),
    "setpoint_pa":     ("setpoint_a", 1e-12),
    "bias_v":          ("bias_v", 1.0),          # already SI; no conversion shown
}


def _si_hint(key: str, value: Any) -> str:
    """The SI value the skills actually take, appended to a human-unit pref.

    Without this the block printed ONLY human units — "扫描尺寸(边长): 50 nm",
    "电流设定点 setpoint: 100 pA" — under a heading that tells the model to
    **prefer these values**, while every skill takes SI (scan_width_m=5e-8,
    setpoint_a=1e-10). The correct number appeared nowhere in the text, so the
    model had nothing to copy and had to convert unaided. That is the same class
    of defect as the 2026-07-27 coordinate incident, and worse than the block
    involved there: live_state at least prints the SI value first and carries a
    MAGNITUDE CHECK guard right after it; this block had neither.

    Shape follows safety_mw's correction text, which already does it right:
    every human unit is immediately followed by its SI equivalent.
    """
    spec = _SI_EQUIV.get(key)
    if spec is None:
        return ""
    param, mult = spec
    if mult == 1.0:                      # value is already SI — nothing to add
        return ""
    try:
        si = float(value) * mult
    except (TypeError, ValueError):
        return ""
    # 用 SI 前缀渲染，**不是** `%.6g`（2026-08-04 改）。原来印的是 `5e-08`，而这个
    # 块的抬头明说「调用技能时用括号里的 SI 值」—— 于是它在直接指示模型写
    # `size_m="5e-08"`，而 size_m 这类整个量程远小于 1 的参数**强制要求 SI 前缀**，
    # 那个值会被 parse_si 当场拒掉。
    #
    # 「照抄块里的数」正是这个函数存在的理由（见上面的 docstring：不给数,模型就得
    # 自己换算,而那正是 2026-07-27 坐标事故的成因）。所以块里印的必须是**能直接
    # 抄进技能参数、且真的能通过解析**的那个形式。
    from mast.core.si_quantity import format_si

    return f"  (= '{format_si(si)}' → {param})"


def format_prefs_block(prefs: dict[str, Any] | None) -> str:
    """Render the operator's default-parameter preferences as a system block.

    Pure function. Returns ``""`` when there is nothing set, so the middleware
    injects nothing on the shipped (unset) default.
    """
    if not prefs:
        return ""
    lines: list[str] = []
    for key, (label, unit, _kind, _rng) in _FIELD_SPEC.items():
        if key in prefs:
            unit_s = f" {unit}" if unit else ""
            si = _si_hint(key, prefs[key])
            lines.append(f"- {label}: {prefs[key]}{unit_s}{si}")
    for key, (label, _choices, disp) in _CHOICE_SPEC.items():
        if key in prefs:
            val = prefs[key]
            lines.append(f"- {label}: {disp.get(val, val)}")
    note = prefs.get(_NOTES_KEY)
    if note:
        lines.append(f"- 其他偏好: {note}")
    if not lines:
        return ""
    return (
        "## 用户实验默认参数偏好\n"
        "用户在「设置 → 实验默认参数」里设定了以下常用默认值。\n"
        "\n"
        "⚠️ **扫描类默认值（速度 / 线数 / setpoint / 角度）已经由 ScanAt 自动应用** ——\n"
        "策略层会按这些偏好和按尺度参数表把参数补齐。**不要手工把这里的数字转传给 "
        "ScanAt**：那会被记成「用户显式指定」并展示给用户，而他并没有在这次对话里"
        "说过它们。只有在直接使用 ConfigureScan 等底层技能时才参考下面的数字。\n"
        "\n"
        "「扫描尺寸」是例外——它是**意图**，不是执行参数：用户没说扫多大时，用它作为 "
        "ScanAt 的 `size_m`。\n"
        "\n"
        "它们是偏好，不是硬上限——安全边界仍由 SafetyLimits 约束，越界的请求照样会被拒。\n"
        "**调用技能时用括号里的 SI 值，不要用前面的人类单位数字**"
        "（技能参数一律 SI：米、安培、伏特）。\n"
        + "\n".join(lines)
    )


# ── Middleware — append the prefs block before every model call ──────────────
class ExperimentPrefsMiddleware(AgentMiddleware):
    """Append the operator's experiment-default-parameter block to the system
    message on every model call (live-read from the process holder).

    A no-op when the holder is empty (the shipped default), so agents that never
    have prefs set are byte-for-byte unchanged. Implements BOTH the sync and
    async ``wrap_model_call`` hooks — LangChain's async base raises
    NotImplementedError for a sync-only middleware, which would crash the
    ``python -m mast`` async dispatch path (see LiveStateMiddleware)."""

    def __init__(self, get_prefs_fn: Callable[[], dict[str, Any]] | None = None):
        super().__init__()
        # Default to the module holder; injectable for tests.
        self._get_prefs = get_prefs_fn or get_prefs

    def _apply(self, request):
        try:
            prefs = self._get_prefs()
        except Exception as exc:  # never break a run over a prefs read
            logger.debug("ExperimentPrefsMiddleware get_prefs failed: %s", exc)
            return request
        block = format_prefs_block(prefs)
        return append_system_block(request, PROMPT_ID, block)

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))


__all__ = [
    "EDITABLE_KEYS",
    "sanitize",
    "set_prefs",
    "get_prefs",
    "format_prefs_block",
    "ExperimentPrefsMiddleware",
]
