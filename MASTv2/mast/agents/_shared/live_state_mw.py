"""LiveStateMiddleware — prepend live instrument state to every LLM call.

The instrument-control agent's static SYSTEM_PROMPT documents the role; this
middleware appends a fresh runtime block immediately before each model call
so the LLM sees the *current* scan frame, bias, setpoint, Z position, etc.

The block exists for one reason: prevent magnitude-of-unit mistakes. If the
scan frame is 100 nm and the LLM asks for a 1-metre-wide configure_scan,
that's a 10⁷× error. Showing the actual frame size in both metres and
nanometres in the system message keeps the LLM grounded.

This middleware needs a callable ``get_state() -> HardwareState`` injected by
the agent builder. If ``get_state`` is None or returns None the middleware is
a no-op (production fallback for offline tests).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from mast.agents._shared.inject import (
    append_human_block,
    append_system_block,
    last_human_index,
    with_appended_text,
)

#: 登记表条目 id。
PROMPT_ID = "mw.live_state"

#: 哪些 agent 挂它（真源；登记表与建图派生）。只有 IC 拿得到 get_state。
AGENTS = ("instrument_control",)

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest, ModelResponse
    from mast.core.types import HardwareState

logger = logging.getLogger(__name__)


def format_live_state_block(state: "HardwareState") -> str:
    """Render a HardwareState as a markdown block for the system message.

    Pure function — easy to unit-test independently from the middleware
    plumbing. Returns an empty string if every relevant field is None.
    """
    if state is None:
        return ""

    # 数值一律用 SI 前缀渲染，**不是** `%.4e`（2026-08-04 改，全块统一）。这个块的
    # 存在意义就是「让模型有个正确的数可以照抄」——不给数，它就得自己换算，而那
    # 正是 2026-07-27 坐标事故的成因。有量纲参数现在走字符串通道，其中整个量程远
    # 小于 1 的那些（坐标、设定点、扫描尺寸…）**强制要求 SI 前缀**：原来印的
    # `5.0000e-08` 抄进 x_m / setpoint_a 会被 parse_si 当场拒掉。
    #
    # 偏压是例外：它天然在 1 附近，普通小数就是它的自然写法（`-2`），
    # 而 format_si 会把它印成 `-2000m` —— 正确但没法看。
    from mast.core.si_quantity import format_si

    lines: list[str] = []
    if state.bias_v is not None:
        lines.append(f"- Bias voltage: {state.bias_v:g} V")
    if state.current_a is not None:
        lines.append(f"- Tunneling current: '{format_si(state.current_a)}' A")
    if state.setpoint_a is not None:
        lines.append(f"- Setpoint: '{format_si(state.setpoint_a)}' A")
    if state.z_pos_m is not None:
        lines.append(f"- Z position: '{format_si(state.z_pos_m)}' m")
    if state.x_pos_m is not None and state.y_pos_m is not None:
        lines.append(
            f"- XY position: ('{format_si(state.x_pos_m)}', "
            f"'{format_si(state.y_pos_m)}') m"
        )
    if state.z_controller_status is not None:
        lines.append(f"- Z controller: {state.z_controller_status}")
    elif state.z_controller_on is not None:
        lines.append(
            f"- Z controller: {'ON' if state.z_controller_on else 'OFF'}"
        )

    # ACTIVE Z-controller identity. A rig can
    # define several Z controllers with one active; engage/approach act on the
    # ACTIVE one, so name it here rather than making the agent hunt for it via
    # GetZCtrlList after being blocked on "Z feedback still closed".
    if state.z_controller_name is not None:
        line = f"- Active Z controller: {state.z_controller_name!r}"
        if state.z_controller_index is not None and state.z_controller_names:
            n = len(state.z_controller_names)
            avail = ", ".join(state.z_controller_names)
            line += (
                f" (index {state.z_controller_index} of {n}; available: {avail})"
            )
        line += (
            " — this is the LIVE feedback channel; engage/approach act on THIS "
            "controller. If feedback won't close, check that the intended "
            "controller is the active one (GetZCtrlList to inspect, "
            "SetActiveZController to switch) before assuming a hardware fault."
        )
        lines.append(line)
    if state.scan_running is not None:
        lines.append(
            f"- Scan: {'RUNNING' if state.scan_running else 'STOPPED'}"
        )

    # Scan-frame geometry + magnitude reminder — the headline feature.
    if state.scan_width_m is not None and state.scan_height_m is not None:
        w_nm = state.scan_width_m * 1e9
        h_nm = state.scan_height_m * 1e9
        lines.append(
            f"- Scan frame size: '{format_si(state.scan_width_m)}' x "
            f"'{format_si(state.scan_height_m)}' m  (= {w_nm:.1f} x {h_nm:.1f} nm)"
        )
    if state.scan_center_x_m is not None and state.scan_center_y_m is not None:
        cx_nm = state.scan_center_x_m * 1e9
        cy_nm = state.scan_center_y_m * 1e9
        lines.append(
            f"- Scan frame center: ('{format_si(state.scan_center_x_m)}', "
            f"'{format_si(state.scan_center_y_m)}') m  "
            f"(= {cx_nm:.1f}, {cy_nm:.1f} nm)"
        )
    if state.scan_angle_deg is not None:
        lines.append(f"- Scan rotation: {state.scan_angle_deg:.2f}°")

    if state.scan_width_m is not None:
        scale_nm = state.scan_width_m * 1e9
        lines.append(
            "- ⚠️ MAGNITUDE CHECK: lengths are METRE quantities, written as "
            "STRINGS WITH AN SI PREFIX. The current scan frame is "
            f"**{scale_nm:.1f} nm wide** (= '{format_si(state.scan_width_m)}'). "
            "Any skill parameter with unit 'm' must be on this order — "
            "typically '1n' … '100n'. Never pass `1` for nm (`1` = 1 metre = "
            "10⁹ nm), and do NOT switch to exponent form — on these parameters "
            "it is rejected outright."
        )

    if not lines:
        return ""

    return "## Live instrument state (refreshed every LLM call)\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# 这两个 helper 2026-08-24 搬进了 ``agents/_shared/inject``，和「拼接 + 记归属」
# 的其余部分放在一起；这里保留同名的 re-export，因为
# ``agents/_shared/alert_delivery_mw`` 从这个模块 import 它们（第二个消费者，
# 2026-08-10），而那段「落点选在最后一条 human 消息、理由是 prompt cache 断点」
# 的推理必须只有一个副本。
#
# 名字上的下划线因此只表示「不是本包对外的 API」，**不表示只有本模块用**。
# ─────────────────────────────────────────────────────────────────────────

_last_human_index = last_human_index
_with_appended_text = with_appended_text


class LiveStateMiddleware(AgentMiddleware):
    """Append a live-state block to the SystemMessage on every model call.

    Implements BOTH sync ``wrap_model_call`` (GUI ``graph.stream()``) and async
    ``awrap_model_call`` (CLI ``await graph.ainvoke()``). LangChain 1.2's base
    async hook raises NotImplementedError when only the sync hook is defined, so
    a sync-only middleware crashes every async agent dispatch — the async twin is
    mandatory for the ``python -m mast`` path.
    """

    def __init__(self, get_state: Callable[[], "HardwareState | None"] | None):
        super().__init__()
        self._get_state = get_state

    def _apply(self, request: "ModelRequest") -> "ModelRequest":
        """Inject the live-state block for THIS call only.

        WHERE it goes matters as much as what it says (2026-07-28 audit, 次级
        「prompt cache 断点落在实时读数之后」). Every field here is formatted
        ``:.4e`` and re-read each call, so the block changes on essentially every
        turn. It used to be appended to the END of the system message — and
        ``AnthropicPromptCachingMiddleware`` puts its cache breakpoint at exactly
        that end (``prompt_caching._apply_caching``: model_settings, end of
        system, last tool). LiveState runs OUTERMOST (``factory.py:224``), so the
        breakpoint landed *after* the volatile text: the 224-tool block hit cache
        while the system message and the entire history missed on every single
        turn — and the 1 h TTL means paying the write premium for a prefix that
        is never reused.

        Fix: keep the system message byte-identical across turns and carry the
        volatile block on the LAST HUMAN message instead. That needs no
        per-provider branch (it is an ordinary message edit, not a role trick),
        it puts the readings closer to the question than they were before, and
        the request's message list is rebuilt rather than mutated so nothing
        reaches the checkpoint.

        Fallback: if there is no human message to carry it (a tool-only tail on
        some agent's first hop), append to the system message as before —
        correctness of the numbers outranks a cache hit.
        """
        if self._get_state is None:
            return request
        try:
            state = self._get_state()
        except Exception as exc:
            logger.info("LiveStateMiddleware get_state() failed: %s", exc)
            return request
        block = format_live_state_block(state) if state is not None else ""
        if not block:
            return request

        out = append_human_block(request, PROMPT_ID, block)
        if out is not None:
            return out
        # 没有 human 消息可挂（某个 agent 第一跳只有工具尾巴）→ 退回 system。
        # 数字的正确性压过一次 cache 命中。
        return append_system_block(request, PROMPT_ID, block)

    def wrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], Any],
    ) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], Any],
    ) -> Any:
        return await handler(self._apply(request))


__all__ = ["LiveStateMiddleware", "format_live_state_block"]
