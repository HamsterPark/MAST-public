"""Signals module skills: read signal metadata and values.

vendored from v1 mast/skills/builtins/signals.py 2026-04-23"""

from __future__ import annotations

import logging
from typing import Any

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)


def _scalar_int(x) -> int | None:
    """One Nanonis scalar → ``int``, unwrapping a 1-element sequence. Never raises.

    Same coercion as :func:`mast.io.nanonis_files.scalar_int`; duplicated rather
    than imported so this module stays free of the numpy/`.sxm` import chain.
    """
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            return None
        x = x[0]
    if isinstance(x, (str, bytes)):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


class GetSignalsAddRT(BaseSkill):
    """Read additional RT signals assigned to Internal 23/24."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSignalsAddRT",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读可用的附加 RT 信号列表，以及当前指派给 Internal 23 与 Internal 24 的名字。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["signals", "rt", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Signals_AddRTGet")
        if record.error:
            return SkillResult(
                skill_name="GetSignalsAddRT", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)):
                # Signals.AddRTGet ResponseTypes = ["i","i","*+c","i","*-c","i","*-c"]
                # so parsed[2] is positional:
                #   [0] additional RT signals names size (int, bytes)
                #   [1] number of additional RT signals (int)
                #   [2] available RT signal names (1D string array)
                #   [3] RT signal 1 name size (int)
                #   [4] Internal 23 assigned signal name (string)
                #   [5] RT signal 2 name size (int)
                #   [6] Internal 24 assigned signal name (string)
                if len(vals) >= 7:
                    names = vals[2]
                    if isinstance(names, (list, tuple)):
                        data["available_rt_signals"] = [str(n) for n in names]
                    else:
                        data["available_rt_signals"] = [str(names)]
                    data["num_rt_signals"] = int(vals[1])
                    data["internal_23_signal"] = str(vals[4])
                    data["internal_24_signal"] = str(vals[6])
                elif len(vals) >= 3 and isinstance(vals[2], (list, tuple)):
                    # Defensive: at least expose the available names list
                    data["available_rt_signals"] = [str(n) for n in vals[2]]
                elif len(vals) >= 1:
                    # Minimal parse: just expose what we got
                    data["available_rt_signals"] = [str(v) for v in vals]
        return SkillResult(
            skill_name="GetSignalsAddRT", success=True,
            data=data, nanonis_calls=[record],
        )


class GetSignalRange(BaseSkill):
    """Read the range limits of a signal by index."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSignalRange",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读一路信号（0-127）的量程上限与下限。",
            parameters=[
                ParameterSpec(
                    name="signal_index",
                    type="int",
                    description="信号序号（0-127）",
                    required=True,
                    min_value=0,
                    max_value=127,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["signals", "range", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = params["signal_index"]
        record = context.safe_call("Signals_RangeGet", idx)
        if record.error:
            return SkillResult(
                skill_name="GetSignalRange", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"signal_index": idx, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                data = {
                    "signal_index": idx,
                    "max_limit": float(vals[0]),
                    "min_limit": float(vals[1]),
                }
        return SkillResult(
            skill_name="GetSignalRange", success=True,
            data=data, nanonis_calls=[record],
        )


class GetSignalValues(BaseSkill):
    """Read the current values of multiple signals at once."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSignalValues",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读若干选定信号的当前值（过采样）。给一份信号序号的列表（0-127）。"
            ),
            parameters=[
                ParameterSpec(
                    name="signal_indexes",
                    type="str",
                    description=(
                        "逗号分隔的信号序号，例如 '0,1,14'。每个序号的范围是 0-127。"
                    ),
                    required=True,
                ),
                ParameterSpec(
                    name="wait_for_newest",
                    type="bool",
                    description=(
                        "为 True 时，丢掉第一个采样，返回一个完全新鲜的值（会更慢）。默认 False。"
                    ),
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["signals", "values", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx_str = str(params["signal_indexes"])
        # 非法通道索引必须报告参数名与接受的格式，不能只向调用方抛裸 int 转换异常。
        indexes: list[int] = []
        bad: list[str] = []
        for piece in idx_str.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                indexes.append(int(piece))
            except ValueError:
                bad.append(piece)
        if bad or not indexes:
            return SkillResult(
                skill_name="GetSignalValues", success=False,
                error=(f"signal_indexes 解不出通道号:{idx_str!r}。"
                       + (f"这几段不是整数:{bad}。" if bad else "一个都没给出。")
                       + "格式是逗号分隔的整数,例如 \"0\" 或 \"0,24,30\"。"
                         "(通道清单见 ListSignalChannels。)"),
                nanonis_calls=[],
            )
        wait = int(params.get("wait_for_newest", False))
        # Signals_ValsGet(Signals_indexes, Wait_for_newest_data)
        record = context.safe_call("Signals_ValsGet", indexes, wait)
        if record.error:
            return SkillResult(
                skill_name="GetSignalValues", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        # ⚠️ 从前这里是 ``"raw": decode_reply(parsed)`` —— 把整个三段信封(含 bytes)
        # 字符串化塞进 data。它是 2026-08-14 那次 HTTP 500 的来源之一,
        # 而且对调用方毫无用处:一串 ``b'…'`` 的字面量。
        # 要看原始回包去看 nanonis_calls,那里本来就有。
        data: dict = {"signal_indexes": indexes}

        def _to_float(v: Any) -> float:
            # decodeArray yields single-element tuples like (1.0,); unwrap them.
            if isinstance(v, (list, tuple)) and len(v) == 1:
                v = v[0]
            return float(v)

        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)):
                # Signals.ValsGet ResponseTypes = ["i", "*f"] so parsed[2] is
                # positional: [0] signals values size (int), [1] values array
                # (1D float32). The actual readings live in vals[1], NOT vals[0]
                # (vals[0] is just the array length).
                if len(vals) >= 2 and isinstance(vals[1], (list, tuple)):
                    data["values"] = [_to_float(v) for v in vals[1]]
                elif len(vals) == 1 and isinstance(vals[0], (list, tuple)):
                    # Defensive: parser sometimes collapses to a bare array
                    data["values"] = [_to_float(v) for v in vals[0]]
                else:
                    # Last resort: treat the remaining entries as scalars
                    data["values"] = [_to_float(v) for v in vals]
            else:
                data["values"] = [_to_float(vals)]
        return SkillResult(
            skill_name="GetSignalValues", success=True,
            data=data, nanonis_calls=[record],
        )


def _is_current_name(name: str) -> bool:
    """Heuristic: does this signal name denote a current channel?

    Nanonis exposes the tunnelling current under several display names
    depending on configuration: 'Current (A)', 'Current', 'LI Demod ... (A)'
    for lock-in current, 'Current 2 (A)', etc. We flag the plain current
    channels (name starts with 'current') so the GUI can pre-select the most
    likely tunnelling-current index without forcing the user to scan all 128.
    """
    n = (name or "").strip().lower()
    return n.startswith("current")


class ListSignalChannels(BaseSkill):
    """Enumerate the 128 available Nanonis signal channels by name + index."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ListSignalChannels",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "列出 128 路可用的 Nanonis 信号（物理输入／输出／内部通道）及其 0-127 序号。"
                "会标出哪些序号是电流通道，好让调用方为高速采集挑出隧道电流那一路。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["signals", "enumerate", "channels", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Signals_NamesGet")
        if record.error:
            return SkillResult(
                skill_name="ListSignalChannels", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        # Signals.NamesGet ResponseTypes = ["i", "i", "*+c"] →
        # Variables (parsed[2]) is positional:
        #   [0] signals names size (int, bytes)
        #   [1] number of signals (int)   ← the INSTRUMENT's own count
        #   [2] signal names (1D string array, prepended-size each)
        names: list[str] = []
        declared: int | None = None
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 3 \
                    and isinstance(vals[2], (list, tuple)):
                names = [str(n) for n in vals[2]]
                declared = _scalar_int(vals[1])
            elif isinstance(vals, (list, tuple)):
                # Defensive: some parser builds collapse to a bare name list.
                names = [str(n) for n in vals if isinstance(n, str)]
        if not names:
            return SkillResult(
                skill_name="ListSignalChannels", success=False,
                error=f"Could not parse signal names from response: {parsed!r}",
                nanonis_calls=[record],
            )

        # ── the count the instrument declared vs the count we decoded ────────
        #
        # `nanonis_patch._patched_decodeStringPrepended` ends the loop with a
        # bare `break` when the body runs out, and clamps the last string with
        # `min(index + str_len, end_of_body)`. Both truncate silently. On the
        # rig the name table came back with 51 entries once — and 51 names is
        # not distinguishable, downstream, from a machine that only has 51
        # signals.
        #
        # That difference is not cosmetic. `monitoring.aux_channels` resolves
        # its channels by matching NAMES against this table, and the operator's
        # lock-in sits at index 86. A list cut at 51 makes the panel say
        # 「这台机器没有 lock-in」 — a statement about the hardware, produced by
        # a parse failure. Reporting the disagreement is what keeps「没解出来」
        # and「本机没有」 two different sentences.
        #
        # It is reported, NOT raised: a partial table still resolves every
        # channel below the cut, and failing outright would take down discovery
        # that used to half-work. The caller decides what a short list means.
        truncated = declared is not None and declared > len(names)
        if truncated:
            logger.warning(
                "Signals_NamesGet: 仪器声明 %d 路信号，只解出 %d 路 —— "
                "名单被截断，索引 %d 及以后的通道在本次结果里不存在",
                declared, len(names), len(names),
            )

        channels = [{"index": i, "name": n} for i, n in enumerate(names)]
        current_indices = [c["index"] for c in channels
                           if _is_current_name(c["name"])]
        return SkillResult(
            skill_name="ListSignalChannels", success=True,
            data={
                "channels": channels,
                "n_channels": len(channels),
                # 仪器自己说有多少路。与 n_channels 不同就是解析出了问题 ——
                # 报出来，而不是用 len() 覆盖掉它（io/nanonis_files.py 对
                # `num_channels` 是同一条纪律）。
                "declared_n": declared,
                "truncated": truncated,
                "current_indices": current_indices,
            },
            nanonis_calls=[record],
        )
