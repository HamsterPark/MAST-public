"""Scan / imaging control skills.

P (Ported) from v1 mast/skills/builtins/imaging.py.
v0.3.12 adds _build_scan_basename helper so .sxm files come out as
``<exp>_<sample>_NNNN.sxm`` instead of ``unnamed####.sxm``.
"""

from __future__ import annotations

import logging
import math
import re

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

_log = logging.getLogger(__name__)


def _safe_basename(text: str, max_len: int = 24) -> str:
    """Sanitise an experiment / sample name into something Nanonis can use
    as a sxm filename prefix. Keeps ASCII letters/digits/underscore/dash;
    replaces everything else with underscore. Truncates to max_len."""
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:max_len] or ""


# ── Nanonis Scan.Props flag encodings ────────────────────────────────────────
# Set and Get do NOT share an encoding. Checked one at a time against the
# nanonis_spm docstrings (2026-08-09) rather than assuming, because a readback
# compared under the wrong table is a check that reports "accepted" either way.
#
#   Scan.PropsSet   continuous/bouncy : 0 = no change, 1 = On,   2 = Off
#                   autosave          : 0 = no change, 1 = All,  2 = Next, 3 = Off
#   Scan.PropsGet   continuous/bouncy : 0 = Off,       1 = On
#                   autosave          : 0 = All,       1 = Next, 2 = Off
#
# The zero is the whole story: in the SET table 0 reads like "off/false" and
# actually means "leave whatever the operator last set in the GUI".
_SET_NO_CHANGE = 0
_SET_OFF = 2
_SET_AUTOSAVE_ALL = 1
_GET_ON = 1
_GET_OFF = 0

# Scan.PropsGet contains flags, length-prefixed text, module metadata and
# per-module parameter arrays. Positional reads must validate the associated
# counts and shapes before interpreting the value. Invalid layouts produce
# unknown rather than a confident controller state; see _scan_props_module_count.
_PROPS_N_FIELDS = 16
_PROPS_IX_SERIES_NAME = 4
_PROPS_IX_MODULES_COUNT = 8
_PROPS_IX_MODULES = 9


def _props_body(parsed):
    """The Variables list out of a ``(err, raw, Variables)`` reply, or None."""
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return None
    body = parsed[2]
    return body if isinstance(body, (list, tuple)) else None


def _unwrap_scalar(value):
    """Array fields arrive as 1-tuples on the real rig; scalars do not."""
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) == 1 else None
    return value


def _continuous_state(flag: "int | None") -> "bool | None":
    """GET-encoded continuous flag → **three states**: True / False / None.

    ``None`` covers BOTH "we could not read it" and "the rig answered something
    that is not in the GET table" — because neither is an answer to "is it on?".

    The second half is not hypothetical padding. The GET table has exactly two
    values (0=Off, 1=On) while the SET table's Off is **2**, and a 2 arriving
    here used to come out as ``2 == _GET_ON`` → ``False`` → "confirmed off".
    That is the same shape as the 2026-08-19 failure this whole path exists for:
    a value that is not an answer, folded into the reassuring answer. A test
    double in this repo was already feeding a 2 (see
    ``tests/v2/unit/skills/builtins/test_scan_direction_up_down.py``), so the
    branch was reachable from inside the repo, not just from a strange rig.
    """
    if flag == _GET_ON:
        return True
    if flag == _GET_OFF:
        return False
    return None


def _scan_props_modules(parsed) -> "list[str]":
    """解析 Scan_PropsGet 返回的保存模块清单，无法读取时返回空表。
    回包是 (error, raw, body)，模块名列表位于 body 内，不能在信封顶层搜索。
    识别完整的字符串列表后保留用户清单，避免解析失败被固定默认清单掩盖。
    """
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return []
    body = parsed[2]
    if not isinstance(body, (list, tuple)):
        return []
    for item in body:
        if (isinstance(item, (list, tuple)) and item
                and all(isinstance(s, str) for s in item)):
            return [str(s) for s in item]
    return []


def _scan_props_module_count(parsed) -> "int | None":
    """The **declared** ``Modules names number``, or None if it can't be trusted.

    Why this exists: ``_scan_props_modules`` returns ``[]`` for two situations
    that are not the same situation —

    * the operator has **zero** modules selected (a fact), and
    * we could not find the array in the reply (an absence of facts).

    Folding those together is the exact move that has cost this repo a day at a
    time, and here it costs something specific: with zero modules selected,
    writing ``[]`` back is a **provable no-op under either possible protocol
    semantics** — if an empty array means "clear the list" it clears an already
    empty list, and if it means "no change" it changes nothing. So that is the
    one case where ``Scan_PropsSet`` can be sent WITHOUT having read a list, and
    conflating it with "unknown" threw that case away.

    Trusted only when the reply has the full declared shape AND the declared
    count agrees with the decoded array — two fields of the same reply that a
    mis-parse has no reason to keep consistent. Disagreement ⇒ None (unknown),
    never a repair.
    """
    body = _props_body(parsed)
    if body is None or len(body) < _PROPS_N_FIELDS:
        return None
    count = _unwrap_scalar(body[_PROPS_IX_MODULES_COUNT])
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    names = body[_PROPS_IX_MODULES]
    if not isinstance(names, (list, tuple)):
        return None
    if len(names) != count or not all(isinstance(s, str) for s in names):
        return None
    return count


def _scan_props_series_name(parsed) -> str:
    """The current series name out of a reply we ALREADY have. "" if unreadable.

    Positional when the reply has the full declared shape, first-string
    otherwise. Exists so ``StartScan`` stops issuing a SECOND ``Scan_PropsGet``
    just for this: the two reads can disagree, and the disagreement is not
    harmless — an empty series name written into ``Scan_PropsSet`` clobbers the
    operator's configured filename prefix down to ``unnamed####`` (review
    2026-07-03). One read, one answer.
    """
    body = _props_body(parsed)
    if body is None:
        return ""
    if len(body) >= _PROPS_N_FIELDS:
        name = body[_PROPS_IX_SERIES_NAME]
        return name if isinstance(name, str) else ""
    for item in body:
        if isinstance(item, str) and item:
            return item
    return ""


def _scan_props_continuous(parsed) -> "int | None":
    """The continuous-scan flag out of a ``Scan_PropsGet`` reply (GET encoding).

    It is the first int of the reply body. Nothing read it until 2026-08-09: the
    reply was already being fetched — for the module names and the series name —
    and the one flag that decides whether a scan ever ENDS was in hand and
    dropped on the floor.

    What that cost: the operator's GUI had
    continuous On, so ``Scan_Action(start)`` did not mean "acquire one frame". It
    meant "acquire frames until told to stop". Four frames came out (three of
    them complete, 78.7 s each, alternating direction because bouncy was On too),
    ``Scan_StatusGet`` never once returned 0, and a wait built on "poll until the
    scan stops" could not succeed even in principle. It burned its whole 300 s
    budget and reported the frame incomplete — true, but for none of the reasons
    the message gave.
    """
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return None
    variables = parsed[2]
    if not isinstance(variables, (list, tuple)):
        return None
    for item in variables:
        if isinstance(item, str):
            return None  # strings begin after the flags — no flag in this reply
        if isinstance(item, (list, tuple)):
            # Array fields arrive as 1-tuples on the real rig; scalars do not,
            # but costing nothing to tolerate both beats a shape assumption.
            item = item[0] if item else None
        if isinstance(item, bool):
            return int(item)
        if isinstance(item, int):
            return int(item)
    return None


# ⚠️ 这里曾经有一个 ``_current_scan_series_name(context, calls)`` —— 它自己再发
# 一次 ``Scan_PropsGet`` 只为取序列名。**删掉了**,因为它构成了「同一件事读两次」:
# 取模块清单那一发成功、取序列名这一发失败时,basename 变成空串,而空的 Series
# name 会把用户配的文件名前缀打回 ``unnamed####``。
# 现在序列名从模块清单**同一份回包**里取:见 :func:`_scan_props_series_name`。


#: How many times ``Scan_PropsGet`` is attempted when the reply comes back
#: unusable. Two, not one, and not more.
#:
#: The reason there is a retry at all: from now on an unreadable reply makes
#: ``StartScan`` REFUSE to start (see :class:`StartScan`), so a single TCP hiccup
#: would abort a scan — and inside an overnight campaign, a run. The reason it is
#: bounded at two: the failure that actually happened (2026-08-19, ``*+i``
#: mis-parse) was **deterministic**, so retrying it is pure cost. A retry buys
#: back the transient case and nothing else, and it should not pretend otherwise.
#:
#: Only transport/shape failures are retried. A reply that decodes but lacks the
#: field we want is a property of that firmware, not of that moment, and asking
#: again is the "probe is not free" mistake with none of the upside.
_PROPS_GET_ATTEMPTS = 2


def _read_scan_props(context, calls: list) -> "tuple[object, str]":
    """``Scan_PropsGet`` → ``(parsed_or_None, error_text)``. Read-only, retried.

    ``parsed`` is the raw ``(err, raw, Variables)`` triple when the reply arrived
    and has a body; ``None`` when it did not. ``error_text`` is non-empty exactly
    when ``parsed`` is None, and says which of the failure modes it was — a
    caller that has to explain itself to an operator needs that difference.
    """
    error = ""
    for attempt in range(1, _PROPS_GET_ATTEMPTS + 1):
        try:
            rec = context.safe_call("Scan_PropsGet")
            calls.append(rec)
            if getattr(rec, "error", ""):
                error = str(rec.error)
            else:
                parsed = getattr(rec, "return_value", None)
                if _props_body(parsed) is not None:
                    return parsed, ""
                error = (f"Scan_PropsGet 回包读不懂"
                         f"({type(parsed).__name__})")
        except Exception as exc:  # noqa: BLE001 — 读失败要变成一句话,不是崩溃
            error = f"{type(exc).__name__}: {exc}"
        if attempt < _PROPS_GET_ATTEMPTS:
            _log.warning("Scan_PropsGet 第 %d 次读失败(%s),再试一次。",
                         attempt, error)
    return None, error or "Scan_PropsGet 失败(原因未知)"


def _build_scan_basename() -> str:
    """Build a sxm basename from the active ExperimentLog (if any).

    Format: ``<exp>_<sample>_`` (trailing underscore so Nanonis appends
    its own zero-padded counter). Falls back to empty string if there's
    no active experiment — caller should be prepared to keep the existing
    Nanonis behaviour in that case.
    """
    try:
        from mast.logging.experiment_log import get_active_log
        log = get_active_log()
        if log is None:
            return ""
        exp_id = getattr(log, "current_experiment_id", None)
        sample_id = getattr(log, "current_sample_id", None)
        exp_name = ""
        sample_name = ""
        if exp_id:
            try:
                row = log._storage.get_experiment(exp_id)
                if row:
                    exp_name = _safe_basename(row.get("name", ""))
            except Exception:
                pass
        if sample_id:
            try:
                row = log._storage.get_sample(sample_id)
                if row:
                    sample_name = _safe_basename(row.get("name", ""))
            except Exception:
                pass
        parts = [p for p in (exp_name, sample_name) if p]
        if not parts:
            return ""
        return "_".join(parts) + "_"
    except Exception:
        return ""


#: 扫描帧读回的相对容差（Nanonis 内部是 float32，别拿等号比）与绝对下限（m）。
_FRAME_TOL_FRAC = 2e-4
_FRAME_TOL_ABS_M = 1e-12
#: 角度容差（度）。
_FRAME_TOL_DEG = 0.05


def frame_extent(center_x_m, center_y_m, width_m, height_m, angle_deg=0.0):
    """扫描框四角在压电坐标里的包络 → (min_x, max_x, min_y, max_y)。

    **必须算角度。** 转了 45° 的框，对角线伸出去 √2 倍；
    只比 center ± size/2 会在转角的帧上系统性地少算。
    """
    hw, hh = float(width_m) / 2.0, float(height_m) / 2.0
    a = math.radians(float(angle_deg or 0.0))
    ca, sa = math.cos(a), math.sin(a)
    xs, ys = [], []
    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        xs.append(float(center_x_m) + dx * ca - dy * sa)
        ys.append(float(center_y_m) + dx * sa + dy * ca)
    return min(xs), max(xs), min(ys), max(ys)


def piezo_half_range_m(calib_m_per_v, limit_low_v=None, limit_high_v=None,
                       limits_enabled=False):
    """由每 DAC 伏的 Piezo Calibration 与当前电压限位计算可达半程。calibration 已包含放大器增益，不能再次乘增益；更窄电压限位相应收紧范围。"""
    if calib_m_per_v is None:
        return None
    v = 10.0
    if limits_enabled and limit_low_v is not None and limit_high_v is not None:
        v = min(v, abs(float(limit_low_v)), abs(float(limit_high_v)))
    return abs(float(calib_m_per_v)) * v


def frame_exceeds(center_x_m, center_y_m, width_m, height_m, angle_deg,
                  half_x_m, half_y_m):
    """检查扫描框是否越出 ±half 范围，返回超出量或 None。
    扫描框读回与运动范围检查解决不同问题：前者检查设置是否被改变，
    后者避免输出电压达到限位而使部分像素失去预期空间采样意义。
    仅凭仪器接受设置并原样回显，不能证明整个框都可达。
    硬件半程读不到时返回 None 表示不能断言，调用方应保留未知状态。
    此闸门用于 ConfigureScan 的新请求；还原路径负责恢复先前读到的设置，
    不能借恢复操作替调用方改变原值。实际开始扫描前应检查其适用性。
    """
    if half_x_m is None or half_y_m is None:
        return None
    x0, x1, y0, y1 = frame_extent(center_x_m, center_y_m, width_m, height_m,
                                  angle_deg)
    over = {}
    if x1 > float(half_x_m):
        over["x_high_m"] = x1 - float(half_x_m)
    if x0 < -float(half_x_m):
        over["x_low_m"] = -float(half_x_m) - x0
    if y1 > float(half_y_m):
        over["y_high_m"] = y1 - float(half_y_m)
    if y0 < -float(half_y_m):
        over["y_low_m"] = -float(half_y_m) - y0
    if not over:
        return None
    over["extent_m"] = [x0, x1, y0, y1]
    over["half_m"] = [float(half_x_m), float(half_y_m)]
    return over


def frame_readback_mismatch(requested, readback):
    """比较请求帧与仪器读回帧，返回差异字典或 None。
    输入均为 (cx, cy, w, h, angle_deg)。请求被接受不等于设置实际生效，
    返回请求值也不能当作读回证据。缺失字段返回 {"unreadable": True}，不能当作一致。
    """
    if readback is None:
        return {"unreadable": True}
    try:
        req = [float(v) for v in requested[:5]]
        got = [float(v) for v in readback[:5]]
    except (TypeError, ValueError, IndexError):
        return {"unreadable": True}
    if len(req) < 5 or len(got) < 5:
        return {"unreadable": True}
    names = ("center_x_m", "center_y_m", "width_m", "height_m", "angle_deg")
    diff = {}
    for i, nm in enumerate(names):
        if not (math.isfinite(req[i]) and math.isfinite(got[i])):
            return {"unreadable": True}
        if nm == "angle_deg":
            if abs(got[i] - req[i]) > _FRAME_TOL_DEG:
                diff[nm] = {"requested": req[i], "readback": got[i]}
            continue
        tol = max(abs(req[i]) * _FRAME_TOL_FRAC, _FRAME_TOL_ABS_M)
        if abs(got[i] - req[i]) > tol:
            diff[nm] = {"requested": req[i], "readback": got[i],
                        "delta_m": got[i] - req[i]}
    return diff or None


def _match_signal(query: str, sig_names: list[str]) -> int | None:
    """Match a channel name like 'Z' or 'Current' against Nanonis signal names
    like 'Z (m)' or 'Current (A)'.  Returns global signal index or None."""
    q = query.strip()
    for idx, name in enumerate(sig_names):
        if name == q:
            return idx
    ql = q.lower()
    for idx, name in enumerate(sig_names):
        nl = name.lower()
        if nl.startswith(ql) and (len(nl) == len(ql) or nl[len(ql)] in (' ', '(')):
            return idx
    return None


class StartScan(BaseSkill):
    """发起一次受控扫描，并检查有限帧工作流所依赖的连续扫描状态。
    ScanAt、FullScan 与 WaitScanComplete 依赖一次开始对应一帧的约定。
    Scan_PropsSet 同时写入多个属性；模块清单未知时不构造猜测值覆盖用户设置。
    因此开始前仍需读回 continuous，不能把未完成配置当成连续扫描已经关闭。
    默认仅在 continuous 明确关闭时放行；开启或未知时拒绝。
    调用方可修复读取、在控制器关闭连续扫描，或显式使用 allow_continuous_scan。
    覆盖会告警并记录 continuous_scan_override_used，使下游知道有限帧前提不再成立。
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StartScan",
            version="1.1.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="用当前参数启动一次扫描。",
            preconditions=["z_controller_on"],
            parameters=[
                ParameterSpec(
                    name="direction", type="str", required=False, default="down",
                    allowed_values=["down", "up"],
                    description=(
                        "扫描方向。``down`` = 从帧顶往下扫(默认);"
                        "``up`` = 从帧底往上扫。"
                        "对应 ``Scan_Action(0, dir)`` 的第二个参数(0=down / 1=up)。"
                    ),
                ),
                ParameterSpec(
                    name="allow_continuous_scan",
                    type="bool",
                    required=False,
                    default=False,
                    description=(
                        "**用户级覆盖,不是重试开关。** 置 True 表示:即使 MAST "
                        "无法确认 Nanonis 的 Continuous scan 已经关掉,也照样发起"
                        "扫描。代价是具体的 —— 扫描可能一帧接一帧永不停止,于是 "
                        "WaitScanComplete 只能等满超时(outcome=restarted),而 "
                        "SaveScan / 扫描地图登记拿到的可能是第 N+2 帧而不是第 N 帧。"
                        "**先试另外两条出口**:把那次读失败当 bug 修掉;或者直接在 "
                        "Nanonis 的 Scan 模块里关掉 Continuous scan(那样下一次 "
                        "StartScan 读到「关」就直接放行)。"
                    ),
                ),
            ],
            estimated_duration_s=1.0,
            rollback_skill="StopScan",
            composition_level=1,
            tags=["scan", "imaging", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []

        # 保存模块清单遵循读到则原样保留、读不到则不猜测写入的原则。
        # Scan_PropsSet 会同时更新多个属性，清单未知时不发整条设置命令。
        # 这也意味着 continuous/autosave 未必完成配置，必须由后续读回与闸门检查。
        # 读取失败、确认零个模块、读到非空清单是三种状态；确认空清单时传空数组不丢失既有选择。
        module_names: list[str] = []
        module_names_source = "unchanged"
        module_count: "int | None" = None
        continuous_before: "int | None" = None
        series_name = ""
        parsed_before, read_error = _read_scan_props(context, calls)
        if parsed_before is not None:
            continuous_before = _scan_props_continuous(parsed_before)
            series_name = _scan_props_series_name(parsed_before)
            module_count = _scan_props_module_count(parsed_before)
            heuristic = _scan_props_modules(parsed_before)
            if module_count is not None:
                # 声明的个数与解出来的数组对得上 ⇒ 按位置取,不靠「找第一个全字符串
                # 的 list」。回包里其实有**两个**全字符串数组(模块名 + 参数表)。
                module_names = [str(s) for s in parsed_before[2][_PROPS_IX_MODULES]]
                module_names_source = "read" if module_names else "read_empty"
            elif heuristic:
                module_names = heuristic
                module_names_source = "read"
            else:
                read_error = "Scan_PropsGet 回包里没有模块名数组"

        if module_names_source == "unchanged":
            # **出声**。原来这里是 `except: pass` + 硬编码兜底,连「我用了兜底值」
            # 都不说 —— 于是用户只能从文件里发现设置被改了。
            _log.warning(
                "StartScan: 读不到当前的「保存哪些模块参数」清单(%s)—— "
                "**不下发 Scan_PropsSet**,用户配的清单保持原样。"
                "代价:本次不设 continuous=Off / autosave=All,它们保持仪器当前设置。",
                read_error or "原因未知")

        # Scan_PropsSet(cont, bouncy, autosave, basename, comment, modules_names, flags)
        # Build a basename from the active experiment + sample (v0.3.12) so files
        # come out as "experiment_sample_0001.sxm". When there's
        # NO active experiment the builder returns "" — the old code then passed
        # "" as the series name, which clobbers the operator's own configured
        # filename prefix down to "unnamed####". Preserve the
        # current series name in that case.
        #
        # ⚠️ 序列名取自**上面那一次读**,不再单开一发 ``Scan_PropsGet``。原来那发
        # 独立于模块清单那一发:两者可以给出不同的答案,而分歧不是无害的 —— 第一发
        # 成功(于是这一整发会下出去)、第二发失败(于是 basename 变成空串)时,
        # 写下去的空串**会把用户配的文件名前缀打回 ``unnamed####``**。
        # 一次读、一个答案,顺便少一个 TCP 往返。
        basename = _build_scan_basename() or series_name
        #
        # Continuous is set OFF, not left alone. Every caller of StartScan —
        # ScanAt, FullScan, WaitScanComplete, the vision monitor, the scan-map
        # registration — is written against "one start, one frame"; leaving the
        # flag to whatever the GUI last had makes that assumption a coin flip
        # decided outside MAST. ContinuousImaging_Auto does NOT want the Nanonis
        # flag either: it loops FullScan at the MAST level, one frame per cycle,
        # so it wants the same Off.
        #
        # To put it back: the argument to answer is not "why did you turn it
        # off", it is "what does MAST do when a scan never ends?" Until there is
        # a wait that can succeed against an endless scan, and callers that can
        # tell frame N from frame N+1, On is a setting that breaks them all.
        props_written = module_names_source in ("read", "read_empty")
        if props_written:
            rec_props = context.safe_call(
                "Scan_PropsSet", _SET_OFF, _SET_NO_CHANGE, _SET_AUTOSAVE_ALL,
                basename, "", module_names, 0,
            )
            calls.append(rec_props)

        # Read back. A rig that refuses the write must not come out looking like
        # one that accepted it — that is the whole reason this is two calls and
        # not one, and it is why the encodings above were checked separately.
        parsed_after, readback_error = _read_scan_props(context, calls)
        continuous_after: "int | None" = (
            None if parsed_after is None
            else _scan_props_continuous(parsed_after))
        # **读不到不是「没开着」。** 2026-08-19：``Scan_PropsGet`` 解析崩掉时
        # continuous_after 是 None，`None == _GET_ON` 为假，于是这一路把
        # `continuous_scan_still_on` 报成 **False** —— 而那一帧的 wait 结果
        # 恰恰是 `restarted`，也就是它其实开着。故障答成了「一切正常」。
        #
        # 现在它是三态,而且**有人读它**:下面那道闸门。此前这个字段全仓没有一个
        # 消费者 —— 生产方写得再仔细,没人读就等于没说。
        still_on = _continuous_state(continuous_after)
        override = bool(params.get("allow_continuous_scan", False))

        data: dict = {
            "saved_modules": module_names,
            # 「这份清单是读到的,还是我们没动」—— 两者的下游动作不同,而从
            # `saved_modules` 本身分不出来(读不到时它是空表,看起来像"没有模块"
            # 而不是"没问到")。要求的那个 bug 就活在这个区分里。
            # 三个值:read(读到 N 个) / read_empty(读到确实是 0 个) / unchanged(问不出来)。
            "module_names_source": module_names_source,
            "module_names_read_error": read_error or None,
            # 仪器**自己声明**的模块个数。它与 saved_modules 的长度是两个独立字段,
            # 因为「数组解出来是空的」和「仪器说有 0 个」不是同一句话。
            "module_names_count_declared": module_count,
            # 读不到清单时整发 Scan_PropsSet 都没下:continuous / autosave 保持
            # 仪器当前设置。下游若依赖自动保存,必须看得见这件事。
            "scan_props_written": props_written,
            # GET encoding (0=Off, 1=On, None=could not read). Reported even
            # when it is the boring 0, so "we turned it off" and "we never
            # managed to look" stay two different answers downstream.
            "continuous_scan_before": continuous_before,
            "continuous_scan_after": continuous_after,
            "continuous_readback_error": readback_error or None,
            # continuous 三态：True 开启、False 确认关闭、None 不可读或非法值。
            # 不能通过与开启值作布尔比较，把 None 错误转换成关闭。
            "continuous_scan_still_on": still_on,
            "continuous_scan_override_used": override and still_on is not False,
        }

        # ══════════════════════════════════════════════════════════════════
        # 闸门:不发起一次「停不下来」的扫描
        # ══════════════════════════════════════════════════════════════════
        gate_error = self._continuous_gate(
            still_on=still_on, override=override,
            continuous_before=continuous_before,
            module_names_source=module_names_source,
            read_error=read_error, readback_error=readback_error,
        )
        if gate_error:
            data["scan_running"] = False
            data["vision_monitor"] = False
            return SkillResult(
                skill_name="StartScan", success=False, error=gate_error,
                data=data, nanonis_calls=calls,
            )

        # 扫描方向必须显式参数化，且与下游几何解释一致。
        # 协议使用 0=down、1=up；up 帧的数组行序需要归位，
        # 否则特征提取返回的物理 y 坐标会翻转。统一通过 sxm_oriented_frames 处理。
        direction = str(params.get("direction", "down") or "down").strip().lower()
        if direction not in ("down", "up"):
            data["scan_running"] = False
            data["vision_monitor"] = False
            return SkillResult(
                skill_name="StartScan", success=False,
                error=(f"direction 只能是 'down' 或 'up',收到 {direction!r}。"
                       "**不猜**:猜错的代价是一整帧扫在错误方向上,而文件头会"
                       "如实记下它,于是后面每一处按方向归位的分析都跟着错。"),
                data=data,
                nanonis_calls=calls,
            )
        record = context.safe_call("Scan_Action", 0, 1 if direction == "up" else 0)
        calls.append(record)
        if record.error:
            data["scan_running"] = False
            data["vision_monitor"] = False
            return SkillResult(
                skill_name="StartScan",
                success=False,
                error=record.error,
                data=data,
                nanonis_calls=calls,
            )

        # Phase 9: spawn the scan-progress vision monitor. At 12.5 % … 100 % it
        # grabs the partial frame, runs M12, and publishes a Chinese narration
        # into the buffer. Fully fail-safe (no buffer / no vision → no-op) and
        # off the graph (own daemon thread), so it can never break the scan.
        monitor_started = False
        try:
            from mast.vision.scan_monitor import start_scan_vision_monitor
            pool = getattr(context, "pool", None)
            abort_event = getattr(context, "_abort", None)
            if pool is not None:
                mon = start_scan_vision_monitor(
                    pool, scan_id=basename, abort_event=abort_event,
                )
                monitor_started = mon is not None
        except Exception:  # noqa: BLE001 — monitoring must never break scanning
            monitor_started = False

        data["scan_running"] = True
        data["vision_monitor"] = monitor_started
        return SkillResult(
            skill_name="StartScan",
            success=True,
            data=data,
            nanonis_calls=calls,
        )

    # ------------------------------------------------------------------
    # The gate. One place, one rule, so "put it back" is one edit.
    # ------------------------------------------------------------------

    @staticmethod
    def _continuous_gate(
        *, still_on: "bool | None", override: bool,
        continuous_before: "int | None", module_names_source: str,
        read_error: str, readback_error: str,
    ) -> str:
        """空串 = 可以起扫;否则返回「为什么不起」。

        The rule is one clause on purpose: **start iff the read-back said Off.**
        Nothing is inferred from "it was Off before and we wrote Off" — that
        would be reasoning around missing evidence instead of going and getting
        it, and getting it is what :data:`_PROPS_GET_ATTEMPTS` is for.

        ⚠️ 这段话分四种,不是两种。**「机器不认这次写入」和「我们根本没写」指向
        完全不同的下一步**(前者去查仪器/权限,后者去查那次读),而把两者说成
        同一句话,就是把人送去查一件没发生的事 —— 本仓在 `WaitScanComplete`
        的「中途停止 vs 从没开始」上已经付过一次这个学费。
        """
        if still_on is False:
            return ""

        written = module_names_source in ("read", "read_empty")
        why_not_written = (f"读「保存哪些模块参数」清单失败:"
                           f"{read_error or '原因未知'} —— 而协议里没有"
                           f"「模块名保持原样」这一档,所以整发 Scan_PropsSet "
                           f"都不敢下,免得清空用户配好的清单")
        if still_on is True and written:
            head = ("Nanonis 的 **Continuous scan 是开着的**,而这次**写了也没关掉**"
                    "(下发 Scan_PropsSet 之后回读仍然是「开」)。")
        elif still_on is True:
            head = (f"Nanonis 的 **Continuous scan 是开着的**,而这次**根本没能写**"
                    f"({why_not_written})。")
        elif written:
            head = (f"下发了 Scan_PropsSet,但**回读不到** Continuous scan 状态"
                    f"({readback_error or '回包里没有这个标志'}),"
                    f"无法确认它已经关掉。")
        else:
            head = (f"**读不到** Nanonis 的 Continuous scan 状态,而且这次也没能写"
                    f"({why_not_written})。")

        if override:
            _log.warning(
                "StartScan: %s 但调用方传了 allow_continuous_scan=True —— "
                "照常发起扫描。这一帧可能永不结束:WaitScanComplete 会等满超时"
                "(outcome=restarted),SaveScan / 地图登记拿到的可能不是这一帧。",
                head)
            return ""

        _log.warning("StartScan: %s **不发起扫描。**", head)
        return (
            f"{head}\n"
            "**不发起这次扫描** —— 一次 start 就不再等于一帧:扫描会一帧接一帧地"
            "跑下去,WaitScanComplete 只能等满超时(outcome=restarted),而 "
            "SaveScan / 扫描地图登记拿到的可能是第 N+2 帧。发出去比不发更坏,"
            "而且坏在看不见的地方。\n"
            f"(读回的 GET 值:before={continuous_before}, 0=关 1=开;"
            f"模块清单来源={module_names_source})\n"
            "三条出口:\n"
            "  1. **修那次读**。读不到通常本身就是 bug(2026-08-19 那次是 "
            "Scan_PropsGet 的 `*+i` 解析崩了),它值得被看见而不是被绕过。\n"
            "  2. **在 Nanonis 的 Scan 模块里手动关掉 Continuous scan**。"
            "下一次 StartScan 读到「关」就直接放行 —— 根本不需要写,也就不用冒"
            "清空「保存哪些模块参数」的险。\n"
            "  3. 明确接受风险:`allow_continuous_scan=True`。**这是用户的决定,"
            "不是自动重试的开关** —— agent 遇到这条应当把上面两条报给用户,"
            "而不是自己把它打开重来一遍(打开之后拿到的那一帧,可能不是你以为的"
            "那一帧)。"
        )


class StopScan(BaseSkill):
    """Stop scanning."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopScan",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="停止当前的扫描。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["scan", "imaging", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # Stop the scan-progress vision monitor promptly (it would otherwise
        # only notice via its next Scan_StatusGet poll). Fail-safe.
        try:
            from mast.vision.scan_monitor import stop_active_monitor
            stop_active_monitor(join_timeout=1.0)
        except Exception:  # noqa: BLE001
            pass
        record = context.safe_call("Scan_Action", 1, 0)
        if record.error:
            return SkillResult(
                skill_name="StopScan",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="StopScan",
            success=True,
            data={"scan_running": False},
            nanonis_calls=[record],
        )


class ConfigureScan(BaseSkill):
    """Configure scan frame parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureScan",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置扫描框：中心、尺寸、角度，以及采集通道。"
                "默认还会顺带设置由 line_time_s 推导出的扫描速度"
                "（线速度 = width_m / line_time_s）；传 set_scan_speed=False "
                "则保持当前扫描速度不动。"
            ),
            parameters=[
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description="扫描中心的 X 坐标，单位米",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description="扫描中心的 Y 坐标，单位米",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="width_m",
                    type="float",
                    description=(
                        "扫描宽度，单位**米**（SI），**不是**纳米。"
                        "换算：100 nm → 100n，50 nm → 50n。像 "
                        "100（=100 m）这样的裸数值是单位错误，会被拒绝。"
                    ),
                    unit="m",
                    required=True,
                    min_value=1e-10,
                    max_value=1e-5,   # = config scan_size_max_m (10 µm); guards nm→m slips
                ),
                ParameterSpec(
                    name="height_m",
                    type="float",
                    description=(
                        "扫描高度，单位**米**（SI），**不是**纳米。"
                        "换算：100 nm → 100n，50 nm → 50n。像 "
                        "100（=100 m）这样的裸数值是单位错误，会被拒绝。"
                    ),
                    unit="m",
                    required=True,
                    min_value=1e-10,
                    max_value=1e-5,   # = config scan_size_max_m (10 µm); guards nm→m slips
                ),
                ParameterSpec(
                    name="angle_deg",
                    type="float",
                    description=(
                        "扫描角度，单位度。**省略**它则保持当前扫描框的"
                        "角度不变（角度会从仪器回读；若这次回读失败，"
                        "这个技能就**失败**，它绝不会臆断为 "
                        "0°）。想要与坐标轴对齐的扫描框，就显式传 "
                        "0。"
                    ),
                    unit="deg",
                    required=False,
                    # 刻意 None 而不是 0.0(同 full_scan.line_time_s 那条注释):
                    # wrap_skill 会把 ParameterSpec.default 物化进 pydantic 字段,
                    # 写着 0.0 时 LLM 路径上「没传」永远到达为「显式 0.0」,
                    # 下面 execute 里「省略=保持当前角度」整条就成了死代码。
                    default=None,
                    min_value=-180.0,
                    max_value=180.0,
                ),
                ParameterSpec(
                    name="channels",
                    type="str",
                    description=(
                        "要采集的通道名，逗号分隔。默认：'Z,Current'。"
                        "做 STS mapping 时加上 lock-in：'Z,Current,LI Demod 1 X,LI Demod 1 Y'"
                    ),
                    required=False,
                    default="Z,Current",
                ),
                # 审查: ConfigureScan used to *silently* overwrite the
                # scan speed to width/0.1s (a hardcoded 0.1 s line time) on every
                # call, with no way for the caller to opt out — a hidden side
                # effect that clobbered any speed the user/LLM had deliberately set
                # via SetScanSpeed. Make it explicit and controllable: set_scan_speed
                # gates the write entirely, and line_time_s exposes the derived line
                # time so the resulting linear speed (width / line_time_s) is no
                # longer a buried magic constant.
                ParameterSpec(
                    name="set_scan_speed",
                    type="bool",
                    description=(
                        "为 True（默认）时，顺带设置由 line_time_s 推导出的"
                        "扫描速度（正扫线速度 = width_m / line_time_s，"
                        "保持每行耗时恒定）。设为 False 则**只**配置扫描框 + 通道，"
                        "并保持当前扫描速度不动"
                        "（要显式控制速度请用 SetScanSpeed）。"
                    ),
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="line_time_s",
                    type="float",
                    description=(
                        "每条扫描线的耗时，单位秒；当 set_scan_speed 为 True "
                        "时用它推导扫描速度。**留空则按这个扫描框尺寸"
                        "套用出厂的 scan tier**（8 nm -> "
                        "原子级档 -> 1.2 s/line）。set_scan_speed 为 False "
                        "时本项忽略。"
                    ),
                    unit="s",
                    required=False,
                    # **不是 0.1**。有默认值的话「没传」与「传了 0.1」在
                    # execute 里分不开，于是查档位表那条路永远到不了 ——
                    # 一段读起来完全正常、却从不执行的代码。
                    default=None,
                    min_value=1e-4,
                    max_value=600.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=1,
            tags=["scan", "imaging", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        center_x = params["center_x_m"]
        center_y = params["center_y_m"]
        width = params["width_m"]
        height = params["height_m"]
        channels_str = params.get("channels", "Z,Current")
        set_scan_speed = params.get("set_scan_speed", True)
        # 线时间按显式参数、档位配置、兜底的顺序解析。
        # 缺省常量不能悄悄覆盖当前任务声明的扫描条件。
        from mast.core.scan_policy import resolve_line_time
        _size_for_tier = max(float(params.get("width_m") or 0.0),
                             float(params.get("height_m") or 0.0))
        line_time, line_time_source = resolve_line_time(
            params.get("line_time_s"), _size_for_tier)

        calls = []

        # 未指定角度时保留当前扫描角度，不能把重定位或缩放隐式变成旋转。
        # 读取异常、错误记录、回包形状异常与字段缺失都会造成未知，均需拒绝。
        # Scan_FrameSet 必须提供角度，因此读不到时不能用零代替；明确要求零角度的调用方应显式传值。
        angle = params.get("angle_deg")
        if angle is None:
            angle = None
            try:
                rec_fg = context.safe_call("Scan_FrameGet")
                calls.append(rec_fg)
                if not rec_fg.error:
                    fg = getattr(rec_fg, "return_value", None)
                    vals = fg[2] if (isinstance(fg, (list, tuple)) and len(fg) > 2) else None
                    if isinstance(vals, (list, tuple)) and len(vals) >= 5:
                        cand = float(vals[4])
                        if math.isfinite(cand):
                            angle = cand
            except Exception:  # pragma: no cover - 读失败与读不懂同等处理
                angle = None
            if angle is None:
                return SkillResult(
                    skill_name=self.metadata().name,
                    success=False,
                    error=(
                        "无法读回当前扫描角度(Scan_FrameGet 失败或回包读不懂),"
                        "拒绝配置扫描帧 —— 用一个假定的 0° 会**静默地把扫描框转正**,"
                        "而调用方要的是「保持当前角度」。"
                        "若确实要 0°,请显式传 angle_deg=0.0。"
                    ),
                    nanonis_calls=calls,
                )

        rec_frame = context.safe_call(
            "Scan_FrameSet", center_x, center_y, width, height, angle,
        )
        calls.append(rec_frame)
        if rec_frame.error:
            return SkillResult(
                skill_name="ConfigureScan",
                success=False,
                error=rec_frame.error,
                nanonis_calls=calls,
            )

        # 帧设置后读取仪器实际接受的中心、尺寸与角度。
        # 返回请求回声不能证明生效；读回一致性与硬件量程检查是不同的约束。
        frame_readback = None
        try:
            rec_fg2 = context.safe_call("Scan_FrameGet")
            calls.append(rec_fg2)
            if not rec_fg2.error:
                fg2 = getattr(rec_fg2, "return_value", None)
                v2 = fg2[2] if (isinstance(fg2, (list, tuple)) and len(fg2) > 2) else None
                if isinstance(v2, (list, tuple)) and len(v2) >= 5:
                    frame_readback = [_unwrap_scalar(x) for x in v2[:5]]
        except Exception:  # pragma: no cover — 读失败与读不懂同等处理
            frame_readback = None

        mismatch = frame_readback_mismatch(
            (center_x, center_y, width, height, angle), frame_readback)
        if mismatch is not None:
            if mismatch.get("unreadable"):
                return SkillResult(
                    skill_name="ConfigureScan", success=False,
                    error=("Scan_FrameSet 发出去了，但**读不回**扫描帧"
                           "(Scan_FrameGet 失败或回包读不懂) —— 拒绝继续。"
                           "「读不到」不是「设上了」：仪器可能已经把框夹到量程内，"
                           "而此后每一张图的坐标都会是假的。"),
                    data={"requested_frame": [center_x, center_y, width, height, angle],
                          "frame_readback": None},
                    nanonis_calls=calls,
                )
            parts = ", ".join(
                "%s 要 %.4g 实得 %.4g" % (k, v["requested"], v["readback"])
                for k, v in mismatch.items())
            return SkillResult(
                skill_name="ConfigureScan", success=False,
                error=("仪器没有接受这个扫描帧，它把框改了：%s。"
                       "**拒绝，不夹紧** —— 按夹紧后的框扫出来的图，坐标与"
                       "请求的对不上，而图看着完全正常。"
                       "把中心挪回量程内，或把 size_m 调小。" % parts),
                data={"requested_frame": [center_x, center_y, width, height, angle],
                      "frame_readback": frame_readback,
                      "frame_mismatch": mismatch},
                nanonis_calls=calls,
            )

        # 越界闸门 —— 见 frame_exceeds 的说明。**读回闸门抓不到这一格**：
        # 仪器照单全收超范围的帧，被夹的是扫描时的输出电压。
        frame_over = None
        half_x = half_y = None
        try:
            rec_cal = context.safe_call("Piezo_CalibrGet")
            calls.append(rec_cal)
            if not rec_cal.error:
                cv = getattr(rec_cal, "return_value", None)
                vals = cv[2] if (isinstance(cv, (list, tuple)) and len(cv) > 2) else None
                if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                    lim = {}
                    try:
                        rec_lim = context.safe_call("Piezo_XYZLimitsGet")
                        calls.append(rec_lim)
                        if not rec_lim.error:
                            lv = getattr(rec_lim, "return_value", None)
                            lvals = (lv[2] if (isinstance(lv, (list, tuple))
                                               and len(lv) > 2) else None)
                            if isinstance(lvals, (list, tuple)) and len(lvals) >= 5:
                                lim = {"on": bool(int(_unwrap_scalar(lvals[0]))),
                                       "xl": float(_unwrap_scalar(lvals[1])),
                                       "xh": float(_unwrap_scalar(lvals[2])),
                                       "yl": float(_unwrap_scalar(lvals[3])),
                                       "yh": float(_unwrap_scalar(lvals[4]))}
                    except Exception:  # pragma: no cover
                        lim = {}
                    half_x = piezo_half_range_m(
                        _unwrap_scalar(vals[0]), lim.get("xl"), lim.get("xh"),
                        lim.get("on", False))
                    half_y = piezo_half_range_m(
                        _unwrap_scalar(vals[1]), lim.get("yl"), lim.get("yh"),
                        lim.get("on", False))
                    frame_over = frame_exceeds(center_x, center_y, width, height,
                                               angle, half_x, half_y)
        except Exception:  # pragma: no cover — 读不到半程 = 判不了，见下
            half_x = half_y = None
        if frame_over:
            over_nm = ", ".join("%s 超 %.0f nm" % (k, v * 1e9)
                                for k, v in frame_over.items()
                                if k.endswith(("_high_m", "_low_m")))
            return SkillResult(
                skill_name="ConfigureScan", success=False,
                error=("扫描帧伸出压电量程：%s（半程 ±%.0f/±%.0f nm）。拒绝该请求，不自动夹紧；设置被接受并原样读回仍不能保证扫描输出不会达到限位。请将中心移回量程内或减小 size_m。"
                       % (over_nm, (half_x or 0) * 1e9, (half_y or 0) * 1e9)),
                data={"requested_frame": [center_x, center_y, width, height, angle],
                      "frame_readback": frame_readback,
                      "frame_exceeds_piezo_range": frame_over,
                      "piezo_half_x_m": half_x, "piezo_half_y_m": half_y},
                nanonis_calls=calls,
            )
        if half_x is None or half_y is None:
            # **「读不到」不是「没超」** —— 留一行，让这一格没验成这件事看得见。
            _log.warning("读不到压电半程(Piezo_CalibrGet)，扫描帧越界这一格未验")

        requested = [ch.strip() for ch in channels_str.split(",") if ch.strip()]
        if requested:
            try:
                rec_signals = context.safe_call("Signals_NamesGet")
                calls.append(rec_signals)
                if not rec_signals.error and isinstance(rec_signals.return_value, (list, tuple)):
                    # return_value is (error_string, raw_bytes, Variables).
                    # Signals.NamesGet ResponseTypes = ["i", "i", "*+c"] → Variables ==
                    # [size, count, [name0, name1, ...]]. The names array is the nested
                    # list-of-strings at parsed[2][2]; scan the Variables for the first
                    # list-of-strings as a defensive fallback.
                    parsed = rec_signals.return_value
                    sig_names: list[str] = []
                    if len(parsed) > 2 and isinstance(parsed[2], list):
                        for v in parsed[2]:
                            if isinstance(v, list) and v and isinstance(v[0], str):
                                sig_names = v
                                break

                    if sig_names:
                        channel_indexes = []
                        for ch in requested:
                            idx = _match_signal(ch, sig_names)
                            if idx is not None:
                                channel_indexes.append(idx)
                            else:
                                _log.warning("Channel '%s' not found in signals, skipping", ch)

                        if channel_indexes:
                            rec_buf = context.safe_call(
                                "Scan_BufferSet", channel_indexes, 0, 0,
                            )
                            calls.append(rec_buf)
                            if rec_buf.error:
                                _log.warning("Scan_BufferSet failed: %s", rec_buf.error)
            except Exception as exc:
                _log.warning("Channel selection failed: %s", exc)

        # Optionally set the scan speed, derived from the (now explicit) line
        # time. This used to run unconditionally with a hardcoded 0.1 s line
        # time, silently clobbering any speed set earlier via SetScanSpeed; it
        # is now opt-out via set_scan_speed and the line time is caller-supplied.
        speed = None
        speed_set = False
        if set_scan_speed:
            try:
                line_time = float(line_time)
            except (TypeError, ValueError):
                line_time = 0.1
            speed = width / line_time if line_time > 0 else 500e-9
            # Keep_parameter_constant=2 → keep the time-per-line constant so the
            # caller's line_time_s stays authoritative (forward/backward speed and
            # time are supplied consistently as width / line_time).
            rec_speed = context.safe_call(
                "Scan_SpeedSet", speed, speed, line_time, line_time, 2, 1.0,
            )
            calls.append(rec_speed)
            if rec_speed.error:
                _log.warning("Scan_SpeedSet failed: %s", rec_speed.error)
            else:
                speed_set = True

        return SkillResult(
            skill_name="ConfigureScan",
            success=True,
            data={
                # 读回值，不是请求值 —— 已与请求逐项核对过（见上面的读回闸门），
                # 所以这四个数是**仪器里真正的帧**，不是回声。
                "center_x_m": frame_readback[0],
                "center_y_m": frame_readback[1],
                "width_m": frame_readback[2],
                "height_m": frame_readback[3],
                "angle_deg": frame_readback[4],
                "frame_verified": True,
                "requested_frame": [center_x, center_y, width, height, angle],
                "frame_exceeds_piezo_range": None,
                "piezo_half_x_m": half_x,
                "piezo_half_y_m": half_y,
                "channels": channels_str,
                "scan_speed_set": speed_set,
                "line_time_s": line_time if set_scan_speed else None,
                "line_time_source": line_time_source if set_scan_speed else None,
                "linear_speed_m_s": speed,
            },
            nanonis_calls=calls,
        )


class SetScanSpeed(BaseSkill):
    """Set scan speed parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetScanSpeed",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置扫描速度：正/反扫速度，或每行耗时。",
            parameters=[
                ParameterSpec(
                    name="fwd_speed",
                    type="float",
                    description="正扫速度，单位 m/s",
                    unit="m/s",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="bwd_speed",
                    type="float",
                    description="反扫速度，单位 m/s",
                    unit="m/s",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="fwd_line_time",
                    type="float",
                    description="正扫每行耗时，单位秒",
                    unit="s",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="bwd_line_time",
                    type="float",
                    description="反扫每行耗时，单位秒",
                    unit="s",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="keep_const",
                    type="int",
                    # Matches Nanonis Scan_SpeedSet "Keep parameter constant"
                    # EXACTLY (审查 — the old labels were off by one
                    # and 2 was unreachable): 0=no change, 1=keep linear SPEED
                    # constant, 2=keep TIME-per-line constant.
                    description=("0 = 不变，1 = 保持线速度恒定，"
                                 "2 = 保持每行耗时恒定"),
                    required=False,
                    default=0,
                    allowed_values=[0, 1, 2],
                ),
                # 审查 [#139]: execute() read params.get("speed_ratio")
                # but the parameter was never declared, so it was a dead read —
                # permanently pinned to the fallback (1) and impossible for the
                # GUI/LLM/composite to set. Declare it (Nanonis Scan_SpeedSet's
                # forward/backward speed-ratio arg) with the neutral 1.0 default
                # so it's actually controllable.
                ParameterSpec(
                    name="speed_ratio",
                    type="float",
                    description="正扫/反扫速度比（1.0 = 对称）",
                    required=False,
                    default=1.0,
                    min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["scan", "speed", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        fwd_speed = params["fwd_speed"]
        bwd_speed = params["bwd_speed"]
        fwd_line_time = params["fwd_line_time"]
        bwd_line_time = params["bwd_line_time"]
        keep_const = params.get("keep_const", 0)

        # [#139]: fallback matches the ParameterSpec default (1.0). Nanonis wants
        # a float32 here; ConfigureScan already passes 1.0, so stay consistent.
        speed_ratio = params.get("speed_ratio", 1.0)
        record = context.safe_call(
            "Scan_SpeedSet",
            fwd_speed,
            bwd_speed,
            fwd_line_time,
            bwd_line_time,
            keep_const,
            speed_ratio,
        )
        if record.error:
            return SkillResult(
                skill_name="SetScanSpeed",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetScanSpeed",
            success=True,
            data={
                "fwd_speed": fwd_speed,
                "bwd_speed": bwd_speed,
                "fwd_line_time": fwd_line_time,
                "bwd_line_time": bwd_line_time,
                "keep_const": keep_const,
                "speed_ratio": speed_ratio,
            },
            nanonis_calls=[record],
        )


class GetScanXYPosition(BaseSkill):
    """Read current scan X/Y position."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetScanXYPosition",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前扫描的 X、Y 位置。",
            parameters=[
                ParameterSpec(
                    name="wait_for_newest",
                    type="bool",
                    description="是否等待最新的数据",
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["scan", "position", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        wait = params.get("wait_for_newest", True)
        record = context.safe_call("Scan_XYPosGet", 1 if wait else 0)
        if record.error:
            return SkillResult(
                skill_name="GetScanXYPosition",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). Scan.XYPosGet
        # ResponseTypes = ["f", "f"] → Variables == [X_m, Y_m]. Read parsed[2][0/1],
        # NOT parsed[0] (error string) or parsed[1] (raw bytes).
        parsed = record.return_value
        x_m = 0.0
        y_m = 0.0
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                x_m = float(vals[0])
                y_m = float(vals[1])
            elif isinstance(vals, (list, tuple)) and len(vals) == 1:
                x_m = float(vals[0])
            elif vals is not None and not isinstance(vals, (list, tuple)):
                x_m = float(vals)
        return SkillResult(
            skill_name="GetScanXYPosition",
            success=True,
            data={"x_m": x_m, "y_m": y_m},
            nanonis_calls=[record],
        )


class ScanBackgroundPaste(BaseSkill):
    """Paste current scan databuffer into the background."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ScanBackgroundPaste",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="把当前的扫描数据缓冲粘贴到 background。",
            parameters=[
                ParameterSpec(
                    name="wait_until_pasted",
                    type="bool",
                    description="等到数据粘贴完成再返回",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="timeout_ms",
                    type="int",
                    description="超时时长，单位毫秒（-1 = 无限等待）",
                    required=False,
                    default=-1,
                ),
            ],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["scan", "background", "paste", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        wait = 1 if params.get("wait_until_pasted", True) else 0
        timeout = params.get("timeout_ms", -1)
        record = context.safe_call("Scan_BackgroundPaste", wait, timeout)
        if record.error:
            return SkillResult(
                skill_name="ScanBackgroundPaste",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). Scan.BackgroundPaste
        # ResponseTypes = ["I"] → Variables == [Timed_out?]. Read parsed[2][0], NOT
        # parsed[0] (= the always-empty error string on success, which made
        # timed_out permanently False even after a real timeout).
        parsed = record.return_value
        timed_out = False
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) > 0:
                timed_out = bool(vals[0])
            elif vals is not None and not isinstance(vals, (list, tuple)):
                timed_out = bool(vals)
        return SkillResult(
            skill_name="ScanBackgroundPaste",
            success=True,
            data={"timed_out": timed_out},
            nanonis_calls=[record],
        )


class ScanBackgroundDelete(BaseSkill):
    """Delete pasted scan background(s)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ScanBackgroundDelete",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="删除最近一次、或全部已粘贴的 scan background。",
            parameters=[
                ParameterSpec(
                    name="wait_until_deleted",
                    type="bool",
                    description="等到数据删除完成再返回",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="timeout_ms",
                    type="int",
                    description="超时时长，单位毫秒（-1 = 无限等待）",
                    required=False,
                    default=-1,
                ),
                ParameterSpec(
                    name="delete_all",
                    type="bool",
                    description="True=删除全部 background，False=只删最近一次",
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["scan", "background", "delete", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        wait = 1 if params.get("wait_until_deleted", True) else 0
        timeout = params.get("timeout_ms", -1)
        which = 1 if params.get("delete_all", False) else 0
        record = context.safe_call(
            "Scan_BackgroundDelete", wait, timeout, which,
        )
        if record.error:
            return SkillResult(
                skill_name="ScanBackgroundDelete",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). Scan.BackgroundDelete
        # ResponseTypes = ["I"] → Variables == [Timed_out?]. Read parsed[2][0], NOT
        # parsed[0] (= the always-empty error string on success, which made
        # timed_out permanently False even after a real timeout).
        parsed = record.return_value
        timed_out = False
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) > 0:
                timed_out = bool(vals[0])
            elif vals is not None and not isinstance(vals, (list, tuple)):
                timed_out = bool(vals)
        return SkillResult(
            skill_name="ScanBackgroundDelete",
            success=True,
            data={"timed_out": timed_out, "delete_all": bool(which)},
            nanonis_calls=[record],
        )
