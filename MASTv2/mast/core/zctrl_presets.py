"""Z-controller parameter presets — the agent picks a NAME, code writes the numbers.

WHY THIS EXISTS
===============
2026-08-03, on the instrument. An agent was asked to set the approach gains and emitted
``p_gain=3`` where its own reasoning said ``3e-12`` — three METRES of proportional
gain, a factor of 10¹² off, silently. A second attempt turned ``0.0000000000030``
into ``0``. The Z loop was open, so nothing moved; with it closed, the first value
drives the tip into the sample the instant feedback engages.

The operator's conclusion, and the design here:

    不让 agent 负责写这个数字。写这个数字的应该是一个 python 脚本，agent 一发话，
    脚本就把默认参数组 A，或者默认参数组 B……写进去。

So the agent's vocabulary for hardware parameters becomes a NAME. There is no
numeric argument to corrupt. This is the same doctrine ``scan_resolver`` already
states for scan parameters ("移除它发明数字的诱因，而不是叮嘱它别乱填") — this
module extends it to the Z loop, which the scan path never covered.

NOT A NEW STORE — A RESOLVER
============================
The tempting design is a table of named groups holding gains and setpoints. It
would be a SECOND SOURCE OF TRUTH for values that already have one, and this repo
has paid for that before (PROGRESS.md, "差点踩的坑：第二真源" — ``tip_type`` was
added to instrument_profile and then reverted wholesale because the tip registry
already owned it).

The scan gains already live in the tier table (``scan_policy``), where the
operator edits them and from which ``ScanAt`` already writes them on every frame.
Putting a "scan" group anywhere else would mean an operator who changes 50 n → 180 n
in one place gets the old value on the next scan.

So a preset NAME resolves against whatever already owns those numbers:

    approach   → instrument_profile.approach_*     (new keys; nothing owned these)
    scan       → the tier for the CURRENT frame size (scan_policy)
    <tier名>   → that tier            (scan_policy)
    <自定义名> → SettingsStore["zctrl_presets"]     (the only new storage)

Custom names may not collide with a reserved name or a tier name; the check is at
CREATE time, so the two hot paths can never be shadowed by a custom group.

WHY THE CUSTOM GROUPS HOLD SI STRINGS
=====================================
A custom group is the one place a MODEL may supply numbers. They are stored and
accepted as SI-prefixed strings (``"3p"``, ``"180n"``) — see ``core.si_quantity``:
with the prefix mandatory, a dropped prefix is a parse ERROR rather than a
plausible number. The operator-maintained sources (instrument_profile, the tier
table) keep plain floats: those values are typed into a form by a human and never
pass through a language model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from mast.core import instrument_profile as _iprof
from mast.core import scan_policy
from mast.core.si_quantity import SIParseError, format_si, parse_si

logger = logging.getLogger(__name__)


# ── Reserved names ──────────────────────────────────────────────────────────
#: The two names that resolve against operator-owned sources. A custom group may
#: not take either, or an agent could shadow the approach parameters with its own.
PRESET_APPROACH = "approach"
PRESET_SCAN = "scan"
RESERVED_NAMES = (PRESET_APPROACH, PRESET_SCAN)

#: Where a resolved value came from. The UI and the tests assert on these.
SOURCE_PROFILE = "instrument_profile"
SOURCE_TIER_OPERATOR = "tier-operator"
SOURCE_TIER_FACTORY = "tier-factory"
SOURCE_CUSTOM = "custom"

MAX_PRESETS = 16
MAX_NAME_CHARS = 32


# ── Envelope ────────────────────────────────────────────────────────────────
# Rejected, never clamped. Silently trimming 8 V to 3 V leaves the operator
# believing they pulsed 8 V — the same argument tip_conditioning_resolver makes
# at length, and the reason instrument_profile's clamp-on-write is NOT relied on
# here (that path exists for a form the operator typed into, not for a value that
# came from a model).
#
# Deliberately wider than any real rig and narrower than SetZCtrlGain's own
# ceiling: this layer rejects "not a Z gain at all", the skill's ParameterSpec
# rejects "wrong for this instrument", and neither is the other's backstop.
_BOUNDS: dict[str, tuple[float, float]] = {
    "p_gain": (1e-15, 1e-6),        # metres
    "i_gain": (1e-12, 1e-3),        # metres/second
    "setpoint_a": (1e-12, 1e-7),    # amperes — aligned with SetSetpoint
}

#: Derived, not stored: Nanonis takes (P, T, I) and T = P / I. Storing T as well
#: would let it drift out of step with the pair that defines it.
_TIME_CONSTANT_BOUNDS = (1e-9, 10.0)


class PresetRejected(ValueError):
    """A preset that must not be stored. Nothing partial is ever written."""


# ── Storage (custom groups only) ────────────────────────────────────────────
_lock = threading.RLock()
_presets: "list[dict[str, Any]]" = []
_persist_sink: "Callable[[list[dict[str, Any]]], None] | None" = None


def _clean_name(raw: object) -> str:
    name = str(raw or "").strip()
    if not name:
        raise PresetRejected("参数组必须有名字。")
    if len(name) > MAX_NAME_CHARS:
        raise PresetRejected(f"参数组名最长 {MAX_NAME_CHARS} 字符。")
    return name


def _check_range(field: str, value: float) -> float:
    lo, hi = _BOUNDS[field]
    if not (lo <= value <= hi):
        raise PresetRejected(
            f"{field} = {value!r}({format_si(value)}) 超出允许范围 "
            f"[{format_si(lo)}, {format_si(hi)}]。**拒绝写入,不会自动夹到边界** —— "
            "被悄悄改小的值会让你以为自己设的是原来那个数。"
        )
    return value


def sanitize_preset(raw: "dict[str, Any]") -> "dict[str, Any]":
    """Validate one custom group. Raises :class:`PresetRejected`; never clamps.

    Values arrive as SI strings (``"3p"``). They are stored that way too, so what
    the operator sees in the UI is what a model would have to write, and a
    round-trip through storage cannot quietly turn a prefix into an exponent.
    """
    name = _clean_name(raw.get("name"))
    lowered = name.lower()
    if lowered in RESERVED_NAMES:
        raise PresetRejected(
            f"'{name}' 是保留名。'approach' 取自仪器档案的进针参数,"
            f"'scan' 取自扫描档位表 —— 自定义组不能遮蔽它们"
            f"(否则你在档位表里改了扫图增益,ApplyZCtrlPreset('scan') 却还用旧值)。"
        )
    try:
        tier_names = {t.lower() for t in scan_policy.tier_names()}
    except Exception:  # noqa: BLE001 — policy unavailable: skip this check only
        tier_names = set()
    if lowered in tier_names:
        raise PresetRejected(
            f"'{name}' 与扫描档位表里的档名重名。档名已经可以直接当参数组名用,"
            f"再建一个同名的自定义组只会让两者永远有一个是死的。"
        )

    out: dict[str, Any] = {"name": name}
    for field in ("p_gain", "i_gain"):
        if field not in raw or raw[field] in (None, ""):
            raise PresetRejected(f"参数组 '{name}' 缺少 {field}。")
        out[field] = _si_field(field, raw[field])

    if raw.get("setpoint_a") not in (None, ""):
        out["setpoint_a"] = _si_field("setpoint_a", raw["setpoint_a"])

    # T = P / I must itself be sane; a pair that is individually in range can
    # still describe a loop that cannot run.
    p = parse_si(out["p_gain"], what="p_gain")
    i = parse_si(out["i_gain"], what="i_gain")
    if i <= 0:
        raise PresetRejected(f"参数组 '{name}' 的 i_gain 必须为正(T = P / I)。")
    t_const = p / i
    lo, hi = _TIME_CONSTANT_BOUNDS
    if not (lo <= t_const <= hi):
        raise PresetRejected(
            f"参数组 '{name}' 的 P/I 组合导出的时间常数 T = P/I = "
            f"{format_si(t_const)}s 不合理(应在 {format_si(lo)}s – {hi:g}s)。"
            f"请检查 p_gain 与 i_gain 的量级。"
        )

    note = str(raw.get("note") or "").strip()
    if note:
        out["note"] = note[:200]
    return out


def _si_field(field: str, raw: object) -> str:
    """Parse-check an SI string and return it UNCHANGED for storage."""
    try:
        value = parse_si(raw, what=field)
    except SIParseError as exc:
        raise PresetRejected(str(exc)) from exc
    _check_range(field, value)
    return str(raw).strip()


def sanitize(raw: object) -> "list[dict[str, Any]]":
    """Validate a whole preset list. One bad row rejects the WHOLE list.

    Same rule as ``scan_policy``: half a table is more dangerous than no table,
    because the half that survived looks authoritative.
    """
    if raw in (None, "", []):
        return []
    if isinstance(raw, dict):
        raw = raw.get("presets", [])
    if not isinstance(raw, (list, tuple)):
        raise PresetRejected("参数组必须是一个列表。")
    if len(raw) > MAX_PRESETS:
        raise PresetRejected(f"参数组最多 {MAX_PRESETS} 组。")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, dict):
            raise PresetRejected("每个参数组必须是一个对象。")
        item = sanitize_preset(row)
        key = item["name"].lower()
        if key in seen:
            raise PresetRejected(f"参数组名重复: '{item['name']}'。")
        seen.add(key)
        out.append(item)
    return out


def set_presets(raw: object) -> None:
    """Install the custom preset list (startup hydration / live-apply)."""
    global _presets
    try:
        cleaned = sanitize(raw)
    except PresetRejected:
        logger.warning("stored zctrl presets rejected; starting with none",
                       exc_info=True)
        cleaned = []
    with _lock:
        _presets = cleaned


def get_presets() -> "list[dict[str, Any]]":
    with _lock:
        return [dict(p) for p in _presets]


def set_persist_sink(fn: "Callable[[list[dict[str, Any]]], None] | None") -> None:
    global _persist_sink
    _persist_sink = fn


def upsert_preset(raw: "dict[str, Any]", *, overwrite: bool = False) -> "dict[str, Any]":
    """Validate and store one custom group. Raises :class:`PresetRejected`."""
    item = sanitize_preset(raw)
    key = item["name"].lower()
    with _lock:
        existing = next((i for i, p in enumerate(_presets)
                         if p["name"].lower() == key), None)
        if existing is not None and not overwrite:
            raise PresetRejected(
                f"参数组 '{item['name']}' 已存在。要替换它请显式传 overwrite=true。"
            )
        if existing is None and len(_presets) >= MAX_PRESETS:
            raise PresetRejected(f"参数组最多 {MAX_PRESETS} 组,请先删除不用的。")
        if existing is None:
            _presets.append(item)
        else:
            _presets[existing] = item
        snapshot = [dict(p) for p in _presets]
    if _persist_sink is not None:
        try:
            _persist_sink(snapshot)
        except Exception:  # noqa: BLE001 — in memory already; report, don't crash
            logger.exception("persisting zctrl presets failed")
    return item


def delete_preset(name: str) -> bool:
    key = str(name or "").strip().lower()
    with _lock:
        before = len(_presets)
        _presets[:] = [p for p in _presets if p["name"].lower() != key]
        removed = len(_presets) != before
        snapshot = [dict(p) for p in _presets]
    if removed and _persist_sink is not None:
        try:
            _persist_sink(snapshot)
        except Exception:  # noqa: BLE001
            logger.exception("persisting zctrl presets failed")
    return removed


# ── Resolution ──────────────────────────────────────────────────────────────

class ResolvedPreset:
    """One preset resolved to numbers, with where each number came from."""

    __slots__ = ("name", "p_gain", "i_gain", "time_constant_s", "setpoint_a",
                 "sources", "notes")

    def __init__(self, name: str, p_gain: float, i_gain: float,
                 setpoint_a: "float | None", sources: "dict[str, str]",
                 notes: "list[str] | None" = None):
        self.name = name
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.time_constant_s = p_gain / i_gain
        self.setpoint_a = setpoint_a
        self.sources = sources
        self.notes = notes or []

    def gain_params(self) -> "dict[str, float]":
        """Exactly what ``SetZCtrlGain`` takes."""
        return {
            "p_gain": self.p_gain,
            "time_constant_s": self.time_constant_s,
            "i_gain": self.i_gain,
        }

    def trace_lines(self) -> "list[str]":
        """Human-readable provenance — every number says where it came from.

        The operator has to be able to answer "why is the loop set to THIS?"
        without reading code, and a number with no stated origin is one nobody
        can check.
        """
        rows = [
            f"p_gain = {format_si(self.p_gain)}m ← {self.sources.get('p_gain', '?')}",
            f"i_gain = {format_si(self.i_gain)}m/s ← {self.sources.get('i_gain', '?')}",
            f"time_constant_s = {format_si(self.time_constant_s)}s ← 由 P/I 导出",
        ]
        if self.setpoint_a is not None:
            rows.append(
                f"setpoint_a = {format_si(self.setpoint_a)}A "
                f"← {self.sources.get('setpoint_a', '?')}"
            )
        else:
            rows.append("setpoint_a = 未配置(保持当前值,不下发)")
        return rows


def available_names() -> "list[str]":
    """Every name ``resolve`` accepts right now, reserved ones first."""
    names = list(RESERVED_NAMES)
    try:
        names.extend(scan_policy.tier_names())
    except Exception:  # noqa: BLE001
        logger.debug("tier names unavailable", exc_info=True)
    names.extend(p["name"] for p in get_presets())
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def _from_profile() -> ResolvedPreset:
    """The approach group, out of the operator's instrument profile."""
    p = _iprof.get_config("approach_p_gain_m", None)
    i = _iprof.get_config("approach_i_gain_m_per_s", None)
    sp = _iprof.get_config("approach_setpoint_a", None)
    missing = [k for k, v in (("P 增益", p), ("I 增益", i)) if v in (None, "")]
    if missing:
        # Fail closed and say where to fix it. Silently doing nothing on an
        # explicit "apply the approach parameters" is a false success.
        raise PresetRejected(
            f"进针参数组尚未配置({', '.join(missing)} 为空)。"
            "请在「设置 → 仪器档案 → 进针参数」里填写 —— 这组数只由用户输入,"
            "不经模型。"
        )
    return ResolvedPreset(
        name=PRESET_APPROACH,
        p_gain=float(p), i_gain=float(i),
        setpoint_a=(float(sp) if sp not in (None, "") else None),
        sources={
            "p_gain": "仪器档案 approach_p_gain_m",
            "i_gain": "仪器档案 approach_i_gain_m_per_s",
            "setpoint_a": "仪器档案 approach_setpoint_a",
        },
    )


def _from_tier(tier: "dict[str, Any]", label: str) -> ResolvedPreset:
    p = tier.get("p_gain")
    t = tier.get("time_constant_s")
    if p in (None, "") or t in (None, ""):
        raise PresetRejected(
            f"扫描档位 '{tier.get('name', label)}' 没有配置 P 增益/时间常数。"
            "请在「设置 → 扫描档位表」里补上,或改用别的参数组。"
        )
    p = float(p)
    t = float(t)
    if t <= 0:
        raise PresetRejected(
            f"扫描档位 '{tier.get('name', label)}' 的时间常数为 0,无法导出 I = P/T。"
        )
    src = (SOURCE_TIER_OPERATOR if tier.get("source") == "operator"
           else SOURCE_TIER_FACTORY)
    origin = f"扫描档位表 '{tier.get('name', label)}'({src})"
    sp = tier.get("setpoint_a")
    return ResolvedPreset(
        name=str(tier.get("name", label)),
        p_gain=p, i_gain=p / t,
        setpoint_a=(float(sp) if sp not in (None, "") else None),
        sources={"p_gain": origin, "i_gain": f"{origin},I = P/T 导出",
                 "setpoint_a": origin},
    )


def _current_frame_size_m(context) -> "float | None":
    """The frame width the scanner is set to, or None if it cannot be read."""
    if context is None:
        return None
    try:
        rec = context.safe_call("Scan_FrameGet")
        if getattr(rec, "error", None):
            return None
        parsed = getattr(rec, "return_value", None)
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 3:
                return float(vals[2])   # (center_x, center_y, width, height, angle)
    except Exception:  # noqa: BLE001
        logger.debug("frame size unreadable", exc_info=True)
    return None


def resolve(name: str, *, context=None) -> ResolvedPreset:
    """Resolve a preset name to numbers. Raises :class:`PresetRejected`.

    ``context`` is only needed by the ``scan`` alias, which asks the scanner what
    size frame it is set to so it can pick the same tier ``ScanAt`` would.
    """
    wanted = str(name or "").strip()
    if not wanted:
        raise PresetRejected(_unknown_message(""))
    lowered = wanted.lower()

    if lowered == PRESET_APPROACH:
        return _from_profile()

    if lowered == PRESET_SCAN:
        size = _current_frame_size_m(context)
        if size is None:
            raise PresetRejected(
                "读不到当前扫描帧的尺寸,无法确定该用哪一档扫图参数。"
                "请直接指定档名(见 ListZCtrlPresets),或先设置扫描范围。"
            )
        tier = scan_policy.get_tier_for_size(size)
        if not tier:
            raise PresetRejected(f"扫描档位表里没有匹配 {format_si(size)}m 的档。")
        res = _from_tier(tier, PRESET_SCAN)
        res.notes.append(
            f"'scan' 按当前帧宽 {format_si(size)}m 选中档位 '{res.name}'"
        )
        return res

    try:
        tier = scan_policy.get_tier_by_name(wanted)
    except Exception:  # noqa: BLE001
        tier = None
    if tier:
        return _from_tier(tier, wanted)

    for item in get_presets():
        if item["name"].lower() == lowered:
            p = parse_si(item["p_gain"], what="p_gain")
            i = parse_si(item["i_gain"], what="i_gain")
            sp = (parse_si(item["setpoint_a"], what="setpoint_a")
                  if item.get("setpoint_a") else None)
            origin = f"自定义参数组 '{item['name']}'"
            return ResolvedPreset(
                name=item["name"], p_gain=p, i_gain=i, setpoint_a=sp,
                sources={"p_gain": origin, "i_gain": origin,
                         "setpoint_a": origin},
                notes=([item["note"]] if item.get("note") else []),
            )

    raise PresetRejected(_unknown_message(wanted))


def _unknown_message(wanted: str) -> str:
    names = available_names()
    return (
        f"没有名为 '{wanted}' 的参数组。当前可用: "
        + ", ".join(f"'{n}'" for n in names)
        + "。(用 ListZCtrlPresets 查看每一组的具体数值。)"
    )


__all__ = [
    "MAX_PRESETS", "PRESET_APPROACH", "PRESET_SCAN", "PresetRejected",
    "RESERVED_NAMES", "ResolvedPreset",
    "available_names", "delete_preset", "get_presets", "resolve",
    "sanitize", "sanitize_preset", "set_persist_sink", "set_presets",
    "upsert_preset",
]
