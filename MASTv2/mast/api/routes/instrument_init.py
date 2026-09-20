"""新仪器初始化 —— 「这台机器还差哪些数」的那一页。

设计文档：``docs/v2/design/new_instrument_initialization.md``
目录与判据：:mod:`mast.core.instrument_init`

写入通道
========
本模块**不新建任何存储**。写入一律转交给既有的两条通道：

  * ``POST /api/settings``（``settings_admin_write.write_settings``）——
    instrument_profile / coarse_drive / current_monitor / scan_policy /
    hardware_modules。校验、PIN 门、live-apply 全在那边，这里一行都不复制。
  * ``POST /api/admin/overrides/{category}``（``admin.write_override``）——
    安全包络。

本模块在那之上加的只有一件事：**回读比对**。

为什么回读比对必须在 Python 里做
================================
同一个模型既可能把 `3e-12` 发成 `3`（错 10¹² 倍），又可能在读回
`3.0` 之后写下「= 3e-12 … 数值通道工作正常」，两次错误方向一致就会互相掩盖
（`docs/v2/fixes/2026-08-03-tool-call-number-corruption.md` 第六点五）。

所以这一页的整条链路是：

    人在输入框里打字 → HTTP JSON → Python 校验 → 既有 store 写入
                     → Python 回读 → Python 比对 → 结果显示给人

**零个 LLM。** agent 也够不到这些端点：它的工具面只有 Nanonis 技能，没有 HTTP
工具（同一条论证已经写在 ``settings_admin_write.write_settings`` 里）。

比对本身也不是「相等」——``instrument_profile.sanitize`` 会**夹紧**越界值、
``coarse_drive.sanitize`` 会**丢弃**越界值，两者都不报错。回读比对把这两种静默
变成三个明确的裁决：match / clamped / dropped。
"""

from __future__ import annotations

import logging
import math
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_admin import OverrideWriteRequest
from mast.api.schemas_instrument_init import (
    AcknowledgeRequest,
    CompleteRequest,
    DerivedSuggestion,
    ImportBundleRequest,
    InitApplyRequest,
    InitApplyResponse,
    InitBundleResponse,
    InitGroupModel,
    InitImportResponse,
    InitItemModel,
    InitProbeResponse,
    InitRecordResponse,
    InputSpec,
    InstrumentInitResponse,
    PreampCheck,
    ProbeField,
    SeverityCount,
    ValueVerdict,
)
from mast.api.schemas_settings_admin_write import SettingsWriteRequest
from mast.core import instrument_init as ii

logger = logging.getLogger(__name__)

router = APIRouter(tags=["instrument-init"])

#: 走 ``POST /api/settings`` 的真源 → 它在 SettingsWriteRequest 上的字段名。
#: ``scan_policy`` 与 ``hardware_modules`` 的载荷形状与其它几个不同（前者是
#: ``{"tiers": [...]}``，后者是 ``{id: bool}``），由 :func:`_settings_payload` 处理。
_SETTINGS_STORES: frozenset[str] = frozenset({
    "instrument_profile", "coarse_drive", "current_monitor",
    "scan_policy", "hardware_modules",
})
#: 走覆写通道的真源。
_OVERRIDE_STORES: dict[str, str] = {"safety_limits": "safety_limits"}

#: 回读比对的相对容差。存储层做的是 float 往返，不是硬件写入，所以可以比
#: ``skills.verify.values_match`` 的 1e-3 严一些；1e-9 只吸收 JSON 往返的
#: 双精度误差，仍然远小于任何一次「夹紧」会造成的差别。
_READBACK_REL_TOL = 1e-9


# ── 取值：每个真源的「用户实际存过什么」 ──────────────────────────────────
#
# 刻意取**存过的值**而不是**生效值**。有出厂默认的项，「没填」和「填的正好等于
# 默认」在生效值上完全一样，而这一页的全部意义就是区分这两者。
def _settings_store(ctx: Any) -> Any:
    return getattr(ctx, "settings_store", None)


def _override_registry(ctx: Any) -> Any:
    reg = getattr(ctx, "override_registry", None)
    if reg is not None:
        return reg
    try:
        from mast.admin.override_store import ConfigOverrideRegistry
        return ConfigOverrideRegistry.get()
    except Exception as exc:  # noqa: BLE001
        logger.debug("override registry unavailable: %s", exc)
        return None


def _stored_dict(store: Any, key: str) -> dict[str, Any]:
    if store is None:
        return {}
    try:
        val = store.get(key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("settings read %s failed: %s", key, exc)
        return {}
    return dict(val) if isinstance(val, dict) else {}


def _stored_values(ctx: Any) -> dict[str, Any]:
    """``item.id`` → 用户**存过**的值（没存过 = 缺席，不是出厂值）。"""
    store = _settings_store(ctx)
    out: dict[str, Any] = {}

    prof = _stored_dict(store, "instrument_profile")
    drive = _stored_dict(store, "coarse_drive")
    monitor = _stored_dict(store, "current_monitor")
    modules = _stored_dict(store, "hardware_modules")

    for it in ii.CATALOG:
        if it.store == "instrument_profile":
            out[it.id] = prof.get(it.key)
        elif it.store == "coarse_drive":
            out[it.id] = drive.get(it.key)
        elif it.store == "current_monitor":
            out[it.id] = monitor.get(it.key)
        elif it.store == "hardware_modules":
            # 「哪些模块装了」的答案是**这张表被填过**，而不是某个模块开着：
            # 全关是一个完全合法的答案（基础控制器什么可选模块都没有）。
            out[it.id] = dict(modules) if modules else None
        elif it.store == "safety_limits":
            out[it.id] = None          # 下面用覆写文件填
        elif it.store == "scan_policy":
            out[it.id] = None          # 下面用 get_stored_policy 填

    reg = _override_registry(ctx)
    raw_limits: dict[str, Any] = {}
    if reg is not None:
        try:
            raw_limits = reg.get_raw("safety_limits.json") or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("safety_limits override read failed: %s", exc)
    for it in ii.CATALOG:
        if it.store == "safety_limits":
            out[it.id] = raw_limits.get(it.key)

    try:
        from mast.core import scan_policy as _sp
        tiers = _sp.get_stored_policy()
    except Exception as exc:  # noqa: BLE001
        logger.debug("scan_policy read failed: %s", exc)
        tiers = []
    for it in ii.CATALOG:
        if it.store == "scan_policy":
            out[it.id] = tiers or None

    return out


def _factory_values() -> dict[str, Any]:
    """``item.id`` → 出厂默认（没有默认 → ``None``）。全部从真源现取，不抄。"""
    out: dict[str, Any] = {}
    try:
        from mast.core import instrument_profile as _ip
    except Exception:  # noqa: BLE001 - pragma: no cover
        _ip = None                                          # type: ignore[assignment]
    try:
        from mast.config import SafetyLimits
        limits = SafetyLimits()
    except Exception:  # noqa: BLE001 - pragma: no cover
        limits = None
    try:
        from mast.monitoring.thresholds import MonitorThresholds
        monitor = MonitorThresholds()
    except Exception:  # noqa: BLE001 - pragma: no cover
        monitor = None

    for it in ii.CATALOG:
        if it.store == "instrument_profile" and _ip is not None:
            out[it.id] = _ip.spec_default(it.key)
        elif it.store == "safety_limits" and limits is not None:
            out[it.id] = getattr(limits, it.key, None)
        elif it.store == "current_monitor" and monitor is not None:
            out[it.id] = getattr(monitor, it.key, None)
        else:
            out[it.id] = None
    return out


def _input_specs() -> dict[str, dict[str, Any]]:
    """``item.id`` → 输入框规格（类型 / 区间 / 枚举）。**全部从真源现取。**

    前端据此渲染并做同样的范围校验。在 TSX 里再抄一份区间就会漂 —— 那正是
    ``test_instrument_profile_frontend_parity`` 钉住的那类事故。
    """
    out: dict[str, dict[str, Any]] = {}

    try:
        from mast.core.instrument_profile import field_specs
        for spec in field_specs():
            kind = spec["type"]
            out[f"instrument_profile.{spec['key']}"] = {
                "type": kind if kind in ("int", "choice", "str") else "float",
                "min": spec.get("min") if kind != "str" else None,
                "max": spec.get("max") if kind != "str" else None,
                "choices": spec.get("choices") or [],
            }
    except Exception as exc:  # noqa: BLE001 - pragma: no cover
        logger.debug("instrument_profile field specs unavailable: %s", exc)

    try:
        from mast.core.coarse_drive import (
            ABSOLUTE_MAX_AMPLITUDE_V, ABSOLUTE_MAX_FREQUENCY_HZ,
        )
        out["coarse_drive.max_amplitude_v"] = {
            "type": "float", "min": 0.0, "max": ABSOLUTE_MAX_AMPLITUDE_V,
            "choices": []}
        out["coarse_drive.expected_frequency_hz"] = {
            "type": "float", "min": 0.0, "max": ABSOLUTE_MAX_FREQUENCY_HZ,
            "choices": []}
    except Exception as exc:  # noqa: BLE001 - pragma: no cover
        logger.debug("coarse_drive bounds unavailable: %s", exc)

    try:
        from mast.monitoring.thresholds import knob_catalog
        for knob in knob_catalog():
            key = str(knob.get("key") or "")
            item_id = f"current_monitor.{key}"
            if item_id in {it.id for it in ii.CATALOG}:
                out[item_id] = {
                    "type": "float", "min": knob.get("min"), "max": knob.get("max"),
                    "choices": []}
    except Exception as exc:  # noqa: BLE001 - pragma: no cover
        logger.debug("monitor knob catalog unavailable: %s", exc)

    # 安全包络是纯浮点，区间由物理决定而不是由一张表决定 —— 不给 min/max
    # （给了就等于在这里发明第二条包络）。
    for it in ii.CATALOG:
        out.setdefault(it.id, {"type": "float", "min": None, "max": None,
                               "choices": []})
    out["scan_policy.tiers"] = {"type": "table", "min": None, "max": None,
                                "choices": []}
    out["hardware_modules.modules"] = {"type": "modules", "min": None, "max": None,
                                       "choices": []}
    return out


def _record(ctx: Any) -> dict[str, Any]:
    return ii.sanitize_record(_stored_dict(_settings_store(ctx), ii.SETTINGS_KEY))


def _save_record(ctx: Any, record: dict[str, Any]) -> bool:
    store = _settings_store(ctx)
    if store is None:
        return False
    try:
        store.update(**{ii.SETTINGS_KEY: record})
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("instrument_init record write failed: %s", exc)
        return False


# ── GET /api/instrument-init ────────────────────────────────────────────────
@router.get("/instrument-init", response_model=InstrumentInitResponse)
def get_instrument_init(request: Request) -> InstrumentInitResponse:
    """整份清单：每一项是什么、现在的值、还差什么、该不该弹。

    ``needs_setup`` 是**权威判据**且**纯内容判据** —— 从值本身算，不看任何标记。
    刻意不用「文件在不在」：覆写目录与设置文件都住在安装目录里
    （KNOWN_ISSUES §3.2），靠「安装包里恰好没有同名文件」活着，所以文件的存在与否
    根本不能回答「这台机器配过没有」。

    永不 500：一份读不出来的清单会让用户以为「没什么要填的」，而那正是最坏的
    答案。任何一步失败都降级成 ``degraded=True`` + 尽量多的已知项。
    """
    ctx = getattr(request.app.state, "ctx", None)
    degraded = _settings_store(ctx) is None

    try:
        values = _stored_values(ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("instrument-init 取值失败: %s", exc)
        values, degraded = {}, True
    factory = _factory_values()
    record = _record(ctx)
    ack = set(record.get("acknowledged") or ())

    statuses = ii.evaluate(values, ack)
    summary = ii.summarise(statuses)
    by_id = {s.id: s for s in statuses}

    inputs = _input_specs()
    items = [
        InitItemModel(
            **{k: v for k, v in raw.items()},
            value=values.get(raw["id"]),
            factory=factory.get(raw["id"]),
            status=by_id[raw["id"]].status,
            na_reason=by_id[raw["id"]].na_reason,
            complete=by_id[raw["id"]].complete,
            input=InputSpec(**inputs.get(raw["id"], {})),
        )
        for raw in ii.catalog_payload()
    ]

    # 前放交叉核对 —— 两个数是同一件事的两种说法，对不上就是有一个填错了。
    check = ii.preamp_consistency(
        values.get("instrument_profile.preamp_gain_v_per_a"),
        values.get("instrument_profile.preamp_full_scale_a"),
    )

    # 由满量程导出的两条下游线（**建议值**，不自动写入）。
    derived: list[DerivedSuggestion] = []
    for target, spec in ii.derived_from_preamp(
            values.get("instrument_profile.preamp_full_scale_a")).items():
        cur = values.get(target)
        if cur is None:
            cur = factory.get(target)
        looser = False
        try:
            looser = cur is not None and float(cur) > float(spec["value"])
        except (TypeError, ValueError):
            looser = False
        derived.append(DerivedSuggestion(
            target=target, value=float(spec["value"]),
            current=float(cur) if isinstance(cur, (int, float)) else None,
            why=str(spec["why"]), current_is_looser=looser,
        ))

    safety_source, restart_pending = "", None
    try:
        from mast.api.safety_view import in_force_safety_limits, pending_restart
        _limits, safety_source = in_force_safety_limits(ctx)
        restart_pending = pending_restart(ctx)
    except Exception as exc:  # noqa: BLE001
        logger.debug("safety provenance unavailable: %s", exc)

    fp_changed = ii.fingerprint_changed(record, _cached_fingerprint(ctx))
    never_completed = not record.get("completed_at")

    return InstrumentInitResponse(
        groups=[InitGroupModel(**g) for g in ii.groups_payload()],
        items=items,
        counts={k: SeverityCount(**v) for k, v in summary["counts"].items()},
        outstanding_required=summary["outstanding_required"],
        needs_setup=bool(summary["needs_setup"]),
        fingerprint_changed=fp_changed,
        should_prompt=bool(summary["needs_setup"]) or fp_changed or never_completed,
        completed_at=record.get("completed_at"),
        rig_fingerprint=record.get("rig_fingerprint"),
        rig_label=str(record.get("rig_label") or ""),
        acknowledged=sorted(ack),
        preamp_check=PreampCheck(**check),
        derived=derived,
        safety_source=safety_source,
        safety_restart_pending=restart_pending,
        degraded=degraded,
    )


# 指纹只在 probe 时真的读硬件；GET 用最近一次 probe 的缓存，免得每次刷新清单都
# 打四个 TCP 往返。缓存活在进程里（换机器必然重启），拿不到就是 None ——
# 而 None 一律不算「变了」。
_FP_CACHE_ATTR = "_instrument_init_fingerprint"


def _cached_fingerprint(ctx: Any) -> "str | None":
    return getattr(ctx, _FP_CACHE_ATTR, None) if ctx is not None else None


# ── POST /api/instrument-init/apply ─────────────────────────────────────────
#: 这些 store 是**整份替换**的 dict —— 只送变动的键会把其余的全部抹掉。
#: ``scan_policy`` 不在里面：那一项的值**就是整张表**，替换才是对的。
_WHOLE_REPLACE_DICT_STORES: frozenset[str] = frozenset({
    "instrument_profile", "coarse_drive", "current_monitor", "hardware_modules",
})


def _settings_payload(ctx: Any, store_name: str, values: dict[str, Any]) -> Any:
    """把页面送来的 ``{key: value}`` 变成对应 store 的载荷形状。

    ⚠️ **整份替换的 store 必须先合并。** ``SettingsStore.update`` 直接
    ``self._data[k] = v`` —— 送 ``{"preamp_gain_v_per_a": 1e9}`` 会把
    ``instrument_profile`` 里其余 40 个键连同运行时学出来的 dI/dV 标定一起删掉，
    而且不报错、不提示，下一次读到的就是 None。

    这正是 ``test_instrument_profile_frontend_parity`` 钉住的那个形状（设置表单
    忘了回显哪个键，编辑别的字段时就把它删掉；qPlus 振幅基线就是这么丢的）。
    这一页按**组**保存，如果不合并，填完信号链再填进针就会把信号链抹掉。

    ``None`` 语义是「清掉这个键」（与表单里清空输入框一致），所以是 pop 而不是
    写一个 None 进去。
    """
    if store_name == "scan_policy":
        # 页面送 {"tiers": [...]}；也接受裸列表（测试里写起来省事）。
        tiers = values.get("tiers", values)
        return {"tiers": tiers} if not isinstance(tiers, dict) else tiers

    if store_name == "hardware_modules":
        mods = values.get("modules", values)
        delta: dict[str, Any] = {str(k): bool(v) for k, v in dict(mods).items()}
    else:
        delta = dict(values)

    if store_name not in _WHOLE_REPLACE_DICT_STORES:
        return delta

    merged = _stored_dict(_settings_store(ctx), store_name)
    for key, val in delta.items():
        if val is None or val == "":
            merged.pop(key, None)
        else:
            merged[key] = val
    return merged


def _readback(ctx: Any, store_name: str) -> dict[str, Any]:
    """写完之后，把**真正在生效的那一份**重新读一遍。

    ⚠️ 刻意读 live-read holder 而**不是** ``ui_settings.json``。
    ``SettingsStore.update`` 原样存下调用方给的 dict；夹紧 / 丢弃发生在
    ``set_profile`` / ``set_declaration`` 那一步（live-apply）。所以一个越界值
    在设置文件里仍是原值，而技能层读到的是被夹过的那个 —— 两者可以差好几个
    数量级，并且两边都不报错。（重启后 hydrate 会再 sanitize 一次，生效值不变。）

    对用户有意义的是**系统接下来会用哪个数**，所以比对必须对着 holder。

    安全包络是例外：覆写要重启才生效（KNOWN_ISSUES §1.1），此刻在跑的仍是旧值，
    所以那里比对的对象是刚写下去的覆写文件本身，并另外回报 ``restart_required``。
    """
    if store_name in _OVERRIDE_STORES:
        reg = _override_registry(ctx)
        if reg is None:
            return {}
        try:
            return dict(reg.get_raw("safety_limits.json") or {})
        except Exception as exc:  # noqa: BLE001
            logger.debug("safety_limits readback failed: %s", exc)
            return {}
    try:
        if store_name == "instrument_profile":
            from mast.core import instrument_profile as _ip
            return _ip.get_profile()
        if store_name == "coarse_drive":
            from mast.core import coarse_drive as _cd
            return _cd.get_declaration()
        if store_name == "current_monitor":
            from mast.monitoring.thresholds import get_monitor_thresholds
            return dict(get_monitor_thresholds().to_mapping())
        if store_name == "scan_policy":
            from mast.core import scan_policy as _sp
            return {"tiers": _sp.get_stored_policy()}
        if store_name == "hardware_modules":
            from mast.skills.hardware_modules import enabled_ids
            return {mid: True for mid in enabled_ids()}
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s readback failed: %s", store_name, exc)
    return _stored_dict(_settings_store(ctx), store_name)


def _compare(requested: Any, stored: Any) -> tuple[str, str]:
    """一个值的裁决：match / clamped / dropped / unverifiable。

    **不是「相等」判断。** ``instrument_profile.sanitize`` 会把越界值**夹紧**、
    ``coarse_drive.sanitize`` 会把越界值**丢弃**，两者都不报错也不返回理由。
    这个函数就是把那两种静默变成用户看得见的三句话。
    """
    # 清空一个字段是**有意的**（「这台机器没有这个量」）—— 清完读不到，正是预期。
    # 不区分这一种，用户每次清字段都会看到一条红色的「没被收下」。
    if requested is None or requested == "":
        if stored is None or stored == "":
            return "match", "已清除（回到「未设置」）。"
        return ("clamped",
                f"要求清除，但读回来仍是 {stored!r} —— 这个键没有被清掉。")
    if stored is None:
        return ("dropped",
                "这个值**没有被收下** —— 通常是超出了允许范围，或者不是这个字段"
                "接受的类型。它现在是「未设置」，不是你填的那个数。")
    if isinstance(requested, (int, float)) and isinstance(stored, (int, float)) \
            and not isinstance(requested, bool) and not isinstance(stored, bool):
        r, s = float(requested), float(stored)
        if not (math.isfinite(r) and math.isfinite(s)):
            return "dropped", "非有限数值，未被收下。"
        if math.isclose(r, s, rel_tol=_READBACK_REL_TOL, abs_tol=0.0) or (r == s):
            return "match", ""
        ratio = (s / r) if r else float("inf")
        return ("clamped",
                f"存下来的是 {s:.6g}，不是你填的 {r:.6g}"
                + (f"（差 {ratio:.4g} 倍）" if math.isfinite(ratio) else "")
                + " —— 该值超出了允许区间，被夹到了边界上。"
                  "**请按存下来的这个数理解系统的行为**，或者回去改一个区间内的值。")
    if str(requested) == str(stored):
        return "match", ""
    return ("clamped",
            f"存下来的是 {stored!r}，不是你填的 {requested!r} —— "
            "请核对是不是写法/取值不被接受。")


@router.post("/instrument-init/apply", response_model=InitApplyResponse)
def apply_instrument_init(request: Request, body: InitApplyRequest) -> InitApplyResponse:
    """写一批值，然后**在 Python 里**回读比对，把结果如实报回来。

    写入本身完全转交既有通道（``write_settings`` / ``write_override``）：校验、
    PIN、live-apply 一行都不复制。本端点只加回读比对那一层。
    """
    ctx = getattr(request.app.state, "ctx", None)
    store_name = str(body.store or "").strip()
    values = dict(body.values or {})

    if store_name not in _SETTINGS_STORES and store_name not in _OVERRIDE_STORES:
        return InitApplyResponse(
            ok=False, store=store_name,
            message=(f"不认识的存储 {store_name!r}。这一页只写既有真源："
                     f"{', '.join(sorted(_SETTINGS_STORES | set(_OVERRIDE_STORES)))}。"))
    if not values:
        return InitApplyResponse(ok=False, store=store_name, message="没有要写的值。")

    restart_required = False
    try:
        if store_name in _OVERRIDE_STORES:
            # 安全包络：覆写文件是**整份替换**，所以要先读旧的再合并，
            # 否则只改一个 setpoint_max_a 会把之前写好的 XY / Z 包络一起抹掉。
            reg = _override_registry(ctx)
            merged = dict((reg.get_raw("safety_limits.json") or {}) if reg else {})
            merged.update(values)
            from mast.api.routes.admin import write_override
            res = write_override(request, _OVERRIDE_STORES[store_name],
                                 OverrideWriteRequest(data=merged))
            if not getattr(res, "ok", False):
                return InitApplyResponse(ok=False, store=store_name, degraded=True,
                                         message="覆写写入失败（没有可用的覆写目录）。")
            # KNOWN_ISSUES §1.1：热重载钩子零订阅者。**说实话** ——
            #「一个只对了一半的『已生效』，比诚实的『要重启』更危险。」
            restart_required = bool(getattr(res, "restart_required", False)
                                    or not getattr(res, "reloaded", False))
        else:
            from mast.api.routes.settings_admin_write import write_settings
            payload = {store_name: _settings_payload(ctx, store_name, values)}
            if body.admin_pin:
                payload["admin_pin"] = body.admin_pin
            res = write_settings(request, SettingsWriteRequest(**payload))
            if getattr(res, "pin_required", False):
                return InitApplyResponse(
                    ok=False, store=store_name, pin_required=True,
                    pin_reason=str(getattr(res, "pin_reason", "")),
                    message=str(getattr(res, "rebuild_note", "")
                                or "这一项需要管理员 PIN。"))
            rejected = dict(getattr(res, "rejected", {}) or {})
            if rejected:
                return InitApplyResponse(ok=False, store=store_name, rejected=rejected,
                                         message="；".join(rejected.values()))
            if not getattr(res, "ok", False):
                return InitApplyResponse(ok=False, store=store_name, degraded=True,
                                         message="设置写入失败（没有可用的设置存储）。")
    except Exception as exc:  # noqa: BLE001 — 写入路径永不 500
        logger.warning("instrument-init apply(%s) failed: %s", store_name, exc)
        return InitApplyResponse(ok=False, store=store_name, degraded=True,
                                 message=f"{type(exc).__name__}: {exc}")

    # ── 回读比对（这一层就是本端点存在的理由）───────────────────────────────
    stored = _readback(ctx, store_name)
    verdicts: list[ValueVerdict] = []
    for key, requested in values.items():
        if store_name in ("scan_policy", "hardware_modules"):
            # 这两个是整表写入，逐键比对没有意义；比「表被收下了没有」。
            got = stored.get("tiers") if store_name == "scan_policy" else stored
            ok = bool(got)
            verdicts.append(ValueVerdict(
                key=key, requested="(整表)", stored="(整表)" if ok else None,
                verdict="match" if ok else "dropped",
                note="" if ok else "整张表没有被收下。"))
            break
        verdict, note = _compare(requested, stored.get(key))
        verdicts.append(ValueVerdict(key=key, requested=requested,
                                     stored=stored.get(key),
                                     verdict=verdict, note=note))

    mismatched = any(v.verdict != "match" for v in verdicts)
    return InitApplyResponse(
        ok=not mismatched, store=store_name, verdicts=verdicts,
        mismatched=mismatched, restart_required=restart_required,
        message=("回读与写入不一致 —— 请照上面每一行的说明核对。"
                 if mismatched else
                 ("已保存。**安全包络覆写要重启才生效**（本进程仍在用旧值）。"
                  if restart_required else "已保存并回读确认。")),
    )


# ── 核对 / 完成 / 重新打开 ──────────────────────────────────────────────────
@router.post("/instrument-init/acknowledge", response_model=InitRecordResponse)
def acknowledge_items(request: Request, body: AcknowledgeRequest) -> InitRecordResponse:
    """把一批「出厂值就是对的」标成已核对。

    为什么需要这个动作：有出厂默认的项，「没填」和「填的正好等于默认」在存储层
    是同一件事，程序分不出来。所以完成与否由**一次明确的核对**决定，而不是猜。
    """
    ctx = getattr(request.app.state, "ctx", None)
    rec = _record(ctx)
    ids = {str(i) for i in (body.item_ids or [])} & set(ii.ITEM_IDS)
    if body.undo:
        rec = ii.sanitize_record(rec)
        rec["acknowledged"] = sorted(set(rec.get("acknowledged") or ()) - ids)
    else:
        rec = ii.acknowledge(rec, ids)
    ok = _save_record(ctx, rec)
    return InitRecordResponse(
        ok=ok, acknowledged=list(rec.get("acknowledged") or []),
        completed_at=rec.get("completed_at"),
        rig_fingerprint=rec.get("rig_fingerprint"),
        rig_label=str(rec.get("rig_label") or ""),
        degraded=not ok,
        message="" if ok else "没有可用的设置存储，核对状态未能保存。")


@router.post("/instrument-init/complete", response_model=InitRecordResponse)
def complete_init(request: Request, body: CompleteRequest) -> InitRecordResponse:
    """盖「这台机器配过了」的戳 + 记下当前硬件指纹。

    **戳只抑制骚扰，不豁免必填。** 必填项的判据永远从值本身算 —— 一个盖过戳的
    机器如果 profile 被抹了，横幅照样回来。
    """
    ctx = getattr(request.app.state, "ctx", None)
    version = ""
    try:
        from mast.api.version import get_version
        version = get_version()
    except Exception:  # noqa: BLE001
        pass
    rec = ii.stamp_completion(
        _record(ctx), fingerprint=_cached_fingerprint(ctx),
        by=body.completed_by or None, app_version=version or None)
    if body.rig_label:
        rec["rig_label"] = str(body.rig_label)[:200]
    ok = _save_record(ctx, rec)
    return InitRecordResponse(
        ok=ok, acknowledged=list(rec.get("acknowledged") or []),
        completed_at=rec.get("completed_at"),
        rig_fingerprint=rec.get("rig_fingerprint"),
        rig_label=str(rec.get("rig_label") or ""),
        degraded=not ok,
        message="" if ok else "没有可用的设置存储，完成记录未能保存。")


@router.post("/instrument-init/reopen", response_model=InitRecordResponse)
def reopen_init(request: Request) -> InitRecordResponse:
    """手动让它重新弹（设置里的入口）。清掉完成戳，**不动任何数值**。"""
    ctx = getattr(request.app.state, "ctx", None)
    rec = ii.sanitize_record(_record(ctx))
    rec.pop("completed_at", None)
    ok = _save_record(ctx, rec)
    return InitRecordResponse(
        ok=ok, acknowledged=list(rec.get("acknowledged") or []),
        completed_at=None, rig_fingerprint=rec.get("rig_fingerprint"),
        rig_label=str(rec.get("rig_label") or ""),
        degraded=not ok,
        message="" if ok else "没有可用的设置存储。")


# ── 导出 / 导入 ─────────────────────────────────────────────────────────────
@router.get("/instrument-init/export", response_model=InitBundleResponse)
def export_bundle(request: Request) -> InitBundleResponse:
    """打一个可带走的配置包（同型号第二台的起点）。

    **剥掉学习量**：``didv_at_contact_v`` 是那根针在那台机器上 EWMA 学出来的、
    ``tilt_cal_*`` 是那个样品托架的响应矩阵、qPlus 实测共振是那一支音叉的。
    搬到另一台机器上，它们不是「起点」，是一个自信的错值。
    """
    ctx = getattr(request.app.state, "ctx", None)
    store = _settings_store(ctx)
    if store is None:
        return InitBundleResponse(ok=False, degraded=True)

    stripped: list[str] = []
    prof = _stored_dict(store, "instrument_profile")
    try:
        from mast.core.instrument_profile import _CALIB_KEYS  # noqa: PLC2701
        learned = set(_CALIB_KEYS)
    except Exception:  # noqa: BLE001 - pragma: no cover
        learned = set()
    # qplus 振幅基线注册在 _CONFIG_SPEC 里、语义却是运行时标定，所以 _CALIB_KEYS
    # 漏掉它 —— 同一个已知裂缝在 clear_tip_bound_state 的 docstring 里写着。
    learned.add("qplus_amplitude_baseline")
    clean_prof = {}
    for k, v in prof.items():
        if k in learned:
            stripped.append(k)
        else:
            clean_prof[k] = v

    stores: dict[str, Any] = {"instrument_profile": clean_prof}
    reg = _override_registry(ctx)
    if reg is not None:
        try:
            stores["safety_limits"] = dict(reg.get_raw("safety_limits.json") or {})
        except Exception as exc:  # noqa: BLE001
            logger.debug("safety_limits export failed: %s", exc)
    try:
        from mast.core import scan_policy as _sp
        tiers = _sp.get_stored_policy()
        if tiers:
            stores["scan_policy"] = {"tiers": tiers}
    except Exception as exc:  # noqa: BLE001
        logger.debug("scan_policy export failed: %s", exc)
    for key in ("current_monitor", "coarse_drive"):
        val = _stored_dict(store, key)
        if val:
            stores[key] = val

    record = _record(ctx)
    version = ""
    try:
        from mast.api.version import get_version
        version = get_version()
    except Exception:  # noqa: BLE001
        pass
    bundle = ii.build_bundle(stores, app_version=version,
                             rig_label=str(record.get("rig_label") or ""))
    return InitBundleResponse(ok=True, bundle=bundle, stripped=sorted(set(stripped)))


@router.post("/instrument-init/import", response_model=InitImportResponse)
def import_bundle(request: Request, body: ImportBundleRequest) -> InitImportResponse:
    """解析一个导入包。**只解析，不落盘。**

    导入是「把数字填进表单」，不是「写进硬件」—— 同型号不等于同一台：前放可能
    不一样，压电标定一定不一样。返回的值进入待复核状态，用户逐组按
    ``/apply`` 确认才真正写下去。
    """
    try:
        parsed = ii.parse_bundle(body.bundle)
    except ii.BundleRejected as exc:
        return InitImportResponse(ok=False, rejected=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("bundle import failed: %s", exc)
        return InitImportResponse(ok=False, rejected=f"{type(exc).__name__}: {exc}")
    return InitImportResponse(
        ok=True, stores=parsed["stores"], rig_label=parsed["rig_label"],
        app_version=parsed["app_version"], exported_at=parsed.get("exported_at"),
        needs_resign=parsed["needs_resign"],
    )


# ── 从仪器读一次，对账 ──────────────────────────────────────────────────────
def _execution_context(ctx: Any) -> Any:
    """与 signals 路由同款：拿得到就拿，拿不到就降级（永不抛）。"""
    pool = getattr(ctx, "connection_pool", None)
    state = getattr(ctx, "instrument_state", None)
    registry = getattr(ctx, "skill_registry", None)
    if pool is None or registry is None:
        return None
    try:
        from mast.core.execution_context import ExecutionContext
        app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
        return ExecutionContext(pool=pool, state=state, registry=registry,
                                abort_event=getattr(app, "_orch_abort", None),
                                owner="新仪器初始化对账")
    except Exception as exc:  # noqa: BLE001
        logger.debug("instrument-init: ExecutionContext build failed: %s", exc)
        return None


def _first_number(raw: Any) -> "float | None":
    """从 Nanonis 的回包里取第一个有限数值（回包形状因命令而异）。"""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw) if math.isfinite(float(raw)) else None
    if isinstance(raw, (list, tuple)):
        for x in raw:
            v = _first_number(x)
            if v is not None:
                return v
    return None


def _numbers(raw: Any) -> list[float]:
    out: list[float] = []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if math.isfinite(float(raw)):
            out.append(float(raw))
    elif isinstance(raw, (list, tuple)):
        for x in raw:
            out.extend(_numbers(x))
    return out


def _envelope_fields(ctx: Any, warnings: list[str]) -> list[ProbeField]:
    """安全包络那几行 —— 由 ``core.envelope_reconcile`` 算，本模块只负责渲染。

    那个模块是开机自动对账用的同一份纯函数。让「按需对账」调它，两边的口径
    （``Piezo_RangeGet`` 给全程、包络存半程；未启用的 Z 限值不算硬件边界）
    就不可能漂开 —— 而两份并存的比较逻辑正是 ``_GLOBAL_CHECKS`` 当年出事的形状。

    它**只产出 findings**：没有任何返回值是 ``SafetyLimits``，也没有任何路径能
    写覆写文件。改不改由人定。
    """
    pool = getattr(ctx, "connection_pool", None)
    safe_call = getattr(pool, "safe_call", None) if pool is not None else None
    if not callable(safe_call):
        return []
    try:
        from mast.api.safety_view import in_force_safety_limits
        from mast.core.envelope_reconcile import reconcile_envelope

        limits, _source = in_force_safety_limits(ctx)
        result = reconcile_envelope(safe_call, limits)
    except Exception as exc:  # noqa: BLE001 — 对账失败不该把这一页打挂
        logger.warning("envelope reconcile (probe) failed: %s", exc)
        return []

    rows: list[ProbeField] = []
    for f in result.findings:
        rows.append(ProbeField(
            key=f"safety_limits.{f.field_name}", label=f.field_name,
            read=f.measured, configured=f.configured, verdict="mismatch",
            note=f.describe() + "（对账只报告，不自动改；若要改，只往收紧的方向改是安全的。）"))
    for name in result.unreadable:
        rows.append(ProbeField(key=f"probe.{name}", label=name, read=None,
                               configured=None, verdict="unread",
                               note="这次没能从仪器读到这个量。"))
    if result.wider:
        warnings.append(
            f"⚠ 有 {len(result.wider)} 项配置比硬件**更宽** —— 那些上限没有在拦"
            "任何东西，它们不是安全网，是一句关于安全网的假话。")
    if not result.findings and not result.unreadable:
        rows.append(ProbeField(
            key="safety_limits", label="安全包络", read="—", configured="—",
            verdict="match", note="配置的包络与实测行程一致。"))
    return rows


@router.post("/instrument-init/probe", response_model=InitProbeResponse)
def probe_rig(request: Request) -> InitProbeResponse:
    """从仪器读一次压电量程 / Z 限位 / 前放增益 / 偏压量程，与登记值对账。

    **只报告不一致，什么都不改。** 若一定要自动化，只能允许**收紧**
    （``skills/builtins/instrument_limits.py``：能放宽自己限值的 agent，
    严格来说没有限值）。

    这一步同时算出**硬件指纹** —— 换一台机器（或重做一次标定）它就会变，
    初始化横幅据此自动回来。指纹只在这里读硬件；清单的 GET 用进程内缓存，
    免得每次刷新都打四个 TCP 往返。
    """
    ctx = getattr(request.app.state, "ctx", None)
    ec = _execution_context(ctx)
    if ec is None:
        return InitProbeResponse(
            ok=False, degraded=True,
            message="内核单例未接线（见服务日志），与 Nanonis 连接状态无关。")

    readings: dict[str, Any] = {}
    warnings: list[str] = []
    z_limits_enabled: "bool | None" = None

    def _run(skill: str) -> dict[str, Any]:
        try:
            res = ec.run(skill, {})
            if getattr(res, "success", False):
                return dict(getattr(res, "data", {}) or {})
            warnings.append(f"{skill} 读取失败：{getattr(res, 'error', '未知错误')}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{skill} 读取异常：{type(exc).__name__}: {exc}")
        return {}

    piezo = _run("GetPiezoConfig")
    zctrl = _run("GetZControllerState")
    misc = _run("GetMiscInstrumentConfig")

    if piezo.get("range") is not None:
        readings["piezo_range"] = piezo["range"]
    if zctrl.get("z_limits") is not None:
        readings["z_limits"] = zctrl["z_limits"]
    if misc.get("current_gains") is not None:
        readings["current_gain_index"] = _first_number(misc["current_gains"])
    if misc.get("bias_range") is not None:
        readings["bias_range"] = misc["bias_range"]

    raw_enabled = zctrl.get("z_limits_enabled")
    n = _first_number(raw_enabled)
    if n is not None:
        z_limits_enabled = bool(n)
        if not z_limits_enabled:
            # KNOWN_ISSUES §1.3 —— 这条一直埋在回包里，没有任何地方在看它。
            warnings.append(
                "⚠ Nanonis 的 Z 软限位**没有启用**。现在不是活的风险（限位值恰好"
                "等于压电全量程），但只要有人为了安全把它收窄，设下去、回读也对、"
                "**却不会生效** —— 而 MAST 整套安全论证依赖的正是「Nanonis 会兜底」"
                "这一条。")

    values = _stored_values(ctx)
    fields: list[ProbeField] = []

    def _cmp_field(key: str, label: str, read: Any, configured: Any,
                   rel_tol: float = 0.05, note_ok: str = "") -> None:
        if read is None:
            fields.append(ProbeField(key=key, label=label, read=None,
                                     configured=configured, verdict="unread",
                                     note="这次没能从仪器读到这个量。"))
            return
        if configured is None:
            fields.append(ProbeField(
                key=key, label=label, read=read, configured=None,
                verdict="not_configured",
                note="仪器上读到了，但 MAST 这边还没登记 —— 填上它。"))
            return
        try:
            same = math.isclose(float(read), float(configured),
                                rel_tol=rel_tol, abs_tol=0.0)
        except (TypeError, ValueError):
            same = str(read) == str(configured)
        fields.append(ProbeField(
            key=key, label=label, read=read, configured=configured,
            verdict="match" if same else "mismatch",
            note=note_ok if same else
            (f"仪器说 {read!r}，MAST 登记的是 {configured!r}。"
             "**对账只报告，不自动改** —— 哪个对由你定；"
             "若要改，只往收紧的方向改是安全的。")))

    # ── 安全包络的比对**复用开机对账那一份实现** ────────────────────────────
    # `mast.core.envelope_reconcile` 是纯函数，与启动时那次自动对账用的是同一段
    # 代码。两份比较逻辑并存就是 `_GLOBAL_CHECKS` 当年漂移的同一个形状 ——
    # 「按需对账」和「开机对账」的口径（全程 vs 半程、未启用的 Z 限值不算数）
    # 必须永远一致，唯一可靠的办法是只有一份。
    fields.extend(_envelope_fields(ctx, warnings))

    # 下面几项**不在**那份实现的范围里：z_range_m 是 instrument_profile 的键
    # （不是 SafetyLimits 字段），bias_range 也不在开机对账读的三条里。
    rng = _numbers(piezo.get("range"))
    if len(rng) >= 3:
        _cmp_field("instrument_profile.z_range_m", "Z 压电总量程",
                   rng[2], values.get("instrument_profile.z_range_m"))

    br = _numbers(misc.get("bias_range"))
    if br:
        _cmp_field("safety_limits.bias_max_v", "偏压上限",
                   max(abs(x) for x in br), values.get("safety_limits.bias_max_v"))

    fp = ii.rig_fingerprint(readings)
    if fp and ctx is not None:
        try:
            setattr(ctx, _FP_CACHE_ATTR, fp)
        except Exception:  # noqa: BLE001 - pragma: no cover
            pass
    changed = ii.fingerprint_changed(_record(ctx), fp)
    if changed:
        warnings.append(
            "⚠ 硬件指纹与上次完成初始化时不同 —— 要么换了机器，要么有人重做了"
            "压电/增益标定。两种情况都值得把这份清单重新过一遍。")

    return InitProbeResponse(
        ok=bool(fields) or bool(readings), fields=fields, rig_fingerprint=fp,
        fingerprint_changed=changed, z_limits_enabled=z_limits_enabled,
        warnings=warnings,
        message=("" if readings else "一个量都没读到 —— 检查 Nanonis 连接。"),
    )
