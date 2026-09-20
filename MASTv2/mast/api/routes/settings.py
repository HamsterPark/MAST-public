"""GET /api/settings — persisted UI settings (SettingsStore whitelist).

Read-only in Phase 2. The POST counterpart (with hot-reload + safety passthrough)
arrives in Phase 3.

Also GET /api/settings/hardware-modules — the optional (licensed-but-maybe-absent)
Nanonis modules and their on/off state. Served from the process-level holder, not
from the store: the holder is what the tool-list gate actually reads, so this
endpoint shows the operator the state that is really in force rather than the
state that was merely written to disk.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from mast.api.admin_pin import pin_is_set
from mast.api.schemas import (
    AdvancedCapabilitiesResponse,
    AdvancedCapabilityState,
    HardwareModulesResponse,
    HardwareModuleState,
    SettingsResponse,
)
from mast.skills.advanced_capabilities import (
    capability_states,
    disabled_skill_names as disabled_advanced_skill_names,
)
from mast.skills.hardware_modules import disabled_skill_names, module_states

router = APIRouter(tags=["settings"])


@router.get("/settings", response_model=SettingsResponse)
def get_settings(request: Request) -> SettingsResponse:
    ctx = request.app.state.ctx
    store = ctx.settings_store
    data: dict = {}
    if store is not None:
        try:
            data = store.load()
        except Exception:
            data = {}
    # SettingsResponse ignores unknown keys and leaves unset ones None.
    return SettingsResponse(**{k: v for k, v in data.items() if k in SettingsResponse.model_fields})


@router.get("/settings/scan-policy")
def get_scan_policy(request: Request) -> dict:
    """扫描参数档位表:当前**生效**的表 + 出厂模板 + 可编辑字段规格。

    返回的是 holder 里真正在用的表(不是磁盘上写着的),所以用户看到的就是
    下一次扫描会用的参数。``stored`` 为空表示他从没改过 —— 界面拿 ``factory``
    做 placeholder,而不是把出厂值填进输入框假装是他设的。

    路径必须留在任何 ``/settings/{key}`` 式路由**之上**:一个路径参数会把它
    遮蔽掉,变成查询一个名叫 "scan-policy" 的设置项(群聊读端点就这么被静默
    变成过一个健康的空列表)。
    """
    from mast.core import scan_policy

    return {
        "tiers": scan_policy.get_policy(),
        "stored": scan_policy.get_stored_policy(),
        "factory": scan_policy.factory_tiers(),
        "customised": scan_policy.is_customised(),
        "min_tiers": scan_policy.MIN_TIERS,
        "max_tiers": scan_policy.MAX_TIERS,
        "size_min_m": scan_policy.SIZE_MIN_M,
        "size_max_m": scan_policy.SIZE_MAX_M,
        "fields": scan_policy.tier_field_specs(),
    }


@router.get("/settings/zctrl-presets")
def get_zctrl_presets(request: Request) -> dict:
    """Z 参数组:每个可用名、它解析出来的数值,以及那些数值**来自哪里**。

    ``sources`` 不是装饰:三个名字可以指向三个不同的存储(仪器档案 / 扫描档位表 /
    自定义组),用户要能一眼看出「改哪里才管用」。

    与 scan-policy 同理,路径必须留在任何 ``/settings/{key}`` 式路由之上。
    """
    from mast.core import zctrl_presets as zp

    rows: list[dict] = []
    for name in zp.available_names():
        row: dict = {"name": name}
        try:
            # 不传 context:'scan' 要读当前帧宽才能定档,而设置页不该触发硬件 I/O。
            res = zp.resolve(name)
            row.update({
                "p_gain": res.p_gain,
                "i_gain": res.i_gain,
                "time_constant_s": res.time_constant_s,
                "setpoint_a": res.setpoint_a,
                "sources": res.sources,
                "trace": res.trace_lines(),
                "usable": True,
            })
        except Exception as exc:  # noqa: BLE001 — 不可用也要列出来并说明原因
            row.update({"usable": False, "why": str(exc)})
        rows.append(row)

    custom = zp.get_presets()
    custom_names = {p["name"] for p in custom}
    for row in rows:
        row["kind"] = (
            "reserved" if row["name"] in zp.RESERVED_NAMES
            else "custom" if row["name"] in custom_names
            else "scan-tier"
        )
    return {
        "presets": rows,
        "custom": custom,
        "reserved": list(zp.RESERVED_NAMES),
        "max_presets": zp.MAX_PRESETS,
    }


@router.get("/settings/scan-policy/preview")
def preview_scan_policy(size_nm: float, purpose: str = "auto") -> dict:
    """「一张 X nm 的图会用什么参数」—— 零硬件成本的档位表预览。

    存在的唯一理由是让用户改完表立刻看到效果,而不必真的扫一张图去验证。
    输入用 nm(界面上的人类单位),内部转 SI。
    """
    from mast.core import scan_resolver

    try:
        out = scan_resolver.preview(float(size_nm) * 1e-9, purpose=purpose)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    out["ok"] = True
    out["size_nm"] = size_nm
    return out


@router.get("/settings/hardware-modules", response_model=HardwareModulesResponse)
def get_hardware_modules(request: Request) -> HardwareModulesResponse:
    """The optional hardware modules, their live on/off state, and what each needs.

    NB the literal path must stay ABOVE any ``/settings/{key}``-style route if one
    is ever added — a path parameter would shadow it and this would silently
    become a lookup of a setting named "hardware-modules" (that shadowing bug
    once turned the group-chat read endpoints into a healthy-looking empty list).
    """
    mods = module_states()
    return HardwareModulesResponse(
        modules=[HardwareModuleState(**m) for m in mods],
        enabled_count=sum(1 for m in mods if m["enabled"]),
        gated_skill_count=len(disabled_skill_names()),
    )


@router.get("/settings/advanced-capabilities", response_model=AdvancedCapabilitiesResponse)
def get_advanced_capabilities(request: Request) -> AdvancedCapabilitiesResponse:
    """高级 → 高级能力: powers that can step around a protection, all OFF by default.

    ``pin_is_set`` False means the toggles cannot be written AT ALL yet — the write
    path is PIN-guarded and fails closed. The UI turns that into "set a PIN first"
    rather than showing switches that would silently refuse.
    """
    caps = capability_states()
    return AdvancedCapabilitiesResponse(
        capabilities=[AdvancedCapabilityState(**c) for c in caps],
        enabled_count=sum(1 for c in caps if c["enabled"]),
        gated_skill_count=len(disabled_advanced_skill_names()),
        pin_is_set=pin_is_set(),
    )
