"""针尖登记 API —— 「当前装的是哪根针」以及换针史。

设计文档：``docs/v2/design/tip_registry_and_hardware_profile.md``

作用域是**仪器**而不是实验：换实验、换样品都未必换针尖。所以这里没有
experiment_id 参数，也没有「切换针尖」——针尖服役期是线性的，
当前针尖就是唯一 ``removed_at IS NULL`` 的那一行（见 storage 建表注释）。

登记新针会清掉绑上一根针的学习标定（dI/dV 接触标定、qPlus 自由振幅基线），
旧值归档进退役那一行。前端在提交前应当把这件事告诉用户。

schemas 内联（与 ``routes/scope.py`` 同款）：这些模型只有这一个消费者，
放进共享 schemas 文件只会让「改一个字段要动两个文件」。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tips"])


# ── schemas ───────────────────────────────────────────────────────────

class TipModel(BaseModel):
    id: str
    tip_index: int | None = None
    name: str = ""
    material: str = ""
    material_detail: str = ""
    fabrication: str = "unknown"
    form: str = "stm_wire"
    wire_diameter_mm: float | None = None
    qplus_sensor_model: str = ""
    qplus_f0_hz: float | None = None
    qplus_q: float | None = None
    qplus_k_n_per_m: float | None = None
    installed_at: str | None = None
    removed_at: str | None = None
    installed_by: str = ""
    note: str = ""
    #: 退役时清掉的学习标定（归档，不是丢弃）。
    retire_snapshot: dict[str, Any] = Field(default_factory=dict)


class CurrentTipResponse(BaseModel):
    tip: TipModel | None = None
    registered: bool = False
    #: 后端不可用时为 True —— 前端据此显示「读不到」而不是「没有针尖」。
    degraded: bool = False
    hint: str = ""


class TipListResponse(BaseModel):
    tips: list[TipModel] = Field(default_factory=list)
    degraded: bool = False


class RegisterTipBody(BaseModel):
    material: str = ""
    fabrication: str = ""
    form: str = ""
    name: str = ""
    material_detail: str = ""
    wire_diameter_mm: float | None = None
    qplus_sensor_model: str = ""
    qplus_f0_hz: float | None = None
    qplus_q: float | None = None
    qplus_k_n_per_m: float | None = None
    #: ISO 日期，可回填（用户常常过一两天才想起来记）。留空 = 现在。
    installed_at: str = ""
    installed_by: str = ""
    note: str = ""


class UpdateTipBody(BaseModel):
    name: str | None = None
    material: str | None = None
    material_detail: str | None = None
    fabrication: str | None = None
    form: str | None = None
    wire_diameter_mm: float | None = None
    qplus_sensor_model: str | None = None
    qplus_f0_hz: float | None = None
    qplus_q: float | None = None
    qplus_k_n_per_m: float | None = None
    installed_at: str | None = None
    note: str | None = None


class TipChangeResult(BaseModel):
    ok: bool = False
    tip: TipModel | None = None
    changed: bool = True
    #: 本次操作清掉的标定键（前端提示「这些需要重新标定」）。
    cleared_calibration: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str = ""


class TipVocabularyResponse(BaseModel):
    """受控词表 —— 前端下拉框的唯一真源，免得两边各写一份分叉。"""
    materials: list[dict[str, str]] = Field(default_factory=list)
    fabrications: list[dict[str, str]] = Field(default_factory=list)
    forms: list[dict[str, str]] = Field(default_factory=list)


# ── helpers ───────────────────────────────────────────────────────────

_NO_TIP_HINT = ("当前没有已登记的针尖。仪器里物理上当然有针，"
                "但没登记它是什么，修针方案只能用通用保守参数。")


def _storage(request: Request):
    try:
        return request.app.state.ctx.experiment_storage
    except Exception:  # noqa: BLE001
        return None


def _model(row: dict | None) -> TipModel | None:
    if not row:
        return None
    import json as _json
    snap = row.get("retire_snapshot")
    if isinstance(snap, str):
        try:
            snap = _json.loads(snap or "{}")
        except Exception:  # noqa: BLE001
            snap = {}
    if not isinstance(snap, dict):
        snap = {}
    return TipModel(
        id=str(row.get("id") or ""),
        tip_index=row.get("tip_index"),
        name=row.get("name") or "",
        material=row.get("material") or "",
        material_detail=row.get("material_detail") or "",
        fabrication=row.get("fabrication") or "unknown",
        form=row.get("form") or "stm_wire",
        wire_diameter_mm=row.get("wire_diameter_mm"),
        qplus_sensor_model=row.get("qplus_sensor_model") or "",
        qplus_f0_hz=row.get("qplus_f0_hz"),
        qplus_q=row.get("qplus_q"),
        qplus_k_n_per_m=row.get("qplus_k_n_per_m"),
        installed_at=row.get("installed_at"),
        removed_at=row.get("removed_at"),
        installed_by=row.get("installed_by") or "",
        note=row.get("note") or "",
        retire_snapshot=snap,
    )


# ── endpoints ─────────────────────────────────────────────────────────
#
# ⚠️ 声明顺序:``/tips/current`` 必须在 ``/tips/{tip_id}`` 之前 —— FastAPI 按
# 声明序匹配,反过来的话 "current" 会被当成一个 tip_id。

@router.get("/tips/current", response_model=CurrentTipResponse)
def current_tip(request: Request) -> CurrentTipResponse:
    """当前装在仪器里的针尖。前端针尖卡片的真源。"""
    st = _storage(request)
    if st is None:
        return CurrentTipResponse(degraded=True, hint="实验记录库不可用。")
    try:
        row = st.get_current_tip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("current tip read failed: %s", exc)
        return CurrentTipResponse(degraded=True, hint="读取当前针尖失败。")
    if not row:
        return CurrentTipResponse(registered=False, hint=_NO_TIP_HINT)
    return CurrentTipResponse(tip=_model(row), registered=True)


@router.get("/tips/vocabulary", response_model=TipVocabularyResponse)
def tip_vocabulary() -> TipVocabularyResponse:
    """材料/制备/形态的受控词表（后端是唯一真源）。"""
    try:
        from mast.core import tip_state as ts
        return TipVocabularyResponse(
            materials=[{"value": v, "label": ts.MATERIAL_LABELS.get(v, v)}
                       for v in ts.TIP_MATERIALS],
            fabrications=[{"value": v, "label": ts.FABRICATION_LABELS.get(v, v)}
                          for v in ts.TIP_FABRICATIONS],
            forms=[{"value": v, "label": ts.FORM_LABELS.get(v, v)}
                   for v in ts.TIP_FORMS],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("tip vocabulary unavailable: %s", exc)
        return TipVocabularyResponse()


@router.get("/tips", response_model=TipListResponse)
def list_tips(request: Request, limit: int = 100) -> TipListResponse:
    """换针史，最近装入的在前。"""
    st = _storage(request)
    if st is None:
        return TipListResponse(degraded=True)
    try:
        rows = st.list_tips(limit=max(1, min(int(limit or 100), 500)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("tip list read failed: %s", exc)
        return TipListResponse(degraded=True)
    return TipListResponse(tips=[m for m in (_model(r) for r in rows) if m])


@router.post("/tips", response_model=TipChangeResult)
def register_tip(body: RegisterTipBody, request: Request) -> TipChangeResult:
    """登记装入一根针尖：退役上一根 + 记新行 + 清掉绑旧针的学习标定。"""
    st = _storage(request)
    if st is None:
        return TipChangeResult(ok=False, error="实验记录库不可用。")
    try:
        from mast.logging import tip_registry as tr
        res = tr.register_tip(
            st,
            material=body.material, fabrication=body.fabrication, form=body.form,
            name=body.name, material_detail=body.material_detail,
            wire_diameter_mm=body.wire_diameter_mm,
            qplus_sensor_model=body.qplus_sensor_model,
            qplus_f0_hz=body.qplus_f0_hz, qplus_q=body.qplus_q,
            qplus_k_n_per_m=body.qplus_k_n_per_m,
            installed_at=(body.installed_at or None),
            installed_by=body.installed_by, note=body.note,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("register tip failed: %s", exc)
        return TipChangeResult(ok=False, error=str(exc))
    if not res.get("ok"):
        return TipChangeResult(ok=False, error=res.get("error", "登记失败"),
                               warnings=res.get("warnings") or [])
    return TipChangeResult(
        ok=True, tip=_model(res.get("tip")),
        cleared_calibration=sorted(res.get("cleared") or {}),
        warnings=res.get("warnings") or [])


@router.patch("/tips/{tip_id}", response_model=TipChangeResult)
def update_tip(tip_id: str, body: UpdateTipBody, request: Request) -> TipChangeResult:
    """补记/更正针尖属性。不清任何标定 —— 那只在登记新针时发生。"""
    st = _storage(request)
    if st is None:
        return TipChangeResult(ok=False, error="实验记录库不可用。")
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        return TipChangeResult(ok=False, error="没有要改的字段。")
    try:
        from mast.logging import tip_registry as tr
        res = tr.update_tip(st, tip_id, fields)
    except Exception as exc:  # noqa: BLE001
        logger.warning("update tip failed: %s", exc)
        return TipChangeResult(ok=False, error=str(exc))
    if not res.get("ok"):
        return TipChangeResult(ok=False, error=res.get("error", "更新失败"),
                               warnings=res.get("warnings") or [])
    return TipChangeResult(ok=True, tip=_model(res.get("tip")),
                           warnings=res.get("warnings") or [])


@router.post("/tips/current/remove", response_model=TipChangeResult)
def remove_current_tip(request: Request) -> TipChangeResult:
    """记录「针尖已取出、还没装新的」。通常不需要 —— 装下一根直接登记即可。"""
    st = _storage(request)
    if st is None:
        return TipChangeResult(ok=False, error="实验记录库不可用。")
    try:
        from mast.logging import tip_registry as tr
        res = tr.remove_current_tip(st)
    except Exception as exc:  # noqa: BLE001
        logger.warning("remove tip failed: %s", exc)
        return TipChangeResult(ok=False, error=str(exc))
    if not res.get("ok"):
        return TipChangeResult(ok=False, error=res.get("error", "退役失败"))
    return TipChangeResult(
        ok=True, changed=bool(res.get("changed", True)),
        cleared_calibration=sorted(res.get("cleared") or {}))


__all__ = ["router"]
