"""Tip registry — 登记一次换针,并把随之失效的东西一起处理掉。

分层(与 sample 那套同构):
  * :mod:`mast.logging.storage` 是持久化(tips 表 + 事务);
  * :mod:`mast.core.tip_state` 是词表 + 进程 holder + 渲染(stdlib,skill 可读);
  * **本模块**是编排 —— 它知道"换针"这件事除了写一行以外还要做什么。

换针不只是记一笔。一批量是**绑上一根针**的,换了就失效:

  ``didv_at_contact_v``      每次成功进针 EWMA 学出来的接触判据,依赖针尖态
  ``qplus_amplitude_baseline`` 撞针判据的**分母**(当前振幅/基线 < 10% 判撞针)
                             —— 拿旧针的自由振幅当分母,判据非错即哑
  qPlus 实测 f₀ / Q          旧那支音叉的共振,不属于新装的这支

清掉它们是必须的,但**不必丢**:清之前先快照进退役针尖那一行的
``retire_snapshot``,日后还能查"上一根针的 dI/dV 标定是多少"。

不清的东西同样要想清楚(完整核查见
``docs/v2/design/tip_registry_and_hardware_profile.md`` §D5):``tilt_cal_*``
是样品/托架的倾斜响应、``qplus_amplitude_signal_index`` 是 Nanonis 信号槽接线、
``tip_crash_tracker`` 是表面位置状态 —— 换针都不会让它们失效。

所有入口 best-effort:登记针尖失败绝不能把一次实验带停,返回带 ``ok`` /
``warnings`` 的 dict(与 ``ScopeChange`` 同样"永不抛"的形状)。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from mast.core import instrument_profile as _iprof
from mast.core import tip_state

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _refresh_holder(storage: Any) -> "dict[str, Any] | None":
    """把 holder 同步到库里的当前针尖。返回当前针尖行。"""
    row = None
    try:
        row = storage.get_current_tip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取当前针尖失败(按未登记处理): %s", exc)
    tip_state.set_current_tip(row)
    return row


def hydrate(storage: Any) -> "dict[str, Any] | None":
    """启动时把当前针尖读进 holder。

    没有 restore_scope 那样的自愈链,因为不需要:当前针尖不是一个可能悬空的
    指针,而是"唯一 removed_at IS NULL 的那一行"—— 查询本身就是真源。
    """
    return _refresh_holder(storage)


def _log_event(storage: Any, text: str) -> None:
    """把换针写进当前实验的对话日志(时间线上看得见)。Best-effort。"""
    try:
        scope = storage.get_active_scope() or {}
        exp_id = scope.get("experiment_id")
        if not exp_id:
            return
        storage.log_conversation(
            role="system", content=text, experiment_id=exp_id,
            sample_id=scope.get("sample_id"), agent="tip_registry")
    except Exception as exc:  # noqa: BLE001 — 记事件失败不影响换针本身
        logger.debug("换针事件写入对话日志失败: %s", exc)


def register_tip(
    storage: Any,
    *,
    material: Any = "",
    fabrication: Any = "",
    form: Any = "",
    material_detail: str = "",
    name: str = "",
    wire_diameter_mm: Any = None,
    qplus_sensor_model: str = "",
    qplus_f0_hz: Any = None,
    qplus_q: Any = None,
    qplus_k_n_per_m: Any = None,
    installed_at: Any = None,
    installed_by: str = "",
    note: str = "",
) -> dict[str, Any]:
    """登记装入一根针尖(= 退役上一根 + 记新行 + 清失效标定 + 刷 holder)。

    词表(材料/制备/形态)在这里归一;认不出的值**不猜**,原样存下并在
    ``warnings`` 里说明,由调用方(工具层)决定是否回给模型候选清单。

    ``installed_at`` 可回填(用户常常过一两天才想起来记)。

    返回 ``{ok, tip_id, tip, cleared, warnings}``,永不抛。
    """
    warnings: list[str] = []

    mat = tip_state.normalize_material(material)
    if mat is None and str(material or "").strip():
        mat = str(material).strip()
        warnings.append(f"材料「{mat}」不在受控词表里，已原样记录（可用 update_tip 更正）")
    fab = tip_state.normalize_fabrication(fabrication)
    if fab is None and str(fabrication or "").strip():
        warnings.append(f"制备方式「{fabrication}」不在受控词表里，已记为 unknown")
    frm = tip_state.normalize_form(form)
    if frm is None and str(form or "").strip():
        warnings.append(f"针尖形态「{form}」不在受控词表里，已记为 stm_wire")

    diameter = None
    if wire_diameter_mm is not None and str(wire_diameter_mm).strip() != "":
        try:
            d = float(wire_diameter_mm)
            if d > 0 and d == d and d != float("inf"):
                diameter = d
            else:
                warnings.append(f"线材直径 {wire_diameter_mm!r} 非正数，已忽略")
        except (TypeError, ValueError):
            warnings.append(f"线材直径 {wire_diameter_mm!r} 不是数字，已忽略")

    def _pos(val: Any, label: str) -> "float | None":
        if val is None or str(val).strip() == "":
            return None
        try:
            f = float(val)
            if f > 0 and f == f and f != float("inf"):
                return f
        except (TypeError, ValueError):
            pass
        warnings.append(f"{label} {val!r} 非法，已忽略")
        return None

    fields = {
        "name": str(name or "").strip(),
        "material": mat or "",
        "material_detail": str(material_detail or "").strip(),
        "fabrication": fab or "unknown",
        "form": frm or "stm_wire",
        "wire_diameter_mm": diameter,
        "qplus_sensor_model": str(qplus_sensor_model or "").strip(),
        "qplus_f0_hz": _pos(qplus_f0_hz, "qPlus 共振频率"),
        "qplus_q": _pos(qplus_q, "qPlus Q 值"),
        "qplus_k_n_per_m": _pos(qplus_k_n_per_m, "qPlus 弹性常数"),
        "installed_at": str(installed_at).strip() if installed_at else _now_iso(),
        "installed_by": str(installed_by or "").strip(),
        "note": str(note or "").strip(),
    }

    # 先取快照再写库:清标定发生在 create_tip 之后,但快照要存进**退役行**,
    # 所以必须在 UPDATE 之前就拿到。
    try:
        snapshot = _iprof.get_tip_bound_state()
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取待归档标定失败: %s", exc)
        snapshot = {}

    try:
        tip_id = storage.create_tip(fields, retire_snapshot=snapshot)
    except Exception as exc:  # noqa: BLE001
        logger.warning("登记针尖失败: %s", exc)
        return {"ok": False, "error": f"登记针尖失败: {exc}",
                "warnings": warnings, "cleared": {}}

    # 名字留空时用序号补一个人能认的默认名(序号由库分配,所以只能事后补)。
    row = None
    try:
        row = storage.get_tip(tip_id)
        if row is not None and not (row.get("name") or "").strip():
            auto = tip_state.auto_name(
                row.get("material") or "tip", row.get("fabrication") or "",
                row.get("tip_index"))
            storage.update_tip(tip_id, {"name": auto})
    except Exception as exc:  # noqa: BLE001
        logger.debug("自动命名失败(不影响登记): %s", exc)

    try:
        cleared = _iprof.clear_tip_bound_state()
    except Exception as exc:  # noqa: BLE001
        logger.warning("清除上一根针的标定失败: %s", exc)
        cleared = {}
        warnings.append("上一根针的学习标定未能清除，请在设置里手动清一次 dI/dV 标定")

    row = _refresh_holder(storage)
    label = (row or {}).get("name") or tip_id[:8]
    _log_event(storage, f"登记装入针尖「{label}」"
                        f"（{fields['material'] or '材料未记录'}/"
                        f"{fields['fabrication']}/{fields['form']}）"
                        + (f"；清除了上一根针的标定 {sorted(cleared)}" if cleared else ""))

    return {"ok": True, "tip_id": tip_id, "tip": row,
            "cleared": cleared, "warnings": warnings}


def remove_current_tip(storage: Any, *, note: str = "") -> dict[str, Any]:
    """物理取出针尖但还没装新的(罕见:通常直接 register_tip 装下一根)。

    与登记同样的副作用 —— 取出后旧标定同样失效。
    """
    try:
        snapshot = _iprof.get_tip_bound_state()
    except Exception:  # noqa: BLE001
        snapshot = {}
    try:
        prev = storage.get_current_tip()
        did = storage.retire_current_tip(retire_snapshot=snapshot)
    except Exception as exc:  # noqa: BLE001
        logger.warning("退役当前针尖失败: %s", exc)
        return {"ok": False, "error": f"退役当前针尖失败: {exc}", "cleared": {}}
    if not did:
        tip_state.set_current_tip(None)
        return {"ok": True, "changed": False, "cleared": {},
                "message": "当前没有已登记的针尖，无需退役。"}
    try:
        cleared = _iprof.clear_tip_bound_state()
    except Exception:  # noqa: BLE001
        cleared = {}
    tip_state.set_current_tip(None)
    _log_event(storage, f"取出针尖「{(prev or {}).get('name') or '未命名'}」"
                        + (f"；{note}" if note else ""))
    return {"ok": True, "changed": True, "cleared": cleared, "previous": prev}


def update_tip(storage: Any, tip_id: str, fields: dict) -> dict[str, Any]:
    """补记针尖属性(线径、型号、备注…)。词表字段同样归一。"""
    clean: dict[str, Any] = {}
    warnings: list[str] = []
    for key in ("name", "material_detail", "qplus_sensor_model", "note", "installed_at"):
        if key in fields and fields[key] is not None:
            clean[key] = str(fields[key]).strip()
    if "material" in fields:
        mat = tip_state.normalize_material(fields["material"])
        if mat is None:
            warnings.append(f"材料「{fields['material']}」不在受控词表里，未改动")
        else:
            clean["material"] = mat
    if "fabrication" in fields:
        fab = tip_state.normalize_fabrication(fields["fabrication"])
        if fab is None:
            warnings.append(f"制备方式「{fields['fabrication']}」不在受控词表里，未改动")
        else:
            clean["fabrication"] = fab
    if "form" in fields:
        frm = tip_state.normalize_form(fields["form"])
        if frm is None:
            warnings.append(f"针尖形态「{fields['form']}」不在受控词表里，未改动")
        else:
            clean["form"] = frm
    for key in ("wire_diameter_mm", "qplus_f0_hz", "qplus_q", "qplus_k_n_per_m"):
        if key in fields and fields[key] is not None and str(fields[key]).strip() != "":
            try:
                f = float(fields[key])
                if f > 0 and f == f and f != float("inf"):
                    clean[key] = f
                else:
                    warnings.append(f"{key} = {fields[key]!r} 非正数，未改动")
            except (TypeError, ValueError):
                warnings.append(f"{key} = {fields[key]!r} 不是数字，未改动")
    if not clean:
        return {"ok": False, "error": "没有可改的字段。", "warnings": warnings}
    try:
        changed = storage.update_tip(tip_id, clean)
    except Exception as exc:  # noqa: BLE001
        logger.warning("更新针尖失败: %s", exc)
        return {"ok": False, "error": f"更新针尖失败: {exc}", "warnings": warnings}
    if not changed:
        return {"ok": False, "error": f"没有 id={tip_id} 的针尖。", "warnings": warnings}
    row = _refresh_holder(storage)
    try:
        updated = storage.get_tip(tip_id)
    except Exception:  # noqa: BLE001
        updated = row
    return {"ok": True, "tip": updated, "updated": sorted(clean), "warnings": warnings}


__all__ = ["hydrate", "register_tip", "remove_current_tip", "update_tip"]
