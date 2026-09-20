"""agent 手里的 conduct 工具 —— 全部 agent 都有，把关在服务端。

## 权限哲学（2026-08-20 定）

本仓从前按角色裁剪工具面：谁能看见 ``approve_plan`` 是在建图时决定的，
理由写在 ``meta_tools.DESIGN_TOOL_NAMES`` 的注释里（「一个 agent 不能批准
自己写的方案」）。这套做法有两个问题：

1. **维护成本随能力数 × 角色数增长。** 每加一个能力都要判 N 次「这个角色该不
   该有」，而这类清单在本仓已经漂过好几次（instrument_lock 的入口计数、
   settings 的 KNOWN_KEYS，都是同一个形状）。
2. **它挡不住真正该挡的东西，却挡住了本该允许的事。** 「不给工具」防的是模型
   *想不到*去做，防不了模型换条路做；而它同时让「夜里没人时谁来批」变成了
   一个无解的问题 —— 那正是全自动实验室的天花板。

所以这一族工具**对全部 agent 可见可调**，把关全部移到服务端：

* **自主度策略** —— attended / supervised / autonomous 决定「谁点头算数」
  （:mod:`mast.conduct.autonomy`）；
* **参数包络** —— validator 的数值界，超界当场拒绝（**拒绝不夹紧**）；
* **op 状态机** —— 不可能的转移显式 ``op_rejected`` 留痕，不静默 no-op；
* **预算闸 / SafetyGate / 三态求值** —— 与人操作时逐字节相同。

每一个动作都进审计流。**自主不等于免审计** —— 恰恰相反：人按按钮时旁边还有
一个人的记忆，agent 按按钮时只有那条事件。

## 这一族**不做**的两件事

* **不直接驱动仪器。** 工具只往 ``conduct_ops`` 意图队列里放东西，由 Director
  在自己的 tick 里消费。仪器动作永远走 ``ExecutionContext`` 的完整管道。
* **不发明数字。** 建一份 conduct 要的参数来自 plan 的编译产物或用户，
  这里只负责把它们递过去；超包络的值会被服务端原样拒绝，而不是夹到边界上
  （夹紧会让「填错了」看起来像「填对了」）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

__all__ = ["make_conduct_tools", "CONDUCT_TOOL_NAMES"]

#: 这一族的工具名。与 :func:`make_conduct_tools` 的返回一一对应，有测试钉着
#: （名单与实物对不上是本仓踩过的形状：清单说有、图上没有）。
CONDUCT_TOOL_NAMES: tuple[str, ...] = (
    "conduct_status",
    "conduct_list",
    "conduct_compile_from_plan",
    "conduct_create",
    "conduct_approve",
    "conduct_post_op",
)


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _service():
    """拿服务句柄；引擎关着时回 None。**惰性 import**。

    这里刻意不在模块顶层 import ``mast.conduct.service``：agent 的工具模块会
    在建图时被导入，而那时 runtime 可能还没起。更要紧的是 ``cd_enabled=0``
    时服务对象根本不存在，而那时这些工具仍然要能诚实作答（「引擎关着」是一个
    答案，不是一次异常）。
    """
    try:
        from mast.conduct.service import get_service

        return get_service()
    except Exception as exc:  # noqa: BLE001
        logger.debug("conduct service 取不到: %s", exc)
        return None


def _readonly_store():
    """引擎关着时的只读通道。

    ``cd_enabled=0`` 只该关掉**推进**，不该关掉**看**：一份跑到一半、进程重启
    之后引擎被关掉的 conduct，它的状态仍然是用户最想知道的东西。设置页那句
    「关掉它不会关掉读端点」在此之前只是一句承诺 —— 服务对象是 None 时读路由
    一律 degraded，谁也读不到。这个函数就是那句承诺的实现。
    """
    try:
        from mast.agents._shared.data_paths import experiment_db_path
        from mast.conduct.store import ConductStore

        return ConductStore(experiment_db_path())
    except Exception as exc:  # noqa: BLE001
        logger.debug("conduct 只读库打不开: %s", exc)
        return None


def _store():
    svc = _service()
    if svc is not None:
        return svc.store, True
    return _readonly_store(), False


def make_conduct_tools(agent_name: str = "") -> list:
    """建这一族工具。

    *agent_name* 只用来在审计流里署名（``agent:<name>``）—— **不用来决定
    给不给**。署名会进 ``approved_by`` / op 的 ``by`` 字段，于是「谁点的头」
    永远答得上来。
    """
    who = f"agent:{agent_name or 'unknown'}"

    @tool
    def conduct_status(conduct_id: str = "") -> str:
        """看一份 conduct 现在到哪一步了（只读）。

        不给 conduct_id 就看当前活跃的那一份。返回：状态、所在阶段/步、
        在等什么、心跳年龄、花销（逐币种）。

        「读不到」和「没有」是两个词：引擎关着、库打不开、这份不存在，
        三种情况的回答各不相同。
        """
        store, live = _store()
        if store is None:
            return _j({"ok": False, "reason": "conduct 库打不开（不是「没有 conduct」）"})
        try:
            row = store.get(conduct_id) if conduct_id else store.active()
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"读库失败: {exc}"})
        if row is None:
            return _j({"ok": True, "engine_running": live, "conduct": None,
                       "note": "现在没有活跃的 conduct" if not conduct_id
                               else f"没有 id={conduct_id} 这一份"})
        wait = row.get("active_wait") or {}
        return _j({
            "ok": True,
            "engine_running": live,
            "conduct": {
                "conduct_id": row.get("conduct_id"),
                "status": row.get("status"),
                "status_reason": row.get("status_reason"),
                "spec_id": row.get("spec_id"),
                "spec_version": row.get("spec_version"),
                "stage_idx": row.get("stage_idx"),
                "step_idx": row.get("step_idx"),
                "attended": row.get("attended"),
                "evidence_epoch": row.get("evidence_epoch"),
                "waiting_for": wait.get("lacking") or wait.get("kind") or None,
                "heartbeat_at": row.get("heartbeat_at"),
                "experiment_id": row.get("experiment_id"),
            },
            "engine_note": None if live else
                "指挥线程没在跑（cd_enabled=0 或还没起）——状态是磁盘上的最后一笔，"
                "它不会自己往前走。",
        })

    @tool
    def conduct_list(status: str = "", limit: int = 20) -> str:
        """列出 conduct（只读）。status 可填 draft/running/waiting_operator/… 或留空。"""
        store, live = _store()
        if store is None:
            return _j({"ok": False, "reason": "conduct 库打不开"})
        try:
            rows = store.list_conducts(status=status or None, limit=max(1, min(200, int(limit))))
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"读库失败: {exc}"})
        return _j({"ok": True, "engine_running": live, "count": len(rows),
                   "conducts": [{k: r.get(k) for k in
                                 ("conduct_id", "status", "spec_id", "title",
                                  "stage_idx", "step_idx", "created_at")}
                                for r in rows]})

    @tool
    def conduct_compile_from_plan(plan_id: str) -> str:
        """把一份已批准的实验方案编译成可执行的 conduct 参数（**试编译，不落库**）。

        编译不通过时返回逐条闭集错误码（哪个槽没填、哪个数超包络、哪个技能
        这台机器上没有），照着改 plan 再试。**这一步不建任何东西。**
        """
        try:
            from mast.conduct.compiler import compile_plan_to_conduct_spec
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"编译器不可用: {exc}"})
        try:
            res = compile_plan_to_conduct_spec(plan_id)
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"编译失败: {exc}"})
        return _j({
            "ok": bool(getattr(res, "ok", False)),
            "spec_id": getattr(getattr(res, "spec", None), "spec_id", None),
            "params": getattr(res, "params", {}),
            "errors": [{"code": e.code, "stage": e.stage_id, "slot": e.slot_name,
                        "detail": e.detail} for e in getattr(res, "errors", ())],
            "warnings": list(getattr(res, "warnings", ())),
        })

    @tool
    def conduct_create(experiment_id: str, spec_id: str = "",
                       params_json: str = "", from_plan_id: str = "",
                       title: str = "") -> str:
        """建一份 conduct（DRAFT 态，**不会自己开跑**）。

        两条路二选一：给 ``from_plan_id`` 由编译器出参数，或者直接给
        ``spec_id`` + ``params_json``。超包络的参数会被**原样拒绝**并告诉你
        哪个字段越了界 —— 不会被悄悄夹到边界上。

        建完是 DRAFT：要跑还得过 ``conduct_approve``，而那一步由自主度策略
        决定谁点头算数。
        """
        svc = _service()
        if svc is None:
            return _j({"ok": False, "reason":
                       "conduct 指挥线程没在跑（cd_enabled=0）：可以读，但建不了新的。"
                       "要开它请在设置页翻开「启用 conduct 指挥线程」。"})
        try:
            params = json.loads(params_json) if params_json else {}
            if not isinstance(params, dict):
                return _j({"ok": False, "reason": "params_json 要是一个 JSON 对象"})
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"params_json 解析不了: {exc}"})

        sid, warns = spec_id, []
        if from_plan_id and not params:
            try:
                from mast.conduct.compiler import compile_plan_to_conduct_spec

                res = compile_plan_to_conduct_spec(from_plan_id)
                if not res.ok:
                    return _j({"ok": False, "reason": "方案编译不通过，没有建",
                               "errors": [{"code": e.code, "slot": e.slot_name,
                                           "detail": e.detail} for e in res.errors]})
                sid = res.spec.spec_id
                params = dict(res.params)
                warns = list(res.warnings)
            except Exception as exc:  # noqa: BLE001
                return _j({"ok": False, "reason": f"编译失败: {exc}"})

        if not sid:
            return _j({"ok": False, "reason": "要么给 spec_id，要么给 from_plan_id"})
        try:
            from mast.conduct.templates import get_template

            spec = get_template(sid)
            if spec is None:
                return _j({"ok": False, "reason": f"这台机器上没有模板 {sid!r}"})
            cid = svc.store.create(
                experiment_id=experiment_id, spec_id=sid,
                spec_version=int(spec.spec_version), params=params,
                title=title or spec.title,
                created_by=(f"compiler:plan={from_plan_id}" if from_plan_id else who))
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"建不出来: {exc}"})
        return _j({"ok": True, "conduct_id": cid, "status": "draft",
                   "warnings": warns,
                   "next": "过 conduct_approve 才会开跑"})

    @tool
    def conduct_approve(conduct_id: str) -> str:
        """批准一份 DRAFT，让它可以开跑。

        能不能批由**自主度策略**决定，不由「你是哪个 agent」决定：
        attended 档下要人来点（这里会如实回「需要人批」并说清怎么改），
        supervised 档下你可以批但点火前留一个撤销窗，autonomous 档下即批即跑。
        **三档下模板 lint、参数包络、approvable 判据完全一样。**
        """
        svc = _service()
        if svc is None:
            return _j({"ok": False, "reason": "conduct 指挥线程没在跑（cd_enabled=0）"})
        try:
            from mast.conduct.autonomy import (
                ignition_payload as _ignition_payload,
                stricter_of,
                who_may_approve,
            )
            from mast.conduct.settings import get_conduct_knobs
            from mast.conduct.templates import get_template
            from mast.conduct.validator import validate_spec
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"批准链路不可用: {exc}"})

        row = svc.store.get(conduct_id)
        if row is None:
            return _j({"ok": False, "reason": f"没有 id={conduct_id} 这一份"})
        if str(row.get("status")) != "draft":
            return _j({"ok": False, "reason":
                       f"只有 DRAFT 能批准，现在是 {row.get('status')}"})
        spec = get_template(str(row.get("spec_id") or ""))
        if spec is None:
            return _j({"ok": False, "reason":
                       f"取不到模板 {row.get('spec_id')!r}"})

        knobs = get_conduct_knobs()
        level = stricter_of(knobs.autonomy, getattr(spec, "max_autonomy", None))
        verdict = who_may_approve(level, by=who,
                                  ignition_delay_s=knobs.cd_ignition_delay_s)
        if not verdict.allowed:
            return _j({"ok": False, "autonomy": level, "reason": verdict.reason})

        rep = validate_spec(spec, skills=None, analyses=None)
        if not rep.approvable:
            return _j({"ok": False, "reason": "模板批不下去",
                       "findings": [str(f) for f in rep.findings],
                       "checks_skipped": list(rep.checks_skipped)})
        try:
            svc.store.record(conduct_id, "approved", changes={
                "status": "approved", "approved_by": who,
                "approved_at": svc.store.now_iso()},
                payload={"by": who, "spec_version": int(spec.spec_version),
                         **_ignition_payload(verdict, svc.store.now_epoch())})
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": str(exc)})
        return _j({"ok": True, "conduct_id": conduct_id, "status": "approved",
                   "autonomy": level,
                   "ignition_delay_s": verdict.ignition_delay_s,
                   "note": verdict.reason or None})

    @tool
    def conduct_post_op(conduct_id: str, op: str, reason: str = "") -> str:
        """给一份 conduct 递一个意图：start / pause / resume / abort / takeover。

        **只是入队**——状态的单写者永远是 Director，它会在自己的 tick 里消费。
        状态不允许的转移会被显式拒绝并留痕（不会静默地什么都不发生）。

        abort 是例外中的例外：它在入队的同时**立刻**置位中止事件，因为
        Director 可能正卡在一次仪器动作里，而那正是最需要它的时刻。
        """
        svc = _service()
        if svc is None:
            return _j({"ok": False, "reason": "conduct 指挥线程没在跑（cd_enabled=0）"})
        allowed = ("start", "pause", "resume", "abort", "takeover")
        if op not in allowed:
            return _j({"ok": False, "reason": f"op 只能是 {allowed} 之一"})
        signalled = False
        if op == "abort":
            try:
                signalled = bool(svc.signal_abort(conduct_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning("conduct abort 立即置位失败（意图仍会入队）: %s", exc)
        try:
            op_id = svc.store.enqueue_op(conduct_id, op,
                                         args={"reason": reason, "by": who}, by=who)
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"入队失败: {exc}"})
        return _j({"ok": True, "conduct_id": conduct_id, "op": op,
                   "op_id": op_id, "abort_signalled": signalled,
                   "note": "已入队；Director 会在下一个 tick 消费它"})

    return [conduct_status, conduct_list, conduct_compile_from_plan,
            conduct_create, conduct_approve, conduct_post_op]
