"""ConductDirector 的**生命周期挂点** —— runtime 持有的那一份。

设计:``campaign_director_design.md`` §3-1(住在哪)。挂点形制照
``executor.start_watchdog`` / ``monitoring.service.start_service`` 的先例:
一个模块级单例 + ``start_service(runtime)`` / ``stop_service()``,由
``CoreRuntime`` 在启动时调、在 ``shutdown()`` 里停。

## 默认关

``cd_enabled`` 默认 0(见 :mod:`mast.conduct.settings`)。**关的时候这里什么
都不建** —— 不建线程、不建 store、连 SQLite 表都不建,于是「关着」逐字节等于
M1-c 之前。开关本身的两侧都有测试钉住。

## 开的时候按什么顺序

1. 建 ``ConductStore``(落在实验库里,与 PlanStore 同一个文件 —— 一份 conduct
   跟着实验记录一起导出);
2. 挂 ``observer``:实验文件夹的 ``progress.jsonl`` + WS 三种帧。**一扇门**,
   见 ``store`` 模块 docstring 第四条;
3. 装真探针(``adapters``)替下 M1-b 的诚实替身;
4. **先清算再起线程**:``reconcile_after_restart()`` 把非终态搬进
   RECOVERY_PENDING(PAUSED / DRAFT / APPROVED 原样保留)。顺序反过来的话,
   线程会在清算之前先推进一步 —— 而那一步的世界状态正是「进程刚死过一次」;
5. 起线程。

## 它不做什么

不 approve、不建 conduct、不替人做任何决定。建与批都在 API 路由上,而
**approve 只接受 UI 来源**(设计 §7)。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SERVICE_LOCK = threading.Lock()
_SERVICE: "ConductService | None" = None


class ConductService:
    """store + director + 外溢接线,一整套。

    构造**不起线程**(与 :class:`~mast.conduct.director.ConductDirector` 同一条
    纪律):起线程是 :meth:`start` 的事,而 :meth:`start` 由 runtime 挂点调。
    """

    def __init__(self, runtime, *, db_path: "str | Path",
                 folder_resolver: "Callable[[str], Path | None] | None" = None,
                 bus=None, board=None):
        from mast.conduct.adapters import (
            ConductCost, ConductNotifier, RuntimeConductCost, RuntimeExecutor,
            RuntimeFrameMetrics, RuntimeLatch, RuntimeLinkProbe,
            RuntimeMonitorEvents, RuntimeTemperature,
        )
        from mast.conduct.director import ConductDirector
        from mast.conduct.llm_seat import make_decide_route
        from mast.conduct.ports import AbortRegistry
        from mast.conduct.store import ConductStore
        from mast.conduct.templates import get_template

        self.runtime = runtime
        self._folder_resolver = folder_resolver
        self.aborts = AbortRegistry()
        self.store = ConductStore(db_path)
        self.store.set_observer(self._on_event)
        self._bus = bus

        self.notifier = ConductNotifier(
            experiment_id_getter=self._active_experiment_id, board=board, bus=bus)
        # 花销读端(M3-d)。逐币种、**不折算**、读不到回 None、**不建账本**。
        # 两个消费方各取所需:``_cost_reader`` 给 director 一个能代表全部花销的
        # USD 数(代表不了就 None),``cost_detail`` 给面板真相。
        self._cost_probe = RuntimeConductCost()
        self.director = ConductDirector(
            self.store,
            spec_provider=get_template,
            executor=RuntimeExecutor(runtime, aborts=self.aborts,
                                     conduct_id_getter=self._active_conduct_id),
            latch=RuntimeLatch(runtime),
            temperature=RuntimeTemperature(runtime),
            notifier=self.notifier,
            abort_registry=self.aborts,
            skill_meta=self._skill_meta,
            cost_reader=self._cost_reader,
            lock_probe=self._lock_probe,
            # 判决席位(M3-a):闭集路由名,**一个数都不给**。建不出模型、超时、
            # 返回值不在闭集里 —— 每一条都由 ``rules.evaluate_gate`` 报「判不了」,
            # 走闸门自己声明的保守去向,而不是在这里兜一个默认。
            decide_route=make_decide_route(context=self._decision_context),
            # 两个**外部证据**探针(M3-b)。它们看的是 conduct 自己的步产不出来
            # 的东西:监控看别人(或没人)在扫的时候电流出了什么事,帧指标看最近
            # 那一帧的质量、不管谁扫的。两个都是「读不到就回 None」。
            monitor_probe=RuntimeMonitorEvents(),
            frame_metrics_probe=RuntimeFrameMetrics(),
            # A1 连接探活(M4-a)。三态:通 / 不通 / **探不出来**。第三种既不是
            # 前两种里的哪一个,而它今天很常见(池子还没建过任何连接)。
            link_probe=RuntimeLinkProbe(runtime),
            # ``recovery_probe`` **永远不接** —— 它是 M1-b 留下的测试替身
            # (一条 TCP 都不发),接上去等于让生产走一条脚本化的假自检。
            # 生产的 A1/A2/A3 各走各的真路:A1=上面这条探针,A2=温度公共口,
            # A3=spec 自己声明的 ``RecoveryPolicy.tip_check``(经 executor.run
            # 完整安全管道,取仪器令牌)。**模板没声明 A3 就报「读不到」**,
            # 停下来问人 —— 这是刻意留着的缺席,不是忘了。
            #
            # ``verify_verdict`` 这个证据源也仍未接,而且**不是忘了**:全仓没有
            # 裁决的持久化真源。裁决要么在步产出里(那已经是 step_data,接成第二
            # 个源就是同一个事实的两个真源),要么在 S2 逐偏压账本那样的文件里 ——
            # 后者要先有一份「裁决登记」的设计。
            #
            # L2 值守席(2026-08-20)。一个阶段栽了、而模板的 ``on_fail.then``
            # 是 ``escalate`` 时,叫醒它一次:它拿一个**只读**的技能视图去看现场,
            # 回一个模板已经授权过的处置。授权来自 ``StageSpec.allowed_escalations``
            # —— 越权的建议按越权记并转人,不悄悄改写成「模型建议叫人」。
            # 建不出来时**不注入**:Director 会照实说「这台机器上没有接 L2」,
            # 而不是假装升级过了。
            escalation_advisor=self._build_escalation_advisor(runtime),
        )

    @staticmethod
    def _build_escalation_advisor(runtime):
        """L2 顾问。建不出来回 ``None`` —— 缺席要看得见,不要被兜掉。"""
        try:
            from mast.conduct.l2_seat import make_escalation_advisor

            def _registry():
                reg = getattr(runtime, "_registry", None)
                if reg is None:
                    raise RuntimeError("这个 runtime 上没有技能注册表")
                return reg

            return make_escalation_advisor(registry_provider=_registry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("L2 诊断席建不出来(escalate 策略会退回问人): %s", exc)
            return None

    # ── 外溢(一扇门)──────────────────────────────────────────────

    def _on_event(self, ev: dict) -> None:
        """一条已提交的事件 → 实验文件夹一行 + WS 一帧。"""
        self._append_progress(ev)
        self._publish_frame(ev)

    def journal(self, conduct_id: str, experiment_id: str = ""):
        from mast.conduct.journal import ConductJournal

        if not experiment_id:
            row = self.store.get(conduct_id) or {}
            experiment_id = str(row.get("experiment_id") or "")
        return ConductJournal(conduct_id, experiment_id,
                               folder_resolver=self._folder_resolver)

    def _append_progress(self, ev: dict) -> None:
        try:
            j = self.journal(str(ev.get("conduct_id") or ""))
            j.append(kind=str(ev.get("kind") or ""), ts=str(ev.get("ts") or ""),
                     status_after=str(ev.get("status_after") or ""),
                     stage_id=str(ev.get("stage_id") or ""),
                     step_id=str(ev.get("step_id") or ""),
                     run_id=str(ev.get("run_id") or ""),
                     payload=dict(ev.get("payload") or {}))
        except Exception as exc:  # noqa: BLE001
            logger.debug("conduct progress 追加跳过: %s", exc)

    def _publish_frame(self, ev: dict) -> None:
        """状态帧。``gate_evaluated`` 单独一种;其余都是 status 帧。

        **帧不带面板状态**(设计 §7):前端收帧后 refetch ``GET /api/conducts/{id}``。
        告警帧不在这里发 —— 它由 notifier 在真正要惊动人的时候发,两者的收件人
        不同(一个是开着面板的人,一个是不在面板前的人)。
        """
        try:
            if self._bus is not None:
                bus = self._bus
            else:
                from mast.core.events import EventBus
                bus = EventBus.get()
            cid = str(ev.get("conduct_id") or "")
            kind = str(ev.get("kind") or "")
            if kind == "gate_evaluated":
                p = dict(ev.get("payload") or {})
                bus.publish_conduct_gate(
                    cid, gate_id=str(p.get("gate_id") or ""),
                    verdict=str(p.get("verdict") or ""),
                    stage_id=str(ev.get("stage_id") or ""))
                return
            bus.publish_conduct_status(
                cid, kind=kind, status=str(ev.get("status_after") or ""),
                status_changed=bool(ev.get("status_changed")),
                stage_id=str(ev.get("stage_id") or ""),
                step_id=str(ev.get("step_id") or ""))
        except Exception as exc:  # noqa: BLE001
            logger.debug("conduct 状态帧跳过: %s", exc)

    # ── 注入给 Director 的小探针 ──────────────────────────────────

    def _active_conduct_id(self) -> str:
        row = self.store.active() or {}
        return str(row.get("conduct_id") or "")

    def _active_experiment_id(self) -> str:
        row = self.store.active() or {}
        return str(row.get("experiment_id") or "")

    def _decision_context(self) -> dict:
        """判决日志那一行的随手快照 —— **只进日志,不进提示词**。

        闸门问什么由 spec 的 ``llm_node.responsibility`` 定死;运行时上下文
        改写不了那个问题。这里给的只是「事后要查这条判决时,得知道它属于谁」。
        读不到就空着:一条少了 conduct_id 的审计仍然比没有审计好。
        """
        row = self.store.active() or {}
        return {"conduct_id": str(row.get("conduct_id") or ""),
                "experiment_id": str(row.get("experiment_id") or ""),
                "spec_id": str(row.get("spec_id") or ""),
                "stage_idx": row.get("stage_idx"),
                "step_idx": row.get("step_idx"),
                "attended": bool(row.get("attended")),
                "evidence_epoch": row.get("evidence_epoch")}

    def _skill_meta(self, name: str):
        """注册表里的技能元数据 —— SAFE 模式对账要它。查不到回 ``None``
        (director 会把「注册表里没有」当成一次**停下来问人**,不是跳过)。"""
        reg = (getattr(self.runtime, "_registry", None)
               or getattr(self.runtime, "skill_registry", None))
        if reg is None:
            return None
        try:
            return reg._get_metadata(reg.get(name))
        except Exception:  # noqa: BLE001
            return None

    def cost_detail(self, conduct_id: str):
        """这份 conduct 的实测花销,**逐币种**(``adapters.ConductCost``)。

        ``None`` = 读不到(账本文件不在 / 读失败)。面板走这一口 —— 它要显示的是
        真相:哪个币种花了多少、为什么合不成一个数。
        """
        return self._cost_probe(conduct_id)

    def _cost_reader(self, conduct_id: str) -> "float | None":
        """这份 conduct 到目前为止的实测花销,**USD,而且必须是完整的 USD**。

        Director 的预算闸拿它和 ``ConductBudget.usd_max`` 比大小,所以这里只有
        两种诚实的回答:一个**能代表全部花销**的 USD 数,或者 ``None``。

        ## 为什么跨币种时回 None 而不是「USD 那一部分」

        账本按 provider 原生币种记账,用哪家由调用时的回退链决定 —— 一份 conduct
        天然可能同时有 CNY 和 USD 两笔。把 USD 那一笔单独交出去,预算闸会拿它去比
        上限、判「没超」,而真正花掉的大头在另一个币种里。**那是一次「检查通过」
        形态的漏报**,比不检查坏得多:director 会把它记成查过了。

        合并两者需要汇率,而 ``usd_to_cny_rate()`` 无覆盖文件时是写死的 7.2,账本
        自己把折算总额标成「never written to the ledger」。所以这里**不折算**,
        如实回 None ⇒ director 记「读不到花销(读不到 ≠ 花了 0)」。

        ⇒ 面板要看真相走 :meth:`cost_detail`;要让这条上限真的能拦,得先裁一次
        计价单位(设计 §11-9:逐币种上限,不接汇率)。
        """
        got = self._cost_probe(conduct_id)
        if got is None:
            return None
        by = dict(getattr(got, "by_currency", None) or {})
        if list(by) == ["USD"]:
            return float(by["USD"])
        # 一条都没有 ⇒ 这份 conduct 确实还没花过钱,那是一个**答得上来的 0**。
        if not by:
            return 0.0
        return None

    def _lock_probe(self) -> bool:
        """仪器令牌现在空着吗。读不到 ⇒ ``False``(当成占着),
        保守方向:让路比抢先安全。"""
        try:
            from mast.core.instrument_lock import instrument_lock

            return instrument_lock().snapshot() is None
        except Exception:  # noqa: BLE001
            return False

    # ── 生命周期 ──────────────────────────────────────────────────

    def start(self) -> None:
        """先清算,再起线程。"""
        try:
            rep = self.director.reconcile_after_restart()
            if rep.actions:
                logger.info("conduct 重启清算: %s", ", ".join(rep.actions))
        except Exception as exc:  # noqa: BLE001
            logger.warning("conduct 重启清算失败(不起线程): %s", exc)
            return
        self.director.start()
        logger.info("ConductDirector started (常规驱动,每步走 executor 完整安全管道)")

    def stop(self, timeout_s: float = 5.0) -> None:
        self.director.stop(timeout_s=timeout_s)

    @property
    def is_running(self) -> bool:
        return bool(self.director.is_running)

    # ── API 线程用的那一条例外通道 ────────────────────────────────

    def signal_abort(self, conduct_id: str) -> bool:
        """**立即**置这份 conduct 当前步的 abort Event。返回是否真置到了一个。

        设计 §4.7 写序纪律的**唯一例外**:abort 意图落表的同时,API 线程不等
        tick 就置位。Director 卡在一次 ``executor.run`` 里时 tick 不会来,而
        abort 按钮必须还能用。
        """
        row = self.store.get(conduct_id) or {}
        run_id = str(row.get("active_run_id") or "")
        if not run_id:
            return False
        return bool(self.aborts.signal(run_id))


# ── 模块级单例(挂点)──────────────────────────────────────────────

def default_db_path(runtime) -> Path:
    """conduct 库落在哪 —— **实验库那个文件**。

    与 ``PlanStore`` 同一个取舍:一份 conduct 的状态跟着实验记录一起导出、
    一起备份。``ConductStore`` 建的是自己的三张表,与实验记录的表不重名。
    """
    p = getattr(getattr(runtime, "config", None), "db_path", None)
    if p:
        return Path(p)
    from mast.agents._shared.data_paths import experiment_db_path

    return Path(experiment_db_path())


def start_service(runtime, **kw) -> "ConductService | None":
    """建(若需要)并起 Director。**永不抛** —— conduct 起不来不该拦住启动。

    ``cd_enabled`` 关着就什么都不做并返回 ``None``。
    """
    global _SERVICE
    try:
        from mast.conduct.settings import get_conduct_knobs

        if not get_conduct_knobs().enabled:
            logger.info("ConductDirector 未启用(conduct.cd_enabled=0)—— 不建线程")
            return None
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = ConductService(
                    runtime, db_path=kw.pop("db_path", None) or default_db_path(runtime),
                    **kw)
            svc = _SERVICE
        if not svc.is_running:
            svc.start()
        return svc
    except Exception:  # noqa: BLE001 —— conduct 绝不能拦住 runtime 起来
        logger.warning("ConductDirector 没能启动", exc_info=True)
        return None


def stop_service(join_timeout: float = 5.0, *, drop: bool = False) -> None:
    """停线程。``drop=True`` 连单例一起丢(shutdown 用)。永不抛。"""
    global _SERVICE
    try:
        svc = _SERVICE
        if svc is not None:
            svc.stop(timeout_s=join_timeout)
        if drop:
            with _SERVICE_LOCK:
                _SERVICE = None
    except Exception:  # noqa: BLE001
        logger.debug("conduct 停止失败(已吞)", exc_info=True)


def get_service() -> "ConductService | None":
    """当前那一份(没起就是 ``None``)。API 层据此报「引擎未启用」。"""
    return _SERVICE


def set_service_for_test(svc: "ConductService | None") -> None:
    global _SERVICE
    with _SERVICE_LOCK:
        _SERVICE = svc


def apply_settings(runtime=None) -> "ConductService | None":
    """设置写入之后调一次:开关翻了就真的起/停。

    没有这个函数的话,用户把开关关掉、以为线程停了,而它还在驱动仪器 ——
    一个「按了没反应」的开关比没有开关更危险。
    """
    from mast.conduct.settings import get_conduct_knobs

    enabled = get_conduct_knobs().enabled
    svc = get_service()
    if not enabled:
        if svc is not None:
            logger.warning("conduct.cd_enabled 被关掉 —— 停 Director 线程")
            stop_service()
        return None
    if svc is not None:
        if not svc.is_running:
            svc.start()
        return svc
    if runtime is None:
        # 开关开了但手上没有 runtime(纯设置写入路径)——**如实记一条**:
        # 下一次进程启动会把它起来,而不是假装已经起来了。
        logger.info("conduct.cd_enabled 打开 —— 下次启动时生效"
                    "(本次写入路径没有 runtime 句柄)")
        return None
    return start_service(runtime)


__all__ = ["ConductService", "start_service", "stop_service", "get_service",
           "set_service_for_test", "apply_settings", "default_db_path"]
