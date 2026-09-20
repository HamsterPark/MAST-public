"""真探针 —— :mod:`mast.conduct.ports` 那四个注入口的**生产实现**。

设计:``campaign_director_design.md`` §3-1/2(住在哪 / 仲裁)、§5(等人通道)、
§7(WS 状态帧)。M1-b 留下的诚实替身(``NullLatch`` / ``NullTemperature`` /
``LoggingNotifier``)**继续保留**:测试仍然用它们,而且一个没接线的部署会因此
如实说「读不到」,不会假装一切正常。

## 四个口分别接到哪

======================  ==========================================================
``ExecutorPort``        ``ExecutionContext.run`` —— 与技能直调 API、agent 工具
                        边界、群聊**同一条**执行路径(第 5 个受 ``instrument_lock``
                        仲裁的入口)。Director **永远没有裸 TCP 权**。
``latch``               ``runtime.emergency_latch_state()`` —— 挂没挂 + abort 位
                        + **为什么**。三个字段一起给,少一个就得靠人猜。
``temperature``         ``runtime.latest_temperature()`` —— 修复项 的公共只读口,
                        返回 ``TempReading``(值 + age_s + 为什么没有值)。
``notifier``            EventBus(照 watchdog ``publish_anomaly`` 的先例)+ 心愿单
                        ``post_agent_request``(等人**绝不走 hitl_bridge**)。
======================  ==========================================================

## 一个不显然的映射:busy ≠ 失败

``ExecutionContext.run`` 把 ``InstrumentBusy`` **吞成一个失败的 SkillResult**
(它必须这么做:agent 那条路要的是一句能读的话)。而对 Director 来说这两件事
必须分开 —— busy 是「下一 tick 再试」,失败是「吃掉一次重试预算」。把两者折叠,
一次正常的并发仲裁就会消耗掉这一步的重试次数,连着几次之后步就「失败」了,
而仪器其实一次都没动过。

所以 ``execution_context`` 在那条分支上留了一个**结构化标记**
``data[INSTRUMENT_BUSY_KEY]``,这里按它判。**不按错误文案判**:错误文案是给人
读的,改一个字就会让这里静默失灵(本仓为手工维护的名字清单付过至少四次学费)。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from mast.conduct.ports import LatchState, Notification, StepOutcome

logger = logging.getLogger(__name__)

#: ``SkillResult.data`` 里那个「这一步被仪器令牌拒了」的键。**单一真源**:
#: 写在 ``core/execution_context.py`` 的 ``InstrumentBusy`` 分支,读在这里。
INSTRUMENT_BUSY_KEY = "instrument_busy"

#: conduct 在仪器令牌里的 owner 前缀。``InstrumentBusy`` 的消息会把它原样
#: 显示给被拒的那一方 —— 于是「谁占着仪器」这个问题有一个具体的答案,
#: 而不是「另一个入口」。
OWNER_PREFIX = "conduct:"

#: 心愿单里 conduct 请求的 agent_id 前缀(``conduct:<id>``)。
WISHLIST_AGENT_PREFIX = OWNER_PREFIX


# ── 执行体 ──────────────────────────────────────────────────────────

class RuntimeExecutor:
    """把一步跑成 ``ExecutionContext.run``。

    ## 为什么每一步新建一个 ExecutionContext

    ``run_id`` 是 ExecutionContext 的构造参数,而 composite 的进度 sidecar 按
    ``(name, run_id)`` 分键 —— 复用一个 ctx 就等于让所有步共享一个 run_id,
    于是「一次跑完的 AutoApproach 让后面每一次进针都短路成假成功」那个洞会
    原样回来。构造成本只是几个字段赋值。

    ## abort 事件是**并集**

    ``[全局急停闩事件, 这一步的 per-run Event]``:E-STOP 停得住它(全局那个),
    而面板上的 abort 按钮也停得住它(per-run 那个,由 API 线程**不经 tick**
    直接置位)。少任何一个都会出现「按了没反应」。
    """

    def __init__(self, runtime, *, aborts, conduct_id_getter: "Callable[[], str]"):
        self._runtime = runtime
        self._aborts = aborts
        self._cid = conduct_id_getter

    # -- 装配 ------------------------------------------------------------

    def _pieces(self):
        rt = self._runtime
        pool = getattr(rt, "_pool", None) or getattr(rt, "connection_pool", None)
        state = getattr(rt, "_state", None) or getattr(rt, "instrument_state", None)
        registry = getattr(rt, "_registry", None) or getattr(rt, "skill_registry", None)
        missing = [n for n, v in (("connection_pool", pool), ("state", state),
                                  ("skill_registry", registry)) if v is None]
        return pool, state, registry, missing

    def _context(self, run_id: str):
        """→ ``(ctx, error)``。缺东西时**说缺什么**,不报成「技能失败」。"""
        pool, state, registry, missing = self._pieces()
        if missing:
            return None, f"执行环境不可用,缺少: {missing}"
        aborts = []
        glob = getattr(self._runtime, "_orch_abort", None)
        if glob is not None:
            aborts.append(glob)
        per_run = self._aborts.get(run_id) if hasattr(self._aborts, "get") else None
        if per_run is None:
            # Director 在启动这一步之前就注册过了;取不到说明注册表被换过 ——
            # 如实记一条,并**照样跑**(全局急停仍然停得住它)。
            logger.warning("conduct: run %s 没有 per-run abort Event —— "
                           "面板 abort 这一步会退化成「下一 tick 才生效」", run_id)
        else:
            aborts.append(per_run)
        try:
            from mast.core.execution_context import ExecutionContext

            ctx = ExecutionContext(pool=pool, state=state, registry=registry,
                                   abort_event=aborts or None, run_id=run_id,
                                   owner=f"{OWNER_PREFIX}{self._cid()}")
        except Exception as exc:  # noqa: BLE001
            return None, f"ExecutionContext 构建失败: {type(exc).__name__}: {exc}"
        # composite 子步骤的地图标记 —— 与三条既有路径同一个记录器。没有它,
        # conduct 打出的每一发脉冲在扫描地图上都不存在。
        try:
            from mast.core.runtime import _attach_marker_sink

            _attach_marker_sink(ctx, self._runtime)
        except Exception as exc:  # noqa: BLE001 — 记地图绝不影响执行
            logger.debug("conduct marker sink 未接上: %s", exc)
        return ctx, ""

    # -- ExecutorPort ----------------------------------------------------

    def run(self, skill: str, params: dict, *, run_id: str) -> StepOutcome:
        ctx, err = self._context(run_id)
        if ctx is None:
            return StepOutcome(ok=False, error=err, run_id=run_id)
        try:
            result = ctx.run(skill, dict(params or {}))
        except Exception as exc:  # noqa: BLE001 —— 技能抛异常是步失败,不是崩溃
            return StepOutcome(ok=False, run_id=run_id,
                               error=f"{type(exc).__name__}: {exc}")
        return self.to_outcome(result, run_id=run_id)

    @staticmethod
    def to_outcome(result: Any, *, run_id: str) -> StepOutcome:
        """``SkillResult`` → ``StepOutcome``。**busy 与失败分开**(见模块 docstring)。"""
        data = getattr(result, "data", None)
        data = dict(data) if isinstance(data, dict) else {}
        busy = bool(data.get(INSTRUMENT_BUSY_KEY))
        ok = bool(getattr(result, "success", False))
        # 针尖事件/撞针:技能自己报的那两个字段。**读不到就是没报**,
        # 这里不去猜 —— 真正拦住 conduct 的是急停闩(E_STOP 必挂闩),
        # 这两个位只是让审计流里看得见「那一步是怎么坏的」。
        tip_event = bool(data.get("tip_event") or data.get("tip_changed"))
        crash = bool(data.get("crash_indicator") or data.get("tip_crash"))
        return StepOutcome(ok=ok and not busy, busy=busy, tip_event=tip_event,
                           crash=crash, run_id=run_id,
                           error=str(getattr(result, "error", "") or ""),
                           data=_jsonable(data))


def _jsonable(obj: Any, _depth: int = 0) -> Any:
    """把技能 data 压成能进 SQLite JSON 的东西。**永不抛。**

    与 ``api/routes/skill_exec.py`` 的同名函数同一个理由:波形数组整条塞进
    审计流会把库撑爆,``bytes`` 会让序列化直接炸掉。这里的产物要进
    ``conduct_events.payload_json``,所以同样的裁剪必须在这一侧也做一遍。
    """
    if _depth > 6:
        return "<嵌套过深,已截断>"
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        import math
        return obj if math.isfinite(obj) else f"<非有限值 {obj!r}>"
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<bytes len={len(bytes(obj))}>"
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        if len(seq) > 512:
            return {"_truncated": True, "len": len(seq),
                    "head": _jsonable(seq[:8], _depth + 1),
                    "tail": _jsonable(seq[-8:], _depth + 1)}
        return [_jsonable(v, _depth + 1) for v in seq]
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return _jsonable(obj.item(), _depth + 1)
        if isinstance(obj, _np.ndarray):
            return _jsonable(obj.tolist(), _depth + 1)
    except Exception:  # noqa: BLE001
        pass
    return repr(obj)[:400]


# ── 急停闩 ──────────────────────────────────────────────────────────

class RuntimeLatch:
    """读 ``runtime.emergency_latch_state()``。

    Director 与闩是**消费者关系**:查它、挂着就停,**从不清它**。清闩的路在
    ``POST /api/safety/clear-emergency`` 上,那是人的动作。
    """

    def __init__(self, runtime):
        self._runtime = runtime

    def state(self) -> LatchState:
        fn = getattr(self._runtime, "emergency_latch_state", None)
        if not callable(fn):
            # 接不上闩 ⇒ **抛**,而不是回一个「没挂」。``_latch_state`` 那边
            # 会把异常记成一次读失败;回「没挂」会让一个读不到闩的部署看起来
            # 像一个闩没挂的部署。
            raise RuntimeError("runtime 没有 emergency_latch_state —— 闩读不到")
        st = fn() or {}
        return LatchState(latched=bool(st.get("latched")),
                          abort_set=bool(st.get("abort_set")),
                          why=str(st.get("why") or ""))


# ── 温度(修复项 公共口)───────────────────────────────────────────────

class RuntimeTemperature:
    """读 ``runtime.latest_temperature()`` —— 只读、不取令牌、永不抛。

    返回的 ``TempReading`` 自带 ``freshness()`` 三态,Director 据此把
    「读不到 / 太旧」判成**判不了**而不是「条件不满足」。
    """

    def __init__(self, runtime):
        self._runtime = runtime

    def read(self, channel: str = ""):
        from mast.core.temperature import NO_SOURCE, TempReading

        fn = getattr(self._runtime, "latest_temperature", None)
        if not callable(fn):
            return TempReading(channel=channel, reason=NO_SOURCE)
        try:
            return fn(channel or None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("conduct 温度读取失败: %s", exc)
            return TempReading(channel=channel, reason=NO_SOURCE)


# ── A1 连接探活(恢复自检)────────────────────────────────────────────

class RuntimeLinkProbe:
    """各 role 只读探活 —— 恢复自检 A1(设计 §8)。

    ``() -> {role: True | False | None}``,三态,**``None`` 是「探不出来」**。
    这三者要人做的事不一样:

    * ``True``  —— 通;
    * ``False`` —— 不通 ⇒ 去看 Nanonis 开着没有、NI Service Locator 启用没有。
      (netstat 会误导:端口 LISTEN 着而 Service Locator 停了 ⇒ TCP 全失败。)
    * ``None``  —— **这一条根本没探成**(连接池取不到、role 名字都不知道)⇒
      去看这台机器上的接线,而不是去看仪器。

    探的是 ``ConnectionPool.health_check``(``Util_VersionGet``,只读、不取
    仪器令牌)。**不自己发 TCP**:Director 永远没有裸 TCP 权,而这条探针连
    「读一个值」都要走池子已有的熔断与 role 锁 —— 一次绕过它们的探活会在
    Nanonis 端口上留下半截事务。
    """

    def __init__(self, runtime, *, roles: "tuple[str, ...] | None" = None):
        self._runtime = runtime
        self._roles = tuple(roles) if roles else ()

    def _pool(self):
        rt = self._runtime
        return getattr(rt, "_pool", None) or getattr(rt, "connection_pool", None)

    def __call__(self) -> dict:
        pool = self._pool()
        if pool is None:
            return {}                      # 一个 role 都报不出来 ⇒ 上层判「读不到」
        roles = self._roles or tuple(getattr(pool, "_connections", {}) or ())
        if not roles:
            # 池子在,但一条连接都没建过。**这不是「都不通」** —— 是「没得探」。
            return {}
        out: dict = {}
        for role in roles:
            try:
                out[role] = bool(pool.health_check(role))
            except Exception as exc:  # noqa: BLE001
                logger.debug("conduct A1 探活 role=%s 失败: %s", role, exc)
                out[role] = None           # 探不出来 ≠ 不通
        return out


# ── 通知 / 心愿单 ───────────────────────────────────────────────────

#: 通知级别 → WS ``conduct_alert`` 的 severity。闭集,别在别处再拼一份。
SEVERITIES = ("info", "warn", "crit")

#: 哪些通知种类要**同时**进心愿单(要人真的去做一件事),哪些只是播报。
#: 「等人」那两条走心愿单;其余是 EventBus 上的告警条。
WISHLIST_KINDS = frozenset({"wait", "wait_renotify"})


class ConductNotifier:
    """EventBus 告警帧 + 心愿单请求。

    ## 两条通道答的是两个问题

    * **EventBus** —— 「现在怎么样」:面板收到帧就 refetch,帧本身不携带状态
      (设计 §7:帧只做触发,不做增量状态源)。人不在场时它没有收件人。
    * **心愿单** —— 「有件事等着人办」:一条持久的、有 open/done 状态的记录。
      等人**绝不走 hitl_bridge**(进程本地、重启即丢、900 s fail-closed —— 三条
      性质对一个可能等一夜的换样品请求全是错的)。

    ## 幂等是靠 id,不是靠猜

    ``request_operator_action`` 返回心愿单请求 id,Director 把它记进
    ``active_wait``。重播(每 4 h)与**重启后重发**都先拿这个 id 问一句
    「那条还开着吗」——开着就不再发一条,只播一帧。心愿单自己也按
    ``(agent_id, message)`` 去重,但 renotify 的文案里带着「还缺什么」会变,
    所以不能只靠它。
    """

    def __init__(self, *, experiment_id_getter: "Callable[[], str] | None" = None,
                 board=None, bus=None):
        self._experiment_id = experiment_id_getter
        self._board = board          # 注入用(测试);None = 用全局心愿单
        self._bus = bus              # 注入用(测试);None = 用 EventBus 单例
        #: 发出去的每一条(排障用:「通知到底发了没有」必须可问)。
        self.sent: list[Notification] = []

    # -- 通道 ------------------------------------------------------------

    def _publish(self, note: Notification) -> None:
        """一帧 ``conduct_alert``。发不出去只记日志,**绝不改变状态机**。"""
        try:
            if self._bus is not None:
                bus = self._bus
            else:
                from mast.core.events import EventBus
                bus = EventBus.get()
            bus.publish_conduct_alert(
                note.conduct_id, code=note.kind,
                severity=note.severity if note.severity in SEVERITIES else "info",
                message=note.message, payload=dict(note.payload or {}))
        except Exception as exc:  # noqa: BLE001
            logger.warning("conduct 告警帧发送失败(状态照旧): %s", exc)

    def _post_wish(self, note: Notification) -> str:
        prev = str((note.payload or {}).get("request_id") or "")
        try:
            if self._board is not None:
                board = self._board
                post = board.post_agent_request
                get_req = board.get_request
            else:
                from mast.wishlist.store import get_request, post_agent_request
                post, get_req = post_agent_request, get_request
            if prev:
                existing = get_req(prev) or {}
                if str(existing.get("status") or "") == "pending":
                    # 还开着 —— 不再发一条。重播只是把告警帧再打一遍。
                    return prev
            exp = ""
            if callable(self._experiment_id):
                try:
                    exp = str(self._experiment_id() or "")
                except Exception:  # noqa: BLE001
                    exp = ""
            rec = post(f"{WISHLIST_AGENT_PREFIX}{note.conduct_id}", note.message,
                       kind="action", experiment_id=exp or None)
            return str((rec or {}).get("id") or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("conduct 心愿单请求失败(状态照旧): %s", exc)
            return prev

    # -- 端口 ------------------------------------------------------------

    def notify(self, note: Notification) -> None:
        self.sent.append(note)
        logger.info("[conduct %s] %s: %s", note.conduct_id, note.kind,
                    note.message)
        self._publish(note)

    def request_operator_action(self, note: Notification) -> str:
        """要人办的事 → 心愿单 + 告警帧。返回心愿单请求 id(没有就空串)。"""
        self.notify(note)
        if note.kind not in WISHLIST_KINDS:
            return ""
        return self._post_wish(note)


# ── 外部证据探针(M3-b)────────────────────────────────────────────
#
# 两个都是 ``(selector, since_epoch_s) -> dict | None``,而 ``None`` **只有一个
# 意思:读不到**。不是「没有告警」,不是「没有帧」——那两个是 ``0``,是答得上来
# 的答案。Director 拿到 None 就把这条证据报成缺席,闸门走它自己声明的保守去向。
#
# 为什么这两个源值得接:它们看的是 conduct **自己的步没产出**的东西 ——
# 监控看的是别人(或没人)在扫的时候电流出了什么事,帧指标看的是最近 N 帧的质量,
# 无论那些帧是谁扫的。``step_data`` 答不了这两问。


class RuntimeMonitorEvents:
    """``monitor_events`` —— 隧道电流监控的告警。

    ## 三态,不是两态

    ================================  ==================================
    监控守护线程没在跑 / 从没跑过      ``None``(**读不到**)
    在跑,窗口里没有告警                ``{"monitor_alert_count": 0, ...}``
    在跑,窗口里有告警                  计数 + 级别 + 规则名
    ================================  ==================================

    第一行与第二行**必须分开**:一个停着的监控守护线程报出来的「零条告警」,
    说的是「没人在看」,不是「这段时间很太平」。而对「要不要接着扫一整夜」这个
    问题,那两句话给的是相反的答案。

    ## 第三道:查询自己炸了(2026-08-15 关上)

    前面两道检查(守护线程在跑 + store 存在)挡不住「store 在、查询炸了」。
    在此之前 ``CurrentMonitorStore.alerts_query`` 自吞异常回 ``([], 0)``,于是
    一次查询失败在这里长得和「零条告警」一模一样 —— 而 ``0`` 是一个**计数**,
    是一句正面断言。现在它抛 ``StoreQueryFailed``,下面那条 except 把它接成
    ``None``:**查询失败 ⇒ 判不了,不是 0。**

    这条 except 保持 ``Exception`` 而不是收窄到 ``StoreQueryFailed``:注入的假
    store(测试)、以及将来换一种存储实现,炸出来的都不会是这个类型;而这里要问
    的问题只有一个「读到了没有」。

    ## 为什么用 ``get_store_if_exists``

    ``get_store()`` 会**建库**(``experiments/current_monitor/``)。一个被动读者
    建出来的空库按构造就没有东西可读,而且在测试里那是往用户的真实实验目录里
    写 —— 那个函数的 docstring 自己点名了这条。conduct 是被动读者。
    """

    #: 告警级别,从轻到重。``selector`` 给谁,就只数那一级**及以上**。
    LEVELS = ("info", "warn", "critical")

    def __init__(self, *, service_getter=None, store_getter=None,
                 limit: int = 200):
        self._service_getter = service_getter
        self._store_getter = store_getter
        self._limit = int(limit)

    def _service(self):
        if self._service_getter is not None:
            return self._service_getter()
        from mast.monitoring.service import get_service

        return get_service()

    def _store(self):
        if self._store_getter is not None:
            return self._store_getter()
        from mast.monitoring.store import get_store_if_exists

        return get_store_if_exists()

    def __call__(self, selector: str, since: float) -> "dict | None":
        svc = self._service()
        if svc is None:
            return None
        try:
            running = bool((svc.status() or {}).get("running"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("monitor status 读不到: %s", exc)
            return None
        if not running:
            # 停着的监控报不出「这段时间很太平」——它只报得出「我没在看」。
            return None
        store = self._store()
        if store is None:
            return None
        try:
            rows, _total = store.alerts_query(since=float(since), limit=self._limit)
        except Exception as exc:  # noqa: BLE001 —— 读不到 ≠ 没有,见类 docstring 第三道
            # **warning 而不是 debug**:这条一响,闸门就要转人,而转人的理由在
            # 面板上只剩一句「读不到」——``probe -> dict | None`` 这个口没有带
            # 原因的槽位。默认日志级别下 debug 不落盘 ⇒ 用户半夜被叫醒,
            # 而「为什么」谁也答不出来。能停不能解释,是本仓记过账的那种停。
            logger.warning("conduct monitor_events 读不到(闸门将按缺证据处理): %s",
                           exc)
            return None
        floor = self._floor_index(selector)
        kept = [r for r in (rows or [])
                if self._level_index(r.get("level")) >= floor]
        levels = [str(r.get("level") or "") for r in kept]
        return {
            "monitor_alert_count": len(kept),
            "monitor_crit_count": sum(1 for lv in levels if lv == "critical"),
            # tuple 而不是 list:``RuleLeaf(op="in")`` 要的是集合,而 spec 的
            # ``in`` 会把右侧转成 tuple —— 两边同一种形状,少一次「为什么没匹配」。
            "monitor_rules": tuple(sorted({str(r.get("rule") or "") for r in kept})),
            "monitor_worst_level": (
                max(levels, key=self._level_index) if levels else "none"),
            "monitor_window_s": max(0.0, float(_now()) - float(since)),
        }

    def _level_index(self, level) -> int:
        try:
            return self.LEVELS.index(str(level or "").lower())
        except ValueError:
            # 不认识的级别**不当成最低**:一个新加的 "fatal" 被当成 info 静默
            # 滤掉,正是「闸门看不见它本该看见的东西」。当成最高。
            return len(self.LEVELS)

    def _floor_index(self, selector: str) -> int:
        s = str(selector or "").strip().lower()
        if not s:
            return 0
        try:
            return self.LEVELS.index(s)
        except ValueError:
            return 0


class RuntimeFrameMetrics:
    """``frame_metrics`` —— **最近那一帧**的质量指标,不管它是谁扫的。

    这正是 ``step_data`` 答不了的问题:step_data 只看得见 conduct 自己那一步的
    产出,而「刚才那张图行相关掉到多少」可能来自一次手动扫描、一次 agent 驱动的
    扫描,或者上一段留下的最后一帧。

    ## 只量一帧,不量 N 帧

    设计里写的是「近 N 帧 metrics」。这里**只做最新一帧**,理由是诚实的成本账:
    ``measure_frame`` 要把 .sxm 读进来跑 numpy,而调用它的是**指挥线程** ——
    多量一帧就多停一次 tick,而急停与暂停只在步边界生效。N 帧要的是一个后台
    预计算的指标表,那是另一件事;在它存在之前,一帧是能诚实付得起的那个数。

    ``selector`` 是通道名(空 = ``Z``)。
    """

    def __init__(self, *, list_scans=None, loader=None, measure=None):
        self._list_scans = list_scans
        self._loader = loader
        self._measure = measure

    def _scans(self, n: int = 20) -> list:
        if self._list_scans is not None:
            return list(self._list_scans(n) or [])
        from mast.core.scan_registry import list_scans

        return list(list_scans(n) or [])

    def __call__(self, selector: str, since: float) -> "dict | None":
        channel = str(selector or "").strip() or "Z"
        try:
            scans = self._scans()
        except Exception as exc:  # noqa: BLE001
            logger.debug("scan_registry 读不到: %s", exc)
            return None
        newest = None
        for rec in scans:                    # 已经是最新在前
            ts = _record_time(rec)
            if ts is None:
                # 时刻答不出来 ⇒ **代次归属答不出来** ⇒ 这一条不能用。
                # 不是「那就当它是新的」—— 那正是把旧帧喂进闸门的那条路。
                continue
            if ts >= float(since):
                newest = rec
                break
        if newest is None:
            # **注意这不是「读不到」**:登记表读到了,只是里面没有一帧属于当前
            # 代次、且在年龄窗口内。那是一个答得上来的答案,而它的意思是
            # 「这一代还没有帧」—— 闸门要的正是这句话,所以照实给 0。
            return {"frame_count": 0, "frame_channel": channel,
                    "frame_scan_id": "", "frame_age_s": None}
        return self._measure_one(newest, channel, since)

    def _measure_one(self, rec: dict, channel: str, since: float) -> "dict | None":
        path = str(rec.get("path") or "")
        if not path:
            return None
        try:
            fr = self._load(path, channel)
        except Exception as exc:  # noqa: BLE001
            logger.debug("帧读不动 %s: %s", path, exc)
            return None
        if fr is None or fr.get("forward") is None:
            # 这个文件里没有这个通道 —— **读不到**这一帧的这一项,不是「指标是 0」。
            return None
        try:
            m = self._run_measure(fr)
        except Exception as exc:  # noqa: BLE001
            logger.debug("measure_frame 失败 %s: %s", path, exc)
            return None
        ts = _record_time(rec)
        return {
            "frame_count": 1,
            "frame_channel": channel,
            "frame_scan_id": str(rec.get("scan_id") or ""),
            "frame_age_s": (None if ts is None else max(0.0, float(_now()) - ts)),
            # 三个都可能是 NaN(``measure_frame`` 用 NaN 表示「没算出来」)。
            # **NaN 原样带出去**:rule 的比较拿 NaN 一律为假,而 spec 的三态求值
            # 会把它读成「不成立」…… 所以这里把 NaN 换成 None,让 lookup 拿到一个
            # 「有这个键但没有值」的读数,与「这一项根本不在证据包里」区分开。
            "frame_rowcorr_median": _finite_or_none(getattr(m, "rowcorr_median", None)),
            "frame_fb_instability": _finite_or_none(getattr(m, "fb_instability", None)),
            "frame_periodic_snr": _finite_or_none(getattr(m, "fine_periodic_snr", None)),
        }

    def _load(self, path: str, channel: str):
        if self._loader is not None:
            return self._loader(path, channel)
        from mast.io.nanonis_files import read_sxm, sxm_oriented_frames

        return sxm_oriented_frames(read_sxm(path), channel)

    def _run_measure(self, fr: dict):
        if self._measure is not None:
            return self._measure(fr)
        from mast.vision.scan_prep import measure_frame

        return measure_frame(fr["forward"], bwd=fr.get("backward"),
                             nm_per_px=fr.get("nm_per_px"))


class ConductCost:
    """一份 conduct 的**实测**花销 —— 逐币种,**没有合计**。

    ## 为什么没有合计

    账本按 provider 的**原生币种**记账(``billing/pricing.py`` 第 3 行:中国
    provider 记 CNY,Anthropic 记 USD),而 conduct 默认判决模型是
    ``kimi-k3`` ⇒ **CNY**。要合成一个数就得有汇率,而 ``usd_to_cny_rate()``
    在没有覆盖文件时返回**写死的 7.2**,账本自己把折算总额标成
    「labelled 折算 …never written to the ledger」。

    所以这里给的是逐币种的实测值 + 一句「能不能合成」。**合不成就说合不成** ——
    在这里挑一个汇率乘出去,会得到一个看起来精确、实际是编的数字,而它会被拿去
    和 ``usd_max`` 比大小,然后决定要不要停掉一个跑了六小时的实验。
    """

    __slots__ = ("by_currency", "count", "all_priced", "reason")

    def __init__(self, by_currency: dict, count: int, all_priced: bool,
                 reason: str = ""):
        #: ``{"CNY": 1.23, "USD": 0.04}`` —— 实测,逐币种。
        self.by_currency = dict(by_currency)
        #: 记了几条调用。
        self.count = int(count)
        #: 这些条目的单价是不是都查得到(``cost_known``)。有 ``approx`` 的
        #: 价目条时它仍是 True —— 那是「查得到但标了近似」,与「没定价」不同。
        self.all_priced = bool(all_priced)
        #: 为什么没有合计 / 为什么这个数不完整。空 = 无话可说。
        self.reason = str(reason)

    @property
    def currencies(self) -> tuple:
        return tuple(sorted(self.by_currency))

    def amount(self, currency: str) -> "float | None":
        """某个币种的实测花销。**这个币种没有记录 ⇒ None,不是 0.0。**

        0.0 的意思是「记过,金额是零」;None 的意思是「这个币种下什么都没有」。
        对「预算超了没有」这个问题,后者答不上来。
        """
        return self.by_currency.get(str(currency).upper())

    def as_dict(self) -> dict:
        return {"by_currency": dict(self.by_currency), "count": self.count,
                "all_priced": self.all_priced, "reason": self.reason,
                "currencies": list(self.currencies)}


class RuntimeConductCost:
    """``(conduct_id) -> ConductCost | None`` —— 读花销账本,**只读**。

    ## 三态,与监控探针同一条纪律

    ==================================  ==================================
    账本文件不存在 / 读不出来            ``None``(**读不到**)
    读到了,这份 conduct 一条都没有      ``ConductCost({}, count=0, …)``
    读到了,有记录                        逐币种实测值
    ==================================  ==================================

    第二行是**答得上来的 0**,不是读不到:账本在、查过了、这份 conduct 还没花过
    钱。第一行是答不上来。两者对「能不能放心继续跑」给的话不一样。

    ## 归集靠 ``source``,而 ``source`` 必须有人写

    账本只有 ``source`` 这一个逐调用归集维度(``meta`` 列写得进去,但
    ``summary()`` / ``recent()`` **都不 SELECT 它** —— 一个没有读端的维度)。
    conduct 的判决调用由 ``llm_seat`` 写成 ``conduct:<id>``。**没写的那些
    进不了这个口径** —— 于是它们既不在这份账里,也不会被误算给别人。

    ## 不建库

    ``billing.ledger.get_ledger()`` 会**建库**(与监控 store 同一个坑,那边的
    ``get_store_if_exists`` docstring 点名了这条:被动读者建出来的空库按构造就
    没有东西可读,而且在测试里那是往用户真实实验目录里写)。这里先看文件在不在,
    不在就报读不到。
    """

    def __init__(self, *, ledger_getter=None, path_getter=None):
        self._ledger_getter = ledger_getter
        self._path_getter = path_getter

    def _ledger(self):
        if self._ledger_getter is not None:
            return self._ledger_getter()
        from pathlib import Path

        from mast.billing.ledger import _default_path, get_ledger

        path = (self._path_getter() if self._path_getter is not None
                else _default_path())
        if not Path(path).exists():
            return None            # 没有账本 ⇒ 读不到,**不建一个空的**
        return get_ledger()

    def __call__(self, conduct_id: str) -> "ConductCost | None":
        from mast.conduct.llm_seat import cost_source

        want = cost_source(conduct_id)
        if not str(conduct_id or "").strip():
            return None            # 没有 id 就没有口径,不是「花了 0」
        try:
            ledger = self._ledger()
        except Exception as exc:  # noqa: BLE001
            logger.debug("花销账本取不到: %s", exc)
            return None
        if ledger is None:
            return None
        try:
            summary = ledger.summary()
        except Exception as exc:  # noqa: BLE001
            logger.debug("花销账本 summary 失败: %s", exc)
            return None
        rows = [r for r in (summary or {}).get("by_source") or []
                if str(r.get("key") or "") == want]
        by_currency: dict = {}
        count = 0
        all_priced = True
        for r in rows:
            cur = str(r.get("currency") or "CNY").upper()
            by_currency[cur] = round(by_currency.get(cur, 0.0)
                                     + float(r.get("cost") or 0.0), 6)
            count += int(r.get("count") or 0)
            all_priced = all_priced and bool(r.get("all_priced", True))
        reason = ""
        if len(by_currency) > 1:
            reason = (f"这份 conduct 的花销跨 {len(by_currency)} 种币种"
                      f"({'、'.join(sorted(by_currency))}),而合并需要一个汇率 ——"
                      f"本仓不自造汇率,所以这里不给合计")
        elif not by_currency:
            reason = "账本读到了,这份 conduct 名下还没有任何一条调用"
        if not all_priced:
            reason = (reason + ";" if reason else "") + \
                "其中有单价查不到的调用(cost_known=0),金额偏低"
        return ConductCost(by_currency, count, all_priced, reason)


def _now() -> float:
    import time

    return time.time()


def _record_time(rec: dict) -> "float | None":
    """一条扫描登记的时刻。``mtime`` 优先(文件真的什么时候写的),
    退回 ``recorded_at``。两个都没有 ⇒ **None**,不猜。"""
    for key in ("mtime", "recorded_at"):
        v = (rec or {}).get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _finite_or_none(v) -> "float | None":
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


__all__ = ["RuntimeExecutor", "RuntimeLatch", "RuntimeTemperature",
           "ConductNotifier", "RuntimeMonitorEvents", "RuntimeFrameMetrics",
           "ConductCost", "RuntimeConductCost",
           "RuntimeLinkProbe",
           "INSTRUMENT_BUSY_KEY", "OWNER_PREFIX",
           "WISHLIST_AGENT_PREFIX", "WISHLIST_KINDS", "SEVERITIES"]
