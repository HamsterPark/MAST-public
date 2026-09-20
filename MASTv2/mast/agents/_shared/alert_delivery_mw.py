"""在每次模型调用前投递尚未送达的电流监控告警。

告警表是唯一数据源；模块查询并折叠近期告警，把提示挂在本次消息副本的最后一条
human 消息上，不进入 checkpoint。WARN 与 CRITICAL 都应能到达模型上下文，
不能要求模型主动查询。模型正常返回后才标记 delivered_agent；人工确认 acked
与模型送达是不同状态。硬件 halt 的运行编号匹配由 runtime 负责。

读表失败时留待下轮，标记失败时允许重复投递，模型异常时不标记。投递不阻塞、不
等待重试，只执行有界的表查询和更新。消息副本避开易变 system 文本对缓存的影响。

送达标记使同一告警通常只注入一次；提示不进入历史，因此 CRITICAL 正文要求模型
把继续或停止的决定写入回复。若需重复通知，应在人工确认或明确处置前采用有限
重发策略，不能把 delivered_agent 误当成已经处置。"""

from __future__ import annotations

import collections
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from mast.agents._shared.inject import append_human_block, append_system_block

#: 登记表条目 id。
PROMPT_ID = "mw.alert_delivery"

#: 哪些 agent 挂它（真源；登记表与建图派生）。
AGENTS = ("instrument_control",)
from mast.monitoring.alert_routing import DeliveryClass, classify, get_alert_routing

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest

logger = logging.getLogger(__name__)


def _hhmmss(ts: float) -> str:
    """本地时钟 HH:MM:SS。告警在日志、面板和用户嘴里都是这个形式。"""
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except Exception:  # noqa: BLE001
        return "??:??:??"


def select_alerts(rows: list[dict], routing=None) -> dict:
    """把原始告警行分成「必达 / 折叠后的 warn / 静音掉的」。纯函数,永不抛。

    返回 ``{"critical": [...], "warn": [...], "muted_n": int,
    "dropped_n": int, "deliver_ids": [...]}``。

    ``warn`` 里每项带 ``count``:同一 ``rule`` 在折叠窗口内的条数。折叠保留**最新**
    的那条正文 —— 「rms_high ×7(最新 970.3 pA)」比七行同样的句子信息量更大。

    ``deliver_ids`` 是**真的被这个块代表了**的行 id,包括被折叠掉的同类。静音掉的
    行**不在里面**:它们没被送到,把它们标成已送达会让那一列不再回答自己的问题。
    """
    routing = routing or get_alert_routing()

    crit: list[dict] = []
    crit_ids: list[int] = []
    #: rule → {"row": 最新一条, "count": n, "ids": 这一组被代表了的行 id}
    folded: dict[str, dict] = {}
    muted_n = 0

    # rows 按 ts DESC 进来(store.undelivered_alerts 的 ORDER BY)。这里不重排,
    # 折叠时「第一次见到某个 rule」就是它最新的那条。
    for r in rows:
        try:
            rule = str(r.get("rule") or "")
            level = str(r.get("level") or "")
            cls = classify(rule, level, routing)
            if cls is DeliveryClass.MUTE:
                muted_n += 1
                continue
            rid = r.get("id")
            rid = int(rid) if rid is not None else None
            if cls is DeliveryClass.ALWAYS:
                crit.append(r)
                if rid is not None:
                    crit_ids.append(rid)
                continue
            # FOLD —— id 记在**组**里,这样一整组被篇幅挤掉时,它的 id 跟着一起
            # 不被标记(不然那些行会被记成「已给你看过」而实际上一个字都没出现)。
            #
            # 折叠范围就是调用方给的这批行(= 回看窗口),没有第二个更窄的窗口:
            # 两个窗口之间的缝里,行会既不被显示也不被标记,每轮重新扫一遍再扔掉。
            # 见 ``alert_routing.DEFAULT_LOOKBACK_S``。
            grp = folded.get(rule)
            if grp is None:
                folded[rule] = {"row": r, "count": 1,
                                "ids": [rid] if rid is not None else []}
            else:
                grp["count"] += 1
                if rid is not None:
                    grp["ids"].append(rid)
        except Exception:  # noqa: BLE001 — 一行坏数据不能让整块消失
            logger.debug("alert row skipped", exc_info=True)

    warn = sorted(folded.values(), key=lambda d: float(d["row"].get("ts") or 0.0),
                  reverse=True)

    # CRITICAL 先占位。装不下的 WARN 组被丢掉,并且**如实报出丢了几组** ——
    # 悄悄截断会让「没有更多」和「还有很多没给你看」长得一模一样。
    budget = max(1, int(routing.max_items))
    keep_warn = warn[: max(0, budget - len(crit))]
    dropped_n = len(warn) - len(keep_warn)

    deliver_ids = list(crit_ids)
    for w in keep_warn:
        deliver_ids.extend(w.get("ids") or [])

    return {"critical": crit, "warn": keep_warn, "muted_n": muted_n,
            "dropped_n": dropped_n, "deliver_ids": deliver_ids}


#: 视觉链路的伪行里放事件 id 的键。带下划线前缀,和告警表的真列区分开 ——
#: 它不是表里的东西,是「这条来自缓冲区的环」的标记。
VISION_ID_KEY = "_vision_event_id"


def vision_rows(buf, routing, *, seen: "set[str] | None" = None,
                now_mono_ns: int | None = None,
                now_wall: float | None = None) -> list[dict]:
    """视觉链路的 CRITICAL,渲染成和告警表同形状的伪行。纯函数,永不抛。

    ## 为什么必须单独走一条

    ``tip_quality_drop`` 这个 kind 下面有**两个判定方**:

    * **电流监控** —— ``alerts.emit_critical`` 既写告警表、又推缓冲区;
    * **视觉** —— ``vision/scan_monitor`` 与 ``buffer/service`` 只推缓冲区,
      **一个字都不进告警表**。

    两条管道互不覆盖；只读告警表会漏掉视觉事件，必须分别读取并去重。

    ## 去重:电流监控来源在这里被跳过

    它们已经由告警表那条路送过了。判据用 ``payload["source"]``,与
    ``runtime.tip_halt_source`` 同一个字段;读不到 source 一律当**视觉**,
    与那边的失败方向一致(宁可多送一条,不可漏一条)。

    ## 时间窗

    ``VisionEvent`` 只有 ``t_mono_ns``(单调钟),没有墙钟。这里按单调差换算出一个
    近似墙钟只为了**显示**;过滤用的是单调差本身,不受系统时间调整影响。

    ``seen`` 是调用方持有的「我已经给 agent 看过哪些 event_id」——
    视觉事件不在告警表里,没有 ``delivered_agent`` 可标,所以这条记忆只能在进程里。
    它**不是真源**,只是一份「别重复说」的备忘;丢了最坏是重复一次。
    """
    if buf is None:
        return []
    try:
        raw = buf.get_event_history(since_seqno=-1, limit=50) or []
    except Exception:  # noqa: BLE001 — 缓冲区读不到不能弄丢告警表那一半
        logger.debug("AlertDelivery: 读视觉事件历史失败(已忽略)", exc_info=True)
        return []

    now_mono_ns = time.monotonic_ns() if now_mono_ns is None else int(now_mono_ns)
    now_wall = time.time() if now_wall is None else float(now_wall)
    horizon_ns = int(float(routing.lookback_s) * 1e9)
    seen = seen if seen is not None else set()

    out: list[dict] = []
    for ev in raw:
        try:
            if str(getattr(getattr(ev, "severity", None), "value",
                           getattr(ev, "severity", ""))).lower() != "critical":
                continue
            payload = getattr(ev, "payload", None) or {}
            source = str(payload.get("source") or "")
            if source == "current_monitor":
                continue                      # 告警表那条路已经送过
            eid = str(getattr(ev, "event_id", "") or "")
            if not eid or eid in seen:
                continue
            t_ns = int(getattr(ev, "t_mono_ns", 0) or 0)
            age_ns = now_mono_ns - t_ns
            if t_ns <= 0 or age_ns > horizon_ns:
                continue
            kind = getattr(getattr(ev, "kind", None), "value",
                           getattr(ev, "kind", "")) or "event"
            out.append({
                "id": None,                   # 不在告警表里 ⇒ 没有表 id
                VISION_ID_KEY: eid,
                "ts": now_wall - age_ns / 1e9,
                "level": "critical",
                # 规则名用 signal,没有就退回 kind —— 视觉那条常常只有 kind。
                "rule": str(payload.get("signal") or kind),
                "summary_zh": str(payload.get("summary_zh") or ""),
                "source": source or "vision",
            })
        except Exception:  # noqa: BLE001 — 一条坏事件不能让整块消失
            logger.debug("AlertDelivery: 跳过一条读不动的视觉事件", exc_info=True)
    return out


def _source_label(row: dict) -> str:
    """渲染来源标签；缺失时明确写来源未标。
    
    读取并展示生产方的 source 字段，不用裸问号冒充完整的告警来源。"""
    src = str(row.get("source") or "").strip()
    if not src:
        return "来源未标"
    return {
        "current_monitor": "电流监控",
        "vision_scan_monitor": "视觉·扫描中途",
        "vision_tip_status": "视觉·针尖状态",
    }.get(src, src)


def format_alert_block(sel: dict) -> str:
    """把 :func:`select_alerts` 的结果渲染成一个 markdown 块。

    纯函数 —— 与 ``live_state_mw.format_live_state_block`` 同一约定,可以脱离
    中间件管道单测。没有任何要说的话时返回空串。
    """
    crit = sel.get("critical") or []
    warn = sel.get("warn") or []
    muted_n = int(sel.get("muted_n") or 0)
    dropped_n = int(sel.get("dropped_n") or 0)
    if not crit and not warn:
        # 只有静音项时**什么都不说**:那一行会在每一轮重复(静音项永不标记已送达),
        # 变成它自己要防的那种噪声。
        return ""

    lines: list[str] = ["## ⚠️ 电流监控告警(上次给你看过之后新增的)"]

    if crit:
        lines.append("")
        lines.append("**CRITICAL —— 先处置,不要继续当前的实验动作:**")
        for r in crit:
            lines.append(
                f"- [{_hhmmss(r.get('ts') or 0.0)}] `{r.get('rule')}`"
                f"({_source_label(r)})—— {r.get('summary_zh') or ''}"
            )
        # 同一个 kind 下有两个判定方,处方与可信度都不同 —— 收到两条时必须能分开看。
        if len({str(r.get("source") or "") for r in crit}) > 1:
            lines.append("")
            lines.append(
                "⚠️ 上面这些 CRITICAL **来自不同的判定方**:电流监控说的是物理越界"
                "(贴轨 / 冻结 / 巨幅瞬变,可从单段自证),视觉说的是图像上的形态突变。"
                "两者可以同时为真,也可以只有一个是真的 —— 别把它们当成互相印证。"
            )
        lines.append("")
        lines.append(
            "CRITICAL 的处方写在告警正文里。**在处置之前,任何「继续扫描 / 继续"
            "取数」动作产生的数据都可能是无效的**。若判断需要停止,用 `StopScan`;"
            "针尖处置用修针类技能。若你认为可以继续,请在回复里写出理由。"
        )

    if warn:
        # CRITICAL 前的不同告警按时间组成恶化序列，并报告提前量；同一规则的重复项仍可折叠，避免重复噪声淹没序列信息。
        newest_crit_ts = max((float(r.get("ts") or 0.0) for r in crit), default=None)
        lead_in = [w for w in warn
                   if newest_crit_ts is not None
                   and float(w["row"].get("ts") or 0.0) < newest_crit_ts]
        lines.append("")
        if lead_in:
            earliest = min(float(w["row"].get("ts") or 0.0) for w in lead_in)
            lead_s = max(0.0, newest_crit_ts - earliest)
            lines.append(
                f"**恶化时间线(这些 WARN 发生在 CRITICAL 之前,最早的早 {lead_s:.0f} 秒)"
                f"—— 它们是这次最早的可行动信号,不是背景噪声:**"
            )
        else:
            lines.append("**WARN(仅供参考,不要求立刻动作):**")
        for w in warn:
            r = w["row"]
            n = int(w.get("count") or 1)
            times = f" ×{n}(下面是最新一条)" if n > 1 else ""
            lines.append(
                f"- [{_hhmmss(r.get('ts') or 0.0)}] `{r.get('rule')}`{times}"
                f"({_source_label(r)})—— {r.get('summary_zh') or ''}"
            )

    if dropped_n > 0:
        lines.append("")
        lines.append(f"(另有 {dropped_n} 类 WARN 因篇幅未列出,监控面板可见。)")
    if muted_n > 0:
        lines.append(
            f"(另有 {muted_n} 条环境类告警按配置不打扰你 —— 它们仍在表里与面板上。)"
        )

    return "\n".join(lines)


class AlertDeliveryMiddleware(AgentMiddleware):
    """每次模型调用前注入未送达的电流监控告警。

    与 :class:`~mast.agents._shared.live_state_mw.LiveStateMiddleware` 同构:
    同步 ``wrap_model_call`` + 异步 ``awrap_model_call`` 都实现(LangChain 的基类
    在只定义同步钩子时会让异步派发直接抛 ``NotImplementedError``,而
    ``python -m mast`` 走的正是异步那条)。

    Parameters
    ----------
    get_store:
        取监控库的可调用对象。``None`` → 用 ``mast.monitoring.store.get_store``。
        注入口存在是为了测试能塞一个临时库进来 —— **校验不能交给会犯这个错的
        那一方**,所以测试用的是真的 ``CurrentMonitorStore``(真 SQLite、真迁移),
        只是换了路径,而不是一个替身。
    get_routing:
        取投递策略的可调用对象。``None`` → ``alert_routing.get_alert_routing``。
    """

    def __init__(self, get_store: Callable[[], Any] | None = None,
                 get_routing: Callable[[], Any] | None = None,
                 get_buffer: Callable[[], Any] | None = None):
        super().__init__()
        self._get_store = get_store
        self._get_routing = get_routing
        self._get_buffer = get_buffer
        #: 已经给 agent 看过的**视觉**事件 id。视觉事件不进告警表,没有
        #: ``delivered_agent`` 可标,所以这份记忆只能在进程里。
        #: **它不是真源**,只是「别重复说」的备忘 —— 丢了最坏是重复一次,
        #: 而重复远好过沉默(与本模块其余部分同一个失败方向)。
        #: 有界:一条长会话不该无限攒 id。
        self._vision_shown: "collections.deque[str]" = collections.deque(maxlen=512)

    @property
    def name(self) -> str:
        return "AlertDeliveryMiddleware"

    # ── internals ────────────────────────────────────────────────────────

    def _store(self):
        """当前的监控库,**没有就是 None —— 绝不新建**。

        用 ``get_store_if_exists`` 而不是 ``get_store``:后者会在
        ``project_root()/experiments/current_monitor/`` 下**创建**库和目录。
        对一个只读的投递中间件,建一个空库什么都没买到(空库里没有告警),
        在测试里却会写进用户真实的 experiments 目录 —— 而 conftest 对这个库
        **没有**像 wishlist / 文献注册表那样的 autouse 守卫。
        """
        if self._get_store is not None:
            return self._get_store()
        from mast.monitoring.store import get_store_if_exists
        return get_store_if_exists()

    def _routing(self):
        if self._get_routing is not None:
            return self._get_routing()
        return get_alert_routing()

    def _buffer(self):
        """视觉事件缓冲区,没有就是 ``None``。绝不新建。"""
        if self._get_buffer is not None:
            return self._get_buffer()
        try:
            from mast.buffer.active import get_active_buffer
            return get_active_buffer()
        except Exception:  # noqa: BLE001 — 纯离线 / 未接视觉服务
            return None

    @staticmethod
    def _shown_vision_ids(sel: dict) -> list[str]:
        """这一块里真的出现了的**视觉**事件 id(含被折叠代表掉的同组)。"""
        out: list[str] = []
        for r in (sel.get("critical") or []):
            eid = r.get(VISION_ID_KEY)
            if eid:
                out.append(str(eid))
        for w in (sel.get("warn") or []):
            eid = (w.get("row") or {}).get(VISION_ID_KEY)
            if eid:
                out.append(str(eid))
        return out

    def _apply(self, request: "ModelRequest") -> tuple["ModelRequest", tuple]:
        """注入告警块。返回 ``(request, (表 id 列表, 视觉事件 id 列表))``。永不抛。

        两份待标记的东西**都不在这里落账** —— 只有当模型调用正常返回之后才标记
        (见 :meth:`wrap_model_call`)。模型调用抛异常时告警必须还在。
        """
        try:
            routing = self._routing()
            rows: list[dict] = []

            # ① 告警表 —— 电流监控的全部(warn + critical),而且是**耐久**的。
            store = self._store()
            if store is not None:
                since = time.time() - float(routing.lookback_s)
                # limit 给得比 max_items 宽:折叠要看到同类的全部条数,静音项也要
                # 被数出来。宽度有界(50)所以这仍然是一次小查询。
                rows.extend(store.undelivered_alerts(since, limit=50))

            # 视觉事件只存在于缓冲区环中，不进入告警表；两条来源都必须接入投递路径。
            rows.extend(vision_rows(self._buffer(), routing,
                                    seen=set(self._vision_shown)))

            if not rows:
                return request, ([], [])
            # 合并后按时间倒序:两条来源的行要按事故的真实顺序排,而不是按来源分堆。
            rows.sort(key=lambda r: float(r.get("ts") or 0.0), reverse=True)
            sel = select_alerts(rows, routing)
            block = format_alert_block(sel)
            if not block:
                return request, ([], [])
        except Exception as exc:  # noqa: BLE001 — 告警投递绝不能弄崩一轮对话
            logger.debug("AlertDelivery: 组装告警块失败(已忽略): %s", exc,
                         exc_info=True)
            return request, ([], [])

        pending = (list(sel.get("deliver_ids") or []),
                   self._shown_vision_ids(sel))

        out = append_human_block(request, PROMPT_ID, block)
        if out is not None:
            return out, pending
        # 没有 human 消息可挂 → 退回 system（告警送达压过一次 cache 命中）。
        try:
            return append_system_block(request, PROMPT_ID, block), pending
        except Exception as exc:  # noqa: BLE001
            logger.debug("AlertDelivery: system 消息注入也失败(%s)——本轮不注入",
                         exc)
            return request, ([], [])

    def _mark(self, pending: tuple) -> None:
        """记下「这些已经给 agent 看过了」。永不抛。

        两半各自记在自己该在的地方:告警表那半落 ``delivered_agent`` 列(耐久,
        跨进程);视觉那半落进程内的 ``_vision_shown``(视觉事件不在表里)。

        失败的方向是**重复**:没标上的下一轮会再出现一次。丢一条 CRITICAL 才是
        不可接受的失败,重复一条不是。
        """
        try:
            ids, vision_ids = pending
        except Exception:  # noqa: BLE001 — 形状不对就当没有
            return
        if vision_ids:
            # deque(maxlen) 自己淘汰最旧的,不会无限攒。
            self._vision_shown.extend(str(v) for v in vision_ids)
        if not ids:
            return
        try:
            store = self._store()
            if store is not None:
                store.mark_alerts_delivered(ids)
        except Exception:  # noqa: BLE001
            logger.debug("AlertDelivery: 标记送达失败(下一轮会重复注入)",
                         exc_info=True)

    # ── hooks ────────────────────────────────────────────────────────────

    def wrap_model_call(self, request: "ModelRequest",
                        handler: Callable[["ModelRequest"], Any]) -> Any:
        request, pending = self._apply(request)
        result = handler(request)
        # handler 抛异常时这一行到不了 —— 于是告警仍是未送达,重试时还在。
        self._mark(pending)
        return result

    async def awrap_model_call(self, request: "ModelRequest",
                               handler: Callable[["ModelRequest"], Any]) -> Any:
        request, pending = self._apply(request)
        result = await handler(request)
        self._mark(pending)
        return result


__all__ = ["AlertDeliveryMiddleware", "format_alert_block", "select_alerts",
           "vision_rows", "VISION_ID_KEY"]
