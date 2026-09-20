"""conduct 的**判决席位** —— 把 :func:`mast.skills.composite.llm_node.decide_route`
接到 :class:`~mast.conduct.director.ConductDirector` 上的那一根线。

设计:``campaign_director_design.md`` §3 的三级执行体第二级(L1 闸门)。

## 这个席位是什么,尤其**不是**什么

模型只在预定义路由中做判决，不拥有物理量或坐标的生成权限。
动作参数预先写在 spec 中，判决席位不得修改这些数值：

* 输出只能是闸门 ``routes`` 里的**一个名字**(闭集,由 ``decide_route`` 的
  ``Literal`` schema 结构保证),每条路由的动作参数在 spec 里预写好;
* 它**不填任何物理量**。这一条不是靠提示词说服,是靠 ``GateResult`` 里根本
  没有能放数字的地方 —— 移除诱因,不劝说。

## 「判不了」的三种来源,只有一种是判决

``decide_route`` 从不抛异常:任何失败都折进 ``escape`` 路由。这对 composite
工作流是对的(工作流要降级到安全出口),但对闸门**不够**:一道 llm 闸门能发出
``detour``,那是半夜把用户叫起来换样品、以小时计。所以这里把
``decide_route`` 的三种 escape 拆开:

======================  ====================================================
``uncertain``           **一次真的弃权** —— 模型看了证据,说自己判不了。
                        照 spec 声明的 escape 路由走(spec 已禁止它映射到
                        ``pass``)。
``off_enum``            返回值不在闭集里。**不是判决**,是判决器坏了。
``error: ...``          调用炸了/没有 API key/provider 500。**不是判决**。
======================  ====================================================

后两种在这里**抛** :class:`SeatUnavailable`,交给
:func:`mast.conduct.rules.evaluate_gate` 已有的那条 except 分支报「判不了」。
不在这里兜一个默认去向:一个「provider 500 于是去换样品」的兜底,合理得让人
看不出兜底发生过 —— 而这正是本仓反复记账的那族错误。

## 超时:两道,不是一道

``executor.run`` 卡死杀不得(强杀会永久损坏 Nanonis 端口),那是一条写下来的
诚实短板。**LLM 调用不在那条豁免里** —— 它是一次 HTTP 往返,停等它就是让整条
指挥线程停在一道闸上。所以:

1. ``request_timeout`` 交给 provider SDK 界定**单次 HTTP**
   (``make_chat_model`` 的 docstring 点名了这条:不给它,一条卡住的连接会把
   工作线程永久停在那里);
2. 本模块再用一个**墙钟上限**界定「Director 最多等多久」。到点就当判不了往下
   走,那条工作线程留给它自己去超时 —— 它是 daemon,而且手上没有任何写权限
   (它唯一的产出是一个没人再读的 dict)。

两道都要:第一道管连接,第二道管**我们**。只有第一道时,一次 SDK 层的重试
链仍然能把 tick 停住。

## 决策日志由**这里**写

``llm_node.decide_route`` 自己**不写** JSONL 决策日志 —— 写的是调用方
(``composite/interpreter.py`` 的 ``_walk_llm``)。所以「接上就白送一份审计」
是假的:不在这里写,conduct 的每一次判决都不会留下任何一行。

写的是 ``experiments/decision_log.jsonl``(与 composite 同一个文件、同一套
字段),加上 conduct 侧的 ``conduct_id`` / ``stage_id`` / ``gate_id``。
**注意**:这个文件今天在 UI 上没有消费者(唯一的读端
``webui/builder_api.py`` 的 ``/builder/decisions`` 随 Gradio 一起成了死代码,
``build_routes()`` 没有任何调用方)。人眼前那一份走的是另一条路 ——
``gate_evaluated`` 事件的 ``llm`` 审计块 → 面板闸门史 + ``progress.jsonl``。
这里这一份是**将来 router-graduation 分类器的训练底料**,事后重建不出来。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Director 最多为一道闸门等多久(墙钟,秒)。到点 ⇒ 判不了。
#:
#: 60 s 的来历:闸门判定跑在 tick 循环里,而 ``tick_interval_s`` 的设计区间是
#: 5-15 s。等一分钟已经是「这一 tick 明显拖长了」,再长就等于让一次
#: provider 抖动把急停/暂停的响应也一起推迟 —— 那两个必须在步边界生效。
DEFAULT_DEADLINE_S = 60.0

#: 交给 provider SDK 的单次 HTTP 上限。比墙钟略小:让 SDK 先自己报错,
#: 我们拿到的就是一句具体的话,而不是一个「等够了」。
DEFAULT_REQUEST_TIMEOUT_S = 45.0

#: 决策日志里标明这条记录来自 conduct 闸门(composite 那边写的是 workflow 名)。
MECHANISM = "conduct_gate"


class SeatUnavailable(RuntimeError):
    """判决**没有发生** —— 不是判决说「不行」,是根本没判成。

    与「模型弃权」严格分开:弃权走 spec 声明的 escape 路由(它可能是
    ``detour``);这个异常走 ``evaluate_gate`` 的「判不了」,也就是有人值守
    时请人来看、无人值守时走 ``gate.unattended_escape``。
    """


#: 花销账本里 conduct 花销的 ``source`` 前缀。**单一真源**:写在这里,
#: 读在 :class:`mast.conduct.adapters.RuntimeConductCost`。
COST_SOURCE_PREFIX = "conduct:"


def cost_source(conduct_id: str) -> str:
    """这份 conduct 在花销账本里的 ``source`` 值。"""
    return f"{COST_SOURCE_PREFIX}{str(conduct_id or '').strip()}"


def _default_model_factory(node: dict, usage_source: str = ""):
    """建一次判决用的模型。**带 request_timeout** —— 见模块 docstring 第二道。

    没有任何 provider key 时 ``make_chat_model`` 抛 ``RuntimeError``:
    那正是我们要的形状(没席位 ⇒ 判不了),别在这里吞掉。

    ``usage_source`` 让这次调用的花销**记在这份 conduct 名下**。不给它的话,
    账本的 ``source`` 由 ``agent`` 推出来 = ``"orchestrator"`` —— 与 supervisor
    自己的路由调用混在一起,事后**分不开**,于是 ``ConductBudget.usd_max``
    没有任何东西可读。账本只有 ``source`` 这一个逐调用归集维度
    (``meta`` 虽然写得进去,但没有任何查询读它)。
    """
    from mast.agents._shared.models import make_chat_model

    return make_chat_model("orchestrator", max_tokens=2048, temperature=0.1,
                           request_timeout=DEFAULT_REQUEST_TIMEOUT_S,
                           usage_source=usage_source or None)


def _build_model(factory, node: dict, usage_source: str):
    """调工厂。**按签名决定给不给 ``usage_source``,不靠 try/except TypeError。**

    靠捕 ``TypeError`` 的话,工厂**内部**抛的 TypeError 会被当成「签名不收这个
    参数」,于是静默退回一次不带归集的调用 —— 花销记到 ``orchestrator`` 名下,
    而没有任何地方会说这件事发生过。签名是查得到的事实,查它。
    """
    import inspect

    if usage_source:
        try:
            params = inspect.signature(factory).parameters
        except (TypeError, ValueError):      # 内建/C 实现取不到签名
            params = {}
        takes = ("usage_source" in params
                 or any(p.kind is inspect.Parameter.VAR_KEYWORD
                        for p in params.values()))
        if takes:
            return factory(node, usage_source=usage_source)
    return factory(node)


def _call_bounded(fn: "Callable[[], Any]", deadline_s: float, what: str) -> Any:
    """在一个 daemon 线程里跑 ``fn``,最多等 ``deadline_s``。

    超时 ⇒ 抛 :class:`SeatUnavailable`。**不杀线程**(杀不得,也不必):它手上
    没有仪器、没有写权限,最坏是多花一次 API 调用的钱,然后把结果丢给没人读的
    变量。这与「不做超时杀步」是同一条纪律,只是这里的代价小到可以直接放手。

    ``what`` 只进错误消息。**建模型也要走这里**,不只是判决那一步:
    ``make_chat_model`` 会 import 一整条 langchain provider 链、读密钥文件,
    这些一样能停住调用它的那条线程 —— 而调用它的正是指挥线程。一道只挡住了
    第二步的超时,挡不住第一步卡住的那种失败。
    """
    box: dict = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 —— 原样带回主线程判
            box["error"] = exc

    t = threading.Thread(target=_run, name="conduct-llm-seat", daemon=True)
    t.start()
    t.join(timeout=max(0.1, float(deadline_s)))
    if t.is_alive():
        raise SeatUnavailable(
            f"{what}超过 {deadline_s:.0f} s 还没回来 —— 判不了。"
            f"(不停等:一道闸门不该拖住急停与暂停的响应)")
    if "error" in box:
        raise SeatUnavailable(f"{what}抛异常: {box['error']}")
    return box.get("value")


def _is_real_abstention(decision: dict) -> bool:
    """这次 escape 是**模型弃权**,还是判决器坏了?

    ``decide_route`` 把三件事都写进 ``escape_reason``:``uncertain`` /
    ``off_enum`` / ``error: ...``。只有第一种是判决。
    """
    return str(decision.get("escape_reason") or "") == "uncertain"


def make_decide_route(
    *,
    context: "Callable[[], dict] | None" = None,
    model_factory: "Callable[[dict], Any] | None" = None,
    deadline_s: float = DEFAULT_DEADLINE_S,
    decide: "Callable[..., dict] | None" = None,
    log: "Callable[[dict], None] | None" = None,
    clock: "Callable[[], float] | None" = None,
) -> "Callable[[dict, dict], dict]":
    """→ 一个能直接喂给 ``ConductDirector(decide_route=...)`` 的可调用。

    ``context()`` 回一份 ``{conduct_id, stage_id, gate_id, spec_id}`` 之类的
    随手快照,只进决策日志、**不进提示词**:闸门问的问题由 spec 的
    ``llm_node.responsibility`` 定死,不由运行时上下文改写。拿不到就写空 ——
    一条少了 conduct_id 的审计仍然比没有审计好。

    全部依赖都可注入,因为这条路径上真的会打 provider:测试注入 ``decide`` 与
    ``log``,一次网络都不打。
    """
    _decide = decide
    _log = log
    _now = clock or time.time

    def decide_route(node: dict, inputs: dict) -> dict:
        nonlocal _decide, _log
        if _decide is None:
            from mast.skills.composite.llm_node import decide_route as _dr
            _decide = _dr
        if _log is None:
            from mast.skills.composite.llm_node import log_decision as _ld
            _log = _ld
        ctx: dict = {}
        if context is not None:
            try:
                ctx = dict(context() or {})
            except Exception as exc:  # noqa: BLE001 —— 审计上下文拿不到不该拦住判决
                logger.debug("conduct 判决上下文取不到(照样判): %s", exc)

        t0 = _now()
        factory = model_factory or _default_model_factory
        # 这次调用的花销记在**这份 conduct** 名下。拿不到 conduct_id 就传空 ——
        # 那样它退回 ``orchestrator``,与 supervisor 的路由调用混在一起。
        # **混了就分不开**,而分不开的那部分不会被算进任何一份 conduct 的预算:
        # 少记比乱记好,但两者都要说得出来(见 ``RuntimeConductCost``)。
        src = cost_source(str(ctx.get("conduct_id") or "")) \
            if ctx.get("conduct_id") else ""
        # **两步分开报,但同一道墙钟都管得着。** 分开是因为两句话不一样:
        # 「建不出判决模型」是配置问题(一个 provider key 都没配),
        # 「判决没回来」是 provider 问题 —— 用户要做的事不同。
        try:
            model = _call_bounded(
                lambda: _build_model(factory, node, src), deadline_s, "建判决模型")
            decision = _call_bounded(
                lambda: _decide(node, inputs, model=model), deadline_s, "LLM 判决")
        except SeatUnavailable as exc:
            _record(_log, ctx, node, inputs, None, _now() - t0,
                    unavailable=str(exc))
            raise
        if not isinstance(decision, dict):
            why = f"LLM 判决返回了 {type(decision).__name__},不是决策 dict"
            _record(_log, ctx, node, inputs, None, _now() - t0, unavailable=why)
            raise SeatUnavailable(why)

        escaped = bool(decision.get("escaped"))
        if escaped and not _is_real_abstention(decision):
            # off_enum / error —— ``decide_route`` 把它折进了 escape 路由,而那条
            # 路由可能通向 ``detour``。**不许**让一次 provider 故障发出
            # 「半夜换样品」的裁决。
            why = str(decision.get("escape_reason") or "未说明")
            _record(_log, ctx, node, inputs, decision, _now() - t0,
                    unavailable=f"escape 不是弃权而是故障: {why}")
            raise SeatUnavailable(f"LLM 判决未完成({why})—— 判不了")

        _record(_log, ctx, node, inputs, decision, _now() - t0)
        return decision

    return decide_route


def _record(log, ctx: dict, node: dict, inputs: dict, decision: "dict | None",
            elapsed_s: float, *, unavailable: str = "") -> None:
    """写一行决策日志。**成败都写** —— 只记成功的那些,日志就会显得判决器从不
    出错,而那正是最该留下的一类记录。永不抛。"""
    rec = {
        "ts": time.time(),
        "mechanism": MECHANISM,
        "node_id": node.get("id"),
        "responsibility": node.get("responsibility"),
        "routes": list((node.get("routes") or {}).keys()),
        "escape": node.get("escape"),
        "inputs": inputs,
        "elapsed_s": round(float(elapsed_s), 3),
    }
    rec.update({k: v for k, v in (ctx or {}).items() if k not in rec})
    if decision:
        rec.update({k: v for k, v in decision.items() if k != "ts"})
    if unavailable:
        # **顺序要紧,而且这一段是被一条测试逼出来的**:先写 decision 再盖标记。
        # 反过来的话,``decide_route`` 折进 escape 的那个路由名会原样留在
        # ``route`` 上 —— 一次 provider 500 于是在训练底料里长得像「模型选了
        # hold」。故障被折叠成一个具体的值,合理得没人会去核。
        rec["unavailable"] = unavailable
        rec["folded_route"] = rec.get("route")
        rec["route"] = None
    try:
        log(rec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("conduct 决策日志写失败(判决照常): %s", exc)


# 「决策里哪几项给人看」不在这里第二次决定 —— 它是 ``rules._audit_of`` 的事
# (那里是消费端:审计块从 ``GateResult`` 进事件 payload)。同一个挑选规则写两份,
# 迟早会有一份少一项,而少的那一项通常正是 ``model``。


__all__ = ["SeatUnavailable", "make_decide_route", "MECHANISM",
           "DEFAULT_DEADLINE_S", "DEFAULT_REQUEST_TIMEOUT_S",
           "COST_SOURCE_PREFIX", "cost_source"]
