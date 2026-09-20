"""仪器聊天的旁白侧信道，为长任务提供持续进度。

旁白写入 conversation_messages 与 EventBus，不进入 agent 的 messages
检查点或 BufferService。调用方仅提供结构化事件，narration_templates
从实际参数生成文字，缺失数字时不编造。

调用线程只做有界的非阻塞入队；后台线程负责落库和推送，错误不应打断
技能执行。MAST_CHAT_NARRATION=0 关闭该通道。

conversation_id 来自 turn_context；裸线程需在启动前 bind，不能在
缺少上下文时猜测最近会话。没有明确会话归属就不发送。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

#: 转录表里的行类型。``ConversationStore.agent_activity()`` 只查 ``kind='message'``，
#: 所以换一个 kind 不会污染任何 agent 的「群聊活动」视图。
NARRATION_KIND = "narration"

#: 队列上限。一次 ForgeAuTip 估计 100–400 条旁白，两千条 ≈ 五次长任务的积压；
#: 真到了这个量级说明写线程已经卡死（DB 锁死），那时**丢最旧**比让仪器线程等着好。
_QUEUE_MAX = 2000

#: ``anchor`` 缓存的会话数上限。同时活跃的私聊会话是个位数，多出来的按插入顺序淘汰。
_ANCHOR_MAX = 32

_QUEUE: "queue.Queue[dict | None]" = queue.Queue(maxsize=_QUEUE_MAX)
_WRITER: "threading.Thread | None" = None
_WRITER_LOCK = threading.Lock()
_BUSY = threading.Event()      # 写线程正在处理一条 —— flush() 用它判「真的空了」

_STORE: Any = None             # 显式注册的 ConversationStore（测试/运行时）
_ANCHORS: "dict[str, int]" = {}
_ANCHOR_LOCK = threading.Lock()

#: 纯统计，没有任何判据读它。故意不加锁：丢一个计数无所谓，而在仪器线程上
#: 抢一把锁是有所谓的。
_STATS: dict[str, int] = {
    "emitted": 0,     # narrate() 被调用且入队
    "no_cid": 0,      # 没有会话 id → 不发
    "no_store": 0,    # 进程里没有转录存储 → 不发
    "off": 0,         # 总开关关着
    "dropped": 0,     # 队列满，丢掉
    "written": 0,     # 真的落库了
    "unknown": 0,     # 不认识的 kind → 不发
    "errors": 0,      # 写线程里出的异常
}


# ── 总开关 ──────────────────────────────────────────────────────────────


def enabled() -> bool:
    """``MAST_CHAT_NARRATION=0`` 全关（与 ``MAST_SCAN_VISION_MONITOR`` 同形）。

    每次都读环境变量而不是 import 时读一次：这样运行中改也生效，测试里
    monkeypatch 也生效 —— 一个只在 import 时生效的开关，在冻结的应用里等于没有。
    """
    return str(os.environ.get("MAST_CHAT_NARRATION", "1")).strip().lower() \
        not in ("0", "false", "off", "no")


# ── 存储解析 ────────────────────────────────────────────────────────────


def set_store(store: Any) -> None:
    """注册转录存储（``ConversationStore``）。``None`` = 摘除（回到自动查找）。

    运行时在建好 ``_conv_store`` 之后调一次；测试传一个建在 tmp 目录上的 store。
    **测试必须传** —— 本仓「测试污染真实用户数据」已经发生五次，而旁白写的正是
    用户真实会话所在的那个库。
    """
    global _STORE
    _STORE = store


def current_store() -> Any:
    """当前该往哪写。显式注册优先，其次问 API 的进程级 ``AppContext``。

    自动查找那一步取的是 ``api.bootstrap`` 已经接好的**同一个** store 对象
    （``ctx.conversation_store = rt._conv_store``），不是新建一个 —— 新建一个会
    在另一个文件里再开一份 WAL，那才是第二真源。standalone dev / 单测里两者都
    没有 ⇒ 返回 None ⇒ 整条链路 no-op。
    """
    if _STORE is not None:
        return _STORE
    try:
        from mast.api.context import get_context

        return getattr(get_context(), "conversation_store", None)
    except Exception:  # noqa: BLE001 — 查不到就是没有
        return None


# ── anchor（回放时插在哪一条消息之后） ──────────────────────────────────


def set_anchor(conversation_id: str, n: int) -> None:
    """记下「此刻这个会话已经渲染出 n 条消息」。由对话引擎每个 super-step 调。

    为什么需要它：``render_history()`` 的输出只有 ``{role, content}``，**没有时间戳**
    （``chat/render.py``），所以刷新之后按 wall-clock 把旁白插回消息之间是做不到的。
    ``anchor`` 是那一刻的**消息条数**，前端据此把旁白插在 ``messages[anchor-1]`` 之后。

    已知近似（写在这里，不当成精确）：一条在**工具执行中途**发出的旁白拿到的
    anchor 是这次工具调用**之前**的值 —— 那个 tool 的结果还没被渲染出来。所以它会
    排在该 tool 的折叠块**之前**。这是对的（结果确实还没发生），但它不是「精确对齐」。
    """
    cid = str(conversation_id or "")
    if not cid:
        return
    try:
        i = int(n)
    except (TypeError, ValueError):
        return
    with _ANCHOR_LOCK:
        _ANCHORS[cid] = i
        if len(_ANCHORS) > _ANCHOR_MAX:
            for k in list(_ANCHORS)[:len(_ANCHORS) - _ANCHOR_MAX]:
                _ANCHORS.pop(k, None)


def _anchor_of(cid: str) -> int:
    """``-1`` = 没有活跃回合（唤醒调度 / 群跑 / 手动触发）→ 前端一律追加到末尾。"""
    with _ANCHOR_LOCK:
        return int(_ANCHORS.get(cid, -1))


def clear_anchor(conversation_id: str) -> None:
    """回合结束时摘掉。留着的话，下一轮开头发的旁白会带上上一轮的条数。"""
    with _ANCHOR_LOCK:
        _ANCHORS.pop(str(conversation_id or ""), None)


# ── 发一条 ──────────────────────────────────────────────────────────────


def _snapshot(value: Any, depth: int = 2) -> Any:
    """把 dict/list 浅拷两层。

    调用方交出来的 ``params`` 是**活的** —— ``ctx.run`` 之后它可能被改，而旁白是在
    写线程上渲染的。拷贝让「句子里的值 = 发出旁白那一刻的值」成立。
    只拷两层是刻意的：深拷一个带 ndarray 的 result 会把「绝不阻塞仪器线程」这条
    保证直接毁掉。
    """
    if depth <= 0:
        return value
    if isinstance(value, dict):
        return {k: _snapshot(v, depth - 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(v, depth - 1) for v in value]
    return value


def _normalise_image(image: Any) -> "dict | None":
    """``{"src": 绝对路径, "origin": "milestone_png"|"sxm"}``，不合规就当没有图。

    ``origin`` 必填且不猜：``milestone_png`` 是**原样读盘**（模型真正看过的那一帧），
    ``sxm`` 是渲染缩略图。猜错的后果不是没有图，是**借了别的图**——
    正是 #76/「历史里所有缩略图静默变成最新那张整图」那个事故。
    """
    if not isinstance(image, dict):
        return None
    src = str(image.get("src") or "").strip()
    origin = str(image.get("origin") or "").strip()
    if not src or origin not in ("milestone_png", "sxm"):
        return None
    return {"src": src, "origin": origin}


def _event_time(event_t: Any) -> float:
    """确定旁白所描述事件的发生时间。

    同步执行器和异步轮询观察到事件的时间可能不同。已知事件时间时使用
    传入值，否则使用当前时间，避免把入队次序误当物理执行次序。
    非数、NaN、无穷和非正值表示时间未知，不夹紧也不猜测。
    """
    try:
        val = float(event_t)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return time.time()
    if val != val or val <= 0.0 or val == float("inf"):
        return time.time()
    return val


def _emit(cid: str, kind: str, image: Any, data: dict,
          event_t: Any = None) -> None:
    """入队一条旁白。**这是仪器线程上唯一跑的东西**，必须又快又不抛。"""
    if not enabled():
        _STATS["off"] += 1
        return
    if not cid:
        _STATS["no_cid"] += 1
        return
    store = current_store()
    if store is None:
        _STATS["no_store"] += 1
        return
    try:
        from mast.core.turn_context import current_turn

        run_id = str(current_turn().get("run_id") or "")
    except Exception:  # noqa: BLE001
        run_id = ""
    item = {
        "cid": cid,
        "kind": str(kind or ""),
        "data": _snapshot(data),
        "image": _normalise_image(image),
        # anchor 在**入队时**取，不在写线程上取:它描述的是「这句话发在哪个
        # 位置」,不是「什么时候写进库的」。
        "anchor": _anchor_of(cid),
        # t = **这件事发生的时刻**。同步发出方不传 ⇒ 现在(对它们二者相同);
        # 轮询发出方(视觉监视器)必须传,否则它的滞后会被当成事件顺序 ——
        # 见 _event_time 的自述。
        "t": _event_time(event_t),
        "run_id": run_id,
        "store": store,
    }
    _ensure_writer()
    try:
        _QUEUE.put_nowait(item)
    except queue.Full:
        # 丢**最旧**的那条 —— 新的旁白比几分钟前的旁白更值钱，而阻塞在这里
        # 意味着让仪器等着一条聊天记录写完。
        try:
            _QUEUE.get_nowait()
        except queue.Empty:  # pragma: no cover — 竞态窗口
            pass
        try:
            _QUEUE.put_nowait(item)
        except queue.Full:  # pragma: no cover
            _STATS["dropped"] += 1
            return
        _STATS["dropped"] += 1
    _STATS["emitted"] += 1


def narrate(kind: str, /, *, image: Any = None, event_t: "float | None" = None,
            **data: Any) -> None:
    """发一条旁白。**永不抛、永不阻塞、没接上就是 no-op。**

    ``kind`` 见 ``narration_templates.TEMPLATES``；不认识的 kind 会在写线程上被丢掉
    （**不发一句通用的**）。``data`` 是结构化数据，句子由模板拼 ——
    这里**没有** ``text=`` 参数，所以没有地方能塞一句编好的话。

    ``event_t``：这条旁白**描述的那件事**发生的 epoch 秒。**只有知道的发出方才
    传** —— 轮询线程(视觉监视器)知道帧的采集时刻,它必须传;同步发出方
    (composite 执行器、技能)不传,因为对它们「现在」就是「发生时刻」。
    不传 / 传垃圾 ⇒ 用现在。理由见 :func:`_event_time`。

    会话 id 自己去 ``turn_context`` 取，不经过 ctx：于是单测里的 ``FakeCtx``、
    无头脚本、以及每一个既有调用方都一个字都不用改。
    """
    try:
        from mast.core.turn_context import current_turn

        cid = str(current_turn().get("conversation_id") or "")
    except Exception:  # noqa: BLE001
        cid = ""
    try:
        _emit(cid, kind, image, data, event_t)
    except Exception:  # noqa: BLE001 — 一条旁白绝不许弄坏调用它的那件事
        logger.debug("narrate(%s) failed", kind, exc_info=True)


class Sink:
    """绑定到一个会话的旁白出口。给**拿不到 ContextVar 的裸线程**用。

    ``bind()`` 在 cid 为空时也返回一个（no-op 的）Sink，所以调用方永远不用写
    ``if sink is not None``。不持有线程、不持有连接，可以随便传。
    """

    __slots__ = ("conversation_id",)

    def __init__(self, conversation_id: str = "") -> None:
        self.conversation_id = str(conversation_id or "")

    def __bool__(self) -> bool:
        """``if sink:`` = 「这个 sink 真的会发东西吗」。"""
        return bool(self.conversation_id)

    def narrate(self, kind: str, /, *, image: Any = None,
                event_t: "float | None" = None, **data: Any) -> None:
        """见 :func:`narrate`。

        ``event_t`` 在这里**尤其重要**:Sink 的存在理由就是「给拿不到 ContextVar
        的裸线程用」,而裸线程正是那些**观测滞后**的发出方(视觉监视器)。
        一个用 Sink 却不传 event_t 的调用方,就是在用自己的轮询节拍冒充事件顺序。
        """
        try:
            _emit(self.conversation_id, kind, image, data, event_t)
        except Exception:  # noqa: BLE001
            logger.debug("sink.narrate(%s) failed", kind, exc_info=True)


def bind(conversation_id: "str | None" = None) -> Sink:
    """取一个可以跨线程带走的 :class:`Sink`。

    ``conversation_id`` 留空 = 用当前上下文里的那个（**必须在还有 ContextVar 的线程
    上调**，也就是起 daemon 线程**之前**）。
    """
    cid = str(conversation_id or "")
    if not cid:
        try:
            from mast.core.turn_context import current_turn

            cid = str(current_turn().get("conversation_id") or "")
        except Exception:  # noqa: BLE001
            cid = ""
    return Sink(cid)


# ── 写线程 ──────────────────────────────────────────────────────────────


def _ensure_writer() -> None:
    global _WRITER
    if _WRITER is not None and _WRITER.is_alive():
        return
    with _WRITER_LOCK:
        if _WRITER is not None and _WRITER.is_alive():
            return
        t = threading.Thread(target=_writer_loop, name="mast-narration",
                             daemon=True)
        _WRITER = t
        t.start()


def _writer_loop() -> None:
    while True:
        item = _QUEUE.get()
        if item is None:  # 停车信号（只有 atexit 会发）
            return
        _BUSY.set()
        try:
            _write_one(item)
        except Exception:  # noqa: BLE001 — 写线程绝不许死
            _STATS["errors"] += 1
            logger.debug("narration write failed", exc_info=True)
        finally:
            _BUSY.clear()


def _write_one(item: dict) -> None:
    from mast.chat import narration_templates as tpl

    rendered = tpl.render(item["kind"], item["data"])
    if rendered is None:
        _STATS["unknown"] += 1
        return
    meta = {
        "v": 1,
        "nk": item["kind"],
        "anchor": item["anchor"],
        "run_id": item["run_id"],
        "tone": rendered.tone,
        # 渲染用到的原始值。存在的理由：让「这句话里的 10 V 是不是真的下发值」
        # 变成一个**可核对**的问题 —— 句子和原始值在同一行里，谁都能拿它去和
        # 记录管线对账。
        "facts": rendered.facts,
        "fold": 1,
    }
    if rendered.degraded:
        meta["degraded"] = True
    if item["image"]:
        meta["image"] = item["image"]
    try:
        meta_json = json.dumps(meta, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 — facts 里混进不可序列化的东西
        meta_json = json.dumps({"v": 1, "nk": item["kind"],
                                "anchor": item["anchor"], "tone": rendered.tone},
                               ensure_ascii=False)
    seq = item["store"].append_message(
        item["cid"],
        kind=NARRATION_KIND,
        # ⚠️ agent_id 必须是空串。前端 ``transcriptRefresh.transcriptCursorFor``
        # 的判据是 ``scope === "transcript" && agent_id === 当前看的 agent``；
        # 写上 "instrument_control" 会让代理对话页因为一条旁白去重刷群聊转录 ——
        # 无害，但那是一条**假因果**。旁白不属于任何 agent：它是系统在向用户解说。
        agent_id="",
        role=NARRATION_KIND,
        text=rendered.text,
        t=item["t"],
        meta=meta_json,
        # 转录推送不发：那条事件是给群聊游标用的，而这里另有自己的 EventType。
        # 一条旁白发两个事件会把只回放 100 条的总线挤爆（一次长任务几百条）。
        notify=False,
    )
    _STATS["written"] += 1
    try:
        from mast.core.events import EventBus

        EventBus.get().publish_chat_narration(item["cid"], int(seq or 0), item["t"])
    except Exception:  # noqa: BLE001 — 掉一次推送 = 晚几秒刷新；抛一次 = 丢一条旁白
        logger.debug("narration notify failed", exc_info=True)


# ── 测试/收尾用 ─────────────────────────────────────────────────────────


def flush(timeout: float = 2.0) -> bool:
    """等队列排空。返回是否真的排空了。**只给测试和 atexit 用**。

    不用 ``Queue.join()``：那需要每条都配一次 ``task_done()``，而写线程里任何一条
    早退路径漏掉它就会让 ``join()`` 永远挂着 —— 在一个「绝不阻塞」的模块里放一个
    可能永远挂着的等待，是自相矛盾的。
    """
    deadline = time.time() + max(0.0, float(timeout))
    while time.time() < deadline:
        if _QUEUE.empty() and not _BUSY.is_set():
            return True
        time.sleep(0.005)
    return _QUEUE.empty() and not _BUSY.is_set()


def stats() -> dict:
    """计数快照（诊断用）。没有任何判据读它。"""
    return dict(_STATS)


def reset_for_tests() -> None:
    """清空队列 / anchor / 计数 / 注册的 store。**每个用例前后都该调。**"""
    set_store(None)
    while True:
        try:
            _QUEUE.get_nowait()
        except queue.Empty:
            break
    with _ANCHOR_LOCK:
        _ANCHORS.clear()
    for k in _STATS:
        _STATS[k] = 0


@atexit.register
def _drain_at_exit() -> None:  # pragma: no cover — 进程退出路径
    try:
        flush(1.0)
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "NARRATION_KIND",
    "Sink",
    "bind",
    "clear_anchor",
    "current_store",
    "enabled",
    "flush",
    "narrate",
    "reset_for_tests",
    "set_anchor",
    "set_store",
    "stats",
]
