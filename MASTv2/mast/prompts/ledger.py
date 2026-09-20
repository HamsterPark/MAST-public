"""Block ledger — 「这段文本由哪些块拼成」，按内容寻址。

## 为什么不是把归属挂在消息对象上

第一版把块清单写进 ``SystemMessage.additional_kwargs["mast_blocks"]``，链内确实
传得过去（``ModelRequest.override`` 是 ``dataclasses.replace``，``_execute_model``
直接把对象交给 ``model.invoke``，``_format_for_tracing`` 只在多模态块上
``model_copy()``——而 ``model_copy`` 保留 additional_kwargs）。

**断点在缓存中间件**：``langchain_anthropic/middleware/prompt_caching.py`` 的
``_tag_system_message`` 末尾是 ``return SystemMessage(content=new_content)`` ——
它重建了对象，additional_kwargs 一起没了。那个中间件在七张图里都排在全部注入器
**内侧**，于是凡 ``isinstance(model, ChatAnthropic)``（Claude / MiniMax）的构建，
台账走到模型回调前就消失。默认模型是 Kimi（ChatOpenAI 子类，不挂该中间件），
所以在本地测得好好的，切到 Claude 静默失效 —— 本仓最怕的那一种。

所以归属**按内容寻址**：注入器发布 ``sha1(文本) -> 块清单``，抓包侧拿到最终文本
再反查。重建对象不改变文本，哈希就还在。

## 边界

这里只回答「这段文本由哪些块组成」。**绝不用它渲染任何文本** —— 展示用的正文
一律来自真实抓包或 registry 的 loader，理由见 :mod:`mast.prompts.registry` 的
诚实性铁律。

## 两种登记

``publish``      每次注入后发布，走 LRU（默认 256 条），够覆盖在飞的请求即可。
``register_base`` 建图时登记**静态基块**的分解（IC = 系统提示 + Nanonis 速查）。
                 每个 agent 一条、不被 LRU 淘汰 —— 它要活到进程结束，因为每一次
                 model call 的 system 都是从它长出来的。
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Any

logger = logging.getLogger(__name__)

#: 在飞请求的量级：群聊 fan-out 最多几个 agent × 每轮几次调用。256 远够，
#: 且每条只是几十字节的块清单（**不存文本**）。
MAX_ENTRIES = 256

#: 块落在哪。与 registry 的 ``position`` 同一套词表。
POS_SYSTEM = "system"
POS_LAST_HUMAN = "last_human"
POS_NEW_HUMAN = "new_human"


@dataclass(frozen=True)
class Block:
    """一段文本里某个注入块占的区间。``start``/``end`` 是**码点**下标。"""

    id: str
    start: int
    end: int
    chars: int
    position: str = POS_SYSTEM

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(content: Any) -> str:
    """消息体 → 文本。**注入侧与抓包侧必须共用这一个函数**。

    内容块（Anthropic 风格 ``[{"type": "text", ...}, …]``）按类型标注后拼接，
    这样 tool_use 块看得见，而不是静静变成空串。

    共用的理由是哈希要对得上：缓存中间件把 str 包成
    ``[{"type": "text", "text": ..., "cache_control": ...}]``，这里拼回来必须
    与注入时那一串**逐字节相同**，否则 lookup 永远落空而且不会报错。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                kind = str(blk.get("type", "block"))
                if kind == "text":
                    parts.append(str(blk.get("text", "")))
                else:
                    parts.append(f"[{kind}] {blk!r}")
            else:
                parts.append(str(blk))
        return "\n".join(parts)
    return str(content)


def fingerprint(text: str) -> str:
    """文本指纹。内容寻址的键。"""
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


class _Ledger:
    def __init__(self, maxlen: int = MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        self._lru: OrderedDict[str, list[Block]] = OrderedDict()
        #: agent -> (fingerprint, blocks)；不淘汰。
        self._pinned: dict[str, tuple[str, list[Block]]] = {}
        self._pinned_by_fp: dict[str, list[Block]] = {}

    def publish(self, text: str, blocks: list[Block]) -> None:
        fp = fingerprint(text)
        with self._lock:
            self._lru[fp] = list(blocks)
            self._lru.move_to_end(fp)
            while len(self._lru) > MAX_ENTRIES:
                self._lru.popitem(last=False)

    def lookup(self, text: str) -> list[Block] | None:
        fp = fingerprint(text)
        with self._lock:
            hit = self._lru.get(fp)
            if hit is not None:
                self._lru.move_to_end(fp)
                return list(hit)
            pinned = self._pinned_by_fp.get(fp)
            return list(pinned) if pinned is not None else None

    def register_base(self, agent: str, parts: list[tuple[str, str]]) -> list[Block]:
        """登记一个 agent 的静态基块分解。``parts`` = [(prompt_id, text), …]。

        文本按 ``parts`` 顺序拼接（**调用方负责拼接方式与建图时一致**），
        返回算好的块清单。重复登记覆盖（重建图就该覆盖）。
        """
        blocks: list[Block] = []
        cursor = 0
        chunks: list[str] = []
        for pid, txt in parts:
            t = txt or ""
            n = len(t)
            blocks.append(Block(id=pid, start=cursor, end=cursor + n, chars=n,
                                position=POS_SYSTEM))
            cursor += n
            chunks.append(t)
        full = "".join(chunks)
        fp = fingerprint(full)
        with self._lock:
            old = self._pinned.get(agent)
            if old is not None and old[0] != fp:
                self._pinned_by_fp.pop(old[0], None)
            self._pinned[agent] = (fp, list(blocks))
            self._pinned_by_fp[fp] = list(blocks)
        return blocks

    def base_for(self, agent: str) -> list[Block] | None:
        with self._lock:
            hit = self._pinned.get(agent)
            return list(hit[1]) if hit else None

    def clear(self) -> None:
        with self._lock:
            self._lru.clear()
            self._pinned.clear()
            self._pinned_by_fp.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._lru) + len(self._pinned)


_LEDGER = _Ledger()


def publish(text: str, blocks: list[Block]) -> None:
    """发布一段文本的块分解。永不抛。"""
    try:
        _LEDGER.publish(text, blocks)
    except Exception:  # noqa: BLE001 — 台账坏了也不能弄坏一次 run
        logger.debug("ledger publish failed (swallowed)", exc_info=True)


def lookup(text: str) -> list[Block] | None:
    """这段文本由哪些块组成？查不到就是 None（不猜、不编）。"""
    try:
        return _LEDGER.lookup(text)
    except Exception:  # noqa: BLE001
        logger.debug("ledger lookup failed (swallowed)", exc_info=True)
        return None


def register_base(agent: str, parts: list[tuple[str, str]]) -> list[Block]:
    try:
        return _LEDGER.register_base(agent, parts)
    except Exception:  # noqa: BLE001
        logger.debug("ledger register_base failed (swallowed)", exc_info=True)
        return []


def base_for(agent: str) -> list[Block] | None:
    try:
        return _LEDGER.base_for(agent)
    except Exception:  # noqa: BLE001
        return None


def get_ledger() -> _Ledger:
    return _LEDGER


__all__ = [
    "MAX_ENTRIES", "POS_LAST_HUMAN", "POS_NEW_HUMAN", "POS_SYSTEM",
    "Block", "base_for", "fingerprint", "get_ledger", "lookup",
    "normalize_text", "publish", "register_base",
]
