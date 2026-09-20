"""唯一的注入入口：往一次 model call 里追加一段上下文，并留下归属。

在这个文件出现之前，**每个注入中间件各自抄了一份**「取出 system 文本 → 拼上
自己的块 → 造一个新 SystemMessage → 尽量走 request.override」。九份副本，
彼此有细微差别（有的漏了 override 只做赋值，于是每次调用触发一条
DeprecationWarning；有的把 list 型 content 直接 ``str()`` 掉，多模态块就没了）。
更要紧的是：**没有任何一份记下「这段文本是谁塞进来的」** —— 抓包里看到的是
一整块 system 文本，要回答「这 22 k 字符里哪 483 个是针尖块」只能靠人肉正则。

于是这里做两件事：

1. **统一拼接**（str / 内容块列表两种 content 都对，一律优先 ``override``）。
2. **登记归属**：块清单同时写进两处 ——

   * ``additional_kwargs["mast_blocks"]``：链内快路，下一个注入器直接接着往后加；
   * :mod:`mast.prompts.ledger`：按内容哈希发布，**跨得过缓存中间件重建对象**
     （理由见 ledger 的模块 docstring；additional_kwargs 在 Claude 路径上会丢）。

抓包侧（:mod:`mast.prompts.capture`）先查 ledger，再退回 additional_kwargs，
两个都没有就如实标 ``blocks_source="none"`` —— 不猜。

## 落点三选一

``append_system_block``
    稳定块（角色、仪器配置、针尖、上游产物）。

``append_human_block``
    **逐轮易变**的块（实时读数、告警、记忆召回）。挂最后一条 human 消息，让
    system 逐轮**逐字节相同** —— Anthropic 的 cache 断点就打在 system 末尾，
    system 一变整段连同历史全 miss。这条推理的原始版本在 ``live_state_mw``
    里，那两个 helper 现在搬到了这里。

``append_new_human``
    必须新增一条消息的守卫类（空转提醒、prefill 续跑、技能图像）。

## 不在这里做的事

不校验 prompt_id 是否登记在册（这是热路径）；那件事由
``tests/v2/unit/prompts/test_manifest_matrix.py`` 的结构闸门保证。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mast.prompts import ledger as _ledger
from mast.prompts.ledger import (
    POS_LAST_HUMAN,
    POS_NEW_HUMAN,
    POS_SYSTEM,
    Block,
    normalize_text,
)

logger = logging.getLogger(__name__)

#: 没有任何登记来源时，已存在的那段文本记成什么。**不冒充某个 agent 的系统提示**
#: —— 「不知道这段是谁的」和「这段是 IC 的系统提示」是两件事。
BASE_BLOCK_ID = "system.base"

#: human 消息的同款兜底。用户自己说的话不是「注入」，账上要看得出区别。
HUMAN_BASE_BLOCK_ID = "user.message"

_BASE_ID_BY_POSITION = {
    POS_SYSTEM: BASE_BLOCK_ID,
    POS_LAST_HUMAN: HUMAN_BASE_BLOCK_ID,
    POS_NEW_HUMAN: HUMAN_BASE_BLOCK_ID,
}


# ─────────────────────────────────────────────────────────────────────────
# 消息级 helper。从 live_state_mw 搬过来（alert_delivery 是第二个消费者，
# 两边都改成 import 这里，副本归零）。
# ─────────────────────────────────────────────────────────────────────────

def last_human_index(messages: list) -> int | None:
    """最后一条 human 消息的下标，没有就 None。

    鸭子判定（``type`` 或类名），因此对任何 langchain_core 版本、以及测试替身
    都成立。
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        kind = getattr(m, "type", None) or type(m).__name__
        if str(kind).lower() in ("human", "humanmessage", "user"):
            return i
    return None


def with_appended_text(msg: Any, block: str) -> Any:
    """*msg* 的**副本**，末尾接上 *block*。

    绝不改原对象 —— 原对象是 checkpoint 里的那一条，而注入块是逐次调用的。
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        new_content: Any = content + "\n\n" + block
    elif isinstance(content, list):
        # 多模态内容块：追加一个 text 块，而不是把图片 str() 掉。
        new_content = list(content) + [{"type": "text", "text": block}]
    else:
        new_content = str(content) + "\n\n" + block
    copier = getattr(msg, "model_copy", None)
    if callable(copier):
        return copier(update={"content": new_content})
    return type(msg)(content=new_content)


# ─────────────────────────────────────────────────────────────────────────
# 归属账
# ─────────────────────────────────────────────────────────────────────────

def _prior_blocks(msg: Any, text: str, *, position: str) -> list[Block]:
    """*msg* 已有文本的块分解：additional_kwargs → ledger → 兜底一整块。"""
    if not text:
        return []
    kw = getattr(msg, "additional_kwargs", None)
    if isinstance(kw, dict):
        raw = kw.get("mast_blocks")
        if isinstance(raw, list) and raw:
            out: list[Block] = []
            for d in raw:
                try:
                    out.append(Block(id=str(d["id"]), start=int(d["start"]),
                                     end=int(d["end"]), chars=int(d["chars"]),
                                     position=str(d.get("position", position))))
                except Exception:  # noqa: BLE001 — 坏的一条不该毁掉整条账
                    return _from_ledger_or_base(text, position=position)
            return out
    return _from_ledger_or_base(text, position=position)


def _from_ledger_or_base(text: str, *, position: str) -> list[Block]:
    hit = _ledger.lookup(text)
    if hit:
        return list(hit)
    base_id = _BASE_ID_BY_POSITION.get(position, BASE_BLOCK_ID)
    return [Block(id=base_id, start=0, end=len(text), chars=len(text),
                  position=position)]


def _kwargs_with_blocks(msg: Any, blocks: list[Block]) -> dict[str, Any]:
    kw = dict(getattr(msg, "additional_kwargs", None) or {}) if msg is not None else {}
    kw["mast_blocks"] = [b.to_dict() for b in blocks]
    return kw


def _set_system(request: Any, new_sm: SystemMessage) -> Any:
    """优先走非弃用的 ``ModelRequest.override()``；测试替身没有它才直接赋值。"""
    override = getattr(request, "override", None)
    if callable(override):
        try:
            return override(system_message=new_sm)
        except Exception:  # noqa: BLE001 — 替身的 override 可能签名不同
            pass
    request.system_message = new_sm
    return request


def _set_messages(request: Any, messages: list) -> Any:
    override = getattr(request, "override", None)
    if callable(override):
        try:
            return override(messages=messages)
        except Exception:  # noqa: BLE001
            pass
    request.messages = messages
    return request


# ─────────────────────────────────────────────────────────────────────────
# 三个入口
# ─────────────────────────────────────────────────────────────────────────

def append_system_block(request: Any, prompt_id: str, text: str) -> Any:
    """把 *text* 追加到 system 消息末尾，并登记它是 *prompt_id*。

    *text* 为空 → 原样返回（「这一块这次没有内容」不是错误，也不该在账上留痕）。
    """
    if not text:
        return request
    try:
        existing = getattr(request, "system_message", None)
        existing_text = (normalize_text(getattr(existing, "content", ""))
                         if existing is not None else "")
        prior = _prior_blocks(existing, existing_text, position=POS_SYSTEM)

        if existing is None:
            new_content: Any = text
        else:
            content = getattr(existing, "content", "")
            if isinstance(content, list):
                new_content = list(content) + [{"type": "text", "text": text}]
            else:
                new_content = existing_text + "\n\n" + text

        new_text = normalize_text(new_content)
        # 用**结果长度**反推区间，与分隔符怎么拼无关；前缀不变，所以旧块偏移仍有效。
        block = Block(id=prompt_id, start=max(0, len(new_text) - len(text)),
                      end=len(new_text), chars=len(text), position=POS_SYSTEM)
        blocks = prior + [block]
        new_sm = SystemMessage(content=new_content,
                               additional_kwargs=_kwargs_with_blocks(existing, blocks))
        _ledger.publish(new_text, blocks)
        return _set_system(request, new_sm)
    except Exception:  # noqa: BLE001 — 注入失败绝不能弄坏一次 run
        logger.debug("append_system_block(%s) failed (swallowed)", prompt_id,
                     exc_info=True)
        return request


def append_human_block(request: Any, prompt_id: str, text: str) -> Any | None:
    """把 *text* 挂到**最后一条 human 消息**末尾，并登记归属。

    返回 ``None`` = 这一轮没有 human 消息可挂（某个 agent 的第一跳可能只有工具
    尾巴）。调用方自行决定回退到 system 还是放弃 —— 「哪个更重要」是调用方的
    判断，不是这里的。
    """
    if not text:
        return request
    try:
        messages = list(getattr(request, "messages", None) or [])
        idx = last_human_index(messages)
        if idx is None:
            return None
        target = messages[idx]
        old_text = normalize_text(getattr(target, "content", ""))
        prior = _prior_blocks(target, old_text, position=POS_LAST_HUMAN)

        new_msg = with_appended_text(target, text)
        new_text = normalize_text(getattr(new_msg, "content", ""))
        block = Block(id=prompt_id, start=max(0, len(new_text) - len(text)),
                      end=len(new_text), chars=len(text), position=POS_LAST_HUMAN)
        blocks = prior + [block]
        try:
            new_msg.additional_kwargs = _kwargs_with_blocks(target, blocks)
        except Exception:  # noqa: BLE001 — 冻结的替身；ledger 仍然记得住
            pass
        _ledger.publish(new_text, blocks)

        messages[idx] = new_msg
        return _set_messages(request, messages)
    except Exception:  # noqa: BLE001
        logger.debug("append_human_block(%s) failed (swallowed)", prompt_id,
                     exc_info=True)
        return None


def append_new_human(request: Any, prompt_id: str, message: Any) -> Any:
    """把**新的一条** human 消息接在消息尾部，并登记归属。

    *message* 可以是字符串，也可以是已经造好的 HumanMessage（技能图像那种
    多模态内容块）。
    """
    try:
        msg = message
        if isinstance(message, str):
            if not message:
                return request
            msg = HumanMessage(content=message)
        text = normalize_text(getattr(msg, "content", ""))
        blocks = [Block(id=prompt_id, start=0, end=len(text), chars=len(text),
                        position=POS_NEW_HUMAN)]
        try:
            msg.additional_kwargs = _kwargs_with_blocks(msg, blocks)
        except Exception:  # noqa: BLE001
            pass
        _ledger.publish(text, blocks)

        messages = list(getattr(request, "messages", None) or []) + [msg]
        return _set_messages(request, messages)
    except Exception:  # noqa: BLE001
        logger.debug("append_new_human(%s) failed (swallowed)", prompt_id,
                     exc_info=True)
        return request


__all__ = [
    "BASE_BLOCK_ID", "HUMAN_BASE_BLOCK_ID",
    "POS_LAST_HUMAN", "POS_NEW_HUMAN", "POS_SYSTEM",
    "append_human_block", "append_new_human", "append_system_block",
    "last_human_index", "with_appended_text",
]
