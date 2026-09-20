"""Chat streaming request contract. The SSE response frames are hand-typed on
the frontend (a stream is not expressible as an OpenAPI response_model):

    {kind: "snapshot", messages: [{role, content}]}   # rendered history, streamed
    {kind: "error", message: str, degraded?: bool}
    {kind: "done"}
"""

from __future__ import annotations

from pydantic import BaseModel


class ChatTurnRequest(BaseModel):
    conversation_id: str
    user_text: str


class NarrationItem(BaseModel):
    """一条旁白 —— 系统在向**用户**解说，不是助手在说话。

    ``anchor`` 是它被发出的那一刻 ``render_history()`` 已经渲染出的消息条数；
    前端把 anchor 相同的旁白插在 ``messages[anchor-1]`` 之后。``-1`` = 没有活跃
    回合（唤醒调度 / 群跑 / 手动触发）→ 追加到末尾。

    ``facts`` 是渲染这句话时用到的**原始值**（``{"params.pulse_v": 10.0}``）。
    它存在的理由是让「这句话里的 10 V 是不是真的下发值」变成一个可核对的问题，
    而不是一件要相信的事。

    列表**不带图片**（``has_image`` 只是一个布尔）—— 缩略图走
    ``/api/chat/narration-image/{seq}``，理由与 ``/api/vision/recent-frame``
    当初分出去时一模一样：一屏几十条 × 每条几十 KB base64，绝大多数没人点开。
    """

    seq: int
    t: float
    text: str
    nk: str = ""
    anchor: int = -1
    tone: str = "info"
    has_image: bool = False
    fold: int = 1
    facts: dict = {}


class NarrationResponse(BaseModel):
    items: list[NarrationItem] = []
    latest_seq: int = 0
    #: True = 这个进程里没有接转录存储（standalone dev）。**不是 500** ——
    #: 与 ``/api/vision/recent`` 同一套降级口径：空但不坏。
    degraded: bool = False


__all__ = ["ChatTurnRequest", "NarrationItem", "NarrationResponse"]
