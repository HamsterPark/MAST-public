"""GET /api/chat/narration —— 仪器 chat 的旁白流（**只读**）。

设计文档：``docs/v2/design/chat_narration_sidechannel.md`` §3.2 / §Q4

旁白是系统在向**用户**解说一个长任务正在做什么（「我们要打一发脉冲」「扫到
50%」）。它落在 ``conversation_messages`` 表的 ``kind='narration'`` 行里，
写端在 ``mast/chat/narration.py``。这里只把它读出来。

⚠️ **两条路径都是字面量，且刻意挂在 ``/api/chat/`` 下。**

本仓在路由遮蔽上踩过两次，两次都是**静默**的（``orchestrator.py`` 的
``/agents/<x>/transcript`` 被先注册的 ``/agents/{agent_id}/...`` 捕获，绑成
``agent_id="run-task"`` 然后返回**另一个 handler** 的结果；``documents.py`` 记了
同款）。遮蔽的症状不是 404 —— 是 **200 + 错的数据**，所以「端点能返回 200」这种
测试对它完全无效。回归测试因此断言的是**路由解析到了哪个函数**
（``tests/v2/unit/api/test_narration_routes.py``）。

``/agents/{agent_id}/narration`` 其实也安全（它和既有的
``/agents/{agent_id}/messages`` 同形），但换个前缀是零风险的，没有理由不换。
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import Response

from mast.api.schemas_chat import NarrationItem, NarrationResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _store(ctx):
    """接线好的 ``ConversationStore``；standalone dev 里是 None。"""
    return getattr(ctx, "conversation_store", None)


def _item(row: dict) -> NarrationItem:
    """一行 → 一条。``meta`` 解析失败**不丢这一行**：句子本身在 ``text`` 里，
    它才是用户要读的东西；meta 只是排版与对账信息。"""
    meta: dict = {}
    raw = row.get("meta") or ""
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                meta = parsed
        except (ValueError, TypeError):
            logger.debug("narration meta unparseable at seq=%s", row.get("seq"))
    image = meta.get("image")
    return NarrationItem(
        seq=int(row.get("seq") or 0),
        t=float(row.get("t") or 0.0),
        text=str(row.get("text") or ""),
        nk=str(meta.get("nk") or ""),
        anchor=int(meta.get("anchor", -1)),
        tone=str(meta.get("tone") or "info"),
        has_image=bool(isinstance(image, dict) and image.get("src")),
        fold=int(meta.get("fold", 1) or 1),
        facts=meta.get("facts") if isinstance(meta.get("facts"), dict) else {},
    )


@router.get("/chat/narration", response_model=NarrationResponse)
def get_narration(
    request: Request,
    conversation_id: str = Query(..., description="哪个会话的旁白"),
    after_seq: int = Query(0, ge=0, description="增量：只要 seq 比这个大的"),
    limit: int = Query(500, ge=1, le=2000),
) -> NarrationResponse:
    """一个会话的旁白，最旧优先。

    增量读（``after_seq``）是常态：前端收到 ``chat_narration`` 事件后只拉新的那几条。
    没有存储 → ``degraded=true`` + 空列表，**不是 500**。
    """
    ctx = getattr(request.app.state, "ctx", None)
    store = _store(ctx)
    if store is None:
        return NarrationResponse(items=[], latest_seq=int(after_seq), degraded=True)
    cid = str(conversation_id or "").strip()
    if not cid:
        return NarrationResponse(items=[], latest_seq=0, degraded=False)
    try:
        rows = store.messages_since(cid, int(after_seq), limit=int(limit))
    except Exception as exc:  # noqa: BLE001 — 读不出来不该把对话页打崩
        logger.warning("narration read failed for %s: %s", cid, exc)
        return NarrationResponse(items=[], latest_seq=int(after_seq), degraded=True)

    # 过滤在这里而不是在 SQL 里：``messages_since`` 是共用的读法，为旁白加一个
    # kind 参数会让每个既有调用方都要想一遍「我该传什么」。一次会话的转录行数
    # 有硬上限（8000），过滤成本可以忽略。
    from mast.chat.narration import NARRATION_KIND

    items = [_item(r) for r in rows if r.get("kind") == NARRATION_KIND]
    # latest_seq 取**这一批读到的所有行**的最大 seq，不只是旁白行的:游标要跨过
    # 中间那些非旁白行，否则每次轮询都会把它们重读一遍(而且永远读不完)。
    latest = max([int(r.get("seq") or 0) for r in rows] + [int(after_seq)])
    return NarrationResponse(items=items, latest_seq=latest, degraded=False)


#: 允许被这个端点读出来的图片来源。**白名单，不是黑名单**：``meta.image.src``
#: 是一条写在 DB 行里的绝对路径，而这个端点会把那条路径的内容发给浏览器。
#: 只放行由我们自己写下的那一类，其余一律 404。
_ALLOWED_ORIGINS = ("milestone_png",)


@router.get("/chat/narration-image/{seq}",
            responses={200: {"content": {"image/png": {}}}},
            response_class=Response)
def get_narration_image(
    request: Request,
    seq: int = Path(..., ge=1),
    conversation_id: str = Query(...),
) -> Response:
    """一条旁白配的缩略图。没有图 → 404（不是一张占位图）。

    **原样读盘，不重渲染。** 那个 PNG 在**那件事发生的当时**就已经落好盘了。
    生产方有两个（2026-08-16 起）：

      · ``scan_monitor._persist_frame_png`` —— 模型**真正看过的那一帧**；
      · ``vision.cluster_panel.render_cluster_panel`` —— 团簇判据的四格分解图，
        在**判读发生的那一刻**画好（它用 Figure/FigureCanvasAgg，不碰 Gcf，
        所以在 composite 的工作线程上画是安全的）。

    重新渲染一张（比如从最新的 .sxm）就正好复刻了 #76/#78 那个事故：扫描途中每一次判读都配着上一张图，
    而新整帧存下来之后，历史里所有缩略图会静默变成那张整图。

    顺带：这条路径上**一行 matplotlib 都没有**。``render_scan_thumbnail`` 用的是
    pyplot 的全局 figure manager（Gcf，非线程安全），而 ``/api/vision/recent``
    已经在 API 线程上调它 —— 再加一个调用方就是两个线程共用 Gcf，症状是随机的
    图错乱或崩在渲染里，而崩的位置离旁白很远（设计文档 §10.1）。
    ``origin="sxm"`` 那一档**至今没有生产方**，所以这里连门都不开。
    (08-16 有人往那一档挂过一次图：发得出去、取不回来，一条挂着图、图却永远
     404 的旁白比没有图更坏 —— 它看起来完全正常。修法就是上面第二个生产方。)

    缓存 24 h + immutable：与 ``/api/vision/recent-frame`` 的 ``no-store`` 不同，
    而这个区别是有理由的 —— 那个端点按位置借最新的 .sxm，同一个 seqno 明天可能
    指向另一张图；这里的 ``src`` 是**写死在那一行里的一条路径**，不会变。
    """
    ctx = getattr(request.app.state, "ctx", None)
    store = _store(ctx)
    cid = str(conversation_id or "").strip()
    if store is None or not cid:
        return Response(status_code=404)
    try:
        rows = store.messages_since(cid, int(seq) - 1, limit=1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("narration image lookup failed: %s", exc)
        return Response(status_code=404)
    if not rows or int(rows[0].get("seq") or 0) != int(seq):
        return Response(status_code=404)

    from mast.chat.narration import NARRATION_KIND

    row = rows[0]
    if row.get("kind") != NARRATION_KIND:
        return Response(status_code=404)
    try:
        meta = json.loads(row.get("meta") or "{}")
        image = meta.get("image") or {}
        src = str(image.get("src") or "")
        origin = str(image.get("origin") or "")
    except (ValueError, TypeError):
        return Response(status_code=404)
    if not src or origin not in _ALLOWED_ORIGINS:
        return Response(status_code=404)

    from pathlib import Path as _P

    p = _P(src)
    try:
        if not p.is_file():
            # 图被清掉了（artifacts 是可清理目录）。404 而不是占位图 ——
            # 前端 onError 会说「这一帧的画面取不到了」，那是一句真话；
            # 一张占位图会被读成「这一帧本来就长这样」。
            return Response(status_code=404)
        data = p.read_bytes()
    except OSError as exc:
        logger.debug("narration image read failed (%s): %s", src, exc)
        return Response(status_code=404)
    return Response(content=data, media_type="image/png", headers={
        "Cache-Control": "public, max-age=86400, immutable",
    })
