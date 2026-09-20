"""SkillImageMiddleware — 让 agent 真的**看见**它刚刚测出来的图。

在这之前,技能→模型的唯一出口是 ``skill_adapter`` 里的
``ToolMessage(content=summary)``,而 ``summary`` 永远是 ``str``。也就是说:
``.sxm→PNG`` 的渲染器一直存在(``webui/scan_preview.render_scan_thumbnail``),
但它的消费者全是 HTTP 路由喂 React ``<img>`` —— 图像从来没有到过模型手里。
``git log -S "image_url" -- MASTv2/mast/agents`` 是**零个 commit**:这条路不是被
删掉的,是从来没接过。

本模块是这条路的**出站那一半**。分工:

  * ``skill_adapter``  —— 把图像的**路径**挂在 ToolMessage 的
    ``additional_kwargs[IMAGES_KEY]`` 上(见 :data:`IMAGES_KEY`);
  * 本模块          —— 在**发请求那一刻**把路径读成 base64 data URI,
    拼成 provider 认的 content block,挂进本次请求。

## 为什么必须拆成两半(而不是在 skill_adapter 里直接塞 base64)

``messages`` 会被 **SqliteSaver 每轮持久化并重放**。一张 512 px 缩略图约 98.6 K
base64 字符;塞进 ToolMessage 就等于让它在**接下来整个 session 的每一轮**里被反复
写盘、读盘、重放。这与 项目规约 的「Checkpoint 不放 tensor / 大对象」是同一条不变式
——tensor 只是这条规则最显眼的那个例子,不是它的全部。

所以:**state 里存路径,像素只在出站请求里出现,且绝不写回 state。**
本模块用 ``request.override(messages=…)``,改的是**本次调用**的消息列表;
graph state 一个字节都不动(与 ``live_state_mw`` / ``prefill_guard_mw`` 同一手法)。

## 为什么图像挂在**新的 user 消息**上,而不是塞进 ToolMessage 的 content

Moonshot 官方文档(platform.kimi.ai/docs/guide/use-kimi-vision-model,
见 ``docs/api_providers/moonshot_kimi_vision.md``)里,图像块**只出现在 user 消息**里:

    {"role": "user", "content": [{"type": "image_url", …}, {"type": "text", …}]}

它**没有**说 ``role: "tool"`` 的消息能不能带图像块。OpenAI 兼容端点在这一点上历来
是不能的。把图塞进 tool 消息,就是拿「看起来合理的类比」当事实 —— 而那正是这个项目
反复栽过的形状,也正是本次任务点名要避开的。代价不对称:猜错的话 agent **每看一张
扫描图就 400**,而且只会在真机上才发现。

所以图像走**文档明确写过的那个形状**:一条新的 user 消息,紧跟在工具结果之后。
文字块里点名这些图来自哪个技能,模型才知道自己在看什么。

## 不支持视觉的模型:降级,但**要出声**

``model_supports_vision()``(单一真源在 ``mast.config``)返回 False 时,本模块
**不发任何 image block** —— 不是 400,不是报错,也不是发一个模型看不懂的块。

但它**不再是静默的**。降级时会往本次请求里追加一句纯文本说明(任何 provider 都收),
告诉模型「有 N 张图没发给你、因为当前模型不支持、不要假装看过」。理由写在
``_apply`` 的 docstring 里:``agent_overrides.json`` 允许把仪器 agent 换成不支持视觉
的模型,而第一版的「逐字节相同」降级会让整晚的图像证据**无声消失**——
提示词还在教它用图像证据,技能还在渲染 PNG,只有 debug 日志知道。

**没有图**的那条路仍然是严格 no-op:不产图的技能不该每轮背一句噪音。
"""
from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

from mast.agents._shared.inject import append_new_human

#: 登记表条目 id —— 送达与未送达是两条不同的条目：一条是图像证据，另一条是
#: 「你没看到图，别假装看过」的诚实声明。抓包里必须分得开。
PROMPT_ID_IMAGES = "mw.skill_image.images"
PROMPT_ID_UNDELIVERED = "mw.skill_image.undelivered"

logger = logging.getLogger(__name__)

#: ToolMessage.additional_kwargs 里放图像路径的键。**生产方(skill_adapter)和消费方
#: (本模块)必须用同一个常量** —— 两边各写一遍字面量,拼错一个字符就变成
#: 「生产方在记、消费方读不到」的静默失败(本仓的 ``silent_fallback_wrong_name``
#: 形状)。所以它只有这一处定义,两边都 import 它。
IMAGES_KEY = "mast_images"

#: 一次请求最多带几张图。图是**每轮都重发**的(它挂在本次请求上,不进 state),
#: 所以这个数直接决定每轮的固定开销。2 张 ≈ 700 token(见 moonshot_kimi_vision.md
#: 的估算区间),对 kimi-k3 的 1 M 上下文可忽略;真要报账用官方 estimate-token-count。
MAX_IMAGES_PER_REQUEST = 2

#: 缩略图边长(px)。经 ``render_scan_thumbnail`` 的缓存(path+mtime+size),同一帧
#: 只渲染一次。
_THUMB_PX = 512

#: 单张图的 base64 字节上限。超了就降采样;降不下来就**丢掉这张图**(退化成纯文本),
#: 绝不把一个超大请求发出去。Moonshot 的硬限制是请求体 100 MB,这里留足余量。
_MAX_B64_BYTES = 4 * 1024 * 1024

#: 直接就是图片文件的后缀 —— 这些只需要读+编码,不需要渲染。
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})

#: Nanonis 原始数据 —— 这些要**复用 render_scan_thumbnail**(它已按 path+mtime+size
#: 缓存)。不在这里写第二个渲染器:两份渲染逻辑必然会在某次改动后给出不同的图,
#: 而「同一个问题两个地方回答」是本仓反复踩的形状。
_NANONIS_SUFFIXES = frozenset({".sxm", ".dat", ".3ds"})

#: (path, mtime) → data URI。图片文件读+编码的缓存;Nanonis 那一支的缓存在
#: render_scan_thumbnail 自己身上。
_URI_CACHE: dict[tuple[str, float], str] = {}


def _downscale_png(raw: bytes) -> bytes:
    """把过大的图缩到 ``_THUMB_PX`` 长边。缩不了就原样返回(调用方再判大小)。"""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if max(im.size) <= _THUMB_PX:
                return raw
            im.thumbnail((_THUMB_PX, _THUMB_PX))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 — 缩不了不该带走整次调用
        logger.debug("vision_mw: downscale failed for a %d-byte image: %s",
                     len(raw), exc)
        return raw


def materialize_data_uri(path: str) -> str | None:
    """一个路径 → ``data:image/…;base64,…``,失败返回 None。

    None 的每一种来源都是**良性**的:文件没了、后缀不认识、图太大、依赖缺席。
    调用方一律当作「这张没有」处理,继续发纯文本 —— 看不到图是遗憾,发不出请求是故障。
    """
    if not path:
        return None
    p = Path(path)
    suffix = p.suffix.lower()

    # Nanonis 原始数据:复用既有渲染器(自带 path+mtime+size 缓存,且返回值本来
    # 就已经是带 "data:image/png;base64," 前缀的完整 data URI —— 正好是 Moonshot
    # 要的那个形状,不需要再拼前缀)。
    if suffix in _NANONIS_SUFFIXES:
        try:
            from mast.webui.scan_preview import render_scan_thumbnail

            return render_scan_thumbnail(str(p), size=_THUMB_PX)
        except Exception as exc:  # noqa: BLE001
            logger.debug("vision_mw: render_scan_thumbnail failed for %s: %s", p, exc)
            return None

    if suffix not in _IMAGE_SUFFIXES:
        logger.debug("vision_mw: 不认识的图像后缀 %r — 跳过", suffix)
        return None

    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    key = (str(p), mtime)
    cached = _URI_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        raw = p.read_bytes()
    except OSError as exc:
        logger.debug("vision_mw: 读不到 %s: %s", p, exc)
        return None
    if not raw:
        return None

    # 技能渲染出来的分析图是一整张 matplotlib figure(标题/色标/坐标轴),可以不小。
    # 先缩到与缩略图同一量级再编码。
    raw = _downscale_png(raw)
    if len(raw) * 4 // 3 > _MAX_B64_BYTES:
        logger.info("vision_mw: %s 编码后仍超过 %d 字节 — 本张丢弃(退化成纯文本)",
                    p.name, _MAX_B64_BYTES)
        return None

    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    uri = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
    _URI_CACHE[key] = uri
    if len(_URI_CACHE) > 64:
        _URI_CACHE.pop(next(iter(_URI_CACHE)))
    return uri


def _collect_recent_images(messages: list) -> list[tuple[str, str]]:
    """(skill_name, path) —— 转录里**最近的** ``MAX_IMAGES_PER_REQUEST`` 张,旧→新。

    从后往前扫,因为要的是最新的证据。同一路径只取一次:一帧被两个技能引用过
    (例如先 AnalyzeScanImage 再 AutoProcessScanBatch)不该占两个名额。
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for msg in reversed(messages):
        kind = getattr(msg, "type", None) or type(msg).__name__
        if str(kind).lower() not in ("tool", "toolmessage"):
            continue
        kwargs = getattr(msg, "additional_kwargs", None)
        if not isinstance(kwargs, dict):
            continue
        paths = kwargs.get(IMAGES_KEY)
        if not isinstance(paths, (list, tuple)):
            continue
        skill = str(getattr(msg, "name", "") or "skill")
        for raw_path in paths:
            path = str(raw_path or "")
            if not path or path in seen:
                continue
            seen.add(path)
            found.append((skill, path))
            if len(found) >= MAX_IMAGES_PER_REQUEST:
                return list(reversed(found))
    return list(reversed(found))


class SkillImageMiddleware(AgentMiddleware):
    """把技能产出的图像挂进出站请求(仅当模型支持视觉)。

    同时实现 ``wrap_model_call``(GUI 的 ``graph.stream()``)与
    ``awrap_model_call``(CLI 的 ``await graph.ainvoke()``)。LangChain 1.2 的基类
    异步钩子在只定义了同步钩子时会抛 NotImplementedError,所以异步孪生体是**必需**
    的,不是可选的(与 ``live_state_mw`` / ``prefill_guard_mw`` 同一条理由)。
    """

    def __init__(self, *, max_images: int = MAX_IMAGES_PER_REQUEST):
        super().__init__()
        self._max_images = max(0, int(max_images))

    # ── 内部 ────────────────────────────────────────────────────────
    @staticmethod
    def _deliver(request: Any, messages: list, extra, prompt_id: str) -> Any:
        """把 *extra* 追加进**本次请求**的消息列表。graph state 一个字节都不动。

        *prompt_id* 只用于注入台账（谁塞的这一条），不影响内容。
        """
        # append_new_human 从 request.messages 取基线，而调用方这里给的 messages
        # 可能已经被本方法的调用链改过 —— 先把它写回请求，再追加。
        override = getattr(request, "override", None)
        if callable(override):
            request = override(messages=list(messages))
        else:                                        # pragma: no cover — 老版本
            request.messages = list(messages)
        return append_new_human(request, prompt_id, extra)

    def _apply(self, request: Any) -> Any:
        """给请求挂上图像;挂不上时**说出来**,而不是安静地少发。

        ## 为什么「安静降级」本身就是 bug(2026-08-11 修正)

        第一版这里三种情况全都 ``return request`` —— 与接线前逐字节相同,只留一行
        ``logger.debug``。那看起来是最保守的选择,其实正是这个仓反复栽的**那一个**
        形状:**一个失效,而它看起来和正常工作一模一样。**

        具体路径:``config/overrides/agent_overrides.json`` 允许把某个 agent 换成
        别的模型。有人把仪器 agent 从 kimi-k3 换成 GLM 或 DeepSeek(两家都不收图像)
        之后 —— 提示词还在教它「怀疑双针尖时用图像证据」,技能还在老老实实渲染 PNG,
        而图**一张都没发出去**。模型不知道,用户不知道,证据只在 debug 日志里。
        模型于是照着它根本没看见的东西继续推断。

        所以现在:图挂不上时,**在给模型的那条消息里写明**(纯文本,任何 provider
        都收得下)。判据是「用户看得到」—— 模型收到这句话就会在回话里提到它,
        而不是要有人去翻日志才发现今晚的图全丢了。日志同时升到 warning。

        没有图的那条路**仍然是严格 no-op**:绝大多数技能不产图,不能让它们每一轮
        都背一句噪音。
        """
        if self._max_images <= 0:
            return request
        try:
            messages = list(getattr(request, "messages", None) or [])
            if not messages:
                return request

            pairs = _collect_recent_images(messages)[-self._max_images:]
            if not pairs:
                return request      # 没有图 —— 真 no-op,不加任何东西

            # 能力闸门。放在 materialize 之前:模型看不了的图连读都不该读。
            from mast.agents._shared.models import model_id_of, model_supports_vision

            model_id = model_id_of(getattr(request, "model", None))
            if not model_id or not model_supports_vision(model_id):
                logger.warning(
                    "vision_mw: 模型 %s 不支持图像输入 —— 本轮 %d 张图未发送"
                    "(能力表 config._VISION_MODELS;出处 docs/api_providers/vision_support.md)",
                    model_id, len(pairs),
                )
                return self._deliver(request, messages, HumanMessage(content=(
                    f"[图像未送达] 本轮有 {len(pairs)} 张技能渲染的图像**没有发给你**:"
                    f"当前模型 `{model_id or '未知'}` 不支持图像输入。"
                    "你只拿到了文字摘要。**不要假装看过这些图**,也不要仅凭它们下结论;"
                    "需要看图时,请告诉用户换成支持视觉的模型(如 kimi-k3)。"
                )), PROMPT_ID_UNDELIVERED)

            blocks: list[dict[str, Any]] = []
            shown: list[str] = []
            unread: list[str] = []
            for skill, path in pairs:
                uri = materialize_data_uri(path)
                if not uri:
                    unread.append(Path(path).name)
                    continue
                blocks.append({"type": "image_url", "image_url": {"url": uri}})
                shown.append(f"{skill} → {Path(path).name}")

            if not blocks:
                # 有路径但一张都读不出来(被删/后缀不认/太大)。同样要说出来。
                logger.warning("vision_mw: %d 张图读不出来,未发送: %s",
                               len(unread), ", ".join(unread))
                return self._deliver(request, messages, HumanMessage(content=(
                    f"[图像未送达] 本轮有 {len(unread)} 张图像**没有发给你**:文件读不出来"
                    f"({', '.join(unread)})—— 可能已被移走、删除,或格式不支持。"
                    "你只拿到了文字摘要,**不要假装看过这些图**。"
                )), PROMPT_ID_UNDELIVERED)

            header = (
                "以下是上面工具结果对应的图像("
                + "; ".join(shown)
                + ")。这是**图像证据**——直接看图判断,不要只依赖上面的数字摘要。"
            )
            if unread:
                # 部分失败也要留痕,否则「少了一张」是彻底看不见的。
                header += f"\n注意:另有 {len(unread)} 张读不出来、未发送({', '.join(unread)})。"
            # 文字块在前:模型先知道自己在看什么,再看到像素。
            # 这个 list 形状同时满足 Moonshot 的硬要求:用视觉时 content 必须是数组。
            content: list[dict[str, Any]] = [{"type": "text", "text": header}]
            content.extend(blocks)

            return self._deliver(request, messages, HumanMessage(content=content),
                                 PROMPT_ID_IMAGES)
        except Exception as exc:  # noqa: BLE001
            # 看图是增强,不是必需。任何意外都退回纯文本,绝不带走这次调用。
            logger.debug("SkillImageMiddleware skipped: %s", exc, exc_info=True)
            return request

    # ── 钩子 ────────────────────────────────────────────────────────
    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._apply(request))


__all__ = [
    "IMAGES_KEY",
    "MAX_IMAGES_PER_REQUEST",
    "SkillImageMiddleware",
    "materialize_data_uri",
]
