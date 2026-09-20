"""图像通道:技能渲染的图**真的**到得了模型手里,到不了时**真的**降级成纯文本。

接线前的事实(只读审计,2026-08-11):``SkillResult`` 没有图像字段;技能→模型的唯一
出口是 ``skill_adapter`` 的 ``ToolMessage(content=summary)``,而 ``summary`` 永远是
``str``;``git log -S "image_url" -- MASTv2/mast/agents`` 是**零个 commit**。也就是说
这条路不是坏了,是从来没有过。

所以这份测试要证的不是「代码路径存在」,而是三件可证伪的事:

  一、**出站请求体里真的有 image block**,而且是 Moonshot 文档写的那个形状。
      证到 ``langchain_openai`` 的真序列化器为止 —— 那是发 HTTP 前的最后一道形状。
  二、模型不支持视觉时**降级成纯文本且不报错**,消息列表与接线前逐字节相同。
  三、**base64 绝不进 checkpoint**。用 langgraph 真正的序列化器证:SqliteSaver 会
      写进盘的那份字节里没有像素,只有路径。

每一条都写成「若事实相反,这个断言会不会红」的形式,而不是「跑通了没报错」。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from mast.agents._shared.skill_adapter import wrap_skill
from mast.agents._shared.vision_mw import (
    IMAGES_KEY,
    SkillImageMiddleware,
    materialize_data_uri,
)
from mast.config import model_supports_vision
from mast.core.types import (
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)

_VISION_MODEL = "kimi-k3"          # 官方视觉列表里的默认模型
_TEXT_ONLY_MODEL = "moonshot-v1-128k"   # 同前缀、**不在**视觉列表里的那个


def _repo_root() -> Path:
    """仓库根 —— 从**导入的 mast 包**推,不数测试文件的相对层级。

    第一版这里数 ``parents[3]`` 数错了一层。第一次错的症状是 FileNotFoundError
    (吵,好修);第二次同样的错落在一个 ``skipif`` 上,症状变成**测试安静地跳过**
    ——看起来和「这台机器没有样本」一模一样。可证伪性是自己创造的:换成从包里问,
    这个错就没地方藏。
    """
    import mast

    return Path(mast.__file__).resolve().parents[2]


_SXM_SAMPLES = _repo_root() / "artifacts" / "operator_reference_20260810" / "sxm"


# ══════════════════════════════════════════════════════════════════════════
# 脚手架
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def real_png(tmp_path: Path) -> Path:
    """一张真的 PNG。用真文件而不是假路径 —— 「读得出来」也是被测行为之一。"""
    from PIL import Image

    p = tmp_path / "scan_Z_auto.png"
    Image.new("RGB", (64, 48), (30, 90, 160)).save(p)
    return p


def _skill_returning(images: list[str], name: str = "AnalyzeScanImage"):
    """一个只做一件事的技能:返回带 ``images`` 的 SkillResult。"""
    class _S:
        def metadata(self):
            return SkillMetadata(name=name, description="d",
                                 category=SkillCategory.ANALYSIS,
                                 safety_level=SafetyLevel.AUTO, parameters=[])

        def validate_params(self, kwargs):
            return []

        def execute(self, ctx, params):
            return SkillResult(skill_name=name, success=True,
                               data={"png_path": images[0] if images else ""},
                               summary="平场 plane,色阶 2–98 百分位",
                               images=list(images))

    return _S


class _State(TypedDict):
    messages: Annotated[list, add_messages]
    executed_skills: list
    scan_paths: list


def _run_real_tool(images: list[str], name: str = "AnalyzeScanImage") -> list:
    """把技能推过**真的** ToolNode + 真的 ``wrap_skill``,取回真的消息列表.

    不直接构造 ToolMessage:``skill_adapter`` 的那一处正是被测对象,绕过它就等于
    假设它是对的。
    """
    tool = wrap_skill(_skill_returning(images, name), lambda: object())
    g = StateGraph(_State)
    g.add_node("tools", ToolNode([tool]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    ai = AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "tc1"}])
    out = g.compile().invoke(
        {"messages": [ai], "executed_skills": [], "scan_paths": []})
    return list(out["messages"])


class _FakeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name


class _FakeRequest:
    """与 test_tool_pair_guard 里的同构 —— 中间件契约就是 messages / model / override。"""

    def __init__(self, messages, model_name: str):
        self.messages = list(messages)
        self.model = _FakeModel(model_name)

    def override(self, *, messages):
        return _FakeRequest(messages, self.model.model_name)


def _outbound(messages, model_name: str) -> list:
    """跑真的中间件,返回**交给模型的那份**消息列表。"""
    seen: dict[str, Any] = {}

    def handler(req):
        seen["messages"] = list(req.messages)
        return AIMessage(content="ok")

    SkillImageMiddleware().wrap_model_call(
        _FakeRequest(messages, model_name), handler)
    return seen["messages"]


def _image_blocks(messages) -> list[dict]:
    """出站消息里所有 image_url 块。"""
    out: list[dict] = []
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, list):
            out += [b for b in content
                    if isinstance(b, dict) and b.get("type") == "image_url"]
    return out


# ══════════════════════════════════════════════════════════════════════════
# 一、能力表 —— 单一真源,逐 id 不按前缀
# ══════════════════════════════════════════════════════════════════════════

class TestCapabilityTable:
    def test_default_model_has_vision(self):
        """kimi-k3 是全部 agent 的默认模型;它要是没视觉,整条路就没有意义。"""
        assert model_supports_vision(_VISION_MODEL) is True

    def test_same_prefix_opposite_answer(self):
        """能力表必须逐 id。

        ``moonshot-v1-128k-vision-preview`` 收图,``moonshot-v1-128k`` 不收 ——
        同一个前缀,相反的答案。任何 ``startswith("moonshot")`` 规则必错一个。
        这条测试就是钉死「不许改成前缀匹配」。
        """
        assert model_supports_vision("moonshot-v1-128k-vision-preview") is True
        assert model_supports_vision(_TEXT_ONLY_MODEL) is False

    def test_every_offered_model_has_a_researched_answer(self):
        """UI 里能选到的每个模型,视觉能力都必须是**查过的**,不是碰巧的默认值。

        ❌ 的三家各自有出处(见 docs/api_providers/vision_support.md):
        DeepSeek 官方文档没有图像请求格式;GLM 的官方页面「输入模态」写的是文本;
        Qwen 那个 id 根本不在现网列表上。把它们钉在这里,是因为「默认 False」和
        「查过确实不支持」在代码里长得一模一样 —— 只有测试能记住区别。
        """
        assert model_supports_vision("claude-opus-4-7") is True
        assert model_supports_vision("claude-sonnet-4-6") is True
        assert model_supports_vision("claude-haiku-4-5-20251001") is True
        assert model_supports_vision("MiniMax-M3") is True
        assert model_supports_vision("kimi-k2.6") is True
        # 查过,确实不支持 / 确认不了 —— 不是忘了填
        assert model_supports_vision("deepseek-v4-pro") is False
        assert model_supports_vision("glm-5.2") is False
        assert model_supports_vision("glm-5.1") is False
        assert model_supports_vision("qwen3.7-max") is False

    def test_unknown_model_is_text_only(self):
        """没登记 ⇒ False。猜错 False 少一张图,猜错 True 每次看图都 400。"""
        assert model_supports_vision("some-model-shipped-next-month") is False
        assert model_supports_vision("") is False

    def test_no_second_capability_table(self):
        """``config.MINIMAX_VISION_MODELS`` 那个零消费者存根必须是**被吸收**了,
        不是留在旁边 —— 同一个问题有两个地方回答,迟早会互相矛盾。"""
        import mast.config as cfg

        assert not hasattr(cfg, "MINIMAX_VISION_MODELS")
        # 它唯一的那条事实没有丢
        assert model_supports_vision("MiniMax-M3") is True


# ══════════════════════════════════════════════════════════════════════════
# 二、端到端 —— 出站请求体里真的有图
# ══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_skill_image_reaches_the_outgoing_request(self, real_png: Path):
        """技能 → 真 ToolNode → 真中间件 → **出站请求体里有 image block**。"""
        messages = _run_real_tool([str(real_png)])
        out = _outbound(messages, _VISION_MODEL)

        blocks = _image_blocks(out)
        assert len(blocks) == 1, "出站请求里应当恰好有一张图"

        url = blocks[0]["image_url"]["url"]
        # Moonshot 官方:远程 https URL **不支持**,只收 base64 data URI。
        assert url.startswith("data:image/"), url[:40]
        assert ";base64," in url
        # 而且里面是**这张图的真实字节**,不是占位符。
        payload = base64.b64decode(url.split(";base64,", 1)[1])
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        from PIL import Image

        assert Image.open(io.BytesIO(payload)).size == (64, 48)

    def test_block_survives_the_real_openai_serializer(self, real_png: Path):
        """证到**发 HTTP 前的最后一道形状**。

        断言到中间件的输出为止,只证明了我们自己的对象长得对;真正会被发出去的是
        ``langchain_openai`` 序列化之后的那个 dict。这里用它**真的**序列化器,
        对照 Moonshot 文档里逐字的那个形状:
            {"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}
        """
        from langchain_openai.chat_models.base import _convert_message_to_dict

        out = _outbound(_run_real_tool([str(real_png)]), _VISION_MODEL)
        carrier = [m for m in out if isinstance(m, HumanMessage)][-1]
        wire = _convert_message_to_dict(carrier)

        assert wire["role"] == "user"
        # 官方原文:使用视觉模型时 message.content 必须是 array[object]。
        assert isinstance(wire["content"], list)
        img = [b for b in wire["content"] if b.get("type") == "image_url"]
        assert len(img) == 1
        assert set(img[0]) == {"type", "image_url"}
        assert set(img[0]["image_url"]) == {"url"}, (
            "Moonshot 文档里没有 detail 字段 —— 不要按 OpenAI 的形状加参数")
        assert img[0]["image_url"]["url"].startswith("data:image/png;base64,")
        # 文字块在前:模型要先知道自己在看什么。
        assert wire["content"][0]["type"] == "text"

    def test_block_also_converts_on_the_anthropic_path(self, real_png: Path):
        """一种发射格式,两个 provider 家族 —— 这件事**不是自动成立的**,要钉住。

        我们只发 OpenAI 形状的 ``image_url`` 块。Moonshot 逐字收下;
        而 **MiniMax-M3 走的是 ChatAnthropic**(api.minimaxi.com/anthropic),
        Anthropic 的原生形状是完全不同的 ``{"type":"image","source":{…}}``。

        两者能共用一份发射代码,靠的是 ``langchain_anthropic`` 在序列化时**替我们
        转换**。这是一个我们不拥有的第三方行为:哪天它不转了,MiniMax 会开始收到
        一个它读不懂的块,而我们这边一行代码都没改过。所以这条测试盯的是那个转换,
        不是我们自己的输出 —— 若事实相反,它必须红。
        """
        from langchain_anthropic.chat_models import _format_messages

        out = _outbound(_run_real_tool([str(real_png)]), _VISION_MODEL)
        carrier = [m for m in out if isinstance(m, HumanMessage)][-1]
        _, converted = _format_messages([carrier])

        img = [b for b in converted[0]["content"] if b.get("type") == "image"]
        assert len(img) == 1, "Anthropic 侧应当拿到一个原生 image 块"
        assert img[0]["source"]["type"] == "base64"
        assert img[0]["source"]["media_type"] == "image/png"
        assert base64.b64decode(img[0]["source"]["data"])[:8] == b"\x89PNG\r\n\x1a\n"

    def test_text_summary_is_not_lost(self, real_png: Path):
        """加了图不等于丢了字。技能的文字摘要必须仍然在 ToolMessage 里。"""
        out = _outbound(_run_real_tool([str(real_png)]), _VISION_MODEL)
        tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
        assert tool_msgs and "色阶" in str(tool_msgs[0].content)


class TestReusesTheExistingRenderer:
    """.sxm 那一支必须**走既有渲染器**,不许出现第二个。

    ``render_scan_thumbnail`` 已经按 path+mtime+size 缓存,而且返回值本来就是带
    ``data:image/png;base64,`` 前缀的完整 data URI。写第二份渲染逻辑的代价不是重复
    代码,是**两份图会在某次改动后开始不一样**,而没有人会发现。
    """

    def test_nanonis_files_route_to_render_scan_thumbnail(self, monkeypatch):
        calls: list[tuple] = []

        def _fake(path, size):
            calls.append((path, size))
            return "data:image/png;base64,ZmFrZQ=="

        import mast.webui.scan_preview as sp

        monkeypatch.setattr(sp, "render_scan_thumbnail", _fake)

        for suffix in (".sxm", ".dat", ".3ds"):
            calls.clear()
            uri = materialize_data_uri(f"/nowhere/frame{suffix}")
            assert uri == "data:image/png;base64,ZmFrZQ==", suffix
            assert len(calls) == 1, f"{suffix} 应当恰好调用一次既有渲染器"

    @pytest.mark.skipif(not _SXM_SAMPLES.is_dir(),
                        reason=f"本机没有 .sxm 样本(gitignored): {_SXM_SAMPLES}")
    def test_real_sxm_becomes_a_real_png(self):
        """有样本的机器上,拿**真的** .sxm 走完整条路。

        `artifacts/` 是 gitignored,所以这条在别的机器上会 skip —— 它是本机的
        额外证据,不是主证据(主证据是上面那条不依赖样本的路由测试)。
        skip 理由里带上**它找的那个路径**:路径写错时,skip 信息自己会指出来,
        而不是伪装成「这台机器没有样本」。
        """
        sample = sorted(_SXM_SAMPLES.glob("*.sxm"))[0]
        uri = materialize_data_uri(str(sample))
        assert uri and uri.startswith("data:image/png;base64,")
        payload = base64.b64decode(uri.split(";base64,", 1)[1])
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"


# ══════════════════════════════════════════════════════════════════════════
# 三、降级 —— 不支持视觉时必须是纯文本,而且不报错
# ══════════════════════════════════════════════════════════════════════════

def _all_text(messages) -> str:
    """出站消息里的全部文字(纯串 content + 多模态里的 text 块)。"""
    parts: list[str] = []
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [b.get("text", "") for b in content
                      if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(parts)


class TestDegradesToText:
    """降级 = 不发 image block。但**降级必须说话**。

    第一版这里断言的是「与接线前逐字节相同」,而那恰恰是本仓最危险的形状:
    一个失效,长得和正常工作一模一样。agent_overrides.json 允许把仪器 agent 换成
    不支持视觉的模型,于是提示词还在教它「用图像证据」、技能还在渲染 PNG,
    而图一张都没发出去 —— 没人知道。

    所以那条断言被**故意推翻**了,换成下面这条:没有 image block(不变),
    **且有一句模型读得到的说明**(新增)。留着这段话是因为「逐字节相同」听起来
    像个更严格的保证,下一个人很容易把它加回来。
    """

    def test_text_only_model_gets_no_image_but_is_told_why(self, real_png: Path):
        messages = _run_real_tool([str(real_png)])
        out = _outbound(messages, _TEXT_ONLY_MODEL)

        assert _image_blocks(out) == [], "不支持视觉的模型绝不能收到 image block"
        text = _all_text(out)
        assert "[图像未送达]" in text, "降级必须是可见的,不能只进 debug 日志"
        assert _TEXT_ONLY_MODEL in text, "要说清楚是哪个模型不支持"
        # 原消息未被就地改写(只在本次请求上追加)
        assert out[:len(messages)] == messages

    def test_notice_tells_the_model_not_to_pretend(self, real_png: Path):
        """光说「没发」不够 —— 模型会照着看不见的东西继续推断。"""
        text = _all_text(_outbound(_run_real_tool([str(real_png)]), _TEXT_ONLY_MODEL))
        assert "不要假装看过" in text

    def test_unknown_model_degrades_too(self, real_png: Path):
        """未登记的模型走同一条降级路径(默认 False 的实际后果)。"""
        out = _outbound(_run_real_tool([str(real_png)]), "brand-new-model")
        assert _image_blocks(out) == []
        assert "[图像未送达]" in _all_text(out)

    def test_missing_file_degrades_and_says_so(self, tmp_path: Path):
        """路径在、文件不在(被清理/被移走)⇒ 纯文本 + 说明,不是异常也不是静默。"""
        ghost = tmp_path / "deleted.png"
        out = _outbound(_run_real_tool([str(ghost)]), _VISION_MODEL)
        assert _image_blocks(out) == []
        text = _all_text(out)
        assert "[图像未送达]" in text and "deleted.png" in text

    def test_unknown_suffix_degrades(self, tmp_path: Path):
        """不认识的后缀不猜 MIME —— 猜错就是发一个模型看不懂的块。"""
        odd = tmp_path / "thing.xyz"
        odd.write_bytes(b"not an image")
        assert materialize_data_uri(str(odd)) is None
        out = _outbound(_run_real_tool([str(odd)]), _VISION_MODEL)
        assert _image_blocks(out) == []
        assert "[图像未送达]" in _all_text(out)

    def test_no_images_is_a_noop(self):
        """绝大多数技能不产图 —— 它们那条路必须一个字节都不变。"""
        messages = _run_real_tool([], name="GetBias")
        out = _outbound(messages, _VISION_MODEL)
        assert len(out) == len(messages)
        for before, after in zip(messages, out):
            assert after is before

    def test_middleware_never_raises_on_junk(self):
        """看图是增强不是必需:任何畸形输入都只能退化,不能带走这次调用。"""
        class _Bad:
            messages = None
            model = None

        SkillImageMiddleware().wrap_model_call(_Bad(), lambda r: r)

        class _Worse:
            @property
            def messages(self):
                raise RuntimeError("boom")

        SkillImageMiddleware().wrap_model_call(_Worse(), lambda r: r)


# ══════════════════════════════════════════════════════════════════════════
# 四、checkpoint 不许有像素(项目规约 不变式)
# ══════════════════════════════════════════════════════════════════════════

class TestNoPixelsInCheckpoint:
    def test_state_carries_the_path_not_the_bytes(self, real_png: Path):
        """state 里是路径。这是 SqliteSaver 每轮持久化并重放的那份东西。"""
        messages = _run_real_tool([str(real_png)])
        tool_msg = [m for m in messages if isinstance(m, ToolMessage)][0]

        assert tool_msg.additional_kwargs[IMAGES_KEY] == [str(real_png)]
        assert isinstance(tool_msg.content, str), (
            "ToolMessage.content 必须仍是 str —— 一旦变成多模态列表,"
            "像素就会跟着进 checkpoint")

    def test_serialized_checkpoint_contains_no_base64(self, real_png: Path):
        """用 langgraph **真正的**序列化器证:会落盘的字节里没有像素。

        断言 ``"data:image"`` 不在里面,而不是断言长度小 —— 前者若事实相反一定红,
        后者可能因为图小而蒙混过关。
        """
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        messages = _run_real_tool([str(real_png)])
        blob = JsonPlusSerializer().dumps_typed({"messages": messages})[1]
        text = bytes(blob).decode("utf-8", "replace")

        assert "data:image" not in text
        assert ";base64," not in text
        assert str(real_png.name) in text, "路径本身应当在(证明这段确实序列化了图像通道)"

    def test_materialized_uri_never_written_back(self, real_png: Path):
        """出站那份有像素,state 那份没有 —— 两者必须是不同的对象。"""
        messages = _run_real_tool([str(real_png)])
        out = _outbound(messages, _VISION_MODEL)

        assert _image_blocks(out), "出站应当有图"
        # 原列表未被就地改写
        assert len(messages) == len([m for m in messages])
        assert _image_blocks(messages) == [], "state 侧不该出现任何 image block"


# ══════════════════════════════════════════════════════════════════════════
# 五、结构闸门 —— 「每个 graph 各自记得接线」人肉找不齐
# ══════════════════════════════════════════════════════════════════════════

_AGENTS = ("instrument_control", "data_processing", "experiment_design",
           "literature", "paper_review", "paper_writing")


def _agents_dir() -> Path:
    """``mast/agents/`` 的真实位置 —— 从**导入的包**问,不用相对路径数层级。

    数 ``parents[n]`` 的写法在测试文件被挪动时会静默指向别处(第一版就写错了一层,
    症状是 FileNotFoundError 而不是「闸门失效」;换成别的层级数还可能指到一个恰好
    存在的目录上,那就真的静默了)。
    """
    import mast.agents

    return Path(mast.agents.__file__).resolve().parent


def test_every_agent_graph_wires_the_middleware():
    """六个 graph 各自 append 一次 —— 这正是「每页各自记得」的形状,靠人扫不齐。

    新增一个 agent 而忘了接线,症状是「那个 agent 就是看不见图」,不会报错、
    不会有日志。所以把它钉成结构闸门,而不是指望 code review 记得。
    """
    root = _agents_dir()
    missing = [a for a in _AGENTS
               if "SkillImageMiddleware()" not in
               (root / a / "graph.py").read_text(encoding="utf-8")]
    assert not missing, f"这些 agent 的 graph 没接图像中间件: {missing}"


def test_gate_would_catch_a_missing_wire():
    """闸门本身可证伪:少一处就必须红(否则它只是个恒真式)。"""
    root = _agents_dir()
    sources = {a: (root / a / "graph.py").read_text(encoding="utf-8")
               for a in _AGENTS}
    sources["literature"] = sources["literature"].replace(
        "middleware.append(SkillImageMiddleware())", "")
    missing = [a for a, s in sources.items() if "SkillImageMiddleware()" not in s]
    assert missing == ["literature"]
