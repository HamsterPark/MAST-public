"""TipContextMiddleware injection tests.

镜像 test_instrument_profile_mw.py:always-inject、sync+async 双钩子、渲染失败
不破坏请求。额外钉住这个块存在的理由 —— 偏压极性未声明时要**明说未声明**。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/agents/_shared/test_tip_context_mw.py -x -v
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import types

import pytest
from langchain_core.messages import SystemMessage

from mast.agents._shared.tip_context_mw import TipContextMiddleware
from mast.core import instrument_profile as ip
from mast.core import tip_state


@pytest.fixture(autouse=True)
def _clean():
    ip.set_persist_sink(None)
    ip.set_profile({})
    tip_state.set_current_tip(None)
    yield
    ip.set_persist_sink(None)
    ip.set_profile({})
    tip_state.set_current_tip(None)


def _fake_request(system_message):
    return types.SimpleNamespace(system_message=system_message)


_W_TIP = {
    "id": "t1", "name": "W-etched #1", "material": "W", "fabrication": "etched",
    "form": "stm_wire", "wire_diameter_mm": 0.25,
    "installed_at": "2026-07-28T09:00:00",
}


def test_injects_even_with_nothing_registered():
    """「针尖未登记」「偏压极性未声明」本身就是模型需要知道的事实。"""
    mw = TipContextMiddleware()
    out = mw._apply(_fake_request(SystemMessage(content="BASE PROMPT")))
    text = out.system_message.content
    assert text.startswith("BASE PROMPT")
    assert "针尖" in text
    assert "未登记" in text


def test_creates_system_message_when_absent():
    mw = TipContextMiddleware()
    out = mw._apply(_fake_request(None))
    assert isinstance(out.system_message, SystemMessage)
    assert "当前针尖" in out.system_message.content


def test_injects_the_registered_tip():
    tip_state.set_current_tip(_W_TIP)
    mw = TipContextMiddleware()
    text = mw._apply(_fake_request(SystemMessage(content="B"))).system_message.content
    assert "W-etched #1" in text
    assert "钨" in text and "电化学腐蚀" in text


def test_unknown_bias_polarity_is_stated_not_assumed():
    """默认 unknown 时,模型必须被告知「不知道」,而不是让它按常规约定断言。"""
    mw = TipContextMiddleware()
    text = mw._apply(_fake_request(None)).system_message.content
    assert "未声明" in text and "不要断言" in text


def test_declared_bias_polarity_reaches_the_model():
    ip.set_profile({"bias_applied_to": "tip"})
    mw = TipContextMiddleware()
    text = mw._apply(_fake_request(None)).system_message.content
    assert "针尖" in text and "反号" in text


def test_preamp_gain_reaches_the_model():
    ip.set_profile({"preamp_model": "FEMTO DLPCA-200", "preamp_gain_v_per_a": 1e9})
    mw = TipContextMiddleware()
    text = mw._apply(_fake_request(None)).system_message.content
    assert "DLPCA-200" in text
    assert "1e+09" in text or "1e9" in text


def test_qplus_tip_carries_the_poke_note_and_it_says_poking_is_routine():
    """qPlus 那一段要真的注入到模型看到的 system message 里。

    ── 2026-08-17 这条翻过面

    原名 ``test_qplus_tip_carries_the_poke_warning``,断言的是 ``allow_on_qplus``
    在正文里 —— 也就是钉着「扎针需要显式许可」那个前提。

    **那个前提是错的**,现场逐字:「我不知道哪里说了 qPlus tip 不能扎,
    实际上 nm 尺度下完全可以扎,所以这个误解从源头上污染了我们的系统。」

    这一层(注入中间件)是这段话真正到达模型的那一跳,所以它要钉的不是措辞,
    而是**那一段确实被注进去了**,并且注进去的是新的那一段。
    """
    from mast.core.tip_state import _QPLUS_POKE_NOTE

    tip_state.set_current_tip(
        {"id": "q", "name": "q1", "material": "PtIr", "fabrication": "cut",
         "form": "qplus", "qplus_sensor_model": "TF-32k", "qplus_f0_hz": 32768.0})
    mw = TipContextMiddleware()
    text = mw._apply(_fake_request(None)).system_message.content
    # 不写死中文 —— 直接对渲染层那份常量,措辞再改也不会误报。
    assert _QPLUS_POKE_NOTE.strip() in text
    assert "allow_on_qplus" not in text, (
        "又把「要显式许可」那句话注给模型了 —— 这正是让它回来问「要不要扎针」的原因")
    assert "32.77 kHz" in text or "3.2768e+04" in text


def test_render_failure_is_a_noop():
    def boom():
        raise RuntimeError("holder read failed")

    mw = TipContextMiddleware(get_tip_fn=boom)
    out = mw._apply(_fake_request(SystemMessage(content="BASE")))
    assert out.system_message.content == "BASE"      # never crashes a run


def test_async_hook_injects():
    import asyncio

    tip_state.set_current_tip(_W_TIP)
    mw = TipContextMiddleware()
    captured = {}

    async def handler(req):
        captured["sm"] = req.system_message
        return "RESP"

    async def drive():
        return await mw.awrap_model_call(
            _fake_request(SystemMessage(content="B")), handler)

    assert asyncio.run(drive()) == "RESP"
    assert "W-etched #1" in captured["sm"].content


def test_sync_hook_injects():
    mw = TipContextMiddleware()
    captured = {}

    def handler(req):
        captured["sm"] = req.system_message
        return "RESP"

    assert mw.wrap_model_call(
        _fake_request(SystemMessage(content="B")), handler) == "RESP"
    assert "当前针尖" in captured["sm"].content


def test_registered_in_the_prompt_inventory():
    """注入源不进 registry = UI 里看不见、查不到实际注入了什么。

    registry 是「清单 + 覆写 + 抓包」而不是调度器,所以接了中间件却忘了登记不会
    报错 —— 只是这一块从上下文查看器里凭空消失。"""
    from mast.prompts.registry import get_entry, render_default

    entry = get_entry("mw.tip_context")
    assert entry is not None, "mw.tip_context 未登记进 prompts registry"
    text, err = render_default(entry)
    assert not err, f"注入块渲染失败: {err}"
    assert "当前针尖" in text


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
