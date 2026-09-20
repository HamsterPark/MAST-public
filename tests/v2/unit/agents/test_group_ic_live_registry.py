"""群聊里的 IC 必须拿到**活的**注册表。

## 这条 bug 的形状

``build_ic`` 的 ``registry`` 参数不传就回落 ``discover_instrument_skills()`` ——
那只 walk ``skills.builtins`` + ``skills.composite`` **两个包**。而运行期注册进来
的东西(builder 页做的声明式 composite、custom .py、覆盖层、以及现在 agent 自己
铸的技能)都是**动态类**,不在任何包里,``discover`` 扫不到。

私聊一直传的是 ``self._registry``,群聊一直没传。于是同一个 agent 两个入口两套
能力,而且**两边都不报错**:群聊 IC 只是「没有那个技能」,看起来就像它从来没被
造出来过。这一组把两半都钉住 —— 接线本身,和它的效果。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
sys.path.insert(0, str(_REPO / "tests" / "v2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import inspect  # noqa: E402

from _forge_fixtures import StubCtx, fresh_registry, two_step_spec  # noqa: E402


def _spec_registry():
    """A registry that also holds one RUNTIME-registered declarative composite."""
    from mast.skills.composite.loader import register_spec
    from mast.skills.composite.spec import CompositeSpec
    reg = fresh_registry()
    register_spec(reg, CompositeSpec.from_dict(two_step_spec("RuntimeOnly")))
    return reg


def _tool_names(registry):
    from mast.agents.instrument_control.tools import build_tools
    return {t.name for t in build_tools(None, (lambda: StubCtx()),
                                        registry=registry, targets=())}


# ── 效果 ────────────────────────────────────────────────────────────

def test_a_runtime_registered_composite_reaches_the_tool_table(tmp_path):
    assert "RuntimeOnly" in _tool_names(_spec_registry())


def test_the_discover_fallback_cannot_see_it():
    """把这条 bug 的**机理**钉下来。

    没有它,上一条测试可能因为别的原因绿(比如 discover 恰好也扫得到),那样
    「必须传 registry」这个结论就没有证据。
    """
    from mast.agents.instrument_control.tools import discover_instrument_skills
    assert not discover_instrument_skills().has("RuntimeOnly")
    assert "RuntimeOnly" not in _tool_names(None)


# ── 接线 ────────────────────────────────────────────────────────────

def test_the_orchestrator_build_accepts_an_instrument_registry():
    from mast.agents.orchestrator.graph import build
    assert "instrument_registry" in inspect.signature(build).parameters


def test_the_orchestrator_hands_that_registry_to_build_ic():
    """源码级断言:传参这一步被删掉的话,症状是「群聊里技能不见了」——没有报错。"""
    from srcref import source_of

    from mast.agents.orchestrator.graph import build
    src = source_of(build)
    assert "registry=instrument_registry" in src, (
        "orchestrator 没有把 instrument_registry 传给 build_ic —— "
        "群聊 IC 会静默回落到 discover(),运行期注册的技能全部消失")


def test_the_runtime_passes_its_live_registry_to_the_orchestrator():
    from srcref import source_of

    from mast.core.runtime import CoreRuntime
    src = source_of(CoreRuntime._build_orchestrator_impl)
    assert "instrument_registry=self._registry" in src, (
        "runtime 建群聊图时没传活的注册表")


def test_the_private_chat_path_still_passes_one():
    """私聊那一半本来就是对的 —— 修群聊时别把它碰坏。"""
    from srcref import source_of

    from mast.core.runtime import CoreRuntime
    src = source_of(CoreRuntime)
    assert "registry=self._registry" in src
