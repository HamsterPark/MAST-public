"""技能名重名:**要么不存在,要么被显式声明**。而且谁赢不许悄悄换人。

## 为什么需要一道闸门而不是一条注释

2026-08-10:``CheckLineQuality`` 的判据被换掉之后,我去查它的调用方,发现
``builtin_composites.py:552`` 的 ``PreScanCheck`` 与 ``prescan_check.py:108`` 的
``PreScanCheck`` **同名**,注册表里赢的是手写那个 —— 也就是说声明式那份里的接线
(包括它传的旧余弦阈值)**根本不会执行**。

这和「零可达调用方」「写了却没人读」是同一族缺陷:**从外面看和正常的一模一样。**

查下来这不是意外,是**写在设计里的**(见 ``builtin_composites`` 模块头):
声明式孪生体是给 builder **编辑/派生**用的,手写版是**运行时基准**,两者共存。
全部 10 个 spec 都被同名手写类遮蔽,这是设计而非事故。

所以这道闸门**不是**「不许重名」——那会把 10 个有意的孪生体判死。它编码的是:

1. 类声明技能之间不许重名(那种一定是事故);
2. spec 之间不许重名;
3. spec 与类同名 **必须**在 ``DECLARATIVE_TWINS`` 里写明,并给出理由;
4. 名单不许发霉:里面的名字必须真的两边都还在;
5. **每个孪生体解析到哪一个,钉死。** 这一条最要紧 —— 注册表按发现顺序
   后来者覆盖,谁赢取决于 import 顺序,而那是可以被一次无关重构悄悄改掉的。
   真要改哪一个赢,应该是一次有意的、看得见的改动。

注册表本身在**类 vs 类**冲突上已经会 warn,
没人看的是 **spec vs 类** 这条跨命名空间的路 —— 那正是这里补的。
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

_DISCOVER = ("mast.skills.builtins", "mast.skills.composite", "mast.skills.paper")

#: spec 与手写类同名的**有意**孪生体 → 为什么允许 + 运行时该由谁赢。
#:
#: 设计出处:``builtin_composites`` 模块头 ——「声明式孪生体供 builder 编辑/派生,
#: 手写版是运行时基准,两者共存」。所以这里每一条的赢家都必须是**手写那个**。
_TWIN_REASON = ("声明式孪生体,供 builder 打开/派生/编辑;手写 Python 版是运行时基准。"
                "设计见 builtin_composites 模块头。")
DECLARATIVE_TWINS: dict[str, str] = {
    "BatchRegionsScan": _TWIN_REASON,
    "ConditionTip": _TWIN_REASON,
    "DemoScanAndSTS": _TWIN_REASON,
    "FullScan": _TWIN_REASON,
    "GridSTS": _TWIN_REASON,
    "PreScanCheck": _TWIN_REASON,
    "ShapeTipOnSurface": _TWIN_REASON,
    "SurveySurface_TileScan": _TWIN_REASON,
    "TipPulse": _TWIN_REASON,
    "TrackDrift_ReferenceScan": _TWIN_REASON,
}

#: 孪生体解析到的模块**必须**是这个前缀(手写版都住在这里)。
_HANDWRITTEN_PREFIX = "mast.skills.composite."
#: 声明式那一份住在这里 —— 它**不该**是任何名字的赢家。
_DECLARATIVE_MODULE = "mast.skills.composite.builtin_composites"


@pytest.fixture(scope="module")
def registry():
    from mast.core.registry import SkillRegistry
    reg = SkillRegistry()
    reg.discover(*_DISCOVER)
    return reg


@pytest.fixture(scope="module")
def spec_names():
    from mast.skills.composite.builtin_composites import reconstructed_composites
    return [s.name for s in reconstructed_composites()]


def test_the_sweep_reaches_both_namespaces(registry, spec_names):
    """自检:一条什么都没梳到的闸门,和「确实没问题」输出一模一样。"""
    assert len(registry.list_skills()) >= 400, "类声明技能太少,discover 姿势多半错了"
    assert len(spec_names) >= 10, f"只找到 {len(spec_names)} 个声明式 spec"


def test_no_two_class_declared_skills_share_a_name(registry):
    """类 vs 类重名一定是事故(复制粘贴 / paper 复用了 builtin 的名字)。"""
    names = [m.name for m in registry.list_skills()]
    dups = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dups, f"这些技能名被多个类声明,一个会静默遮蔽另一个:{dups}"


def test_no_two_specs_share_a_name(spec_names):
    dups = [n for n, c in collections.Counter(spec_names).items() if c > 1]
    assert not dups, f"声明式 spec 内部重名:{dups}"


def test_every_spec_class_collision_is_declared(registry, spec_names):
    """spec 与手写类同名 —— 允许,但**必须写明**。

    没写明的重名 = 有人无意中用了一个已经存在的名字,而它**从外面看完全正常**:
    技能表里有它,调用它也不报错,只是执行的是另一份代码。
    """
    class_names = {m.name for m in registry.list_skills()}
    undeclared = sorted(set(spec_names) & class_names - set(DECLARATIVE_TWINS))
    assert not undeclared, (
        f"这些声明式 spec 与手写技能同名,而没有在 DECLARATIVE_TWINS 里声明:"
        f"{undeclared} —— 其中一份会被静默遮蔽。若是有意的,加进名单并写清理由;"
        f"若不是,改名。")


def test_the_twin_list_has_not_gone_stale(registry, spec_names):
    """名单里的名字必须真的两边都还在 —— 过期的名单在骗人。"""
    class_names = {m.name for m in registry.list_skills()}
    live = set(spec_names) & class_names
    stale = sorted(set(DECLARATIVE_TWINS) - live)
    assert not stale, f"这些名字已经不再是重名了,请从 DECLARATIVE_TWINS 删除:{stale}"
    for name, reason in DECLARATIVE_TWINS.items():
        assert reason and len(reason) > 15, f"{name} 的理由太短,说不清为什么允许"


@pytest.mark.parametrize("name", sorted(DECLARATIVE_TWINS))
def test_the_handwritten_version_is_the_one_that_wins(registry, name):
    """**谁赢,钉死。**

    注册表按发现顺序后来者覆盖,所以"哪一份会真的执行"取决于 import 顺序 ——
    一次无关的重构就能把它换个人,而且**不会有任何症状**:技能还在,调用还成功,
    只是跑的是另一份接线。设计上运行时基准是手写版(见 builtin_composites 模块头),
    这里把它钉住。要改,应该是一次有意的、看得见的改动。
    """
    cls = registry.get(name)
    assert cls is not None, f"{name} 消失了"
    mod = cls.__module__
    assert mod != _DECLARATIVE_MODULE, (
        f"{name} 现在解析到**声明式**那一份({mod})—— 运行时基准应该是手写版。"
        f"这种翻转不会有任何症状,所以必须由闸门抓。")
    assert mod.startswith(_HANDWRITTEN_PREFIX), (
        f"{name} 解析到 {mod},不在手写 composite 包里 —— 请确认这是有意的。")
