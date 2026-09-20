"""后台 run 的准入判据：拦的是「无人盯着的模型判断驱动仪器」。

## 这条禁令为什么值得一道测试

`BACKGROUNDABLE` 是一个 frozenset 白名单加一句 `raise ValueError`，它是全框架
**唯一**能让工作脱离 HTTP 请求生命周期活下去的机制上的门。历史上它的注释写的是
「everything EXCEPT instrument_control (the sole hardware agent)」——这句话把
判据说成了「哪个 agent」，于是很容易被读成「后台线程不许碰硬件」。

那个读法是错的，而且代价不小：按它推理，任何多天无人值守的仪器工作都无路可走。
真实的判据是**下一步由什么决定**：

* 确定性代码线程驱动技能 —— 早有在产先例（SafetyWatchdog、ConductDirector），
* 无人盯着时由一次 LLM 采样决定动哪个仪器 —— 这才是被拦的东西。

判据句现在写在 `background_runs.py` 的注释里。**注释会漂**，所以这里同时钉
行为和那句话本身：一个只在 docstring 里活着的判据，和没有判据的区别只是它更
难被发现已经不成立了。
"""

from __future__ import annotations

import inspect

import pytest

from mast.core.background_runs import BackgroundRunManager


def _mgr() -> BackgroundRunManager:
    return BackgroundRunManager(run_fn=lambda *a, **k: None)


def test_the_hardware_agent_cannot_be_detached():
    assert "instrument_control" not in BackgroundRunManager.BACKGROUNDABLE
    mgr = _mgr()
    with pytest.raises(ValueError):
        mgr.spawn(instruction="扫一张图", agents=("instrument_control",))


def test_a_request_mixing_ic_with_others_keeps_only_the_others():
    """混着报名不该整单拒绝，也不该把 IC 偷偷放进去。"""
    mgr = _mgr()
    rec = mgr.spawn(instruction="读文献顺便扫张图",
                    agents=("literature", "instrument_control"))
    assert "instrument_control" not in rec["agents"]
    assert "literature" in rec["agents"]


def test_the_refusal_says_where_unattended_hardware_work_should_go():
    """拒绝要给出路。

    「不行」而不说「那该走哪」，下一个人会去改这张表——本仓在
    「能停不能解 = 死锁」上吃过这个亏。conduct 层就是那条出路。
    """
    mgr = _mgr()
    with pytest.raises(ValueError) as ei:
        mgr.spawn(instruction="过夜扫描", agents=("instrument_control",))
    msg = str(ei.value)
    assert "conduct" in msg, "拒绝消息没有指出无人值守的仪器工作该走哪一层"


def test_the_criterion_is_written_down_next_to_the_whitelist():
    """判据句必须在源码里，而且说的是「模型判断」不是「哪个 agent」。"""
    src = inspect.getsource(BackgroundRunManager)
    head = src[: src.index("def __init__")]
    assert "模型判断" in head or "model decides" in head, (
        "BACKGROUNDABLE 旁边没有写下判据。少了它，这张表读起来像"
        "「后台线程不许碰硬件」——那个读法会把 conduct 那一层一起否掉。"
    )
    # 反向：不许退回成「因为它是硬件 agent」这种同义反复。
    assert "the sole hardware agent, which stays foreground" not in head, (
        "判据退回成了同义反复（「它是硬件 agent 所以它是前台的」）。"
        "要说的是：无人盯着 + 下一步由一次模型采样决定 ⇒ 不许脱离请求。"
    )


@pytest.mark.parametrize("agent", ["literature", "research_director"])
def test_a_read_only_planning_agent_is_allowed_to_be_backgrounded(agent):
    """判据的正面：不碰仪器的角色可以后台跑。

    research_director（科研策划）2026-08-21 落地，进这张表走的是**同一条理由**
    —— 它的工具面里没有执行面，所以「无人盯着时模型会不会驱动仪器」在它身上是
    结构性的「不会」。那半边由
    ``tests/v2/unit/agents/test_research_director.py`` 的结构断言钉着；这里钉的
    是准入本身。两半缺一，这条判据就退化成一份白名单。
    """
    mgr = _mgr()
    rec = mgr.spawn(instruction="查一下 synthetic_sample 的能隙", agents=(agent,))
    assert tuple(rec["agents"]) == (agent,)
