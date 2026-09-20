"""把「保存了管理员覆写」接到「东西真的换了」上。

KNOWN_ISSUES §1.1：``ConfigOverrideRegistry.register_reload_hook`` 在整个生产代码
里**零订阅者**，``signal_reload()`` 是打进一个空列表的。写入接口报
``reloaded=True``（一个字面量），而四个消费者一个都没换。v6.0.1 把**报告**改成了
说实话；这个模块补上**机制**。

四个消费者，三种不同的刷新方式
==============================

一份 ``SafetyLimits`` 覆写要生效，得让下面每一处都换掉。它们不是一类东西，所以没有
一个统一的「reload」能覆盖：

============================== ================================ =====================
消费者                          刷新方式                          本模块做什么
============================== ================================ =====================
``SafetyGuard``（手动执行路径）  重新合并即可                      直接 ``reload()``
``ExecutionContext._safety_    每次建 context 现造一个新 guard    **本来就是活的**
guard``（composite 子步）                                        （无需干预）
私聊 agent 图                    工具 schema 在 build 时定死       ``engine.invalidate()``
                                → 必须重建                        → 下一回合重建
群聊 orchestrator 图             同上                              后台线程重建
============================== ================================ =====================

**为什么 agent 侧非重建不可**：``skill_adapter._schema_from_metadata`` 把当时的包络
写成 pydantic 的 ``Field(ge=, le=)``。那是真约束，不是注释 —— 放宽覆写之后，旧
schema 会在 SafetyGate 看到参数**之前**就把它拒掉。所以对 agent 而言，只重新合并
gate 的限值是不够的：放宽根本到不了模型手上，收紧倒是能立刻拦住（旧 schema 更宽 →
放行 → 新 gate 拦下）。换句话说，只做一半的失效方向是**保守**的，但仍然不是「已
生效」。

**为什么可以不同步到「正在跑的那一回合」**：一个已经在飞的 turn 持有它自己的图对象，
重建只影响下一回合。这件事重启也做不到（重启会直接杀掉那一回合），所以它不是
``restart_required`` 要回答的问题。

诚实
====
``agent_refresh_pending()`` 报的是**观察到的调度结果**，不是意图：任务运行中导致
orchestrator 没重建、或者根本没接线，它就报 True。这条纪律是这一整条链最初出错的
地方（``reloaded=True`` 是字面量），所以不重犯：**不知道就报 None，别报 False。**
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# orchestrator 侧的调度结果。写成常量是因为它会进日志和判定分支，
# 拼错一个字母就是一次静默的语义漂移。
ORCH_REBUILDING = "rebuilding"          # 后台重建已启动
ORCH_ABSENT = "absent"                  # 没有 orchestrator（离线/纯 API 模式）
ORCH_SKIPPED_TASK_ACTIVE = "skipped_task_active"   # 任务运行中，刻意不重建
ORCH_FAILED = "failed"                  # 调度或重建抛了
ORCH_PENDING_QUEUED = "pending_queued"  # 任务运行中，已排队，任务结束时自动补做


@dataclass(frozen=True)
class RefreshOutcome:
    """上一次 override 保存之后，各条路径实际发生了什么。"""

    guard_reloaded: bool          # 手动执行路径的 SafetyGuard 是否真的重新合并了
    chat_invalidated: bool        # 私聊图缓存是否已清（下一回合重建）
    orchestrator: str             # 上面那四个常量之一
    at: float                     # time.time()

    @property
    def agent_path_pending(self) -> bool:
        """agent 侧是不是还没跟上（True ＝ 现在还没生效）。

        ``ORCH_PENDING_QUEUED`` **也算 pending**。排队了不等于生效了 —— 这正是
        这个模块开头那段「不知道就报 None，别报 False」的同一条纪律：一个已经
        安排好、但还没发生的事，报 False 就是在替未来打包票。
        """
        if not self.chat_invalidated:
            return True
        return self.orchestrator in (ORCH_SKIPPED_TASK_ACTIVE, ORCH_FAILED,
                                     ORCH_PENDING_QUEUED)

    def describe(self) -> str:
        bits = [
            f"手动路径={'已刷新' if self.guard_reloaded else '未刷新'}",
            f"私聊图={'下一回合重建' if self.chat_invalidated else '未清缓存'}",
            f"群聊图={self.orchestrator}",
        ]
        return "；".join(bits)


# 模块级状态。故意用弱引用持 runtime —— 这个模块不该让一个 CoreRuntime 活得比它
# 自己的所有者更久（测试里会反复建 runtime）。
_runtime_ref: "weakref.ReferenceType[Any] | None" = None
_last: RefreshOutcome | None = None
_lock = threading.Lock()


def _reload_guard(rt: Any) -> bool:
    """手动执行路径：重新合并 SafetyGuard 的限值 + 全局检查。"""
    guard = getattr(rt, "_safety", None)
    reload_fn = getattr(guard, "reload", None)
    if not callable(reload_fn):
        return False
    try:
        reload_fn()
        return True
    except Exception:  # noqa: BLE001 — 一条路径失败不能挡住其余两条
        logger.exception("safety guard reload failed")
        return False


def _invalidate_chat(rt: Any) -> bool:
    """私聊 agent 图：清缓存，下一回合用新包络重建工具 schema。

    对正在跑的那一回合无影响 —— 它持有的是图对象本身，不是缓存里的槽位。
    """
    eng = getattr(rt, "_conv_engine", None)
    invalidate = getattr(eng, "invalidate", None)
    if not callable(invalidate):
        return False
    try:
        invalidate()
        return True
    except Exception:  # noqa: BLE001
        logger.exception("conversation engine invalidate failed")
        return False


def _rebuild_orchestrator(rt: Any) -> str:
    """群聊 orchestrator 图：走既有的后台重建通道。

    复用 ``_request_composite_rebuild`` 而不是自己起线程：它已经处理了三件我们
    同样需要的事 —— 任务运行中就不动、``_orchestrator`` 不存在就直接返回、重建
    在守护线程里跑且由 ``_orch_build_lock`` 串行化。它返回的中文后缀正好把
    「实际发生了什么」告诉了我们。
    """
    # Preferred: the structured API . It answers with a code
    # instead of a sentence, and it QUEUES rather than drops when a task is
    # streaming. Ask for it first.
    structured = getattr(rt, "request_agent_rebuild", None)
    if callable(structured):
        try:
            code, _suffix = structured("admin override change")
        except Exception:  # noqa: BLE001
            logger.exception("orchestrator rebuild request failed")
            return ORCH_FAILED
        return str(code or ORCH_ABSENT)

    # Legacy branch, kept for runtimes that predate request_agent_rebuild
    # (test doubles, older embedders). It matches a SUBSTRING of a Chinese
    # sentence — an unguarded coupling that nothing else pins, which is exactly
    # why the structured path above exists. test_orch_suffix_substring_contract
    # keeps this branch honest.
    request = getattr(rt, "_request_composite_rebuild", None)
    if not callable(request):
        return ORCH_ABSENT
    try:
        suffix = request() or ""
    except Exception:  # noqa: BLE001
        logger.exception("orchestrator rebuild request failed")
        return ORCH_FAILED
    if "任务运行中" in suffix:
        return ORCH_SKIPPED_TASK_ACTIVE
    if suffix:
        return ORCH_REBUILDING
    return ORCH_ABSENT


def _do_refresh(rt: Any, *, invalidate_catalogs: bool = False) -> RefreshOutcome:
    """把一次「东西换了」推到三条消费者链上。

    ``invalidate_catalogs`` 只在**技能集合**变了时为 True：管理员覆写改的是包络，
    技能目录本身没变；而覆盖层会增删技能，UI 目录和 /agents/tools 都得重算。
    """
    outcome = RefreshOutcome(
        guard_reloaded=_reload_guard(rt),
        chat_invalidated=_invalidate_chat(rt),
        orchestrator=_rebuild_orchestrator(rt),
        at=time.time(),
    )
    if invalidate_catalogs:
        for mod_name, fn_name in (("mast.webui.builder_api", "invalidate_catalog"),
                                  ("mast.webui.agents_api", "invalidate_agent_tools")):
            try:
                mod = __import__(mod_name, fromlist=[fn_name])
                getattr(mod, fn_name)()
            except Exception:  # noqa: BLE001 — 一条缓存没清不该挡住其余
                logger.warning("%s.%s 失效失败", mod_name, fn_name, exc_info=True)
    return outcome


def refresh_after_skill_change(reason: str = "") -> RefreshOutcome:
    """技能集合变了（覆盖层重载 / 热注册）之后，把三条链都推一遍。

    与 ``_on_override_reload`` 共用底下同一套 —— 两个入口两份实现，迟早一边改了
    另一边没跟上。区别只有一个：这里**还要**让 UI 目录和 /agents/tools 失效，
    因为技能集合本身变了。
    """
    global _last
    rt = _runtime_ref() if _runtime_ref is not None else None
    if rt is None:
        logger.debug("refresh_after_skill_change：没有活的 runtime")
        return RefreshOutcome(guard_reloaded=False, chat_invalidated=False,
                              orchestrator=ORCH_ABSENT, at=time.time())
    outcome = _do_refresh(rt, invalidate_catalogs=True)
    with _lock:
        _last = outcome
    if outcome.agent_path_pending:
        logger.warning("技能变更后刷新（%s）：%s（agent 侧未跟上）",
                       reason, outcome.describe())
    else:
        logger.info("技能变更后刷新（%s）：%s", reason, outcome.describe())
    return outcome


def _on_override_reload() -> None:
    """The single hook registered with the override registry.

    Module-level (not a bound method) on purpose: ``register_reload_hook`` is
    idempotent by callable identity, so re-wiring — which a test harness or a
    second ``build_live_context`` will do — cannot stack duplicates or pin a
    dead runtime alive.
    """
    global _last
    rt = _runtime_ref() if _runtime_ref is not None else None
    if rt is None:
        logger.debug("override reload hook fired with no live runtime")
        return
    # 覆写改的是**包络**，技能集合没变 —— 所以不必让目录失效。
    outcome = _do_refresh(rt, invalidate_catalogs=False)
    with _lock:
        _last = outcome
    if outcome.agent_path_pending:
        # 大声说 —— 这正是「以为已生效，其实没有」会长出来的地方。
        logger.warning("管理员覆写热重载：%s（agent 侧未跟上，重启才保险）",
                       outcome.describe())
    else:
        logger.info("管理员覆写热重载：%s", outcome.describe())


def wire_override_reload(runtime: Any, registry: Any | None = None) -> bool:
    """让保存覆写这件事真的推动到运行中的组件上。接线一次，进程内长期有效。

    幂等：重复调用只是把 runtime 指向最新的那一个。返回是否接上了。
    """
    global _runtime_ref
    if runtime is None:
        return False
    if registry is None:
        try:
            from mast.admin.override_store import ConfigOverrideRegistry

            registry = ConfigOverrideRegistry.get()
        except Exception:  # noqa: BLE001
            logger.warning("override reload wiring: no registry", exc_info=True)
            return False
    register = getattr(registry, "register_reload_hook", None)
    if not callable(register):
        return False
    _runtime_ref = weakref.ref(runtime)
    register(_on_override_reload)
    logger.info("override reload wiring: SafetyGuard + agent graphs subscribed")
    return True


def last_refresh() -> RefreshOutcome | None:
    """上一次热重载各路径的结果（从未跑过则 None）。"""
    with _lock:
        return _last


def agent_refresh_pending() -> bool | None:
    """agent 侧是否还没跟上覆写。**None ＝ 判断不了，绝不能报 False。**

    True 的两种情形都是真实发生过的调度结果，不是猜测：任务运行中（
    ``_request_composite_rebuild`` 刻意不重建）或重建调度抛了。

    False 的含义要说准：私聊图缓存已清、群聊图后台重建已启动 —— 于是**下一回合**
    的 agent 用的就是新包络，不需要重启。它不承诺此刻正在飞的那一回合也换了；
    那件事重启同样做不到（重启会直接杀掉它）。
    """
    with _lock:
        outcome = _last
    if outcome is None:
        return None
    return outcome.agent_path_pending


def reset_for_tests() -> None:
    """Drop the module-level runtime ref + last outcome (tests only)."""
    global _runtime_ref, _last
    with _lock:
        _runtime_ref = None
        _last = None


__all__ = [
    "ORCH_ABSENT",
    "refresh_after_skill_change",
    "ORCH_FAILED",
    "ORCH_PENDING_QUEUED",
    "ORCH_REBUILDING",
    "ORCH_SKIPPED_TASK_ACTIVE",
    "RefreshOutcome",
    "agent_refresh_pending",
    "last_refresh",
    "reset_for_tests",
    "wire_override_reload",
]
