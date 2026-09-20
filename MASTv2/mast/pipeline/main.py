"""MAST top-level entry — `python -m mast` launches the Orchestrator graph.

Minimum-viable form. This entry covers:

  - Build the orchestrator with all 7 agent subgraphs wired
  - Connect a real Nanonis ConnectionPool (or skip with --no-hardware)
  - Connect a BufferService (async, with aiosqlite WAL)

Usage:
  .venv-v2-py313/Scripts/python.exe -m mast --help
  .venv-v2-py313/Scripts/python.exe -m mast --instruction "What's the bias?"

界面不在这里：``--gui`` 曾经启动的 Gradio 聊天随 UI 的 TypeScript 重写一并删除了，
服务入口是 ``python -m mast.api``（FastAPI + WS），前端在仓库根 ``frontend/``。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_minimal_environment(
    *, no_hardware: bool, no_buffer: bool = False
):
    """Build the v2 runtime environment: pool / state / buffer / context_provider.

    Returns (buf, context_provider, registry).
    """
    from mast.config import NanonisConfig, Paths
    from mast.core.connection import ConnectionPool
    from mast.core.execution_context import ExecutionContext
    from mast.core.registry import SkillRegistry
    from mast.core.state import InstrumentState
    from mast.buffer.service import BufferService

    paths = Paths()

    # SkillRegistry — required by both XD's catalog browse and IC's tool wrap
    registry = SkillRegistry()
    registry.discover("mast.skills.builtins")
    logger.info("Discovered %d skills from mast.skills.builtins", len(registry.list_skills()))

    # ConnectionPool: real Nanonis or fake
    if no_hardware:
        pool = _make_fake_pool()
        state = _make_fake_state()
    else:
        try:
            pool = ConnectionPool(NanonisConfig())
            results = pool.connect_all()
            connected = sum(1 for v in results.values() if v)
            logger.info("Nanonis connected: %d/4 ports", connected)
            state = InstrumentState(pool)
            try:
                state.refresh()
            except Exception as e:
                logger.warning("Initial state refresh failed: %s", e)
        except Exception as e:
            logger.error("Nanonis pool setup failed: %s — falling back to fake", e)
            pool = _make_fake_pool()
            state = _make_fake_state()

    # BufferService
    if no_buffer:
        buf = None
    else:
        from mast.config import BufferConfig
        buf = BufferService(wal_path=BufferConfig().wal_path)
        # Caller is responsible for `await buf.start()` if running async

    # Register the active buffer process-wide so producers without an
    # orchestrator reference (e.g. the scan-progress vision monitor spawned
    # from the StartScan skill) can publish into it. None in no_buffer mode.
    try:
        from mast.buffer.active import set_active_buffer
        set_active_buffer(buf)
    except Exception:  # noqa: BLE001
        pass

    # 修复项 review fix: one SHARED abort Event for every ExecutionContext on
    # the headless path (previously each context got a private dead Event —
    # nothing could ever stop a running composite here), wired to E_STOP
    # buffer events via the synchronous critical hook.
    import threading
    abort_event = threading.Event()
    if buf is not None:
        try:
            from mast.buffer.schemas import VisionEventType

            def _estop_sets_abort(ev):
                if ev.kind is VisionEventType.E_STOP:
                    abort_event.set()

            buf.register_critical_hook(_estop_sets_abort)
        except Exception:  # noqa: BLE001 — best-effort wiring
            logger.warning("pipeline E_STOP abort hook wiring failed",
                           exc_info=True)

    # ExecutionContext factory (fresh per-tool-call)
    def context_provider():
        return ExecutionContext(pool=pool, state=state, registry=registry,
                                abort_event=abort_event)

    return buf, context_provider, registry


class _FakePool:
    """Stub ConnectionPool that returns canned NanonisCallRecord on every call."""

    def safe_call(self, method_name, *args, role="main"):
        from mast.core.types import NanonisCallRecord
        # Return a "no error" record with reasonable defaults
        return NanonisCallRecord(
            method=method_name,
            args=args,
            return_value=("", b"", [0.0]),
            error="",
        )

    def connect_all(self):
        return {"main": True, "monitor": True, "data": True, "emergency": True}

    def close_all(self):
        pass


def _make_fake_pool():
    return _FakePool()


def _make_fake_state():
    """Stub InstrumentState that returns a default HardwareState."""

    from mast.core.types import HardwareState

    class _FakeState:
        def __init__(self):
            self._cache = HardwareState(
                bias_v=0.1, current_a=1e-9, z_pos_m=5e-7,
                z_controller_on=True, scan_running=False,
                setpoint_a=10e-12,
            )

        def snapshot(self):
            return self._cache

        def refresh(self):
            return self._cache

        def history(self, channel):
            return []

    return _FakeState()


async def _run_instruction_async(instruction: str, *, no_hardware: bool):
    """Run a single instruction through the orchestrator, returning the final state.

    ``async`` only because the buffer's lifecycle (``buf.start()`` / ``buf.stop()``)
    is a coroutine. **The graph itself is driven synchronously** (see below).

    Why that distinction is worth a paragraph
    -----------------------------------------
    This function used to ``await graph.ainvoke(...)``, and it was the **only**
    async consumer of an agent graph in the whole repo — every other driver (the
    群聊 SSE bridge, private chat, voice, background runs) is a synchronous
    generator on a worker thread. LangChain 1.2's ``AgentMiddleware`` does not
    proxy its async hooks to the sync ones, so that single call site obliged
    **every** middleware to ship a second, async implementation of every hook —
    21 ``awrap_*`` twins at last count.

    Twice that obligation was met halfway and the CLI died on its first dispatch
    with ``NotImplementedError``: once for ``prefill_guard``, once (pre-emptively
    caught) for ``LiveStateMiddleware`` / ``SafetyGateMiddleware``. The sync tests
    could not see it, because the sync path was complete.

    Driving the graph synchronously here removes the obligation at its root: with
    no async consumer left, the twins become dead code that Step 0.4 of the
    LangGraph exit deletes. ``asyncio.to_thread`` keeps the (never-hot) CLI event
    loop responsive; a blocking call would work too, this is just tidier.
    """
    import asyncio

    from langgraph.checkpoint.memory import InMemorySaver

    from mast.agents.orchestrator.graph import build

    buf, context_provider, _registry = _build_minimal_environment(no_hardware=no_hardware)
    if buf is not None:
        await buf.start()
    try:
        # Thread the live-state snapshot into the orchestrator so the
        # instrument_control agent's SafetyGateMiddleware enforces state
        # preconditions (z_controller_off / scan_not_running …) on the CLI agent
        # path too. context_provider closes over the same ExecutionContext.state,
        # so we reach its snapshot via a fresh context() call (cheap, cached).
        def _cli_get_state():
            return context_provider().state.snapshot()

        graph = build(
            buf=buf,
            context_provider=context_provider,
            checkpointer=InMemorySaver(),
            get_state=_cli_get_state,
        )
        initial_state = {
            "messages": [("user", instruction)],
            "visit_count": {},
            "executed_skills": [],
            "scan_paths": [],
            "scan_metadata": {},
            "error_log": [],
            "pending_approvals": {},
        }
        config = {"configurable": {"thread_id": "cli-1"}}
        # 同步 invoke（见 docstring）：这条路径曾是全仓唯一的 async graph 消费者。
        return await asyncio.to_thread(graph.invoke, initial_state, config)
    finally:
        if buf is not None:
            await buf.stop()


def _v2_cli_enabled() -> bool:
    """第五个切换面：CLI 单条指令走不走新编排器。

    与另外四个同一套纪律 —— **默认关、读不到当作关、失败退回旧路径并留痕**。

    为什么它也要有开关，而不是「CLI 无所谓，直接换」
    ------------------------------------------------
    删除闸门的第②条问的是「四个开关全开吗」，而那句话真正的含义是**「旧路径已经不是
    任何一个面的生产了吗」**。CLI 若是一个没有开关的第五个面，这条判据就有了一个它
    看不见的角落：要么 CLI 永远走旧图（于是删除被它一个人卡住），要么 CLI 无声地成了
    新引擎的唯一未 bake 面。两种都让闸门的答复不再对应它想表达的事。
    """
    try:
        from mast.webui.settings_store import settings_store_for_runtime

        return bool(settings_store_for_runtime().get("engine_v2_cli"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("engine_v2_cli 读不到（%s），按关处理", exc)
        return False


async def _run_instruction_v2(instruction: str, *, no_hardware: bool) -> str:
    """v2 路径：``OrchestratorLoop`` 驱动一条 CLI 指令，返回最终文本。

    与旧路径的三个可见差别（都是本次迁移刻意要的）：

    * 结局有名字。旧路径把「路由不出来」「撞上限」「分支挂了」都答成同一坨 state，
      读的人只能从有没有正文去猜；这里 ``TaskResult.outcome`` 明说是哪一种。
    * 单位是真的。步数不再按 super-step 计价（那个数字随中间件数量漂）。
    * 仪器 agent 走**专门装配**（``ic_assembly``），少一件安全件就抛 —— 而不是
      装配出一个「安全件没挂上但照跑」的循环。
    """
    from mast.agentruntime.background import run_background_task
    from mast.agents._shared.roster import AGENT_NAMES

    buf, context_provider, registry = _build_minimal_environment(
        no_hardware=no_hardware)
    if buf is not None:
        await buf.start()
    try:
        def _cli_get_state():
            return context_provider().state.snapshot()

        def _build_loop(agent_id: str):
            if agent_id == "instrument_control":
                from mast.agentruntime.ic_assembly import build_instrument_loop

                return build_instrument_loop(
                    buf=buf, get_state=_cli_get_state,
                    context_provider=context_provider, registry=registry)
            from mast.agentruntime.assembly import build_agent_loop

            return build_agent_loop(agent_id, buf=buf)

        from mast.agents._shared.models import make_chat_model

        lines: list[str] = []

        def _emit(agent: str, role: str, text: str) -> None:
            # CLI 的 transcript 就是 stdout。带上是谁说的 —— 旧路径这里只有一坨
            # messages，读的人分不出哪句是哪个 agent 说的。
            if text:
                lines.append(f"[{agent}] {text}" if agent else text)

        import threading

        final = await asyncio.to_thread(
            run_background_task,
            instruction=instruction, agents=list(AGENT_NAMES), emit=_emit,
            abort=threading.Event(), router_model=make_chat_model("orchestrator"),
            build_loop=_build_loop, run_id="cli-1",
        )

        # ★ 转录 **加** 最终文本，不是「二选一」（2026-08-27 实测修）。
        #
        # 第一版写的是 ``run_background_task(...) or "\n".join(lines)`` ——
        # 最终文本非空时，**中途所有行整批被丢掉**。实测两跳的 run：
        #
        #     CLI 打印出来的：'data_processing 说的话'
        #
        # 第一个 agent（literature）说的话**整个不见了**。而 v1 的 CLI 打印的是
        # ``result["messages"]`` 里**每一条** —— 也就是说 v2 这条路打印得**更少**，
        # 而少掉的正是「前一个 agent 做了什么」。
        #
        # 丢掉的还不止正文：别的分支的「提前结束」说明也走 ``emit``，一并没了 ——
        # 而那正是第一条★刻意变更（静默死分支 → 显式结局）要让人看见的东西。
        parts = [*lines]
        if final and final not in lines and not any(final in ln for ln in lines):
            parts.append(final)
        return "\n".join(parts) if parts else (final or "")
    finally:
        if buf is not None:
            await buf.stop()


def _launch_gui(no_hardware: bool) -> int:
    """``--gui`` 的落点。**目标已经不存在了。**

    这个分支从 Phase 7 起指向 ``mast.webui.app.launch``（最小可用 Gradio 聊天）。
    UI 转 TypeScript 时 Gradio 整条链被删掉，``mast/webui/app.py`` 随之消失，而这个
    入口没人动过 —— 于是 ``python -m mast --gui`` 会以一个 ``ImportError`` 收场，
    错误信息指向一个没人认得的模块名。

    与其留一条会崩的路，不如**说清楚发生了什么并指向真正的入口**。删掉整个 flag
    是另一个选择，但那会让照着旧文档敲命令的人得到「unrecognized arguments」，
    比现在这句话更难懂。
    """
    print("--gui 已失效：Gradio 界面随 UI 的 TypeScript 重写一并删除了。\n"
          "现在的服务入口是：\n"
          "  .venv-v2-py313/Scripts/python.exe -m mast.api      # FastAPI + WS 后端\n"
          "前端在仓库根 frontend/（npm run dev）。")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mast", description="MAST entry point")
    parser.add_argument("--instruction", "-i", help="Single instruction to run (no GUI)")
    parser.add_argument("--gui", action="store_true",
                        help="（已失效：Gradio 界面已删除，见 mast.api）")
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="Use fake Nanonis pool (offline / dev mode)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.gui:
        return _launch_gui(no_hardware=args.no_hardware)

    if args.instruction:
        if _v2_cli_enabled():
            try:
                print(asyncio.run(_run_instruction_v2(
                    args.instruction, no_hardware=args.no_hardware)))
                return 0
            except Exception as exc:  # noqa: BLE001
                # ★ 回退**留痕**。静默兜底会让「新引擎一直在挂」看起来和「工作正常」
                #   一模一样 —— 那正是这次迁移要根除的形状。
                logger.warning("v2 CLI 引擎失败，已回退旧图：%s: %s",
                               type(exc).__name__, exc)
                print(f"（v2 引擎失败已回退：{type(exc).__name__}: {exc}）",
                      file=sys.stderr)

        result = asyncio.run(_run_instruction_async(args.instruction, no_hardware=args.no_hardware))
        for msg in result.get("messages", []):
            content = getattr(msg, "content", None)
            if content:
                print(content)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
