"""``agentruntime`` —— agent 会话运行时（把承重从 langchain/langgraph 搬回仓库）。

定位取自 ``docs/v2/architecture/v2.md`` 的「agent 会话运行时」一节：day 层编排、
执行管道、恢复、等待、仲裁、产物流水线**都不依赖 LangGraph**；它今天唯一的职责是
model-tool 循环 + middleware 栈 + 对话持久化 + 流式 + provider 的 chat 抽象。这个包
就是那六件事的新家。

**现在处在 strangler 迁移的第 0 步**：这里的模块与既有运行时**并存**，逐个调用面
切换，每一步可回退。本包内已经落地的部分不依赖任何尚未落地的部分——先有存储，
再有循环，最后才有编排器。

设计立场（三条，写在这里是为了每个新模块都照着来）
--------------------------------------------------
1. **隐性语义一律显式化**。静默死分支 → 显式 outcome 枚举；super-step 汇率 →
   真实单位（模型调用数 / 工具调用数）；reducer 海关 → 一处显式 merge；
   interrupt 重放 → 原地阻塞续行；checkpoint 即消息库 → 独立消息表。
2. **控制流用返回值，不用异常/短路**。交棒是返回值而非 ``Command(PARENT)`` 短路，
   限流是 outcome 而非 ``jump_to:end``。「适配框架语义」的防御层因此整类消失。
3. **全同步**。生产四条驱动链本来就全在可阻塞的工作线程上；唯一的 async 消费者
   （CLI）改同步之后，中间件不再需要 sync/async 双实现。

对照测试在 ``tests/v2/agents/contract/`` —— 那是新旧行为等价的机器判据。
"""
from __future__ import annotations

__all__: list[str] = []
