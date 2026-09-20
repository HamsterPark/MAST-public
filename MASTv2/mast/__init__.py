"""MAST — LangGraph 1.x multi-agent STM autonomy system.

Architecture (see docs/v2/architecture/v2.md):
  day ── long-term memory (SQLite experiments + 15-domain knowledge + citations)
  min ── LangGraph StateGraph supervisor + 7 agents (SqliteSaver checkpointer)
  s   ── BufferService (asyncio + aiosqlite WAL)
  ms  ── VisionModule (DINOv3 ViT + task heads; legacy wrapper for older ckpts)
  μs  ── Nanonis V5e FPGA (PID servo, tip-crash protection) — UNTOUCHED

Seven agents under Orchestrator:
  Literature Reading / Experiment Design / Instrument Control /
  Data Processing / Paper Writing / Paper Review / Research Director

Note the checkpointer: the design document specifies PostgresSaver, but the
shipped implementation is SqliteSaver. Nothing serialisable-hostile may enter
checkpointed state either way — no tensors, ndarrays, file handles, sockets or
live instrument clients.
"""

# pyarrow must initialise BEFORE the langchain/torch import stack: on Windows,
# importing pyarrow.lib after `langchain.agents` (which drags in torch,
# transformers, tokenizers, …) dies with an access violation, and pandas 3.x
# routes every string column through pyarrow — so a literature search (parquet
# read) would hard-crash the whole process mid-experiment. Eager-loading it
# here fixes the DLL load order for every entry point (dev, tests, frozen).
try:
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover - pyarrow genuinely absent
    pass

try:
    from mast._buildinfo import VERSION as __version__
    from mast._buildinfo import RELEASED_AT as __release_date__
except ImportError:
    __version__ = "6.5.0"
    __release_date__ = ""
