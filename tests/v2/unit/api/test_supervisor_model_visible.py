"""The supervisor has a model too — and the table has to say which.

Symptom — the rendered table for SUP showed blank model/thinking columns::

    智能体  模型  思考  状态  线程深度  操作
    SUP    编排与协调  —   —   空闲   0

It read like a rendering bug. It was not. ``AGENT_MODEL`` has carried
``orchestrator: kimi-k3`` all along; ``_resolve_agent_models`` iterates
``_AGENT_IDS``, and that tuple listed the six worker agents plus
``buffer_summarizer`` — never the supervisor. The registry was never asked.

Same shape as several defects found on 2026-07-28: the data exists, the lookup
list does not include it, and nothing errors — the row just renders blank.

The UI names the row ``_supervisor`` (components/agents/registry.tsx SUP_ID)
while the registry key is ``orchestrator``, so the payload carries BOTH. A
rename on one side only would reintroduce the exact same silent failure: a row
whose id matches nothing renders blank, indistinguishable from the bug.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/api/test_supervisor_model_visible.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mast.api.app import create_app  # noqa: E402

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of  # noqa: E402


@pytest.fixture(scope="module")
def models() -> dict:
    r = TestClient(create_app()).get("/api/agents/models")
    assert r.status_code == 200
    return {a["agent_id"]: a for a in r.json().get("agents", [])}


# ════════════════════════════════════════════════════════════════════════════
# The row the operator was looking at
# ════════════════════════════════════════════════════════════════════════════

def test_the_supervisor_is_in_the_table(models):
    assert "_supervisor" in models, (
        f"界面查的是 _supervisor,而接口只给了 {sorted(models)} —— "
        "那一行还是会显示「—」")


def test_the_supervisor_reports_a_real_model_and_thinking(models):
    sup = models["_supervisor"]
    assert sup["model"], "模型为空 —— 和修复前的「—」没有区别"
    assert sup["thinking"], "思考等级为空"
    # Not a placeholder: it must be the SAME model the graph actually builds.
    from mast.agents._shared import models as M
    assert sup["model"] == str(M.AGENT_MODEL["orchestrator"])


def test_both_ids_are_present_and_agree(models):
    """The UI key and the registry key must not be able to drift apart."""
    assert "orchestrator" in models
    assert models["orchestrator"]["model"] == models["_supervisor"]["model"]
    assert models["orchestrator"]["thinking"] == models["_supervisor"]["thinking"]


def test_the_worker_agents_are_still_all_there(models):
    """The fix adds a row; it must not have dropped one."""
    for aid in ("literature", "experiment_design", "instrument_control",
                "data_processing", "paper_writing", "paper_review"):
        assert aid in models, f"{aid} 掉了"
        assert models[aid]["model"]


def test_the_model_the_graph_builds_is_the_model_reported():
    """The table must reflect what the supervisor node ACTUALLY constructs —
    `make_chat_model("orchestrator", …)` — not a separate hard-coded guess."""
    import inspect

    from mast.agents.orchestrator import graph as G

    src = source_of(G.build)
    assert 'make_chat_model(\n                "orchestrator"' in src \
        or 'make_chat_model("orchestrator"' in src, (
        "编排器换了取模型的方式 —— 这张表可能又在报一个没人用的值")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
