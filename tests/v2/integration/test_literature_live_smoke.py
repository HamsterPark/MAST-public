"""Live smoke for the literature full-text pipeline — real Kimi K3 + DashScope.

Self-skips unless ``MAST_LIT_LIVE=1``, so CI stays green on a machine with no
keys and no network, and this is a one-command check on a machine that has them:

    set MAST_LIT_LIVE=1
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/integration/test_literature_live_smoke.py -q -s

What it covers (the four questions this pipeline was built to answer):
  1. the agent finds real papers in the 50k index, and the DOIs it reports exist;
  2. it fetches an open-access full text itself and ingests it;
  3. it refuses to bypass a paywall, and tells the operator instead;
  4. it deep-reads several full texts (with their SI) in parallel.

Cost: a handful of Kimi K3 calls (roughly ¥1–2 for the whole file) plus DashScope
embeddings. That cost lands in a temp ledger, not the operator's.

Isolation, in the order that matters:
  * ``MAST2_PROJECT_ROOT`` is deliberately NOT redirected. Two key-directory
    constants freeze at import time, so moving the root hides the Kimi key and
    ``make_chat_model`` quietly falls back to another provider — the run would
    look fine and test the wrong model.
  * every writable store is redirected by its own env var, and the singletons
    that captured a directory at construction are reset;
  * the billing ledger has neither an env var nor a fingerprint guard, so it is
    redirected explicitly — a live run otherwise writes real spend into the
    operator's ledger (the fifth instance of this class of leak);
  * the big index is COPIED (269 MB): the fetch/ingest half of the pipeline
    promotes into it, and promotion must not touch the real one.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (canonical block for tests/v2/) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import os
import re
import shutil

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MAST_LIT_LIVE") != "1",
    reason="live literature smoke — set MAST_LIT_LIVE=1 (needs Kimi + DashScope "
           "keys, network, and the 269 MB big index) to run",
)

REPO = Path(__file__).resolve().parents[3]
REAL_INDEX = REPO / "MASTv2" / "artifacts" / "literature_index"

#: Confirmed open access (arXiv copy) — the OA resolver should find a PDF.
OA_DOI = "10.1126/science.1102896"
#: Confirmed paywalled — must be refused, never bypassed.
CLOSED_DOI = "10.1103/PhysRevLett.49.57"

#: Filled in by the ``live`` fixture and re-applied per test (see _keep_the_sandbox).
_SANDBOX_ENV: dict[str, str] = {}


def _read_key(name: str) -> str:
    try:
        for line in (REPO / "api key" / name).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                return s.split("=", 1)[1].strip() if "=" in s else s
    except Exception:
        pass
    return ""


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """A sandbox that owns every writable store, with the real keys in env."""
    sandbox = tmp_path_factory.mktemp("lit_live")
    idx = sandbox / "literature_index"
    if not (REAL_INDEX / "vectors.npy").is_file():
        pytest.skip(f"big index not provisioned at {REAL_INDEX}")
    shutil.copytree(REAL_INDEX, idx)

    _SANDBOX_ENV.update({
        "MAST_LITERATURE_INDEX_DIR": str(idx),
        "MAST_LITERATURE_LIBS_DIR": str(sandbox / "libs"),
        "MAST_PAPERS_DIR": str(sandbox / "papers"),
        "MAST_PAPER_CORPUS": str(sandbox / "papers"),
        "MAST_EXPERIMENT_ROOT": str(sandbox / "experiments"),
        "MAST_EXPERIMENT_DB": str(sandbox / "experiments" / "exp.db"),
    })
    for var, val in _SANDBOX_ENV.items():
        os.environ[var] = val
    for var, fname in (("MOONSHOT_API_KEY", "kimi.env"),
                       ("DASHSCOPE_API_KEY", "dashscope.env")):
        if not os.environ.get(var):
            k = _read_key(fname)
            if k:
                os.environ[var] = k
    if not os.environ.get("MOONSHOT_API_KEY"):
        pytest.skip("no Moonshot/Kimi key configured")

    import importlib
    for mod, fn in (("mast.knowledge.libraries", "reset_default_registry"),
                    ("mast.knowledge.fetch_board", "reset_default_board"),
                    ("mast.documents", "reset_caches")):
        try:
            getattr(importlib.import_module(mod), fn)()
        except Exception:
            pass
    from mast.billing.ledger import UsageLedger, set_ledger_for_test
    set_ledger_for_test(UsageLedger(str(sandbox / "usage.sqlite")))

    yield sandbox

    try:
        from mast.billing.ledger import get_ledger
        s = get_ledger().summary() or {}
        print(f"\n[live] spend: {s.get('count', 0)} calls, "
              f"CNY {s.get('combined_cny', 0)}")
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _keep_the_sandbox(live):
    """Re-apply the sandbox for every test in this file.

    The suite-wide ``_isolate_literature_data`` fixture is autouse and
    FUNCTION-scoped: it re-points the literature env vars at a fresh tmp dir and
    resets the registry/board singletons before each test. That is exactly right
    for unit tests and exactly wrong here, where the board written by one test is
    the input to the next — without this, the fulfilment test finds an empty
    board and blames the test before it.

    Conftest fixtures are set up before test-module fixtures at the same scope,
    so re-applying here wins. The singleton reset makes the board re-read the
    sandbox's own JSON, which preserves state rather than discarding it.
    """
    import importlib
    for var, val in _SANDBOX_ENV.items():
        os.environ[var] = val
    for mod, fn in (("mast.knowledge.libraries", "reset_default_registry"),
                    ("mast.knowledge.fetch_board", "reset_default_board"),
                    ("mast.documents", "reset_caches")):
        try:
            getattr(importlib.import_module(mod), fn)()
        except Exception:
            pass
    yield


@pytest.fixture(scope="module")
def agent(live):
    """Built exactly as the private-chat path builds it, on the real model."""
    from langgraph.checkpoint.memory import InMemorySaver
    from mast.agents.literature.graph import build
    return build(None, checkpointer=InMemorySaver(), standalone=True)


def _turn(agent, thread: str, text: str, *, limit: int = 40):
    state = agent.invoke({"messages": [("user", text)]},
                         config={"configurable": {"thread_id": thread},
                                 "recursion_limit": limit})
    msgs = state.get("messages", [])
    tools = [t.get("name", "") for m in msgs
             for t in (getattr(m, "tool_calls", None) or [])]
    content = getattr(msgs[-1], "content", "") if msgs else ""
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                          for b in content)
    return tools, str(content or "")


# ── preflight ────────────────────────────────────────────────────────────

def test_model_is_really_kimi_k3(live):
    """A silent provider fallback would make every later assertion meaningless."""
    from langchain_core.messages import HumanMessage
    from mast.agents._shared.models import make_chat_model
    m = make_chat_model("literature", max_tokens=16000, request_timeout=90.0)
    model_id = str(getattr(m, "model_name", None) or getattr(m, "model", ""))
    assert "kimi" in model_id.lower(), f"fell back to {model_id!r}"
    assert m.invoke([HumanMessage(content="回答一个字：好")]).content


def test_semantic_search_is_not_degraded(live):
    """Degraded retrieval still returns plausible rows — assert the flag, not the rows.

    A DashScope hiccup turns semantic search into keyword matching. Results keep
    arriving with titles, years and DOIs, so "we got results" proves nothing about
    whether anything was actually retrieved by meaning.
    """
    from mast.knowledge.literature_index import search_with_status
    hits, status = search_with_status("Au(111) herringbone reconstruction STM", k=5)
    assert not status.get("degraded"), status
    assert len(hits) >= 3
    scores = {round(float(h.get("score", 0) or 0), 3) for h in hits}
    assert len(scores) > 1, f"ranking collapsed: {scores}"


# ── Q1: does it find papers? ─────────────────────────────────────────────

def test_agent_finds_real_papers(agent):
    from mast.knowledge.literature_index import work_id_for_doi
    tools, reply = _turn(agent, "q1",
                         "帮我查一下 Au(111) 表面上分子自组装的 STM 研究，"
                         "给我 3-5 篇最相关的，列出 DOI 和标题。")
    assert "search_local_corpus" in tools
    dois = list(dict.fromkeys(re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", reply)))
    assert len(dois) >= 3, reply[:500]
    # Every DOI must exist in the index — a plausible-looking DOI is the easiest
    # thing in the world for a model to invent.
    real = [d for d in dois if work_id_for_doi(d.rstrip(".,;)"))]
    assert len(real) >= 3, f"unverifiable DOIs: {set(dois) - set(real)}"


# ── Q2: does it get the full text itself? ────────────────────────────────

def test_open_access_fetch_layer_downloads_a_pdf(live):
    from mast.knowledge.fetch import try_fetch_fulltext
    res = try_fetch_fulltext(OA_DOI, auto_oa=True, timeout=30.0)
    assert res.get("status") == "ok_pdf", res
    assert Path(str(res.get("pdf_path", ""))).is_file()


def test_paywalled_paper_is_refused_not_bypassed(live):
    from mast.knowledge.fetch import try_fetch_fulltext
    res = try_fetch_fulltext(CLOSED_DOI, auto_oa=True, timeout=30.0)
    assert res.get("status") in ("blocked", "ok_metadata", "unavailable", "error")
    assert not res.get("pdf_path"), "a paywalled PDF must never be retrieved"


def test_agent_fetches_before_asking(agent):
    tools, _reply = _turn(
        agent, "q2",
        f"我需要 DOI {OA_DOI} 这篇论文里方法部分的具体实验参数。"
        f"请设法拿到全文再回答，不要只用摘要。")
    assert "fetch_fulltext_oa" in tools, tools
    # Asking the operator for something it could have fetched is the failure mode.
    if "request_fulltext" in tools:
        assert tools.index("fetch_fulltext_oa") < tools.index("request_fulltext")


# ── Q3: does it ask, and then carry on? ──────────────────────────────────

def test_asks_the_operator_when_it_cannot_fetch(agent, live):
    from mast.knowledge import fetch_board as board_mod
    tools, _reply = _turn(
        agent, "q3",
        f"我需要 DOI {CLOSED_DOI} 这篇的全文里关于隧道结间距和偏压的具体数值。"
        f"如果开源渠道拿不到，就向我索取原文。")
    assert "fetch_fulltext_oa" in tools, "it must try itself first"
    assert "request_fulltext" in tools
    pending = board_mod.list_requests("pending")
    assert pending and pending[0]["reason"], "the ask must record WHY"


def test_resumes_and_finishes_once_the_paper_arrives(agent, live):
    """The whole point of the board: fulfilment must un-block the parked task."""
    import fitz
    from mast.core.fetch_resume import build_resume_instruction
    from mast.knowledge import fetch_board as board_mod
    from mast.knowledge.fulfilment import curate_ingested_fulltext
    from mast.knowledge.ingest import ingest_pdf

    body = ("Experimental Methods\n"
            "The tunnel junction was operated with a gap width of 0.6 nm.\n"
            "The tunnel bias was held at 30 mV and the setpoint current at 1.0 nA,\n"
            "giving a junction resistance of 30 MOhm. The tip was prepared by\n"
            "electrochemical etching of tungsten and cleaned by field emission.\n"
            "Samples were sputtered at 600 eV and annealed at 900 K for 30 min.\n"
            "Images were taken in constant-current mode at 2 nm/s.\n") * 3
    pdf = live / "upload.pdf"
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(50, 50, 545, 780), body, fontsize=9)
    doc.save(str(pdf))
    doc.close()

    pending = board_mod.list_requests("pending")
    assert pending, "run test_asks_the_operator_when_it_cannot_fetch first"
    wid = pending[0]["work_id"]

    ir = ingest_pdf(str(pdf), work_id=wid, library_id="", source="user_pdf",
                    promote=True)
    assert getattr(ir, "status", "") in ("ingested", "replaced", "noop"), ir.status
    cur = curate_ingested_fulltext(
        getattr(ir, "work_id", wid) or wid, source="user_pdf",
        reason="operator upload", slug=Path(str(ir.slug_dir)).name, added_by="user")
    assert cur["fulfilled_requests"] >= 1, "the board must close"
    assert cur["fulltext_readable"] is True

    tools, reply = _turn(agent, "q3",
                         build_resume_instruction([dict(pending[0],
                                                        fulltext_readable=True)]))
    assert any(t in tools for t in ("read_paper_section", "extract_protocol",
                                    "search_papers", "deep_read_papers"))
    # It went and got the numbers it had been waiting for.
    assert "30 mV" in reply.replace(" ", " ") or "0.6" in reply, reply[:600]


def test_unreadable_upload_does_not_send_it_hunting(agent, live):
    """A scanned PDF ingests fine and reads as nothing.

    Told "your full text arrived", the agent worked through every reading tool
    looking for text that was not there and only stopped at the recursion limit
    (live run, 2026-08-01). It must now be told the truth and simply say so.
    """
    import fitz
    from mast.core.fetch_resume import build_resume_instruction

    row = {"request_id": "fr-scan", "work_id": "W_scan_live",
           "title": "A scanned reprint", "reason": "需要正文里的偏压数值",
           "fulltext_readable": False}
    tools, reply = _turn(agent, "q3scan", build_resume_instruction([row]), limit=20)
    assert not tools, f"it should not go hunting: {tools}"
    assert any(k in reply for k in ("文本层", "无法提取", "OCR", "扫描")), reply[:400]


# ── Q4: does it deep-read several papers, SI included? ───────────────────

def test_parallel_deep_read_uses_the_supplement(agent, live):
    import fitz
    from mast.knowledge import attachments as att
    from mast.knowledge.paths import papers_dir

    papers = {
        "W_live_a": ("Ordered self-assembly on Au(111)",
                     "Methods.\nImages were recorded at 4.5 K with a sample bias of\n"
                     "-1.20 V and a setpoint of 20 pA. Molecules were deposited from\n"
                     "a Knudsen cell at 420 K.\nResults.\nPeriodicity 1.4 nm.\n",
                     "Supplementary Information.\nThe crystal was sputtered at 1.0 keV\n"
                     "and annealed at 750 K for 15 minutes.\ndI/dV used 10 mV at 731 Hz.\n"),
        "W_live_b": ("NaCl bilayers on Cu(111)",
                     "Methods.\nMeasured at 5 K with a bias of 0.50 V and a setpoint\n"
                     "of 5 pA.\nResults.\nApparent height 0.30 nm.\n", None),
    }
    for slug, (title, body, si) in papers.items():
        d = papers_dir() / slug
        d.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        doc.new_page().insert_textbox(fitz.Rect(50, 50, 545, 780), body, fontsize=9)
        doc.save(str(d / "source.pdf"))
        doc.close()
        (d / "fulltext.txt").write_text(body, encoding="utf-8")
        (d / "meta.json").write_text(f'{{"title": "{title}"}}', encoding="utf-8")
        if si:
            sp = live / f"{slug}_si.pdf"
            sdoc = fitz.open()
            sdoc.new_page().insert_textbox(fitz.Rect(50, 50, 545, 780), si, fontsize=9)
            sdoc.save(str(sp))
            sdoc.close()
            assert att.attach_si(slug, sp, label="Supplementary Information")["ok"]

    tools, reply = _turn(
        agent, "q4",
        "请仔细精读本地这两篇论文的全文和补充材料：W_live_a 和 W_live_b。"
        "关注点是测量条件与制样参数，并注明每个数值出自正文还是 SI。", limit=30)

    assert "deep_read_papers" in tools, tools
    notes = list(papers_dir().rglob("deep_read_notes.md"))
    assert len(notes) >= 2, [str(n) for n in notes]
    text = "\n".join(n.read_text(encoding="utf-8") for n in notes)
    # Numbers that exist ONLY in the supplement: their presence is proof the SI
    # was read rather than skipped.
    assert "750" in text and "731" in text, "the SI was not read"
    assert "20 pA" in reply or "0.50" in reply, reply[:600]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
