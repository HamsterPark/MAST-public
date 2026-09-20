"""Parallel deep reading of several papers (body + SI) in one tool call.

The properties that matter, and why:

* the papers really are read **concurrently** — pinned with a Barrier that can
  only be crossed if N workers are in flight at once, so a regression to serial
  reading deadlocks the test instead of quietly costing four times the wall clock;
* one paper failing, timing out, or not being on disk **stays local** to that
  paper;
* the pool is never joined without a timeout, so a wedged provider call cannot
  hang the agent;
* notes are written beside the paper and never overwrite an earlier reading.

Everything is injected — no network, no real model.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/literature/test_deep_read.py -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (canonical block for tests/v2/) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import threading

import pytest

from mast.agents.literature import deep_read as dr

_NOTE = """## 一句话结论
在 Au(111) 上通过退火得到有序单层。

## 关键参数表
| 参数 | 值 | 出处 |
| bias | -0.8 V | 正文 |
| setpoint | 50 pA | SI |
"""


class _FakeLLM:
    """Records what it was asked; answers with a canned note."""

    def __init__(self, reply=_NOTE, on_invoke=None):
        self._reply = reply
        self._on_invoke = on_invoke
        self.prompts: list[str] = []

    def invoke(self, messages):
        text = "\n".join(getattr(m, "content", "") for m in messages)
        self.prompts.append(text)
        if self._on_invoke is not None:
            self._on_invoke(text)

        class _R:
            content = self._reply

        return _R()


@pytest.fixture
def papers(tmp_path, monkeypatch):
    d = tmp_path / "papers"
    d.mkdir()
    monkeypatch.setenv("MAST_PAPERS_DIR", str(d))
    monkeypatch.setenv("MAST_PAPER_CORPUS", str(d))
    return d


def _make_paper(papers: Path, slug: str, text: str = "Methods. Bias -0.8 V.") -> Path:
    p = papers / slug
    p.mkdir(parents=True, exist_ok=True)
    (p / "source.pdf").write_bytes(b"%PDF-1.4 x")
    (p / "fulltext.txt").write_text(text, encoding="utf-8")
    return p


# ── concurrency ──────────────────────────────────────────────────────────

def test_papers_are_read_concurrently(papers):
    """Three workers must be in flight at once — serial execution deadlocks here."""
    for s in ("W1", "W2", "W3"):
        _make_paper(papers, s)

    barrier = threading.Barrier(3, timeout=15)
    notes = dr.deep_read_batch(
        ["W1", "W2", "W3"],
        llm_factory=lambda: _FakeLLM(on_invoke=lambda _t: barrier.wait()))

    assert [n.status for n in notes] == ["ok", "ok", "ok"]


def test_results_follow_the_requested_order(papers):
    for s in ("Wa", "Wb", "Wc"):
        _make_paper(papers, s)
    notes = dr.deep_read_batch(["Wc", "Wa", "Wb"], llm_factory=lambda: _FakeLLM())
    assert [n.ref for n in notes] == ["Wc", "Wa", "Wb"]


def test_each_paper_gets_its_own_model_instance(papers):
    """Callbacks a model carries (billing, prompt capture) are not thread-safe."""
    made: list = []

    def _factory():
        m = _FakeLLM()
        made.append(m)
        return m

    for s in ("W1", "W2"):
        _make_paper(papers, s)
    dr.deep_read_batch(["W1", "W2"], llm_factory=_factory)
    assert len(made) == 2 and made[0] is not made[1]


# ── isolation of failures ────────────────────────────────────────────────

def test_one_failing_paper_does_not_sink_the_batch(papers):
    for s in ("W1", "W2", "W3"):
        _make_paper(papers, s)

    def _factory():
        calls = {"n": 0}

        def _on(text):
            if "W2" in text:
                raise RuntimeError("provider said no")

        calls  # noqa: B018 — kept for readability of the closure
        return _FakeLLM(on_invoke=_on)

    notes = {n.ref: n for n in dr.deep_read_batch(["W1", "W2", "W3"],
                                                  llm_factory=_factory)}
    assert notes["W1"].status == "ok" and notes["W3"].status == "ok"
    assert notes["W2"].status == "failed" and "provider said no" in notes["W2"].error


def test_one_slow_paper_times_out_without_stalling_the_others(papers):
    """A wedged read must not hold the batch past its budget."""
    for s in ("W1", "W2"):
        _make_paper(papers, s)

    def _on(text):
        if "W2" in text:
            threading.Event().wait(30)  # never set — parks this worker

    import time
    t0 = time.monotonic()
    notes = {n.ref: n for n in dr.deep_read_batch(
        ["W1", "W2"], llm_factory=lambda: _FakeLLM(on_invoke=_on),
        paper_timeout_s=0.5, total_timeout_s=2.0)}
    elapsed = time.monotonic() - t0

    assert notes["W1"].status == "ok"
    assert notes["W2"].status == "timeout"
    # The point of not joining the pool: we return on our own schedule.
    assert elapsed < 10, f"batch waited {elapsed:.1f}s on a parked worker"


def test_model_construction_failure_is_reported_per_paper(papers):
    _make_paper(papers, "W1")

    def _factory():
        raise RuntimeError("no api key")

    notes = dr.deep_read_batch(["W1"], llm_factory=_factory)
    assert notes[0].status == "failed" and "no api key" in notes[0].error


def test_empty_model_reply_is_a_failure_not_an_empty_note(papers):
    _make_paper(papers, "W1")
    notes = dr.deep_read_batch(["W1"], llm_factory=lambda: _FakeLLM(reply="   "))
    assert notes[0].status == "failed"


# ── resolving what to read ───────────────────────────────────────────────

def test_missing_paper_says_how_to_get_it(papers):
    notes = dr.deep_read_batch(["W_absent"], llm_factory=lambda: _FakeLLM())
    assert notes[0].status == "not_found"
    assert "fetch_fulltext_oa" in notes[0].error


def test_openalex_url_and_bare_id_resolve_alike(papers):
    _make_paper(papers, "W1")
    notes = dr.deep_read_batch(["https://openalex.org/W1"],
                               llm_factory=lambda: _FakeLLM())
    assert notes[0].status == "ok" and notes[0].slug == "W1"


def test_hand_dropped_pdf_is_readable(papers, monkeypatch):
    """A paper that never went through ingest still has a paper_id."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Methods. Setpoint 50 pA.", fontsize=11)
    doc.save(str(papers / "Chen2020.pdf"))
    doc.close()

    notes = dr.deep_read_batch(["Chen2020"], llm_factory=lambda: _FakeLLM())
    assert notes[0].status == "ok"


def test_batch_size_is_capped_and_the_excess_is_reported(papers):
    """Silently reading 4 of 6 would read as 'all done'."""
    for i in range(6):
        _make_paper(papers, f"W{i}")
    notes = dr.deep_read_batch([f"W{i}" for i in range(6)],
                               llm_factory=lambda: _FakeLLM())
    assert len([n for n in notes if n.status == "ok"]) == dr.MAX_PAPERS
    extra = [n for n in notes if n.status == "failed"]
    assert len(extra) == 2 and "最多精读" in extra[0].error


def test_empty_input_returns_nothing(papers):
    assert dr.deep_read_batch([]) == []
    assert dr.deep_read_batch(["  "]) == []


# ── SI ───────────────────────────────────────────────────────────────────

def test_si_attachments_are_read_with_the_paper(papers):
    from mast.knowledge import attachments as att
    _make_paper(papers, "W1")
    src = papers / "si_src.pdf"
    src.write_bytes(b"%PDF si")
    att.attach_si("W1", src, label="Supplementary Note 3",
                  extractor=lambda p: "SI: annealed at 600 C for 20 min. " * 30)

    llm = _FakeLLM()
    notes = dr.deep_read_batch(["W1"], llm_factory=lambda: llm)
    assert notes[0].status == "ok"
    assert "Supplementary Note 3" in llm.prompts[0]
    assert "annealed at 600 C" in llm.prompts[0]
    assert notes[0].si_files == ["Supplementary Note 3"]


def test_paper_without_si_says_so(papers):
    _make_paper(papers, "W1")
    llm = _FakeLLM()
    dr.deep_read_batch(["W1"], llm_factory=lambda: llm)
    assert "没有 SI 附件" in llm.prompts[0]


def test_legacy_si_sibling_is_picked_up(papers):
    fitz = pytest.importorskip("fitz")
    for name, body in (("Chen2020.pdf", "Main text. Bias -1 V."),
                       ("Chen2020_SI.pdf", "Supplementary. Anneal 600 C.")):
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), body, fontsize=11)
        doc.save(str(papers / name))
        doc.close()

    llm = _FakeLLM()
    notes = dr.deep_read_batch(["Chen2020"], llm_factory=lambda: llm)
    assert notes[0].status == "ok"
    assert "Chen2020_SI.pdf" in notes[0].si_files
    assert "Anneal 600 C" in llm.prompts[0]


# ── long papers ──────────────────────────────────────────────────────────

def test_long_paper_is_read_in_segments_then_merged(papers):
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 400 for i in range(200))
    _make_paper(papers, "W1", text=body)
    llm = _FakeLLM()
    notes = dr.deep_read_batch(["W1"], llm_factory=lambda: llm)
    assert notes[0].status == "ok"
    assert notes[0].llm_calls >= 2, "a long paper must be segmented"
    assert "合并成" in llm.prompts[-1], "the last call must be the merge"


def test_oversized_paper_is_truncated_and_says_so(papers):
    body = "x" * (dr.PAPER_INPUT_BUDGET_CHARS + 50_000)
    _make_paper(papers, "W1", text=body)
    llm = _FakeLLM()
    notes = dr.deep_read_batch(["W1"], llm_factory=lambda: llm)
    assert notes[0].truncated is True
    assert "截断" in llm.prompts[0]
    assert "过长" in (papers / "W1" / "deep_read_notes.md").read_text(encoding="utf-8")


# ── notes on disk ────────────────────────────────────────────────────────

def test_note_is_written_beside_the_paper(papers):
    _make_paper(papers, "W1")
    notes = dr.deep_read_batch(["W1"], focus="针尖处理", llm_factory=lambda: _FakeLLM())
    written = (papers / "W1" / "deep_read_notes.md").read_text(encoding="utf-8")
    assert "针尖处理" in written and "一句话结论" in written
    assert notes[0].note_path.endswith("deep_read_notes.md")


def test_second_reading_keeps_the_first(papers):
    """Re-reading must not destroy the earlier note — it may have been cited."""
    _make_paper(papers, "W1")
    dr.deep_read_batch(["W1"], llm_factory=lambda: _FakeLLM(reply="## 一句话结论\n第一次"))
    dr.deep_read_batch(["W1"], llm_factory=lambda: _FakeLLM(reply="## 一句话结论\n第二次"))

    files = sorted(p.name for p in (papers / "W1").glob("deep_read_notes*.md"))
    assert len(files) == 2
    assert "第二次" in (papers / "W1" / "deep_read_notes.md").read_text(encoding="utf-8")
    archived = next(p for p in (papers / "W1").glob("deep_read_notes_*.md"))
    assert "第一次" in archived.read_text(encoding="utf-8")


def test_focus_reaches_the_prompt(papers):
    _make_paper(papers, "W1")
    llm = _FakeLLM()
    dr.deep_read_batch(["W1"], focus="针尖处理与制样", llm_factory=lambda: llm)
    assert "针尖处理与制样" in llm.prompts[0]
    assert "与当前任务的相关性" in llm.prompts[0]


def test_brief_is_sliced_from_the_note_not_regenerated(papers):
    """A summary of the summary would double the cost of every paper."""
    _make_paper(papers, "W1")
    llm = _FakeLLM()
    notes = dr.deep_read_batch(["W1"], llm_factory=lambda: llm)
    assert notes[0].llm_calls == 1
    assert "有序单层" in notes[0].brief


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
