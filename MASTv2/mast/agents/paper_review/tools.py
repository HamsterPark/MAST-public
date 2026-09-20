"""Paper Review agent tool list — Phase 6 real implementation.

The Paper Review agent itself IS the rubric reasoner; tool calls only do the
work the LLM cannot do unaided (read files, count cross-references) and frame
the REAL manuscript content against fixed rubrics. No tool returns a fabricated
verdict: check_methodology / check_data_reasoning return the actual section
text paired with their rubric (or an honest "no text" notice), and
produce_review loads the real draft and returns a report template for the agent
to fill in — never a hardcoded ACCEPT/REVISE/REJECT.

Tools:
  1. load_draft(doc_id)                  — document store reader (+ legacy
                                           Markdown / PyMuPDF fallback)
  2. check_methodology(section_text)     — rubric scaffold over real section
  3. check_data_reasoning(section_text)  — rubric scaffold over real section
  4. check_citations(doc_id)             — regex cross-reference check
  5. produce_review(rubric, doc_id)      — report template over real draft
  6. save_review(target_doc_id, verdict, — persist the ReviewReport as a
                 report_markdown,          document linked to what it reviewed
                 doc_id)
  7. buffer tools (if buf supplied)      — live tip status / scan progress
  8. handoff_to_paper_writing            — send review back to PW for revision
  9. handoff_to_supervisor               — return control to orchestrator

What a review is attached to (2026-07-29 rewrite — design doc
``docs/v2/design/document_and_library_management.md``)
------------------------------------------------------------------------

A review used to be a file named after the draft's file STEM:
``Au111_report_v002_review_v001.md``. Because the stem carried the draft's
version number, every new draft version started a NEW review family, so the
rounds of one manuscript were scattered across unrelated filename prefixes and
``load_review`` (which matched by prefix and sorted by mtime) could return the
review of a different version.

Now a review is a document like any other, and the link is explicit:
``doc.json`` records ``target_doc_id`` + ``target_version``. All the reviews of
one manuscript share the same ``target_doc_id`` regardless of how the titles or
version numbers move.

Cross-agent import rule: this file must NOT import from any other agent package.
Only mast.agents._shared.*, mast.agents.state and non-agent packages
(mast.documents here) are permitted.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from langchain_core.tools import InjectedToolCallId, tool

from mast.agents._shared.artifact_channel import (
    ArtifactToolReturn,
    doc_ref_from_save,
)
from mast.agents._shared.buffer_tools import make_buffer_tools
from mast.agents._shared.figure_tools import list_figures
from mast.agents._shared.data_paths import (
    version_sort_key as _version_sort_key,
    drafts_dir as _shared_drafts_dir,
)
from mast.agents._shared.handoff import make_handoff
from mast.documents import store as _doc_store

if TYPE_CHECKING:
    from mast.buffer.service import BufferService

logger = logging.getLogger(__name__)

#: The kinds this agent reviews. ``experiment_report`` is the common case (an
#: internal record); ``paper_draft`` is a manuscript headed for submission.
_REVIEWABLE_KINDS: tuple[str, ...] = ("experiment_report", "paper_draft")


# ─────────────────────────────────────────────────────────────────────
# Draft resolution — document store first, legacy directory second.
#
# The legacy path stays READ-ONLY on purpose: drafts saved by an older build
# still live in ``data/drafts/`` and a review round in progress must not
# dead-end just because the store has not been populated yet. Nothing here
# ever WRITES there (the old private _repo_root()
# walk resolved to the install dir in frozen builds, so load_draft could never
# see what paper_writing had saved — the shared data_paths resolver fixed that
# and remains the one answer for the legacy location).
# ─────────────────────────────────────────────────────────────────────

def _drafts_dir() -> Path:
    """The LEGACY drafts directory, read-only. Honours MAST_DRAFTS_DIR env."""
    return _shared_drafts_dir()


def _truncate(s: str, n: int = 1500) -> str:
    return s if len(s) <= n else s[:n] + "...[truncated]"


def _read_pdf(path: Path) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _resolve_draft(ref: str) -> tuple[str | None, str, str]:
    """Resolve a loose draft reference to ``(text, identity, error)``.

    ``identity`` is the human/LLM-facing one-liner naming what was loaded (it
    carries the doc_id, which the caller must pass back to ``save_review`` as
    ``target_doc_id``). On failure ``text`` is None and ``error`` explains what to
    do next; the tools turn that into their own message rather than raising.
    """
    r = str(ref or "").strip()
    try:
        entry = _doc_store().resolve_ref(r, kinds=_REVIEWABLE_KINDS)
    except Exception as exc:  # noqa: BLE001 — never let store trouble kill a review
        logger.warning("load_draft: document store unavailable: %r", exc)
        entry = None
    if entry is not None:
        text = entry.read_text()
        if text is not None:
            return (
                text,
                f"doc_id={entry.doc_id} · v{entry.latest_version} · "
                f"《{entry.meta.title}》 · {entry.version_path()}",
                "",
            )
        logger.warning("load_draft: %s has metadata but no readable version",
                       entry.doc_id)

    # ── legacy read-only fallback: data/drafts/<stem>.md|.pdf ──
    dir_ = _drafts_dir()
    if not dir_.is_dir():
        return None, "", (
            f"还没有任何已保存的报告/草稿（文档库里没有，旧目录 {dir_} 也不存在）。"
            "请让 paper_writing 调它的 save_draft 工具存一版，并把返回的 doc_id 告诉你。"
        )
    candidates = [p for p in dir_.iterdir() if p.suffix.lower() in (".md", ".pdf")]
    if not candidates:
        return None, "", (
            f"还没有任何已保存的报告/草稿（文档库里没有，旧目录 {dir_} 也是空的）。"
            "请让 paper_writing 调它的 save_draft 工具存一版，并把返回的 doc_id 告诉你。"
        )
    if r in ("", "current", "latest"):
        target = max(candidates, key=_version_sort_key)
    else:
        target = next((p for p in candidates if p.stem == r), None)
        if target is None:
            avail = ", ".join(sorted(p.stem for p in candidates))
            return None, "", (
                f"找不到 {ref!r}。文档库里没有这个 doc_id；旧目录里可选：{avail}。"
                "正确的用法是传 paper_writing 的 save_draft 返回的 doc_id，"
                "或用 'current' 取最近一份。"
            )
    try:
        text = (_read_pdf(target) if target.suffix.lower() == ".pdf"
                else target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, "", f"读取 {target} 失败：{type(exc).__name__}: {exc}"
    return text, f"旧格式文件 {target.name}（{target}）· 没有 doc_id", ""


# ─────────────────────────────────────────────────────────────────────
# Domain tools (Phase 6 v0 stubs)
# ─────────────────────────────────────────────────────────────────────

@tool("load_draft")
def load_draft(doc_id: str = "current") -> str:
    """Load a report / manuscript for review.

    Args:
        doc_id: The doc_id paper_writing's save_draft returned (quote it from the
                handoff message), or "current" for the most recently updated
                report/draft in the active experiment. Legacy file stems and old
                .md/.pdf drafts still resolve too.

    **The first line of the output is the doc_id.** Keep it: ``save_review``
    needs it as ``target_doc_id``, and that is the only thing linking your review
    to the manuscript it reviewed. Do not reconstruct it from the title.

    Returns the text plus a brief stats block (section count via `^#`, word
    count, bibliography line count). Returns an honest "nothing saved yet" note —
    with the next action spelled out — if no document can be resolved.
    """
    text, identity, error = _resolve_draft(doc_id)
    if text is None:
        return f"load_draft: {error}"

    sections = re.findall(r"(?m)^#{1,6}\s+(.+)$", text)
    bib_block = re.search(
        r"(?im)^#+\s*(references|bibliography)\s*$([\s\S]*)\Z",
        text,
    )
    bib_lines = (
        len([ln for ln in bib_block.group(2).splitlines() if ln.strip()])
        if bib_block else 0
    )
    word_count = len(re.findall(r"\b\w+\b", text))

    header = (
        f"Loaded {identity}\n"
        "  ↑ 把上面这个 doc_id 作为 save_review 的 target_doc_id 传回去。\n"
        f"  sections: {len(sections)}\n"
        f"  words:    {word_count}\n"
        f"  bibliography lines: {bib_lines}\n"
    )
    return _truncate(header + "\n---\n" + text, n=4000)


# Rubric checklists. These are the criteria the reviewing agent reasons over;
# the tool returns the REAL section text framed against the rubric so the LLM
# evaluates the actual manuscript, never a canned verdict. (审查
# finding [16]: the old stubs returned hardcoded, draft-independent "issues".)
_METHODOLOGY_RUBRIC = (
    "1. Are all sample preparation parameters stated (substrate, anneal/deposition "
    "temperatures, pressure, dosing)?\n"
    "2. Is the tip preparation / conditioning protocol described and reproducible?\n"
    "3. Are measurement conditions given (bias, setpoint, temperature, lock-in "
    "parameters where relevant)?\n"
    "4. Is error analysis present (error bars, repeat counts, uncertainty sources)?\n"
    "5. Are instrument/software versions and calibration steps specified?"
)

_DATA_REASONING_RUBRIC = (
    "1. Is every quantitative claim supported by the data or a cited measurement?\n"
    "2. Are necessary controls / reference measurements present?\n"
    "3. Do the conclusions follow logically from the figures and numbers shown "
    "(no overreach, no unsupported causal claims)?\n"
    "4. Are alternative explanations considered and ruled out?\n"
    "5. Are statistical/fitting choices appropriate and their assumptions stated?"
)


@tool("check_methodology")
def check_methodology(section_text: str) -> str:
    """Frame a methods section against the methodology rubric for review.

    Args:
        section_text: The raw text of the methods section to check.

    This tool does NOT invent findings. It returns the actual section text
    paired with the methodology rubric checklist so the Paper Review agent
    (the rubric reasoner) evaluates the REAL manuscript content. If no section
    text is supplied it returns an honest "cannot evaluate" notice rather than
    a fabricated verdict.
    """
    text = (section_text or "").strip()
    if not text:
        return (
            "check_methodology: no section text provided — cannot evaluate. "
            "Call load_draft first and pass the methods section text."
        )
    return (
        "Methodology review — apply this rubric to the section text below and "
        "report ONLY issues actually evidenced by the text (cite the offending "
        "passage); if a criterion is satisfied, say so.\n\n"
        "Rubric:\n" + _METHODOLOGY_RUBRIC + "\n\n"
        "--- methods section under review ---\n" + _truncate(text, n=3000)
    )


@tool("check_data_reasoning")
def check_data_reasoning(section_text: str) -> str:
    """Frame a results/discussion section against the data-reasoning rubric.

    Args:
        section_text: The raw text of the results or discussion section.

    This tool does NOT invent findings. It returns the actual section text
    paired with the data-reasoning rubric so the Paper Review agent evaluates
    the REAL claims in the manuscript. If no section text is supplied it returns
    an honest "cannot evaluate" notice rather than a fabricated verdict.
    """
    text = (section_text or "").strip()
    if not text:
        return (
            "check_data_reasoning: no section text provided — cannot evaluate. "
            "Call load_draft first and pass the results/discussion section text."
        )
    return (
        "Data-reasoning review — apply this rubric to the section text below and "
        "report ONLY issues actually evidenced by the text (quote the specific "
        "claim and why it is unsupported); if the reasoning is sound, say so.\n\n"
        "Rubric:\n" + _DATA_REASONING_RUBRIC + "\n\n"
        "--- results/discussion section under review ---\n" + _truncate(text, n=3000)
    )


@tool("check_citations")
def check_citations(doc_id: str = "current") -> str:
    """Cross-reference in-text citation keys against the bibliography.

    Args:
        doc_id: Same as load_draft — a doc_id, or "current" for the most recently
                updated report/draft.

    Detects:
      - Citation keys appearing in the body but absent from the bibliography
        (probable missing reference)
      - Bibliography entries that no body paragraph cites (probable orphan)

    Recognised citation patterns (case-insensitive):
      [Author2024]   ([\\w]+\\d{4})
      [author2024]   (lowercase)
      Author et al. (2024)

    Returns a Markdown summary with two bullet lists. DOI validation against
    CrossRef is intentionally NOT wired — that needs network access; do it as
    a manual follow-up.
    """
    text, identity, error = _resolve_draft(doc_id)
    if text is None:
        return f"check_citations: {error}"

    bib_match = re.search(
        r"(?im)^#+\s*(references|bibliography)\s*$([\s\S]*)\Z",
        text,
    )
    bib = bib_match.group(2) if bib_match else ""
    body = text[: bib_match.start()] if bib_match else text

    body_keys = set()
    for m in re.finditer(r"\[([A-Za-z][\w-]*\d{4}[a-z]?)\]", body):
        body_keys.add(m.group(1).lower())
    for m in re.finditer(r"([A-Z][\w-]+)\s+et\s+al\.?\s*\((\d{4})\)", body):
        body_keys.add(f"{m.group(1).lower()}{m.group(2)}")

    bib_keys = set()
    for m in re.finditer(r"\[([A-Za-z][\w-]*\d{4}[a-z]?)\]", bib):
        bib_keys.add(m.group(1).lower())
    for m in re.finditer(
        r"([A-Z][\w-]+)\s*[,\.]?\s*[A-Z]\.[^()]*?\((\d{4})\)",
        bib,
    ):
        bib_keys.add(f"{m.group(1).lower()}{m.group(2)}")

    missing = sorted(body_keys - bib_keys)
    orphans = sorted(bib_keys - body_keys)

    lines = [f"check_citations on {identity}:"]
    if missing:
        lines.append("  cited but missing from bibliography:")
        for k in missing:
            lines.append(f"    - [{k}]")
    else:
        lines.append("  cited but missing from bibliography: (none)")
    if orphans:
        lines.append("  in bibliography but never cited (orphans):")
        for k in orphans:
            lines.append(f"    - [{k}]")
    else:
        lines.append("  orphans in bibliography: (none)")
    if not body_keys and not bib_keys:
        lines.append("  no citation keys detected — draft may use unrecognised format.")
    return "\n".join(lines)


_REVIEW_PROFILES = {
    "fast": ("methodology", "citations"),
    "standard": ("methodology", "data_reasoning", "citations"),
    "deep": ("methodology", "data_reasoning", "citations", "statistics"),
}


@tool("produce_review")
def produce_review(rubric: str = "standard", doc_id: str = "current") -> str:
    """Frame the loaded draft against the review rubric so the agent can write
    the final ReviewReport.

    Args:
        rubric: Rubric profile — "fast" (methodology + citations), "standard"
                (adds data reasoning, default), or "deep" (adds statistics).
        doc_id: Same as load_draft — the report's doc_id, or "current".

    This tool does NOT return a verdict and NEVER fabricates findings. It builds
    an empty ReviewReport template (verdict placeholder + one empty section per
    active rubric dimension) and attaches the REAL draft text when it can be
    loaded, instructing the Paper Review agent to fill in ACCEPT / REVISE /
    REJECT and a per-issue list grounded in the actual manuscript and in the
    check_* / check_citations results it has gathered. If no draft can be loaded
    it says so honestly inline instead of inventing a report body.

    The attached draft block starts with the manuscript's doc_id — pass that to
    ``save_review`` as ``target_doc_id`` when you persist the report.
    """
    profile = (rubric or "standard").strip().lower()
    dims = _REVIEW_PROFILES.get(profile)
    if dims is None:
        return (
            f"produce_review: unknown rubric profile '{rubric}'. "
            f"Choose one of: {', '.join(sorted(_REVIEW_PROFILES))}."
        )

    # Load the real draft text — the verdict must be grounded in actual content.
    draft = load_draft.invoke({"doc_id": doc_id})
    draft_failed = draft.startswith("load_draft:") or draft.startswith("load_draft failed")
    if draft_failed:
        # Honest inline notice; the template still guides the agent but makes
        # clear NO manuscript content was available, so it must not invent one.
        draft_block = (
            "(no draft could be loaded — " + draft + "\n"
            "Do NOT produce a verdict without the manuscript text; load a draft first.)"
        )
    else:
        draft_block = draft

    dim_labels = {
        "methodology": "Methodology Issues",
        "data_reasoning": "Data-Reasoning Issues",
        "citations": "Missing / Incorrect Citations",
        "statistics": "Statistical-Analysis Issues",
    }
    template_sections = "\n".join(
        f"## {dim_labels[d]}\n(list ONLY issues evidenced in the draft above; "
        f"write 'none' if none)\n" for d in dims
    )
    return (
        "Write the final ReviewReport for the draft below. Base EVERY statement on "
        "the actual draft text and on the results you obtained from the check_* / "
        "check_citations tools — do not invent issues. Use this exact structure:\n\n"
        "## Overall Verdict: <ACCEPT | REVISE | REJECT>\n\n"
        + template_sections
        + "## Required Revisions\n(numbered, actionable; empty if ACCEPT)\n\n"
        f"Active rubric profile: {profile} ({', '.join(dims)}).\n\n"
        "Then call save_review(target_doc_id=<the doc_id on the first line of the "
        "draft block below>, verdict=…, report_markdown=…). Without that doc_id "
        "the review is not linked to anything.\n\n"
        "--- draft under review ---\n" + draft_block
    )


# ─────────────────────────────────────────────────────────────────────
# save_review — persist the final ReviewReport (versioned)
# ─────────────────────────────────────────────────────────────────────

@tool("save_review")
def save_review(target_doc_id: str, verdict: str, report_markdown: str,
                doc_id: str = "",
                tool_call_id: Annotated[str, InjectedToolCallId] = "",
                ) -> "ArtifactToolReturn | str":
    """Persist the final ReviewReport to disk. MUST be called after writing
    the report — a review that only exists in the conversation is lost to the
    operator and to future review rounds ("评审记录也应该保存").

    Args:
        target_doc_id:   The doc_id of the REPORT/DRAFT you reviewed — take it
                         from the first line of load_draft's output. This is what
                         links the review to the manuscript; all the rounds on one
                         manuscript share it. "current" also works if you truly
                         reviewed the latest one.
        verdict:         "ACCEPT", "REVISE" or "REJECT".
        report_markdown: The FULL ReviewReport markdown you wrote.
        doc_id:          Leave EMPTY for a normal review — each round is its own
                         review document. Pass a review's own doc_id only when you
                         are correcting/reissuing THAT review, to store it as a
                         new version of it.

    Saved as ``<experiment>/reports/<doc-dir>/vNNN.md`` next to the manuscript
    versions, never overwriting anything. ``doc.json`` records target_doc_id +
    the manuscript version reviewed, so paper_writing's load_review can find the
    review OF a given report without guessing from filenames. Returns the
    review's doc_id and path; include both in your handoff.
    """
    v = (verdict or "").strip().upper()
    if v not in ("ACCEPT", "REVISE", "REJECT"):
        return f"save_review: verdict must be ACCEPT/REVISE/REJECT, got {verdict!r}."
    text = (report_markdown or "").strip()
    if not text:
        return "save_review: refusing to save an empty report."

    s = _doc_store()
    ref = str(target_doc_id or "").strip()
    target = None
    try:
        if ref:
            target = s.resolve_ref(ref, kinds=_REVIEWABLE_KINDS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("save_review: target resolve failed: %r", exc)

    if target is not None:
        title = f"{target.meta.title} 评审"
        tgt_id: str | None = target.doc_id
        tgt_ver: int | None = target.latest_version
        note = f"评审 {target.doc_id} v{target.latest_version} · {v}"
    else:
        # The report itself must NOT be lost because the link could not be
        # resolved: the text only exists in the conversation, and a refusal
        # throws away work. Save it unlinked and say so plainly.
        title = "评审报告"
        tgt_id = tgt_ver = None
        note = f"目标未解析（target_doc_id={ref!r}）· {v}"

    # The `<!-- verdict: X -->` first line is a load-bearing format: the frontend
    # and the documents API parse the verdict out of it. Do not drop it.
    body = f"<!-- verdict: {v} -->\n\n{text}\n"
    try:
        res = s.save(
            text=body, kind="review", title=title, doc_id=str(doc_id or "").strip(),
            created_by="agent:paper_review", note=note,
            target_doc_id=tgt_id, target_version=tgt_ver,
        )
    except Exception as e:  # noqa: BLE001 — a tool never raises at the model
        return f"save_review: write failed ({type(e).__name__}: {e})"
    if not res.ok:
        return f"save_review: 保存失败 —— {res.error}"

    # A new VERSION of an existing review document keeps doc.json's header from
    # _create, so a re-issue against a newer manuscript version has to be patched
    # in explicitly. (The per-version note above already records it either way.)
    if not res.created_new and tgt_id is not None:
        try:
            s.patch(res.doc_id, target_doc_id=tgt_id, target_version=tgt_ver)
        except Exception as exc:  # noqa: BLE001 — the review is already on disk
            logger.warning("save_review: target patch failed: %r", exc)

    logger.info("paper_review: review %s v%d saved → %s (verdict=%s, target=%s)",
                res.doc_id, res.version, res.path, v, tgt_id)

    lines = [
        f"已保存评审报告 v{res.version}（判决 {v}）—— doc_id = {res.doc_id}",
        f"文件：{res.path}",
    ]
    if target is not None:
        lines.append(f"评审对象：《{target.meta.title}》doc_id {tgt_id} v{tgt_ver}"
                     " —— paper_writing 用这个 doc_id 调 load_review 就能读到本报告。")
    else:
        lines.append(
            f"⚠ target_doc_id={ref!r} 解析不到任何报告/草稿。评审内容一个字都没丢，"
            "但它**没有和任何手稿建立关联**，paper_writing 按手稿 doc_id 查不到它。"
            "请先用 load_draft 拿到正确的 doc_id，再带上本报告的 doc_id="
            f"{res.doc_id} 重存一次以补上关联。"
        )
    if res.doc_id_unknown:
        lines.append(f"⚠ 你传的 doc_id {str(doc_id).strip()!r} 找不到，已另存为新的评审"
                     f"文档 {res.doc_id}。")
    if res.root_kind == "unfiled":
        lines.append("⚠ 当前没有活跃实验，已存入未归属区（_unfiled），内容没有丢。")
    lines.append("交接消息里写上上面的 doc_id 和路径，用户在 实验记录 → 报告 里能读到。")
    summary = "\n".join(lines)

    # Publish on the inter-agent channel so paper_writing is TOLD the review
    # exists and which document it judges. Its prompt has always said "read the
    # saved review, do not trust the handoff text" — this is what makes that
    # instruction followable without the model having to carry the id by hand.
    ref = doc_ref_from_save(
        res, kind="review", produced_by="paper_review",
        summary=(f"判决 {v}" + (f"；评审对象 doc_id={tgt_id} v{tgt_ver}"
                                if tgt_id else "；未关联到手稿")))
    if ref is None:
        return summary
    return ArtifactToolReturn(summary, {"review": ref},
                              tool_call_id=tool_call_id, name="save_review")


# ─────────────────────────────────────────────────────────────────────
# Exported list of pure domain tools (no buffer, no handoff)
# Used by tests for name-checking without constructing buf.
# ─────────────────────────────────────────────────────────────────────

AGENT_TOOLS: list = [
    load_draft,
    check_methodology,
    check_data_reasoning,
    check_citations,
    # Confirm that figures a draft references actually EXIST. load_draft was
    # this agent's only file entry point, so a draft citing a figure that was
    # never rendered could not be caught here (2026-07-27).
    list_figures,
    produce_review,
    save_review,
]


# ─────────────────────────────────────────────────────────────────────
# Top-level tool list assembly
# ─────────────────────────────────────────────────────────────────────

def build_tools(buf: "BufferService | None") -> list:
    """Assemble the full tool list for the Paper Review agent.

    Order: domain review tools first, buffer reads second (live context),
    handoffs last (terminal routing actions).

    Args:
        buf: BufferService for live tip/scan context, or None in offline tests.
    """
    tools: list = list(AGENT_TOOLS)

    if buf is not None:
        tools = tools + make_buffer_tools(buf)

    tools = tools + [
        make_handoff(
            "paper_writing",
            (
                "Send the ReviewReport back to the Paper Writing agent to "
                "address required revisions. Include the numbered issues list "
                "in the reason string."
            ),
        ),
        make_handoff(
            "supervisor",
            (
                "Return control to the orchestrator/supervisor. Use when the "
                "draft is accepted (ACCEPT verdict) or when a final REJECT "
                "decision is reached."
            ),
        ),
    ]

    logger.info(
        "paper_review: built %d tools (buf=%s)",
        len(tools),
        buf is not None,
    )
    return tools


__all__ = [
    "load_draft",
    "check_methodology",
    "check_data_reasoning",
    "check_citations",
    "produce_review",
    "save_review",
    "AGENT_TOOLS",
    "build_tools",
]
