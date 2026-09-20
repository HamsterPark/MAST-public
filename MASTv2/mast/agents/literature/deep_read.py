"""Reading several papers properly, at the same time.

``read_paper_section`` answers "what does the methods section say" with a regex
over a heading. That is the right tool for pulling one number out of one paper.
It is the wrong tool for "read these three papers and their supplements and tell
me how their preparation recipes differ" — which needs an actual reading of each
whole paper, and which done one-at-a-time costs minutes of wall clock per paper.

So: one tool call fans out to N **deep-read workers**, one per paper, each a
plain LLM conversation with no tools of its own, each producing a structured
reading note. They never talk to each other; the batch is just N independent
reads that happen to overlap in time.

Why the parallelism lives inside a single tool call rather than in the graph:

* LangGraph's parallel branches join at a super-step barrier, and everything
  crossing that barrier must be checkpointable. Executors and futures are not.
* Keeping it inside the call means nothing unserialisable ever reaches the
  checkpointer, and the agent sees one tool result rather than N.

The wall-clock discipline that follows from "the agent is blocked while this
runs":

* every worker holds an absolute deadline and checks it before each LLM call;
* the pool is never used as a context manager — ``__exit__`` waits for stragglers
  with no timeout, which is exactly the blocking call the v2 rules forbid;
* the model carries its own per-request timeout, because cancelling a future does
  not interrupt a thread already parked in a socket read.

A paper that fails, times out, or is not on disk comes back as a note saying so.
It never takes the other papers down with it.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = ["PaperNote", "deep_read_batch", "MAX_PAPERS"]

# ── budget ───────────────────────────────────────────────────────────────
#: Hard cap on papers per call. A deep read costs a full LLM pass over each
#: paper; four is already a minute or two and a few hundred thousand tokens.
MAX_PAPERS = 4
#: Concurrency. Matches MAX_PAPERS so a full batch runs in one wave.
MAX_WORKERS = 4
#: One paper's share of wall clock, and the batch's.
PAPER_TIMEOUT_S = 240.0
TOTAL_TIMEOUT_S = 300.0
#: Read in one pass below this; above it, map-reduce over segments.
SINGLE_PASS_CHARS = 60_000
SEGMENT_CHARS = 50_000
#: Absolute ceiling on characters fed to one paper's read (body + SI).
PAPER_INPUT_BUDGET_CHARS = 120_000
#: Per-supplement ceiling. SI can be enormous (raw tables, spectra dumps).
SI_MAX_CHARS = 30_000

_NOTE_BASENAME = "deep_read_notes"


@dataclass
class PaperNote:
    """One paper's reading, or an honest account of why there isn't one."""

    ref: str
    status: str = "ok"          # ok | failed | timeout | not_found
    slug: str = ""
    title: str = ""
    note_md: str = ""
    brief: str = ""
    error: str = ""
    note_path: str = ""
    llm_calls: int = 0
    truncated: bool = False
    si_files: list[str] = field(default_factory=list)


@dataclass
class _Paper:
    ref: str
    slug: str
    title: str
    text: str
    si: list[tuple[str, str]] = field(default_factory=list)
    truncated: bool = False


class _Deadline:
    """An absolute wall-clock budget, monotonic so a clock change cannot skew it."""

    def __init__(self, seconds: float) -> None:
        self._end = _monotonic() + max(0.0, float(seconds))

    def remaining(self) -> float:
        return max(0.0, self._end - _monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


def _monotonic() -> float:
    import time
    return time.monotonic()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── loading a paper ──────────────────────────────────────────────────────

def _load_si(slug: str) -> list[tuple[str, str]]:
    """``[(label, text)]`` for a paper's supplements. Never raises, never OCRs."""
    if not slug:
        return []
    out: list[tuple[str, str]] = []
    try:
        from mast.knowledge import attachments as att
        for row in att.list_attachments(slug):
            fname = str(row.get("file", ""))
            if not fname:
                continue
            text = att.attachment_text(slug, fname) or ""
            if not text.strip():
                continue
            label = str(row.get("label", "") or "").strip() or fname
            out.append((label, text[:SI_MAX_CHARS]))
    except Exception as exc:  # noqa: BLE001 — SI is an enrichment, not a gate
        logger.info("SI load failed for %s: %s", slug, exc)
    return out


def _load_legacy_si(pdf: "Path | None") -> list[tuple[str, str]]:
    """Supplements dropped next to a paper by hand (``Foo.pdf`` + ``Foo_SI.pdf``)."""
    if pdf is None:
        return []
    out: list[tuple[str, str]] = []
    try:
        from mast.agents.literature import tools as littools
        for sib in littools._legacy_si_siblings(pdf):
            text, kind, _ = littools._paper_text(sib.stem)
            if kind in ("fulltext", "pdf") and text.strip():
                out.append((sib.name, text[:SI_MAX_CHARS]))
    except Exception as exc:  # noqa: BLE001
        logger.info("legacy SI load failed for %s: %s", pdf, exc)
    return out


def _resolve_paper(ref: str) -> "_Paper | PaperNote":
    """Find a paper by work_id / slug / hand-dropped stem, and load its text."""
    from mast.agents.literature import tools as littools

    raw = (ref or "").strip()
    if not raw:
        return PaperNote(ref=ref, status="not_found", error="空的论文标识")

    candidates: list[str] = []
    try:
        from mast.knowledge.ingest import slug_for_work_id
        slug = slug_for_work_id(raw)
        if slug:
            candidates.append(slug)
    except Exception:  # noqa: BLE001 — ingest deps missing; the raw ref may still hit
        pass
    if raw not in candidates:
        candidates.append(raw)

    for cand in candidates:
        text, kind, detail = littools._paper_text(cand)
        if kind in ("fulltext", "pdf") and text.strip():
            pdf = littools._find_paper(cand)
            si = _load_si(cand) + _load_legacy_si(pdf)
            title = _title_for(cand) or cand
            budget = PAPER_INPUT_BUDGET_CHARS - sum(len(t) for _, t in si)
            truncated = len(text) > max(budget, SEGMENT_CHARS)
            if truncated:
                text = text[:max(budget, SEGMENT_CHARS)]
            return _Paper(ref=raw, slug=cand, title=title, text=text,
                          si=si, truncated=truncated)
        if kind == "error":
            return PaperNote(ref=raw, status="failed", slug=cand, error=detail)

    return PaperNote(
        ref=raw, status="not_found",
        error=(f"本地没有 «{raw}» 的全文。先用 fetch_fulltext_oa(doi, work_id) 试开源"
               f"渠道，或 request_fulltext 请用户上传。"))


def _title_for(slug: str) -> str:
    try:
        import json
        from mast.knowledge.paths import papers_dir
        meta = papers_dir() / slug / "meta.json"
        if meta.is_file():
            return str(json.loads(meta.read_text(encoding="utf-8")).get("title", "") or "")
    except Exception:  # noqa: BLE001
        pass
    return ""


# ── prompts ──────────────────────────────────────────────────────────────

_SYSTEM = """你是文献精读员，为一个自主 STM（扫描隧道显微镜）实验系统仔细阅读论文全文。

你的读者是要照着做实验的人，所以**参数的准确性高于一切**：
- 数值一律照抄原文并带单位；原文没写就写「未报告」，**绝不推测补全**。
- 正文与补充材料（SI）矛盾时，两个值都列出来并注明各自出处。
- 不确定就说不确定。宁可少写一条，不可编一条。"""

_NOTE_SCHEMA = """输出 markdown 笔记，包含且仅包含这些小节：

## 一句话结论
## 实验体系（材料 / 衬底 / 晶面 / 制备与处理条件）
## 实验方法与流程（逐步骤；含针尖处理与测量条件）
## 关键参数表（bias / setpoint / 温度 / 扫描尺寸 / 扫描速率 / 锁相参数…；
   每个值标注出处＝「正文」或「SI」；没报告的写「未报告」）
## 主要结论与证据
## SI 补充了什么（没有 SI 附件就写「无 SI 附件」）
## 可信度与局限"""


def _read_prompt(paper: _Paper, focus: str, body: str, *,
                 part: str = "", with_si: bool = True) -> str:
    head = [f"# 论文：{paper.title or paper.slug}"]
    if focus.strip():
        head.append(f"\n【本次精读的关注点】{focus.strip()}\n"
                    f"（笔记末尾追加一节 `## 与当前任务的相关性` 回答这个关注点。）")
    if part:
        head.append(f"\n【注意】这是全文的{part}，只就你看到的部分作答。")
    if paper.truncated:
        head.append("\n【注意】全文过长已被截断，未读到的部分不要臆测。")
    head.append("\n---\n正文：\n" + body)
    if with_si and paper.si:
        for label, text in paper.si:
            head.append(f"\n---\n【SI 附件：{label}】\n{text}")
    elif with_si:
        head.append("\n---\n（本篇没有 SI 附件。）")
    head.append("\n---\n" + _NOTE_SCHEMA)
    return "\n".join(head)


def _segments(text: str) -> list[str]:
    """Split a long paper on paragraph boundaries near SEGMENT_CHARS."""
    if len(text) <= SEGMENT_CHARS:
        return [text]
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for para in re.split(r"\n\s*\n", text):
        p = para.strip()
        if not p:
            continue
        if size + len(p) > SEGMENT_CHARS and buf:
            out.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        out.append("\n\n".join(buf))
    return out or [text[:SEGMENT_CHARS]]


# ── one worker ───────────────────────────────────────────────────────────

def _invoke(llm, system: str, user: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    res = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = getattr(res, "content", res)
    if isinstance(content, list):  # some providers return content blocks
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return "".join(parts).strip()
    return str(content or "").strip()


def _brief_of(note_md: str, limit: int = 400) -> str:
    """A short gist for the batch summary — sliced from the note, not re-generated.

    Asking the model for a summary of its own summary would double the cost of
    every paper for text the agent can already see.
    """
    lines = [ln.rstrip() for ln in (note_md or "").splitlines()]
    out: list[str] = []
    grab = False
    for ln in lines:
        if ln.startswith("## "):
            if out:
                break
            grab = ln.strip().startswith("## 一句话结论")
            continue
        if grab and ln.strip():
            out.append(ln.strip())
    text = " ".join(out) or " ".join(x.strip() for x in lines if x.strip())[:limit]
    return text[:limit]


def _deep_read_one(paper: _Paper, focus: str, llm_factory: Callable[[], Any],
                   deadline: _Deadline) -> PaperNote:
    note = PaperNote(ref=paper.ref, slug=paper.slug, title=paper.title,
                     truncated=paper.truncated,
                     si_files=[label for label, _ in paper.si])
    try:
        llm = llm_factory()
    except Exception as exc:  # noqa: BLE001
        note.status = "failed"
        note.error = f"模型不可用：{type(exc).__name__}: {exc}"
        return note

    try:
        if len(paper.text) <= SINGLE_PASS_CHARS:
            if deadline.expired():
                note.status = "timeout"
                note.error = "还没开始读就超时了"
                return note
            note.note_md = _invoke(llm, _SYSTEM, _read_prompt(paper, focus, paper.text))
            note.llm_calls = 1
        else:
            segs = _segments(paper.text)
            drafts: list[str] = []
            for i, seg in enumerate(segs, 1):
                if deadline.expired():
                    break
                drafts.append(_invoke(
                    llm, _SYSTEM,
                    _read_prompt(paper, focus, seg,
                                 part=f"第 {i}/{len(segs)} 段", with_si=False)))
                note.llm_calls += 1
            if not drafts:
                note.status = "timeout"
                note.error = "分段阅读未能在时限内开始"
                return note
            if len(drafts) < len(segs):
                note.truncated = True
            merged = "\n\n---\n\n".join(drafts)
            body = ("以下是同一篇论文分段阅读产生的草稿笔记。请把它们合并成**一份**"
                    "笔记，去重、消解重复小节，保留全部数值与出处标注。\n\n" + merged)
            note.note_md = _invoke(llm, _SYSTEM,
                                   _read_prompt(paper, focus, body))
            note.llm_calls += 1
    except Exception as exc:  # noqa: BLE001 — one paper's failure stays local
        note.status = "failed"
        note.error = f"{type(exc).__name__}: {exc}"
        return note

    if not (note.note_md or "").strip():
        note.status = "failed"
        note.error = "模型没有返回内容"
        return note

    note.brief = _brief_of(note.note_md)
    note.note_path = _persist_note(paper, note, focus)
    return note


def _persist_note(paper: _Paper, note: PaperNote, focus: str) -> str:
    """Write the note beside the paper. Old notes are renamed, never overwritten."""
    if not paper.slug:
        return ""
    try:
        from mast.knowledge.paths import papers_dir
        d = papers_dir() / paper.slug
        if not d.is_dir():
            return ""
        target = d / f"{_NOTE_BASENAME}.md"
        if target.exists():
            stamp = _now_iso().replace(":", "").replace("-", "")
            target.rename(d / f"{_NOTE_BASENAME}_{stamp}.md")
        header = [
            f"# 精读笔记：{paper.title or paper.slug}",
            f"- 日期：{_now_iso()}",
            f"- 关注点：{focus.strip() or '（通用精读）'}",
            f"- SI 附件：{', '.join(note.si_files) if note.si_files else '无'}",
        ]
        if note.truncated:
            header.append("- 注意：全文过长，本次只读了前一部分。")
        target.write_text("\n".join(header) + "\n\n" + note.note_md,
                          encoding="utf-8")
        return str(target)
    except Exception as exc:  # noqa: BLE001 — the note in the reply still stands
        logger.info("deep-read note write failed for %s: %s", paper.slug, exc)
        return ""


# ── the batch ────────────────────────────────────────────────────────────

def _default_llm_factory() -> Callable[[], Any]:
    def _make():
        from mast.agents._shared.models import make_chat_model, resolve_effective_model_id
        try:
            model_id = resolve_effective_model_id("literature")
        except Exception:  # noqa: BLE001 — fall back to the coded default
            model_id = None
        # A fresh instance per paper: the callbacks a model carries (billing,
        # prompt capture) are not built to be shared across threads.
        return make_chat_model("literature", model_id=model_id, max_tokens=8192,
                               temperature=0.2,
                               request_timeout=max(30.0, PAPER_TIMEOUT_S - 30.0))
    return _make


def deep_read_batch(refs: list[str], focus: str = "", *,
                    llm_factory: Callable[[], Any] | None = None,
                    paper_timeout_s: float = PAPER_TIMEOUT_S,
                    total_timeout_s: float = TOTAL_TIMEOUT_S,
                    max_workers: int = MAX_WORKERS) -> list[PaperNote]:
    """Read every paper in ``refs`` concurrently; return one note per paper.

    Order follows ``refs``. Papers that cannot be read come back with a status
    saying which way they failed, so the caller can report honestly rather than
    quietly returning fewer notes than it was asked for.
    """
    wanted = [str(r).strip() for r in (refs or []) if str(r).strip()]
    if not wanted:
        return []
    dropped = wanted[MAX_PAPERS:]
    wanted = wanted[:MAX_PAPERS]

    resolved: list[tuple[str, Any]] = [(r, _resolve_paper(r)) for r in wanted]
    readable = [(r, p) for r, p in resolved if isinstance(p, _Paper)]
    notes: dict[str, PaperNote] = {
        r: p for r, p in resolved if isinstance(p, PaperNote)}

    if readable:
        factory = llm_factory or _default_llm_factory()
        deadline = _Deadline(total_timeout_s)
        ex = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(readable))),
                                thread_name_prefix="deep-read")
        try:
            futs = {
                ex.submit(_deep_read_one, paper, focus, factory,
                          _Deadline(min(paper_timeout_s, total_timeout_s))): ref
                for ref, paper in readable
            }
            pending = set(futs)
            while pending:
                left = deadline.remaining()
                if left <= 0:
                    break
                done, pending = wait(pending, timeout=left,
                                     return_when=FIRST_COMPLETED)
                if not done:
                    break
                for fut in done:
                    ref = futs[fut]
                    try:
                        notes[ref] = fut.result(timeout=0)
                    except Exception as exc:  # noqa: BLE001
                        notes[ref] = PaperNote(ref=ref, status="failed",
                                               error=f"{type(exc).__name__}: {exc}")
            for fut in pending:
                ref = futs[fut]
                notes.setdefault(ref, PaperNote(
                    ref=ref, status="timeout",
                    error=f"精读超时（>{int(total_timeout_s)}s），本篇没有结果"))
        finally:
            # NEVER `with ThreadPoolExecutor(...)`: its __exit__ joins with no
            # timeout, so one wedged provider call would hang the tool — and with
            # it the agent — indefinitely. Workers that are still in flight are
            # abandoned; the model's own request timeout ends them.
            ex.shutdown(wait=False, cancel_futures=True)

    out = [notes.get(r) or PaperNote(ref=r, status="failed", error="未知错误")
           for r in wanted]
    for ref in dropped:
        out.append(PaperNote(
            ref=ref, status="failed",
            error=f"一次最多精读 {MAX_PAPERS} 篇，这篇未被读取（请分批调用）"))
    return out
