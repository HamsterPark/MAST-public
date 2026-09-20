"""Paper Writing agent tool list — Phase 6 real implementations.

Live tools:
  1. query_experiment_records(experiment_id, limit) — ExperimentStorage SQL
  2. lookup_citation(key)                            — small curated reference
                                                       map (4 SPM-ML refs). On
                                                       an unknown key it returns
                                                       an HONEST "not found" note
                                                       listing the known keys —
                                                       it never fabricates a
                                                       reference for a key it
                                                       does not know.
  3. draft_section(section, context)                 — honest fill-in SCAFFOLD
                                                       (ordered [FILL: …] slots +
                                                       operator context); makes
                                                       NO LLM call and emits no
                                                       prose. The agent fills the
                                                       markers itself.
  4. embed_figure(scan_path, caption)                — copy figure into a
                                                       run-scoped figures dir
                                                       and return a markdown
                                                       embed line; on a name
                                                       clash with DIFFERENT bytes
                                                       it disambiguates with a
                                                       content hash (no silent
                                                       overwrite).
  5. save_draft(title, markdown_text,               — persist the report / draft
                doc_id, kind)                        as a new VERSION of a
                                                       document in the current
                                                       experiment's folder
                                                       (``reports/<doc-dir>/vNNN.md``,
                                                       never overwrites) so PR's
                                                       load_draft and the human
                                                       operator can find it.
  6. buffer tools (if buf supplied)                  — live tip status
  7. handoff_to_paper_review                         — send draft to PR
  8. handoff_to_supervisor                           — return to orchestrator

Where the documents live (2026-07-29 rewrite — design doc
``docs/v2/design/document_and_library_management.md``)
------------------------------------------------------------------------

Everything used to land in a GLOBAL ``data/drafts`` / ``data/reviews`` /
``data/figures``, keyed by the LLM-supplied title. Holding an
``Au111_report_v003.md`` you could not tell which experiment or which
conversation produced it, rewording the title forked the version history in
two, and two unrelated experiments picking the same title merged into one.

Now a document is a DIRECTORY inside the owning experiment's folder, and its
identity is a ``doc_id`` (ULID) that the title cannot influence:

    <experiment>/reports/rpt__2026-07-29__Au111形貌__ab12cd34/
        v001.md  v002.md  …      ← versions, never overwritten
        doc.json                 ← mutable header (title, kind, provenance)
        versions.jsonl           ← the authority on what versions exist
    <experiment>/reports/_assets/    ← figure pool embed_figure copies into

``mast.documents.store`` owns that protocol (per-doc lock + ``open(…, 'x')``
claim, so two concurrent savers cannot land on the same version number —
``data_paths.next_version_path`` could, and silently overwrote the loser).

Cross-agent import rule: this file must NOT import from any other agent
package. Only mast.agents._shared.*, mast.agents.state and non-agent packages
(mast.documents here) are permitted.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
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
    experiment_db_path as _shared_experiment_db_path,
    project_root as _project_root,
)
from mast.agents._shared.handoff import make_handoff
from mast.documents import KIND_LABELS, normalize_kind, store as _doc_store
from mast.documents.paths import (
    assets_dir_for,
    assets_rel_prefix,
    current_scope,
    doc_home,
)

if TYPE_CHECKING:
    from mast.buffer.service import BufferService

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Path helpers — all delegate to agents._shared.data_paths (the private
# _repo_root() walk resolved to the INSTALL dir in frozen builds, so every
# derived path was wrong on deployments; ).
# ─────────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return _project_root()


def _experiment_db_path() -> Path:
    """Resolve the SQLite experiment-log DB path.

    Honours MAST_EXPERIMENT_DB env (used by tests + production override)
    before falling back to <project_root>/experiments/mast_experiments.db —
    the canonical location mast.config uses (the old data/experiments.db
    guess matched no real deployment, so query_experiment_records always
    reported "experiment database not found")."""
    return _shared_experiment_db_path()


def _assets_dir() -> Path:
    """The current experiment's figure pool — where embed_figure copies TO.

    ``<experiment>/reports/_assets/``, or ``_unfiled/documents/_assets/`` when no
    experiment is active (same fail-open rule the document store uses: an
    unfiled figure is a bookkeeping problem, a lost one is not recoverable).
    """
    eid = current_scope()[0]
    home, root_kind = doc_home("experiment_report", eid, create=True)
    return assets_dir_for(home, root_kind, create=True)


def _truncate(s: str, n: int = 1500) -> str:
    return s if len(s) <= n else s[:n] + "...[truncated]"


# ─────────────────────────────────────────────────────────────────────
# Built-in citation map (minimal, until BibTeX wiring lands)
# ─────────────────────────────────────────────────────────────────────

_CITATION_MAP: dict[str, dict] = {
    "smalley2024": {
        "authors": "Smalley, R. et al.",
        "year": 2024,
        "title": "WSe2 defect ML segmentation",
        "venue": "APL Mach. Learn.",
        "vol": "2",
        "pages": "036104",
        "doi": "",
    },
    "krull2020": {
        "authors": "Krull, A., Hirsch, P., Rother, C., Schiffrin, A., Krull, C.",
        "year": 2020,
        "title": "Artificial-intelligence-driven scanning probe microscopy",
        "venue": "Communications Physics",
        "vol": "3",
        "pages": "54",
        "doi": "10.1038/s42005-020-0317-3",
    },
    "rashidi2018": {
        "authors": "Rashidi, M., Wolkow, R. A.",
        "year": 2018,
        "title": "Autonomous scanning probe microscopy in situ tip conditioning through machine learning",
        "venue": "ACS Nano",
        "vol": "12",
        "pages": "5185-5189",
        "doi": "10.1021/acsnano.8b02208",
    },
    "ramachandra2024": {
        "authors": "Ramachandra, S., Yang, T. C.-K., Cooper, V. R. et al.",
        "year": 2024,
        "title": "Recent advances in machine learning for scanning probe microscopy",
        "venue": "Beilstein J. Nanotechnol.",
        "vol": "15",
        "pages": "456-471",
        "doi": "",
    },
}


def _format_acs(entry: dict) -> str:
    pieces = [
        f"{entry['authors']}",
        f"({entry['year']})",
        f"\"{entry['title']}.\"",
        entry.get("venue", ""),
    ]
    vol = entry.get("vol", "")
    pages = entry.get("pages", "")
    if vol and pages:
        pieces.append(f"{vol}, {pages}.")
    elif vol:
        pieces.append(f"{vol}.")
    elif pages:
        pieces.append(f"{pages}.")
    if entry.get("doi"):
        pieces.append(f"doi:{entry['doi']}")
    return " ".join(p for p in pieces if p)


# ─────────────────────────────────────────────────────────────────────
# query_experiment_records — real SQLite via ExperimentStorage
# ─────────────────────────────────────────────────────────────────────

@tool("query_experiment_records")
def query_experiment_records(
    experiment_id: str | None = None,
    limit: int = 10,
) -> str:
    """Retrieve experiment records from the SQLite experiment log.

    Args:
        experiment_id: Optional UUID or name fragment. If None, returns the
                       most recent ``limit`` records.
        limit:         Maximum records to return (default 10).

    Returns a Markdown bulleted list with experiment id / name / status /
    sample count. The DB path is taken from ``MAST_EXPERIMENT_DB`` env or
    defaults to ``<repo>/data/experiments.db``. Returns a "no DB" note if
    the SQLite file is missing.
    """
    db_path = _experiment_db_path()
    if not db_path.is_file():
        return (
            f"query_experiment_records: SQLite DB not found at {db_path}. "
            "Set MAST_EXPERIMENT_DB or run an experiment first to create it."
        )
    try:
        from mast.logging.storage import ExperimentStorage
    except Exception as e:
        return f"query_experiment_records: storage import failed ({type(e).__name__}: {e})"

    try:
        store = ExperimentStorage(db_path)
        rows = store.list_experiments_with_counts(limit=max(1, min(limit, 200)))
    except Exception as e:
        return f"query_experiment_records: query failed ({type(e).__name__}: {e})"

    if experiment_id:
        # filter by id substring (case-insensitive)
        eid_lower = experiment_id.lower()
        rows = [
            r for r in rows
            if eid_lower in r.get("id", "").lower()
            or eid_lower in r.get("name", "").lower()
        ]

    if not rows:
        return (
            f"query_experiment_records: no records matched "
            f"(experiment_id={experiment_id!r}, limit={limit})."
        )

    lines = [f"Experiment records ({len(rows)}/{limit}) from {db_path.name}:"]
    for r in rows:
        sc = r.get("sample_count", 0)
        lines.append(
            f"  - {r['id'][:8]}…  '{r.get('name', '')}'  "
            f"[{r.get('status', '?')}, {sc} samples, {r.get('start_time', '')[:19]}]"
        )
    return _truncate("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# lookup_citation — small curated reference map (honest on miss)
# ─────────────────────────────────────────────────────────────────────

@tool("lookup_citation")
def lookup_citation(key: str) -> str:
    """Look up a citation by its short key.

    The reference table is a small CURATED map of SPM-ML references (4
    entries). It is deliberately NOT a general bibliography: on an unknown
    key this returns an honest "not found" note listing the keys it does
    know, and it NEVER fabricates a reference. Do not rely on it for
    arbitrary citations — verify any reference you cite against the original
    source.

    Args:
        key: Short citation key, e.g. "smalley2024", "krull2020",
             "rashidi2018", "ramachandra2024". Lower-case author + year.

    Returns an ACS-style formatted reference for a known key, or an honest
    "not found" note (with the list of known keys) for an unknown one.
    """
    norm = key.strip().lower()
    entry = _CITATION_MAP.get(norm)
    if entry is None:
        keys = ", ".join(sorted(_CITATION_MAP.keys()))
        return (
            f"lookup_citation: '{key}' not found. This tool only knows a small "
            f"curated set of SPM-ML references: {keys}. For any other citation, "
            "obtain the reference from the original source — do NOT invent one."
        )
    return _format_acs(entry)


# ─────────────────────────────────────────────────────────────────────
# draft_section — honest fill-in scaffold (NOT prose)
#
# This tool does NOT write prose and makes no LLM call. It returns an
# explicit, labelled scaffold: an ordered list of the slots a given section
# must contain, each annotated with what to put there. The Paper-Writing
# agent (the writer) fills these slots from the supplied context + the
# experiment records / citations it has gathered. The output never pretends
# to be finished text — every unfilled slot is rendered as a visible
# `[FILL: …]` marker so a downstream reader can see at a glance what is
# still missing (审查: the old templates emitted bare
# `{topic}`-style placeholders that were silently returned unfilled and read
# as finished sentences).
# ─────────────────────────────────────────────────────────────────────

# Each entry: ordered (slot_key, guidance) pairs describing what the section
# needs. Rendered into a checklist the writer fills, not pre-written prose.
_SECTION_SLOTS: dict[str, tuple[tuple[str, str], ...]] = {
    "introduction": (
        ("topic", "the phenomenon / material this paper studies"),
        ("prior_art_summary", "what the literature (LIT agent) already reports"),
        ("gap", "the open question this work addresses"),
        ("contribution", "what this work adds, in one sentence"),
    ),
    "methods": (
        ("instrument", "instrument + controller (e.g. Nanonis V5e), temperature/UHV"),
        ("sample_prep", "substrate, anneal/deposition temps, pressure, dosing"),
        ("tip_prep", "tip material + conditioning protocol (reproducible)"),
        ("imaging_params", "bias, setpoint, scan rate/size for topography"),
        ("sts_params", "spectroscopy: bias range, lock-in modulation, stabilisation"),
        ("error_analysis", "repeat counts, error bars, uncertainty sources"),
    ),
    "results": (
        ("topo_summary", "what the topography shows (features, periodicity, defects)"),
        ("sts_summary", "what spectroscopy reveals (gaps, peaks, dI/dV features)"),
        ("key_params", "quantitative values extracted FROM THE DATA (with units)"),
    ),
    "discussion": (
        ("interpretation", "what the results mean physically"),
        ("citations", "supporting refs (use lookup_citation keys)"),
        ("novel_finding", "how this extends / differs from prior work"),
        ("caveats", "limitations, alternative explanations not ruled out"),
    ),
    "conclusion": (
        ("summary", "the demonstrated result in one sentence"),
        ("next_steps", "concrete follow-up experiments"),
    ),
}

# Marker the writer must replace; also let us detect unfilled slots later.
_FILL_MARKER = "[FILL: {key} — {guidance}]"


@tool("draft_section")
def draft_section(section: str, context: str) -> str:
    """Build an honest fill-in scaffold for a manuscript section.

    This tool does NOT write prose and makes NO LLM call. It returns an
    ordered checklist of the slots the section requires (heading + one
    ``[FILL: key — guidance]`` line per slot) followed by the operator's
    context verbatim. The Paper-Writing agent then writes the section by
    replacing every ``[FILL: …]`` marker with real content drawn from the
    context, the experiment records (query_experiment_records), and the
    citations (lookup_citation). Any ``[FILL: …]`` marker left in the final
    draft is a visible signal that the section is incomplete.

    Args:
        section: One of "introduction", "methods", "results", "discussion",
                 "conclusion". Case-insensitive.
        context: Free-text describing what to put in the section. Appended
                 verbatim under an "Operator context" heading so the writer
                 can map it onto the slots.

    Returns the scaffold (heading + FILL markers + operator context).
    """
    sec = section.strip().lower()
    slots = _SECTION_SLOTS.get(sec)
    if slots is None:
        return (
            f"draft_section: unknown section '{section}'. "
            f"Use one of: {', '.join(_SECTION_SLOTS.keys())}."
        )

    heading = sec.capitalize()
    lines = [
        f"## {heading}",
        "",
        "<!-- SCAFFOLD ONLY — replace every [FILL: …] marker with real "
        "content; do not leave markers in the final draft. -->",
        "",
    ]
    for key, guidance in slots:
        lines.append(_FILL_MARKER.format(key=key, guidance=guidance))
    body = "\n".join(lines)
    body += (
        "\n\n### Operator context (verbatim — map onto the slots above)\n\n"
        + context.strip()
    )
    return _truncate(body, n=2000)


# ─────────────────────────────────────────────────────────────────────
# embed_figure — copy image into a figures dir + emit Markdown line
# ─────────────────────────────────────────────────────────────────────

def _file_sha8(path: Path) -> str:
    """First 8 hex chars of the file's SHA-256 — enough to disambiguate
    same-named-but-different figures without bloating the filename."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _resolve_figure_dest(src: Path, dest_dir: Path) -> tuple[Path, str]:
    """Pick a destination path that never silently overwrites a DIFFERENT
    figure that already lives under the same name.

    Returns (dest_path, disposition) where disposition is one of:
      "new"        — name was free; copy as-is
      "reused"     — a file with the same name AND identical bytes exists;
                     reuse it (idempotent, no copy needed)
      "hashed"     — a file with the same name but DIFFERENT bytes exists;
                     disambiguate the new copy with a "<stem>.<sha8><suffix>"
                     name so the existing figure is preserved.
    """
    plain = dest_dir / src.name
    if not plain.exists():
        return plain, "new"

    src_hash = _file_sha8(src)
    # Same name already present — is it the same bytes?
    if _file_sha8(plain) == src_hash:
        return plain, "reused"

    # Name clash with different content → hash-suffix the new file. If a
    # hash-suffixed copy of THIS content already exists, reuse it.
    hashed = dest_dir / f"{src.stem}.{src_hash}{src.suffix}"
    if hashed.exists() and _file_sha8(hashed) == src_hash:
        return hashed, "reused"
    return hashed, "hashed"


@tool("embed_figure")
def embed_figure(scan_path: str, caption: str) -> str:
    """Embed a figure into the report / manuscript.

    Args:
        scan_path: Absolute path to a PNG / JPEG / PDF / SXM-export image.
                   Get it from ``list_figures()`` — do not guess a filename.
        caption:   Figure caption (newlines and Markdown allowed).

    Copies the source image into the CURRENT EXPERIMENT's figure pool
    (``<experiment>/reports/_assets/``) and returns a Markdown image-embed line
    ready to paste into a draft. The emitted link is ``../_assets/<name>``,
    relative to the document's own directory.

    Why the figure travels with the experiment (2026-07-29): the old
    implementation copied into the global ``data/figures/`` and emitted
    ``../figures/x.png``. That link only resolved for a draft sitting in
    ``data/drafts/`` — move the experiment folder to another machine, or open the
    document from anywhere else, and every figure in the report was broken while
    the file itself sat one directory over. Inside the experiment folder the
    document and its figures move together.

    With no active experiment the pool is ``_unfiled/documents/_assets/`` (the
    figure is never dropped); pick or create an experiment and the next embed
    lands in it.

    Name-collision handling: if a figure with the
    same filename but DIFFERENT content already exists in the pool, the new copy
    is written under a content-hashed name ("<stem>.<sha8><ext>") instead of
    silently clobbering the existing one. If the same-named file has identical
    bytes, the existing copy is reused (idempotent — re-embedding the same figure
    does not duplicate it).

    Returns a "source missing" error if the source path does not exist.
    """
    src = Path(scan_path)
    if not src.is_file():
        return f"embed_figure: source file not found: {scan_path}"

    try:
        dest_dir = _assets_dir()
    except Exception as e:  # noqa: BLE001 — a tool never raises at the model
        return (f"embed_figure: 无法准备实验图池（{type(e).__name__}: {e}）。"
                "图**没有**被复制，请先确认实验文件夹可写。")
    try:
        dest, disposition = _resolve_figure_dest(src, dest_dir)
    except Exception as e:
        return f"embed_figure: hashing failed ({type(e).__name__}: {e})"

    try:
        if disposition != "reused" and src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
    except Exception as e:
        return f"embed_figure: copy failed ({type(e).__name__}: {e})"

    # The link is read by whatever renders the .md, and that resolves relative
    # paths against the FILE'S OWN directory. A document lives at
    # <exp>/reports/<doc-dir>/vNNN.md and the pool at <exp>/reports/_assets/, so
    # the correct link is ../_assets/<name> — computed by assets_rel_prefix()
    # rather than spelled out here, because a plan document sits one level deeper
    # and needs ../../reports/_assets/.
    rel_str = assets_rel_prefix("experiment_report") + dest.name
    note = {
        "new": f"figure copied to {dest}",
        "reused": f"figure already present (identical bytes); reusing {dest}",
        "hashed": (
            f"name '{src.name}' already taken by a different figure — "
            f"saved this one as {dest.name} to avoid overwriting it"
        ),
    }[disposition]
    return (
        f"![{caption}]({rel_str})\n"
        f"\n*Caption:* {caption}\n"
        f"\n({note})"
    )


# ─────────────────────────────────────────────────────────────────────
# save_draft — persist as a new VERSION of a document in the experiment folder
# ─────────────────────────────────────────────────────────────────────

#: The two kinds this agent is allowed to write. ``review`` belongs to
#: paper_review, ``experiment_plan`` to the planner, ``literature_report`` to the
#: literature agent — a writer that can mint any kind makes the kind meaningless.
_WRITABLE_KINDS: tuple[str, ...] = ("experiment_report", "paper_draft")


@tool("save_draft")
def save_draft(title: str, markdown_text: str, doc_id: str = "",
               kind: str = "experiment_report",
               tool_call_id: Annotated[str, InjectedToolCallId] = "",
               ) -> "ArtifactToolReturn | str":
    """Persist the report / manuscript to disk. MUST be called once the text is
    complete — a report that only exists in the conversation is invisible to the
    Paper Review agent's load_draft/check_citations tools and to the human
    operator ("看不到论文在哪里").

    Args:
        title:         Short human-readable title (e.g. "Au(111) 形貌与 STS").
                       It is a DISPLAY NAME only — it does not decide identity,
                       and on a revision (``doc_id`` given) it is ignored, since
                       renaming a document is the operator's call.
        markdown_text: The FULL text in Markdown, with no [FILL: …] scaffold
                       markers left. Every save stores the whole document, not
                       a diff.
        doc_id:        EMPTY = create a NEW document. NON-EMPTY = save a new
                       version of THAT document. **When you revise something you
                       already saved, pass back the doc_id the previous save
                       returned** — otherwise you get a second, unrelated
                       document instead of v002 of the first one.
        kind:          "experiment_report" (default) or "paper_draft".
                       - ``experiment_report``: the normal case — an internal
                         record of what was measured, for the operator.
                       - ``paper_draft``: only when the operator is preparing a
                         MANUSCRIPT FOR SUBMISSION. If they did not say so, it is
                         a report.

    Saved as ``<experiment>/reports/<doc-dir>/vNNN.md``: a new version file every
    time, never an overwrite, inside the folder of the experiment
    that is currently active. Returns the doc_id, the version and the path.
    """
    text = (markdown_text or "").strip()
    if not text:
        return "save_draft: refusing to save an empty draft."
    if "[FILL:" in text:
        return (
            "save_draft: draft still contains [FILL: …] scaffold markers — "
            "complete every section before saving."
        )

    k = normalize_kind(kind, "experiment_report")
    kind_note = ""
    if k not in _WRITABLE_KINDS:
        kind_note = (f"（注：kind={kind!r} 不是本工具能写的类型，已按 "
                     f"experiment_report 保存）")
        k = "experiment_report"

    try:
        res = _doc_store().save(
            text=text, kind=k, title=(title or "").strip(),
            doc_id=(doc_id or "").strip(), created_by="agent:paper_writing",
        )
    except Exception as e:  # noqa: BLE001 — a tool never raises at the model
        return f"save_draft: write failed ({type(e).__name__}: {e})"
    if not res.ok:
        return f"save_draft: 保存失败 —— {res.error}"

    label = KIND_LABELS.get(res.kind, res.kind)
    logger.info("paper_writing: %s %s v%d saved → %s (%d words)",
                res.kind, res.doc_id, res.version, res.path, res.words)

    lines = [
        f"已保存{label}「{res.title}」v{res.version} —— doc_id = {res.doc_id}",
        f"文件：{res.path}（{res.words} 词）{kind_note}",
    ]
    if res.doc_id_unknown:
        lines.append(
            f"⚠ 你传的 doc_id {str(doc_id).strip()!r} 在库里找不到。内容一个字都没丢，"
            f"但它被**另存为一个新文档** {res.doc_id} —— 它不是任何已有文档的新版本。"
            "如果你本意是修订某一份，请核对那一份的 doc_id 再存一次。"
        )
    if res.root_kind == "unfiled":
        lines.append(
            "⚠ 当前没有活跃实验，已存入**未归属区**（_unfiled），内容没有丢。"
            "建议先选定或新建一个实验；也可以稍后在 实验记录 → 报告 里认领它。"
        )
    lines.append(
        f"**下次修订这一份，必须把 doc_id={res.doc_id} 原样传回 save_draft**（连同修订后的"
        "全文）。不传 doc_id 就会另立一个新文档，而不是接上这一份的版本历史。"
    )
    lines.append(
        f"交接消息里写上 doc_id={res.doc_id} 和上面的路径 —— paper_review 用这个 doc_id "
        "调 load_draft，用户在 实验记录 → 报告 里能读到。"
    )
    summary = "\n".join(lines)

    # Publish the pointer on the inter-agent channel: paper_review is SHOWN this
    # doc_id in its own context, and so is this agent on its next hop — which is
    # what stops a revision from silently forking a second document because the
    # id had been compacted out of the conversation.
    ref = doc_ref_from_save(res, kind=res.kind, produced_by="paper_writing",
                            summary=f"{label}，{res.words} 词")
    if ref is None:
        return summary
    return ArtifactToolReturn(summary, {"draft": ref},
                              tool_call_id=tool_call_id, name="save_draft")


# ─────────────────────────────────────────────────────────────────────
# export_report_html — ONE file the operator can actually hand to someone
# ─────────────────────────────────────────────────────────────────────

@tool("export_report_html")
def export_report_html(doc_id: str = "current") -> str:
    """把报告/草稿导出成**单个自包含 HTML 文件**（图片内嵌），双击即可查看、可直接分享。

    markdown 是工作稿：可在「对象编辑」里改、有版本、能看 diff。但它**不是**
    交付物 —— 图片存在旁边的图池里，把 .md 发给别人图就丢了，而且对方还得有
    markdown 阅读器。本工具把图片以 base64 嵌进 HTML，产出的单文件不依赖任何
    外部文件。

    Args:
        doc_id: save_draft 返回的 doc_id；也接受 "current"（= 当前实验里最近
                更新的那一份报告/草稿）。导出的总是该文档的**最新版本**。

    写到 ``<实验>/exports/<名字>_vNNN_<时间戳>.html``。**带时间戳，多份共存** ——
    旧实现是固定文件名直接覆盖，上一份交付件就这么没了。

    返回路径与内嵌图片数 —— 若有图片找不到，会**明确报告缺了几张**，因为一份看着
    完整、实则少了数据的报告比报错更糟。在交接消息里写上这个路径。
    """
    from mast.agents._shared.report_html import render_html

    try:
        s = _doc_store()
        entry = s.resolve_ref(doc_id, kinds=_WRITABLE_KINDS)
        if entry is None:
            return ("export_report_html: 找不到可导出的报告/草稿"
                    f"（doc_id={doc_id!r}）。先 save_draft 存一版，再用它返回的 "
                    "doc_id 调本工具。")
        md = entry.read_text()
        src = entry.version_path()
        if md is None or src is None:
            return (f"export_report_html: 文档 {entry.doc_id} 有元数据但读不到版本"
                    f"文件（目录 {entry.dir}）。请检查该目录，或先 save_draft 重存一版。")

        # base_dir 必须是**版本文件所在的目录**：markdown 渲染器按文件自己的目录
        # 解析相对路径，`../_assets/x.png` 只有从这里算起才落在实验图池上。
        doc, inlined, missing = render_html(
            md, base_dir=src.parent, title=entry.meta.title,
            subtitle=f"v{entry.latest_version} · doc_id {entry.doc_id} · 由 MAST 导出")

        dest = s.export_path(entry, ".html")
        dest.write_text(doc, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return f"export_report_html failed: {type(e).__name__}: {e}"

    note = f"，内嵌图片 {inlined} 张" if inlined else "，无图片"
    if missing:
        note += (f"；**{missing} 张图片找不到**，已在文中标出 —— "
                 "请检查图是否还在实验的 reports/_assets/ 图池里")
    logger.info("paper_writing: report exported → %s (%d inlined, %d missing)",
                dest, inlined, missing)
    return (f"已导出自包含 HTML：{dest}\n"
            f"（{dest.stat().st_size // 1024} KB{note}）"
            "。这是单个文件，可直接双击打开或发给他人，不依赖图片目录。")


@tool("export_report_docx")
def export_report_docx(doc_id: str = "current") -> str:
    """把报告/草稿导出成 **Word 文档（.docx）**，供他人用修订/批注来改。

    什么时候用它，而不是 export_report_html：

    * 要让**别人改**（导师、合作者审阅，走 Word 的修订与批注流程）→ 本工具。
    * 要让**别人看**（附在邮件里、丢共享盘、双击就开）→ export_report_html。

    HTML 交付件覆盖不了「被人改」这一半：它没有修订功能，期刊也不收 HTML 投稿。

    Args:
        doc_id: save_draft 返回的 doc_id；也接受 "current"（= 当前实验里最近
                更新的那一份报告/草稿）。导出的总是该文档的**最新版本**。

    全文挂 **Word 内置样式**（``Heading 1``/``Normal``/``Quote``/``Caption``/
    ``Table Grid`` …），从不硬编码字号颜色 —— 所以收件人改一次样式就能重排全文，
    套期刊模板只是一次样式替换；导航窗格、自动目录、交叉引用也都能用。

    写到 ``<实验>/exports/<名字>_vNNN_<时间戳>.docx``，带时间戳多份共存。
    图片嵌进文档，题注用 Word 的 ``Caption`` 样式。若有图嵌不进去（找不到、或是
    Word 不支持的格式如 PDF），会**在文中标出并在返回值里报数** —— 一份看着完整、
    实则少了数据的报告比报错更糟。
    """
    try:
        from mast.agents._shared.report_docx import render_docx
    except ImportError as e:  # python-docx 未装（或打包漏了它的模板数据）
        return ("export_report_docx: Word 导出不可用 —— python-docx 未安装"
                f"（{e}）。装它：pip install python-docx。"
                "改用 export_report_html 仍然可以产出可分享的交付件。")
    try:
        s = _doc_store()
        entry = s.resolve_ref(doc_id, kinds=_WRITABLE_KINDS)
        if entry is None:
            return ("export_report_docx: 找不到可导出的报告/草稿"
                    f"（doc_id={doc_id!r}）。先 save_draft 存一版，再用它返回的 "
                    "doc_id 调本工具。")
        md = entry.read_text()
        src = entry.version_path()
        if md is None or src is None:
            return (f"export_report_docx: 文档 {entry.doc_id} 有元数据但读不到版本"
                    f"文件（目录 {entry.dir}）。请检查该目录，或先 save_draft 重存一版。")

        # base_dir 必须是**版本文件所在的目录** —— 与 HTML 分支同一条理由：
        # `../_assets/x.png` 只有从这里算起才落在实验图池上。
        blob, embedded, missing = render_docx(
            md, base_dir=src.parent, title=entry.meta.title,
            subtitle=f"v{entry.latest_version} · doc_id {entry.doc_id} · 由 MAST 导出")

        dest = s.export_path(entry, ".docx")
        dest.write_bytes(blob)
    except Exception as e:  # noqa: BLE001
        return f"export_report_docx failed: {type(e).__name__}: {e}"

    note = f"，嵌入图片 {embedded} 张" if embedded else "，无图片"
    if missing:
        note += (f"；**{missing} 张图片未能嵌入**，已在文中逐条标出原因 —— "
                 "常见是图不在实验的 reports/_assets/ 里，或是 Word 不支持的格式（如 PDF）")
    logger.info("paper_writing: docx exported → %s (%d embedded, %d missing)",
                dest, embedded, missing)
    return (f"已导出 Word 文档：{dest}\n"
            f"（{dest.stat().st_size // 1024} KB{note}）"
            "。全文使用 Word 内置样式，收件人可直接用修订/批注审阅，也能套期刊模板。")


# ─────────────────────────────────────────────────────────────────────
# load_review — read the reviewer's report back (closes the revision loop)
# ─────────────────────────────────────────────────────────────────────

@tool("load_review")
def load_review(doc_id: str = "latest") -> str:
    """Load the Paper Review agent's saved ReviewReport so you can revise against
    the ACTUAL report rather than from memory.

    Args:
        doc_id: Either
                - the doc_id of a REPORT/DRAFT you saved → returns the most
                  recent review OF THAT document (this is the normal call: pass
                  the doc_id save_draft gave you);
                - the doc_id of a review itself → returns that review;
                - "latest" (default) → the most recent review of any document.

    Why this exists (2026-07-11): paper_review persists every ReviewReport to
    disk, but paper_writing had NO tool to read one — the revision loop depended
    entirely on the issue list surviving in the handoff message text. That text
    passes through the conversation, which the compaction middleware may summarise
    away on a long run, so a late revision round could be done half-blind. Reading
    the saved file makes the loop robust: the report is on disk, in full, forever.

    Reviews are linked to what they reviewed by ``target_doc_id``, so asking with
    a report's doc_id always returns a review OF THAT REPORT. (The old version
    matched review FILENAME PREFIXES and sorted by mtime, which could hand back
    the review of a different manuscript, or an older round of this one.)

    Use it whenever paper_review hands work back with a REVISE verdict, and
    address EVERY numbered item it lists.
    """
    s = _doc_store()
    ref = str(doc_id or "").strip()

    target_of = ""
    entry = None
    if ref in ("", "latest", "current"):
        entry = s.latest(kinds=("review",))
    else:
        named = s.get(ref)
        if named is not None and named.meta.kind == "review":
            entry = named
        elif named is not None:
            # A report/draft doc_id → the newest review whose target is it.
            entry = _latest_review_of(s, named.doc_id)
            if entry is None:
                return (f"load_review: 还没有针对《{named.meta.title}》"
                        f"（doc_id {named.doc_id}）的评审 —— paper_review 还没存过。"
                        "如果你要的是别的文档的评审，请传那一份的 doc_id；"
                        "或者用 'latest' 取最近一份评审。")
            target_of = f"（针对《{named.meta.title}》）"
        else:
            entry = s.resolve_ref(ref, kinds=("review",))

    if entry is None:
        avail = s.list(kind="review")
        if not avail:
            return ("load_review: 目前一份评审都没有 —— paper_review 还没保存过 "
                    "ReviewReport。")
        hint = "；".join(f"{e.meta.title}（doc_id {e.doc_id}）" for e in avail[:5])
        return f"load_review: 找不到 {doc_id!r} 对应的评审。已有的评审：{hint}"

    text = entry.read_text()
    if text is None:
        return (f"load_review: 评审 {entry.doc_id} 有元数据但读不到版本文件"
                f"（目录 {entry.dir}）。")

    tgt = entry.meta.target_doc_id
    tgt_line = (f"评审对象：doc_id {tgt} v{entry.meta.target_version or '?'}"
                if tgt else "评审对象：未记录（这份评审没有登记它评的是哪一版）")
    return (f"ReviewReport doc_id={entry.doc_id} v{entry.latest_version}"
            f"{target_of} —— 《{entry.meta.title}》\n"
            f"{tgt_line}\n"
            f"文件：{entry.version_path()}\n"
            "逐条处理下面每一个编号项，然后带着**报告自己的 doc_id** 调 save_draft "
            "存一个新版本（不要另存新文档）。\n"
            f"---\n{_truncate(text, 6000)}")


def _latest_review_of(s, target_doc_id: str):
    """The most recent review whose ``target_doc_id`` is *target_doc_id*.

    Ordering never touches mtime — that is how the OLD implementation handed back
    the wrong round (70.5% of back-to-back writes on this machine land on an
    identical mtime). But ``updated_at`` alone is not enough either: the store
    stamps it with ``timespec="seconds"``, so two rounds saved in the same second
    TIE, and a stable sort then falls back to directory order, which returns the
    OLDEST. Measured, not theorised: two save_review calls back to back reproduce
    it every time.

    The tiebreak is the doc_id. It is a ULID, so lexicographic order IS creation
    order — no clock re-reading, no filesystem timestamps.
    """
    cands = [e for e in s.list(kind="review")
             if e.meta.target_doc_id == target_doc_id and e.versions]
    if not cands:
        return None
    cands.sort(key=lambda e: (e.meta.updated_at or e.meta.created_at or "", e.doc_id),
               reverse=True)
    return cands[0]


# ─────────────────────────────────────────────────────────────────────
# Top-level assembly
# ─────────────────────────────────────────────────────────────────────

AGENT_TOOLS: list = [
    query_experiment_records,
    lookup_citation,
    draft_section,
    # embed_figure needs an ABSOLUTE path and this agent had no way to obtain
    # one — its only source was data_processing spelling the path out in a
    # handoff message, which a compaction or a fresh session silently breaks.
    # list_figures is that missing source (2026-07-27).
    list_figures,
    embed_figure,
    save_draft,
    export_report_html,
    export_report_docx,
    load_review,
]


def build_tools(buf: "BufferService | None") -> list:
    tools: list = list(AGENT_TOOLS)
    if buf is not None:
        tools = tools + make_buffer_tools(buf)
    tools = tools + [
        make_handoff(
            "paper_review",
            (
                "Hand the completed draft to the Paper Review agent for "
                "rigorous methodology, data-reasoning, and citation checks."
            ),
        ),
        make_handoff(
            "supervisor",
            (
                "Return control to the orchestrator/supervisor. Use when the "
                "review cycle is complete and the paper is ready for submission, "
                "or when the orchestrator needs to redirect the pipeline."
            ),
        ),
    ]
    logger.info("paper_writing: built %d tools (buf=%s)", len(tools), buf is not None)
    return tools


__all__ = [
    "query_experiment_records",
    "lookup_citation",
    "draft_section",
    "embed_figure",
    "save_draft",
    "export_report_html",
    "export_report_docx",
    "load_review",
    "AGENT_TOOLS",
    "build_tools",
]
