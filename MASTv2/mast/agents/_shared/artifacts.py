"""The shared-artifact registry — DERIVED from what the agents' tools actually do.

Why this module exists (, and the 2026-07-11 scout audit):

The old model was a hand-written one-to-one ``{agent: artifact}`` table in the
API layer. It was wrong in every way a model can be wrong:

  * it claimed each artifact had exactly ONE writer, and that every other agent
    read it — so paper_review "read" the .sxm scan files, which it never opens;
  * 5 of its 6 artifact ids named ``MASTState`` fields that are DEAD (zero reads,
    zero writes anywhere in the tree — the Pydantic models are never even
    instantiated);
  * it had to be maintained by hand, so it drifted from reality silently.

This module fixes the *cause*, not the symptom. An artifact is declared once —
with the concrete place it lives — and the read/write edges are DERIVED by
looking at which tools each agent is actually holding. Give an agent a tool and
its edge appears in the graph; take the tool away and the edge goes. There is no
second table to keep in sync (the same principle ``tool_skills.py`` uses to
bridge agent tools into the skill menu).

Consequences the derivation surfaces rather than hides — both are real:
  * several artifacts genuinely have MULTIPLE writers (figures: data_processing
    renders them, paper_writing copies them into the manuscript; the experiment
    DB and the experiment plan: instrument_control and experiment_design both
    hold the tools; the long-term memory: every agent carries the middleware).
  * an agent's edges move the moment its tool grant moves — ``experiment_design``
    became a writer of ``experiment_plan`` when the runtime handed it
    ``create_plan`` (2026-07-27), and this graph reports that only because it
    derives from the grant (``meta_tools.DESIGN_TOOL_NAMES``) instead of from a
    remembered summary of it. Until 2026-08-14 it derived from ``LIFECYCLE`` —
    a set nothing granted — and so kept drawing an agent that could not persist
    its own design, months after that had stopped being true.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

#: ``<stem>_v003`` → ``<stem>``. The version suffix the legacy naming scheme used;
#: ``documents/migrate.py`` groups a whole family under the stripped base, so a
#: "已迁移?" check has to compare on the same key. Mirrors the identical regex in
#: ``documents/store.resolve_ref``.
_LEGACY_VER_RE = re.compile(r"_v\d+$")

logger = logging.getLogger(__name__)

Access = Literal["read", "write"]

#: Agents that hold NO vision-buffer tools. The buffer trio (tip status / scan
#: progress / tip history) is handed to every agent that takes a ``buf`` —
#: research_director does not, and its ``build_tools`` has no buffer parameter at
#: all: live tip and scan state are not Campaign-layer inputs.
#:
#: Named rather than inferred, and checked against the agent's REAL assembly by
#: ``test_artifact_class_status.py`` — an exception that only agrees with itself
#: is how this module's edges went fictional in the first place.
NO_BUFFER_AGENTS: frozenset[str] = frozenset({"research_director"})

PIPELINE: tuple[str, ...] = (
    "research_director",
    "literature",
    "experiment_design",
    "instrument_control",
    "data_processing",
    "paper_writing",
    "paper_review",
)


@dataclass(frozen=True)
class Artifact:
    """One shared artifact. ``store`` names where it PHYSICALLY lives, so every
    claim in this registry can be checked against the disk — a store must never
    name a MASTState slot that nothing reads or writes."""

    id: str
    label: str
    kind: Literal["file", "db", "index", "buffer"]
    store: str
    #: True when the operator can meaningfully EDIT it and an agent will then
    #: read the edit back (file-backed artifacts only). A DB/buffer artifact is
    #: not operator-editable — pretending otherwise is what made the old
    #: artifact editor a write-only black hole.
    editable: bool = False
    #: Set for artifacts written by the system rather than by any agent.
    system_written: bool = False
    #: This class arrives in BULK — one ordinary run produces tens to hundreds.
    #: A list that flattens it buries every other artifact ("制品
    #: 区域就应该分目录，否则看起来太乱了"). Consumers group by class and start
    #: a high-volume group collapsed; how to render it is still the UI's call,
    #: but WHICH classes flood is a fact about the artifact, so it lives here.
    high_volume: bool = False


# ── The artifacts, and where they really live ────────────────────────────────
ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        # The Campaign layer's product (2026-08-21). It is upstream of every other
        # artifact here — everything below records 做了什么 / 做出了什么; this one
        # records 为什么做, and it is the only one whose lifetime is weeks to months.
        id="research_campaign", label="科研纲领（campaign）", kind="db",
        store="experiments/mast_experiments_v2.db · campaigns 表（logging/v2）",
    ),
    Artifact(
        id="literature_library", label="文献库（指针集）", kind="index",
        # LIT *writes* the library registry (sets of work_id pointers). The 36k
        # OpenAlex index itself is a read-only shipped asset — crediting LIT with
        # writing it would be crediting it with writing something it only reads.
        store="artifacts/literature_libs/registry.json",
    ),
    Artifact(
        id="literature_report", label="文献报告", kind="file",
        # LIT's first written artifact. Before 2026-07-29 the literature agent had
        # no file-writing tool at all, so a survey it spent a dozen searches
        # building existed only in the conversation — and the compaction
        # middleware could summarise it away.
        store="实验文件夹 reports/<doc-dir>/vNNN.md（kind=literature_report）",
        editable=True,
    ),
    Artifact(
        id="experiment_plan", label="实验方案", kind="db",
        # NOT MASTState.experiment_plan — that field is dead. The plan that
        # actually persists goes through mast.planning.plan_store.
        store="planning DB + 实验文件夹 plans/<doc-dir>/（v001.md 定义 + progress.md 进度）",
        # The definition markdown is a document version like any other, so an
        # operator edit is a new version the tools read back — it used to be a
        # file PlanStore overwrote on every progress tick, which is why this said
        # editable=False.
        editable=True,
    ),
    Artifact(
        id="scan_files", label="扫描数据 (.sxm)", kind="file",
        store="Nanonis 会话目录（core.scan_registry 记录）",
        # The only class that floods: every SaveScan adds one, so a night's run
        # is hundreds of rows against a handful of everything else.
        high_volume=True,
    ),
    Artifact(
        id="experiment_records", label="实验记录", kind="db",
        store="experiments/mast_experiments.db",
    ),
    Artifact(
        id="figures", label="图表", kind="file",
        # PW no longer copies into data/figures/ — embed_figure puts the figure in
        # the experiment's own pool so the document and its images move together.
        store="data/figures/ + artifacts/mosaics|montages/（DP 渲染）"
              " → 实验文件夹 reports/_assets/（PW 入稿）",
    ),
    Artifact(
        # Covers both document kinds this class holds: experiment_report (the
        # default — an internal record) and paper_draft (a manuscript for
        # submission). On disk they are told apart by doc.json.kind and the
        # rpt__ / draft__ directory prefix; before 2026-07-29 they were literally
        # indistinguishable, differing only in how the LLM worded the title.
        id="draft", label="实验报告 / 论文草稿", kind="file",
        store="实验文件夹 reports/<doc-dir>/vNNN.md（kind=experiment_report|paper_draft）",
        editable=True,
    ),
    Artifact(
        id="review", label="评审报告", kind="file",
        store="实验文件夹 reports/<doc-dir>/vNNN.md（kind=review，doc.json 记 target_doc_id）",
        editable=True,
    ),
    Artifact(
        id="memory", label="长期记忆 / 洞察", kind="db",
        store="cognition store（memory 中间件）",
    ),
    Artifact(
        id="vision_buffer", label="视觉缓冲（针尖/扫描事件）", kind="buffer",
        store="vision_buffer.wal.sqlite",
        # Standing invariant: agents never write the buffer. The old
        # one-writer-per-artifact table literally could not express "no agent
        # writes this".
        system_written=True,
    ),
)

ARTIFACT_BY_ID: dict[str, Artifact] = {a.id: a for a in ARTIFACTS}


# ── Tool → artifact access. THE single place the graph is declared. ──────────
#
# Keys are tool names as registered on the agents (``@tool("name")``). The edges
# are then derived by asking each agent which of these tools it actually holds,
# so an edge can never claim an access the agent has no way to perform.
TOOL_ACCESS: dict[str, tuple[str, Access]] = {
    # ── research_director: the campaign row (the hypothesis + its commission) ──
    #    Lives in _shared/campaign_tools.py, so any agent the orchestrator hands
    #    them to becomes a reader/writer here automatically — the whole point of
    #    deriving the graph instead of writing it down.
    "campaign_list": ("research_campaign", "read"),
    "campaign_get": ("research_campaign", "read"),
    "campaign_create": ("research_campaign", "write"),
    "campaign_update": ("research_campaign", "write"),
    "campaign_request_plan": ("research_campaign", "write"),
    # ── literature: the library registry (pointer sets into the big index) ──
    "lib_create": ("literature_library", "write"),
    "lib_add": ("literature_library", "write"),
    "lib_remove": ("literature_library", "write"),
    "lib_switch": ("literature_library", "write"),
    "lib_list": ("literature_library", "read"),
    "lib_search": ("literature_library", "read"),
    "search_local_corpus": ("literature_library", "read"),
    "search_papers": ("literature_library", "read"),
    "propose_citations": ("literature_library", "read"),
    "lookup_citation": ("literature_library", "read"),
    # ── the literature SURVEY itself (2026-07-29). LIT had zero write edges to
    #    anything but the pointer registry: the prose it produced was never
    #    persisted, so "综述只活在对话里" was a structural fact, not an oversight
    #    by any one run. Implemented in literature/tools.py.
    "save_literature_report": ("literature_report", "write"),
    # ── experiment plan: all of these are META-tools. instrument_control gets
    #    the full set; experiment_design gets DESIGN_TOOL_NAMES.
    #
    #    2026-08-28: this comment used to say DESIGN_TOOL_NAMES excludes
    #    approve_plan ("an agent must not approve its own plan"). That stopped
    #    being true on 2026-08-20, when the permission model moved from "trim
    #    the tool face by role" to "full tool face + server-side envelope
    #    adjudication" — approve_plan is in DESIGN_TOOL_NAMES today. Read the
    #    constant, not this comment; it is the source of truth and
    #    test_xd_design_tool_surface.py pins it.
    "create_plan": ("experiment_plan", "write"),
    "approve_plan": ("experiment_plan", "write"),
    "advance_plan": ("experiment_plan", "write"),
    "pause_plan": ("experiment_plan", "write"),
    "resume_plan": ("experiment_plan", "write"),
    "list_plans": ("experiment_plan", "read"),
    "get_plan_progress": ("experiment_plan", "read"),
    "show_plan_on_map": ("experiment_plan", "read"),
    # ── scans ──
    "SaveScan": ("scan_files", "write"),          # an IC SKILL, not a @tool
    "GetLatestScanFile": ("scan_files", "read"),  # ditto
    "load_scan_file": ("scan_files", "read"),     # meta-tool
    "get_latest_scan_info": ("scan_files", "read"),
    "load_scan": ("scan_files", "read"),
    "get_latest_scan_file": ("scan_files", "read"),
    "fft_2d": ("scan_files", "read"),
    "plane_subtract": ("scan_files", "read"),
    "detect_defects": ("scan_files", "read"),
    "fit_sts_peaks": ("scan_files", "read"),
    "find_flat_region": ("scan_files", "read"),
    "assess_cluster_roundness": ("scan_files", "read"),
    # §10 on-request analysis library (2026-07-27). All read-only w.r.t. the
    # measurement: the preprocessing ones write a NEW .npy beside it, never over
    # it, which is why they are "read" on scan_files and not "write".
    # ── experiment records: IC holds the full meta-tool set, XD holds the
    #    lifecycle part of DESIGN_TOOL_NAMES — so BOTH write this DB (a genuine
    #    multi-writer).
    "start_experiment": ("experiment_records", "write"),
    "end_experiment": ("experiment_records", "write"),
    "rename_experiment": ("experiment_records", "write"),
    "start_sample": ("experiment_records", "write"),
    "end_sample": ("experiment_records", "write"),
    "rename_sample": ("experiment_records", "write"),
    "query_experiment_records": ("experiment_records", "read"),
    "query_past_experiments": ("experiment_records", "read"),
    "lookup_sample": ("experiment_records", "read"),
    # research_director reads the SAME records through the v2 store instead of a
    # fourth copy of the v1 SQL (paper_writing has one, experiment_design copied
    # it, and the copy's comment says so). Same artifact, different door.
    "campaign_experiments": ("experiment_records", "read"),
    "campaign_claims": ("experiment_records", "read"),
    # ── figures: DP renders them, PW copies them into the manuscript ──
    # DP's baseline deliverable — the two most-used figure producers were missing
    # from this table entirely (2026-07-27). The figures edge already existed via
    # mosaic/montage so no drawn edge changes; the table just stopped under-reporting.
    "plot_scan": ("figures", "write"),
    "plot_spectrum": ("figures", "write"),
    # The first READ edge this artifact has ever had. paper_writing could only
    # embed a figure whose absolute path it was handed in conversation; nothing
    # could enumerate what had been rendered (2026-07-27).
    "list_figures": ("figures", "read"),
    "mosaic_scans": ("figures", "write"),
    "embed_figure": ("figures", "write"),
    # ── the PW ⇄ PR revision loop ──
    "save_draft": ("draft", "write"),
    "draft_section": ("draft", "write"),
    "load_draft": ("draft", "read"),
    "check_citations": ("draft", "read"),
    "check_methodology": ("draft", "read"),
    "check_data_reasoning": ("draft", "read"),
    "produce_review": ("draft", "read"),
    "save_review": ("review", "write"),
    "load_review": ("review", "read"),   # added 2026-07-11 — see below
    # ── 交付件导出（2026-07-28 html / 2026-07-29 docx）。两者都是对 draft 的
    #    **读**：产物落 <exp>/exports/，那不是被追踪的 artifact 类（它是交付
    #    快照，不是任何 agent 的输入）。此前 export_report_html 完全不在这张表
    #    里 —— 守卫只查「表里的名字必须存在」这一个方向，查不出漏登记。
    "export_report_html": ("draft", "read"),
    "export_report_docx": ("draft", "read"),
    # ── vision buffer: READ-ONLY for agents, always ──
    "read_latest_tip_status": ("vision_buffer", "read"),
    "get_scan_progress": ("vision_buffer", "read"),
    "get_tip_history_since": ("vision_buffer", "read"),
    # ── long-term memory: every agent carries the memory tools ──
    # These are the names ``memory_tools.make_memory_tools`` really registers.
    # Until 2026-07-28 this said ``remember_insight`` / ``recall_insights`` —
    # two names that appear NOWHERE else in the tree. The edges they drew were
    # invented, and the guard test could not catch it because
    # ``agent_tool_names()`` hard-coded the same two ghosts into every agent's
    # set: the map was checked against a copy of itself.
    "memory_write": ("memory", "write"),
    "memory_read": ("memory", "read"),
    "memory_list": ("memory", "read"),
    "memory_search": ("memory", "read"),
}


@dataclass
class Flow:
    """The derived read/write edges of one artifact."""

    artifact: Artifact
    writers: list[str] = field(default_factory=list)
    readers: list[str] = field(default_factory=list)

    @property
    def multi_writer(self) -> bool:
        return len(self.writers) > 1


def _tool_names(tools: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for t in tools or ():
        name = getattr(t, "name", None) or (t if isinstance(t, str) else None)
        if name:
            out.add(str(name))
    return out


def _module_tool_names(mod: Any) -> set[str]:
    """Every LangChain tool object defined at a module's top level.

    Used where an agent's tools are only assembled inside ``build_tools(...)``
    with arguments we don't have (experiment_design wants a registry): scan the
    module for the tool objects themselves rather than guessing their names.
    """
    out: set[str] = set()
    for attr in dir(mod):
        obj = getattr(mod, attr, None)
        name = getattr(obj, "name", None)
        # a LangChain tool: has .name and .invoke, and is not a class
        if isinstance(name, str) and callable(getattr(obj, "invoke", None)):
            out.add(name)
    return out


def agent_tool_names() -> dict[str, set[str]]:
    """Tool names each pipeline agent actually holds.

    Uses the agents' REAL assembly (``build_tools``), not the ``AGENT_TOOLS``
    constant: for several agents that constant is only part of the story (the
    literature agent's library tools and experiment_design's introspection tools
    are added inside build_tools), and instrument_control has no static list at
    all — its tools are the SkillRegistry's skills, wrapped at build time. Reading
    the constant instead of the assembly is how a model quietly stops describing
    the system.

    On top of that the runtime grants meta-tools asymmetrically: IC receives the
    FULL set, experiment_design the DESIGN_TOOL_NAMES slice (lifecycle + tip
    read-only + plan drafting + knowledge, no approvals and no writes to the
    surface). That asymmetry is DATA here — read from the same constant the
    runtime filters by, so the graph derives it rather than asserting it.

    Any failure degrades to fewer edges for that agent; a read-only view must
    never crash.
    """
    out: dict[str, set[str]] = {a: set() for a in PIPELINE}

    # One discovered registry serves both instrument_control (whose tools ARE the
    # skills) and experiment_design (whose build_tools wants one).
    registry = None
    try:
        # discover_instrument_skills(), NOT a bare SkillRegistry().discover().
        # The bare call loads builtins + composite + paper/ (418); IC only ever
        # mounts builtins + composite (384). The 34-skill difference is the whole
        # paper/ analysis library — FitGap_BCS, UnmixSpectra, DetectAtoms_FCN … —
        # which this table was reporting as instrument_control tools even though
        # IC cannot call a single one of them. They belong to data_processing
        # (now bridged there under snake_case @tool names, 2026-07-27).
        #
        # Latent rather than active: TOOL_ACCESS is keyed by @tool name and the
        # ghosts are class names, so ghosts ∩ TOOL_ACCESS = [] and derive_flow()
        # draws no wrong edge today. But this function is the public answer to
        # "which agent holds what", and it has been answering wrong.
        from mast.agents.instrument_control.tools import discover_instrument_skills
        registry = discover_instrument_skills()
    except Exception as exc:  # noqa: BLE001
        logger.debug("artifact flow: skill registry unavailable: %s", exc)

    for agent in PIPELINE:
        if agent == "instrument_control":
            continue  # its tools are the registry's skills — added below
        try:
            mod = __import__(f"mast.agents.{agent}.tools",
                             fromlist=["build_tools", "AGENT_TOOLS"])
            build = getattr(mod, "build_tools", None)
            names: set[str] = set()
            if callable(build):
                # Call the REAL assembly. Signatures differ (XD also wants the
                # registry, and its @tools are built by factory functions inside
                # build_tools — they are not module attributes at all, so nothing
                # short of calling it can learn their names).
                for args in ((None,), (None, registry)):
                    try:
                        names = _tool_names(build(*args))
                        break
                    except TypeError:
                        continue
            out[agent] = names or _tool_names(getattr(mod, "AGENT_TOOLS", ()))
        except Exception as exc:  # noqa: BLE001
            logger.debug("artifact flow: %s tools unavailable: %s", agent, exc)

    # instrument_control: its tools ARE the registry's skills (wrap_skill wraps
    # every one of them at build time).
    if registry is not None:
        try:
            out["instrument_control"] |= {m.name for m in registry.list_skills()}
        except Exception as exc:  # noqa: BLE001
            logger.debug("artifact flow: skill list unavailable: %s", exc)

    # Meta-tools (runtime wiring): IC = the full set, XD = DESIGN_TOOL_NAMES.
    # The SAME constant runtime.py filters by — before, this line derived from
    # `meta & LIFECYCLE`, which is not what runtime grants, so the graph was
    # missing XD's write edge into experiment_plan while a green test pinned the
    # absence. A derivation that defines its own input can only agree with itself.
    try:
        from mast.agents._shared.meta_tools import (
            DESIGN_TOOL_NAMES, META_TOOL_NAMES,
        )
        meta = set(META_TOOL_NAMES)
        design = set(DESIGN_TOOL_NAMES)
    except Exception as exc:  # noqa: BLE001
        logger.debug("artifact flow: meta-tools unavailable: %s", exc)
        meta, design = set(), set()
    out["instrument_control"] |= meta
    out["experiment_design"] |= (meta & design)

    # Buffer tools (make_buffer_tools) and memory tools (make_memory_tools) are
    # attached to every agent at build time — the factories are called here for
    # the same reason build_tools() is: a hand-typed name list is a second table
    # that drifts. It drifted: the memory pair was written as remember_insight /
    # recall_insights, names no factory has ever produced, and because THIS set
    # was the thing the guard test compared TOOL_ACCESS against, the fiction
    # validated itself. Both factories only dereference their argument inside a
    # tool body, so None is enough to learn the names.
    #
    # ``skip`` is what keeps this blanket add honest: the memory tools really do
    # go to every agent (the orchestrator's ``_shared()`` hands them out
    # unconditionally), but the buffer trio only goes to agents that take a
    # ``buf``. Adding it to one that does not would draw a ``vision_buffer`` READ
    # edge for an agent holding no such tool — precisely the invented edge this
    # module exists to stop. See :data:`NO_BUFFER_AGENTS`.
    for factory_mod, factory_name, skip in (
        ("mast.agents._shared.buffer_tools", "make_buffer_tools",
         NO_BUFFER_AGENTS),
        ("mast.agents._shared.memory_tools", "make_memory_tools", frozenset()),
    ):
        try:
            mod = __import__(factory_mod, fromlist=[factory_name])
            names = _tool_names(getattr(mod, factory_name)(None))
        except Exception as exc:  # noqa: BLE001 — fewer edges, never a crash
            logger.debug("artifact flow: %s unavailable: %s", factory_name, exc)
            continue
        for agent in PIPELINE:
            if agent in skip:
                continue
            out[agent] |= names
    return out


def derive_flow() -> list[Flow]:
    """Build the artifact data-flow graph from the agents' live tool lists.

    An edge exists iff an agent HOLDS a tool that performs that access. Nothing
    is asserted that no tool can do.
    """
    held = agent_tool_names()
    flows = {a.id: Flow(artifact=a) for a in ARTIFACTS}
    for agent in PIPELINE:
        for tool in held.get(agent, ()):  # noqa: PLC0206
            edge = TOOL_ACCESS.get(tool)
            if edge is None:
                continue
            art_id, access = edge
            flow = flows.get(art_id)
            if flow is None:
                continue
            bucket = flow.writers if access == "write" else flow.readers
            if agent not in bucket:
                bucket.append(agent)
    # keep the pipeline's own order, so the graph reads left→right
    order = {a: i for i, a in enumerate(PIPELINE)}
    for f in flows.values():
        f.writers.sort(key=lambda a: order.get(a, 99))
        f.readers.sort(key=lambda a: order.get(a, 99))
        # A writer is implicitly a reader of its own artifact; don't double-list.
        f.readers = [r for r in f.readers if r not in f.writers]
    return [flows[a.id] for a in ARTIFACTS]


# ── The artifacts that actually EXIST on disk right now ──────────────────────

#: ``documents`` kind → artifact CLASS. The two report kinds share the ``draft``
#: class on purpose: they are the same thing to every consumer of this list (a
#: versioned markdown document the operator can open and edit), and doc.json's
#: kind is what tells them apart where the difference matters.
KIND_ARTIFACT: dict[str, str] = {
    "literature_report": "literature_report",
    "experiment_plan": "experiment_plan",
    "experiment_report": "draft",
    "paper_draft": "draft",
    "review": "review",
}


def list_existing() -> list[dict]:
    """Enumerate the REAL artifacts currently on disk.

    Replaces ``task["artifacts"]``, which had no producer anywhere in the tree
    and was therefore always ``{}`` — the "produced artifacts" panel could only
    ever be empty, and a mock in the test suite hid that. Everything below is
    read from the true stores; every entry is a file you can open.

    Documents (reports, drafts, reviews, literature surveys, plans) come from
    ``mast.documents.store``, whose ``doc_id`` is a real stable identity rather
    than the ``"<class>:<file stem>"`` string this function used to synthesise —
    that string changed whenever the title was reworded, so an id handed to the
    editor could stop resolving. The old directory walks over
    ``data/{drafts,reviews}`` and ``plans/plan_*.md`` are kept as a DEGRADATION
    path: they run only for a class the store returned nothing for, so
    pre-migration files still show up while migrated ones are not listed twice.
    """
    from mast.agents._shared.data_paths import drafts_dir, figures_dir, reviews_dir

    out: list[dict] = []

    def _add(art_id: str, path: Path, *, doc_id: str = "", name: str = "",
             editable: bool | None = None) -> None:
        try:
            st = path.stat()
        except OSError:
            return
        art = ARTIFACT_BY_ID.get(art_id)
        if art is None:
            return
        label = name or path.name
        out.append({
            # The CLASS of artifact ("draft") — what the flow graph draws.
            "artifact_id": art_id,
            # The identity of THIS ONE DOCUMENT/FILE. Five drafts all carrying
            # artifact_id="draft" are indistinguishable to anything that wants to
            # OPEN one — the editor would always land on whichever the backend
            # guessed. This is the id every per-file endpoint (documents.py,
            # artifacts_edit.py) addresses.
            "doc_id": doc_id or f"{art_id}:{path.stem}",
            "name": label,
            # What the UI shows as the row's preview line. Same string as `name`;
            # named separately so a consumer does not have to know that a
            # document's "filename" (v002.md) is useless to a reader.
            "preview": label,
            "path": str(path),
            "bytes": st.st_size,
            "modified_at": st.st_mtime,
            # Per ROW, not per class: the class says "this KIND of thing can be
            # edited", but a legacy file that was never imported into the document
            # store has no endpoint that would accept an edit of it. Offering the
            # button anyway is how the artifact editor became a write-only black
            # hole in the first place.
            "editable": art.editable if editable is None else editable,
        })

    # ── documents: the store is the primary enumerator ──
    migrated_stems: set[str] = set()
    try:
        from mast.documents import store as _doc_store
        for entry in _doc_store().list():
            art_id = KIND_ARTIFACT.get(entry.meta.kind)
            if art_id is None:
                continue
            p = entry.version_path()
            if p is None:
                continue  # metadata with no readable version — nothing to open
            title = entry.meta.title or entry.dir.name
            unfiled = " · 未归属" if entry.meta.root_kind == "unfiled" else ""
            _add(art_id, p, doc_id=entry.doc_id,
                 name=f"{title} v{entry.latest_version}{unfiled}")
            stem = entry.meta.legacy_stem
            if stem:
                migrated_stems.add(stem)
                migrated_stems.add(_LEGACY_VER_RE.sub("", stem))
    except Exception as exc:  # noqa: BLE001 — a read-only view never crashes
        logger.debug("artifact listing: document store unavailable: %s", exc)

    def _already_migrated(p: Path) -> bool:
        """Has this legacy file already been imported into the document store?

        Keyed on ``legacy_stem`` — the field ``documents/migrate.py`` writes for
        exactly this purpose — and on the version-stripped family base, because
        ``migrate_legacy_markdown`` groups ``<stem>_v001/_v002/…`` into ONE
        document and records the family base. Same two-step the store's own
        ``resolve_ref`` does.
        """
        if not migrated_stems:
            return False
        return (p.stem in migrated_stems
                or _LEGACY_VER_RE.sub("", p.stem) in migrated_stems)

    # ── legacy directories ──
    #
    # These used to be skipped WHOLESALE as soon as the store held ANY document
    # of that class (`if art_id in doc_classes: continue`). Two facts turn that
    # into silent data loss rather than a tidy de-duplication:
    #
    #   * ``documents/migrate.py`` has zero production callers, so on a real rig
    #     the legacy files were NEVER imported — this branch was the only thing
    #     listing them; and
    #   * the guard is per CLASS, not per FILE. Writing ONE new draft puts
    #     "draft" in the set and every pre-migration draft vanishes from the
    #     artifact list in the same instant. The bytes are still on disk; nothing
    #     enumerates them, and nothing says so.
    #
    # De-duplication is a per-FILE question, so ask it per file: skip only the
    # legacy files a migration actually imported (``legacy_stem``). Migration
    # COPIES and leaves the original in place ("复制导入，原文件不动"), so without
    # this check a migrated file would legitimately appear twice.
    for art_id, d, suffixes in (
        ("draft", drafts_dir(), (".md", ".pdf")),
        ("review", reviews_dir(), (".md",)),
    ):
        try:
            if d.is_dir():
                for p in d.iterdir():
                    if p.suffix.lower() in suffixes and not _already_migrated(p):
                        _add(art_id, p)
        except OSError:
            pass

    # figures are NOT documents — this directory is their only enumerator.
    try:
        fd = figures_dir(create=False)
        if fd.is_dir():
            for p in fd.iterdir():
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf"):
                    _add("figures", p)
    except OSError:
        pass

    # scans: whatever the run actually saved (the registry the IC skills feed)
    try:
        from mast.core.scan_registry import recent_scan_paths
        for s in recent_scan_paths(20):
            _add("scan_files", Path(s))
    except Exception:  # noqa: BLE001
        pass

    # plans: pre-migration PlanStore rendered every saved plan to
    # plans/plan_<id>.md next to the experiment DB (PlanStore's own default:
    # db_path.parent / "plans"). The DB row is the source of truth, but the
    # markdown is the thing an operator can OPEN — and without it the 实验方案
    # class had no enumerator at all, so it was reported as "not produced yet"
    # forever, including right after create_plan had just written one
    #. Plans that are already documents come from the store above.
    # Same per-FILE rule as the drafts/reviews walk above: a plan that HAS been
    # migrated carries legacy_stem="plan_<id>" and is listed by the store; one
    # that has not is listed from here. Gating on "does the class have any
    # document at all" hid every un-migrated plan the moment one plan document
    # existed.
    try:
        for p in sorted(plans_dir().glob("plan_*.md")):
            if _already_migrated(p):
                continue
            # editable=False even though the CLASS is now editable: a plan that
            # is already a document has a versioned definition file an operator
            # edit lands on, but one of these legacy files is a whole-file
            # re-render PlanStore owned — no endpoint takes an edit of it.
            _add("experiment_plan", p, editable=False)
    except OSError:
        pass

    out.sort(key=lambda e: e["modified_at"], reverse=True)
    return out


def plans_dir() -> Path:
    """The LEGACY global plan directory — ``<experiment db>/../plans``.

    Superseded 2026-07-29: plans are documents in the owning experiment's folder
    (``<experiment>/plans/<doc-dir>/``). Kept so plans written by earlier builds
    are still enumerated until they are migrated.

    Derived from :func:`data_paths.experiment_db_path` rather than hard-coded so
    it follows ``MAST_EXPERIMENT_DB``, exactly as ``PlanStore``'s own default
    (``db_path.parent / "plans"``) does."""
    from mast.agents._shared.data_paths import experiment_db_path

    return experiment_db_path().parent / "plans"


# ── Per-CLASS production status ──────────────────────────────────────────────
#
# (2026-07-27): every one of the artifact classes sat at
# 待产出 forever. Two independent causes, and this function addresses the second.
#
#   1. The UI compared per-FILE ids ("draft:Au111_v001") against CLASS ids
#      ("draft"). Disjoint id spaces, so the "already produced?" test could never
#      be true. Fixed in the frontend.
#   2. FIVE classes had no enumerator at all. ``list_existing()``
#      walks directories, and literature_library / experiment_records / memory /
#      vision_buffer do not live in directories — they live in SQLite tables and
#      a JSON registry. Nothing anywhere reported them as existing, so even with
#      (1) fixed they would still read "尚未产出" while their stores filled up.
#
# So each class is asked ITS OWN store. The rule that keeps this honest: an
# unreadable store reports ``known=False``, never ``count=0``. "I cannot tell"
# and "there is nothing" are different answers, and only one of them is a lie
# when the store is simply locked.


@dataclass(frozen=True)
class ClassStatus:
    """Whether one artifact CLASS has anything in it, read from its own store."""

    artifact: Artifact
    #: Meaningful only when ``known``. Never a guess.
    count: int
    #: False when the store could not be probed at all.
    known: bool
    #: One honest line: what was counted, or why it could not be.
    detail: str

    @property
    def produced(self) -> bool:
        return self.known and self.count > 0


def _sqlite_count(db: Path, sql: str) -> int | None:
    """Row count over a READ-ONLY connection. None = could not read.

    Read-only on purpose. ``sqlite3.connect`` creates the file when it is
    missing, so a plain connect would turn "there is no experiment database yet"
    into "an experiment database with 0 rows" — and leave a real empty DB behind
    to make the lie permanent.

    A short timeout because this feeds an endpoint the UI polls every 6 s: a
    locked DB must degrade, never block the event loop.
    """
    import sqlite3

    if not db.is_file():
        return 0  # a store that does not exist really is empty
    try:
        conn = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=0.5)
    except Exception as exc:  # noqa: BLE001
        logger.debug("artifact status: cannot open %s: %s", db, exc)
        return None
    try:
        return int(conn.execute(sql).fetchone()[0])
    except sqlite3.OperationalError as exc:
        # "no such table" in a DB that exists = the writer has never run. That
        # is a genuine zero, not an unknown.
        if "no such table" in str(exc).lower():
            return 0
        logger.debug("artifact status: %s on %s: %s", sql, db, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("artifact status: %s on %s: %s", sql, db, exc)
        return None
    finally:
        conn.close()


def _count_literature_members() -> tuple[int | None, str]:
    """Pinned works across every library — NOT the number of libraries.

    ``LibraryRegistry`` always materialises a GLOBAL library, so a virgin
    install already has one and "1 library" would read as "produced" before
    anything happened. What LIT actually produces here is MEMBERSHIP (lib_add),
    so membership is what gets counted."""
    from mast.knowledge.libraries import list_libraries

    libs = list_libraries()
    total = sum(int(lib.get("member_count", 0) or 0) for lib in libs)
    return total, f"{len(libs)} 个文献库 · 共 {total} 篇入库"


def _count_experiments() -> tuple[int | None, str]:
    from mast.agents._shared.data_paths import experiment_db_path

    db = experiment_db_path()
    n = _sqlite_count(db, "SELECT COUNT(*) FROM experiments")
    return n, (f"{n} 条实验记录" if n is not None else f"无法读取 {db}")


def _count_campaigns() -> tuple[int | None, str]:
    """Research campaigns in the v2 records DB.

    Two things it must NOT do, both of which ``logging.v2.storage.open_store()``
    would do for us:
      * **create the store.** ``open_store`` runs the whole DDL as a side effect of
        being asked where the file is, so a status probe would leave a real empty
        v2 database behind and make "there is nothing yet" permanently true-looking.
        ``_sqlite_count`` opens ``mode=ro`` and answers 0 for a missing file.
      * **resolve the path from the CWD.** ``open_store``'s default is
        ``Path(".")``, which no test can redirect without chdir'ing the process —
        so this probe would read the developer's real campaigns inside a fixture
        that carefully redirected every other store. ``v2_experiment_db_path``
        honours ``MAST_DATA_DIR`` then ``project_root()``; see its docstring for
        why the two resolutions agree in every real deployment.
    """
    from mast.agents._shared.data_paths import v2_experiment_db_path

    db = v2_experiment_db_path()
    n = _sqlite_count(db, "SELECT COUNT(*) FROM campaigns")
    return n, (f"{n} 个科研纲领" if n is not None else f"无法读取 {db}")


def _count_memories() -> tuple[int | None, str]:
    """Long-term memory rows. MemoryStore shares the experiment DB file."""
    from mast.agents._shared.data_paths import experiment_db_path

    db = experiment_db_path()
    n = _sqlite_count(db, "SELECT COUNT(*) FROM memory")
    return n, (f"{n} 条长期记忆" if n is not None else f"无法读取 {db}")


def _count_buffer_events() -> tuple[int | None, str]:
    from mast._runtime_paths import project_root

    db = project_root() / "experiments" / "vision_buffer.wal.sqlite"
    ev = _sqlite_count(db, "SELECT COUNT(*) FROM event_journal")
    tips = _sqlite_count(db, "SELECT COUNT(*) FROM tip_status_journal")
    if ev is None and tips is None:
        return None, f"无法读取 {db}"
    total = (ev or 0) + (tips or 0)
    return total, f"{ev or 0} 条事件 · {tips or 0} 条针尖状态"


#: artifact_id → probe. File-backed classes are absent: they are counted from
#: ``list_existing()``, which already enumerates them file by file.
_STORE_PROBES: dict[str, Any] = {
    "research_campaign": _count_campaigns,
    "literature_library": _count_literature_members,
    "experiment_records": _count_experiments,
    "memory": _count_memories,
    "vision_buffer": _count_buffer_events,
}


def class_status(existing: list[dict] | None = None) -> list[ClassStatus]:
    """Production status of EVERY artifact class in :data:`ARTIFACTS`, in registry
    order — a class the backend omits is a class the UI cannot show a state for.

    Pass the result of :func:`list_existing` to avoid walking the directories
    twice (the API does exactly one pass and shares it).
    """
    rows = list_existing() if existing is None else existing
    counted: dict[str, int] = {}
    for r in rows:
        counted[r["artifact_id"]] = counted.get(r["artifact_id"], 0) + 1

    out: list[ClassStatus] = []
    for art in ARTIFACTS:
        probe = _STORE_PROBES.get(art.id)
        if probe is None:
            n = counted.get(art.id, 0)
            out.append(ClassStatus(artifact=art, count=n, known=True,
                                   detail=f"{n} 项"))
            continue
        try:
            n, detail = probe()
        except Exception as exc:  # noqa: BLE001 — a status view never crashes
            logger.debug("artifact status: %s probe failed: %s", art.id, exc)
            n, detail = None, f"无法读取（{exc.__class__.__name__}）"
        out.append(ClassStatus(artifact=art, count=n or 0, known=n is not None,
                               detail=detail))
    return out


__all__ = [
    "Access", "Artifact", "ARTIFACTS", "ARTIFACT_BY_ID", "ClassStatus",
    "KIND_ARTIFACT", "TOOL_ACCESS", "Flow", "PIPELINE", "NO_BUFFER_AGENTS",
    "agent_tool_names",
    "class_status", "derive_flow", "list_existing", "plans_dir",
]
