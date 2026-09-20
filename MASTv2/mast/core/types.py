"""Shared types: SkillResult, ParameterSpec, SafetyLevel, etc.

vendored from v1 mast/core/types.py 2026-04-23. Zero behavioural changes.
The v1 SkillMetadata / SkillResult / HardwareState / NanonisCallRecord shapes
are required by mast.skills.base.BaseSkill and consumed by every existing v1
skill. wrap_skill (mast.agents._shared.skill_adapter) introspects these.

Phase 4 may extend (not replace) — additions go below the "v2 extensions" mark
at file end. v1 schema is frozen during migration to keep the 130 builtin
skills working unmodified.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SafetyLevel(Enum):
    """Skill safety classification."""
    AUTO = "auto"           # Read-only, and small repeatable writes — execute immediately
    CONFIRM = "confirm"     # Scan params, spectroscopy, tip conditioning, lateral coarse
    DANGEROUS = "dangerous" # Must have human approval. Only 2 skills in the whole repo.

    # ⚠️ 2026-08-25 更正 DANGEROUS 这一行的注释。它原来写的是
    # "Tip conditioning, large pulses" —— 而**没有任何一个修针技能是 DANGEROUS**：
    # 2026-06-11 那次重新界定把它们放到了 CONFIRM，注释没跟上，于是这句话在
    # 代码里当了两个多月的假话，还被 experiment_design 的提示词抄了过去
    # （那边把 BiasPulse / TipShape / MotorMove 举成 DANGEROUS 的例子，
    #  而它们实际是 AUTO / AUTO / CONFIRM）。
    #
    # 真正标 DANGEROUS 的只有 CreateZCtrlPreset 与 LockNanonisUI ——
    # 都与针尖损伤无关。要看当下的真实分档就去枚举 metadata，别读这行注释。


class SkillCategory(Enum):
    """Skill operation category."""
    READ = "read"           # Only reads instrument state
    WRITE = "write"         # Modifies instrument state
    COMPOSITE = "composite" # Calls other skills
    ANALYSIS = "analysis"   # Data processing, no hardware interaction


class CitationType(Enum):
    """Type of citable work."""
    PAPER = "paper"
    SOFTWARE = "software"
    DATASET = "dataset"


# BibTeX/LaTeX special characters that would break compilation if left raw.
# We escape *selectively*: only characters that are virtually never intended as
# LaTeX markup in bibliographic metadata. We deliberately DO NOT touch
# ``\ { } ~ ^ $`` because curated entries legitimately use them for accented
# names (e.g. ``Jestil\"a``), en-dash page ranges, and the like. Escaping those
# would corrupt existing, intentional LaTeX.
_BIBTEX_ESCAPE = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
}


def _bibtex_escape(value: str) -> str:
    """Escape the LaTeX special characters that are unsafe in free-text BibTeX
    fields (author/title/journal/note). Leaves backslashes and braces alone so
    deliberate LaTeX accents survive."""
    if not value:
        return value
    return "".join(_BIBTEX_ESCAPE.get(ch, ch) for ch in value)


@dataclass
class Citation:
    """A citable reference (paper, software, or dataset)."""
    key: str                         # Unique ID, e.g. "DeepSPM_2020"
    authors: str                     # "Krull, A. et al."
    title: str
    year: int
    citation_type: CitationType = CitationType.PAPER
    journal: str = ""
    volume: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""
    note: str = ""                   # E.g. "Used for tip quality CNN"

    def to_bibtex(self) -> str:
        """Generate BibTeX entry."""
        if self.citation_type == CitationType.PAPER:
            entry_type = "article"
        elif self.citation_type == CitationType.SOFTWARE:
            entry_type = "software"
        else:
            entry_type = "misc"

        # Free-text fields get selective LaTeX escaping; identifier-like fields
        # (key, doi, url, numeric volume/pages) are left verbatim — escaping a
        # URL's "_" would break the link, and pages use intentional "--".
        lines = [f"@{entry_type}{{{self.key},"]
        lines.append(f"  author = {{{_bibtex_escape(self.authors)}}},")
        lines.append(f"  title = {{{_bibtex_escape(self.title)}}},")
        lines.append(f"  year = {{{self.year}}},")
        if self.journal:
            lines.append(f"  journal = {{{_bibtex_escape(self.journal)}}},")
        if self.volume:
            lines.append(f"  volume = {{{self.volume}}},")
        if self.pages:
            lines.append(f"  pages = {{{self.pages}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        if self.note:
            lines.append(f"  note = {{{_bibtex_escape(self.note)}}},")
        lines.append("}")
        return "\n".join(lines)

    def to_plaintext(self) -> str:
        """Generate plain-text citation string."""
        # Some entries (e.g. software with a TODO author) carry an empty author
        # string; emitting a bare "." reads as a rendering bug, so skip it.
        parts: list[str] = []
        if self.authors:
            parts.append(f"{self.authors}.")
        parts.append(f'"{self.title}."')
        if self.journal:
            parts.append(f"*{self.journal}*")
            if self.volume:
                parts.append(f"{self.volume}")
            if self.pages:
                parts.append(f"({self.pages})")
        parts.append(f"({self.year}).")
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        return " ".join(parts)


@dataclass
class ParameterSpec:
    """Specification for a single skill parameter."""
    name: str
    type: str                    # "float", "int", "str", "bool"
    description: str = ""
    unit: str = ""
    required: bool = True
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None


@dataclass
class SkillMetadata:
    """Complete metadata describing a skill's interface and safety profile.

    ``composition_level`` (added 2026-05-18) sorts skills by orchestration
    depth so the GUI / LLM can filter and reason about them:

      L0 — 1:1 Nanonis TCP wrapper (e.g. SetBias, Current_Get)
      L1 — small sequence of 2-3 L0 calls, no branching (e.g. WaitScanComplete,
           SaveScan)
      L2 — pure data / analysis, no Nanonis writes (e.g. FindFlatRegion,
           AssessClusterRoundness, AssessImageQuality)
      L3 — multi-step hardware workflow (e.g. TipPulse, FullScan, GridSTS,
           ShapeTipOnSurface)
      L4 — long autonomous / RL / paper-replication procedures
           (e.g. ConditionTip_DQN, ContinuousImaging_Auto)
      L5+ — overnight campaigns chaining dozens of L3/L4 skills

    Default = 0 so unmigrated skills don't break (they show up as "atomic").
    """
    name: str
    version: str = "1.0.0"
    category: SkillCategory = SkillCategory.READ
    safety_level: SafetyLevel = SafetyLevel.CONFIRM
    description: str = ""
    parameters: list[ParameterSpec] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    estimated_duration_s: float = 1.0
    rollback_skill: str | None = None
    tags: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    composition_level: int = 0
    # Capability tags for global operating-mode gating (added 2026-07-07).
    # Values in use: "bias_pulse" (electrical-pulse tip skills) and
    # "tip_shaping" (mechanical Z-plunge tip skills). Read by the mode-aware
    # safety / HITL middlewares to decide what SAFE / SEMI operating modes block
    # or route to confirmation. Empty for skills that touch neither — the common
    # case — so every existing skill is unaffected (frozen-schema-safe: this is
    # an additive field with a default, not a replacement).
    capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass
class NanonisCallRecord:
    """Record of a single Nanonis TCP call."""
    method: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    return_value: Any = None
    error: str = ""
    elapsed_s: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HardwareState:
    """Snapshot of instrument state at a point in time."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # True when NONE of the hardware reads in the last refresh succeeded (link
    # down): every value below is then carried forward from an older cache and
    # must NOT be treated as live. ``timestamp`` in that case is preserved from
    # the last good read (not bumped to "now"), so consumers can see the real
    # age instead of a falsely-fresh clock.
    stale: bool = False
    bias_v: float | None = None
    current_a: float | None = None
    z_pos_m: float | None = None
    x_pos_m: float | None = None
    y_pos_m: float | None = None
    z_controller_on: bool | None = None
    z_controller_status: str | None = None  # Off/On/Hold/SwitchingOff/SafeTip/Withdrawing
    # The active Z controller identifies the feedback channel used by engage
    # and approach. Surface it in live state instead of requiring a separate list
    # query. The identity fields are serializable scalars and strings.
    z_controller_name: str | None = None          # active controller name
    z_controller_index: int | None = None         # active controller index
    z_controller_names: list[str] | None = None   # all available controllers
    withdrawn: bool | None = None
    scan_running: bool | None = None
    setpoint_a: float | None = None
    #: Lock-in modulation state; None means unread, not off.
    #: Poll the controller because the operator may change it outside MAST.
    #: Modulation ripples must not be mistaken for spontaneous tip instability.
    lockin_mod_on: bool | None = None
    # Scan frame geometry — critical context so the LLM doesn't ask for nm
    # scans while the instrument is set to µm range (or vice versa).
    scan_center_x_m: float | None = None
    scan_center_y_m: float | None = None
    scan_width_m: float | None = None
    scan_height_m: float | None = None
    scan_angle_deg: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Result of a skill execution."""
    skill_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_s: float = 0.0
    state_before: HardwareState | None = None
    state_after: HardwareState | None = None
    nanonis_calls: list[NanonisCallRecord] = field(default_factory=list)
    # Optional one-line human summary for the chat ToolMessage. When None the
    # tool adapter falls back to str(data) (existing behaviour for every skill
    # that doesn't set it). Skills with large data (e.g. TipShapeWithReadback's
    # current/z traces) set this so chat stays concise.
    summary: str | None = None
    #: Rendered images this skill produced, as FILESYSTEM PATHS — the channel
    #: that lets an agent actually SEE what it measured (added 2026-08-11).
    #:
    #: ⚠️ PATHS, NEVER PIXELS. The list travels into the graph state and every
    #: message there is persisted and replayed by the SqliteSaver each turn, so a
    #: base64 payload parked here would be re-serialised for the rest of the
    #: session — the same rule that keeps tensors out of the checkpointer
    #: (项目规约 invariant). The bytes are materialised into a data URI only on
    #: the outbound request, by ``agents._shared.vision_mw``, and are never
    #: written back to state.
    #:
    #: Empty for every skill that doesn't set it, so this is additive: the 130
    #: builtins that predate it keep behaving exactly as before.
    images: list[str] = field(default_factory=list)


@dataclass
class SampleRecord:
    """Record of a physical sample within an experiment."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    name: str = ""
    description: str = ""
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str | None = None
    status: str = "active"  # active/completed/failed
    sample_type: str = ""      # category id, e.g. "clean_metal"
    sample_subtype: str = ""   # specific material, e.g. "Au(111)"


@dataclass
class ActionRecord:
    """Complete record of a single skill execution for experiment logging."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    sample_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    skill_name: str = ""
    skill_version: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    result: SkillResult | None = None
    state_before: HardwareState | None = None
    state_after: HardwareState | None = None
    nanonis_calls: list[NanonisCallRecord] = field(default_factory=list)
    context: str = ""  # LLM prompt or parent skill that triggered this
    duration_s: float = 0.0
    approval_source: str = "auto"  # "auto" / "llm" / "human"


@dataclass
class SensorReading:
    """A single reading from an environment sensor."""
    value: float
    unit: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "ok"  # "ok", "warning", "error", "unavailable"


# ─────────────────────────────────────────────────────────────────────
# v2 extensions (additions below this line; v1 schema above is frozen)
# ─────────────────────────────────────────────────────────────────────


class OperatingMode(Enum):
    """Global operating-mode / aggressiveness tier for autonomous experiments.

    Controls how the agent may treat the STM tip on the *autonomous* path
    (added 2026-07-07). Persisted as ``autonomy_mode`` in ui_settings.json and
    read live via a ``get_mode`` callable threaded into the safety / HITL /
    belief middlewares, so switching mode takes effect without rebuilding the
    agent graph.

      SAFE — no tip processing at all, enforced in three layers (2026-08-01):
             (1) the tip VERDICTS themselves are rewritten to "good" at their
             producers — see :mod:`mast.core.operating_mode` and
             ``docs/v2/design/safe_mode_tip_verdict_override.md`` — so the agent
             never receives evidence arguing for a repair, and a bad tip no
             longer halts the very experiment SAFE told it to focus on;
             (2) the belief block still states the tip is fine;
             (3) electrical pulses and tip shaping are hard-blocked as a
             backstop. Scan artifacts, raw measurements and every PHYSICAL
             SAFETY signal (current saturation/freeze/spike, E_STOP) stay
             truthful and still act — SAFE means "do not repair the tip", never
             "switch off the protections".
      SEMI — shallow, purely-mechanical tip shaping is allowed; a too-deep
             plunge is refused. Electrical pulses RUN and are announced in the
             diagnostics ledger (since 2026-08-08 — see
             :func:`mast.core.safety.mode_refusal`, the enforcing function;
             this docstring used to say "routed to HITL", which stopped being
             true when that change landed).
      AUTO — everything allowed. This is the current behaviour and the default,
             so an absent / unknown setting is treated as AUTO.

    The string values match the persisted ``autonomy_mode`` key exactly; do not
    rename them to e.g. ``semi_auto`` without updating the settings schema and
    the frontend in lockstep.
    """
    SAFE = "safe"
    SEMI = "semi"
    AUTO = "auto"

    @classmethod
    def coerce(cls, value: "OperatingMode | str | None") -> "OperatingMode":
        """Best-effort parse from a persisted/UI string; unknown → AUTO.

        Fail-open to AUTO (current behaviour) rather than raising, so a
        corrupt/stale settings value can never wedge the agent.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                return cls.AUTO
        return cls.AUTO
