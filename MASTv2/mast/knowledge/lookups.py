"""Query, matching, and formatting functions for sample-type workflows."""

from __future__ import annotations

import copy
import importlib
import re
from typing import Any

# ── Module registry ────────────────────────────────────────────────────
# Maps category id -> (module_name, loaded_module | None)
# 公开版：样品类型知识模块不随仓发布（见 docs/OPEN_SOURCE_NOTES.md）。
# 表留空 —— 所有查询返回「未知材料」，调用方本来就要处理这一种情况。
_CATEGORY_MODULES: dict[str, str] = {}

_module_cache: dict[str, Any | None] = {}

# Cached material name -> (type_id, mat_name) index, built on first use
_material_index: dict[str, tuple[str, str]] | None = None


def _load_module(type_id: str) -> Any | None:
    """Lazily import and cache a sample-type module."""
    if type_id in _module_cache:
        return _module_cache[type_id]
    mod_name = _CATEGORY_MODULES.get(type_id)
    if mod_name is None:
        _module_cache[type_id] = None
        return None
    mod = importlib.import_module(mod_name)
    _module_cache[type_id] = mod
    return mod


def _get_material_index() -> dict[str, tuple[str, str]]:
    """Build and cache a lowercase material name -> (type_id, name) index."""
    global _material_index
    if _material_index is not None:
        return _material_index
    index: dict[str, tuple[str, str]] = {}
    for type_id in _CATEGORY_MODULES:
        mod = _load_module(type_id)
        if mod and hasattr(mod, "MATERIALS"):
            for mat_name in mod.MATERIALS:
                index[mat_name.lower()] = (type_id, mat_name)
    _material_index = index
    return index


# ── Public API ─────────────────────────────────────────────────────────

def get_all_categories() -> list[dict]:
    """Return list of all CATEGORY dicts (lazy-loaded)."""
    cats: list[dict] = []
    for type_id in _CATEGORY_MODULES:
        mod = _load_module(type_id)
        if mod and hasattr(mod, "CATEGORY"):
            cats.append(mod.CATEGORY)
    return cats


def _apply_category_override(cat: dict, type_id: str) -> dict:
    """Merge admin override into a category dict (if available)."""
    try:
        from mast.admin.override_store import ConfigOverrideRegistry, deep_merge
        ovr = ConfigOverrideRegistry.get().get_knowledge_override(type_id)
        if ovr:
            return deep_merge(cat, ovr)
    except (ImportError, Exception):
        pass
    return cat


def get_category(type_id: str) -> dict | None:
    """Return CATEGORY dict for a given type_id, or None."""
    mod = _load_module(type_id)
    if mod and hasattr(mod, "CATEGORY"):
        return _apply_category_override(mod.CATEGORY, type_id)
    return None


def get_material(material_name: str) -> dict | None:
    """Return material dict by exact name (searches all categories)."""
    for type_id in _CATEGORY_MODULES:
        mod = _load_module(type_id)
        if mod and hasattr(mod, "MATERIALS"):
            if material_name in mod.MATERIALS:
                mat = mod.MATERIALS[material_name]
                # Apply material-level overrides if available
                try:
                    from mast.admin.override_store import ConfigOverrideRegistry, deep_merge
                    ovr = ConfigOverrideRegistry.get().get_knowledge_override(type_id)
                    if ovr and "materials" in ovr and material_name in ovr["materials"]:
                        return deep_merge(mat, ovr["materials"][material_name])
                except (ImportError, Exception):
                    pass
                return mat
    return None


def get_materials_in(type_id: str) -> dict[str, dict]:
    """Return all MATERIALS in a given category."""
    mod = _load_module(type_id)
    if mod and hasattr(mod, "MATERIALS"):
        return mod.MATERIALS
    return {}


def get_completeness(type_id: str) -> str:
    """Return 'full', 'stub', or '' for a category."""
    cat = get_category(type_id)
    return cat.get("completeness", "") if cat else ""


_ALIASES: dict[str, str] = {
    "金": "Au(111)", "铜": "Cu(111)", "银": "Ag(111)", "铂": "Pt(111)",
    "硅": "Si(111)-7x7", "锗": "Ge(100)", "砷化镓": "GaAs(110)",
    "石墨烯": "2d_material", "graphene": "2d_material",
    "金属": "clean_metal", "半导体": "semiconductor",
    "分子": "molecular_adsorbate", "超导": "superconductor",
    "拓扑": "topological", "磁性": "magnetic_spm",
    "薄膜": "thin_film", "二维": "2d_material",
    "氧化物": "oxide_surface", "表面合成": "on_surface_synthesis",
    "herringbone": "Au(111)", "shockley": "Au(111)",
    "kondo": "molecular_adsorbate",
    "vortex": "superconductor", "涡旋": "superconductor",
    "dirac": "topological", "狄拉克": "topological",
    "sp-stm": "magnetic_spm", "spin": "magnetic_spm",
    "skyrmion": "magnetic_spm", "spin spiral": "magnetic_spm",
    "磁畴": "magnetic_spm", "自旋螺旋": "magnetic_spm",
    "mn/w": "Mn/W(110)", "fe/ir": "Fe/Ir(111)",
    "pdfe": "PdFe/Ir(111)",
}


def match_material(query: str, _depth: int = 0) -> tuple[str, str] | None:
    """Fuzzy-match a query string to (type_id, material_name).

    Tries exact name match first, then substring/alias matching.
    Returns None if no match found.

    _depth guards the alias-resolution recursion (Pass 3): a self- or
    mutually-referential alias would otherwise recurse forever.
    """
    query = query.strip()
    if not query or _depth > 3:
        return None

    q_lower = query.lower()

    # Pass 1: exact material name match (O(1) via cached index)
    index = _get_material_index()
    if q_lower in index:
        return index[q_lower]

    # Pass 2: category id match
    if q_lower in _CATEGORY_MODULES:
        return (q_lower, "")

    # Pass 3: alias table for common Chinese/English terms
    for alias, target in _ALIASES.items():
        if alias in q_lower:
            if target in _CATEGORY_MODULES:
                return (target, "")
            result = match_material(target, _depth + 1)
            if result:
                return result

    # Pass 4: substring in material name or description
    for mat_lower, (type_id, mat_name) in index.items():
        if q_lower in mat_lower:
            return (type_id, mat_name)
    for type_id in _CATEGORY_MODULES:
        mod = _load_module(type_id)
        if not mod or not hasattr(mod, "MATERIALS"):
            continue
        for mat_name, mat_data in mod.MATERIALS.items():
            desc = mat_data.get("description", "").lower()
            if q_lower in desc:
                return (type_id, mat_name)

    # Pass 5: match against category name/name_en/description
    for type_id in _CATEGORY_MODULES:
        cat = get_category(type_id)
        if not cat:
            continue
        if (q_lower in cat.get("name", "").lower()
                or q_lower in cat.get("name_en", "").lower()
                or q_lower in cat.get("description", "").lower()):
            return (type_id, "")

    return None


def list_material_candidates(limit: "int | None" = 40) -> list[str]:
    """The controlled vocabulary a material / sample_type lookup matches against:
    the top-level category ids first (the coarse types), then every known
    material name. Sorted, de-duplicated, optionally capped.

    This is the SINGLE source for "what are the valid choices?", so every tool
    that rejects (or accepts-and-normalises) a material can offer the SAME
    candidate list — the fix for the three inconsistent enum paths (start_sample
    rejected without candidates; one tool accepted anything; a third listed
    candidates). Now they all agree, and a miss always returns candidates.
    """
    index = _get_material_index()  # lower -> (type_id, name)
    names = sorted({name for (_t, name) in index.values() if name})
    out: list[str] = list(_CATEGORY_MODULES.keys())  # categories first
    for n in names:
        if n not in out:
            out.append(n)
    if limit is not None and len(out) > limit:
        return out[:limit]
    return out


def material_candidates_hint(limit: int = 30) -> str:
    """A compact one-line hint listing the material/sample-type controlled
    vocabulary, for embedding in a tool's miss response."""
    cands = list_material_candidates(limit=limit)
    if not cands:
        return ""
    return "可选的受控词表(材料名或样品类别): " + "、".join(cands)


def get_merged_phases(type_id: str, material_name: str = "") -> list[dict]:
    """Return fully-merged phase list for a specific material.

    Starts from CATEGORY.phases, then applies MATERIALS[name].phases_override.
    """
    cat = get_category(type_id)
    if not cat:
        return []

    # Deep-copy base phases
    phases = copy.deepcopy(cat.get("phases", []))

    if not material_name:
        return phases

    mat = get_material(material_name)
    if not mat:
        # Material not found — try within this category
        mats = get_materials_in(type_id)
        mat = mats.get(material_name)
    if not mat:
        return phases

    overrides = mat.get("phases_override", {})
    if not overrides:
        return phases

    # Apply overrides by phase id
    for i, phase in enumerate(phases):
        pid = phase.get("id", "")
        if pid in overrides:
            ovr = overrides[pid]
            # Merge: override replaces steps, notes, success_criteria, on_fail
            for key in ("steps", "notes", "success_criteria", "on_fail"):
                if key in ovr:
                    phases[i][key] = copy.deepcopy(ovr[key])

    return phases


def get_constants(type_id: str, material_name: str = "") -> dict:
    """Return physical constants for a category or specific material.

    If *material_name* is given, returns that material's constants dict.
    If only *type_id*, returns the full CONSTANTS dict for the category.
    Returns {} if the module has no CONSTANTS.
    """
    mod = _load_module(type_id)
    if not mod or not hasattr(mod, "CONSTANTS"):
        return {}
    constants = mod.CONSTANTS
    if material_name:
        return constants.get(material_name, {})
    return constants


def get_diagnostics(type_id: str, material_name: str) -> list[dict]:
    """Return diagnostics list for a specific material.

    Each item is ``{"symptom": ..., "cause": ..., "remedy": ...}``.
    Returns [] if not found.
    """
    mod = _load_module(type_id)
    if not mod or not hasattr(mod, "MATERIALS"):
        return []
    mat = mod.MATERIALS.get(material_name)
    if not mat:
        return []
    return mat.get("diagnostics", [])


def format_workflow_for_llm(
    type_id: str, material_name: str = ""
) -> str:
    """Format a workflow as a compact text summary for LLM system prompt.

    Output is ~200-300 tokens, suitable for injection into planner context.
    """
    cat = get_category(type_id)
    if not cat:
        return ""

    mat = get_material(material_name) if material_name else None
    title = material_name if material_name else cat.get("name", type_id)
    cat_label = cat.get("name_en", cat.get("name", type_id))

    lines: list[str] = [f"## Recommended Workflow: {title} ({cat_label})"]

    # Tip requirements
    tip = cat.get("tip_requirements", "")
    if tip:
        lines.append(f"Tip: {tip}")

    # Merged phases
    phases = get_merged_phases(type_id, material_name)
    lines.append("Phases:")
    for i, phase in enumerate(phases, 1):
        pname = phase.get("name", phase.get("id", ""))
        steps_str = _format_steps_compact(phase.get("steps", []))
        criteria = phase.get("success_criteria", "")
        line = f"  {i}. {pname}"
        if steps_str:
            line += f" — {steps_str}"
        lines.append(line)
        if criteria:
            lines.append(f"     Success: {criteria}")

    # Material-specific features
    if mat:
        features = mat.get("key_features", [])
        if features:
            lines.append("Key features:")
            for f in features:
                lines.append(f"  - {f}")
        prep = mat.get("prep", {})
        if prep:
            lines.append("Preparation:")
            for k, v in prep.items():
                lines.append(f"  - {k}: {v}")

    # Quality criteria (material-specific success indicators)
    if mat:
        qc = mat.get("quality_criteria", {})
        if qc:
            lines.append("Quality criteria:")
            for label, criterion in qc.items():
                lines.append(f"  - {label}: {_format_qc(criterion)}")

    # Common issues
    issues = cat.get("common_issues", [])
    if issues:
        lines.append("Common issues:")
        for issue in issues:
            lines.append(f"  - {issue}")

    return "\n".join(lines)


def format_conceptual_for_llm(
    type_id: str, material_name: str = ""
) -> str:
    """Format a workflow as conceptual background for LLM system prompt.

    Unlike format_workflow_for_llm(), this version:
    - Shows physical phenomena, quality markers, and diagnostics
    - Phase list only shows conceptual purpose (name + description), NO parameters
    - Ends with explicit disclaimer that parameters are literature reference values
    """
    cat = get_category(type_id)
    if not cat:
        return ""

    mat = get_material(material_name) if material_name else None
    title = material_name if material_name else cat.get("name", type_id)
    cat_label = cat.get("name_en", cat.get("name", type_id))

    lines: list[str] = [f"## {title} — 物理背景 ({cat_label})"]

    # Tip requirements
    tip = cat.get("tip_requirements", "")
    if tip:
        lines.append(f"针尖要求: {tip}")

    # Material-specific key features as "物理现象"
    if mat:
        features = mat.get("key_features", [])
        if features:
            lines.append("")
            lines.append("物理现象:")
            for f in features:
                lines.append(f"  - {f}")

    # Quality criteria
    if mat:
        qc = mat.get("quality_criteria", {})
        if qc:
            lines.append("")
            lines.append("质量标志:")
            for label, criterion in qc.items():
                lines.append(f"  - {label}: {_format_qc(criterion)}")

    # Common issues (diagnostics are instrument-independent)
    issues = cat.get("common_issues", [])
    if issues:
        lines.append("")
        lines.append("常见问题诊断:")
        for issue in issues:
            lines.append(f"  - {issue}")

    # Phases: concept only — name + description, NO steps/parameters
    phases = get_merged_phases(type_id, material_name)
    if phases:
        lines.append("")
        lines.append("实验阶段 (概念):")
        for i, phase in enumerate(phases, 1):
            pname = phase.get("name", phase.get("id", ""))
            desc = phase.get("description", "")
            line = f"  {i}. {pname}"
            if desc:
                line += f" — {desc}"
            lines.append(line)

    # Material prep info
    if mat and mat.get("prep"):
        lines.append("")
        lines.append("制备条件:")
        for k, v in mat["prep"].items():
            lines.append(f"  - {k}: {v}")

    # Explicit disclaimer
    lines.append("")
    lines.append("⚠ 具体参数 (偏压、电流、扫描范围等) 为文献参考值，请向用户确认仪器实际值。")

    return "\n".join(lines)


def format_literature_params(
    type_id: str, material_name: str = "", phase_id: str = ""
) -> str:
    """Format literature reference parameters for a specific phase.

    Returns step-by-step parameters with explicit literature-reference warning.
    Used by the get_literature_parameters meta-tool.
    """
    cat = get_category(type_id)
    if not cat:
        return ""

    phases = get_merged_phases(type_id, material_name)
    if not phases:
        return ""

    # If phase_id given, filter to that phase; otherwise show all
    if phase_id:
        matched = [p for p in phases if p.get("id", "") == phase_id]
        if not matched:
            # Fuzzy: try matching name
            matched = [
                p for p in phases
                if phase_id.lower() in p.get("name", "").lower()
                or phase_id.lower() in p.get("name_en", "").lower()
                or phase_id.lower() in p.get("id", "").lower()
            ]
        if not matched:
            return f"未找到阶段 '{phase_id}'。可用阶段: {', '.join(p.get('id', '') for p in phases)}"
        phases = matched

    title = material_name if material_name else cat.get("name", type_id)
    lines: list[str] = []

    for phase in phases:
        pname = phase.get("name", phase.get("id", ""))
        lines.append(f"## 文献参考参数: {title} / {pname}")
        lines.append("⚠ 以下为文献参考值，请向用户确认后再使用。")
        lines.append("")

        steps = phase.get("steps", [])
        if not steps:
            lines.append("  (此阶段无具体步骤)")
            continue

        lines.append("步骤:")
        for j, step in enumerate(steps, 1):
            skill = step.get("skill")
            if skill:
                params = step.get("params", {})
                if params:
                    param_str = ", ".join(
                        f"{k}={_format_value(v)}" for k, v in params.items()
                    )
                    step_text = f"{skill}({param_str})"
                else:
                    step_text = f"{skill}()"
            else:
                action = step.get("action", "manual step")
                step_text = f"[{action}]"

            notes = step.get("notes", "")
            note_part = f" — {notes}" if notes else ""
            lines.append(f"  {j}. {step_text}{note_part}")

        # Phase notes
        phase_notes = phase.get("notes", "")
        if phase_notes:
            lines.append(f"  Note: {phase_notes}")

        # Success criteria
        criteria = phase.get("success_criteria", "")
        if criteria:
            lines.append(f"  Success: {criteria}")

        lines.append("")

    return "\n".join(lines)


def _format_qc(criterion: Any) -> str:
    """Format a quality_criteria value (str or dict) for display."""
    if isinstance(criterion, str):
        return criterion
    if isinstance(criterion, dict):
        parts: list[str] = []
        if "check" in criterion:
            parts.append(criterion["check"])
        if "threshold" in criterion:
            parts.append(f"[{criterion['threshold']}]")
        if "method" in criterion:
            parts.append(f"({criterion['method']})")
        return " ".join(parts) if parts else str(criterion)
    return str(criterion)


def _format_steps_compact(steps: list[dict]) -> str:
    """Format steps as a compact string: skill1(params) -> skill2(params)."""
    parts: list[str] = []
    for step in steps:
        skill = step.get("skill")
        if skill:
            params = step.get("params", {})
            if params:
                param_str = ", ".join(
                    f"{_format_value(v)}" for v in params.values()
                )
                parts.append(f"{skill}({param_str})")
            else:
                parts.append(skill)
        else:
            action = step.get("action", "")
            if action:
                parts.append(f"[{action}]")
    return " -> ".join(parts) if parts else ""


def _format_value(v: Any) -> str:
    """Format a parameter value compactly."""
    if isinstance(v, float):
        if abs(v) < 1e-6:
            return f"{v:.0e}"
        if abs(v) < 0.01:
            return f"{v:.1e}"
        return f"{v:g}"
    if isinstance(v, bool):
        return str(v)
    return str(v)
