# v2 note: get_assembler() removed — assembler.py is D-discarded.
# "expert" mode goes to ExperimentDesign agent; "simple/normal" to Orchestrator.
"""Sample-type workflow knowledge base for MAST.

Provides recommended experiment workflows for different STM sample types,
from clean metals to topological insulators. Each sample type defines
category-level generic phases and material-specific overrides.
"""

from mast.knowledge.lookups import (
    get_all_categories,
    get_category,
    get_completeness,
    get_constants,
    get_diagnostics,
    get_material,
    get_materials_in,
    get_merged_phases,
    list_material_candidates,
    match_material,
    material_candidates_hint,
    format_workflow_for_llm,
    format_conceptual_for_llm,
    format_literature_params,
)
from mast.knowledge.skill_guidance import (
    get_skill_guidance,
    format_skill_guidance_for_llm,
    get_measurement_template,
    format_measurement_template_for_llm,
)

__all__ = [
    # Lookups
    "get_all_categories",
    "get_category",
    "get_completeness",
    "get_constants",
    "get_diagnostics",
    "get_material",
    "get_materials_in",
    "get_merged_phases",
    "list_material_candidates",
    "match_material",
    "material_candidates_hint",
    "format_workflow_for_llm",
    "format_conceptual_for_llm",
    "format_literature_params",
    # Skill guidance
    "get_skill_guidance",
    "format_skill_guidance_for_llm",
    # Measurement templates
    "get_measurement_template",
    "format_measurement_template_for_llm",
]


# 公开版：基于 chunk 的检索（chunk_registry / retriever）不随仓发布，
# get_chunk_store / get_retriever 随之移除；调用方都有 ImportError 兜底。
