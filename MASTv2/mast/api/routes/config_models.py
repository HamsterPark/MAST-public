"""GET /api/config/models — the model-capability table.

Fully real data from ``mast.config`` (pure functions, no heavy deps), so this
endpoint is the Phase-2 end-to-end type-flow proof: Pydantic → OpenAPI → TS →
rendered in the Settings model picker.
"""

from __future__ import annotations

from fastapi import APIRouter

from mast.api.schemas import ModelInfo, ModelsResponse

router = APIRouter(tags=["config"])


@router.get("/config/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    from mast.config import (
        DEFAULT_MODEL_ALIAS,
        MODEL_PRESETS,
        THINKING_PRESETS,
        model_input_context,
        model_output_limit,
        model_thinking_mode,
    )

    models: list[ModelInfo] = []
    for alias, (model_id, provider, description, default_max_tokens) in MODEL_PRESETS.items():
        models.append(
            ModelInfo(
                alias=alias,
                model_id=model_id,
                provider=provider,
                description=description,
                default_max_tokens=default_max_tokens,
                thinking_mode=model_thinking_mode(model_id),  # type: ignore[arg-type]
                output_limit=model_output_limit(model_id),
                input_context=model_input_context(model_id),
                is_default=(alias == DEFAULT_MODEL_ALIAS),
            )
        )

    return ModelsResponse(
        default_alias=DEFAULT_MODEL_ALIAS,
        thinking_presets=dict(THINKING_PRESETS),
        models=models,
    )
