"""GET /api/skills/catalog — lightweight skill index for the Codex browser.

Phase 2: degrades to an empty index in standalone dev (no live SkillRegistry
wired). Phase 3 wires this to the live registry via builder_api (admin overrides
applied), keeping the same contract so the frontend never changes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from mast.api.schemas import SkillCatalogResponse, SkillIndexEntry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["skills"])


@router.get("/skills/catalog", response_model=SkillCatalogResponse)
def get_skill_catalog(request: Request) -> SkillCatalogResponse:
    ctx = request.app.state.ctx
    if ctx.skill_registry is None:
        # Standalone dev / not yet wired — empty but not broken.
        return SkillCatalogResponse(degraded=True)

    try:
        # builder_api owns the cached catalog build off the live registry. It
        # uses its own module-global registry, set by the live app at startup.
        from mast.webui.builder_api import get_catalog  # type: ignore[attr-defined]

        raw = get_catalog()  # {"index": [...], "cards": {...}}
        index = [
            SkillIndexEntry(
                name=row.get("name", ""),
                domain=row.get("domain", "其他"),
                source=row.get("source", "other"),
                # builder_api.build_catalog 写入的键名是 safety，值为小写
                # auto / confirm / dangerous。读取其他键会丢失真实分级。
                # 未知分级不能显示成 AUTO，应使用保守的展示回退值。
                safety_level=str(
                    row.get("safety") or row.get("safety_level") or "UNKNOWN"
                ).upper(),
                composition_level=row.get("composition_level"),
                summary=row.get("summary"),
            )
            for row in (raw.get("index") or [])
            if row.get("name")
        ]
        return SkillCatalogResponse(index=index, count=len(index), degraded=False)
    except Exception as exc:  # any wiring/shape mismatch → degrade, never 500
        logger.warning("skill catalog build failed: %s", exc)
        return SkillCatalogResponse(degraded=True)
