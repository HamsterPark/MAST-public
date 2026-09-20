"""Literature libraries + semantic search — Domain G of the typed API seam.

Mirrors the real 文献库 Gradio tab (``mast.webui.literature_panel``) but returns
typed JSON instead of HTML. The owner directive holds: **the big library is the
real library; user libraries are pointer-sets of ``work_id``s into it.**

Endpoints (read + write contract):
  GET    /literature/libraries                 list libraries (+ active id)
  POST   /literature/libraries                 create a custom/experiment library
  PATCH  /literature/libraries/{id}            rename a library
  DELETE /literature/libraries/{id}            delete (global is undeletable)
  POST   /literature/libraries/{id}/members    add work_id/DOI pointers
  POST   /literature/search                     semantic search (degrade-safe)
  GET    /literature/abstract/{work_id}        full abstract for one work

Graceful degradation (mandatory): the API must boot STANDALONE with no live
core. Every handler checks the context for a wired literature backend; if it is
absent — or any lazy import / core call raises — it returns a valid
empty/degraded payload (``degraded=True``) and NEVER 500s. The heavy
``mast.knowledge`` modules are lazy-imported INSIDE each handler, exactly like
``routes/skills.py`` lazy-imports ``builder_api``.

No business logic / safety checks live here — handlers only marshal to/from the
core. Real wiring to the live singleton + safety passthrough is the integrator's
integration step.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from mast.api.schemas_literature import (
    AbstractResponse,
    AddMembersRequest,
    CreateLibraryRequest,
    LibrariesResponse,
    LibrarySummary,
    MemberMutationResult,
    MutationResult,
    PatchLibraryRequest,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["literature"])


# ── wiring gate ───────────────────────────────────────────────────────
def _is_wired(ctx) -> bool:
    """True when a live literature backend has been shared into this process.

    Standalone dev leaves this unset → endpoints degrade (empty, not broken).
    The leader wires ``ctx.library_registry`` (and/or a truthy
    ``ctx.literature_wired``) at integration time; the contract stays identical.
    """
    try:
        return bool(
            getattr(ctx, "library_registry", None) is not None
            or getattr(ctx, "literature_wired", False)
        )
    except Exception:  # pragma: no cover - defensive
        return False


def _registry(ctx):
    """The injected live registry, or ``None`` (core falls back to its
    JSON-backed process singleton)."""
    return getattr(ctx, "library_registry", None)


# ── libraries: list ───────────────────────────────────────────────────
@router.get("/literature/libraries", response_model=LibrariesResponse)
def list_libraries(request: Request) -> LibrariesResponse:
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return LibrariesResponse(degraded=True)
    try:
        from mast.knowledge import libraries as lib_mod

        reg = _registry(ctx)
        rows = lib_mod.list_libraries(registry=reg)
        active = None
        try:
            active = lib_mod.get_active(registry=reg).get("library_id")
        except Exception:  # active is best-effort
            active = None
        # The id an add/search with no explicit library_id actually hits right now.
        # Resolved fresh on every request — it follows the active experiment with no
        # subscription anywhere, which is why activating a scope needs no hook here.
        effective, eff_source = None, ""
        try:
            from mast.knowledge.experiment_library import resolve_effective_library
            effective, eff_source = resolve_effective_library(registry=reg)
        except Exception as exc:  # best-effort; the list is still useful without it
            logger.info("effective library unresolved: %s", exc)
        names = _experiment_names(rows or [])
        libs = [
            LibrarySummary(
                library_id=str(r.get("library_id", "")),
                name=str(r.get("name", "") or ""),
                scope=str(r.get("scope", "custom") or "custom"),
                experiment_id=r.get("experiment_id"),
                experiment_name=names.get(str(r.get("experiment_id") or ""), ""),
                created_at=str(r.get("created_at", "") or ""),
                index_dir=r.get("index_dir"),
                member_count=int(r.get("member_count", 0) or 0),
                is_active=bool(r.get("is_active", False)),
            )
            for r in (rows or [])
            if r.get("library_id")
        ]
        return LibrariesResponse(
            libraries=libs, active_library_id=active,
            effective_library_id=effective, effective_source=eff_source,
            count=len(libs), degraded=False,
        )
    except Exception as exc:  # any wiring/shape mismatch → degrade, never 500
        logger.warning("list_libraries failed: %s", exc)
        return LibrariesResponse(degraded=True)


def _experiment_names(rows: list) -> dict[str, str]:
    """``{experiment_id: name}`` for the experiment libraries in *rows*.

    One storage instance for the whole list, and only when at least one row needs
    it — an experiment library labelled by its bare ``exp_1a2b3c4d`` id is useless
    in the UI. A missing experiment row yields no entry (the library keeps its own
    frozen name); the lookup never raises.
    """
    ids = {str(r.get("experiment_id") or "") for r in rows}
    ids.discard("")
    if not ids:
        return {}
    out: dict[str, str] = {}
    try:
        from mast.documents.paths import storage
        st = storage()
        for eid in ids:
            try:
                exp = st.get_experiment(eid) or {}
            except Exception:  # noqa: BLE001 — per-id, one bad row loses one label
                continue
            name = str(exp.get("name") or "").strip()
            if name:
                out[eid] = name
    except Exception as exc:  # noqa: BLE001 — no DB → no labels, still a valid list
        logger.info("experiment names unavailable: %s", exc)
    return out


# ── libraries: create ─────────────────────────────────────────────────
@router.post("/literature/libraries", response_model=MutationResult)
def create_library(request: Request, body: CreateLibraryRequest) -> MutationResult:
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return MutationResult(ok=False, degraded=True, message="literature backend not wired")
    name = (body.name or "").strip()
    if not name:
        return MutationResult(ok=False, message="library name must be non-empty")
    try:
        from mast.knowledge import libraries as lib_mod

        rec = lib_mod.create_library(
            name, scope=(body.scope or "custom"), registry=_registry(ctx)
        )
        summary = LibrarySummary(
            library_id=str(rec.get("library_id", "")),
            name=str(rec.get("name", "") or ""),
            scope=str(rec.get("scope", "custom") or "custom"),
            created_at=str(rec.get("created_at", "") or ""),
            index_dir=rec.get("index_dir"),
        )
        return MutationResult(ok=True, message=f"created {summary.name}", library=summary)
    except Exception as exc:
        # LibraryError (bad scope / dup) is a normal 'no' — return ok=False, not 500.
        logger.info("create_library rejected: %s", exc)
        return MutationResult(ok=False, message=str(exc))


# ── libraries: rename (PATCH) ─────────────────────────────────────────
@router.patch("/literature/libraries/{library_id}", response_model=MutationResult)
def patch_library(
    request: Request, library_id: str, body: PatchLibraryRequest
) -> MutationResult:
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return MutationResult(ok=False, degraded=True, message="literature backend not wired")
    new_name = (body.new_name or "").strip()
    if not new_name:
        return MutationResult(ok=False, message="new name must be non-empty")
    try:
        from mast.knowledge import libraries as lib_mod

        rec = lib_mod.rename_library(library_id, new_name, registry=_registry(ctx))
        summary = LibrarySummary(
            library_id=str(rec.get("library_id", "")),
            name=str(rec.get("name", "") or ""),
            scope=str(rec.get("scope", "custom") or "custom"),
            created_at=str(rec.get("created_at", "") or ""),
            index_dir=rec.get("index_dir"),
        )
        return MutationResult(ok=True, message=f"renamed to {summary.name}", library=summary)
    except Exception as exc:
        logger.info("patch_library rejected: %s", exc)
        return MutationResult(ok=False, message=str(exc))


# ── libraries: delete ─────────────────────────────────────────────────
@router.delete("/literature/libraries/{library_id}", response_model=MutationResult)
def delete_library(request: Request, library_id: str) -> MutationResult:
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return MutationResult(ok=False, degraded=True, message="literature backend not wired")
    try:
        from mast.knowledge import libraries as lib_mod

        ok = bool(lib_mod.delete_library(library_id, registry=_registry(ctx)))
        return MutationResult(ok=ok, message="deleted" if ok else "not deleted")
    except Exception as exc:
        # global library undeletable → LibraryError; surface as ok=False, never 500.
        logger.info("delete_library rejected: %s", exc)
        return MutationResult(ok=False, message=str(exc))


# ── libraries: add members ────────────────────────────────────────────
@router.post(
    "/literature/libraries/{library_id}/members", response_model=MemberMutationResult
)
def add_members(
    request: Request, library_id: str, body: AddMembersRequest
) -> MemberMutationResult:
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return MemberMutationResult(
            ok=False, degraded=True, message="literature backend not wired"
        )
    ids = [str(w).strip() for w in (body.work_ids or []) if str(w).strip()]
    if not ids:
        return MemberMutationResult(ok=False, message="provide at least one work_id / DOI")
    try:
        from mast.knowledge import libraries as lib_mod

        res = lib_mod.add_members(
            ids,
            library_id=library_id or None,
            reason=body.reason or "",
            added_by=(body.added_by if body.added_by in ("agent", "user") else "user"),
            registry=_registry(ctx),
        )
        if not isinstance(res, dict):
            res = {}
        return MemberMutationResult(
            ok=True,
            library_id=str(res.get("library_id", library_id) or library_id),
            added=[str(x) for x in (res.get("added") or [])],
            skipped=[str(x) for x in (res.get("skipped") or [])],
            rejected=[str(x) for x in (res.get("rejected") or [])],
            member_count=int(res.get("member_count", 0) or 0),
            at_cap=bool(res.get("at_cap", False)),
            message=f"added {len(res.get('added') or [])} pointer(s)",
        )
    except Exception as exc:
        logger.info("add_members rejected: %s", exc)
        return MemberMutationResult(ok=False, message=str(exc))


# ── search ────────────────────────────────────────────────────────────
@router.post("/literature/search", response_model=SearchResponse)
def search(request: Request, body: SearchRequest) -> SearchResponse:
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return SearchResponse(degraded=True)
    query = (body.query or "").strip()
    if not query:
        return SearchResponse(degraded=False)
    k = max(1, min(int(body.k or 8), 200))
    try:
        from mast.knowledge import literature_index

        # Big-library search; optionally filter to a library's member pointer-set
        # (mirrors literature_panel.search_big_library_html: over-fetch + filter).
        # Compare on the CANONICAL work_id BOTH sides: members are stored bare
        # (#125 canonical normalisation) while search results carry the URL form
        # from metadata.parquet — a raw `in` would miss every hit otherwise.
        member_set = None
        if body.library_id:
            try:
                from mast.knowledge import libraries as lib_mod
                from mast.knowledge.literature_index import canonical_work_id

                lib = lib_mod.get_library(body.library_id, registry=_registry(ctx))
                member_set = {
                    canonical_work_id(m.get("work_id") if isinstance(m, dict) else m)
                    for m in (lib.get("members") or [])
                }
            except Exception:
                member_set = None

        fetch_k = max(k * 8, k + 50) if member_set is not None else k
        rows = literature_index.search(query, k=fetch_k, material=(body.material or None))
        rows = rows or []
        if member_set is not None:
            from mast.knowledge.literature_index import canonical_work_id
            rows = [r for r in rows if canonical_work_id(r.get("work_id")) in member_set]
        rows = rows[:k]

        # degraded when the semantic index fell back to keyword matching
        degraded = bool(rows and rows[0].get("degraded"))
        hits = [
            SearchHit(
                work_id=str(r.get("work_id", "") or ""),
                title=str(r.get("title", "") or ""),
                year=int(r.get("year", 0) or 0),
                journal=str(r.get("journal", "") or ""),
                doi=str(r.get("doi", "") or ""),
                cited=int(r.get("cited", 0) or 0),
                score=float(r.get("score", 0.0) or 0.0),
                source=str(r.get("source", "") or ""),
                abstract_excerpt=str(r.get("abstract_excerpt", "") or ""),
                retrieval=str(r.get("retrieval", "semantic") or "semantic"),
            )
            for r in rows
        ]
        return SearchResponse(hits=hits, count=len(hits), degraded=degraded)
    except Exception as exc:
        logger.warning("literature search failed: %s", exc)
        return SearchResponse(degraded=True)


# ── abstract ──────────────────────────────────────────────────────────
@router.get("/literature/abstract/{work_id}", response_model=AbstractResponse)
def fetch_abstract(request: Request, work_id: str) -> AbstractResponse:
    ctx = request.app.state.ctx
    wid = (work_id or "").strip()
    if not _is_wired(ctx):
        return AbstractResponse(work_id=wid, found=False, degraded=True)
    if not wid:
        return AbstractResponse(work_id="", found=False)
    try:
        from mast.knowledge import literature_index

        rec = literature_index.fetch_abstract(wid)
        if not isinstance(rec, dict):
            rec = {}

        def _i(key: str) -> int:
            try:
                return int(rec.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0

        return AbstractResponse(
            work_id=str(rec.get("work_id", wid) or wid),
            found=bool(rec.get("found", False)),
            title=str(rec.get("title", "") or ""),
            abstract=str(rec.get("abstract", "") or ""),
            user_abstract=str(rec.get("user_abstract", "") or ""),
            abstract_provenance=str(rec.get("abstract_provenance", "") or ""),
            authors=str(rec.get("authors", "") or ""),
            first_author=str(rec.get("first_author", "") or ""),
            year=str(rec.get("year", "") or ""),
            journal=str(rec.get("journal", "") or ""),
            doi=str(rec.get("doi", "") or ""),
            source=str(rec.get("source", "") or ""),
            cited_by_count=_i("cited_by_count"),
            note=str(rec.get("note", "") or ""),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("fetch_abstract failed: %s", exc)
        return AbstractResponse(work_id=wid, found=False, degraded=True)
