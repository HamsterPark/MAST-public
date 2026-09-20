"""Load stored declarative composite specs into a live SkillRegistry.

A spec saved by the builder/version store becomes a real, runnable skill: this
module binds each spec to a :class:`SpecComposite` subclass (via
``make_spec_skill``) and registers it, so the orchestrator / instrument-control
agent can invoke it exactly like a hand-written composite. Re-loading after an
edit registers the new version; the registry returns the latest by semver.
"""

from __future__ import annotations

import logging

from mast.skills.composite.interpreter import make_spec_skill
from mast.skills.composite.version_store import CompositeVersionStore

logger = logging.getLogger(__name__)


def register_spec(registry, spec) -> None:
    """Register a single CompositeSpec as a runnable skill.

    Refuses when *spec.name* collides with an already-registered NON-spec
    skill: a composite named e.g. "SetBias" would shadow the builtin in the
    registry, and a later delete/hot-unregister would then drop the BUILTIN
    (修复项 hardening, 2026-06-11). Rename the composite instead."""
    from mast.skills.composite.interpreter import SpecComposite
    if registry.has(spec.name):
        try:
            existing = registry.get(spec.name)
        except KeyError:  # pragma: no cover — has() raced
            existing = None
        if existing is not None and not issubclass(existing, SpecComposite):
            raise ValueError(
                f"composite name {spec.name!r} collides with a registered "
                f"non-composite skill "
                f"({existing.__module__}.{existing.__qualname__}) — rename it")
    # 传 registry:安全级要从子步骤技能继承(声明只能收紧不能放松)。
    registry.register(make_spec_skill(spec, registry))


def _is_builtin_composite_twin(registry, name: str) -> bool:
    """True iff *name* is already registered as a hand-written builtin composite
    (a ``CompositeSkillGraph`` that is NOT a ``SpecComposite``). Such a stored
    spec is a declarative TWIN kept for editing — it must not register over its
    builtin. (A spec colliding with a NON-composite builtin, e.g. ``SetBias``,
    is NOT a twin and still flows through register_spec's safety refusal.)"""
    if not registry.has(name):
        return False
    try:
        existing = registry.get(name)
    except KeyError:  # pragma: no cover — has() raced
        return False
    from mast.skills.composite._base import CompositeSkillGraph
    from mast.skills.composite.interpreter import SpecComposite
    return (isinstance(existing, type)
            and issubclass(existing, CompositeSkillGraph)
            and not issubclass(existing, SpecComposite))


def _collect_step_skills(nodes) -> set[str]:
    """Recursively collect every ``step`` node's skill name from a spec tree.

    Delegates to :func:`mast.skills.composite.spec.collect_step_skills` — the
    single source of truth for walking a spec tree. The local version here used
    to recurse into ``if`` and ``loop`` ONLY, so a step buried in a ``try`` body,
    an ``llm`` route or a ``human`` route was invisible to the
    missing-skill check below and could register anyway (then crash at runtime,
    after firing whatever hardware steps came before it — exactly what that
    check exists to prevent).
    """
    from mast.skills.composite.spec import collect_step_skills
    return collect_step_skills(nodes)


def _missing_step_skills(registry, spec) -> list[str]:
    """Referenced ``step`` skills that don't exist in the registry.

    A spec that names a non-existent skill used to register anyway and only fail
    at RUNTIME — after firing any earlier hardware step (e.g. a bias pulse) then
    crashing on the missing step. Refusing to register it
    keeps the broken skill out of the callable set entirely."""
    try:
        skills = _collect_step_skills(getattr(spec, "nodes", None))
    except Exception:  # pragma: no cover - defensive
        return []
    missing = []
    for sk in sorted(skills):
        try:
            if not registry.has(sk):
                missing.append(sk)
        except Exception:  # pragma: no cover
            missing.append(sk)
    return missing


def load_spec_skills(registry, store: CompositeVersionStore | None = None,
                     *, seed: bool = True) -> list[str]:
    """Register every stored spec into *registry*. Returns the names loaded.

    When ``seed`` is True, built-in templates are written into an empty store
    first so a fresh install has examples available immediately. Best-effort:
    a bad spec is logged and skipped, never fatal."""
    store = store or CompositeVersionStore()
    if seed:
        try:
            from mast.skills.composite.templates import seed_templates
            # Pass the registry so a stale seeded template that references
            # non-existent skills gets repaired rather than surviving forever
            # ().
            seed_templates(store, registry)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.debug("seed_templates failed: %s", exc)

    # Only enforce the unknown-skill refusal when the registry was ALREADY
    # populated with its builtins BEFORE this call (production: runtime.py runs
    # discover() first). An initially-EMPTY registry hasn't loaded builtins yet,
    # so we can't tell "unknown skill" from "not-loaded-yet" — and it grows as we
    # register specs, so checking per-spec would only skip the first (review
    # 2026-07-03). Snapshot the decision once, up front.
    try:
        _enforce_skill_refusal = bool(registry.list_skills())
    except Exception:  # pragma: no cover - defensive
        _enforce_skill_refusal = False

    loaded: list[str] = []
    rejected: list[dict] = []
    for summary in store.list_specs():
        name = summary.get("name")
        if not name:
            continue
        try:
            spec = store.load(name)
            problems = spec.validate()
            if problems:
                logger.warning("composite spec %s invalid, not registered: %s",
                               name, "; ".join(problems))
                rejected.append({"name": name, "reason": "invalid",
                                 "detail": "; ".join(problems)})
                continue
            # Refuse to register a spec that references skills which don't exist:
            # otherwise invoking it fires the earlier hardware steps then crashes
            # on the missing one.
            missing = _missing_step_skills(registry, spec) if _enforce_skill_refusal else []
            if missing:
                logger.warning(
                    "composite spec %s references unknown skill(s) %s — not "
                    "registered (would fire hardware then crash mid-run)",
                    name, ", ".join(missing))
                rejected.append({"name": name, "reason": "unknown_skills",
                                 "detail": ", ".join(missing),
                                 "missing_skills": list(missing)})
                continue
            # P5: a reconstructed declarative twin of a builtin composite (same
            # name) lives in the store for EDITING only — skip quietly so it
            # never tries to register over its hand-written builtin.
            if _is_builtin_composite_twin(registry, name):
                logger.debug("composite spec %s: declarative twin of a builtin "
                             "composite — kept in store, not registered", name)
                continue
            register_spec(registry, spec)
            loaded.append(name)
        except Exception as exc:
            logger.warning("failed to load composite spec %s: %s", name, exc)
            rejected.append({"name": name, "reason": "load_error",
                             "detail": f"{type(exc).__name__}: {exc}"})
    if loaded:
        logger.info("Loaded %d declarative composite(s): %s",
                    len(loaded), ", ".join(loaded))
    _record_rejections(rejected)
    return loaded


# ── refused specs, readable instead of log-only () ──
#
# A refused spec simply does not appear in the skill list, so the UI cannot
# distinguish "this composite is broken" from "this composite was never
# designed". Five specs — including both automatic tip-repair loops — were
# refused at every single startup for months and the only trace was a WARNING.
# Keep the last load's refusals in memory so a status endpoint / panel can show
# them. Plain dicts only; never hardware handles, never checkpointed.
_LAST_REJECTED: list[dict] = []


def _record_rejections(rejected: list[dict]) -> None:
    global _LAST_REJECTED
    _LAST_REJECTED = list(rejected)


def last_rejected_specs() -> list[dict]:
    """Specs refused by the most recent :func:`load_spec_skills` call.

    Each entry: ``{"name", "reason", "detail"[, "missing_skills"]}`` where
    ``reason`` is ``"invalid" | "unknown_skills" | "load_error"``.
    """
    return list(_LAST_REJECTED)


__all__ = ["register_spec", "load_spec_skills", "last_rejected_specs"]
