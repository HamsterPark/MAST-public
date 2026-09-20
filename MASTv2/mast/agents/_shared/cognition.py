"""CognitionContext — one-line assembly of MAST's "cognitive services".

This is the *wiring* layer that bundles the already-implemented memory backend
into a single, easy-to-attach service:

  * :class:`~mast.memory.store.MemoryStore`     — persistent agent memory
  * :class:`~mast.memory.sharding.PhaseManager` — conversation phase summaries
  * :class:`~mast.memory.dreaming.DreamingService` — async background consolidation

plus the agent-facing memory *tools*
(:func:`~mast.agents._shared.memory_tools.make_memory_tools`).

The orchestrator / GUI builds **one** ``CognitionContext`` per experiment DB and
then attaches tools to any agent in a single line::

    cog = CognitionContext.from_storage(storage, author="planner")
    tools = cog.tools(experiment_id=exp_id)          # 4 memory tools
    cog.start_dreaming(should_dream=lambda: not busy) # idle-gated background pass

Everything works **offline / without an LLM key** — both the phase summariser and
the dream consolidator default to the dependency-free rule-based implementations
shipped in ``mast.memory``. The GUI may later inject a real LLM summariser /
consolidator via :meth:`set_summarizer` / :meth:`set_consolidator`.

Lives under ``agents/_shared`` (a shared module) so it can be attached to several
agents without crossing the agent boundary. It imports ONLY ``mast.memory.*`` and
``mast.agents._shared.memory_tools`` — never a concrete agent.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable

from mast.agents._shared.memory_tools import make_memory_tools
from mast.memory.dreaming import DreamingService
from mast.memory.sharding import PhaseManager
from mast.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def _namespace_for(experiment_id: str | None) -> str:
    """Namespace rule shared by provider/tools: per-experiment, else global."""
    return f"experiment:{experiment_id}" if experiment_id else "global"


class CognitionContext:
    """Assembled cognitive services for ONE experiment database.

    Build once per DB; cheap to hold. All three backends share the same SQLite
    file (the experiment record), so a project's whole cognitive context lives in
    — and exports with — its record.
    """

    def __init__(self, db_path: str | Path, *, author: str = "agent"):
        self._db_path = Path(db_path)
        self._author = author or "agent"

        # The three cognitive backends, all bound to the same DB file.
        self.store = MemoryStore(self._db_path)
        self.phases = PhaseManager(self._db_path, memory_store=self.store)
        self.dreaming = DreamingService(self._db_path, self.store)

        # Injection points (default to the rule-based ones already inside the
        # backends; only overwritten if the app injects a real LLM later).
        self._summarizer: Callable[[list[dict]], str] | None = None
        self._consolidator: Callable[[dict], list] | None = None

        # start/stop idempotency guard for the background dreaming thread.
        self._dream_lock = threading.Lock()
        self._dreaming_on = False

        # Semantic memory vector index (lazy — built on first remember/recall so
        # GUI startup never blocks on an embedder probe / model load). None until
        # resolved; stays None when the embedder tier is "substring".
        self._vec = None
        self._vec_init = False
        self._vec_lock = threading.Lock()
        self._needs_backfill = False

    # ── factory ───────────────────────────────────────────────────────
    @classmethod
    def from_storage(cls, storage, **kw) -> "CognitionContext":
        """Build from an ``ExperimentStorage`` (shares its DB file).

        Reads the storage's private ``_db_path`` — the same attribute
        :meth:`MemoryStore.from_storage` relies on — so cognition lives in the
        very same database as the experiment record.
        """
        db_path = getattr(storage, "_db_path", None)
        if db_path is None:
            raise ValueError("storage has no _db_path; cannot build CognitionContext")
        return cls(db_path, **kw)

    # ── memory provider / tools ───────────────────────────────────────
    def memory_provider(self, *, experiment_id: str | None = None,
                        author: str | None = None) -> Callable[[], dict]:
        """Return a ``provider()`` callable for :func:`make_memory_tools`.

        Namespace rule: ``experiment:{experiment_id}`` when an experiment id is
        given, else ``"global"``. The callable is re-evaluated on every tool call
        so the binding (store/namespace/author) is always current.
        """
        ns = _namespace_for(experiment_id)
        who = author or self._author

        def _provider() -> dict:
            return {
                "store": self.store,
                "namespace": ns,
                "experiment_id": experiment_id,
                "author": who,
                # `writer` indexes into the semantic store on write; `recaller`
                # does namespace-scoped semantic recall. Both best-effort.
                "writer": self.remember,
                "recaller": (lambda q: self.recall(q, experiment_id=experiment_id)),
            }

        return _provider

    def tools(self, *, experiment_id: str | None = None,
              author: str | None = None) -> list:
        """The 4 agent-facing memory tools bound to this experiment/namespace."""
        return make_memory_tools(
            self.memory_provider(experiment_id=experiment_id, author=author))

    # ── LLM injection points (default rule-based) ─────────────────────
    def set_summarizer(self, fn: Callable[[list[dict]], str] | None) -> None:
        """Inject an LLM phase summariser into the :class:`PhaseManager`.

        ``fn(messages: list[{role, content}]) -> str``. Pass ``None`` to revert
        to the backend's rule-based default. The PhaseManager itself already
        guards against a misbehaving summariser (falls back to rule-based on
        exception), so this is safe to wire to a flaky LLM.
        """
        self._summarizer = fn
        if fn is not None:
            self.phases._summarizer = fn
        else:
            # restore the backend default rather than leave a stale LLM hook
            from mast.memory.sharding import rule_based_summary
            self.phases._summarizer = rule_based_summary

    def set_consolidator(self, fn: Callable[[dict], list] | None) -> None:
        """Inject an LLM dream consolidator into the :class:`DreamingService`.

        ``fn(context: dict) -> list[{path, title, content, kind}]``. Pass
        ``None`` to revert to the rule-based default. ``dream_once`` already
        catches a misbehaving consolidator, so this is safe to wire to an LLM.
        """
        self._consolidator = fn
        if fn is not None:
            self.dreaming._consolidate = fn
        else:
            from mast.memory.dreaming import rule_based_consolidate
            self.dreaming._consolidate = rule_based_consolidate

    # ── dreaming lifecycle (idle-gated, idempotent) ───────────────────
    def start_dreaming(self, should_dream: Callable[[], bool] | None = None) -> None:
        """Start the background consolidation pass (idempotent).

        ``should_dream`` is an idle predicate forwarded to the service so the
        dream cycle only fires when the system is quiet (e.g. ``lambda: not
        orchestrator.busy``). Calling this repeatedly is a no-op while running;
        the predicate, if given, is (re)applied each call.
        """
        with self._dream_lock:
            if should_dream is not None:
                self.dreaming.set_should_dream(should_dream)
            if self._dreaming_on:
                return
            self.dreaming.start()
            self._dreaming_on = True

    def stop_dreaming(self) -> None:
        """Stop the background pass and join its thread cleanly (idempotent)."""
        with self._dream_lock:
            if not self._dreaming_on:
                # still call through so a thread started out-of-band is cleaned up
                self.dreaming.stop()
                return
            self.dreaming.stop()
            self._dreaming_on = False

    def dream_once(self) -> list[dict]:
        """Run one consolidation cycle synchronously (convenience pass-through)."""
        return self.dreaming.dream_once()

    # ── context-assembly helpers (thin pass-throughs) ─────────────────
    def memory_index(self, *, experiment_id: str | None = None,
                     max_lines: int = 60) -> str:
        """MEMORY.md-style index for the experiment's namespace (session start)."""
        return self.store.index_markdown(
            _namespace_for(experiment_id), max_lines=max_lines)

    # ── semantic memory index (lazy, best-effort) ─────────────────────
    def _ensure_vec(self):
        """Resolve the embedder + open the vector store once. Returns it or None.

        Picks the best embedder tier (DashScope→local→substring); for the
        substring tier returns None (recall falls back to ``MemoryStore.search``).
        Persists ``(backend, dim)`` in a manifest; if a prior run used a different
        backend/dim the existing vectors are incomparable, so they are wiped and
        re-indexed lazily.
        """
        if self._vec_init:
            return self._vec
        with self._vec_lock:
            if self._vec_init:
                return self._vec
            self._vec_init = True
            try:
                from mast.logging.v2.vector_search import open_search
                from mast.memory.embeddings import make_embedder
                embedder, dim, backend = make_embedder()
                if embedder is None or dim <= 0:
                    logger.info("cognition: semantic recall disabled (backend=%s)", backend)
                    return None
                vec_path = self._db_path.parent / "memory_vectors.db"
                manifest = self._db_path.parent / "memory_vectors_manifest.json"
                prev = None
                if manifest.exists():
                    try:
                        prev = json.loads(manifest.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        prev = None
                if prev and (prev.get("backend") != backend or prev.get("dim") != dim):
                    for ext in ("", "-wal", "-shm"):
                        p = Path(str(vec_path) + ext)
                        if p.exists():
                            try:
                                p.unlink()
                            except OSError:
                                pass
                    prev = None
                self._vec = open_search(str(vec_path), embedder, dim=dim)
                if prev is None:
                    manifest.write_text(json.dumps({"backend": backend, "dim": dim}),
                                        encoding="utf-8")
                    self._needs_backfill = True
                logger.info("cognition: semantic memory index ready (backend=%s dim=%d)",
                            backend, dim)
                return self._vec
            except Exception as exc:  # noqa: BLE001 — recall degrades to substring
                logger.warning("cognition: vector index unavailable (%s); "
                               "recall falls back to substring", exc)
                self._vec = None
                return None

    def _index_one(self, namespace: str, path: str, title: str, content: str) -> None:
        vec = self._ensure_vec()
        if vec is None:
            return
        eid = f"{namespace}/{path}"
        text = (f"{title}\n{content}" if title else content or "").strip()
        if not text:
            return
        try:
            vec.delete_entity(entity_kind="memory", entity_id=eid)
            vec.index(entity_kind="memory", entity_id=eid, text=text)
        except Exception as exc:  # noqa: BLE001 — indexing never breaks a write
            logger.debug("memory index failed for %s: %s", eid, exc)

    def _maybe_backfill(self) -> None:
        """One-time lazy index of all existing memory (dreams/phases/notes)."""
        if not self._needs_backfill:
            return
        self._needs_backfill = False
        try:
            for ns in self.store.namespaces():
                for row in self.store.list(ns, limit=2000):
                    self._index_one(ns, row["path"], row.get("title", ""),
                                    row.get("content", ""))
        except Exception as exc:  # noqa: BLE001
            logger.debug("memory backfill failed: %s", exc)

    def remember(self, namespace: str, path: str, content: str, *, title: str = "",
                 kind: str = "note", tags: list | None = None,
                 experiment_id: str | None = None, author: str | None = None,
                 pinned: bool = False) -> dict:
        """Write a memory AND index it for semantic recall (best-effort index)."""
        r = self.store.write(namespace, path, content, title=title, kind=kind,
                             tags=tags or [], experiment_id=experiment_id,
                             author=author or self._author, pinned=pinned)
        self._index_one(r["namespace"], r["path"], title, content)
        return r

    def recall(self, query: str, *, experiment_id: str | None = None,
               k: int = 5) -> list[dict]:
        """Semantic recall over memory, scoped to the experiment ns + 'global'.

        Uses the vector index when available (knn is global over the whole file,
        so results are post-filtered by ``entity_id`` namespace prefix to prevent
        cross-experiment leakage); falls back to substring search otherwise.
        Returns full MemoryStore rows.
        """
        ns = _namespace_for(experiment_id)
        namespaces = {ns, "global"}
        vec = self._ensure_vec()
        if vec is not None:
            try:
                self._maybe_backfill()
                out, seen = [], set()
                for h in vec.knn(query, k=max(k * 3, k)):
                    eid = h.get("entity_id", "")
                    if "/" not in eid:
                        continue
                    hns, _, hpath = eid.partition("/")
                    if hns not in namespaces or eid in seen:
                        continue
                    seen.add(eid)
                    row = self.store.read(hns, hpath)
                    if row:
                        out.append(row)
                    if len(out) >= k:
                        break
                if out:
                    return out
            except Exception as exc:  # noqa: BLE001
                logger.debug("semantic recall failed (%s); substring fallback", exc)
        # substring fallback (namespace-scoped, deduped)
        uniq: dict = {}
        for n in (ns, "global"):
            for r in self.store.search(query, namespace=n, limit=k):
                uniq[(r["namespace"], r["path"])] = r
        return list(uniq.values())[:k]


def from_storage(storage, **kw) -> CognitionContext:
    """Module-level factory: build a :class:`CognitionContext` from a storage."""
    return CognitionContext.from_storage(storage, **kw)


__all__ = ["CognitionContext", "from_storage"]
