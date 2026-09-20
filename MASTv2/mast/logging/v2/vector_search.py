"""Semantic search over experiment records via sqlite-vec.

Per compass §4.2. Designed to remain useful when sqlite-vec is missing —
the module exposes a NumpyFallbackSearch that does brute-force cosine
similarity on small corpora (sufficient for MAST's expected 10⁴–10⁵
embeddings).

Embedding source is intentionally pluggable: pass any callable that maps
strings to ``list[float]`` of fixed dim. The default ``HashEmbedder`` is a
deterministic 128-d hashing-trick embedder useful for tests; production
should pass a real model (sentence-transformers, sqlite-lembed, etc.).
"""
from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from mast.logging.v2.ulid import ulid_now

logger = logging.getLogger(__name__)


# ── Embedder protocol ─────────────────────────────────────────────────

Embedder = Callable[[str], list[float]]


@dataclass
class HashEmbedder:
    """Deterministic 128-d hash embedding — for tests and offline demos."""
    dim: int = 128

    def __call__(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        if not text:
            return vec
        tokens = text.lower().split()
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for i in range(self.dim):
                # Sign: bit 0 of byte i%32, magnitude: byte (i+8)%32 / 255
                byte = digest[i % 32]
                mag = digest[(i + 8) % 32] / 255.0
                sign = 1.0 if (byte & 0x01) else -1.0
                vec[i] += sign * mag
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ── sqlite-vec backend ────────────────────────────────────────────────

def have_sqlite_vec() -> bool:
    try:
        import sqlite_vec  # noqa: F401
        return True
    except ImportError:
        return False


class SqliteVecSearch:
    """Vector search backed by sqlite-vec virtual tables.

    Stores embeddings in a separate db file to avoid VACUUM contention with
    the main log db (compass §6.3 risk row "sqlite-vec single-file膨胀").
    """

    def __init__(
        self,
        embedding_db_path: str | Path,
        embedder: Embedder,
        *,
        dim: int = 128,
    ):
        if not have_sqlite_vec():
            raise RuntimeError("sqlite-vec not installed")
        import sqlite_vec
        self.dim = dim
        self.embedder = embedder
        self.path = Path(embedding_db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init()

    def _init(self) -> None:
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0("
            "  id TEXT PRIMARY KEY,"
            f"  embedding FLOAT[{self.dim}]"
            ")"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_meta ("
            "  id TEXT PRIMARY KEY,"
            "  entity_kind TEXT NOT NULL,"
            "  entity_id TEXT NOT NULL,"
            "  text_excerpt TEXT NOT NULL,"
            "  embedded_at TEXT NOT NULL"
            ")"
        )
        self.conn.commit()

    @staticmethod
    def _pack_vec(vec: list[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    def index(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        text: str,
    ) -> str:
        from datetime import datetime, timezone
        vec = self.embedder(text)
        if len(vec) != self.dim:
            raise ValueError(f"Embedder returned dim {len(vec)}, expected {self.dim}")
        emb_id = ulid_now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO embeddings(id, embedding) VALUES (?, ?)",
                (emb_id, self._pack_vec(vec)),
            )
            self.conn.execute(
                "INSERT INTO embedding_meta(id, entity_kind, entity_id, text_excerpt, embedded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (emb_id, entity_kind, entity_id, text[:512],
                 datetime.now(timezone.utc).isoformat()),
            )
        return emb_id

    def knn(self, query_text: str, k: int = 10) -> list[dict]:
        qvec = self.embedder(query_text)
        # sqlite-vec (>=0.1.x) requires the KNN limit expressed as a `k = ?`
        # constraint on the vec0 table, NOT a plain LIMIT — do the vector match in
        # a CTE first, then join metadata (a bare `LIMIT ?` raises
        # "A LIMIT or 'k = ?' constraint is required on vec0 knn queries").
        rows = self.conn.execute(
            "WITH knn AS ("
            "  SELECT id, distance FROM embeddings WHERE embedding MATCH ? AND k = ?"
            ") "
            "SELECT knn.id, knn.distance, m.entity_kind, m.entity_id, m.text_excerpt "
            "FROM knn JOIN embedding_meta m ON knn.id = m.id "
            "ORDER BY knn.distance",
            (self._pack_vec(qvec), k),
        ).fetchall()
        return [
            {
                "embedding_id": r[0],
                "distance": r[1],
                "entity_kind": r[2],
                "entity_id": r[3],
                "text_excerpt": r[4],
            }
            for r in rows
        ]

    def delete_entity(self, *, entity_kind: str, entity_id: str) -> int:
        """Delete all vectors for an (entity_kind, entity_id). Returns rows removed.

        ``index()`` is INSERT-only, so an upserted memory (same ns/path rewritten,
        e.g. by the dream/phase loops) would otherwise accrete stale duplicate
        vectors. Callers delete-then-reindex on every memory write/delete.
        """
        ids = [r[0] for r in self.conn.execute(
            "SELECT id FROM embedding_meta WHERE entity_kind=? AND entity_id=?",
            (entity_kind, entity_id)).fetchall()]
        if not ids:
            return 0
        with self.conn:
            qs = ",".join("?" for _ in ids)
            self.conn.execute(f"DELETE FROM embeddings WHERE id IN ({qs})", ids)
            self.conn.execute(f"DELETE FROM embedding_meta WHERE id IN ({qs})", ids)
        return len(ids)

    def close(self):
        self.conn.close()


# ── NumPy fallback (no extension required) ────────────────────────────

class NumpyFallbackSearch:
    """Brute-force cosine similarity over an in-process matrix.

    Suitable for ≤ 10⁴ embeddings; falls back gracefully when sqlite-vec is
    not installed. Persistence is via a plain sqlite table keyed by entity.
    """

    def __init__(
        self,
        embedding_db_path: str | Path,
        embedder: Embedder,
        *,
        dim: int = 128,
    ):
        self.dim = dim
        self.embedder = embedder
        self.path = Path(embedding_db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  id TEXT PRIMARY KEY, "
            "  entity_kind TEXT NOT NULL, "
            "  entity_id TEXT NOT NULL, "
            "  text_excerpt TEXT NOT NULL, "
            "  embedded_at TEXT NOT NULL, "
            "  vec BLOB NOT NULL"
            ")"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emb_entity ON embeddings(entity_kind, entity_id)"
        )
        self.conn.commit()

    @staticmethod
    def _pack(vec: list[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _unpack(blob: bytes, dim: int) -> list[float]:
        return list(struct.unpack(f"{dim}f", blob))

    def index(self, *, entity_kind: str, entity_id: str, text: str) -> str:
        from datetime import datetime, timezone
        vec = self.embedder(text)
        if len(vec) != self.dim:
            raise ValueError(f"Embedder returned dim {len(vec)}, expected {self.dim}")
        emb_id = ulid_now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO embeddings(id, entity_kind, entity_id, text_excerpt, embedded_at, vec) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (emb_id, entity_kind, entity_id, text[:512],
                 datetime.now(timezone.utc).isoformat(), self._pack(vec)),
            )
        return emb_id

    def knn(self, query_text: str, k: int = 10) -> list[dict]:
        qvec = self.embedder(query_text)
        results: list[tuple[float, sqlite3.Row]] = []
        for row in self.conn.execute("SELECT * FROM embeddings"):
            vec = self._unpack(row["vec"], self.dim)
            dot = sum(a * b for a, b in zip(qvec, vec))
            # qvec & vec are L2-normalised by HashEmbedder; for arbitrary embedders, normalise.
            results.append((1.0 - dot, row))
        results.sort(key=lambda x: x[0])
        return [
            {
                "embedding_id": r["id"],
                "distance": d,
                "entity_kind": r["entity_kind"],
                "entity_id": r["entity_id"],
                "text_excerpt": r["text_excerpt"],
            }
            for d, r in results[:k]
        ]

    def delete_entity(self, *, entity_kind: str, entity_id: str) -> int:
        """Delete all vectors for an (entity_kind, entity_id). Returns rows removed.

        Mirrors ``SqliteVecSearch.delete_entity`` so memory upserts/deletes can
        clear stale vectors before re-indexing (``index()`` is INSERT-only).
        """
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM embeddings WHERE entity_kind=? AND entity_id=?",
                (entity_kind, entity_id))
            return cur.rowcount

    def close(self):
        self.conn.close()


# ── Factory ───────────────────────────────────────────────────────────

def open_search(
    embedding_db_path: str | Path,
    embedder: Embedder | None = None,
    *,
    dim: int = 128,
    prefer_sqlite_vec: bool = True,
):
    """Return the best available backend (sqlite-vec if installed, else NumPy)."""
    embedder = embedder or HashEmbedder(dim=dim)
    if prefer_sqlite_vec and have_sqlite_vec():
        try:
            return SqliteVecSearch(embedding_db_path, embedder, dim=dim)
        except Exception as exc:
            logger.warning("sqlite-vec init failed (%s); falling back to NumPy", exc)
    return NumpyFallbackSearch(embedding_db_path, embedder, dim=dim)


# ── Bulk indexing helper ──────────────────────────────────────────────

def bulk_index_observations(
    backend,
    items: Iterable[dict],
    *,
    text_fields: tuple[str, ...] = ("observable", "result_summary_json"),
) -> int:
    """Index a stream of observation rows into the embedding store.

    Each item dict is expected to have keys ``id`` and any of *text_fields*.
    Returns the number of items indexed.
    """
    n = 0
    for item in items:
        text = " ".join(str(item.get(f, "") or "") for f in text_fields)
        if not text.strip():
            continue
        backend.index(entity_kind="observation", entity_id=item["id"], text=text)
        n += 1
    return n
