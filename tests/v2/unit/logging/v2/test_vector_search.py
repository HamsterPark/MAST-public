"""Vector search smoke tests against the NumPy fallback."""
import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
while _MASTV2_ROOT in sys.path:
    sys.path.remove(_MASTV2_ROOT)
sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest

from mast.logging.v2.vector_search import (
    HashEmbedder,
    NumpyFallbackSearch,
    open_search,
)


def test_hash_embedder_deterministic():
    e = HashEmbedder(dim=64)
    a = e("STM tip preparation on Si(111)")
    b = e("STM tip preparation on Si(111)")
    assert a == b
    assert len(a) == 64
    # L2 normalised
    norm = sum(v * v for v in a) ** 0.5
    assert 0.95 < norm < 1.05


def test_numpy_fallback_knn(tmp_path):
    backend = NumpyFallbackSearch(tmp_path / "emb.db", HashEmbedder(dim=64), dim=64)
    backend.index(entity_kind="experiment", entity_id="e1",
                  text="topography of Au(111) surface")
    backend.index(entity_kind="experiment", entity_id="e2",
                  text="STS spectroscopy on superconductor")
    backend.index(entity_kind="experiment", entity_id="e3",
                  text="topography measurement on Au(111) clean")
    results = backend.knn("topography of Au(111)", k=2)
    assert len(results) == 2
    # The two Au(111) topography records should beat the superconductor one.
    ids = {r["entity_id"] for r in results}
    assert "e2" not in ids or ids == {"e1", "e3"}


def test_open_search_falls_back(tmp_path):
    backend = open_search(tmp_path / "emb.db", dim=32, prefer_sqlite_vec=False)
    backend.index(entity_kind="action", entity_id="a1", text="set_bias to -2V")
    out = backend.knn("set_bias query", k=1)
    assert out and out[0]["entity_id"] == "a1"
