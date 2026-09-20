"""Embedder factory for cross-conversation memory semantic recall.

Three tiers, tried in order (see ``make_embedder``):

  A. **DashScope ``text-embedding-v3`` (cloud, PRIMARY)** — reuses the qwen/
     DashScope key + base url already configured for the Qwen chat provider
     (``mast.agents._shared.models``). OpenAI-compatible, so called via the
     ``openai`` SDK. dim 1024, L2-normalised here.
  B. **Local ``transformers`` mean-pooling (best-effort MIDDLE)** — bare
     AutoModel + mean pool (sentence-transformers is NOT installed). Only used
     when the model is ALREADY in the HF cache (``local_files_only``) OR
     ``MAST_MEMORY_ALLOW_EMBED_DOWNLOAD=1`` is set — a first-use download is
     slow/blocked on the campus VPN, so we never trigger one silently.
  C. **Substring (FINAL fallback)** — no embedding; the recall middleware uses
     ``MemoryStore.search`` (SQL LIKE). ``make_embedder`` returns
     ``(None, 0, "substring")`` to signal this.

Every embedding tier L2-normalises its output (``NumpyFallbackSearch`` cosine
assumes unit vectors). The returned ``dim`` is fixed per backend; the caller
(``vector_search.open_search(dim=…)``) bakes ONE dim into the store, so a tier
change requires a re-index — the CognitionContext manifest handles that.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Callable

logger = logging.getLogger(__name__)

_DASHSCOPE_MODEL = "text-embedding-v3"
_DASHSCOPE_DIM = 1024
_DASHSCOPE_MAX_CHARS = 8000  # text-embedding-v3 caps ~8192 tokens; truncate input

_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"
_LOCAL_DIM = 512


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class DashScopeEmbedder:
    """Cloud embedder via DashScope text-embedding-v3 (OpenAI-compatible API)."""

    dim = _DASHSCOPE_DIM
    backend = "dashscope"

    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def __call__(self, text: str) -> list[float]:
        cli = self._ensure_client()
        resp = cli.embeddings.create(
            model=_DASHSCOPE_MODEL,
            input=(text or "")[:_DASHSCOPE_MAX_CHARS],
            dimensions=_DASHSCOPE_DIM,
        )
        return _l2_normalise(list(resp.data[0].embedding))


class LocalTransformersEmbedder:
    """Local mean-pooling embedder over a small HF model (no sentence-transformers).

    Loads lazily. Raises on first call if the model isn't cached and downloads
    aren't permitted — the caller treats embedding as best-effort and degrades.
    """

    dim = _LOCAL_DIM
    backend = "local"

    def __init__(self, model_name: str = _LOCAL_MODEL):
        self._model_name = model_name
        self._tok = None
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch  # noqa: F401  (ensure torch present)
        from transformers import AutoModel, AutoTokenizer
        allow_dl = os.environ.get("MAST_MEMORY_ALLOW_EMBED_DOWNLOAD", "") == "1"
        kw = {} if allow_dl else {"local_files_only": True}
        self._tok = AutoTokenizer.from_pretrained(self._model_name, **kw)
        self._model = AutoModel.from_pretrained(self._model_name, **kw)
        self._model.eval()

    def __call__(self, text: str) -> list[float]:
        import torch
        self._ensure_model()
        enc = self._tok(text or "", return_tensors="pt", truncation=True,
                        max_length=512, padding=True)
        with torch.no_grad():
            out = self._model(**enc)
        # mean-pool last_hidden_state masked by attention_mask
        hidden = out.last_hidden_state  # [1, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()  # [1, T, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        vec = (summed / counts).squeeze(0).tolist()
        return _l2_normalise(vec)


def _dashscope_available() -> tuple[bool, str, str]:
    """(ok, api_key, base_url) — whether a DashScope key is configured."""
    try:
        from mast.agents._shared.models import PROVIDER_BASE_URL, load_provider_key
        key = load_provider_key("qwen")
        base = PROVIDER_BASE_URL.get("qwen", "")
        return bool(key and base), key, base
    except Exception as exc:  # noqa: BLE001
        logger.debug("dashscope availability check failed: %s", exc)
        return False, "", ""


def _local_available() -> bool:
    """True if local embedding is permitted AND feasible (cached or download ok)."""
    if os.environ.get("MAST_MEMORY_ALLOW_EMBED_DOWNLOAD", "") == "1":
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False
    # Only feasible offline if the model is already cached — probe via local_files_only.
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(_LOCAL_MODEL, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def make_embedder(prefer: str = "auto") -> tuple[Callable[[str], list[float]] | None, int, str]:
    """Resolve the best available embedder.

    Returns ``(embedder, dim, backend_name)`` where ``backend_name`` is one of
    ``"dashscope" | "local" | "substring"``. For the substring tier the embedder
    is ``None`` and dim is 0 — the recall middleware then uses ``MemoryStore.search``.

    ``prefer`` lets callers/tests force a tier: ``"dashscope" | "local" |
    "substring" | "auto"`` (default).
    """
    if prefer in ("dashscope", "auto"):
        ok, key, base = _dashscope_available()
        if ok:
            logger.info("memory embedder: DashScope text-embedding-v3 (dim=%d)", _DASHSCOPE_DIM)
            return DashScopeEmbedder(key, base), _DASHSCOPE_DIM, "dashscope"
        if prefer == "dashscope":
            logger.warning("memory embedder: DashScope forced but no key; using substring")
            return None, 0, "substring"

    if prefer in ("local", "auto"):
        if _local_available():
            logger.info("memory embedder: local %s (dim=%d)", _LOCAL_MODEL, _LOCAL_DIM)
            return LocalTransformersEmbedder(), _LOCAL_DIM, "local"
        if prefer == "local":
            logger.warning("memory embedder: local forced but model unavailable; using substring")
            return None, 0, "substring"

    logger.info("memory embedder: substring fallback (no vector recall)")
    return None, 0, "substring"


__all__ = [
    "make_embedder",
    "DashScopeEmbedder",
    "LocalTransformersEmbedder",
]
