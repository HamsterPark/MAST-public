"""OCR fallback for scanned / image-only literature PDFs (design §7, EXTRACT).

A minority of user PDFs are *scanned* — pages are images with no text layer, so
PyMuPDF's ``page.get_text()`` returns (near-)nothing and the ingest pipeline's
``warning:no_text_layer`` guard refuses them. This module is the honest recovery
path: render each page to an image and read it with Alibaba Cloud **qwen-vl-ocr**
(DashScope's vision-language OCR model), then hand the recovered text back to the
normal EXTRACT → RESOLVE → CHUNK → EMBED pipeline.

Model / transport
-----------------
* Model ``qwen3.5-ocr`` — Alibaba's current vision-language OCR generation
  (the older ``qwen-vl-ocr`` snapshots are free-tier-exhausted on our key; see
  ``_OCR_MODEL``). Called via the **OpenAI-compatible** DashScope endpoint
  ``https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`` — the
  SAME base URL the ``qwen`` provider + the embedder already use
  (``config.PROVIDER_BASE_URL['qwen']``). The key is resolved through the
  canonical ``config._load_provider_key('dashscope')`` loader (env var →
  ``api key/dashscope.env``); we never invent a second key path.
* Each page is sent as one multimodal user turn: an ``image_url`` part carrying a
  ``data:image/png;base64,…`` URI (+ ``min_pixels`` / ``max_pixels`` budget) and a
  short text instruction. The recognized text comes back in
  ``choices[0].message.content``.

Graceful degradation (house rule 2 — OCR must NEVER crash or block an upload)
---------------------------------------------------------------------------
Every failure mode returns an empty string (``ocr_pdf``) / empty list
(``ocr_pdf_pages``) and logs a warning — it never raises:
  * PyMuPDF (fitz) missing / PDF unreadable → no pages rendered → ``""``.
  * httpx missing / no DashScope key → skip OCR → ``""``.
  * Per-page HTTP / network / quota error → that page contributes ``""``; a run
    of consecutive failures with no success (systemic auth/network) aborts early
    so a dead key can't trigger dozens of doomed calls.
The caller (``knowledge.ingest.ingest_pdf``) only *upgrades* to OCR text when it
clears the same non-whitespace-char threshold a real text layer must clear, so a
partial/garbled OCR result never silently pollutes the big index.

Cost / time guards
------------------
Pages are rendered at a moderate DPI and capped (``max_pages``); ``max_pixels``
bounds per-image vision tokens (32×32 px == 1 token for qwen-vl-ocr). Both the
page renderer and the per-page OCR call are injectable so tests exercise the loop
with no real PDF, no fitz raster, and no DashScope network.

Pure sync I/O — no event-loop assumptions (ingest wraps the whole pipeline in a
worker thread / ``asyncio.to_thread``).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── qwen OCR model + endpoint (docs/api_providers/ — DashScope compat mode) ──
# Model family: Alibaba's vision-language OCR. ``qwen3.5-ocr`` is the CURRENT
# generation and is what we default to — a live smoke (2026-07-20) confirmed it
# returns 200 + correct text on our account, while the older ``qwen-vl-ocr`` /
# ``qwen-vl-ocr-latest`` snapshots 403 with ``AllocationQuota.FreeTierOnly``
# ("free quota exhausted") on this key. Both speak the identical compatible-mode
# request/response shape, so switching back (once qwen-vl-ocr is funded) is a
# one-liner. Override per-call via ``ocr_pdf(..., model=…)``.
_OCR_MODEL = "qwen3.5-ocr"
_OCR_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# ── Backends, in preference order ────────────────────────────────────────────
# (provider, model, endpoint, sends_pixel_hints)
#
# The MAIN chat model comes first. Kimi K3 is multimodal, and a live test on a
# real paper (2026-07-27, AISTM-reference/2023_Qian_PRB.pdf page 1 at 150 DPI)
# returned the title, authors, journal/volume/year and the full abstract
# correctly. That matters for one blunt reason: **the moonshot key is the one
# key this system is guaranteed to have** — it drives every agent. DashScope is
# an optional provider whose key has already been exhausted once on this
# account (``qwen-vl-ocr`` snapshots 403 with AllocationQuota.FreeTierOnly), and
# an OCR path that dies with a provider nobody configured is a path that is only
# discovered when a scanned PDF finally arrives.
#
# Kept as a FALLBACK rather than deleted: qwen3.5-ocr is a purpose-built OCR
# model, and on a dense scan it may well beat a general multimodal model.
#
# Worth knowing how rarely this runs at all: of the 38 papers in
# AISTM-reference/, the 12 sampled all yield 34k–500k characters straight from
# PyMuPDF — **none of them needs OCR**. Modern publisher PDFs carry a text
# layer; this path exists for old scans, so its job is to work when it is
# finally called, on whatever key the operator actually has.
#
# `sends_pixel_hints`: min_pixels/max_pixels are a DashScope vision extension.
# Moonshot's OpenAI-compatible endpoint does not define them, so they are only
# sent to backends that document them.
_OCR_BACKENDS: tuple[tuple[str, str, str, bool], ...] = (
    ("moonshot", "kimi-k3",
     "https://api.moonshot.cn/v1/chat/completions", False),
    ("dashscope", _OCR_MODEL, _OCR_ENDPOINT, True),
)


def _pick_backend() -> tuple[str, str, str, bool] | None:
    """First backend in preference order that actually has a key; None if none."""
    try:
        from mast.config import _load_provider_key
    except Exception as e:  # pragma: no cover — config import guard
        logger.warning("OCR: config import failed: %s", e)
        return None
    for provider, model, endpoint, hints in _OCR_BACKENDS:
        try:
            if (_load_provider_key(provider) or "").strip():
                return (provider, model, endpoint, hints)
        except Exception:  # noqa: BLE001 — a broken provider must not mask the next
            continue
    return None

# Per-image pixel budget (qwen-vl-ocr doc defaults). 32×32 px == 1 vision token,
# so max_pixels == 8_388_608 caps a single page at ~8192 vision tokens; smaller
# images are enlarged to min_pixels. We render at a moderate DPI (below) so a
# typical A4 page lands well under the ceiling.
_OCR_MIN_PIXELS = 3072
_OCR_MAX_PIXELS = 8_388_608
# Output cap per page — OCR text is bounded; keeps a runaway response in check
# (the model's hard ceiling is 32_768).
_OCR_MAX_TOKENS = 8192
# Page render resolution. 150 DPI A4 ≈ 1240×1754 ≈ 2.1 MPx ≈ ~2100 vision tokens
# — legible for OCR while keeping per-page token cost moderate.
_OCR_DPI = 150
# Cost/time guard for very long scans (a 500-page image PDF must not bill for 500
# vision calls). Callers can override.
_OCR_MAX_PAGES = 50
# Slow campus/VPN links to aliyun (see docs/api_providers/dashscope_realtime_voice.md
# — TCP handshake ~17 s observed). Generous per-page read timeout, longer connect.
#
# 90 s was sized for a purpose-built OCR model, which answers directly. The
# preferred backend is now the main chat model, and Kimi K3 is an always-on
# reasoning model: it thinks before it answers, on every page. Measured on one
# real paper page at 150 DPI (2026-07-27): 48.9 s once, 98.5 s on a repeat —
# the same request, twice the time. That variance is the CoT, not the network.
#
# OCR is a background ingest step, never an interactive one, so waiting is
# cheap and a timeout is expensive: it burns the tokens the page already cost
# and returns nothing. Sized to hold the observed spread with room to spare.
_OCR_TIMEOUT_S = 240.0
_OCR_CONNECT_TIMEOUT_S = 30.0
# Minimal, deterministic instruction so the model returns raw text (not prose).
_OCR_PROMPT = (
    "请识别并输出图片中的所有文字内容，保持自然的阅读顺序，"
    "不要添加任何解释、翻译或额外说明。"
)
# Fail-fast breaker: if the first N page calls all raise (systemic auth / quota /
# network) with zero successes so far, stop — a dead key must not fan out.
_OCR_MAX_CONSECUTIVE_FAILS = 2


# ── key loading (canonical loader — never raises) ────────────────────
def _load_key() -> str:
    """Key for the FIRST configured OCR backend; ``""`` when none is configured.

    Was DashScope-only, which made the whole OCR path unavailable to an operator
    who never set up an optional provider — while the multimodal model that
    drives every agent sat right there (see ``_OCR_BACKENDS``).

    Reuses ``config._load_provider_key`` (env-var precedence → ``<provider>.env``)
    exactly like the voice layer and the embedder — no bespoke key path.
    """
    picked = _pick_backend()
    if picked is None:
        return ""
    try:
        from mast.config import _load_provider_key

        return (_load_provider_key(picked[0]) or "").strip()
    except Exception as e:  # pragma: no cover - config import guard
        logger.warning("OCR: %s key load failed: %s", picked[0], e)
        return ""


def ocr_available() -> bool:
    """Best-effort readiness probe: fitz (render) + httpx (transport) + a key.

    Used by the ingest-status endpoint so the panel can tell the operator whether
    a scanned PDF can be OCR-recovered. Never raises.
    """
    try:
        import fitz  # noqa: F401  (PyMuPDF)
        import httpx  # noqa: F401
    except Exception:
        return False
    return bool(_load_key())


# ── page rendering (PyMuPDF → PNG bytes) ─────────────────────────────
def _render_pages(pdf_path: str | Path, *, dpi: int, max_pages: int) -> list[bytes]:
    """Render up to ``max_pages`` pages of *pdf_path* to PNG bytes via PyMuPDF.

    Returns ``[]`` on any failure (fitz missing / file absent / open error) — the
    caller degrades to "no OCR". A single page that fails to rasterize is skipped
    (it contributes no image) rather than aborting the whole document.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover - exercised only when fitz absent
        logger.warning("OCR: PyMuPDF (fitz) unavailable — cannot render PDF: %s", e)
        return []

    p = Path(pdf_path)
    if not p.is_file():
        logger.warning("OCR: PDF not found: %s", p)
        return []

    try:
        doc = fitz.open(str(p))
    except Exception as e:
        logger.warning("OCR: cannot open PDF %s: %s", p, e)
        return []

    out: list[bytes] = []
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                logger.info(
                    "OCR: page cap %d reached for %s — remaining pages skipped",
                    max_pages, p.name,
                )
                break
            try:
                pix = page.get_pixmap(dpi=dpi)
                out.append(pix.tobytes("png"))
            except Exception as e:  # pragma: no cover - per-page raster failure
                logger.warning("OCR: render page %d of %s failed: %s", i, p.name, e)
    finally:
        try:
            doc.close()
        except Exception:  # pragma: no cover
            pass
    return out


# ── per-page OCR call (DashScope qwen-vl-ocr, OpenAI-compatible) ──────
def _default_page_ocr(data_uri: str, *, key: str, model: str, client=None) -> str:
    """OCR one page image (a ``data:image/png;base64,…`` URI) via the OCR model.

    Raises on transport / HTTP error (the caller counts it toward the fail-fast
    breaker); returns ``""`` when the response carries no text. When *client* is
    supplied it is reused (one TCP+TLS handshake amortised across all pages of a
    document — the per-page connect dominates latency on slow links); otherwise a
    short-lived client is created and closed here.
    """
    import httpx  # lazy

    # Route by the model that was picked, not by a module constant — the two
    # backends have different endpoints and only one of them defines the pixel
    # hints (see _OCR_BACKENDS).
    endpoint, send_hints = _OCR_ENDPOINT, True
    for _prov, _model, _ep, _hints in _OCR_BACKENDS:
        if _model == model:
            endpoint, send_hints = _ep, _hints
            break

    image_part: dict = {"type": "image_url", "image_url": {"url": data_uri}}
    if send_hints:
        image_part["min_pixels"] = _OCR_MIN_PIXELS
        image_part["max_pixels"] = _OCR_MAX_PIXELS

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_part,
                    {"type": "text", "text": _OCR_PROMPT},
                ],
            }
        ],
        "max_tokens": _OCR_MAX_TOKENS,
        "stream": False,
    }
    own = client is None
    if own:
        timeout = httpx.Timeout(_OCR_TIMEOUT_S, connect=_OCR_CONNECT_TIMEOUT_S)
        client = httpx.Client(timeout=timeout)
    try:
        r = client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        r.raise_for_status()
        data = r.json()
    finally:
        if own:
            client.close()
    try:  # book OCR cost (VLM tokens) into the 用量·花销 ledger — fail-safe
        _u = (data or {}).get("usage") or {}
        from mast.billing.capture import record_ocr
        record_ocr(model=model,
                   input_tokens=int(_u.get("prompt_tokens") or 0),
                   output_tokens=int(_u.get("completion_tokens") or 0),
                   pages=1)
    except Exception:  # noqa: BLE001
        pass
    return _content_text(data)


def _content_text(data: dict) -> str:
    """Pull the assistant text out of a chat-completions response body.

    ``content`` is normally a plain string; defend against the multimodal
    list-of-parts shape too (``[{"type":"text","text":…}, …]``).
    """
    choices = (data or {}).get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


# ── public API ───────────────────────────────────────────────────────
# An injectable per-page OCR callable: ``data_uri -> text``. When provided we do
# NOT require a DashScope key (tests / custom transports pass their own).
PageOCR = Callable[[str], str]


def ocr_pdf_pages(
    pdf_path: str | Path,
    *,
    model: Optional[str] = None,
    key: Optional[str] = None,
    dpi: int = _OCR_DPI,
    max_pages: int = _OCR_MAX_PAGES,
    page_ocr: Optional[PageOCR] = None,
) -> list[str]:
    """OCR each rendered page of *pdf_path* → per-page text list.

    Never raises. Returns ``[]`` (or a partial list) on any degradation: fitz
    missing, PDF unreadable, no key, or a systemic run of failed page calls.

    Args:
        model:    explicit vision-model id. ``None`` → the first configured
                  backend in ``_OCR_BACKENDS`` (main chat model first, then the
                  purpose-built OCR model).
        key:      explicit provider key; ``None`` → canonical loader for whichever
                  backend was picked. Ignored when ``page_ocr`` is injected.
        dpi:      page render resolution (higher = sharper but pricier).
        max_pages: hard cap on pages rendered/OCR'd (cost guard).
        page_ocr: injectable ``data_uri -> text`` (tests / custom transport).
                  ``None`` → the default DashScope qwen-vl-ocr call.
    """
    pages_png = _render_pages(pdf_path, dpi=dpi, max_pages=max_pages)
    if not pages_png:
        return []

    # Default path: bind key/model to a SHARED httpx client so N pages amortise a
    # single TCP+TLS handshake. Injected page_ocr bypasses transport entirely.
    call = page_ocr
    shared_client = None
    resolved_model = model
    if call is None:
        picked = _pick_backend()
        if resolved_model is None:
            resolved_model = picked[1] if picked else _OCR_MODEL
        resolved_key = key if key is not None else _load_key()
        if not resolved_key:
            logger.warning(
                "OCR: no vision-model key configured (tried %s) — skipping OCR "
                "for %s",
                ", ".join(p for p, *_ in _OCR_BACKENDS), Path(pdf_path).name,
            )
            return []
        logger.info("OCR: %s via %s", Path(pdf_path).name, resolved_model)
        try:
            import httpx  # lazy

            shared_client = httpx.Client(
                timeout=httpx.Timeout(_OCR_TIMEOUT_S, connect=_OCR_CONNECT_TIMEOUT_S)
            )
        except Exception as e:  # pragma: no cover - httpx missing
            logger.warning("OCR: httpx unavailable — skipping OCR: %s", e)
            return []

        def call(data_uri: str) -> str:  # bind key/model/client for the default path
            return _default_page_ocr(
                data_uri, key=resolved_key, model=resolved_model,
                client=shared_client
            )

    texts: list[str] = []
    consecutive_fails = 0
    got_any = False
    try:
        for i, png in enumerate(pages_png):
            if not png:
                texts.append("")
                continue
            data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
            try:
                txt = call(data_uri) or ""
            except Exception as e:
                consecutive_fails += 1
                logger.warning(
                    "OCR: page %d call failed (%d consecutive, %s): %s",
                    i, consecutive_fails, Path(pdf_path).name, e,
                )
                texts.append("")
                # A dead key / down endpoint fails identically on every page — bail
                # out once we've seen enough failures with nothing to show for it.
                if not got_any and consecutive_fails >= _OCR_MAX_CONSECUTIVE_FAILS:
                    logger.warning(
                        "OCR: %d consecutive failures with no success — aborting %s",
                        consecutive_fails, Path(pdf_path).name,
                    )
                    break
                continue
            consecutive_fails = 0
            if txt.strip():
                got_any = True
            texts.append(txt)
    finally:
        if shared_client is not None:
            try:
                shared_client.close()
            except Exception:  # pragma: no cover
                pass
    return texts


def ocr_pdf(pdf_path: str | Path, **kwargs) -> str:
    """OCR a scanned / image-only PDF → one recovered-text string.

    The task-facing contract (``path -> str``). Joins the non-empty per-page
    results. Never raises; returns ``""`` on any failure / degradation so a
    caller can treat "no text recovered" uniformly. See :func:`ocr_pdf_pages`
    for the injectable knobs.
    """
    try:
        pages = ocr_pdf_pages(pdf_path, **kwargs)
    except Exception as e:  # pragma: no cover - ocr_pdf_pages already guards
        logger.warning("OCR: ocr_pdf failed for %s: %s", pdf_path, e)
        return ""
    return "\n\n".join(t for t in pages if t and t.strip())


__all__ = [
    "ocr_pdf",
    "ocr_pdf_pages",
    "ocr_available",
    "_OCR_MODEL",
]
