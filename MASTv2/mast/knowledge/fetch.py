"""Experimental, best-effort, ToS-respecting full-text fetch (Decision 2).

Design: literature_library_design.md §8.2.

This is a **clearly experimental** convenience: "paste DOI / URL → MAST tries to
fetch the full text", offered ALONGSIDE manual upload, never replacing it.

Hard rules baked in here (non-negotiable — see §1.2 / §8.2):

  * **Legality gate (hard).** Before fetching anything, the target host's
    ``robots.txt`` is consulted and only legitimately-accessible content is
    retrieved: an **open-access** PDF copy, a publisher **landing page**
    (metadata/abstract only), or content the **user is licensed** for. Publisher
    ToS / robots are respected.
  * **Never** bypass a paywall, log in, send credentials/cookies, or scrape
    content behind authentication. There is no auth code path in this module by
    construction.
  * On any failure (paywall, robots disallow, no OA copy, network error) the
    fetch **does not retry aggressively or work around it** — it returns a clear
    "please upload manually" message and leaves the manual path open.

The literature agent does NOT call this directly — it only *requests* a fetch
(``request_fulltext_from_user`` with a ``url_hint``); the human / this helper
makes the legality call.

Return contract (always a plain JSON-serializable dict)::

    {
      "status":   "ok_pdf" | "ok_metadata" | "blocked" | "error" | "unavailable",
      "pdf_path": "<path>" | None,     # set only when status == "ok_pdf"
      "metadata": {...} | None,        # landing-page / OA metadata when available
      "message":  "<human-readable note>",
      "url":      "<resolved url>" | "",
      "source":   "user_url",
    }

Degradation:
  * Missing ``httpx`` → returns ``status="unavailable"`` (no crash, no network).
  * All network I/O is wrapped in try/except; no ``time.sleep``; non-blocking.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)

# httpx is an optional dependency here — degrade gracefully if absent.
try:  # pragma: no cover - import guard
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


# ── Constants ────────────────────────────────────────────────────────
_USER_AGENT = "MAST-LiteratureFetcher/1.0 (+research; respects robots.txt)"
_DOI_PREFIX = "https://doi.org/"
_DEFAULT_TIMEOUT = 20.0
_MAX_PDF_BYTES = 60 * 1024 * 1024  # 60 MB safety cap on a downloaded PDF

# Signals on a landing page / HTTP response that we hit a paywall / auth wall.
# If we see these we REFUSE to go further (never bypass).
_PAYWALL_MARKERS = (
    "paywall",
    "purchase pdf",
    "buy article",
    "get access",
    "institutional login",
    "sign in to read",
    "subscribe to view",
    "rent this article",
    "access through your institution",
    "log in to wiley",
)
_AUTH_STATUS = frozenset({401, 402, 403})


# ── Path resolution ──────────────────────────────────────────────────
# Both of these used to derive from a private repo-root walk local to this file
# (one of five copies across ``knowledge/*``); they now go through the ONE shared
# resolver, ``knowledge/paths.py``. Same answers as before (verified 2026-07-29).
_HERE = Path(__file__).resolve()


def _find_repo_root() -> Path:
    """The install/index base — NOT the user-data root.

    Kept as a named function because the ``api key/`` probe below wants this base
    and only this base (see ``ingest._load_dashscope_key`` for the field
    report on probing the wrong one).
    """
    from mast.knowledge.paths import base_dir
    return base_dir()


def _papers_dir() -> Path:
    """``MASTv2/data/papers/`` (design §2.3). Created lazily on first save."""
    from mast.knowledge.paths import papers_dir
    return papers_dir()


# ── Input validation / slug helpers ──────────────────────────────────
_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug_for(doi_or_url: str) -> str:
    """A filesystem-safe directory slug for an as-yet-unbound paper.

    Path-traversal-proof: the result contains only ``[A-Za-z0-9._-]`` and can
    never escape the papers dir. We derive it from a stable hash of the input
    plus a short readable prefix.
    """
    cleaned = _SLUG_SAFE.sub("-", (doi_or_url or "").strip()).strip("-._")
    cleaned = cleaned[:40] or "paper"
    h = hashlib.sha1((doi_or_url or "").encode("utf-8")).hexdigest()[:10]
    return f"{cleaned}-{h}"


def _normalize_doi(s: str) -> str:
    """Strip a DOI down to its bare form (``10.xxxx/...``) or return ''."""
    s = (s or "").strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^doi:\s*", "", s, flags=re.IGNORECASE)
    m = re.match(r"(10\.\d{4,9}/\S+)", s)
    return m.group(1) if m else ""


def _looks_like_url(s: str) -> bool:
    try:
        u = urlparse((s or "").strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _resolve_candidate_url(doi_or_url: str) -> str:
    """Map a raw DOI/URL input to a candidate landing URL (§8.2 step 1).

    Returns '' if the input is neither a valid http(s) URL nor a parseable DOI.
    Only ``http``/``https`` schemes are ever returned (no ``file:``/``ftp:``).
    """
    raw = (doi_or_url or "").strip()
    if not raw:
        return ""
    doi = _normalize_doi(raw)
    if doi:
        return _DOI_PREFIX + doi
    if _looks_like_url(raw):
        return raw
    return ""


# ── Legality gate: robots.txt ─────────────────────────────────────────
def _robots_allows(url: str, *, fetcher=None) -> bool:
    """Return True iff ``robots.txt`` for *url*'s host allows our UA to fetch it.

    Hard gate (§8.2 step 2). *fetcher* is an injectable ``(robots_url) -> str``
    callable returning the robots.txt body — tests pass a fake so NO real
    request is made. In production it defaults to an httpx GET.

    Conservative semantics:
      * robots.txt fetched & parsed → honour its verdict for our UA.
      * robots.txt explicitly says ``Disallow: /`` for us → False.
      * robots.txt missing / unreadable → **allow** (the IETF default: absence
        of a rule is permission), but the higher-level fetch still refuses
        anything that looks like auth/paywall content.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        body: Optional[str] = None
        if fetcher is not None:
            body = fetcher(robots_url)
        elif httpx is not None:
            try:
                with httpx.Client(
                    timeout=_DEFAULT_TIMEOUT,
                    follow_redirects=True,
                    headers={"User-Agent": _USER_AGENT},
                ) as c:
                    r = c.get(robots_url)
                    if r.status_code == 200:
                        body = r.text
            except Exception as e:  # network failure → treat as "no robots.txt"
                logger.debug("robots.txt fetch failed for %s: %s", robots_url, e)
                body = None

        if not body:
            # No robots.txt available → default-allow (RFC 9309 §2.4).
            return True

        rp = RobotFileParser()
        rp.parse(body.splitlines())
        return rp.can_fetch(_USER_AGENT, url)
    except Exception as e:  # never raise out of the gate
        logger.debug("robots check error for %s: %s", url, e)
        # On an unexpected parse error, be conservative and DISALLOW.
        return False


def _looks_paywalled(text: str) -> bool:
    """Heuristic: does this landing-page HTML scream 'paywall / login required'?"""
    low = (text or "").lower()
    return any(marker in low for marker in _PAYWALL_MARKERS)


def _result(
    status: str,
    *,
    message: str,
    pdf_path: Optional[str] = None,
    metadata: Optional[dict] = None,
    url: str = "",
) -> dict[str, Any]:
    """Build the canonical serializable result dict."""
    return {
        "status": status,
        "pdf_path": pdf_path,
        "metadata": metadata,
        "message": message,
        "url": url,
        "source": "user_url",
    }


#: Downloads land in a staging sub-directory rather than beside the ingested
#: papers. They are keyed by DOI, while ``ingest_pdf`` keys its own copy by
#: work_id, so a fetched-then-ingested paper used to sit in the corpus TWICE —
#: once complete, once as a bare PDF with no full text, chunks or metadata, each
#: answering to a different paper_id. Corpus scans skip this directory
#: (``literature/tools._NON_PAPER_DIRS``).
STAGING_DIRNAME = "_incoming"


def _save_pdf(content: bytes, slug: str) -> str:
    """Write a downloaded PDF to ``data/papers/_incoming/<slug>/source.pdf``.

    Returns the absolute path string. The slug is sanitized so the write can
    never escape the papers dir. Atomic (temp + replace).
    """
    base = _papers_dir() / STAGING_DIRNAME
    # slug is already _SLUG_SAFE-cleaned, but resolve-and-check to be sure.
    target_dir = (base / slug).resolve()
    if base.resolve() not in target_dir.parents and target_dir != base.resolve():
        raise ValueError(f"refusing to write outside papers dir: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    final = target_dir / "source.pdf"
    tmp = target_dir / "source.pdf.part"
    tmp.write_bytes(content)
    tmp.replace(final)  # atomic on same filesystem
    return str(final)


# ── OA location resolver (OpenAlex / Unpaywall) ──────────────────────
# Without this, the "only reliably-legal PDF source" (oa_url) is never
# populated, so a bare paywalled DOI almost always blocks. OpenAlex needs no
# key (optional polite mailto); Unpaywall is tried only when an email exists.
_OPENALEX_WORK = "https://api.openalex.org/works/doi:{doi}"
_UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"


def _oa_email() -> str:
    """A polite contact email for OpenAlex/Unpaywall.

    Optional for OpenAlex (just upgrades to the polite pool); REQUIRED by
    Unpaywall. Read from env ``MAST_OA_EMAIL`` or ``api key/openalex.env``.
    Returns '' when unset (OpenAlex still works without it).
    """
    import os
    e = (os.environ.get("MAST_OA_EMAIL") or "").strip()
    if e:
        return e
    try:
        p = _find_repo_root() / "api key" / "openalex.env"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:  # KEY=value form
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
                return line.strip('"').strip("'")
    except Exception:  # pragma: no cover - best effort
        pass
    return ""


def _json_of(resp) -> Any:
    """Best-effort JSON from an httpx-like response (or a test fake)."""
    try:
        return resp.json()
    except Exception:
        try:
            import json as _json
            return _json.loads(getattr(resp, "text", "") or "")
        except Exception:
            return None


def _pdf_url_from_openalex(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for loc_key in ("best_oa_location", "primary_location"):
        loc = data.get(loc_key)
        if isinstance(loc, dict):
            u = str(loc.get("pdf_url") or "").strip()
            if _looks_like_url(u):
                return u
    oa = data.get("open_access")
    if isinstance(oa, dict):
        u = str(oa.get("oa_url") or "").strip()
        if _looks_like_url(u):
            return u
    return ""


def _pdf_url_from_unpaywall(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    loc = data.get("best_oa_location")
    if isinstance(loc, dict):
        for k in ("url_for_pdf", "url"):
            u = str(loc.get(k) or "").strip()
            if _looks_like_url(u):
                return u
    return ""


def resolve_oa_pdf_url(
    doi_or_url: str,
    *,
    client=None,
    email: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    """Resolve a DOI to an open-access PDF URL via OpenAlex (then Unpaywall).

    Returns a candidate PDF URL or '' (never raises). The CALLER still runs the
    robots/paywall gate before downloading — this only *locates* a legal OA PDF.
    ``client`` is injectable for tests; OpenAlex requires no key.
    """
    doi = _normalize_doi(doi_or_url or "")
    if not doi:
        return ""
    if httpx is None and client is None:
        return ""
    mail = email if email is not None else _oa_email()
    own = False
    c = client
    try:
        if c is None:
            c = httpx.Client(  # type: ignore[union-attr]
                timeout=timeout, follow_redirects=True,
                headers={"User-Agent": _USER_AGENT})
            own = True
        # 1) OpenAlex — no key needed (optional polite mailto)
        try:
            url = _OPENALEX_WORK.format(doi=doi)
            if mail:
                url += f"?mailto={mail}"
            r = c.get(url)
            if getattr(r, "status_code", 0) == 200:
                u = _pdf_url_from_openalex(_json_of(r))
                if u:
                    return u
        except Exception as e:  # pragma: no cover - network best effort
            logger.debug("openalex oa resolve failed for %s: %s", doi, e)
        # 2) Unpaywall — needs an email
        if mail:
            try:
                r = c.get(_UNPAYWALL.format(doi=doi) + f"?email={mail}")
                if getattr(r, "status_code", 0) == 200:
                    u = _pdf_url_from_unpaywall(_json_of(r))
                    if u:
                        return u
            except Exception as e:  # pragma: no cover
                logger.debug("unpaywall oa resolve failed for %s: %s", doi, e)
        return ""
    finally:
        if own and c is not None:
            try:
                c.close()
            except Exception:
                pass


# ── Public API ───────────────────────────────────────────────────────
def try_fetch_fulltext(
    doi_or_url: str,
    *,
    oa_url: str = "",
    auto_oa: bool = False,
    timeout: float = _DEFAULT_TIMEOUT,
    client=None,
    robots_fetcher=None,
) -> dict[str, Any]:
    """Best-effort, ToS-respecting full-text fetch for a DOI or URL.

    Args:
        doi_or_url:    a DOI (``10.x/...`` or ``https://doi.org/...``) or a URL.
        oa_url:        an OPTIONAL known open-access PDF URL (e.g. from
                       Unpaywall / OpenAlex ``open_access.oa_url``). When given
                       and robots-allowed, it is tried first — this is the only
                       reliably-legal PDF source.
        auto_oa:       when True and no ``oa_url`` is given, resolve one from the
                       DOI via OpenAlex/Unpaywall before fetching (the GUI passes
                       True). Default False keeps the call purely driven by its
                       network inputs (and existing tests unaffected).
        timeout:       per-request timeout (seconds).
        client:        an OPTIONAL pre-built ``httpx.Client``-like object
                       (used by tests to monkeypatch the network). Must support
                       ``.get(url) -> response`` with ``.status_code``,
                       ``.headers``, ``.content``, ``.text``.
        robots_fetcher: OPTIONAL ``(robots_url) -> str|None`` to supply robots
                       bodies without a real request (tests).

    Returns a JSON-serializable dict (see module docstring). Never raises.

    Flow (§8.2):
        1. Resolve a candidate URL from the DOI/URL.
        2. Hard legality gate — robots.txt; refuse paywall/auth content.
        3. Retrieve only legitimately-accessible content (OA PDF / landing page).
        4/5/6. Success-pdf / success-metadata / failure-please-upload.
    """
    raw = (doi_or_url or "").strip()
    if not raw:
        return _result(
            "error",
            message="未提供 DOI 或 URL。请粘贴有效的 DOI / 链接，或改为手动上传。",
        )

    # Degrade if no HTTP stack and no injected client.
    if httpx is None and client is None:
        return _result(
            "unavailable",
            message=(
                "网络组件 httpx 不可用，无法自动获取。请手动下载 PDF 后上传。"
            ),
        )

    landing_url = _resolve_candidate_url(raw)
    if not landing_url:
        return _result(
            "error",
            message=(
                "无法识别为有效的 DOI 或 http(s) URL。请检查输入，或改为手动上传。"
            ),
        )

    slug = _slug_for(raw)
    doi_norm = _normalize_doi(raw)
    own_client = False
    c = client
    try:
        if c is None:
            c = httpx.Client(  # type: ignore[union-attr]
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            )
            own_client = True

        # ── 0) Auto-resolve an OA PDF URL from the DOI (OpenAlex/Unpaywall) ──
        # so a bare DOI has a real open-access target; the robots/paywall gate
        # below still vets whatever we resolve before downloading.
        if auto_oa and not oa_url and doi_norm:
            try:
                oa_url = resolve_oa_pdf_url(raw, client=c, timeout=timeout) or ""
            except Exception:
                oa_url = ""

        # ── 1) Prefer a known OA PDF URL when supplied & robots-allowed ──
        if oa_url:
            if not _robots_allows(oa_url, fetcher=robots_fetcher):
                logger.info("robots.txt disallows OA url %s — skipping", oa_url)
            else:
                pdf_res = _try_pdf(c, oa_url, slug, doi_norm)
                if pdf_res is not None:
                    return pdf_res

        # ── 2) Legality gate on the landing URL ──
        if not _robots_allows(landing_url, fetcher=robots_fetcher):
            return _result(
                "blocked",
                message=(
                    "目标站点的 robots.txt 不允许自动抓取。"
                    "请手动下载 PDF 后上传。"
                ),
                url=landing_url,
            )

        # ── 3) Fetch the landing page (metadata only; never bypass) ──
        try:
            resp = c.get(landing_url)
        except Exception as e:
            logger.info("landing fetch failed for %s: %s", landing_url, e)
            return _result(
                "error",
                message=(
                    "网络获取失败（超时/连接错误）。请稍后重试或改为手动上传。"
                ),
                url=landing_url,
            )

        status_code = getattr(resp, "status_code", 0)
        if status_code in _AUTH_STATUS:
            return _result(
                "blocked",
                message=(
                    "无法合法自动获取（付费墙 / 需登录 / ToS）。请手动下载后上传。"
                ),
                url=landing_url,
            )
        if status_code >= 400:
            return _result(
                "error",
                message=(
                    f"目标返回 HTTP {status_code}。请改为手动上传。"
                ),
                url=landing_url,
            )

        headers = _lower_headers(getattr(resp, "headers", {}) or {})
        ctype = headers.get("content-type", "")

        # Landing URL itself served a PDF (rare but possible for OA).
        if "application/pdf" in ctype:
            content = getattr(resp, "content", b"") or b""
            if _pdf_ok(content):
                return _save_and_result(content, slug, landing_url, doi_norm)

        body_text = _safe_text(resp)

        # Paywall / login wall detected → refuse (never bypass).
        if _looks_paywalled(body_text):
            return _result(
                "blocked",
                message=(
                    "检测到付费墙 / 登录墙，按 ToS 不予绕过。请手动下载后上传。"
                ),
                url=landing_url,
                metadata=_extract_landing_metadata(body_text, landing_url, doi_norm),
            )

        # ── 4) Look for an OA PDF link advertised on the landing page ──
        pdf_link = _find_pdf_link(body_text, landing_url)
        if pdf_link and _robots_allows(pdf_link, fetcher=robots_fetcher):
            pdf_res = _try_pdf(c, pdf_link, slug, doi_norm)
            if pdf_res is not None:
                return pdf_res

        # ── 5) Success: metadata only (no legally-reachable PDF) ──
        meta = _extract_landing_metadata(body_text, landing_url, doi_norm)
        if meta:
            return _result(
                "ok_metadata",
                message=(
                    "仅获取到元数据/摘要（未找到可合法获取的全文 PDF）。"
                    "如需全文请手动上传。"
                ),
                metadata=meta,
                url=landing_url,
            )

        # ── 6) Failure: nothing legally obtainable ──
        return _result(
            "unavailable",
            message=(
                "无法合法自动获取全文（无开放获取副本 / 付费墙 / ToS）。"
                "请手动下载后上传。"
            ),
            url=landing_url,
        )
    finally:
        if own_client and c is not None:
            try:
                c.close()
            except Exception:
                pass


# ── Internal retrieval helpers ───────────────────────────────────────
def _try_pdf(client, url: str, slug: str, doi_norm: str) -> Optional[dict[str, Any]]:
    """GET *url*, and if it is a valid PDF, save it and return an ok_pdf result.

    Returns ``None`` if it is not a usable PDF (caller falls through). On an
    auth/paywall status, returns a ``blocked`` result (never bypass).
    """
    try:
        resp = client.get(url)
    except Exception as e:
        logger.info("pdf fetch failed for %s: %s", url, e)
        return None

    status_code = getattr(resp, "status_code", 0)
    if status_code in _AUTH_STATUS:
        return _result(
            "blocked",
            message="该 PDF 需登录/付费，按 ToS 不予绕过。请手动下载后上传。",
            url=url,
        )
    if status_code >= 400:
        return None

    headers = _lower_headers(getattr(resp, "headers", {}) or {})
    ctype = headers.get("content-type", "")
    content = getattr(resp, "content", b"") or b""

    if "application/pdf" in ctype or _pdf_ok(content):
        if _pdf_ok(content):
            return _save_and_result(content, slug, url, doi_norm)
    return None


def _save_and_result(content: bytes, slug: str, url: str, doi_norm: str) -> dict[str, Any]:
    if len(content) > _MAX_PDF_BYTES:
        return _result(
            "error",
            message="PDF 超过大小上限，未保存。请手动处理。",
            url=url,
        )
    try:
        path = _save_pdf(content, slug)
    except Exception as e:
        logger.warning("failed to save fetched pdf: %s", e)
        return _result(
            "error",
            message="保存 PDF 失败。请改为手动上传。",
            url=url,
        )
    return _result(
        "ok_pdf",
        message="已合法获取开放获取全文 PDF。",
        pdf_path=path,
        metadata={"doi": doi_norm, "source_url": url} if doi_norm else {"source_url": url},
        url=url,
    )


def _pdf_ok(content: bytes) -> bool:
    """A minimal sanity check that *content* is actually a PDF."""
    return bool(content) and content[:5] == b"%PDF-" and len(content) <= _MAX_PDF_BYTES


def _lower_headers(headers) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        items = headers.items()
    except Exception:
        return out
    for k, v in items:
        try:
            out[str(k).lower()] = str(v).lower()
        except Exception:
            continue
    return out


def _safe_text(resp) -> str:
    try:
        t = getattr(resp, "text", "")
        return t if isinstance(t, str) else ""
    except Exception:
        return ""


# Citation/meta tags commonly used by publishers & OA repos.
_META_PATTERNS = {
    "title": (
        r'<meta[^>]+name=["\'](?:citation_title|dc\.title|og:title)["\'][^>]+'
        r'content=["\']([^"\']+)["\']'
    ),
    "doi": r'<meta[^>]+name=["\']citation_doi["\'][^>]+content=["\']([^"\']+)["\']',
    "journal": (
        r'<meta[^>]+name=["\']citation_journal_title["\'][^>]+'
        r'content=["\']([^"\']+)["\']'
    ),
    "year": (
        r'<meta[^>]+name=["\']citation_(?:publication_date|date|year)["\'][^>]+'
        r'content=["\'](\d{4})'
    ),
}
_PDF_META_RE = re.compile(
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _extract_landing_metadata(html: str, url: str, doi_norm: str) -> dict[str, Any]:
    """Pull abstract/metadata from a landing page's citation meta tags.

    Only reads what the page openly serves (no auth). Returns ``{}`` when there
    is nothing useful. Best-effort regex — no HTML parser dependency.
    """
    if not html:
        return {}
    meta: dict[str, Any] = {}
    for field, pattern in _META_PATTERNS.items():
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            meta[field] = m.group(1).strip()

    # citation_abstract / og:description as a lightweight abstract.
    m = re.search(
        r'<meta[^>]+name=["\'](?:citation_abstract|dc\.description|og:description)'
        r'["\'][^>]+content=["\']([^"\']{20,})["\']',
        html,
        re.IGNORECASE,
    )
    if m:
        meta["abstract"] = m.group(1).strip()

    if doi_norm and "doi" not in meta:
        meta["doi"] = doi_norm
    if meta:
        meta.setdefault("source_url", url)
    return meta


def _find_pdf_link(html: str, base_url: str) -> str:
    """Find an advertised OA PDF link on the landing page, absolutized.

    Prefers the ``citation_pdf_url`` meta tag (the standard OA signal). Returns
    '' if none. Only ``http``/``https`` links are returned.
    """
    if not html:
        return ""
    m = _PDF_META_RE.search(html)
    if m:
        link = m.group(1).strip()
        absolute = urljoin(base_url, link)
        if _looks_like_url(absolute):
            return absolute
    return ""


__all__ = ["try_fetch_fulltext", "resolve_oa_pdf_url"]
