"""Markdown report → ONE self-contained HTML file (images inlined as data URIs).

Why this exists
---------------
Reports are written as markdown with the figures living beside them in
``data/figures/``. That is a good WORKING format — editable in 对象编辑,
versioned, diffable — but a poor DELIVERABLE: send someone the .md and the
figures do not travel with it, and the reader needs a markdown renderer at all.

This produces a single .html that opens in any browser, survives being emailed
or dropped in a shared folder, and cannot lose its figures because they are
embedded in the file.

Why a hand-written renderer
---------------------------
No markdown library is installed and none is added: the input here is markdown
that MAST itself produced (draft_section / embed_figure), not arbitrary user
input, so the subset below covers it. A new dependency would also land in a
2.4 GB frozen build.

The rule for anything unrecognised is **emit it as a paragraph, escaped** —
degrade the styling, never drop the content. A report that loses a sentence is
worse than one with a plain-looking table.
"""
from __future__ import annotations

import base64
import html
import logging
import mimetypes
import re
from pathlib import Path

logger = logging.getLogger(__name__)

#: Skip absurdly large images rather than produce a browser-hostile file.
_MAX_INLINE_BYTES = 12 * 1024 * 1024

_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_LINK_RE = re.compile(r"(?<!!)\[(?P<txt>[^\]]+)\]\((?P<href>[^)]+)\)")
_BOLD_RE = re.compile(r"\*\*(?P<t>[^*]+)\*\*")
_ITAL_RE = re.compile(r"(?<!\*)\*(?P<t>[^*]+)\*(?!\*)")
_CODE_RE = re.compile(r"`(?P<t>[^`]+)`")

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 2.5rem 1.5rem 6rem; max-width: 52rem;
  font: 16px/1.75 -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC",
        "Hiragino Sans GB", sans-serif;
  color: #1a1a1a; background: #fff;
}
h1 { font-size: 1.9rem; margin: 0 0 .3em; line-height: 1.25; }
h2 { font-size: 1.35rem; margin: 2.2em 0 .5em; padding-bottom: .3em;
     border-bottom: 1px solid #e3e3e3; }
h3 { font-size: 1.1rem; margin: 1.8em 0 .4em; }
p  { margin: 0 0 1em; }
ul, ol { margin: 0 0 1em; padding-left: 1.6em; }
li { margin: .25em 0; }
code { font: .9em/1.5 ui-monospace, Consolas, "Courier New", monospace;
       background: #f2f2f2; padding: .12em .38em; border-radius: 3px; }
pre { background: #f7f7f7; border: 1px solid #e6e6e6; border-radius: 6px;
      padding: .9em 1.1em; overflow-x: auto; }
pre code { background: none; padding: 0; }
figure { margin: 1.8em 0; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #e6e6e6;
             border-radius: 4px; }
figcaption { margin-top: .6em; font-size: .9rem; color: #555; }
table { border-collapse: collapse; width: 100%; margin: 1.2em 0;
        font-size: .94rem; display: block; overflow-x: auto; }
th, td { border: 1px solid #ddd; padding: .45em .7em; text-align: left; }
th { background: #f5f5f5; font-weight: 600; }
blockquote { margin: 1em 0; padding: .1em 1em; border-left: 3px solid #ccc;
             color: #555; }
hr { border: 0; border-top: 1px solid #e3e3e3; margin: 2.5em 0; }
.meta { margin: .2em 0 2.5em; color: #666; font-size: .88rem; }
.missing { color: #a33; font-size: .9rem; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e6; background: #181818; }
  h2 { border-bottom-color: #333; }
  code { background: #262626; }
  pre { background: #202020; border-color: #333; }
  figure img { border-color: #333; }
  th { background: #232323; } th, td { border-color: #383838; }
  .meta, figcaption, blockquote { color: #a8a8a8; }
}
@media print {
  body { max-width: none; padding: 0; }
  figure { break-inside: avoid; }
  h2 { break-after: avoid; }
}
"""


def _data_uri(path: Path) -> str | None:
    """File → ``data:<mime>;base64,…``; None when unusable (never raises)."""
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
        if size == 0 or size > _MAX_INLINE_BYTES:
            logger.warning("report_html: skipping %s (%d bytes)", path, size)
            return None
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_html: cannot inline %s: %s", path, exc)
        return None


def _inline(text: str) -> str:
    """Escape, then apply inline markup. Escaping FIRST is what keeps a report
    that legitimately contains ``<`` or ``&`` from corrupting the document."""
    t = html.escape(text)
    t = _CODE_RE.sub(lambda m: f"<code>{m.group('t')}</code>", t)
    t = _BOLD_RE.sub(lambda m: f"<strong>{m.group('t')}</strong>", t)
    t = _ITAL_RE.sub(lambda m: f"<em>{m.group('t')}</em>", t)
    t = _LINK_RE.sub(_link, t)
    return t


#: 链接只放行 http(s)、mailto、页内锚点与相对路径。报告正文会来自外部 agent（交接报告），
#: ``[看这里](javascript:…)`` 在导出的 HTML 里点一下就会执行脚本。
_SAFE_HREF = re.compile(r"^(?:https?://|mailto:|#|\.{0,2}/|[^:/?#\s]+(?:[/?#]|$))", re.IGNORECASE)


def _link(m: re.Match) -> str:
    href = m.group("href")
    probe = re.sub(r"[\x00-\x20]", "", html.unescape(href))   # 浏览器会剥掉的空白与控制字符
    if not _SAFE_HREF.match(probe):
        return m.group("txt")
    return f'<a href="{href}">{m.group("txt")}</a>'


def _row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render_html(md: str, *, base_dir: Path, title: str = "实验报告",
                subtitle: str = "") -> tuple[str, int, int]:
    """Render *md* to a standalone HTML document.

    ``base_dir`` is what relative image paths resolve against — normally the
    directory the draft lives in, matching how a markdown renderer would read
    it. Returns ``(html, images_inlined, images_missing)``; the counts let the
    caller report honestly instead of silently shipping a figure-less report.
    """
    out: list[str] = []
    inlined = missing = 0
    lines = md.replace("\r\n", "\n").split("\n")
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        # fenced code
        if s.startswith("```"):
            i += 1
            buf: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        # image on its own line → <figure>; a caption line right after it
        # ("*Caption:* …", the shape embed_figure emits) folds into figcaption.
        m = _IMG_RE.fullmatch(s)
        if m:
            src, alt = m.group("src"), m.group("alt")
            cap = alt
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                nxt = lines[j].strip()
                cm = re.fullmatch(r"\*Caption:\*\s*(?P<c>.+)", nxt)
                if cm:
                    cap = cm.group("c")
                    i = j
            uri = None
            if not src.startswith(("http://", "https://", "data:")):
                uri = _data_uri((base_dir / src).resolve())
            if uri:
                inlined += 1
                out.append(
                    f'<figure><img src="{uri}" alt="{html.escape(alt)}">'
                    f"<figcaption>{_inline(cap)}</figcaption></figure>")
            elif src.startswith(("http://", "https://")):
                out.append(
                    f'<figure><img src="{html.escape(src)}" '
                    f'alt="{html.escape(alt)}">'
                    f"<figcaption>{_inline(cap)}</figcaption></figure>")
            else:
                # Say so in the document. A silently absent figure is how a
                # report ends up looking complete while missing its data.
                missing += 1
                out.append(
                    f'<p class="missing">[图片缺失：{html.escape(src)}]</p>')
            i += 1
            continue

        # heading
        hm = re.match(r"(#{1,6})\s+(.*)", s)
        if hm:
            lv = len(hm.group(1))
            out.append(f"<h{lv}>{_inline(hm.group(2))}</h{lv}>")
            i += 1
            continue

        if re.fullmatch(r"(\*\s*){3,}|(-\s*){3,}|(_\s*){3,}", s):
            out.append("<hr>")
            i += 1
            continue

        # table: header row + separator
        if s.startswith("|") and i + 1 < n and re.fullmatch(
                r"\|[\s:|-]+\|?", lines[i + 1].strip()):
            head = _row(s)
            i += 2
            body: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                body.append(_row(lines[i].strip()))
                i += 1
            th = "".join(f"<th>{_inline(c)}</th>" for c in head)
            trs = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                for r in body)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
            continue

        # lists
        if re.match(r"[-*+]\s+", s) or re.match(r"\d+[.)]\s+", s):
            ordered = bool(re.match(r"\d+[.)]\s+", s))
            items: list[str] = []
            while i < n:
                t = lines[i].strip()
                mm = re.match(r"(?:[-*+]|\d+[.)])\s+(.*)", t)
                if not mm:
                    break
                items.append(f"<li>{_inline(mm.group(1))}</li>")
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue

        if s.startswith(">"):
            quote: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote><p>" + _inline(" ".join(quote)) + "</p></blockquote>")
            continue

        # paragraph — the fallback for everything unrecognised
        para: list[str] = []
        while i < n and lines[i].strip() and not re.match(
                r"#{1,6}\s|```|\||[-*+]\s|\d+[.)]\s|>", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        if para:
            body_txt = _inline(" ".join(para))
            body_txt = _IMG_RE.sub(
                lambda m: f"[图：{html.escape(m.group('alt'))}]", body_txt)
            out.append(f"<p>{body_txt}</p>")
        else:
            out.append(f"<p>{_inline(s)}</p>")
            i += 1

    sub = f'<p class="meta">{html.escape(subtitle)}</p>' if subtitle else ""
    doc = (
        "<!doctype html>\n"
        '<html lang="zh-CN"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        f"{sub}\n" + "\n".join(out) + "\n</body></html>\n"
    )
    return doc, inlined, missing


__all__ = ["render_html"]
