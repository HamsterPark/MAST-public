"""A report you can hand to someone: one file, figures inside it.

The requirement: papers and reports need figures inline, so the deliverable format is HTML.

The working format stays markdown (editable in 对象编辑, versioned, diffable).
What was missing was a DELIVERABLE: send someone the .md and the figures do not
travel with it, and they need a markdown renderer to see anything at all.

Two things this pins:

* **Self-containment.** Zero external references — no http(s), no file paths.
  A single .html that opens anywhere. If an ``<img src="…">`` ever points
  outside the document again, the export has quietly stopped being shareable.
* **Honest degradation.** A figure that cannot be inlined is marked IN THE
  DOCUMENT, and anything the renderer does not understand is emitted as escaped
  text rather than dropped. A report that looks complete while missing a
  measurement is worse than an ugly one.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_report_html_export.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import base64  # noqa: E402
import re  # noqa: E402

import pytest  # noqa: E402

from tests.v2.toolcall import tool_call
from mast.agents._shared.report_html import render_html  # noqa: E402

#: Smallest valid PNG (1×1). Enough to prove the inlining path.
_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    b"IQAAAABJRU5ErkJggg==")


@pytest.fixture()
def figdir(tmp_path):
    d = tmp_path / "figures"
    d.mkdir()
    (d / "topo.png").write_bytes(_PNG)
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    return drafts, d


# ════════════════════════════════════════════════════════════════════════════
# Self-containment — the whole point
# ════════════════════════════════════════════════════════════════════════════

def test_no_external_references_at_all(figdir):
    drafts, _ = figdir
    md = "# 报告\n\n![形貌](../figures/topo.png)\n\n*Caption:* 5 nm 视场\n"
    doc, inlined, missing = render_html(md, base_dir=drafts)
    assert inlined == 1 and missing == 0
    assert not re.search(r'(?:src|href)="https?://', doc), "外部 URL"
    assert not re.search(r'src="(?!data:)', doc), "图片指向外部文件 — 发出去就丢图"


def test_the_image_bytes_are_really_in_the_file(figdir):
    drafts, _ = figdir
    doc, _i, _m = render_html("![t](../figures/topo.png)\n", base_dir=drafts)
    m = re.search(r'src="data:image/png;base64,([^"]+)"', doc)
    assert m, "no inlined png"
    assert base64.b64decode(m.group(1)) == _PNG, "内嵌的不是原图字节"


def test_it_is_a_complete_html_document(figdir):
    drafts, _ = figdir
    doc, _i, _m = render_html("# T\n\n正文\n", base_dir=drafts, title="实验报告")
    assert doc.startswith("<!doctype html>")
    assert "<style>" in doc, "样式必须内联,否则换台机器就没格式"
    assert 'charset="utf-8"' in doc, "缺 charset — 中文会乱码"
    assert "<title>实验报告</title>" in doc


# ════════════════════════════════════════════════════════════════════════════
# Honest degradation
# ════════════════════════════════════════════════════════════════════════════

def test_a_missing_figure_is_declared_not_hidden(figdir):
    """The failure that matters: a report that reads as complete but lost a
    measurement."""
    drafts, _ = figdir
    doc, inlined, missing = render_html(
        "![丢了](../figures/nope.png)\n", base_dir=drafts)
    assert (inlined, missing) == (0, 1)
    assert "图片缺失" in doc and "nope.png" in doc


def test_unknown_markup_is_kept_as_text(figdir):
    """Degrade the styling, never drop the content."""
    drafts, _ = figdir
    doc, _i, _m = render_html("::: 某种未知语法 :::\n\n真实结论在这里\n",
                              base_dir=drafts)
    assert "真实结论在这里" in doc
    assert "某种未知语法" in doc


def test_html_in_the_report_is_escaped(figdir):
    """A report legitimately containing < or & must not corrupt the document."""
    drafts, _ = figdir
    doc, _i, _m = render_html("电流 <1 pA & 偏压 >0\n", base_dir=drafts)
    assert "&lt;1 pA" in doc and "&amp;" in doc
    assert "<1 pA" not in doc


@pytest.mark.parametrize("href", [
    "javascript:alert(1)", "JaVaScRiPt:alert(1)", " javascript:alert(1)",
    "java\tscript:alert(1)", "data:text/html,<b>x</b>", "vbscript:msgbox(1)",
])
def test_a_script_link_is_not_emitted_as_a_link(figdir, href):
    """报告正文会来自外部 agent（交接报告）：导出的 HTML 里点一下链接不许执行脚本。"""
    drafts, _ = figdir
    doc, _i, _m = render_html(f"[看这里]({href})\n", base_dir=drafts)
    assert "看这里" in doc
    assert "<a " not in doc, doc[-400:]


@pytest.mark.parametrize("href", ["https://example.org/x", "http://127.0.0.1:7862/",
                                  "mailto:a@example.org", "#sec", "notes/x.md", "../y.md"])
def test_ordinary_links_stay_links(figdir, href):
    drafts, _ = figdir
    doc, _i, _m = render_html(f"[链接]({href})\n", base_dir=drafts)
    assert '<a href="' in doc


def test_oversized_image_is_skipped_not_embedded(figdir, monkeypatch):
    """A 100 MB data URI would make the file unusable; say so instead."""
    import mast.agents._shared.report_html as R
    drafts, fd = figdir
    monkeypatch.setattr(R, "_MAX_INLINE_BYTES", 4)
    doc, inlined, missing = render_html("![big](../figures/topo.png)\n",
                                        base_dir=drafts)
    assert inlined == 0 and missing == 1
    assert "图片缺失" in doc


# ════════════════════════════════════════════════════════════════════════════
# Markdown subset actually used by the report writer
# ════════════════════════════════════════════════════════════════════════════

def test_headings_lists_table_and_emphasis(figdir):
    drafts, _ = figdir
    md = ("# 标题\n\n## 方法\n\n偏压 **0.5 V**,setpoint `1e-10 A`。\n\n"
          "- 第一条\n- 第二条\n\n| 参数 | 值 |\n|---|---|\n| 视场 | 5 nm |\n")
    doc, _i, _m = render_html(md, base_dir=drafts)
    assert "<h1>标题</h1>" in doc and "<h2>方法</h2>" in doc
    assert "<strong>0.5 V</strong>" in doc and "<code>1e-10 A</code>" in doc
    assert doc.count("<li>") == 2
    assert "<th>参数</th>" in doc and "<td>5 nm</td>" in doc


def test_caption_line_folds_into_the_figure(figdir):
    """embed_figure emits '*Caption:* …' on its own line right after the image;
    it belongs to the figure, not to a stray paragraph."""
    drafts, _ = figdir
    doc, _i, _m = render_html(
        "![形貌](../figures/topo.png)\n\n*Caption:* 5 nm 视场\n", base_dir=drafts)
    assert "<figcaption>5 nm 视场</figcaption>" in doc
    assert doc.count("<figure>") == 1


def test_code_fence_is_not_interpreted(figdir):
    drafts, _ = figdir
    doc, _i, _m = render_html("```\n# 这不是标题\n```\n", base_dir=drafts)
    assert "<pre><code>" in doc
    assert "<h1>" not in doc


# ════════════════════════════════════════════════════════════════════════════
# The tool wrapper
# ════════════════════════════════════════════════════════════════════════════

def _embed(tmp_path, name: str = "topo.png") -> str:
    """Put a real figure in the experiment pool via the tool that owns that job,
    and return the markdown link it emitted. Hand-writing ``../_assets/x.png``
    here would let the export and the embed drift apart without a test noticing —
    the exact failure that made every figure in every report a broken image."""
    from mast.agents.paper_writing.tools import embed_figure

    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / name).write_bytes(_PNG)
    out = embed_figure.invoke({"scan_path": str(src / name), "caption": "形貌"})
    return out.split("](")[1].split(")")[0]


def test_export_tool_writes_one_file(tmp_path, documents_root):
    from mast.agents.paper_writing.tools import export_report_html, save_draft

    link = _embed(tmp_path)
    save_draft.invoke(tool_call(save_draft, {"title": "Rep", "markdown_text": f"# 报告\n\n![形貌]({link})\n"}))

    out = export_report_html.invoke({"doc_id": "current"})
    assert "已导出" in out and "内嵌图片 1 张" in out
    files = list(documents_root.rglob("*.html"))
    assert len(files) == 1 and files[0].stat().st_size > len(_PNG)
    # Exports live beside the experiment, timestamped so re-exporting keeps the
    # previous deliverable (the old fixed name overwrote it).
    assert files[0].parent.name == "exports"
    assert "_v001_" in files[0].name


def test_two_exports_of_the_same_document_coexist(tmp_path, documents_root):
    """The old export wrote a FIXED name under data/reports and overwrote it, so
    re-exporting destroyed a deliverable that may already have been handed to
    someone. The name now carries a timestamp — but the timestamp is only to the
    second, so two exports in the SAME second must still coexist. No clock
    trickery here on purpose: back-to-back is the case that actually broke."""
    from mast.agents.paper_writing.tools import export_report_html, save_draft

    save_draft.invoke(tool_call(save_draft, {"title": "Rep", "markdown_text": "# 报告\n\n正文\n"}))
    export_report_html.invoke({"doc_id": "current"})
    export_report_html.invoke({"doc_id": "current"})
    files = sorted(p.name for p in documents_root.rglob("*.html"))
    assert len(files) == 2, f"the second export overwrote the first: {files}"


def test_export_reports_missing_figures_to_the_caller(tmp_path, documents_root):
    from mast.agents.paper_writing.tools import export_report_html, save_draft

    save_draft.invoke(tool_call(save_draft, {"title": "Rep",
                       "markdown_text": "![x](../_assets/gone.png)\n"}))
    out = export_report_html.invoke({"doc_id": "current"})
    assert "找不到" in out or "缺失" in out, (
        "导出静默丢了图 — 调用方会以为报告是完整的")


def test_export_without_any_draft_says_so(tmp_path, documents_root):
    from mast.agents.paper_writing.tools import export_report_html
    out = export_report_html.invoke({"doc_id": "current"})
    assert "save_draft" in out, "没告诉调用方下一步该做什么"


def test_a_figure_embedded_by_the_tool_survives_the_export(tmp_path,
                                                           documents_root):
    """End to end on the link that used to break: embed_figure writes the pool
    entry and the link, save_draft stores the document one directory below the
    pool, and the export must resolve ``../_assets/`` from the VERSION FILE's own
    directory. Get the base_dir wrong and this silently reports 0 inlined."""
    from mast.agents.paper_writing.tools import export_report_html, save_draft

    link = _embed(tmp_path, "au111.png")
    save_draft.invoke(tool_call(save_draft, {"title": "有图的报告",
                       "markdown_text": f"# T\n\n![形貌]({link})\n"}))
    out = export_report_html.invoke({"doc_id": "current"})
    assert "内嵌图片 1 张" in out, f"图没被内嵌进交付件：{out}"
    doc = next(documents_root.rglob("*.html")).read_text(encoding="utf-8")
    assert base64.b64encode(_PNG).decode() in doc


def test_export_is_in_the_agents_toolbox():
    from mast.agents.paper_writing.tools import AGENT_TOOLS
    assert "export_report_html" in {t.name for t in AGENT_TOOLS}, (
        "工具存在但没挂上 — 这个仓库最常见的缺陷形状")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
