"""Markdown 报告 → ``.docx``（**全部用 Word 内置样式**）。

为什么需要它（2026-07-29）
--------------------------

设计文档 §6 原本把 docx 列为非目标，理由是「HTML 自包含导出已覆盖『发给别人』的
需求」。那句话只对了一半：HTML 覆盖了**发给别人看**，但没覆盖**让别人改** ——
导师或合作者要在 Word 里用**修订和批注**，或者期刊要求 ``.docx`` 投稿，HTML 都办不到。
所以这不是重复功能，是一个真缺口。

为什么用内置样式，而不是自己写格式
----------------------------------

约束是使用 Word 内置样式。这条约束不是审美问题，是**可编辑性**问题：

* 收件人改一次 ``Heading 1`` 就能重排全文标题；硬编码在每个 run 上的字号改不动。
* 修订/审阅流程、导航窗格、自动目录、交叉引用**全都依赖内置样式**。用假标题
  （放大加粗的正文）做出来的文档，在 Word 里是一份没有结构的文档。
* 期刊模板的做法就是替换内置样式的定义 —— 只要正文挂的是内置样式，套模板是一次
  样式替换，而不是重排全文。

所以这里**从不设置 font/size/color**，只挂样式名。用到的全是 python-docx 默认模板
里就有的（已逐一核对存在性）：

===============  ==========================================================
Markdown         Word 内置样式
===============  ==========================================================
标题行           ``Title`` / ``Subtitle``
``#`` … ``####`` ``Heading 1`` … ``Heading 4``（更深的降到 4）
正文             ``Normal``
``**粗**``       ``Strong``（字符样式，不是 ``run.bold``）
``*斜*``         ``Emphasis``
`` `代码` ``     ``Macro Text Char`` —— Word 内置的等宽字符样式（Courier）
代码块           ``No Spacing`` 段落 + ``Macro Text Char`` 的 run
``- ``           ``List Bullet``
``1. ``          ``List Number``
``> ``           ``Quote``
表格             ``Table Grid``，表头单元格的 run 挂 ``Strong``
图片说明         ``Caption`` —— Word 的题注样式，能被「插入题注」体系接管
===============  ==========================================================

两个内置样式**不存在**，如实处理而不是伪造：

* **没有代码段落样式。** 默认模板只有 ``Macro Text Char``（字符级），没有对应的
  段落样式。所以代码块用 ``No Spacing``（内置，段前段后为 0，正好适合连续多行）
  承载，字符样式仍是内置的那个。
* **没有 ``Hyperlink`` 字符样式。** 所以链接**做成真的超链接**（关系 + ``w:hyperlink``），
  但 run 不挂任何字符样式 —— 可点击是实质，颜色是外观，Word 在用户编辑时会自己
  补上它的 Hyperlink 样式。宁可少一层外观，也不自己发明一个叫 ``Hyperlink`` 的
  样式去冒充内置的。

其余规则与 ``report_html.py`` 逐条对齐
--------------------------------------

同一份 markdown 子集、同一个渲染循环形状、同一条兜底规则：**认不出来的东西一律
当段落输出，降级样式、绝不丢内容**。一份丢了一句话的报告比一份表格长得朴素的报告
更糟。图片缺失、格式不支持都**写进文档正文**，因为「看着完整、实则少了数据」的报告
比报错更危险。
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

#: 超过这个大小的图片不嵌入 —— 与 ``report_html`` 同一口径，理由也一样：
#: 一份打不开的交付件不如一份少一张图但能打开的。
_MAX_IMAGE_BYTES = 12 * 1024 * 1024

#: python-docx 的 ``add_picture`` 只认这几类；``embed_figure`` 允许 PDF，所以
#: 必须有一条「嵌不进去就如实说」的路径。
_EMBEDDABLE = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".emf", ".wmf"}

_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_RE = re.compile(r"\*Caption:\*\s*(?P<c>.+)")

#: 行内标记的一次性扫描。顺序有意义：代码在最前，否则 ``**`` 会先吃掉 `` ` `` 里的星号。
_TOKEN_RE = re.compile(
    r"`(?P<code>[^`]+)`"
    r"|\*\*(?P<bold>.+?)\*\*"
    r"|(?<!\*)\*(?P<ital>[^*]+)\*(?!\*)"
    r"|(?<!!)\[(?P<txt>[^\]]+)\]\((?P<href>[^)]+)\)"
)

_STRONG, _EMPH, _CODE = "Strong", "Emphasis", "Macro Text Char"


def _style(doc, name: str, fallback: str = "Normal") -> str:
    """样式名存在就用它，否则退到 ``fallback``。

    这是「降级样式、绝不丢内容」在样式层的应用：某个模板（比如期刊模板）少一个
    样式时，段落照样写出来，只是长得朴素 —— 而不是抛异常把整份报告丢掉。
    """
    try:
        doc.styles[name]
        return name
    except KeyError:
        logger.debug("report_docx: 样式 %r 不在模板里，退到 %r", name, fallback)
        return fallback


def _run(paragraph, text: str, style: str | None = None):
    r = paragraph.add_run(text)
    if style:
        try:
            r.style = style
        except KeyError:  # 模板缺这个字符样式 —— 文字仍在，只是不带那层外观
            logger.debug("report_docx: 字符样式 %r 缺失，按正文输出", style)
    return r


def _add_hyperlink(paragraph, url: str, text: str) -> None:
    """插入一个**真的**超链接（外部关系 + ``w:hyperlink``）。

    python-docx 没有高层 API，所以这里落到 oxml。不挂字符样式：默认模板没有内置的
    ``Hyperlink``（已核对），而自己造一个同名样式就是在冒充内置。可点击是实质。
    """
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    try:
        r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    except Exception as exc:  # noqa: BLE001 — 链接建不了也不能丢文字
        logger.debug("report_docx: 超链接 %r 建立失败：%r", url, exc)
        _run(paragraph, f"{text}（{url}）")
        return
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    run.append(OxmlElement("w:rPr"))
    node = OxmlElement("w:t")
    node.text = text
    node.set(qn("xml:space"), "preserve")
    run.append(node)
    link.append(run)
    paragraph._p.append(link)


def _add_hr(doc) -> None:
    """一条横线。Word 没有「横线」样式，只有段落下边框 —— 所以这里落到 oxml，
    但仍然不碰字体，只加一条边框。"""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    p = doc.add_paragraph()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"),
                 ("w:color", "auto")):
        bottom.set(qn(k), v)
    borders.append(bottom)
    p._p.get_or_add_pPr().append(borders)


def _add_inline(paragraph, text: str) -> None:
    """把一行 markdown 拆成带样式的 run。识别不了的片段原样输出为正文 run。"""
    pos = 0
    for m in _TOKEN_RE.finditer(text):
        if m.start() > pos:
            _run(paragraph, text[pos:m.start()])
        if m.group("code") is not None:
            _run(paragraph, m.group("code"), _CODE)
        elif m.group("bold") is not None:
            _run(paragraph, m.group("bold"), _STRONG)
        elif m.group("ital") is not None:
            _run(paragraph, m.group("ital"), _EMPH)
        else:
            _add_hyperlink(paragraph, m.group("href"), m.group("txt"))
        pos = m.end()
    if pos < len(text):
        _run(paragraph, text[pos:])


def _usable_width(doc):
    """正文可用宽度 —— 图片不能宽过它，否则 Word 里会溢出到页面外。"""
    from docx.shared import Inches

    try:
        sec = doc.sections[0]
        w = sec.page_width - sec.left_margin - sec.right_margin
        return w if w and w > 0 else Inches(6.0)
    except Exception:  # noqa: BLE001
        return Inches(6.0)


def _row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render_docx(md: str, *, base_dir: Path, title: str = "实验报告",
                subtitle: str = "") -> tuple[bytes, int, int]:
    """把 *md* 渲染成一个 ``.docx``，返回 ``(bytes, 嵌入图数, 缺失图数)``。

    ``base_dir`` 是相对图片路径的解析基准 —— 传**版本文件所在目录**，与
    ``report_html.render_html`` 完全一致（文档里写的是 ``../_assets/x.png``，
    markdown 渲染器按文件自己的目录解析）。

    两个计数让调用方能**如实**汇报，而不是默默交付一份少了图的报告。
    """
    from docx import Document

    doc = Document()
    embedded = missing = 0
    max_w = _usable_width(doc)

    if title:
        doc.add_paragraph(title, style=_style(doc, "Title"))
    if subtitle:
        doc.add_paragraph(subtitle, style=_style(doc, "Subtitle"))

    lines = md.replace("\r\n", "\n").split("\n")
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        # 围栏代码块 —— No Spacing 段落 + Macro Text Char 的 run（两者都是内置）
        if s.startswith("```"):
            i += 1
            buf: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            for code_line in (buf or [""]):
                p = doc.add_paragraph(style=_style(doc, "No Spacing"))
                _run(p, code_line, _CODE)
            continue

        # 独占一行的图片 → 嵌图 + Caption 段落；紧随其后的 "*Caption:* …"
        # （embed_figure 产出的形状）折进题注
        m = _IMG_RE.fullmatch(s)
        if m:
            src, alt = m.group("src"), m.group("alt")
            cap = alt
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                cm = _CAPTION_RE.fullmatch(lines[j].strip())
                if cm:
                    cap = cm.group("c")
                    i = j
            ok, note = _add_picture(doc, src, base_dir, max_w)
            if ok:
                embedded += 1
                if cap:
                    p = doc.add_paragraph(style=_style(doc, "Caption"))
                    _add_inline(p, cap)
            else:
                # 写进正文。一张静默消失的图，正是「报告看着完整、实则少了数据」
                # 的来源。
                missing += 1
                doc.add_paragraph(note, style=_style(doc, "Normal"))
            i += 1
            continue

        hm = re.match(r"(#{1,6})\s+(.*)", s)
        if hm:
            lv = min(len(hm.group(1)), 4)          # 默认模板 Heading 1..4 已核实
            p = doc.add_paragraph(style=_style(doc, f"Heading {lv}"))
            _add_inline(p, hm.group(2))
            i += 1
            continue

        if re.fullmatch(r"(\*\s*){3,}|(-\s*){3,}|(_\s*){3,}", s):
            _add_hr(doc)
            i += 1
            continue

        # 表格：表头 + 分隔行
        if s.startswith("|") and i + 1 < n and re.fullmatch(
                r"\|[\s:|-]+\|?", lines[i + 1].strip()):
            head = _row(s)
            i += 2
            body: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                body.append(_row(lines[i].strip()))
                i += 1
            _add_table(doc, head, body)
            continue

        # 列表
        if re.match(r"[-*+]\s+", s) or re.match(r"\d+[.)]\s+", s):
            ordered = bool(re.match(r"\d+[.)]\s+", s))
            style = _style(doc, "List Number" if ordered else "List Bullet")
            while i < n:
                mm = re.match(r"(?:[-*+]|\d+[.)])\s+(.*)", lines[i].strip())
                if not mm:
                    break
                p = doc.add_paragraph(style=style)
                _add_inline(p, mm.group(1))
                i += 1
            continue

        if s.startswith(">"):
            quote: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip("> ").rstrip())
                i += 1
            p = doc.add_paragraph(style=_style(doc, "Quote"))
            _add_inline(p, " ".join(quote))
            continue

        # 段落 —— 认不出来的一切都落到这里
        para: list[str] = []
        while i < n and lines[i].strip() and not re.match(
                r"#{1,6}\s|```|\||[-*+]\s|\d+[.)]\s|>", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        text = " ".join(para) if para else s
        if not para:
            i += 1
        # 行内图片降级成一句可见的说明，而不是无声消失
        text = _IMG_RE.sub(lambda mm: f"[图：{mm.group('alt')}]", text)
        p = doc.add_paragraph(style=_style(doc, "Normal"))
        _add_inline(p, text)

    buf_io = io.BytesIO()
    doc.save(buf_io)
    return buf_io.getvalue(), embedded, missing


def _add_picture(doc, src: str, base_dir: Path, max_w) -> tuple[bool, str]:
    """嵌一张图。返回 ``(成功, 失败说明)``；**永不抛**。"""
    if src.startswith(("http://", "https://", "data:")):
        # 远程图不去联网抓：仪器机常常离线，而一次挂住的 HTTP 请求比一张缺图更糟。
        return False, f"[外部图片未嵌入：{src}]"
    path = (Path(base_dir) / src).resolve()
    try:
        if not path.is_file():
            return False, f"[图片缺失：{src}]"
        size = path.stat().st_size
        if size == 0:
            return False, f"[图片为空：{src}]"
        if size > _MAX_IMAGE_BYTES:
            return False, f"[图片过大未嵌入（{size // 1024 // 1024} MB）：{src}]"
        if path.suffix.lower() not in _EMBEDDABLE:
            # embed_figure 允许 PDF，而 Word 嵌不进 PDF —— 如实说，并给出路径，
            # 用户自己能插。
            return False, f"[Word 不支持嵌入 {path.suffix} 格式，原图在：{path}]"
        doc.add_picture(str(path), width=max_w)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_docx: 嵌图失败 %s: %r", path, exc)
        return False, f"[图片嵌入失败：{src}（{type(exc).__name__}）]"


def _add_table(doc, head: list[str], body: list[list[str]]) -> None:
    """``Table Grid``（内置）+ 表头 run 挂 ``Strong``（内置字符样式）。

    刻意不用 ``Light Grid Accent 1`` 那类带底纹的内置表格样式：``Table Grid`` 是最
    朴素、最好改的一个，而这份文档的用途是**被人改**。
    """
    cols = max(len(head), max((len(r) for r in body), default=0)) or 1
    table = doc.add_table(rows=1, cols=cols)
    try:
        table.style = doc.styles["Table Grid"]
    except KeyError:
        logger.debug("report_docx: Table Grid 缺失，用模板默认表格样式")
    # 不写 ``cell.text = ""``：新单元格本来就只有一个空段落、零个 run，赋空串反而
    # 会造出一个空 run，让「表头 run 都挂 Strong」这个断言里混进一堆 Default。
    for cell, text in zip(table.rows[0].cells, head + [""] * cols):
        _run(cell.paragraphs[0], text, _STRONG)
    for row in body:
        cells = table.add_row().cells
        for cell, text in zip(cells, row + [""] * cols):
            _add_inline(cell.paragraphs[0], text)
