"""把 MAST 的「给人读的那一半」编成一本自包含的 HTML 电子书。

## 收哪些内容

1. **STM 操作实验指南** —— `docs/v2/stm_knowledge/`，由 Claude 在真机验收里口述
   整理的十篇。**每条事实都标了来源**（【通识】/【本机】/【操作员】/【存疑】），
   这本书把那套标记原样保留**并按四档上色**（`_colorize_marks`）——
   指南自己说「来源比内容重要」，那就得一眼分得开，【存疑】最刺眼。
2. **九个 agent 的系统提示词** —— 这是模型每一次调用真正读到的那份文本，不是
   它的说明书。放进来是为了让人能**逐句校对它到底被告知了什么**。
3. **知识库** —— agent 用查询工具取得到的东西。**按用途分两档**：操作员会去查、
   也会去改的运行规程（工作流序列 / 质量标准 / 验收判据 / 常见假象 / 诊断阈值）
   **全量收**；材料库、故障库、文献索引**只给索引**（有什么、多少条、哪个工具
   取）—— 那是约 150 种材料的参数表，全量抄进来会让这本书翻倍。
4. **上下文注入矩阵** —— 哪一块内容送给哪个角色。
5. **全部仪器技能** —— 按「工具包」分组（与按需加载的分包同一套），每个技能带
   描述、参数、安全等级、复合层级。**每个技能只出现一次**（落一个主包，其余
   归属写在标题旁）—— 按包重复列会让 `id` 撞车、锚点失效。

## 为校对准备的两件事

* 搜索框同时过滤**目录与技能条目**；
* **「只看待译」**开关 —— 待译条目散在 13 个包里，没有它得一个包一个包翻。
  「待译」只标**英文散文**：整条是 Nanonis 字面取值的（`0 = X/Y，1 = R/phi`）
  本来就该是英文，标它只会让人去追不该改的东西（判据见 `needs_translation`）。

## 为什么是一个单文件

它要能拷到 U 盘、发给人、离线打开、Ctrl+F 全文搜。任何外部依赖（CDN 的字体、
JS 库）在没网的机器上都会静默降级成一堆无样式的文字。所以：CSS 与 JS 全部内联，
零外部请求。

## 重跑

内容会变（技能描述正在中文化、提示词在改），所以这个脚本设计成**随时可重跑**：
它每次都从仓库现状重新生成，不做增量。

    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/build_handbook.py
    # → docs/handbook/MAST手册.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "MASTv2"))
sys.path.insert(0, str(_REPO))

CJK = re.compile(r"[一-鿿]")

# ── 「待译」该标谁 ────────────────────────────────────────────────────────
# 只有**英文散文**算待译。整条就是 Nanonis 字面取值的（`0 = X/Y，1 = R/phi`、
# `bright | dark | auto`）本来就该是英文 —— 标成待译会让校对的人去追一堆不该
# 改的东西，也会让进度分母是假的。
#: 分隔符：空白、数字、各种中英标点。用 chr() 拼是为了避开转义。
_SEP = set(" " + chr(9) + chr(10)
           + "0123456789=|/,，、.。:：;；-+*()（）[]{}'\"<>%^_")
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")
#: 出现这些虚词就是句子，不是取值清单。
_PROSE_WORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "if", "when", "set", "get", "returns", "return", "value", "use", "used",
    "must", "should", "will", "not", "only", "this", "that", "each", "from",
    "before", "after", "than", "then", "is", "are", "be", "can", "may",
}


def needs_translation(text: str, label: str = "") -> bool:
    """这条是不是**还没译**（整条英文散文，或中文里夹着成句英文）。

    ⚠️ **只问「有没有中文」是不够的** —— 这一版之前就是那样，漏掉三类：

    * 中文译文后面挂着一截没替换掉的英文（批量替换少截一行）；
    * 「英文第一句 + 中文其余」的半译；
    * 整条英文、只因句中嵌了一句判据就被判成已译
      （`AssessAtomicLines` 就是这么漏的）。

    混排那一侧的判据与 ``check_mixed_language.py`` **共用**，不另立一份 ——
    两份判据一定会漂。
    """
    t = (text or "").strip()
    if not t:
        return False
    if CJK.search(t):
        # 有中文 ⇒ 只剩「里面还夹着成句英文吗」这一问
        return _mixed_language_leftover(t, label)
    words = [w.lower() for w in _IDENT.findall(t)]
    if not words:
        return False                       # 纯符号/纯数字
    if any(w in _PROSE_WORDS for w in words):
        return True                        # 有虚词 → 是句子
    # 剥掉标识符后只剩分隔符 ⇒ 是取值清单，不该翻
    return not all(ch in _SEP for ch in _IDENT.sub("", t))


def _mixed_language_leftover(text: str, label: str) -> bool:
    """中文字段里还夹着成句英文吗（Nanonis 字面名与厂商引证除外）。"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_cml", Path(__file__).with_name("check_mixed_language.py"))
        mod = sys.modules.get("_cml")
        if mod is None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_cml"] = mod
            spec.loader.exec_module(mod)
    except Exception:                                            # noqa: BLE001
        return False                       # 取不到判据就不标，别造假红
    if label and label in mod.KNOWN_OK:
        return False
    return bool(mod.ENGLISH_RUN.search(text))



# ════════════════════════════════════════════════════════════════════════
# 取材
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Section:
    slug: str
    title: str
    body_html: str
    source: str = ""
    subsections: tuple = ()


def _md(text: str) -> str:
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False, "linkify": False})
    md.enable("table")
    md.enable("strikethrough")
    return md.render(text)


#: 来源标记 → CSS 类。知识库说明页：**来源比内容重要**。
#: 复合标记（`【本机 + 通识】`、`【通识 + 这一天的教训】`）按**先命中的那一档**
#: 上色，顺序即优先级：存疑 > 操作员 > 本机 > 通识 —— 最该让人停下来的排最前。
_MARK_CLASS = [
    ("存疑", "m-doubt"),
    ("操作员", "m-operator"),
    ("本机", "m-machine"),
    ("通识", "m-common"),
]
_MARK_RE = re.compile(r"【([^】]{1,40})】")


def _colorize_marks(html_text: str) -> str:
    """给渲染好的 HTML 里的来源标记上色。

    **必须在 markdown 渲染之后调**：渲染前插 HTML 会被 ``html:False`` 的
    MarkdownIt 整段转义成字面标签。
    """
    def one(m):
        inner = m.group(1)
        for key, cls in _MARK_CLASS:
            if key in inner:
                return (f"<span class='mark {cls}'>【{inner}】</span>")
        return m.group(0)

    return _MARK_RE.sub(one, html_text)


def _slug(text: str) -> str:
    s = re.sub(r"[^\w一-鿿-]+", "-", text.strip()).strip("-")
    return s[:60] or "x"


# ── 1. STM 操作知识库 ────────────────────────────────────────────────────

def collect_stm_guide() -> list[Section]:
    base = _REPO / "docs" / "v2" / "stm_knowledge"
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        first = raw.splitlines()[0] if raw.splitlines() else p.stem
        title = first.lstrip("# ").strip() or p.stem
        # 去掉正文里的一级标题（章节标题已经在导航里）
        body = re.sub(r"^#\s+.*\n", "", raw, count=1)
        out.append(Section(slug=f"stm-{_slug(p.stem)}", title=title,
                           body_html=_colorize_marks(_md(body)),
                           source=f"docs/v2/stm_knowledge/{p.name}"))
    return out


# ── 2. agent 系统提示词 ──────────────────────────────────────────────────

_AGENT_ORDER = [
    ("orchestrator", "编排器 · 路由提示词"),
    ("research_director", "科研策划 RD"),
    ("literature", "文献 LIT"),
    ("experiment_design", "实验设计 XD"),
    ("instrument_control", "仪器控制 IC"),
    ("data_processing", "数据处理 DP"),
    ("paper_writing", "论文写作 PW"),
    ("paper_review", "论文审稿 PR"),
    ("buffer_summarizer", "缓冲区摘要 BUF"),
]


def _agent_prompt(agent: str) -> tuple[str, str]:
    """``(正文, 出处)``。"""
    if agent == "orchestrator":
        from mast.agents.orchestrator.graph import _ROUTER_PROMPT
        return _ROUTER_PROMPT, "MASTv2/mast/agents/orchestrator/graph.py:_ROUTER_PROMPT"
    mod = __import__(f"mast.agents.{agent}.prompts", fromlist=["SYSTEM_PROMPT"])
    return (getattr(mod, "SYSTEM_PROMPT", ""),
            f"MASTv2/mast/agents/{agent}/prompts.py:SYSTEM_PROMPT")


def collect_agents() -> list[Section]:
    out = []
    for agent, label in _AGENT_ORDER:
        try:
            text, src = _agent_prompt(agent)
        except Exception as exc:  # noqa: BLE001
            out.append(Section(slug=f"agent-{agent}", title=label,
                               body_html=f"<p class='warn'>读不到：{html.escape(str(exc))}</p>"))
            continue
        if not text:
            continue
        lines = text.splitlines()
        zh = sum(1 for ln in lines if CJK.search(ln))
        en = sum(1 for ln in lines if ln.strip() and not CJK.search(ln))
        meta = (f"<p class='meta'>{len(text):,} 字符 · 中文行 {zh} / 英文行 {en}"
                f" · 出处 <code>{html.escape(src)}</code></p>")
        # note 里的 ** 要真的变粗体，所以走 _md()；但 _md 是 html:False，
        # 外层 <p> 若一起喂进去会被转义成字面文本（曾经就是这样）。
        note = _md("下面是**模型每一次调用真正读到的那份文本**，不是它的说明书。"
                   "校对时请逐句读：这里写错一句，仪器就会照着做。")
        body = (meta + f"<div class='note'>{note}</div>"
                + "<pre class='prompt'>" + html.escape(text) + "</pre>")
        out.append(Section(slug=f"agent-{agent}", title=label, body_html=body,
                           source=src))
    return out


# ── 3. 全部仪器技能 ──────────────────────────────────────────────────────

_SAFETY_LABEL = {"auto": "AUTO 自动", "confirm": "CONFIRM 留痕",
                 "dangerous": "DANGEROUS 危险"}
_LEVEL_LABEL = {
    0: "L0 原子（1:1 Nanonis TCP）", 1: "L1 短序列", 2: "L2 纯数据/分析",
    3: "L3 多步硬件流程", 4: "L4 长自治流程", 5: "L5+ 通宵战役",
}


def collect_skills() -> tuple[list[Section], dict]:
    import importlib

    importlib.import_module("tests.v2.conftest")   # mock nanonis_spm
    from mast.agents._shared import tool_packs as tp
    from mast.agents.instrument_control.tools import discover_instrument_skills

    reg = discover_instrument_skills()
    metas = {m.name: m for m in reg.list_skills()}

    # ── 每个技能只列一次 ────────────────────────────────────────────────
    # 一个技能可以属于多个包。按包逐个列会让它出现 N 次：457 个技能曾产出
    # 613 条，`id` 撞车、锚点失效，校对时同一段文字要读三遍。
    # 所以落**一个主包**（核心优先，其余按包的声明顺序），全部归属放到标题旁。
    _pack_rank = {tp.CORE: -1}
    _pack_rank.update({pk.name: i for i, pk in enumerate(tp.PACKS)})

    by_pack: dict[str, list] = {}
    packs_of: dict[str, list[str]] = {}
    for name, m in metas.items():
        where = sorted(tp.classify(name, m),
                       key=lambda x: _pack_rank.get(x, 999))
        packs_of[name] = where
        by_pack.setdefault(where[0], []).append(m)

    # 「到位」= 已译成中文，或本来就该是英文的取值清单。
    stats = {
        "total": len(metas),
        "zh_desc": sum(1 for m in metas.values()
                       if not needs_translation(m.description or "", f"{m.name}.description")),
        "params": sum(len(m.parameters or []) for m in metas.values()),
        "zh_params": sum(1 for m in metas.values() for p in (m.parameters or [])
                         if not needs_translation(p.description or "", f"{m.name}.{p.name}")),
        "todo_desc": sum(1 for m in metas.values()
                         if needs_translation(m.description or "", f"{m.name}.description")),
        "todo_params": sum(1 for m in metas.values() for p in (m.parameters or [])
                           if needs_translation(p.description or "", f"{m.name}.{p.name}")),
    }

    order = [tp.CORE] + [p.name for p in tp.PACKS]
    labels = {tp.CORE: "核心（每次调用都可见）"}
    labels.update({p.name: f"{p.name} · {p.label}" for p in tp.PACKS})

    out = []
    for pack in order:
        items = by_pack.get(pack)
        if not items:
            continue
        items.sort(key=lambda m: (-getattr(m, "composition_level", 0), m.name))
        rows = []
        for m in items:
            lvl = getattr(m, "composition_level", 0)
            sl = getattr(getattr(m, "safety_level", None), "value", "")
            desc = (m.description or "").strip()
            untranslated = (" <span class='tag-en'>待译</span>"
                            if needs_translation(desc, f"{m.name}.description") else "")
            params = ""
            if m.parameters:
                prows = "".join(
                    "<tr><td><code>{n}</code></td><td>{u}</td><td>{r}</td>"
                    "<td>{d}</td></tr>".format(
                        n=html.escape(p.name),
                        u=html.escape(p.unit or "—"),
                        r=html.escape(_range_of(p)),
                        d=(html.escape((p.description or "").strip() or "—")
                           + (" <span class='tag-en'>待译</span>"
                              if needs_translation(p.description or "", f"{m.name}.{p.name}") else "")))
                    for p in m.parameters)
                params = ("<table class='params'><thead><tr><th>参数</th>"
                          "<th>单位</th><th>范围</th><th>说明</th></tr></thead>"
                          f"<tbody>{prows}</tbody></table>")
            tags = " ".join(f"<span class='tag'>{html.escape(t)}</span>"
                            for t in (getattr(m, "tags", None) or [])[:8])
            # 它同时还属于哪些包 —— 只列一次，但归属信息不能丢
            also = [x for x in packs_of.get(m.name, []) if x != pack]
            also_html = ("<span class='also'>也在 "
                         + "、".join(html.escape(x) for x in also)
                         + "</span>") if also else ""
            rows.append(
                f"<article class='skill' id='skill-{html.escape(m.name)}'>"
                f"<h4>{html.escape(m.name)}{untranslated}"
                f"<span class='badges'>"
                f"<span class='b-{sl}'>{html.escape(_SAFETY_LABEL.get(sl, sl))}</span>"
                f"<span class='b-lvl'>{html.escape(_LEVEL_LABEL.get(lvl, f'L{lvl}'))}</span>"
                f"{also_html}"
                f"</span></h4>"
                f"<p class='desc'>{html.escape(desc) or '—'}</p>"
                f"{params}<p class='tags'>{tags}</p></article>")
        head = (f"<p class='meta'>{len(items)} 个技能"
                f"<span class='hitcount'></span></p>")
        out.append(Section(slug=f"pack-{_slug(pack)}",
                           title=labels.get(pack, pack),
                           body_html=head + "".join(rows)))
    return out, stats


def _range_of(p) -> str:
    lo, hi = getattr(p, "min_value", None), getattr(p, "max_value", None)
    if lo is None and hi is None:
        allowed = getattr(p, "allowed_values", None)
        return " / ".join(str(a) for a in allowed) if allowed else "—"
    return f"{'' if lo is None else lo} … {'' if hi is None else hi}"


# ── 3.5 知识库（agent 查得到的）───────────────────────────────────────────

#: 全量收的运行规程：模块属性名 → (标题, 一句话说明它回答什么问题)。
#: 这些是**人**会去查、也会去改的东西。
_KB_OPERATIONAL = [
    ("experiment_design", "WORKFLOW_SEQUENCES", "工作流序列",
     "一次实验从微米走到原子分辨的分级协议，每级带扫描尺寸/偏压/电流/"
     "go-nogo 判据"),
    ("experiment_design", "QUALITY_STANDARDS", "质量标准",
     "什么样的图/谱算合格"),
    ("experiment_design", "VERIFICATION_CRITERIA", "验收判据",
     "怎么确认「真的做到了」而不是「看起来像」"),
    ("experiment_design", "STM_ARTIFACTS", "常见假象",
     "双针尖、漂移、行噪…… 以及怎么和真实的样品特征区分"),
    ("experiment_design", "DIAGNOSTIC_THRESHOLDS", "诊断阈值",
     "各项判据的数值门槛"),
    ("experiment_design", "ANOMALY_RESPONSE", "异常处置",
     "出了状况先做什么"),
    ("experiment_design", "MEASUREMENT_STRATEGIES", "测量策略",
     "按目的选测法"),
    ("experiment_design", "REFERENCE_EXPERIMENTS", "参考实验",
     "可照着做的样板"),
]

#: 只给索引的：模块 → (标题, 取它的工具)。内容在工具里，书里不复制一份。
_KB_INDEXED = [
    ("clean_metal", "洁净金属", "query_knowledge"),
    ("semiconductor", "半导体", "query_knowledge"),
    ("molecular_adsorbate", "分子吸附", "query_knowledge"),
    ("superconductor", "超导体", "query_knowledge"),
    ("topological", "拓扑材料", "query_knowledge"),
    ("magnetic_spm", "磁性 SPM", "query_knowledge"),
    ("thin_film", "薄膜", "query_knowledge"),
    ("two_d_material", "二维材料", "query_knowledge"),
    ("oxide_surface", "氧化物表面", "query_knowledge"),
    ("on_surface_synthesis", "表面合成", "query_knowledge"),
    ("fault_diagnosis", "故障诊断库", "get_fault_diagnosis"),
    ("skill_guidance", "技能指引 / 决策树 / 测量模板",
     "get_skill_guidance · get_measurement_template"),
    ("stm_noise", "噪声目录", "get_noise_reference"),
    ("safety_constraints", "安全约束", "（服务端包络裁决用）"),
    ("hardware_profile", "硬件档案", "query_knowledge"),
    ("reference_index", "深研报告索引",
     "search_deep_reference · read_reference_section"),
    ("image_databases", "图像数据库", "query_knowledge"),
]


def _render_value(v, depth: int = 0) -> str:
    """把知识库里的嵌套 dict/list 渲染成可读的 HTML，不是一坨 JSON。"""
    if isinstance(v, dict):
        rows = "".join(
            f"<tr><td class='k'>{html.escape(str(k))}</td>"
            f"<td>{_render_value(x, depth + 1)}</td></tr>"
            for k, x in v.items())
        return f"<table class='kv'>{rows}</table>"
    if isinstance(v, (list, tuple)):
        if all(not isinstance(x, (dict, list, tuple)) for x in v):
            return "、".join(html.escape(str(x)) for x in v)
        return "".join(f"<div class='li'>{_render_value(x, depth + 1)}</div>"
                       for x in v)
    return html.escape(str(v))


def collect_knowledge() -> list[Section]:
    import importlib

    out: list[Section] = []

    intro = _md(
        "这一章是 **agent 用查询工具取得到的东西** —— 不是它每次都读到的。"
        "v2 的知识走**拉取式**：提示词里不装知识，模型需要时去查。\n\n"
        "**取舍**：操作员会去查、也会去改的**运行规程**全量收在下面；"
        "**材料库 / 故障库 / 文献索引**只给索引（有什么、多少条、哪个工具取）"
        "—— 那是约 150 种材料的参数表，全量抄进来会让这本书翻倍，"
        "而它们的真源在代码里，工具随时取得到。\\n\\n"
        "**这一章的中英混排是原样**：`name` / `description` 是中文，"
        "`purpose` / `go_nogo` 这些字段是英文 —— 知识库数据本身就长这样，"
        "这次的中文化范围是 **agent 提示词与技能描述**，没有动知识库。"
        "要不要一并中文化，是一个单独的决定。")

    rows = []
    for mod_name, label, tool in _KB_INDEXED:
        try:
            mod = importlib.import_module(f"mast.knowledge.{mod_name}")
        except Exception:                                        # noqa: BLE001
            continue
        n = 0
        for attr in ("MATERIALS", "FAULT_CATEGORIES", "SKILL_EXTRA",
                     "REFERENCE_INDEX", "NOISE_BENCHMARKS", "CONTROLLERS",
                     "EXPERIMENTAL_DATASETS", "COARSE_MOTION_RULES"):
            got = getattr(mod, attr, None)
            if got:
                n = len(got)
                break
        rows.append(f"<tr><td>{html.escape(label)}</td>"
                    f"<td class='num'>{n or '—'}</td>"
                    f"<td><code>{html.escape(tool)}</code></td>"
                    f"<td><code>mast/knowledge/{html.escape(mod_name)}.py</code></td>"
                    f"</tr>")
    index_tbl = (
        "<h3>只给索引的部分</h3>"
        "<table class='kb-index'><thead><tr><th>库</th><th>条目</th>"
        "<th>取它的工具</th><th>源文件</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>")

    out.append(Section(slug="kb-intro", title="知识库是什么 · 收了哪些",
                       body_html=intro + index_tbl,
                       source="MASTv2/mast/knowledge/"))

    for mod_name, attr, label, why in _KB_OPERATIONAL:
        try:
            mod = importlib.import_module(f"mast.knowledge.{mod_name}")
            data = getattr(mod, attr)
        except Exception as exc:                                 # noqa: BLE001
            out.append(Section(slug=f"kb-{_slug(attr)}", title=label,
                               body_html=f"<p class='warn'>读不到："
                                         f"{html.escape(str(exc))}</p>"))
            continue
        n = len(data)
        # 出处由 Section.source 渲染，这里别再写一遍（曾经重了一次）
        head = f"<p class='meta'>{n} 条</p>" + _md(why)
        out.append(Section(slug=f"kb-{_slug(attr)}", title=f"{label}（{n}）",
                           body_html=head + _render_value(data),
                           source=f"mast/knowledge/{mod_name}.py:{attr}"))
    return out


# ── 4. 注入矩阵（谁收到什么）─────────────────────────────────────────────

def collect_matrix() -> Section | None:
    try:
        from mast.prompts import manifest, registry as reg
    except Exception:  # noqa: BLE001
        return None
    m = manifest.matrix()
    agents = m["agents"]
    head = "".join(f"<th>{html.escape(manifest.label_for(a).split()[-1])}</th>"
                   for a in agents)
    rows = []
    for r in m["rows"]:
        e = r["entry"]
        cells = "".join(
            "<td class='{cls}'>{mark}</td>".format(
                cls="c-all" if r["shared"] and r["cells"][a] else
                    ("c-yes" if r["cells"][a] else "c-no"),
                mark="○" if (r["shared"] and r["cells"][a])
                     else ("●" if r["cells"][a] else "·"))
            for a in agents)
        rows.append(f"<tr><td class='blk'>{html.escape(e.label)}"
                    f"<code>{html.escape(e.id)}</code></td>{cells}</tr>")
    body = (
        "<p class='note'>行 = 一段注入进上下文的文本，列 = 收到它的角色。"
        "<b>●</b> = 定向给这个角色 · <b>○</b> = 全员都收 · <b>·</b> = 不发给它。</p>"
        f"<div class='scroll'><table class='matrix'><thead><tr><th>注入块</th>"
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    return Section(slug="matrix", title="上下文注入矩阵", body_html=body,
                   source="mast/prompts/manifest.py:matrix()")


# ════════════════════════════════════════════════════════════════════════
# 渲染
# ════════════════════════════════════════════════════════════════════════

_CSS = """
:root{--bg:#fbfaf8;--panel:#fff;--text:#1b1b1c;--muted:#5c5c62;--faint:#8b8b93;
--border:#e3e0da;--accent:#8a5a2b;--accent-soft:#f5ede3;--warn:#8a5a00;
--danger:#9b2c2c;--ok:#2f6b45;--code:#f4f1ec;}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#16161a;--panel:#1d1d22;--text:#e8e6e1;--muted:#a5a3a0;--faint:#77757a;
--border:#2e2e36;--accent:#d8a76a;--accent-soft:#2a2118;--warn:#d8a24a;
--danger:#e08585;--ok:#7fbf9a;--code:#232329;}}
:root[data-theme=dark]{--bg:#16161a;--panel:#1d1d22;--text:#e8e6e1;--muted:#a5a3a0;
--faint:#77757a;--border:#2e2e36;--accent:#d8a76a;--accent-soft:#2a2118;
--warn:#d8a24a;--danger:#e08585;--ok:#7fbf9a;--code:#232329;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.75 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}
#wrap{display:flex;min-height:100vh}
#nav{width:290px;flex:0 0 290px;background:var(--panel);border-right:1px solid var(--border);
position:sticky;top:0;height:100vh;overflow:auto;padding:18px 0 40px}
#nav h1{font-size:17px;margin:0 18px 4px;letter-spacing:.02em}
#nav .sub{font-size:12px;color:var(--faint);margin:0 18px 14px}
#nav .grp{font-size:11px;letter-spacing:.12em;text-transform:uppercase;
color:var(--faint);margin:18px 18px 6px}
#nav a{display:block;padding:5px 18px;color:var(--muted);text-decoration:none;
font-size:14px;border-left:2px solid transparent}
#nav a:hover{background:var(--accent-soft);color:var(--text)}
#nav a.on{border-left-color:var(--accent);color:var(--text);background:var(--accent-soft)}
#q{width:calc(100% - 36px);margin:0 18px 10px;padding:7px 10px;font-size:13px;
border:1px solid var(--border);border-radius:7px;background:var(--bg);color:var(--text)}
main{flex:1;min-width:0;padding:40px 52px 120px;max-width:980px}
section{margin-bottom:64px;scroll-margin-top:20px}
h2{font-size:26px;margin:0 0 6px;padding-bottom:8px;border-bottom:2px solid var(--accent)}
h3{font-size:19px;margin:32px 0 8px}
h4{font-size:15px;margin:22px 0 4px;display:flex;flex-wrap:wrap;align-items:center;gap:8px}
p,li{color:var(--text)}
.meta,.src{font-size:12px;color:var(--faint);margin:2px 0 14px}
.src code{font-size:11px}
.note{background:var(--accent-soft);border-left:3px solid var(--accent);
padding:10px 14px;border-radius:0 7px 7px 0;font-size:14px}
code{background:var(--code);padding:1px 5px;border-radius:4px;
font:13px/1.6 "SF Mono",Consolas,monospace}
pre{background:var(--code);padding:14px 16px;border-radius:9px;overflow:auto;
border:1px solid var(--border)}
pre.prompt{white-space:pre-wrap;word-break:break-word;font-size:13px;line-height:1.7;
max-height:none}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px}
th,td{border:1px solid var(--border);padding:6px 9px;text-align:left;vertical-align:top}
th{background:var(--accent-soft);font-weight:600}
.params{margin:8px 0 4px}
.params td:first-child{white-space:nowrap}
.skill{border:1px solid var(--border);border-radius:10px;padding:12px 16px;
margin:14px 0;background:var(--panel)}
.skill .desc{margin:4px 0 8px;color:var(--text)}
.badges{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.badges span{font-size:11px;padding:2px 7px;border-radius:20px;border:1px solid var(--border);
color:var(--muted);font-weight:400}
.b-dangerous{color:var(--danger);border-color:var(--danger)}
.b-auto{color:var(--ok);border-color:var(--ok)}
.tag{font-size:11px;background:var(--code);color:var(--muted);padding:1px 6px;
border-radius:4px;margin-right:4px}
.tags{margin:6px 0 0}
.also{font-size:11px;color:var(--muted);margin-left:6px}
.onlytodo{display:flex;align-items:center;gap:6px;font-size:12px;
          color:var(--muted);margin:0 0 8px;cursor:pointer;user-select:none}
.onlytodo input{cursor:pointer}
.hitcount{color:var(--accent)}
.kv{width:100%;border-collapse:collapse;margin:2px 0}
.kv td{padding:3px 8px;border-top:1px solid var(--border);
       vertical-align:top;font-size:13px}
.kv td.k{color:var(--muted);white-space:nowrap;width:1%;font-weight:600}
.li{border-left:2px solid var(--border);padding-left:10px;margin:8px 0}
.kb-index{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px}
.kb-index th,.kb-index td{padding:5px 8px;border-bottom:1px solid var(--border);
                          text-align:left}
.kb-index td.num{text-align:right;color:var(--accent)}
/* 来源标记：来源比内容重要，所以四档必须一眼分得开 */
.mark{font-weight:600;font-size:.92em;padding:1px 3px;border-radius:3px;
      white-space:nowrap}
.m-common{color:var(--muted)}
.m-machine{color:var(--accent);background:var(--accent-soft)}
.m-operator{color:var(--ok);box-shadow:inset 0 0 0 1px var(--ok)}
/* 【存疑】= 别当结论用。四档里唯一需要让人停下来的，给它最强的对比。 */
.m-doubt{color:#fff;background:var(--danger);
         box-shadow:inset 0 0 0 1px var(--danger)}
.tag-en{font-size:11px;color:var(--warn);border:1px solid var(--warn);
padding:1px 6px;border-radius:4px;font-weight:400}
.warn{color:var(--danger)}
.scroll{overflow-x:auto}
.matrix{font-size:12px}
.matrix td{text-align:center}
.matrix td.blk{text-align:left;white-space:nowrap}
.matrix td.blk code{display:block;font-size:10px;color:var(--faint);background:none;padding:0}
.c-all{color:var(--faint)}.c-yes{color:var(--accent);font-weight:700}
.c-no{color:var(--border)}
blockquote{border-left:3px solid var(--border);margin:12px 0;padding:2px 14px;color:var(--muted)}
hr{border:0;border-top:1px solid var(--border);margin:28px 0}
#top{position:fixed;right:22px;bottom:22px;background:var(--panel);color:var(--muted);
border:1px solid var(--border);border-radius:50%;width:42px;height:42px;cursor:pointer;
font-size:16px}
@media print{#nav,#top{display:none}main{max-width:none;padding:0}
section{page-break-inside:auto}pre.prompt{white-space:pre-wrap}}
@media(max-width:900px){#wrap{flex-direction:column}#nav{width:auto;height:auto;
position:static;border-right:0;border-bottom:1px solid var(--border)}
main{padding:24px 18px 80px}}
"""

_JS = """
(function(){
 var q=document.getElementById('q'),links=[].slice.call(document.querySelectorAll('#nav a'));
 var skills=[].slice.call(document.querySelectorAll('article.skill'));
 var only=document.getElementById('onlytodo');
 // 每个技能的可搜文本预算一次：457 条 × 每次按键重算 textContent 会卡。
 skills.forEach(function(a){a.__t=a.textContent.toLowerCase();
                            a.__todo=!!a.querySelector('.tag-en');});
 function apply(){
   var v=q.value.trim().toLowerCase(), todo=only&&only.checked;
   links.forEach(function(a){
     a.style.display=(!v||a.textContent.toLowerCase().indexOf(v)>=0)?'':'none';});
   skills.forEach(function(a){
     var hit=(!v||a.__t.indexOf(v)>=0)&&(!todo||a.__todo);
     a.style.display=hit?'':'none';});
   // 一个包里全被过滤掉就把整节收起来，免得留下一串空标题
   [].slice.call(document.querySelectorAll('section')).forEach(function(sec){
     var own=sec.querySelectorAll('article.skill');
     if(!own.length)return;
     var vis=0;
     [].slice.call(own).forEach(function(a){if(a.style.display!=='none')vis++;});
     sec.style.display=vis?'':'none';
     var n=sec.querySelector('.hitcount');
     if(n)n.textContent=(v||todo)?(' · 命中 '+vis):'';
   });
 }
 q.addEventListener('input',apply);
 if(only)only.addEventListener('change',apply);
 var secs=[].slice.call(document.querySelectorAll('section'));
 var byId={};links.forEach(function(a){byId[a.getAttribute('href').slice(1)]=a;});
 var io=new IntersectionObserver(function(es){
   es.forEach(function(e){
     var a=byId[e.target.id];if(!a)return;
     if(e.isIntersecting){links.forEach(function(x){x.classList.remove('on');});
       a.classList.add('on');}
   });},{rootMargin:'-10% 0px -85% 0px'});
 secs.forEach(function(s){io.observe(s);});
 document.getElementById('top').onclick=function(){window.scrollTo({top:0,behavior:'smooth'});};
})();
"""


def render(groups: list[tuple[str, list[Section]]], stats: dict) -> str:
    nav, body = [], []
    for gname, secs in groups:
        if not secs:
            continue
        nav.append(f"<div class='grp'>{html.escape(gname)}</div>")
        for s in secs:
            nav.append(f"<a href='#{s.slug}'>{html.escape(s.title)}</a>")
            src = (f"<p class='src'>出处 <code>{html.escape(s.source)}</code></p>"
                   if s.source else "")
            body.append(f"<section id='{s.slug}'><h2>{html.escape(s.title)}</h2>"
                        f"{src}{s.body_html}</section>")
    return (
        "<title>MAST 手册</title>\n"
        f"<style>{_CSS}</style>\n"
        "<div id='wrap'>\n<aside id='nav'>"
        "<h1>MAST 手册</h1>"
        f"<p class='sub'>STM 操作知识 · {len(_AGENT_ORDER)} 个 agent 角色 · "
        f"{stats.get('total', 0)} 个仪器技能</p>"
        "<label class='onlytodo'>"
        "<input type='checkbox' id='onlytodo'> 只看待译</label>"
        "<input id='q' placeholder='搜技能名 / 目录…' autocomplete='off'>"
        + "".join(nav) + "</aside>\n<main>" + "".join(body) + "</main></div>"
        "<button id='top' title='回到顶部'>↑</button>"
        f"<script>{_JS}</script>"
    )


def build(out_path: Path) -> dict:
    guide = collect_stm_guide()
    agents = collect_agents()
    skills, stats = collect_skills()
    knowledge = collect_knowledge()
    matrix = collect_matrix()

    intro = Section(
        slug="intro", title="这本手册是什么",
        body_html=_md(
            "这本手册把 MAST 里**给人读的那一半**收在一起，五部分：\n\n"
            "1. **STM 操作实验指南** —— 由 Claude 在真机验收里口述整理的十篇。"
            "**每条事实都标了来源**，四档在正文里各有配色："
            "【通识】领域普遍知识 · 【本机】rig 上实测 · "
            "【操作员】现场明确说的（权威）· 【存疑】推断未验证"
            "（**不要当结论用**）。读到【存疑】而你正要据此动硬件 —— 停下来问人。\n"
            "2. **agent 角色** —— 九个角色的系统提示词**原文**。这是模型每一次"
            "调用真正读到的文本，不是它的说明书。写错一句，仪器就会照着做。\n"
            "3. **知识库** —— agent 用查询工具**取得到**的东西（不是每次都读到的："
            "v2 的知识走拉取式）。运行规程全量收；材料库/故障库/文献索引只给索引。\n"
            "4. **上下文注入** —— 哪一块内容送给哪个角色。\n"
            "5. **仪器技能** —— 全部技能按「工具包」分组（与按需加载的分包同一套），"
            "带描述、参数、单位、范围、安全等级。**每个技能只出现一次**。\n\n"
            "校对提示：左上角搜索框**同时过滤目录与技能条目**；"
            "勾上**「只看待译」**能把还没译的挑出来"
            + (f"（当前 {stats['todo_desc'] + stats['todo_params']} 条）。\n\n"
               if (stats["todo_desc"] + stats["todo_params"])
               else "——**当前一条都没有**，这个开关留着是为了下次有人加了英文"
                    "描述时能一眼找出来。\n\n")
            + f"**中文化进度**：技能描述 {stats['zh_desc']}/{stats['total']}，"
            f"参数说明 {stats['zh_params']}/{stats['params']}"
            + ("。\n\n"
               if not (stats["todo_desc"] + stats["todo_params"])
               else f"（还差 {stats['todo_desc']} + {stats['todo_params']} 条）。\n\n")
            + "「到位」的口径：**已译成中文，或本来就该是英文的字面取值**。"
            "整条是 Nanonis 取值清单的（`0 = X/Y，1 = R/phi`、`bright | dark | "
            "auto`）不算漏译 —— 把它们标成「待译」只会让人去追一堆不该改的东西。"
            "剩下夹在中文里的英文是**技能名、参数名、通道名、单位、化学式**，"
            "那些必须保持原样：模型是照着它们发调用的。\n\n"
            "> 这本书由 `MASTv2/scripts/build_handbook.py` 从仓库现状生成，"
            "**可以随时重跑**。内容会随代码变化 —— 它是一份快照，不是真源。"))

    groups = [
        ("开始", [intro]),
        ("STM 操作知识库", guide),
        ("agent 角色（系统提示词原文）", agents),
        ("知识库（agent 查得到的）", knowledge),
        ("上下文注入", [matrix] if matrix else []),
        ("仪器技能", skills),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(groups, stats), encoding="utf-8")
    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "guide": len(guide), "agents": len(agents), "packs": len(skills),
        "knowledge": len(knowledge),
        **stats,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成 MAST HTML 手册")
    ap.add_argument("-o", "--out",
                    default=str(_REPO / "docs" / "handbook" / "MAST手册.html"))
    args = ap.parse_args(argv)
    info = build(Path(args.out))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
