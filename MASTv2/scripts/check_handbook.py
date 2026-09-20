"""手册的结构体检 —— 光 grep 它不够，人是要**打开它读**的。

    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/check_handbook.py
    # 或指定一份： ... check_handbook.py 某个.html

查这几件在阅读时真的会硌人的事：

1. 目录链接指向不存在的锚点（点了没反应）
2. `id` 重复（浏览器只跳第一个 —— 曾经 457 个技能被列成 613 条）
3. HTML 标签被当文本转义（页面上印出字面的 `<p class=...>` —— 发生过）
4. 空章节
5. 外部请求（这本书要能离线打开）
6. **提示词逐字一致** —— 见下
7. 「待译」只标英文散文，没误标 Nanonis 字面取值
8. 来源标记四档都上了色

## 第 6 条为什么单列

agent 那一章的**全部价值**就在「这是模型每一次调用真正读到的那份文本」。
只要它与源码差一个字，这一章就从「可以逐句校对的原文」退化成「大致是那么回事
的摘录」，而读的人**不会知道**。所以逐字比对，不比长度、不比相似度。

报红时先看是不是**快照过期**（书是在提示词改动之前生成的）—— 重跑
`build_handbook.py` 即可。重跑后仍不一致，才是转写 bug。
"""
import html as _html
import importlib
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "MASTv2"))
sys.path.insert(0, str(_REPO))

path = Path(sys.argv[1] if len(sys.argv) > 1
            else "docs/handbook/MAST手册.html")
doc = path.read_text(encoding="utf-8")
print(f"{path}  {len(doc):,} 字符 / {path.stat().st_size / 1024:.0f} KB\n")

bad = 0


def fail(msg):
    global bad
    bad += 1
    print(f"✗ {msg}")


def ok(msg):
    print(f"✓ {msg}")


# ── 1. 锚点 ──────────────────────────────────────────────────────────────
ids = re.findall(r"""\bid=['"]([^'"]+)['"]""", doc)
links = re.findall(r"""\bhref=['"]#([^'"]+)['"]""", doc)
idset = set(ids)
dangling = sorted({ln for ln in links if ln not in idset})
if dangling:
    fail(f"{len(dangling)} 个目录链接指向不存在的锚点：{dangling[:8]}")
else:
    ok(f"{len(links)} 个内部链接全部有落点")

dupes = [k for k, n in Counter(ids).items() if n > 1]
if dupes:
    fail(f"{len(dupes)} 个重复 id（浏览器只跳第一个）：{dupes[:8]}")
else:
    ok(f"{len(ids)} 个 id 无重复")

# ── 2. 被转义的 HTML 标签（会以字面文本印在页面上）────────────────────────
leaks = re.findall(r"&lt;(p|div|span|table|h[1-6])\b[^&]{0,40}&gt;", doc)
# 提示词原文里可能合法地含尖括号（<step>、<exact skill name>），只报常见 HTML 标签
if leaks:
    fail(f"{len(leaks)} 处 HTML 标签被当文本转义，会印在页面上："
         f"{sorted(set(leaks))[:6]}")
else:
    ok("没有被误转义成字面文本的 HTML 标签")

# ── 3. 空章节 ────────────────────────────────────────────────────────────
sections = re.findall(r"<section[^>]*id=['\"]([^'\"]+)['\"][^>]*>(.*?)</section>",
                      doc, re.S)
empties = [sid for sid, body in sections
           if len(re.sub(r"<[^>]+>", "", body).strip()) < 40]
if empties:
    fail(f"{len(empties)} 个空章节：{empties[:8]}")
else:
    ok(f"{len(sections)} 个 <section> 都有实质内容"
       if sections else "（没有用 <section> 分块，跳过）")

# ── 4. 离线自包含 ────────────────────────────────────────────────────────
ext = re.findall(r"""(?:src|href)=['"](https?:)?//[^'"]+['"]""", doc)
ext += re.findall(r"""@import\s+url\(['"]?https?://""", doc)
if ext:
    fail(f"{len(ext)} 个外部请求 —— 这本书就不能离线看了")
else:
    ok("零外部请求（可离线打开）")

# ── 5. 五部分内容都在 ────────────────────────────────────────────────────
need = {
    "STM 操作实验指南": r"stm-\d",
    "agent 提示词": r"agent-(instrument_control|data_processing)",
    "知识库": r"id='kb-",
    "技能条目": r"id='skill-",
    "注入矩阵": r"上下文注入",
}
for label, pat in need.items():
    n = len(re.findall(pat, doc))
    (ok if n else fail)(f"{label}：{n} 处")

# ── 5.5 提示词逐字一致（这一章的全部价值所在）───────────────────────────
_AGENTS = ["orchestrator", "research_director", "literature",
           "experiment_design", "instrument_control", "data_processing",
           "paper_writing", "paper_review", "buffer_summarizer"]


def _source_prompt(agent: str) -> str:
    if agent == "orchestrator":
        from mast.agents.orchestrator.graph import _ROUTER_PROMPT
        return _ROUTER_PROMPT
    mod = __import__(f"mast.agents.{agent}.prompts", fromlist=["SYSTEM_PROMPT"])
    return getattr(mod, "SYSTEM_PROMPT", "")


try:
    importlib.import_module("tests.v2.conftest")     # mock nanonis_spm
    drift = []
    for agent in _AGENTS:
        src = _source_prompt(agent)
        if not src:
            continue
        i = doc.find(f"id='agent-{agent}'")
        if i < 0:
            drift.append(f"{agent}: 书里没有这一章")
            continue
        j = doc.find("<pre class='prompt'>", i)
        k = doc.find("</pre>", j)
        shown = _html.unescape(doc[j + len("<pre class='prompt'>"):k])
        if shown == src:
            continue
        where = next((n for n, (a, b) in enumerate(zip(src, shown)) if a != b),
                     min(len(src), len(shown)))
        drift.append(f"{agent}: 第 {where} 字符起不一致"
                     f"（源码 {len(src):,} / 书里 {len(shown):,}）")
    if drift:
        fail("提示词与源码不逐字一致：\n     " + "\n     ".join(drift)
             + "\n     先重跑 build_handbook.py —— 多半只是快照过期；"
               "重跑后仍不一致才是转写 bug")
    else:
        ok(f"{len(_AGENTS)} 份提示词与源码逐字一致")
except Exception as exc:                                         # noqa: BLE001
    fail(f"提示词逐字比对跑不起来：{type(exc).__name__}: {exc}")

# ── 6. 待译标记只落在英文散文上（抽样验）─────────────────────────────────
marked = re.findall(r"([^<>]{0,80})\s*<span class='tag-en'>待译</span>", doc)
cjk = re.compile(r"[一-鿿]")
wrong = [m.strip() for m in marked if cjk.search(_html.unescape(m))]
if wrong:
    fail(f"{len(wrong)} 处「待译」标在了含中文的条目上：{wrong[:3]}")
else:
    ok(f"{len(marked)} 处「待译」都标在英文条目上")

# ── 7. 来源标记上了色 ────────────────────────────────────────────────────
for cls, label in [("m-doubt", "存疑"), ("m-operator", "操作员"),
                   ("m-machine", "本机"), ("m-common", "通识")]:
    n = len(re.findall(f"class='mark {cls}'", doc))
    (ok if n else fail)(f"来源标记【{label}】上色 {n} 处")

print()
print("⛔ 有问题，见上" if bad else "✓ 结构体检全过")
sys.exit(1 if bad else 0)
