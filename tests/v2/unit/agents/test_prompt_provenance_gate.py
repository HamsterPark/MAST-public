"""提示词里只装**规则**，不装**论证**。

## 为什么这是一道闸门而不是一条口头约定

IC 的系统提示词长到 18 183 字符，其中相当一部分是**给开发者看的事故溯源**：
「（2026-07-27 15:04）」、「commit 0000000」、「Measured 2026-08-04,
12 trials」、整整一节「这条规则为什么归你（IC）管、不归编排器管」、以及用户
的原话引用。

这些内容**该留**——它们是这条规则为什么存在的唯一记录，删了下一个人只会看到
一段没有来由的指令。但它们该留在 `prompts.py` 的 **Python 注释**里，不是留在
**字符串常量**里：

* 模型不需要知道我们哪一天量的、量了几次、谁在什么时候抱怨过什么；
* 每一轮、每一次模型调用，这些字都在重新付一遍钱；
* 更要紧的是它稀释了**规则本身** —— 判据读起来像新的指令，事故编号
  读起来像可以查询的东西，而模型两样都做不了。

判据走 AST，只看**会进模型上下文**的字符串常量；注释与 docstring 一概不判
（那是这些内容该去的地方，逼人删掉解释的闸门是负收益）。

## 保留机理，删掉日期与次数

「裸数字在这条 API 上会丢指数，所以前缀是校验和」是**机理**，模型需要它才能
理解为什么格式不能商量 —— 保留。「Measured 2026-08-04, 12 trials，十二次里
十二次」是**证据**，模型对它无能为力 —— 进注释。

Run from repo root::

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/agents/test_prompt_provenance_gate.py -q
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_MASTV2 = str(_REPO / "MASTv2")
if sys.path and sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)

#: 会进模型上下文的提示词真源。加一个 agent 就要加一行 —— 这份名单由
#: :func:`test_the_source_list_covers_every_agent` 对着目录核。
_SOURCES: tuple[str, ...] = (
    "mast/agents/instrument_control/prompts.py",
    "mast/agents/experiment_design/prompts.py",
    "mast/agents/data_processing/prompts.py",
    "mast/agents/literature/prompts.py",
    "mast/agents/research_director/prompts.py",
    "mast/agents/paper_writing/prompts.py",
    "mast/agents/paper_review/prompts.py",
    "mast/agents/brainstorm/prompts.py",
    "mast/agents/buffer_summarizer/prompts.py",
    "mast/agents/orchestrator/graph.py",      # _ROUTER_PROMPT 住在这儿
    "mast/agents/_shared/mode_mw.py",
    "mast/agents/_shared/tool_packs.py",
)

#: 只判**开发者溯源**，不判领域内容。每一条都窄，而且都配了「该去哪」。
_MARKERS: tuple[tuple[str, str], ...] = (
    (r"commit\s+[0-9a-f]{7,}", "commit 哈希 —— 模型查不了它"),
    (r"现场反馈\s*#\d+", "现场反馈编号 —— 进注释"),
    (r"\bfeedback\s*#\d+", "feedback 编号 —— 进注释"),
    (r"20\d\d-\d\d-\d\d", "日期 —— 规则不因日期而不同；进注释"),
    (r"\d+\s*trials\b", "实验次数 —— 保留机理，删掉次数"),
    (r"twelve times out of twelve", "同上（英文写法）"),
    (r"\d+\s*/\s*\d+\s*次(里|中)", "同上（中文写法）"),
    (r"用户(的)?原话", "判据引用 —— 它读起来像新指令；进注释"),
    (r"审查\s*#\d+|review\s*#\d+", "评审编号 —— 进注释"),
)

#: 太短的常量多半是字典键、日志格式、正则，不是提示词。
_MIN_CONST_CHARS = 200


def _read(rel: str) -> str:
    p = _REPO / "MASTv2" / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _prompt_strings(source: str) -> list[str]:
    """源码里**会进模型上下文**的字符串常量；docstring 与注释一律排除。

    docstring 之所以排除，不是因为它们不重要，而是因为它们**必须**能记下
    「这里原来有过什么」；注释根本不在 AST 里，自动就被排除了。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return []
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings and len(n.value) >= _MIN_CONST_CHARS]


def _offending(source: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in _prompt_strings(source):
        for pat, why in _MARKERS:
            for m in re.finditer(pat, s):
                lo = max(0, m.start() - 40)
                out.append((s[lo:m.end() + 40].replace("\n", " ").strip(), why))
    return out


def _offending_text(text: str) -> list[tuple[str, str]]:
    """自检用：把一段裸文本当成一个足够长的提示词常量来判。"""
    padded = text + " " * max(0, _MIN_CONST_CHARS - len(text))
    out: list[tuple[str, str]] = []
    for pat, why in _MARKERS:
        if re.search(pat, padded):
            out.append((text, why))
    return out


# ── 自检 ────────────────────────────────────────────────────────────────

def test_the_gate_actually_reads_the_prompts():
    """一个扫不到文件的闸门永远绿。先证明它有东西可扫。"""
    present = [rel for rel in _SOURCES if _read(rel)]
    assert len(present) >= 10, f"提示词真源只找到 {len(present)} 个：{present}"
    total = sum(len(_read(rel)) for rel in present)
    assert total > 80_000, f"扫到的源码只有 {total} 字符，名单多半错了"
    strings = sum(len(_prompt_strings(_read(rel))) for rel in present)
    assert strings >= 9, f"只抽出了 {strings} 个提示词常量"


# Synthetic markers are assembled at runtime so export scrubbing cannot
# remove the planted mutation before the provenance gate gets to inspect it.
_SYNTHETIC_COMMIT = "".join(("comm", "it ", "f" * 7))
_SYNTHETIC_DATE = "-".join(("2099", "01", "02"))
_SYNTHETIC_FEEDBACK = "".join(("现场", "反馈 ", "#", "9001"))
_SYNTHETIC_QUOTE = "".join(("用户", "原话", "：合成测试输入。"))


@pytest.mark.parametrize("planted", [
    _SYNTHETIC_COMMIT,
    _SYNTHETIC_FEEDBACK,
    "Measured " + _SYNTHETIC_DATE + ", " + str(7) + " trials",
    " ".join(("twelve", "times", "out", "of", "twelve")),
    _SYNTHETIC_QUOTE,
    "".join(("re", "view ", "#", "9002")),
])
def test_the_gate_recognises_planted_provenance(planted):
    """Every synthetic forbidden marker must trip the actual gate."""
    assert _offending_text(planted), f"gate missed synthetic marker: {planted!r}"


def test_provenance_may_live_in_comments_and_docstrings():
    """Developer-only context does not become a prompt constant."""
    src = (
        repr(_SYNTHETIC_COMMIT + " " + _SYNTHETIC_DATE) + "\n"
        + "# " + _SYNTHETIC_FEEDBACK + "\n"
        + 'SYSTEM_PROMPT = "' + "写清规则就好。" * 40 + '"\n'
    )
    assert _offending(src) == []


def test_the_gate_still_catches_it_inside_a_real_prompt_constant():
    """The same synthetic marker in a long prompt must be rejected."""
    body = "规则。" * 90 + _SYNTHETIC_COMMIT
    assert len(body) > _MIN_CONST_CHARS
    assert _offending("SYSTEM_PROMPT = " + repr(body) + "\n")


def test_short_constants_are_not_scanned():
    """字典键、日志格式、正则里出现日期不该判红。"""
    assert _offending('LOG_FMT = "%(asctime)s 2026-08-24 %(message)s"\n') == []


# ── 主判据 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", _SOURCES)
def test_no_prompt_constant_carries_developer_provenance(rel):
    hits = _offending(_read(rel))
    assert not hits, (
        f"{rel} 的提示词常量里有 {len(hits)} 处开发者溯源：\n"
        + "\n".join(f"  · {why}: …{frag}…" for frag, why in hits)
        + "\n把它们移到紧挨那条规则的 Python 注释里 —— 内容要留，位置要换。")


def test_the_source_list_covers_every_agent():
    """名单不能漏掉一个 agent —— 漏掉的那个正好是没人守的那个。"""
    agents_dir = _REPO / "MASTv2" / "mast" / "agents"
    found = {f"mast/agents/{p.parent.name}/prompts.py"
             for p in agents_dir.glob("*/prompts.py")}
    missing = found - set(_SOURCES)
    assert not missing, f"有 agent 的 prompts.py 不在扫描名单里：{sorted(missing)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
