"""⑰ 的结构闸门:**不许再有一句话叫模型去等一个不会到来的批准**。

## 为什么这条要有闸门,而不是靠改的时候记得

审批链路割掉之后,提示词里每一句「这个动作需要人工审批 / 批准后才会执行 / 等待
批准」都变成了**会指挥行动的假话**:模型读到它,就会停下来等,或者绕开一个它以为
要等人的工具。而这类句子散落在 6 个 agent 的 prompts、prompt registry 的说明、
中间件的信念块里 —— 「每一页各自记得」的接线,人肉清单永远找不齐(本仓一天逮到
过四次)。

## 判据

走 AST 取**会进模型上下文的字符串**(赋值/拼接里的字符串常量),然后在里面找一批
「等人」措辞。命中即红,除非同一个字符串里带着否定标记(见 :data:`_ALLOWED_CONTEXT`)。

**为什么是 AST 而不是逐行 grep**,两条都是写这个闸门时当场撞上的:

* 注释和 docstring 里**必须**能写清楚「这里原来有过什么」,否则下一个人只看到一段
  没有来由的代码。逐行扫会把这些历史说明一起判红 —— 那样的闸门会逼人删掉解释,
  正好删掉最该留的东西(前端那一批刚栽过同一个形状:闸门被自己注释里那句
  「从前这是一个 `<Modal>`」判红)。
* 逐行扫**会被换行切开**:作者写的 "Never stop and wait for an approval — none
  will ever arrive." 在源码里折了行,于是带 "wait for an approval" 的那一行不带否定
  词,闸门当场误判。按**整个字符串常量**判就没有这个问题。

## 自检

一个匹配不到任何东西的正则永远是绿的,而它和「全部通过」长得一模一样。所以闸门
先断言**自己确实扫到了东西**(文件数、字符数、以及至少能认出一句故意留下的假话)。

Run from repo root::

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/agents/test_no_stale_approval_wording.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest

#: 会进模型上下文的提示词真源。加一个新 agent 就要加一行 —— 这份名单本身由
#: :func:`test_the_prompt_source_list_covers_every_agent` 对着目录核。
_PROMPT_SOURCES: tuple[str, ...] = (
    "mast/agents/instrument_control/prompts.py",
    "mast/agents/experiment_design/prompts.py",
    "mast/agents/data_processing/prompts.py",
    "mast/agents/literature/prompts.py",
    "mast/agents/research_director/prompts.py",
    "mast/agents/paper_writing/prompts.py",
    "mast/agents/paper_review/prompts.py",
    "mast/agents/brainstorm/prompts.py",
    "mast/agents/orchestrator/prompts.py",
    "mast/agents/buffer_summarizer/prompts.py",
    "mast/agents/_shared/mode_mw.py",
    "mast/agents/_shared/ask_tools.py",
    "mast/skills/composite/persona.py",
)

#: 会让模型「停下来等」的措辞。刻意窄:只抓**指示行为**的句子,不抓单纯提到
#: 「审批」两个字的历史说明 —— 后者是本仓要求写下来的东西。
_WAIT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"等待(人工)?(批准|审批)", "叫模型等一个不会到来的批准"),
    (r"(批准|审批)后(才会|再)(执行|继续)", "承诺「批准后才执行」——没有批准这一步了"),
    (r"需要\*{0,2}人工确认\*{0,2}后", "承诺「人工确认后」才动"),
    (r"routes? to the operator'?s? approval pane", "指向已删除的审批面板"),
    (r"wait(ing)? for (an? )?approval", "叫模型等审批"),
)

#: 允许命中的上下文 —— 一句话里同时出现这些标记,说明它正在**否定**这件事
#: (「不要等待批准」「没有任何批准会到来」),那是要保留的。
_ALLOWED_CONTEXT: tuple[str, ...] = (
    "不会到来", "没有任何批准", "不要", "别", "已经不存在", "不再",
    "Never stop and wait", "do NOT wait", "no approval exists",
    "used to", "gone",
)


def _read(rel: str) -> str:
    p = _REPO / "MASTv2" / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _prompt_strings(source: str) -> list[str]:
    """源码里**会进模型上下文**的字符串常量;docstring 与注释一律排除。

    docstring 之所以排除,不是因为它们不重要,而是因为它们**必须**能记下「这里
    原来有过什么」;注释根本不在 AST 里,自动就被排除了。
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover — 仓里的文件都编译得过
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
    return [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docstrings
    ]


#: 否定标记的搜索半径(字符)。**不能按整个字符串判** —— 一份 SYSTEM_PROMPT 是一个
#: 上万字符的常量,里面任何一处「不要」都会把整份提示词豁免掉。写这个闸门时第一版
#: 就是那样,变异验证当场证明它是个摆设:把「CONFIRM — 等待人工批准，批准后才会
#: 执行」塞回 IC 的 SYSTEM_PROMPT,18 条测试全绿。
_CONTEXT_RADIUS = 120


def _offending(source: str) -> list[tuple[str, str]]:
    """返回 (片段, 命中理由)。**每一处**命中单独判,否定标记只看它附近。"""
    out: list[tuple[str, str]] = []
    for s in _prompt_strings(source):
        for pat, why in _WAIT_PATTERNS:
            for m in re.finditer(pat, s):
                lo = max(0, m.start() - _CONTEXT_RADIUS)
                near = s[lo:m.end() + _CONTEXT_RADIUS]
                if any(mark in near for mark in _ALLOWED_CONTEXT):
                    continue
                frag_lo = max(0, m.start() - 40)
                out.append((s[frag_lo:m.end() + 40].replace("\n", " ").strip(),
                            why))
    return out


def _offending_lines(text: str) -> list[tuple[str, str]]:
    """自检用:把一段裸文本当成一个字符串常量来判。"""
    out: list[tuple[str, str]] = []
    for pat, why in _WAIT_PATTERNS:
        if re.search(pat, text):
            if any(mark in text for mark in _ALLOWED_CONTEXT):
                return []
            out.append((text.strip(), why))
            break
    return out


# ── 自检:闸门自己扫到了东西吗 ─────────────────────────────────────────

def test_the_gate_actually_reads_something():
    """一个扫不到文件的闸门永远绿。先证明它有东西可扫。"""
    present = [rel for rel in _PROMPT_SOURCES if _read(rel)]
    assert len(present) >= 8, f"提示词真源只找到 {len(present)} 个:{present}"
    total = sum(len(_read(rel)) for rel in present)
    assert total > 50_000, f"扫到的提示词总量只有 {total} 字符,名单多半错了"


def test_the_gate_recognises_a_deliberately_stale_sentence():
    """变异自检:塞一句典型的假话进去,闸门必须逮到。

    没有这一条,上面那些正则写错一个字也不会有人知道。"""
    for bad in (
        "  DANGEROUS 技能会等待人工批准，批准后才会执行。",
        "  这个动作需要**人工确认**后才会执行。",
        "  any DANGEROUS skill routes to the operator's approval pane.",
        "  Stop and wait for an approval before continuing.",
    ):
        assert _offending_lines(bad), f"闸门漏掉了:{bad!r}"


def test_the_gate_does_not_flag_a_sentence_that_denies_it():
    """反向自检:说「不要等批准」的句子不能被判红,否则闸门会逼人删掉解释。"""
    for good in (
        "  这不是等待审批，没有任何批准会到来。",
        "  Never stop and wait for an approval — none will ever arrive.",
        "  It used to route to the operator's approval pane; that pane is gone.",
    ):
        assert not _offending_lines(good), f"闸门误判了:{good!r}"


# ── 主判据 ────────────────────────────────────────────────────────────

def test_the_gate_ignores_docstrings_and_comments():
    """历史说明必须能写。写在 docstring / 注释里的旧文案不判红。"""
    src = (
        '"""这里原来写着「批准后才会执行」,⑰ 之后没有了。"""\n'
        '# 旧文案:需要人工确认后才动。\n'
        'GOOD = "直接执行并留痕。"\n'
    )
    assert _offending(src) == []


def test_the_gate_still_catches_it_in_a_real_prompt_constant():
    """而写进**提示词常量**的同一句话必须被逮到 —— 上一条不能把闸门变成摆设。"""
    src = 'SYSTEM_PROMPT = "DANGEROUS 技能会等待人工批准。"\n'
    assert _offending(src), "闸门对提示词常量失效了"


def test_a_denial_far_away_does_not_excuse_a_stale_sentence():
    """**闸门自己的变异测试。**

    第一版按**整个字符串**找否定标记,而一份 SYSTEM_PROMPT 是上万字符的单个常量
    —— 里面任何一处「不要」都会把整份提示词豁免掉。变异验证当场证明它是摆设:
    把一句「等待人工批准，批准后才会执行」塞进 IC 的 SYSTEM_PROMPT,18 条全绿。

    现在否定标记只在命中处 ±``_CONTEXT_RADIUS`` 内找。这条测试就是那次变异,
    钉住别再退回去。
    """
    far = "不要" + "。" * (_CONTEXT_RADIUS + 50)
    src = f'SYSTEM_PROMPT = "{far}CONFIRM — 等待人工批准，批准后才会执行。"\n'
    assert _offending(src), "远处的一句「不要」又把整份提示词豁免掉了"

    # …而近处的否定仍然放过(否则闸门会逼人删掉解释)。
    near = 'SYSTEM_PROMPT = "这不是等待人工批准，没有任何批准会到来。"\n'
    assert _offending(near) == []


@pytest.mark.parametrize("rel", _PROMPT_SOURCES)
def test_no_prompt_tells_the_model_to_wait_for_an_approval(rel):
    """⑰ 之后没有任何批准会到来 —— 提示词里不能还有一句话叫模型去等。"""
    hits = _offending(_read(rel))
    assert not hits, (
        f"{rel} 里还有 {len(hits)} 处会让模型等批准的提示词:\n" +
        "\n".join(f"  · {why}: …{frag}…" for frag, why in hits))


def test_the_prompt_source_list_covers_every_agent():
    """名单不能漏掉一个 agent —— 漏掉的那个正好是没人守的那个。"""
    agents_dir = _REPO / "MASTv2" / "mast" / "agents"
    found = {
        f"mast/agents/{p.parent.name}/prompts.py"
        for p in agents_dir.glob("*/prompts.py")
    }
    missing = found - set(_PROMPT_SOURCES)
    assert not missing, f"有 agent 的 prompts.py 不在扫描名单里:{sorted(missing)}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
