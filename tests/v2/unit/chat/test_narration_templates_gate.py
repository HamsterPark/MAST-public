"""旁白模板的**结构闸门** —— 人肉扫不出来的那几种坏法。

这三条坏法有一个共同点：**代码全绿、语法全对、跑起来一句话也不少**，只是那句话
永远不带数字，或者带的是一个编出来的数字。它们只能靠结构挡，靠读是读不出来的。

① key 写错 → **永远走 fallback**。``params.puls_v`` 拼错一个字母，模板不会报错，
   它会一辈子说「我们要打脉冲修针（电压/宽度没记下来）」。这是本仓
   ``lookup(错名字) || 默认`` 那一类的第三次（StartScan 的 ``"Piezo"``、
   settings 的 KNOWN_KEYS 各一次）。这里拿**真的 SkillMetadata** 对账。

② 句子里写死数字。``f"我们要打一发 10 V 脉冲"`` 读起来和真的一模一样。
   用 AST 断言 render 函数体里的字符串**一个数字都不许有**。

③ 在 render 里手写单位换算（``v * 1e12``）。本仓已经为「LLM 工具调用丢指数
   （3e-12 → 3 米）」付过一次学费。用 AST 断言 render 体里**没有任何字面数字**，
   于是换算只能走 :func:`narration_templates.si`（它共用 ``core.si_quantity``
   的前缀表，且有 round-trip 测试）。

闸门而不是人肉扫，是本仓已经用过的手法（``test_forge_scan_working_point``）。

    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/chat/test_narration_templates_gate.py -q
"""
from __future__ import annotations

import ast
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

import pytest  # noqa: E402

from mast.chat import narration_templates as T  # noqa: E402
from mast.core.registry import SkillRegistry  # noqa: E402
from mast.core.si_quantity import format_si, parse_si  # noqa: E402

_SRC = Path(T.__file__)
_DISCOVER = ("mast.skills.builtins", "mast.skills.composite")


# ── ① key 必须是真的 ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def metas():
    reg = SkillRegistry()
    reg.discover(*_DISCOVER)
    return {m.name: m for m in reg.list_skills()}


def test_every_mapped_skill_exists(metas):
    """``BEGIN_KIND_FOR_SKILL`` 的键必须是真技能名。

    映射到一个不存在的技能不会报错 —— 那一条只是**永远不触发**，
    于是「已经接上了」这句话在整棵树上没有任何证据支持。
    """
    unknown = sorted(set(T.BEGIN_KIND_FOR_SKILL) - set(metas))
    assert not unknown, (
        f"这些技能不存在（改名了？）：{unknown}。"
        f"映射到一个不存在的技能 = 这条旁白永远不会出现，而且不会有人发现。")


def test_every_mapped_kind_has_a_template():
    unknown = sorted({k for k in T.BEGIN_KIND_FOR_SKILL.values()
                      if k not in T.TEMPLATES}
                     | {k for k in T.RESULT_KIND_FOR_SKILL.values()
                        if k not in T.TEMPLATES})
    assert not unknown, f"映射到了不存在的模板 kind：{unknown}"


def test_every_required_params_key_is_a_real_parameter(metas):
    """``requires`` / ``records`` 里每一个 ``params.X`` 都必须是那个技能**真有**的参数。

    这是本文件最重要的一条。它把「拼错一个字母 → 永远走 fallback」从一个
    只有真机才会暴露的静默失效，变成一条测试红线。
    """
    problems: list[str] = []
    for skill_name, kind in sorted(T.BEGIN_KIND_FOR_SKILL.items()):
        meta = metas.get(skill_name)
        tpl = T.TEMPLATES.get(kind)
        if meta is None or tpl is None:
            continue        # 由上面两条断言负责
        real = {p.name for p in (getattr(meta, "parameters", None) or [])}
        for path in tpl.requires + tpl.records:
            if not path.startswith("params."):
                continue
            key = path.split(".", 1)[1]
            if key not in real:
                problems.append(
                    f"{kind}: {skill_name} 没有参数 {key!r}（它有：{sorted(real)}）")
    assert not problems, "\n".join(problems)


def test_a_template_with_no_requires_says_the_same_thing_no_matter_what():
    """``requires=()`` 的模板：**给它什么参数，它都说同一句话**。

    ``scan_start`` 就是这种 —— StartScan 的扫描参数是仪器当前的那份，不在
    ``params`` 里，所以这句话不带数字**不是降级**，是它本来就没有可说的数字。

    这条断言故意**不去查 StartScan 有几个参数**：那是别人的 schema，会长
    （2026-08-11 它就多了一个 ``direction``），而模板的正确性不该挂在别人的
    参数表大小上。要钉的是模板自己的性质：它不会因为多来了一个字段就开始
    声称一个它没读过的值。
    """
    for kind, tpl in T.TEMPLATES.items():
        if tpl.requires:
            continue
        plain = T.render(kind, {})
        noisy = T.render(kind, {"params": {"direction": "down", "size_m": 5e-8},
                                "summary_zh": ""})
        assert plain is not None and noisy is not None
        assert plain.text == noisy.text, (
            f"{kind} 的句子随参数变了，但它没有声明任何 requires —— "
            f"那意味着它在用一个没被 requires 保护的字段")
        assert plain.degraded is False, f"{kind} 无需必需字段，不该报降级"


# ── ② / ③ render 体里不许有字面数字 ────────────────────────────────────


def _walk_sentence(node: ast.AST):
    """遍历句子代码,**剪掉 f-string 的格式串子树**。

    ``ast.walk`` 是广度优先、不支持剪枝:在它身上写 ``continue`` 只跳过当前节点,
    子节点照样会被 yield 出来。所以 ``f"{x:.3f}"`` 里那个 ``'.3f'`` 仍会被当成
    「句子里写死了数字」。自己走一遍才剪得掉。

    (剪的是 ``format_spec``,**不是** ``FormattedValue`` 整枝 —— 后者会把被格式化
     的那个表达式也一起剪掉,而那里面正可能藏着一个 ``* 1e9``。)
    """
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        for child in ast.iter_child_nodes(cur):
            if isinstance(cur, ast.FormattedValue) and child is cur.format_spec:
                continue
            stack.append(child)


def _strip_docstring(fn: ast.AST) -> list[ast.AST]:
    """函数体,去掉 docstring。

    docstring 里当然会有数字(它就是用来解释那些数字的)。不去掉的话,
    每一条带说明的 helper 都会被判违规,而闸门一旦开始误报,下一步就是被放宽 ——
    那才是它真正的死法。
    """
    body = list(getattr(fn, "body", []))
    if body and isinstance(body[0], ast.Expr)             and isinstance(body[0].value, ast.Constant)             and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _render_bodies() -> list[tuple[str, list[ast.AST]]]:
    """每个 ``render=`` 背后**真正会跑到的那段句子代码**。

    ⚠️ 2026-08-17 扩了射程。原来只取 ``render=`` 后面挂的那个表达式,于是
    ``render=lambda d: ...`` 查得到,而 ``render=_cluster_sentence`` 只看到一个
    ``Name`` 节点 —— **句子体整个在闸门视野之外**。

    这不是理论风险:我当天写 ``_find_spot_sentence`` / ``_pulse_result_sentence``
    时,把七处 ``* 1e9`` 全写在了那两个函数里,而闸门一声不吭地绿着。
    **把违规挪到闸门看不见的地方,比原地违规更坏** —— 前者还会让人以为查过了。

    ## 只跟一层,而且只跟「句子体」

    跟进 ``render=`` 直接指向的那个函数,**不再往里跟它调用的 helper**
    (``si`` / ``num`` / ``nm`` / ``count_zh``)。那些 helper 正是这道闸门要你去
    用的东西 —— 它们内部当然有 ``1e9``,那是**给它起了名字之后**该待的地方,
    而且它们各自有测试。跟进去只会让闸门骂自己的解决方案。

    (第一版扩射程时我就是这么错的:一口气跟进了所有被引用的名字,于是
     ``num`` 的 docstring 都成了违规项。误报的闸门下一步就是被放宽。)
    """
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    funcs: dict[str, ast.AST] = {
        n.name: n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def _followed(render: ast.AST) -> "str | None":
        """``render=`` 直接指的那个模块级句子函数名(没有就 None)。"""
        if isinstance(render, ast.Name):
            return render.id
        # ``render=lambda d: _foo(d)`` —— 一层薄包装,同样要跟进去。
        if isinstance(render, ast.Lambda) and isinstance(render.body, ast.Call)                 and isinstance(render.body.func, ast.Name):
            return render.body.func.id
        return None

    out: list[tuple[str, list[ast.AST]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") != "Template":
            continue
        kind = ""
        render = None
        for kw in node.keywords:
            if kw.arg == "kind" and isinstance(kw.value, ast.Constant):
                kind = str(kw.value.value)
            if kw.arg == "render":
                render = kw.value
        if render is None:
            continue

        name = _followed(render)
        if name is None:
            out.append((kind, [render]))          # 句子直接写在 lambda 里
            continue
        target = funcs.get(name)
        # 跟不进去就是**失败**,不是跳过:一个「找不到就当过了」的闸门,
        # 在有人换个写法之后会永远绿着 —— 而那正是它要防的事。
        assert target is not None or not name.startswith("_"), (
            f"{kind}: render 指向 {name},但这个模块里找不到它的定义 —— "
            f"闸门跟不进去,句子体等于没被检查")
        if target is None:
            out.append((kind, [render]))          # 指向外部 helper,按表达式查
        else:
            out.append((f"{kind}→{name}", _strip_docstring(target)))
    return out


def test_the_gate_can_actually_see_the_templates():
    """闸门自检 —— 一个扫不到任何东西的闸门永远是绿的。

    本仓刚为这句话付过学费（``_check_spec_preimport.py`` 引用了早就删掉的模块，
    烂了很久没人发现，正因为没有任何东西跑它）。
    """
    bodies = _render_bodies()
    kinds = {k.split("→")[0] for k, _ in bodies}
    assert kinds == set(T.TEMPLATES), (
        f"AST 扫到的 kind 与注册表对不上 —— 差集 "
        f"{kinds ^ set(T.TEMPLATES)};有模板不是用 `Template(kind=..., render=...)` "
        f"的写法建的，闸门看不见它")
    # 具名 render 函数必须被跟进去 —— 否则句子体在视野之外(2026-08-17 的教训)。
    followed = [k for k, _ in bodies if "→" in k]
    assert followed, (
        "一个具名 render 函数都没跟进去 —— 要么写法变了，要么跟踪那段坏了。"
        "句子体一旦跟不进去，这道闸门就只剩下 lambda 那几条还在被检查。")


def test_no_numeric_literal_anywhere_in_a_render_body():
    """render 体里出现任何字面数字都算违规 —— 换算一律走 si()/pct()/count_zh()。

    这条比「只禁字符串里的数字」严一档，且是故意的：``v * 1e12`` 里那个 1e12
    不在字符串里，但它正是「3e-12 → 3 米」那一类事故的形状。要算就给它起个名字，
    放到模块级 helper 里 —— helper 有名字就会有测试。
    """
    bad: list[str] = []
    for kind, nodes in _render_bodies():
        for stmt in nodes:
            for n in _walk_sentence(stmt):
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                        and not isinstance(n.value, bool):
                    bad.append(f"{kind}: 字面数字 {n.value!r}")
    assert not bad, (
        "\n".join(bad)
        + "\n→ 用 si()/nm()/ratio()/count_zh() 这些模块级 helper，别在句子里算。")


def test_no_digits_inside_the_chinese_sentences():
    """句子里不许出现数字字面量；格式串（``:.4g``）不算。

    ``f"我们要打一发 10 V 脉冲"`` 读起来和真的完全一样，而它说的是一个
    **没有任何人下发过**的值。
    """
    bad: list[str] = []
    for kind, nodes in _render_bodies():
        for stmt in nodes:
            # ``_walk_sentence`` 已经把格式串(``:.3f``)剪掉了，
            # 剩下的就是真正会被用户读到的文字。
            for n in _walk_sentence(stmt):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    if any(ch.isdigit() for ch in n.value):
                        bad.append(f"{kind}: 句子里写死了数字 {n.value!r}")
    assert not bad, "\n".join(bad)


def test_fallback_sentences_carry_no_numbers():
    """``fallback`` 是「必需字段没读到」时说的那句话，它**必须**不带数字。

    缺 duration_s 时说「我们要打一发 10 V 脉冲」就是把一个没读到的值说成读到的。
    """
    bad = [t.kind for t in T.TEMPLATES.values()
           if any(ch.isdigit() for ch in t.fallback)]
    assert not bad, f"这些 fallback 里带了数字：{bad}"


def test_narrate_has_no_text_parameter():
    """``narrate()`` 里**没有**一个能写进一句现成话的参数。

    这不是风格问题：只要有 ``text=``，一个偷懒的技能（或模型）就有地方写一个
    编出来的数字，而模板这一整套约束就绕过去了。
    """
    import inspect

    from mast.chat.narration import Sink, narrate

    for fn in (narrate, Sink.narrate):
        params = set(inspect.signature(fn).parameters)
        assert "text" not in params, f"{fn} 多了一个 text= 参数"
        assert "message" not in params
        assert "sentence" not in params


# ── si() 与 core.si_quantity 是同一个物理量 ────────────────────────────


@pytest.mark.parametrize("value", [3e-12, 5e-8, 0.5, 10.0, -0.05, 1.5e-9, 2e6])
def test_si_agrees_with_the_repo_wide_formatter(value):
    """``si()`` 只换了读法，没换值。

    它和 ``format_si`` 是两个目标（可读 vs 可反解），所以字符串不同；但**物理量
    必须相同**，否则旁白里的数字和仪器里的数字就是两回事了。
    """
    shown = T.si(value, "V")
    mantissa, _, tail = shown.partition(" ")
    prefix = tail[:-1]                       # 去掉单位 V
    back = parse_si(f"{mantissa}{prefix}") if prefix else float(mantissa)
    assert back == pytest.approx(value, rel=1e-6)
    assert parse_si(format_si(value)) == pytest.approx(value, rel=1e-6)


def test_si_reads_like_a_person_wrote_it():
    """这几条是 ``format_si`` 给不出来的读法 —— 也是不直接用它的理由。"""
    assert T.si(10.0, "V") == "10 V"          # format_si(10.0) == "10000m"
    assert T.si(0.5, "s") == "500 ms"
    assert T.si(3e-12, "m") == "3 pm"
    assert T.si(5e-8, "m") == "50 nm"


# ── ④ 术语:用户看到的字里不许有口语说法 ────────────────────────────────
#
# 要求:统一使用术语——「多针尖」而非「针尖劈了」等口语说法，全文尽量使用术语。
#
# 这条闸门的**射程是句子,不是文件**:docstring、注释、日志、以及知识库里
# 「自旋-轨道劈裂」那种**真正的物理术语**都不在内 —— 复用上面的 `_render_bodies`,
# 它只看 render 背后真正会跑到的那段句子代码。
#
# 为什么钉:措辞是最容易在一次「顺手润色」里悄悄退回去的东西,而退回去之后
# 没有任何东西会报错。

#: 口语 → 该用的术语。**每一条都要写出替代词** —— 一张只有黑名单的表会让下一个人
#: 知道不许写什么,却不知道该写什么,于是他会绕过去而不是改对。
COLLOQUIAL: dict[str, str] = {
    "劈了": "多针尖",
    "劈开": "出现重影 / 多针尖",
    "劈裂": "重影(判据语境;物理上的能级劈裂不在此列)",
    "单尖": "单针尖",
    "打动了": "改变了针尖",
    "没打动": "未改变针尖",
    "针尖动了": "针尖已改变",
    "没扎上": "未接触",
    "图上没东西": "图上没有可判读的团簇",
    "换地方": "换落点 / 换位置",
    "变浅": "减小下压深度",
    "搬多了": "材料转移过量",
    "往表面里": "下压深度",
    "收工": "判据要求 / 判定完成",
    "钝了": "锐度不足",
    "没过验收": "未通过验收",
    "换区了": "已粗动换区",
    "当场停手": "立即停止",
    "请人来看": "需人工介入排查",
    "没修好": "未达标",
    "撞到失控保险": "触发失控保险",
    "停在这里": "流程停止",
}


def test_no_colloquialisms_in_operator_facing_sentences():
    """旁白句子里不许出现口语说法 —— 每一条都有对应的术语。"""
    bad: list[str] = []
    for kind, nodes in _render_bodies():
        for stmt in nodes:
            for n in _walk_sentence(stmt):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    for word, better in COLLOQUIAL.items():
                        if word in n.value:
                            bad.append(f"{kind}: 「{word}」→ 用「{better}」"
                                       f"  ({n.value[:40]!r})")
    assert not bad, "\n".join(bad) + "\n→ 能使用术语尽量使用术语。"


def test_the_outcome_table_is_written_in_terminology_too():
    """站点/整跑结局那张表也是用户看到的字,同一把尺子量。"""
    bad = [f"{code}: 「{w}」→ 用「{b}」  ({txt!r})"
           for code, txt in T.FORGE_OUTCOME_ZH.items()
           for w, b in COLLOQUIAL.items() if w in txt]
    assert not bad, "\n".join(bad)


def test_the_gate_can_actually_see_a_colloquialism():
    """闸门自检:它真的认得出违规。

    一条永远为真的断言和没有断言是同一件事 —— 而这条闸门扫的是 AST,
    最容易的坏法是 `_render_bodies` 有一天不再返回句子体(2026-08-17 真发生过)。
    """
    assert _render_bodies(), "一个 render 体都没扫到 —— 闸门在空跑"
    sample = ast.parse('x = "这根针尖劈了"').body
    found = [w for _n in _walk_sentence(sample[0])
             for w in COLLOQUIAL
             if isinstance(_n, ast.Constant) and isinstance(_n.value, str)
             and w in _n.value]
    assert "劈了" in found, "闸门认不出「劈了」——判据坏了"
