"""地图工具的米量纲参数必须具有量级护栏，并接受明确的 SI 字符串。

逐一核验所有注册工具的参数分类、护栏调用、字符串支持和已有正确数值输入。
扫描范围覆盖全部工具模块，防止只检查单个文件形成盲区。"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.agents._shared.meta_tools import (  # noqa: E402
    _METRE_POSITION_ARGS,
    _METRE_SIZE_ARGS,
    _check_metre_magnitudes,
    resolve_metre_args,
)

_PKG = Path(_MASTV2_ROOT) / "mast"
_GUARDS = ("_check_metre_magnitudes", "_check_plan_step_magnitudes",
           "resolve_metre_args")
#: 只有这两个别名会把「怎么写」的说明带进 provider payload(实测
#: ``Annotated[..., Field(description=...)]`` 能穿过 ``convert_to_openai_tool``)。
#: 手写一个 ``float | str`` 能通过类型校验却对模型一言不发,所以闸门认别名。
_METRE_ANNOTATIONS = ("MetrePos", "MetreSize")

#: 明确豁免:带 ``*_m`` 参数但**不应该**判量级的工具。空表也要在,因为一条
#: 「只编码了必须有护栏」的闸门,下一轮会被一个正当的例外逼着整条删掉。
_EXEMPT: dict[str, str] = {}


def _tool_functions() -> list[tuple[Path, ast.FunctionDef]]:
    """全仓所有 ``@tool(...)`` 函数 —— 不是某一个文件里的。"""
    out: list[tuple[Path, ast.FunctionDef]] = []
    for path in sorted(_PKG.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(isinstance(d, ast.Call) and getattr(d.func, "id", None) == "tool"
                   for d in node.decorator_list):
                out.append((path, node))
    return out


def _metre_params(fn: ast.FunctionDef) -> list[ast.arg]:
    args = fn.args
    return [a for a in (list(args.posonlyargs) + list(args.args)
                        + list(args.kwonlyargs)) if a.arg.endswith("_m")]


def _body(path: Path, fn: ast.FunctionDef) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[fn.lineno - 1:fn.end_lineno])


# ── 结构闸门 ────────────────────────────────────────────────────────────

def test_the_sweep_itself_found_something():
    """闸门自检:一条匹配不到任何东西的闸门会一直绿。

    见 [[mutation_must_prove_it_landed]] —— 先证明这把梳子梳到了头发。
    额外要求**跨到第二个文件**:上一版只梳 meta_tools.py,梳子本身是对的,
    但它够不到 find_flat_region,于是一个真实的洞在绿灯下活了一天。
    """
    fns = _tool_functions()
    assert len(fns) >= 40, f"只解析出 {len(fns)} 个 @tool,AST 姿势多半错了"
    with_metre = [(p.name, f.name) for p, f in fns if _metre_params(f)]
    assert len(with_metre) >= 4, f"带米量纲参数的工具只找到 {with_metre}"
    files = {p for p, f in fns if _metre_params(f)}
    assert len(files) >= 2, (
        f"米量纲 @tool 只在 {files} 一个文件里 —— 闸门退回成了「梳一个文件」")


def test_every_tool_with_a_metre_arg_calls_the_magnitude_guard():
    missing = []
    for path, fn in _tool_functions():
        if not _metre_params(fn) or fn.name in _EXEMPT:
            continue
        if not any(g in _body(path, fn) for g in _GUARDS):
            missing.append(
                f"{path.name}::{fn.name}({','.join(a.arg for a in _metre_params(fn))})")
    assert not missing, (
        "这些工具收米量纲参数却不判量级 —— 量级错了它们不会报错,"
        f"会给一个看起来合理的错答案:{missing}")


def test_every_metre_arg_name_is_classified():
    """没分类的名字连判都不会被判 —— 那正是 radius_m 的死法。"""
    known = set(_METRE_POSITION_ARGS) | set(_METRE_SIZE_ARGS)
    unclassified = sorted({
        a.arg for _p, fn in _tool_functions() if fn.name not in _EXEMPT
        for a in _metre_params(fn)} - known)
    assert not unclassified, (
        f"这些米量纲参数名不在 _METRE_POSITION_ARGS / _METRE_SIZE_ARGS 里:"
        f"{unclassified} —— 护栏按名字判,不在表里 = 送进去也看不见")


def test_every_metre_arg_accepts_the_si_string_form():
    """裸 ``float`` = 模型**只有**一种写法,而它写不对那一种。

    这是护栏的另一半。护栏是事后拦截,拦得再准也只是把错的挡回去;模型下一次
    还是只能写 ``5e-8``。给它 ``'50n'`` 这条写法是 [[remove_incentive_not_persuade]]:
    移除诱因,不要在提示词里说服它。
    """
    bare = []
    for path, fn in _tool_functions():
        if fn.name in _EXEMPT:
            continue
        for a in _metre_params(fn):
            ann = ast.unparse(a.annotation) if a.annotation else "<none>"
            if ann not in _METRE_ANNOTATIONS:
                bare.append(f"{path.name}::{fn.name}.{a.arg}: {ann}")
    assert not bare, (
        f"这些米量纲参数不是 {_METRE_ANNOTATIONS} 之一:{bare} —— "
        "裸 float 收不下 '50n',模型于是只剩「自己写对 5e-8」这一条路;"
        "而只有这两个别名会把写法说明带进 provider payload。")


# ── 事后拦截:量级 ──────────────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("frame_size_m", 8.0),   # 合成的米级错误输入，应被量级护栏拒绝
    ("radius_m", 1.4),
    ("x_m", 2.0),
    ("min_separation_m", 0.1),            # find_flat_region:十厘米的「最小间距」
])
def test_a_metre_value_eight_orders_out_is_refused_not_answered(field, value):
    errors, _warn = _check_metre_magnitudes("t", {field: value})
    assert errors, f"{field}={value} 米被放行了"
    assert field in errors[0] and "量级" in errors[0]


def test_a_legitimate_nanometre_value_still_passes():
    """护栏不能把正常调用也挡掉 —— 只朝一侧失败的判据分不出好坏。"""
    errors, warnings = _check_metre_magnitudes(
        "t", {"x_m": -1.32e-7, "y_m": -8.7e-8, "radius_m": 50e-9,
              "frame_size_m": 5e-8, "min_separation_m": 1e-7})
    assert errors == [] and warnings == []


def test_nan_is_an_error_not_a_skip():
    """NaN 过得了 float(),然后毒化下游每一次比较。"""
    errors, _ = _check_metre_magnitudes("t", {"radius_m": float("nan")})
    assert errors and "有限数值" in errors[0]


# ── 事前的那一半:安全写法 ──────────────────────────────────────────────

@pytest.mark.parametrize("written,expect", [
    ("50n", 50e-9),          # 新写法:量级由字母承载
    ("200n", 200e-9),
    ("1u", 1e-6),
    ("-132n", -132e-9),
    ("0p", 0.0),             # 有意的零
])
def test_the_si_string_form_parses(written, expect):
    parsed, errors = resolve_metre_args("t", {"x_m": written})
    assert errors == []
    assert parsed["x_m"] == pytest.approx(expect, rel=1e-12, abs=1e-18)


@pytest.mark.parametrize("written", [5e-8, 5.0e-08, 1.4e-06, 0.0])
def test_the_number_form_still_works(written):
    """``5e-8`` 今天是对的写法,加了新写法不许把旧写法弄坏。"""
    parsed, errors = resolve_metre_args("t", {"frame_size_m": written})
    assert errors == []
    assert parsed["frame_size_m"] == pytest.approx(written)


@pytest.mark.parametrize("written,expect", [
    ("5e-8", 5e-8), ("0.00000005", 5e-8), ("1.4e-06", 1.4e-6), ("0", 0.0),
])
def test_the_string_number_form_still_works(written, expect):
    """字符串形式的科学计数法今天**也**是能用的(pydantic 宽松转换),同样不许弄坏。

    这条不是理论:``float`` 字段实测接受 ``'5e-8'``。把它拒掉就是拿一个新写法
    换掉一个在用的写法。
    """
    parsed, errors = resolve_metre_args("t", {"x_m": written})
    assert errors == []
    assert parsed["x_m"] == pytest.approx(expect, abs=1e-18)


@pytest.mark.parametrize("junk", ["50nm", "abc", "", "n", "5 nm", "1e-7 m"])
def test_unparseable_never_falls_back_to_a_default(junk):
    """解析不了就报错 —— 不许悄悄变成默认值。

    ``lookup(name) || DEFAULT`` 的形状:``'50nm'``(多写了个 m)如果回退成
    ``radius_m`` 的 50 nm 默认值,和写对了长得一模一样,没有任何人会发现。
    """
    parsed, errors = resolve_metre_args("t", {"radius_m": junk})
    assert errors, f"{junk!r} 被静静接受了"
    assert "radius_m" in errors[0]
    assert "radius_m" not in parsed, "解析失败还往结果里塞了一个值"


def test_the_string_is_parsed_before_the_magnitude_is_judged():
    """顺序是承重的,而且**只有这个值证得出来**。

    ``_check_metre_magnitudes`` 读的是 ``float(raw)``,``float('5k')`` 会抛,而它的
    except 分支是 ``continue`` —— 也就是先判后解析的话,一个字符串会被护栏**静静
    跳过**,然后解析成 5000 米照样落库。

    注意 ``'5'`` 证不出这件事(两种顺序都会拒),``'50n'`` 也证不出(两种顺序都放行)。
    要证顺序,必须找一个**解析后才显形**的量级错误。
    """
    _parsed, errors = resolve_metre_args("t", {"radius_m": "5k"})
    assert errors, "'5k'(=5000 米)在字符串形式下溜过了量级护栏"
    assert "量级" in errors[0]


# ── 纳米单位的那个工具:换算之后也要过同一道闸 ────────────────────────

def _plan_scan_batch():
    from mast.agents._shared.meta_tools import make_meta_tools
    return {t.name: t for t in make_meta_tools(lambda: {})}["plan_scan_batch"]


@pytest.mark.parametrize("kw,wrote", [
    ({"kind": "survey", "n_images": 2, "size_nm": 5e7}, "size_nm"),
    ({"kind": "zoomin", "final_size_nm": 1e9, "feature": "defect"}, "final_size_nm"),
])
def test_plan_scan_batch_judges_the_converted_metres(kw, wrote):
    """``size_nm`` 是纳米,但 ``plan_batch`` 对尺寸**只判 <= 0**,没有上界。

    换算完就是米,所以送同一道护栏 —— 不新引一个阈值。5e7 nm = 5 cm 的扫描框
    原本会一路排完计划、返回 plan_json、还叫模型「原样执行」。
    """
    import json
    res = json.loads(_plan_scan_batch().invoke(kw))
    assert res.get("success") is False
    assert wrote in res["error"] and "纳米" in res["error"]


def test_plan_scan_batch_error_speaks_nanometres_not_metres():
    """借护栏的裁决,不借它的措辞。

    护栏自带的提示是按米写的(「若本意是 0.05 µm,应写 5e-08」)。这个参数是
    纳米,照搬过去就是一句**对这个参数错误**的建议 —— 模型照做会再错一次,而且
    是在一句看起来很权威的提示指导下错的。
    """
    import json
    res = json.loads(_plan_scan_batch().invoke(
        {"kind": "survey", "n_images": 2, "size_nm": 5e7}))
    assert "应写 5e-08" not in res["error"], "把米量纲的换算建议原样搬给了纳米参数"
    assert "50 nm 的图就写 50" in res["error"]


def test_a_good_plan_is_not_collateral_damage():
    """闸门不能把正常的 50 nm 规划也挡掉。"""
    import json
    res = json.loads(_plan_scan_batch().invoke(
        {"kind": "repeat", "n_images": 2, "size_nm": 20}))
    assert res.get("success") is True
    assert res.get("published_to_map") == 2
    assert "publish_refused" not in res


def test_strict_prefix_would_have_broken_a_working_form():
    """被否掉的方案钉在这里:米量纲**不**强制 SI 前缀。

    见 [[pin_rejected_designs_as_tests]]。skill 那一侧对米量纲是 ``strict=True``
    (前缀不可省),照抄过来看起来更安全,但会把今天在用的 ``'5e-8'`` 拒掉。
    这条路上不需要前缀当校验和:掉了前缀的 ``'50n'`` → ``'50'`` → 50 米,正好落进
    量级护栏的拒绝区。**量级护栏对米量纲比前缀规则更强 —— 它不管你怎么写的。**

    要推翻这个决定,需要回答:有哪个米量纲的合法值,量级护栏放行、而它又是一次
    掉前缀的产物?([[make_decisions_refutable]])
    """
    from mast.core.si_quantity import SIParseError, parse_quantity
    with pytest.raises(SIParseError):
        parse_quantity("5e-8", strict=True, what="x_m")      # strict 会拒掉它
    assert parse_quantity("5e-8", strict=False, what="x_m") == pytest.approx(5e-8)
    # 而掉了前缀的写法照样被拦下 —— 拦它的是量级,不是前缀规则。
    for dropped in ("50", "200", "1"):
        _p, errors = resolve_metre_args("t", {"radius_m": dropped})
        assert errors, f"掉前缀的 {dropped!r} 没被量级护栏接住"
