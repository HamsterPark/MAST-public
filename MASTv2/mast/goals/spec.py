"""谓词目录与 ``done_when`` 的解析 —— 闭集在这里，判断不在这里。

形状：``done_when`` 要么是一条谓词，要么是 ``{all|any|not: [...]}`` 组合；
列表是 ``all`` 的简写（最常见的写法，也是最不容易写错的那个默认）。

组合算子直接取 :data:`mast.conduct.spec.RULE_COMBINATORS`，不另开一份 ——
「同一个记号两套语法」是本仓反复付过的税。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from mast.conduct.spec import RULE_COMBINATORS, RuleLeaf, RuleTree


class GoalSpecError(ValueError):
    """``done_when`` 不合法。带着逐条理由 —— 它要原样回到调用方眼前。"""

    def __init__(self, errors: "list[str]") -> None:
        super().__init__("；".join(errors) if errors else "done_when 不合法")
        self.errors = list(errors)


# ── 解析后的形状（纯 JSON 可回环，可哈希） ────────────────────────────

@dataclass(frozen=True)
class Predicate:
    """一条谓词：``kind`` + 已归一化的参数。"""

    kind: str
    args: tuple[tuple[str, Any], ...] = ()

    def arg(self, name: str, default: Any = None) -> Any:
        for k, v in self.args:
            if k == name:
                return v
        return default

    def as_dict(self) -> dict:
        return {"kind": self.kind, **{k: v for k, v in self.args}}


@dataclass(frozen=True)
class Combo:
    """``all`` / ``any`` / ``not`` 组合。"""

    op: str
    children: tuple = ()

    def as_dict(self) -> dict:
        return {self.op: [c.as_dict() for c in self.children]}


DoneWhen = Predicate | Combo


# ── 目录 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PredSpec:
    """一条谓词的完整契约。

    ``leaf`` 把谓词编成一个 ``RuleLeaf``；``evidence`` 是这条谓词需要收集器
    填进证据包的字段名。两者必须对得上——``evidence`` 里的名字就是 ``leaf``
    要 lookup 的名字，写岔了的后果是**永远 UNDECIDABLE**（一个永远判不了的
    判据看起来像「还没到」，而不是像一个 bug）。
    """

    kind: str
    reads: str
    required: tuple[str, ...] = ()
    optional: "Mapping[str, Any]" = None  # type: ignore[assignment]
    one_of: tuple[tuple[str, ...], ...] = ()
    evidence: tuple[str, ...] = ()
    leaf: "Callable[[Predicate], RuleLeaf]" = None  # type: ignore[assignment]
    validate: "Callable[[Predicate, Any], list[str]] | None" = None
    describe: "Callable[[Predicate], str]" = None  # type: ignore[assignment]

    def arg_names(self) -> tuple[str, ...]:
        opt = tuple((self.optional or {}).keys())
        flat = tuple(n for g in self.one_of for n in g)
        return self.required + opt + flat


def _pos_int(pred: Predicate, name: str) -> "list[str]":
    v = pred.arg(name)
    try:
        n = int(v)
    except Exception:  # noqa: BLE001
        return [f"{name} 要一个整数，收到 {v!r}"]
    if n < 1:
        return [f"{name} 要 >= 1，收到 {n}"]
    return []


def _artifact_field(pred: Predicate, artifact_fields) -> "list[str]":
    field = pred.arg("field")
    if not isinstance(field, str) or not field.strip():
        return ["field 要一个非空产物名"]
    if artifact_fields is None:
        # 读不到可等产物清单 —— **拒绝**，不放行。写入口拒绝无害（今天的行为
        # 就是「没有 done_when」）；读入口拒绝会让这条判据变成 unknown，而
        # unknown 在唤醒准入上是「照旧唤醒」，同样是安全方向。
        return ["读不到可等待产物清单（artifact_channel 不可用），无法校验 field"]
    if field not in artifact_fields:
        return [f"field={field!r} 不是可等待的产物；可选：{sorted(artifact_fields)}"]
    return []


def _default_artifact_fields():
    """可等待产物的闭集，惰性取自产物通道（那里是真源，不在这里抄第二份）。"""
    try:
        from mast.agents._shared.artifact_channel import WAITABLE_FIELDS

        return tuple(WAITABLE_FIELDS)
    except Exception:  # noqa: BLE001
        return None


CATALOG: "dict[str, PredSpec]" = {
    "artifact_present": PredSpec(
        kind="artifact_present",
        reads="产物通道 + 实验文件夹（state 有=铁证；state 没有再问磁盘）",
        required=("field",),
        optional={"allow_preexisting": False},
        evidence=("new_since_baseline",),
        leaf=lambda p: RuleLeaf(field="new_since_baseline", op="==", value=True),
        validate=_artifact_field,
        describe=lambda p: f"产出了 {p.arg('field')}"
                           + ("（含目标设定之前就有的）"
                              if p.arg("allow_preexisting") else ""),
    ),
    "artifact_count_at_least": PredSpec(
        kind="artifact_count_at_least",
        reads="实验文件夹里该产物类的计数（相对基线的增量）",
        required=("field", "n"),
        evidence=("delta_count",),
        leaf=lambda p: RuleLeaf(field="delta_count", op=">=",
                                value=int(p.arg("n"))),
        validate=lambda p, af: _artifact_field(p, af) + _pos_int(p, "n"),
        describe=lambda p: f"至少又产出了 {p.arg('n')} 份 {p.arg('field')}",
    ),
    "operator_confirmed": PredSpec(
        kind="operator_confirmed",
        reads="用户在本次 run 里的确认（ask_operator）",
        evidence=("answer",),
        leaf=lambda p: RuleLeaf(field="answer", op="==", value="yes"),
        describe=lambda p: "用户确认「够了」",
    ),
    "conduct_completed": PredSpec(
        kind="conduct_completed",
        reads="conducts 表（只读，conduct 引擎关着也读得到）",
        one_of=(("spec_id", "conduct_id"),),
        optional={"min_count": 1},
        evidence=("completed_count",),
        leaf=lambda p: RuleLeaf(field="completed_count", op=">=",
                                value=int(p.arg("min_count") or 1)),
        validate=lambda p, af: _pos_int(p, "min_count"),
        describe=lambda p: (
            f"跑完了 {p.arg('conduct_id') or p.arg('spec_id')}"
            + (f"（{p.arg('min_count')} 份）"
               if int(p.arg("min_count") or 1) > 1 else "")),
    ),
    "best_frame_settled": PredSpec(
        kind="best_frame_settled",
        reads="best_frames/<tag>.json（TrackBestFrame 落的那份，单写者）",
        required=("tag",),
        optional={"dry_limit": None},
        evidence=("good_enough_to_stop",),
        leaf=lambda p: RuleLeaf(field="good_enough_to_stop", op="==", value=True),
        # ``dry_limit`` 给了就必须是正整数：``0`` 会让 ``dry_rounds >= 0``
        # 恒真 —— 连记录文件都不存在时这条谓词就 done。别的整数参数（``n`` /
        # ``min_count``）都过 ``_pos_int``，第一版只漏了它。
        validate=lambda p, af: (
            ([] if isinstance(p.arg("tag"), str) and p.arg("tag").strip()
             else ["tag 要一个非空字符串"])
            + ([] if p.arg("dry_limit") is None else _pos_int(p, "dry_limit"))),
        describe=lambda p: f"追猎 {p.arg('tag')} 连着若干轮没更好了（差不多了）",
    ),
    "claims_supported": PredSpec(
        kind="claims_supported",
        reads="v2 记录库的 claims 表（status ∈ supported/verified）",
        required=("min_count",),
        evidence=("count",),
        leaf=lambda p: RuleLeaf(field="count", op=">=",
                                value=int(p.arg("min_count"))),
        validate=lambda p, af: _pos_int(p, "min_count"),
        describe=lambda p: f"至少 {p.arg('min_count')} 条论断被证据支持",
    ),
}


def catalog_json() -> "list[dict]":
    """给模型/前端看的目录。**在调用点渲染**，不靠记忆。

    这是「移除诱因」的那一半：模型在它写错的地方立刻拿到闭集，而不是被一句
    「请从目录里选」劝说。
    """
    out = []
    for spec in CATALOG.values():
        args: dict[str, Any] = {n: "必填" for n in spec.required}
        for g in spec.one_of:
            for n in g:
                args[n] = f"这几个里必须且只能有一个：{list(g)}"
        for n, d in (spec.optional or {}).items():
            args[n] = f"可选，默认 {d!r}"
        out.append({"kind": spec.kind, "args": args, "reads": spec.reads})
    return out


#: 给模型/前端看的示例。**刻意不用任何真实样品名** —— MAST 是一套 STM 系统，
#: 不是为某一种样品写的；样品名只许出现在 conduct 模板层。结构测试
#: ``test_no_sample_names_in_generic_layer`` 钉着这条边界，而且这次就是它逮到我
#: 在这里写了一个真实样品的模板名（**它连注释一起扫**，这是对的：注释里的名字
#: 照样会被复制进提示词和文档）。换成两条与样品无关的谓词之后，示例本身也更能
#: 说明「组合」是怎么写的。
EXAMPLE: dict = {
    "all": [
        {"kind": "artifact_present", "field": "analysis"},
        {"kind": "claims_supported", "min_count": 2},
    ]
}


# ── 解析 ──────────────────────────────────────────────────────────────

def _norm_one(raw: Any, path: str, artifact_fields,
              errors: "list[str]") -> "Predicate | None":
    if not isinstance(raw, Mapping):
        errors.append(f"[{path}] 要一个对象，收到 {type(raw).__name__}")
        return None
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in CATALOG:
        errors.append(
            f"[{path}] kind={kind!r} 不在目录；可选：{sorted(CATALOG)}")
        return None
    spec = CATALOG[kind]
    known = set(spec.arg_names()) | {"kind"}
    extra = [k for k in raw if k not in known]
    if extra:
        errors.append(f"[{path}] {kind} 不认识这些参数：{sorted(extra)}")
    args: dict[str, Any] = {}
    for name in spec.required:
        if name not in raw:
            errors.append(f"[{path}] {kind} 缺必填参数 {name}")
        else:
            args[name] = raw[name]
    for group in spec.one_of:
        present = [n for n in group if raw.get(n) not in (None, "")]
        if len(present) != 1:
            errors.append(
                f"[{path}] {kind} 需要 {list(group)} 中**恰好一个**，"
                f"收到 {present or '零个'}")
        for n in present:
            args[n] = raw[n]
    for name, default in (spec.optional or {}).items():
        args[name] = raw.get(name, default)
    pred = Predicate(kind=kind, args=tuple(sorted(args.items())))
    if spec.validate is not None:
        for msg in spec.validate(pred, artifact_fields):
            errors.append(f"[{path}] {kind} {msg}")
    return pred


def _norm(raw: Any, path: str, artifact_fields,
          errors: "list[str]", depth: int = 0) -> "DoneWhen | None":
    if depth > 4:
        errors.append(f"[{path}] 嵌套太深（>4 层）")
        return None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        # 列表 = all 的简写。**默认必须是 all**：any 会让一条便宜的谓词短路
        # 整个目标，那正是「太早停」。
        before = len(errors)
        kids = [k for k in (_norm(r, f"{path}[{i}]", artifact_fields, errors,
                                  depth + 1)
                            for i, r in enumerate(raw)) if k is not None]
        if not kids:
            # 只有在**真的什么都没写**时才说「空列表」。子项全部不合法时，
            # 每一条的理由已经在 errors 里了 —— 再补一句「你没写判据」是在
            # 描述一件没发生的事，而读报文的人（模型）会照着它去改错地方。
            if len(errors) == before:
                errors.append(f"[{path}] 空列表 —— 没有判据就别写 done_when")
            return None
        return Combo(op="all", children=tuple(kids))
    if isinstance(raw, Mapping):
        ops = [k for k in raw if k in RULE_COMBINATORS]
        if ops and "kind" in raw:
            errors.append(f"[{path}] 既像组合又像谓词（同时有 kind 和 {ops}）")
            return None
        if len(ops) > 1:
            errors.append(f"[{path}] 一个节点只能有一个组合算子，收到 {ops}")
            return None
        if ops:
            op = ops[0]
            body = raw[op]
            extra = [k for k in raw if k != op]
            if extra:
                errors.append(f"[{path}] 组合节点多了这些键：{sorted(extra)}")
            items = body if isinstance(body, Sequence) and not isinstance(
                body, (str, bytes)) else [body]
            before = len(errors)
            kids = [k for k in (_norm(r, f"{path}.{op}[{i}]", artifact_fields,
                                      errors, depth + 1)
                                for i, r in enumerate(items)) if k is not None]
            if not kids:
                if len(errors) == before:      # 同上：别描述没发生的事
                    errors.append(f"[{path}] {op} 至少要有一个子判据")
                return None
            if op == "not" and len(kids) != 1:
                errors.append(f"[{path}] not 只能有一个子判据，收到 {len(kids)}")
                return None
            return Combo(op=op, children=tuple(kids))
        return _norm_one(raw, path, artifact_fields, errors)
    errors.append(f"[{path}] 看不懂的节点：{type(raw).__name__}")
    return None


def normalise_done_when(
    raw: Any, *, artifact_fields: "Sequence[str] | None" = None,
) -> "tuple[DoneWhen | None, list[str]]":
    """``(spec, errors)``。**永不抛。**

    ``raw`` 为空（None / 空串 / 空 dict）⇒ ``(None, [])`` —— 「没有 done_when」
    是合法的，而且就是今天所有调用方的样子。

    ``errors`` 非空时 ``spec`` 不可信，调用方**整体拒绝**：丢掉不合法的那几条
    会让目标被悄悄削弱（``all`` 少一个合取项 = 更早达成 = 错误抑制唤醒）。
    """
    if raw is None or raw == "" or raw == {} or raw == []:
        return None, []
    fields = (tuple(artifact_fields) if artifact_fields is not None
              else _default_artifact_fields())
    errors: list[str] = []
    spec = _norm(raw, "done_when", fields, errors)
    if errors:
        return None, errors
    return spec, []


def done_when_to_json(spec: "DoneWhen | None") -> Any:
    return None if spec is None else spec.as_dict()


def fingerprint(spec: "DoneWhen | None") -> str:
    """判据的稳定指纹 —— 事件去重键用。判据变了，指纹就得变。"""
    payload = json.dumps(done_when_to_json(spec), sort_keys=True,
                         ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def describe_done_when(spec: "DoneWhen | None") -> str:
    """一行人读的判据摘要（进 CampaignRef 的 bounded 字段、进提示词）。"""
    if spec is None:
        return ""
    if isinstance(spec, Predicate):
        return CATALOG[spec.kind].describe(spec)
    joiner = {"all": " 且 ", "any": " 或 "}.get(spec.op, " ")
    if spec.op == "not":
        return f"并非（{describe_done_when(spec.children[0])}）"
    return joiner.join(describe_done_when(c) for c in spec.children)


def iter_predicates(spec: "DoneWhen | None") -> "list[Predicate]":
    """按渲染顺序摊平所有叶子谓词。命名空间下标就是这个顺序。"""
    if spec is None:
        return []
    if isinstance(spec, Predicate):
        return [spec]
    out: list[Predicate] = []
    for c in spec.children:
        out.extend(iter_predicates(c))
    return out


def compile_rule(spec: "DoneWhen") -> Any:
    """``DoneWhen`` → ``RuleLeaf`` / ``RuleTree``，字段名带命名空间。

    命名空间不是装饰：两条 ``artifact_present`` 的证据字段名都是
    ``new_since_baseline``，不分开的话第二条会读到第一条的答案。下标即
    :func:`iter_predicates` 的顺序。
    """
    counter = {"i": 0}

    def _go(node):
        if isinstance(node, Predicate):
            idx = counter["i"]
            counter["i"] += 1
            leaf = CATALOG[node.kind].leaf(node)
            return RuleLeaf(field=f"p{idx}.{leaf.field}", op=leaf.op,
                            value=leaf.value)
        return RuleTree(op=node.op, children=tuple(_go(c) for c in node.children))

    return _go(spec)
