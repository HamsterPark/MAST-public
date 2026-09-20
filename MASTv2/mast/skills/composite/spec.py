"""Declarative composite-skill spec (IR) + a safe expression evaluator.

The historical composites express conditionals/loops as opaque Python in
``plan_dynamic`` — that cannot be serialised, versioned, templated or drawn.
This module introduces a *declarative* intermediate representation so a
composite's control flow becomes **data**:

  - it can be rendered as a flow graph (see ``graph_render.py``),
  - version-controlled (see ``version_store.py``),
  - cloned from a template (``CompositeSpec.clone``),
  - and executed by a single generic interpreter (``interpreter.py``),

without writing or importing any new Python per composite.

A spec is a tree of nodes:

    step  — run a sub-skill with params (params may be expressions)
    if    — branch on a boolean expression (then / else node lists)
    loop  — repeat-N / for-each / while a body of nodes
    set   — bind a variable to an expression's value

Expressions (conditions, loop counts/iterables, ``set`` values, and any param
written as ``{"$expr": "..."}``) are evaluated by :func:`safe_eval`, an
**allowlist AST evaluator**. This is safety-critical: a composite drives a real
STM, so the evaluator forbids attribute access, arbitrary calls, comprehensions,
imports, and dunders — only arithmetic/comparison/boolean ops, indexing,
literals, context variable lookups, and a tiny whitelist of pure builtins are
permitted. There is no ``eval``/``exec`` anywhere on this path.
"""

from __future__ import annotations

import ast
import copy
import operator
import re
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────
# Safe expression evaluator
# ─────────────────────────────────────────────────────────────────────────

class ExprError(ValueError):
    """Raised when an expression is unsafe or fails to evaluate."""


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_,
}
_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}

# Pure, side-effect-free builtins. range is capped at eval time. NO open/eval/
# getattr/__import__ etc.
_SAFE_FUNCS: dict[str, Any] = {
    "len": len, "abs": abs, "min": min, "max": max, "round": round,
    "int": int, "float": float, "str": str, "bool": bool, "sum": sum,
    "sorted": sorted, "any": any, "all": all, "range": range,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
}

_MAX_POW_EXP = 1000        # guard against 2**huge
_MAX_RANGE = 1_000_000     # guard against range(huge)
_MAX_SEQ_MUL = 1_000_000   # guard against sequence-repetition bombs ("a"*1e9)


def safe_eval(expr: str, context: dict[str, Any]) -> Any:
    """Evaluate *expr* against *context* with an allowlist AST walker.

    Only literals, context names, arithmetic/comparison/boolean ops, indexing,
    list/tuple/dict/set literals, ternary, and whitelisted pure-function calls
    are allowed. Anything else (attribute access, lambda, comprehension, call to
    a non-whitelisted name, etc.) raises :class:`ExprError`.
    """
    if not isinstance(expr, str):
        raise ExprError(f"expression must be a string, got {type(expr).__name__}")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"syntax error in {expr!r}: {exc}") from exc
    return _ev(tree.body, context)


def _ev(node: ast.AST, ctx: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ctx:
            return ctx[node.id]
        if node.id in ("True", "False", "None"):  # py constants, defensive
            return {"True": True, "False": False, "None": None}[node.id]
        raise ExprError(f"unknown variable {node.id!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ExprError(f"operator {type(node.op).__name__} not allowed")
        left, right = _ev(node.left, ctx), _ev(node.right, ctx)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and right > _MAX_POW_EXP:
            raise ExprError("exponent too large")
        if isinstance(node.op, ast.Mult):
            # sequence-repetition size guard: "a"*1e9 / [0]*1e9 would OOM.
            for seq, n in ((left, right), (right, left)):
                if isinstance(seq, (str, bytes, list, tuple)) and isinstance(n, int):
                    if n * max(1, len(seq)) > _MAX_SEQ_MUL:
                        raise ExprError("sequence repetition too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ExprError(f"unary {type(node.op).__name__} not allowed")
        return op(_ev(node.operand, ctx))
    if isinstance(node, ast.BoolOp):
        vals = node.values
        if isinstance(node.op, ast.And):
            result = True
            for v in vals:
                result = _ev(v, ctx)
                if not result:
                    return result
            return result
        else:  # Or
            result = False
            for v in vals:
                result = _ev(v, ctx)
                if result:
                    return result
            return result
    if isinstance(node, ast.Compare):
        left = _ev(node.left, ctx)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise ExprError(f"comparison {type(op_node).__name__} not allowed")
            right = _ev(comparator, ctx)
            if not op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _ev(node.body, ctx) if _ev(node.test, ctx) else _ev(node.orelse, ctx)
    if isinstance(node, ast.Subscript):
        value = _ev(node.value, ctx)
        key = _ev(node.slice, ctx)
        try:
            return value[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExprError(f"bad subscript: {exc}") from exc
    if isinstance(node, ast.List):
        return [_ev(e, ctx) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_ev(e, ctx) for e in node.elts)
    if isinstance(node, ast.Set):
        return {_ev(e, ctx) for e in node.elts}
    if isinstance(node, ast.Dict):
        return {_ev(k, ctx): _ev(v, ctx) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExprError("only direct calls to whitelisted functions allowed")
        fn = _SAFE_FUNCS.get(node.func.id)
        if fn is None:
            raise ExprError(f"call to {node.func.id!r} not allowed")
        if node.keywords:
            raise ExprError("keyword args not allowed in expressions")
        args = [_ev(a, ctx) for a in node.args]
        if fn is range:
            # cap range size to avoid building a huge list downstream
            rng = range(*args)
            if len(rng) > _MAX_RANGE:
                raise ExprError("range too large")
            return rng
        return fn(*args)
    raise ExprError(f"expression element {type(node).__name__} not allowed")


# ─────────────────────────────────────────────────────────────────────────
# Param resolution
# ─────────────────────────────────────────────────────────────────────────

def resolve_value(value: Any, context: dict[str, Any]) -> Any:
    """Resolve a spec value against *context*.

    Conventions:
      * ``{"$expr": "..."}``  → evaluate the expression.
      * a plain str / number / bool / None → returned literally.
      * a list / dict → resolved element-wise (so nested ``$expr`` works).
    """
    if isinstance(value, dict):
        if set(value.keys()) == {"$expr"}:
            return safe_eval(value["$expr"], context)
        return {k: resolve_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_value(v, context) for v in value]
    return value


def resolve_params(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Resolve every value in a step's params dict."""
    return {k: resolve_value(v, context) for k, v in (params or {}).items()}


# ─────────────────────────────────────────────────────────────────────────
# Spec model
# ─────────────────────────────────────────────────────────────────────────

NODE_TYPES = ("step", "if", "loop", "set", "llm", "human", "agent",
              "try", "break", "continue", "succeed", "fail")

#: 节点类型 → 装着子节点列表的键。子图可以藏在六种容器里,少数一个就意味着
#: 「遍历整棵树」的工具会漏掉那一支 —— 而漏掉的那支照样会在运行时打到硬件上。
_CHILD_LIST_KEYS: dict[str, tuple[str, ...]] = {
    "if": ("then", "else"),
    "loop": ("body",),
    "try": ("body", "finally"),
    "agent": ("on_error",),
    "llm": ("on_error",),          # data 模式;route 模式的分支在 routes 里
}
#: 节点类型 → 装着 {路由名: 子节点列表} 的键。
_CHILD_MAP_KEYS: dict[str, tuple[str, ...]] = {
    "human": ("routes",),
    "llm": ("routes",),
}


def walk_nodes(nodes: Any):
    """深度优先遍历一棵 spec 节点树,逐个 yield 节点(含全部嵌套分支)。

    覆盖 if/loop/try/human-routes/llm-routes/on_error 六类容器。之前各处自己写的
    遍历只认 if/loop 两种,于是藏在 try 体、llm 路由、human 路由里的 step 对
    「这棵树引用了哪些技能」这类问题是隐形的。
    """
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        yield node
        ntype = node.get("type")
        for key in _CHILD_LIST_KEYS.get(ntype, ()):
            yield from walk_nodes(node.get(key) or [])
        for key in _CHILD_MAP_KEYS.get(ntype, ()):
            branches = node.get(key)
            if isinstance(branches, dict):
                for sub in branches.values():
                    yield from walk_nodes(sub or [])


def collect_step_skills(nodes: Any) -> set[str]:
    """一棵 spec 树里 ``step`` 节点引用的全部技能名(含所有嵌套分支)。"""
    out: set[str] = set()
    for node in walk_nodes(nodes):
        if node.get("type") == "step":
            skill = node.get("skill")
            if isinstance(skill, str) and skill:
                out.add(skill)
    return out
LOOP_MODES = ("repeat", "foreach", "while")
LLM_MODES = ("route", "data")
LLM_OUTPUT_TYPES = ("str", "float", "int", "bool")
# route 名进 prompt/枚举/step 前缀，必须是干净标识符；"uncertain" 是保留的
# 结构性弃权选项（P2-B，RFC §9）。
_ROUTE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
DEFAULT_MAX_ITER = 10_000


@dataclass
class ParamSpec:
    """One declared input parameter of a composite (for the UI form)."""
    name: str
    type: str = "number"          # number | int | string | bool
    default: Any = None
    description: str = ""
    required: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "default": self.default,
                "description": self.description, "required": self.required}

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSpec":
        return cls(name=str(d.get("name", "")), type=str(d.get("type", "number")),
                   default=d.get("default"), description=str(d.get("description", "")),
                   required=bool(d.get("required", False)))


@dataclass
class CompositeSpec:
    """A declarative composite skill.

    Attributes:
      name:         skill name (registry key), e.g. "AutoConditionAndScan".
      description:  human description (shows in metadata / UI).
      version:      monotonically-increasing integer (the version store bumps it).
      safety_level: "auto" | "confirm" | "dangerous" (maps to SkillMetadata).
      params:       declared inputs (for the form + defaults context).
      nodes:        the control-flow tree (list of node dicts).
      estimated_duration_s / tags / author / notes: metadata.
    """
    name: str
    description: str = ""
    version: int = 1
    safety_level: str = "confirm"
    params: list[ParamSpec] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    # P4: 声明式输出签名 [{name, expr, description?}] —— 工作流结束时对最终
    # 变量上下文求值（safe_eval），进入 SkillResult.data 与技能卡。这是
    # "工作流即技能"拿到类型化对外接口的第一步。
    outputs: list[dict] = field(default_factory=list)
    # P5 控制流扩展（2026-06-26）：spec 级最终裁决。``success_when`` 若非空，
    # 在工作流结束时对最终变量上下文 safe_eval，决定 SkillResult.success；
    # ``fail_message`` 是失败原因表达式。二者皆空 ⇒ 沿用"所有步骤跑完即成功"。
    success_when: str = ""
    fail_message: str = ""
    estimated_duration_s: float = 0.0
    tags: list[str] = field(default_factory=list)
    author: str = ""
    notes: str = ""

    # -- (de)serialisation --
    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description,
            "version": self.version, "safety_level": self.safety_level,
            "params": [p.to_dict() for p in self.params],
            "nodes": copy.deepcopy(self.nodes),
            "outputs": copy.deepcopy(self.outputs),
            "success_when": self.success_when,
            "fail_message": self.fail_message,
            "estimated_duration_s": self.estimated_duration_s,
            "tags": list(self.tags), "author": self.author, "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompositeSpec":
        return cls(
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            version=int(d.get("version", 1)),
            safety_level=str(d.get("safety_level", "confirm")),
            params=[ParamSpec.from_dict(p) for p in (d.get("params") or [])],
            nodes=copy.deepcopy(d.get("nodes") or []),
            outputs=copy.deepcopy(d.get("outputs") or []),
            success_when=str(d.get("success_when", "") or ""),
            fail_message=str(d.get("fail_message", "") or ""),
            estimated_duration_s=float(d.get("estimated_duration_s", 0.0)),
            tags=list(d.get("tags") or []),
            author=str(d.get("author", "")),
            notes=str(d.get("notes", "")),
        )

    def default_context(self) -> dict[str, Any]:
        """The starting variable context: each param's default."""
        return {p.name: p.default for p in self.params}

    # -- template / clone --
    def clone(self, new_name: str, *, author: str = "") -> "CompositeSpec":
        """Return a deep copy under *new_name*, version reset to 1.

        This is the "use myself as a blueprint" primitive — the cloned spec is
        fully independent and starts its own version history."""
        c = CompositeSpec.from_dict(self.to_dict())
        c.name = new_name
        c.version = 1
        if author:
            c.author = author
        c.notes = (f"Cloned from {self.name} v{self.version}."
                   + (f" {c.notes}" if c.notes else ""))
        return c

    # -- validation --
    def validate(self) -> list[str]:
        """Return a list of human-readable problems ([] if valid).

        Checks node shape, expression parseability, loop bounds, and step
        id uniqueness — so the version store / UI can reject a bad spec
        before it ever drives hardware."""
        problems: list[str] = []
        if not self.name or not self.name.strip():
            problems.append("name is empty")
        if self.safety_level not in ("auto", "confirm", "dangerous"):
            problems.append(f"invalid safety_level {self.safety_level!r}")
        seen_ids: set[str] = set()
        self._validate_nodes(self.nodes, problems, seen_ids, path="root")
        # P4: outputs 签名 —— name 是干净标识符、expr 可解析。
        out_names: set[str] = set()
        for i, o in enumerate(self.outputs or []):
            here = f"outputs[{i}]"
            if not isinstance(o, dict):
                problems.append(f"{here}: must be an object")
                continue
            nm = o.get("name")
            if not isinstance(nm, str) or not _ROUTE_NAME.match(nm):
                problems.append(f"{here}: invalid output name {nm!r}")
            elif nm in out_names:
                problems.append(f"{here}: duplicate output name {nm!r}")
            else:
                out_names.add(nm)
            self._check_expr(o.get("expr"), problems, here, "expr")
        # P5: optional spec-level verdict expressions.
        if self.success_when and self.success_when.strip():
            self._check_expr(self.success_when, problems, "success_when", "success_when")
        if self.fail_message and self.fail_message.strip():
            self._check_expr(self.fail_message, problems, "fail_message", "fail_message")
        # P5: 'break'/'continue' are only valid inside a loop body.
        self._check_break_continue(self.nodes, False, problems, "root")
        return problems

    def _validate_nodes(self, nodes: Any, problems: list[str],
                        seen_ids: set[str], path: str) -> None:
        if not isinstance(nodes, list):
            problems.append(f"{path}: nodes must be a list")
            return
        for i, node in enumerate(nodes):
            here = f"{path}[{i}]"
            if not isinstance(node, dict):
                problems.append(f"{here}: node must be an object")
                continue
            ntype = node.get("type")
            if ntype not in NODE_TYPES:
                problems.append(f"{here}: invalid node type {ntype!r}")
                continue
            nid = node.get("id")
            if nid:
                if nid in seen_ids:
                    problems.append(f"{here}: duplicate node id {nid!r}")
                seen_ids.add(nid)
            if ntype == "step":
                if not node.get("skill"):
                    problems.append(f"{here}: step missing 'skill'")
                self._check_param_exprs(node.get("params"), problems, here)
            elif ntype == "if":
                self._check_expr(node.get("cond"), problems, here, "cond")
                self._validate_nodes(node.get("then") or [], problems, seen_ids, here + ".then")
                self._validate_nodes(node.get("else") or [], problems, seen_ids, here + ".else")
            elif ntype == "loop":
                mode = node.get("mode")
                if mode not in LOOP_MODES:
                    problems.append(f"{here}: invalid loop mode {mode!r}")
                if mode == "repeat":
                    self._check_expr(node.get("count"), problems, here, "count")
                elif mode == "foreach":
                    self._check_expr(node.get("iterable"), problems, here, "iterable")
                    if not node.get("var"):
                        problems.append(f"{here}: foreach missing 'var'")
                elif mode == "while":
                    self._check_expr(node.get("cond"), problems, here, "cond")
                self._validate_nodes(node.get("body") or [], problems, seen_ids, here + ".body")
            elif ntype == "set":
                if not node.get("var"):
                    problems.append(f"{here}: set missing 'var'")
                self._check_expr(node.get("value"), problems, here, "value")
            elif ntype == "llm":
                self._validate_llm(node, problems, seen_ids, here)
            elif ntype == "agent":
                from mast.skills.composite.agent_node import DELEGATABLE_AGENTS
                aid = node.get("agent")
                if aid == "instrument_control":
                    problems.append(
                        f"{here}: instrument_control 不可作为 agent 节点委托"
                        "——仪器动作必须以 step 节点逐技能出现在图上"
                        "（最小能动性）")
                elif aid not in DELEGATABLE_AGENTS:
                    problems.append(f"{here}: invalid agent {aid!r} "
                                    f"(可委托：{DELEGATABLE_AGENTS})")
                task = node.get("task")
                if not isinstance(task, str) or not task.strip():
                    problems.append(f"{here}: agent missing 'task'")
                self._check_param_exprs(node.get("inputs"), problems, here)
                self._validate_nodes(node.get("on_error") or [], problems,
                                     seen_ids, here + ".on_error")
            elif ntype == "human":
                msg = node.get("message")
                if not isinstance(msg, str) or not msg.strip():
                    problems.append(f"{here}: human missing 'message'")
                self._check_param_exprs(node.get("inputs"), problems, here)
                routes = node.get("routes")
                if routes is not None:
                    if not isinstance(routes, dict) or not routes:
                        problems.append(f"{here}: human 'routes' must be a "
                                        "non-empty object when present")
                    else:
                        for rname, rlist in routes.items():
                            if (not isinstance(rname, str)
                                    or not _ROUTE_NAME.match(rname)):
                                problems.append(f"{here}: invalid human route "
                                                f"name {rname!r}")
                            self._validate_nodes(rlist or [], problems,
                                                 seen_ids,
                                                 f"{here}.routes.{rname}")
            elif ntype == "try":
                # P5: try/finally — body runs non-aborting (interpreter forces
                # body steps optional), finally always runs. Both are node lists.
                self._validate_nodes(node.get("body") or [], problems,
                                     seen_ids, here + ".body")
                self._validate_nodes(node.get("finally") or [], problems,
                                     seen_ids, here + ".finally")
            elif ntype in ("succeed", "fail"):
                # P5: early verdict; optional 'reason' expression.
                rsn = node.get("reason")
                if rsn is not None:
                    self._check_expr(rsn, problems, here, "reason")
            # 'break' / 'continue' carry no fields; loop-placement is checked
            # separately by _check_break_continue.

    def _validate_llm(self, node: dict, problems: list[str],
                      seen_ids: set[str], here: str) -> None:
        """`llm` 节点（P2-B，RFC §9）：受限 LLM 决策。

        route 模式：闭集 routes（命名子图槽位）+ escape 必指向其中一个 route
        （结构性保证「不确定 → 安全出口」，不靠模型自觉）；data 模式：
        类型化 output_schema + 可选 on_error 槽。inputs 沿用 step params 的
        {"$expr": ...} 约定。"""
        resp = node.get("responsibility")
        if not isinstance(resp, str) or not resp.strip():
            problems.append(f"{here}: llm missing 'responsibility'")
        mode = node.get("mode", "route")
        if mode not in LLM_MODES:
            problems.append(f"{here}: invalid llm mode {mode!r}")
            return
        self._check_param_exprs(node.get("inputs"), problems, here)
        if mode == "route":
            routes = node.get("routes")
            if not isinstance(routes, dict) or not routes:
                problems.append(f"{here}: llm route mode needs non-empty 'routes'")
                return
            for rname in routes:
                if not isinstance(rname, str) or not _ROUTE_NAME.match(rname):
                    problems.append(
                        f"{here}: invalid route name {rname!r} "
                        "(identifier, ≤41 chars)")
                elif rname == "uncertain":
                    problems.append(f"{here}: route name 'uncertain' is "
                                    "reserved (structural abstention)")
            esc = node.get("escape")
            if esc not in routes:
                problems.append(f"{here}: llm 'escape' must name one of its "
                                f"routes (got {esc!r})")
            for rname, rlist in routes.items():
                self._validate_nodes(rlist or [], problems, seen_ids,
                                     f"{here}.routes.{rname}")
        else:  # data
            schema = node.get("output_schema")
            if not isinstance(schema, dict) or not schema:
                problems.append(f"{here}: llm data mode needs non-empty "
                                "'output_schema'")
            else:
                for k, v in schema.items():
                    if v not in LLM_OUTPUT_TYPES:
                        problems.append(
                            f"{here}: output_schema[{k!r}] type {v!r} not in "
                            f"{LLM_OUTPUT_TYPES}")
            self._validate_nodes(node.get("on_error") or [], problems,
                                 seen_ids, here + ".on_error")

    @staticmethod
    def _check_expr(expr: Any, problems: list[str], path: str, field_name: str) -> None:
        if not isinstance(expr, str) or not expr.strip():
            problems.append(f"{path}: {field_name} must be a non-empty expression string")
            return
        try:
            ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            problems.append(f"{path}: {field_name} syntax error: {exc}")

    @classmethod
    def _check_break_continue(cls, nodes: Any, in_loop: bool,
                              problems: list[str], path: str) -> None:
        """P5: walk the tree and flag any 'break'/'continue' not inside a loop."""
        if not isinstance(nodes, list):
            return
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            nt = node.get("type")
            here = f"{path}[{i}]"
            if nt in ("break", "continue") and not in_loop:
                problems.append(f"{here}: '{nt}' is only valid inside a loop body")
            elif nt == "loop":
                cls._check_break_continue(node.get("body") or [], True, problems, here + ".body")
            elif nt == "if":
                cls._check_break_continue(node.get("then") or [], in_loop, problems, here + ".then")
                cls._check_break_continue(node.get("else") or [], in_loop, problems, here + ".else")
            elif nt == "try":
                cls._check_break_continue(node.get("body") or [], in_loop, problems, here + ".body")
                cls._check_break_continue(node.get("finally") or [], in_loop, problems, here + ".finally")
            elif nt in ("llm", "human"):
                for rname, rlist in (node.get("routes") or {}).items():
                    cls._check_break_continue(rlist or [], in_loop, problems, f"{here}.routes.{rname}")
                cls._check_break_continue(node.get("on_error") or [], in_loop, problems, here + ".on_error")
            elif nt == "agent":
                cls._check_break_continue(node.get("on_error") or [], in_loop, problems, here + ".on_error")

    @classmethod
    def _check_param_exprs(cls, params: Any, problems: list[str], path: str) -> None:
        if params is None:
            return
        if not isinstance(params, dict):
            problems.append(f"{path}: params must be an object")
            return
        for k, v in params.items():
            if isinstance(v, dict) and set(v.keys()) == {"$expr"}:
                cls._check_expr(v["$expr"], problems, path, f"params.{k}")


__all__ = [
    "safe_eval", "ExprError", "resolve_value", "resolve_params",
    "ParamSpec", "CompositeSpec",
    "NODE_TYPES", "LOOP_MODES", "DEFAULT_MAX_ITER",
]
