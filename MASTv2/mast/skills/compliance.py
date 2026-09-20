"""技能合规判据 —— 单个 .py 技能 / 单份组合 spec / 单个 contrib 投稿目录。

三个入口，一套判据：

* :func:`check_python_source` —— 一段技能源码（``config/custom_skills/<名>.py`` 的形状）；
* :func:`check_spec`          —— 一份 CompositeSpec（JSON 数据，``POST /api/composites`` 的形状）；
* :func:`check_contrib_dir`   —— ``contrib/skills/<名>/``：``manifest.json`` + ``skill.py`` 或 ``spec.json``。

调用方：``scripts/skill_check.py``（投稿者本地、公开仓 CI）、
``tests/v2/unit/skills/test_contrib_compliance.py``、外部 agent 网关的提议 / 草稿端点。

## 判据不另写第二份

每一条都 import 调用仓里已有的那一份实现：技能形状
``skill_forge_tools._baseskill_shape_problems``、spec 设计期全量校验
``skill_forge_tools._validate``（内含 ``validate_spec_payload`` 与 ``agent_track_lint``）、
撞名 ``_name_collision_problems``、拒绝名单 ``SkillAuthor._ast_safety_check``、名字
``_validate_skill_name``、全局包络 ``core.safety._GLOBAL_CHECKS``、前置条件词表
``core.preconditions.precondition_recognized``、读 / 写动词 ``core.execution_context._is_read``、
强制前缀 ``core.si_quantity.needs_strict_prefix``。测试树里已有几条全仓扫描的结构判据
（``safe_call`` 动词是字面量、命令名在 ``nanonis_spm`` 里存在、``SkillResult`` 关键字合法、
不交裸信封、strict 参数描述不教指数写法）——这里是它们的**单文件版本**：同一个判据，
作用域从「整棵树」换成「这一个投稿」。

## 静态优先

默认不执行被检代码（``run_smoke=False``）。冒烟执行（X01）只在拒绝名单（S03）通过之后
才跑；而拒绝名单是防呆不是沙箱 —— 执行等于以本进程的全部权限运行那段代码，所以只在
调用方明确要求时做（投稿目录检查默认做，网关的提议端点不做）。

## 足迹（V03）

``pure-analysis`` / ``hardware-read-only`` / ``hardware-write``，由源码静态推出：
字面量 ``safe_call`` 动词按 ``_is_read`` 分读写，``context.run("X")`` 的子技能递归取足迹，
读 ``context.state`` 算只读。执行上下文被交给看不透的函数、被存起来、或动词 / 子技能名是
变量时，结论是 ``unknown`` —— 硬件只经执行上下文可达，上下文不外泄，足迹就是看得见的那些。
已注册的官方技能静态看不透时，按其 ``category`` 保守推断（:attr:`SkillFootprint.effective`）。

每条 :class:`Finding` 的级别：FAIL 拦、WARN 提醒、INFO 说明。``report.ok`` ⇔ 没有 FAIL。
代号表见 :data:`CODES`。
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import logging
import operator
import re
import sys
import threading
import time
import traceback
import types
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "CODES",
    "FOOTPRINTS",
    "UNKNOWN",
    "Finding",
    "ComplianceReport",
    "SkillFootprint",
    "check_python_source",
    "check_spec",
    "check_contrib_dir",
    "skill_footprint",
    "default_registry",
    "MANIFEST_KEYS",
]

FAIL, WARN, INFO = "FAIL", "WARN", "INFO"
_LEVEL_ORDER = {FAIL: 0, WARN: 1, INFO: 2}

#: 三类足迹。``unknown`` 不在其中：它表示「静态分析看不透」，不是第四类。
FOOTPRINTS: tuple[str, ...] = ("pure-analysis", "hardware-read-only", "hardware-write")
UNKNOWN = "unknown"
_FP_RANK = {"pure-analysis": 0, "hardware-read-only": 1, "hardware-write": 2}

#: 判据代号表。代号是对外契约（CLI 输出、CI 日志、网关回执都带它），只增不改。
CODES: dict[str, str] = {
    "S01": "safety_level 显式声明",
    "S02": "恰好一个 BaseSkill 子类，有 metadata() 与 execute()，metadata() 直接构造 SkillMetadata",
    "S03": "AST 拒绝名单（与自定义技能加载器同一个检查器）",
    "S04": "名字合法，且 安装文件名 = 类名 = metadata.name",
    "S05": "不与注册表里的技能或内置模板种子撞名",
    "P01": "带量纲参数：unit + min/max，报命中的全局包络行",
    "P02": "强制 SI 前缀的参数，描述里不出现指数写法",
    "P03": "前置条件在词表里",
    "P04": "发脉冲 / 修针的技能声明了对应能力标签",
    "V01": "safe_call 的动词是字符串字面量",
    "V02": "动词在 nanonis_spm（或补丁）里存在",
    "V03": "足迹可三分类，且与 category / manifest 一致",
    "V04": "写后回读（启发式，只提醒）",
    "R01": "SkillResult 的关键字都是真字段",
    "R02": "不把 Nanonis 的整个回包信封当数据交出去",
    "X01": "冒烟执行（拒绝名单通过之后才跑）",
    "C01": "spec 设计期全量校验，且子步只引用官方来源的技能",
    "M01": "manifest.json 结构",
    "M02": "manifest 与代码 / 目录一致",
    "M03": "政策声明：原创、无机器参数默认值、接受入站许可；不收论文移植",
    "E01": "判据的运行环境（注册表 / 依赖是否齐全）",
}

#: manifest.json 的键：{键: 是否必填}。
MANIFEST_KEYS: dict[str, bool] = {
    "schema": True,
    "name": True,
    "kind": True,
    "version": True,
    "summary": True,
    "summary_zh": False,
    "safety_level": True,
    "footprint": True,
    "authors": True,
    "license": True,
    "verification": True,
    "hardware_notes": False,
    "tests": True,
    "policy": True,
}
_MANIFEST_KINDS = ("python", "spec")
_MANIFEST_VERIFICATION = ("unit-tested", "contributor-hardware")
_MANIFEST_POLICY_KEYS = ("original_work", "no_machine_specific_defaults", "accepts_inbound_license")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_SAFETY_NAMES = ("auto", "confirm", "dangerous")

#: 执行上下文上**不通向硬件**、或本身就是被计入足迹的那几个属性。
_CTX_ATTRS_OK = frozenset({
    "safe_call", "urgent_call", "run", "check_abort", "abort_reason",
    "narrate", "emit_progress", "run_id", "state",
})
#: BaseSkill 自带、接收执行上下文但只轮询中止的辅助方法。
_SELF_HELPERS_OK = frozenset({"abortable_sleep"})
_VERB_CALLS = ("safe_call", "urgent_call")

#: 被检代码 import 了这些，就能绕开执行上下文碰到仪器（厂商库、MAST 的连接 / 运行时 / 执行层），
#: 或经网络、串口够到别的东西（包括 MAST 自己的 HTTP API）—— 足迹从此无法静态判断。
#: ``os`` / ``socket`` / ``subprocess`` 等已由拒绝名单拦下，这里不重复。
_BYPASS_IMPORTS: tuple[str, ...] = (
    "nanonis_spm", "serial", "pyvisa", "usb",
    "urllib", "http", "requests", "httpx", "aiohttp", "websocket", "websockets",
    "ftplib", "smtplib", "telnetlib", "xmlrpc",
    "mast.core.connection", "mast.core.runtime", "mast.core.execution_context", "mast.core.executor",
    "mast.core.instrument_lock", "mast.core.state", "mast.api", "mast.pipeline", "mast.agents",
    "mast.instruments", "mast.environment",
)

#: 与 tests/v2/unit/skills/test_strict_param_descriptions.py 同一个判据（指数写法）。
_EXPONENT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?[eE]-\d+(?![\w])")
#: 名字后缀像带量纲、却没写 unit 的浮点参数（只提醒）。
_UNIT_SUFFIX = re.compile(r"_(m|nm|v|mv|a|pa|na|s|ms|hz|k|deg)$")

#: 论文移植的形状（M03）。
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+|doi\.org/|\bdoi:\s*10\.", re.I)
_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/|\barXiv:\s*\d{4}\.\d{4,5}", re.I)
_SOFT_PORT = re.compile(r"\breproduce[sd]?\b|\bported from\b|\bet al\.", re.I)

#: 发脉冲 / 修针的 Nanonis 动词 → 必须声明的能力标签（与 core.safety 的两个常量同名）。
_CAPABILITY_VERBS: dict[str, str] = {
    "Bias_Pulse": "bias_pulse",
    "TipShaper_Start": "tip_shaping",
}

_SMOKE_TIMEOUT_S = 15.0


# ─────────────────────────────────────────────────────────────────────────
# 报告
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Finding:
    code: str
    level: str
    message: str
    line: int | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "level": self.level, "message": self.message, "line": self.line}


@dataclass
class ComplianceReport:
    target: str
    kind: str                       # "python" | "spec" | "contrib"
    skill_name: str = ""
    findings: list[Finding] = field(default_factory=list)
    footprint: str = UNKNOWN
    verbs: list[str] = field(default_factory=list)
    sub_skills: list[str] = field(default_factory=list)
    #: 没能执行的判据（缺库、没给注册表……）。**跳过不等于通过**，CLI 会单独大声列出。
    skipped: list[str] = field(default_factory=list)
    #: 实际评估过的代号。给「零 FAIL 是因为全都查过了，而不是什么都没查」留证据。
    checks_run: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(f.level == FAIL for f in self.findings)

    def add(self, code: str, level: str, message: str, line: int | None = None) -> None:
        self.findings.append(Finding(code, level, message, line))
        self.ran(code)

    def ran(self, *codes: str) -> None:
        for c in codes:
            if c not in self.checks_run:
                self.checks_run.append(c)

    def codes(self, level: str = FAIL) -> set[str]:
        return {f.code for f in self.findings if f.level == level}

    def merge(self, other: "ComplianceReport") -> None:
        self.findings.extend(other.findings)
        self.ran(*other.checks_run)
        for s in other.skipped:
            if s not in self.skipped:
                self.skipped.append(s)
        self.skill_name = self.skill_name or other.skill_name
        if other.footprint != UNKNOWN or self.footprint == UNKNOWN:
            self.footprint = other.footprint
        self.verbs = sorted(set(self.verbs) | set(other.verbs))
        self.sub_skills = sorted(set(self.sub_skills) | set(other.sub_skills))
        self.extra.update(other.extra)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "kind": self.kind,
            "skill_name": self.skill_name,
            "ok": self.ok,
            "footprint": self.footprint,
            "verbs": list(self.verbs),
            "sub_skills": list(self.sub_skills),
            "skipped": list(self.skipped),
            "checks_run": list(self.checks_run),
            "findings": [f.to_dict() for f in self.sorted_findings()],
            "extra": _jsonable(self.extra),
        }

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (_LEVEL_ORDER.get(f.level, 9), f.code, f.line or 0))

    def render(self, *, verbose: bool = False) -> str:
        head = "PASS" if self.ok else "FAIL"
        name = f"{self.skill_name} · " if self.skill_name else ""
        lines = [f"[{head}] {self.target}  ({name}{self.footprint})"]
        for f in self.sorted_findings():
            if f.level == INFO and not verbose:
                continue
            where = f"L{f.line}" if f.line else "   "
            lines.append(f"  {f.level:<4} {f.code} {where:>5}  {f.message}")
        for s in self.skipped:
            lines.append(f"  SKIPPED  {s}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SkillFootprint:
    """一个已注册技能的足迹。

    ``footprint`` 是纯静态结论（可能是 ``unknown``）；``declared`` 是按 ``category``
    保守推出的（ANALYSIS→pure-analysis、READ→hardware-read-only、WRITE/COMPOSITE→
    hardware-write）；:attr:`effective` 先取静态，看不透时取声明。
    """

    footprint: str
    verbs: tuple[str, ...] = ()
    verbs_known: bool = False
    sub_skills: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    declared: str = UNKNOWN

    @property
    def effective(self) -> str:
        return self.footprint if self.footprint != UNKNOWN else self.declared

    @property
    def basis(self) -> str:
        if self.footprint != UNKNOWN:
            return "static"
        return "declared" if self.declared != UNKNOWN else "none"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────
# 复用的判据（惰性 import：本模块被 CLI / 网关 import 时不该拖进整条依赖链）
# ─────────────────────────────────────────────────────────────────────────

def _forge():
    from mast.agents._shared import skill_forge_tools
    return skill_forge_tools


def _deny_list(code: str) -> list[str]:
    from mast.llm.skill_author import SkillAuthor
    return list(SkillAuthor._ast_safety_check(SkillAuthor.__new__(SkillAuthor), code))


def _name_error(name: str) -> str:
    try:
        from mast.llm.skill_author import _validate_skill_name
        _validate_skill_name(name)
    except ValueError as exc:
        return str(exc)
    return ""


def _is_read_verb(verb: str) -> bool:
    from mast.core.execution_context import _is_read
    return bool(_is_read(verb))


class _NoSkills:
    """空注册表：只让 ``_name_collision_problems`` 做「模板种子」那一半。"""

    def has(self, name: str) -> bool:
        return False

    def get(self, name: str, version: str | None = None):
        raise KeyError(name)

    def list_skills(self) -> list:
        return []


# ─────────────────────────────────────────────────────────────────────────
# 注册表
# ─────────────────────────────────────────────────────────────────────────

_REG_LOCK = threading.Lock()
_REG_CACHE: tuple[Any, tuple[Finding, ...]] | None = None


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # noqa: BLE001
            pass


def default_registry() -> tuple[Any, tuple[Finding, ...]]:
    """本进程一份、只建一次的注册表：``SkillRegistry().discover()``（内置 + 官方组合 + 论文包）。

    只含随代码发出的技能 —— 不加载本机的 spec 库、自定义技能与覆盖层，所以本机与 CI
    给出同一个答案。返回 ``(registry | None, findings)``；findings 里是 E01（环境）结论：
    某个技能包整包导入失败、单个模块导入失败、注册表为空，都会在这里被点名。
    """
    global _REG_CACHE
    with _REG_LOCK:
        if _REG_CACHE is not None:
            return _REG_CACHE
        findings: list[Finding] = []
        try:
            from mast.core.registry import _DEFAULT_DISCOVER_PACKAGES, SkillRegistry
        except Exception as exc:  # noqa: BLE001
            findings.append(Finding("E01", FAIL, f"注册表模块导入失败（{type(exc).__name__}: {exc}）"
                                                 "——撞名与官方子步两项无从判断"))
            _REG_CACHE = (None, tuple(findings))
            return _REG_CACHE
        for pkg in _DEFAULT_DISCOVER_PACKAGES:
            try:
                importlib.import_module(pkg)
            except ModuleNotFoundError as exc:
                if exc.name == pkg:
                    findings.append(Finding("E01", INFO, f"技能包 {pkg} 不在这份代码里（未随仓），跳过"))
                else:
                    findings.append(Finding(
                        "E01", FAIL,
                        f"技能包 {pkg} 整包导入失败（缺 {exc.name}）——这个包里的技能全都不在注册表里，"
                        "撞名与官方子步检查会漏掉它们。装齐依赖再跑。"))
            except Exception as exc:  # noqa: BLE001
                findings.append(Finding("E01", FAIL, f"技能包 {pkg} 整包导入失败（{type(exc).__name__}: {exc}）"))
        cap = _Capture()
        reg_logger = logging.getLogger("mast.core.registry")
        reg_logger.addHandler(cap)
        try:
            reg = SkillRegistry()
            n = reg.discover()
        except Exception as exc:  # noqa: BLE001
            findings.append(Finding("E01", FAIL, f"注册表发现过程抛异常（{type(exc).__name__}: {exc}）"))
            _REG_CACHE = (None, tuple(findings))
            return _REG_CACHE
        finally:
            reg_logger.removeHandler(cap)
        bad = [m for m in cap.messages if m.startswith(("Failed to import", "Failed to register"))]
        if bad:
            findings.append(Finding(
                "E01", WARN,
                f"{len(bad)} 个技能模块没能加载，它们的名字不在撞名检查的视野里："
                + "；".join(b[:120] for b in bad[:6]) + ("…" if len(bad) > 6 else "")))
        if n == 0:
            findings.append(Finding("E01", FAIL, "注册表是空的（一个技能都没发现）——检查依赖是否装齐"))
        else:
            findings.append(Finding("E01", INFO, f"注册表：{n} 个随代码发出的技能（fresh discover）"))
        _REG_CACHE = (reg, tuple(findings))
        return _REG_CACHE


def _resolve_registry(registry, rep: ComplianceReport):
    rep.ran("E01")
    if registry is None:
        reg, env = default_registry()
        for f in env:
            rep.findings.append(f)
        return reg
    try:
        n = len(list(registry.list_skills()))
    except Exception as exc:  # noqa: BLE001
        rep.add("E01", FAIL, f"传入的注册表读不了（{type(exc).__name__}: {exc}）")
        return None
    if n == 0:
        rep.add("E01", FAIL, "传入的注册表是空的——撞名与官方子步两项无从判断")
        return None
    return registry


# ─────────────────────────────────────────────────────────────────────────
# 字面量求值（只认常量与常量的算术；其余一律「求不出」）
# ─────────────────────────────────────────────────────────────────────────

class _Unresolved:
    def __repr__(self) -> str:
        return "<无法静态求值>"


_UNRES = _Unresolved()


@dataclass(frozen=True)
class _EnumRef:
    owner: str
    member: str


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.FloorDiv: operator.floordiv,
}


def _ev(node: ast.AST | None, names: Mapping[str, Any]) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand, names)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
        return _UNRES
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _ev(node.left, names), _ev(node.right, names)
        if a is _UNRES or b is _UNRES:
            return _UNRES
        num = (int, float)
        if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
            try:
                return _BINOPS[type(node.op)](a, b)
            except Exception:  # noqa: BLE001
                return _UNRES
        if isinstance(a, str) and isinstance(b, str) and isinstance(node.op, ast.Add):
            return a + b
        return _UNRES
    if isinstance(node, ast.Name):
        return names.get(node.id, _UNRES)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        owner = node.value.id
        if owner in ("SafetyLevel", "SkillCategory"):
            return _EnumRef(owner, node.attr)
        return names.get(f"{owner}.{node.attr}", _UNRES)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        vals = [_ev(e, names) for e in node.elts]
        return _UNRES if any(v is _UNRES for v in vals) else vals
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            return _UNRES
        ks = [_ev(k, names) for k in node.keys]
        vs = [_ev(v, names) for v in node.values]
        if any(x is _UNRES for x in ks + vs):
            return _UNRES
        try:
            return dict(zip(ks, vs))
        except TypeError:
            return _UNRES
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("frozenset", "set", "list", "tuple") and not node.keywords):
        if not node.args:
            return []
        if len(node.args) == 1:
            v = _ev(node.args[0], names)
            return list(v) if isinstance(v, (list, tuple)) else _UNRES
    return _UNRES


def _callee_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _callee_repr(func: ast.AST) -> str:
    try:
        return ast.unparse(func)
    except Exception:  # noqa: BLE001
        return _callee_name(func) or "?"


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return "<缺>"
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "<表达式>"


def _field_names(cls_path: str) -> tuple[str, ...]:
    from mast.core import types as _t
    return tuple(f.name for f in dataclasses.fields(getattr(_t, cls_path)))


def _call_args(call: ast.Call, names_in_order: tuple[str, ...]) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for i, a in enumerate(call.args):
        if isinstance(a, ast.Starred):
            out["*"] = a
            continue
        if i < len(names_in_order):
            out[names_in_order[i]] = a
    for kw in call.keywords:
        out["**" if kw.arg is None else kw.arg] = kw.value
    return out


# ─────────────────────────────────────────────────────────────────────────
# 源码事实：动词、子技能、上下文外泄
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class _Facts:
    verbs: dict[str, list[int]] = field(default_factory=dict)
    dynamic_verbs: list[tuple[int, str]] = field(default_factory=list)
    runs: dict[str, list[int]] = field(default_factory=dict)
    dynamic_runs: list[tuple[int, str]] = field(default_factory=list)
    escapes: list[tuple[int, str]] = field(default_factory=list)
    state_lines: list[int] = field(default_factory=list)


def _params_of(fn: ast.AST) -> list[str]:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return []
    a = fn.args
    return [p.arg for p in list(a.posonlyargs) + list(a.args)]


def _context_names(units: list[ast.AST], seed: set[str], funcs: Mapping[str, ast.AST],
                   methods: Mapping[str, ast.AST], classes: Mapping[str, ast.ClassDef]) -> set[str]:
    """执行上下文在这些代码里叫什么名字：种子 + 别名赋值 + 经本地函数 / 方法传参后的形参名。"""
    names = set(seed)
    for _ in range(8):
        before = len(names)
        for unit in units:
            for node in ast.walk(unit):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in names:
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
                if not isinstance(node, ast.Call):
                    continue
                target, offset = None, 0
                if isinstance(node.func, ast.Name):
                    if node.func.id in funcs:
                        target = funcs[node.func.id]
                    elif node.func.id in classes:
                        init = next((b for b in classes[node.func.id].body
                                     if isinstance(b, ast.FunctionDef) and b.name == "__init__"), None)
                        target, offset = init, 1
                elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                      and node.func.value.id in ("self", "cls") and node.func.attr in methods):
                    target, offset = methods[node.func.attr], 1
                if target is None:
                    continue
                params = _params_of(target)
                for i, a in enumerate(node.args):
                    if isinstance(a, ast.Name) and a.id in names and i + offset < len(params):
                        names.add(params[i + offset])
                for kw in node.keywords:
                    if kw.arg and isinstance(kw.value, ast.Name) and kw.value.id in names:
                        names.add(kw.arg)
        if len(names) == before:
            break
    return names


def _collect_facts(units: list[ast.AST], ctx_names: set[str], local_callables: set[str],
                   local_methods: set[str]) -> _Facts:
    f = _Facts()
    seen_nodes: set[int] = set()
    for unit in units:
        for node in ast.walk(unit):
            if id(node) in seen_nodes:
                continue
            seen_nodes.add(id(node))
            if isinstance(node, ast.Call):
                name = _callee_name(node.func)
                first = node.args[0] if node.args else None
                if first is None:
                    first = next((k.value for k in node.keywords
                                  if k.arg in ("method_name", "skill_name")), None)
                if name in _VERB_CALLS:
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        f.verbs.setdefault(first.value, []).append(node.lineno)
                    else:
                        f.dynamic_verbs.append((node.lineno, f"{name}({_unparse(first)}, …)"))
                on_ctx = (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                          and node.func.value.id in ctx_names)
                if on_ctx and name == "run":
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        f.runs.setdefault(first.value, []).append(node.lineno)
                    else:
                        f.dynamic_runs.append((node.lineno, f"run({_unparse(first)}, …)"))
                is_local = ((isinstance(node.func, ast.Name) and node.func.id in local_callables)
                            or (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                                and node.func.value.id in ("self", "cls")
                                and node.func.attr in (local_methods | _SELF_HELPERS_OK)))
                if not (is_local or on_ctx):
                    for a in list(node.args) + [k.value for k in node.keywords]:
                        if isinstance(a, ast.Starred):
                            a = a.value
                        if isinstance(a, ast.Name) and a.id in ctx_names:
                            f.escapes.append((node.lineno, f"执行上下文被交给了 {_callee_repr(node.func)}(…)"))
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ctx_names:
                if node.attr == "state":
                    f.state_lines.append(node.lineno)
                elif node.attr not in _CTX_ATTRS_OK:
                    f.escapes.append((node.lineno, f"直接访问了执行上下文的 .{node.attr}"))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if isinstance(value, ast.Name) and value.id in ctx_names:
                    for t in targets:
                        if not isinstance(t, ast.Name):
                            f.escapes.append((node.lineno, f"执行上下文被存进了 {_unparse(t)}"))
            elif isinstance(node, (ast.Return, ast.Yield)) and isinstance(getattr(node, "value", None), ast.Name) \
                    and node.value.id in ctx_names:
                f.escapes.append((node.lineno, "执行上下文被返回 / 交出去了"))
    return f


def _own_footprint(verbs: Iterable[str], state_read: bool) -> str:
    vs = list(verbs)
    if any(not _is_read_verb(v) for v in vs):
        return "hardware-write"
    if vs or state_read:
        return "hardware-read-only"
    return "pure-analysis"


def _combine(fps: Iterable[str]) -> str:
    best = "pure-analysis"
    for fp in fps:
        if fp not in _FP_RANK:
            return UNKNOWN
        if _FP_RANK[fp] > _FP_RANK[best]:
            best = fp
    return best


def _declared_footprint(category: Any) -> str:
    v = str(getattr(category, "value", category) or "").lower()
    return {"analysis": "pure-analysis", "read": "hardware-read-only",
            "write": "hardware-write", "composite": "hardware-write"}.get(v, UNKNOWN)


# ─────────────────────────────────────────────────────────────────────────
# 已注册技能的足迹（spec 子步、Python 技能的 context.run 子技能、网关的技能卡）
# ─────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _parse_file(path: str, mtime_ns: int) -> ast.Module | None:
    try:
        return ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    except Exception:  # noqa: BLE001
        return None


def _module_tree(cls) -> ast.Module | None:
    """定义 ``cls`` 的模块的 AST。

    打包版（PyInstaller）里 ``inspect.getsourcefile`` 给的 ``.py`` 并不存在（模块在 PYZ 里），
    只靠它会让仪器上的每个技能都读成 ``unknown``；``mast.*`` 的类退到随包发出的源码副本
    （``mast.pyexec._srcfiles.resolve``，开发树里就是包目录本身）。
    """
    import inspect
    try:
        path = inspect.getsourcefile(cls)
    except (TypeError, OSError):
        path = None
    if not path or not Path(path).is_file():
        path = None
        module = str(getattr(cls, "__module__", "") or "")
        if module == "mast" or module.startswith("mast."):
            try:
                from mast.pyexec._srcfiles import resolve

                found = resolve(module)
                path = str(found) if found is not None else None
            except Exception:  # noqa: BLE001 — 源码副本不在：照旧读成 unknown
                path = None
    if not path:
        return None
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        return None
    return _parse_file(path, mtime)


def skill_footprint(registry, name: str, *, _seen: frozenset[str] = frozenset()) -> SkillFootprint:
    """已注册技能 ``name`` 的足迹。永不抛。

    声明式 spec 取各子步之并；Python 技能读其定义模块：类体 + 同模块的基类 + 从类体可达的
    模块级函数 / 类。``context.run("X")`` 递归；看不透的（上下文外泄、动词是变量、``execute``
    来自别的模块的框架类）给 ``unknown`` 并写明原因，``declared`` 另按 category 给出。
    """
    if name in _seen or len(_seen) > 8:
        return SkillFootprint(UNKNOWN, reasons=(f"{name}：子技能循环引用或嵌套太深",))
    seen = _seen | {name}
    try:
        cls = registry.get(name)
        meta = registry._get_metadata(cls)
    except Exception:  # noqa: BLE001
        return SkillFootprint(UNKNOWN, reasons=(f"注册表里没有 {name}",))
    declared = _declared_footprint(getattr(meta, "category", None))
    try:
        from mast.skills.composite.interpreter import SpecComposite
        is_spec = isinstance(cls, type) and issubclass(cls, SpecComposite)
    except Exception:  # noqa: BLE001
        is_spec = False
    if is_spec:
        steps = sorted(_forge()._spec_step_skills_of(cls))
        subs = [skill_footprint(registry, s, _seen=seen) for s in steps]
        static = _combine(s.footprint for s in subs) if subs else UNKNOWN
        eff = _combine(s.effective for s in subs) if subs else declared
        verbs = sorted({v for s in subs for v in s.verbs})
        reasons = tuple(r for s in subs for r in s.reasons)
        return SkillFootprint(static, tuple(verbs), all(s.verbs_known for s in subs), tuple(steps),
                              reasons, eff)
    tree = _module_tree(cls)
    if tree is None:
        return SkillFootprint(UNKNOWN, reasons=(f"{name}：源码读不到（动态生成的类？）",), declared=declared)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    node = classes.get(getattr(cls, "__name__", ""))
    if node is None:
        return SkillFootprint(UNKNOWN, reasons=(f"{name}：类不在模块顶层",), declared=declared)
    owner = next((k for k in getattr(cls, "__mro__", ()) if "execute" in vars(k)), None)
    if owner is None or getattr(owner, "__module__", "") != getattr(cls, "__module__", "") \
            or getattr(owner, "__name__", "") not in classes:
        where = f"{getattr(owner, '__module__', '?')}.{getattr(owner, '__name__', '?')}"
        return SkillFootprint(UNKNOWN, reasons=(f"{name}：execute 由 {where} 提供，静态看不透",),
                              declared=declared)
    # 同模块的基类（execute 可能在那里）
    units: list[ast.AST] = [node]
    stack = [node]
    while stack:
        cur = stack.pop()
        for b in cur.bases:
            bn = b.id if isinstance(b, ast.Name) else ""
            if bn in classes and classes[bn] not in units:
                units.append(classes[bn])
                stack.append(classes[bn])
    # 从类体可达的模块级函数 / 类
    frontier = list(units)
    while frontier:
        cur = frontier.pop()
        for n in ast.walk(cur):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                tgt = funcs.get(n.id) or classes.get(n.id)
                if tgt is not None and tgt not in units:
                    units.append(tgt)
                    frontier.append(tgt)
    methods: dict[str, ast.AST] = {}
    for u in units:
        if isinstance(u, ast.ClassDef):
            for b in u.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.setdefault(b.name, b)
    exec_def = methods.get("execute")
    seed = set(_params_of(exec_def)[1:2]) if exec_def is not None else set()
    ctx = _context_names(units, seed, funcs, methods, classes)
    facts = _collect_facts(units, ctx, set(funcs) | set(classes), set(methods))
    return _footprint_from_facts(registry, name, facts, declared, seen)


def _footprint_from_facts(registry, name: str, facts: _Facts, declared: str,
                          seen: frozenset[str]) -> SkillFootprint:
    reasons: list[str] = []
    reasons += [f"{name} L{ln}：动词是变量 {txt}" for ln, txt in facts.dynamic_verbs]
    reasons += [f"{name} L{ln}：子技能名是变量 {txt}" for ln, txt in facts.dynamic_runs]
    reasons += [f"{name} L{ln}：{txt}" for ln, txt in facts.escapes]
    subs = {s: skill_footprint(registry, s, _seen=seen) for s in sorted(facts.runs)}
    own = _own_footprint(facts.verbs, bool(facts.state_lines))
    verbs = set(facts.verbs)
    for sf in subs.values():
        verbs |= set(sf.verbs)
    if reasons:
        static = UNKNOWN
    else:
        static = _combine([own] + [sf.effective for sf in subs.values()])
    known = not facts.dynamic_verbs and not facts.escapes and not facts.dynamic_runs \
        and all(sf.verbs_known for sf in subs.values())
    return SkillFootprint(static, tuple(sorted(verbs)), known, tuple(sorted(subs)), tuple(reasons), declared)


# ─────────────────────────────────────────────────────────────────────────
# nanonis_spm 的方法名（从磁盘 AST 取，绕开测试替身）
# ─────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _nanonis_methods() -> frozenset[str] | None:
    """nanonis_spm 暴露的方法名 ∪ ``core/nanonis_patch.py`` 挂上去的名字。没装库时返回 None。

    **不 import**：测试与 CLI 在缺库时会往 ``sys.modules`` 塞一个 MagicMock，它的任何属性都
    存在，拿它判「这个方法在不在」恒真。
    """
    import sysconfig

    pkg: Path | None = None
    for key in ("purelib", "platlib"):
        try:
            cand = Path(sysconfig.get_paths()[key]) / "nanonis_spm"
        except Exception:  # noqa: BLE001
            continue
        if cand.is_dir():
            pkg = cand
            break
    if pkg is None:
        for entry in sys.path:
            cand = Path(entry or ".") / "nanonis_spm"
            if cand.is_dir():
                pkg = cand
                break
    if pkg is None:
        return None
    names: set[str] = set()
    for src in pkg.glob("*.py"):
        try:
            tree = ast.parse(src.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
    patch = Path(__file__).resolve().parents[1] / "core" / "nanonis_patch.py"
    try:
        names |= set(re.findall(r"^\s*Nanonis\.(\w+)\s*=", patch.read_text(encoding="utf-8"), re.M))
    except OSError:
        pass
    return frozenset(names) if names else None


# ─────────────────────────────────────────────────────────────────────────
# Python 源码的静态模型
# ─────────────────────────────────────────────────────────────────────────

class _PySource:
    def __init__(self, code: str) -> None:
        self.code = code
        self.tree = ast.parse(code)
        self.funcs = {n.name: n for n in self.tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.classes = {n.name: n for n in self.tree.body if isinstance(n, ast.ClassDef)}
        self.names: dict[str, Any] = {}
        for n in self.tree.body:
            self._bind(n, prefix=())
        self.skill_cls: ast.ClassDef | None = next(
            (n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef) and any(
                (isinstance(b, ast.Name) and b.id == "BaseSkill")
                or (isinstance(b, ast.Attribute) and b.attr == "BaseSkill") for b in n.bases)), None)
        if self.skill_cls is not None:
            for n in self.skill_cls.body:
                self._bind(n, prefix=("self", "cls", self.skill_cls.name))
        self.meta_call: ast.Call | None = self._find_meta_call()
        self.meta: dict[str, ast.AST] = (
            _call_args(self.meta_call, _field_names("SkillMetadata")) if self.meta_call is not None else {})
        self.methods: dict[str, ast.AST] = {}
        for c in self.classes.values():
            for b in c.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.methods.setdefault(b.name, b)

    def _bind(self, n: ast.AST, *, prefix: tuple[str, ...]) -> None:
        target, value = None, None
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            target, value = n.targets[0].id, n.value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            target, value = n.target.id, n.value
        if target is None:
            return
        v = _ev(value, self.names)
        if prefix:
            for p in prefix:
                self.names[f"{p}.{target}"] = v
        else:
            self.names[target] = v

    def _find_meta_call(self) -> ast.Call | None:
        if self.skill_cls is None:
            return None
        for item in self.skill_cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "metadata":
                calls = [n for n in ast.walk(item)
                         if isinstance(n, ast.Call) and _callee_name(n.func) == "SkillMetadata"]
                for n in ast.walk(item):
                    if isinstance(n, ast.Return) and isinstance(n.value, ast.Call) \
                            and _callee_name(n.value.func) == "SkillMetadata":
                        return n.value
                return calls[0] if calls else None
        return None

    def value(self, key: str) -> Any:
        node = self.meta.get(key)
        return None if node is None else _ev(node, self.names)

    def line_of(self, key: str) -> int | None:
        node = self.meta.get(key)
        return getattr(node, "lineno", None) if node is not None else getattr(self.meta_call, "lineno", None)

    def params(self) -> tuple[list[dict[str, Any]], str]:
        """[{字段: 值, "_line": 行号}]，以及一句「读不出来」的原因（空 = 读出来了）。"""
        node = self.meta.get("parameters")
        if node is None:
            return [], ""
        if not isinstance(node, (ast.List, ast.Tuple)):
            return [], "parameters 不是字面量列表（写成 [ParameterSpec(...), ...]，好让审阅与判据都读得到）"
        names = _field_names("ParameterSpec")
        out: list[dict[str, Any]] = []
        for el in node.elts:
            if not (isinstance(el, ast.Call) and _callee_name(el.func) == "ParameterSpec"):
                return [], f"parameters 里有一项不是 ParameterSpec(...)：{_unparse(el)[:60]}"
            args = _call_args(el, names)
            row: dict[str, Any] = {k: _ev(v, self.names) for k, v in args.items() if k in names}
            for k, spec_default in (("description", ""), ("unit", ""), ("required", True), ("default", None),
                                    ("min_value", None), ("max_value", None), ("allowed_values", None)):
                row.setdefault(k, spec_default)
            row["_line"] = el.lineno
            out.append(row)
        return out, ""


# ─────────────────────────────────────────────────────────────────────────
# check_python_source
# ─────────────────────────────────────────────────────────────────────────

def check_python_source(code: str, *, filename: str = "skill.py", registry=None,
                        run_smoke: bool = False, contribution: bool = False) -> ComplianceReport:
    """一段技能源码的合规报告。

    ``filename`` 是**安装名**（``<技能名>.py``，即 ``config/custom_skills/`` 里的文件名、
    ``enabled.json`` 里的启用键）。取缺省值 ``skill.py`` 时不做「文件名 = 类名」那一半。
    ``registry=None`` ⇒ 用 :func:`default_registry`。``run_smoke=True`` ⇒ 拒绝名单通过后
    执行一次（X01）。``contribution=True`` ⇒ 追加投稿政策的源码扫描（M03）。
    """
    rep = ComplianceReport(target=filename, kind="python")
    stem = Path(filename).stem

    # S03 —— 拒绝名单先于一切：它不通过，后面任何需要执行的判据都不跑。
    rep.ran("S03")
    try:
        violations = _deny_list(code)
    except Exception as exc:  # noqa: BLE001 — 检查器不可用时保守拒绝（与加载器同一策略）
        violations = [f"拒绝名单检查器不可用（{type(exc).__name__}: {exc}）"]
        rep.add("E01", FAIL, f"拒绝名单检查器不可用（{type(exc).__name__}: {exc}）")
    for v in violations:
        m = re.search(r"\(line (\d+)\)", v)
        rep.add("S03", FAIL, f"拒绝名单：{v}（加载器加载前会重跑同一个检查，不过就不加载，只留一行日志）",
                int(m.group(1)) if m else None)
    s03_ok = not violations

    # S02 —— 形状
    rep.ran("S02")
    try:
        shape = _forge()._baseskill_shape_problems(code)
    except Exception as exc:  # noqa: BLE001
        shape = [f"形状判据不可用（{type(exc).__name__}: {exc}）"]
    for p in shape:
        rep.add("S02", FAIL, p)
    try:
        src = _PySource(code)
    except SyntaxError as exc:
        if not shape:
            rep.add("S02", FAIL, f"代码语法错误：{exc}", exc.lineno)
        return rep
    if src.skill_cls is not None and src.meta_call is None:
        rep.add("S02", FAIL, "metadata() 里没有直接构造 SkillMetadata(...)——判据与审阅都要能静态读到它",
                src.skill_cls.lineno)
    if shape or src.skill_cls is None or src.meta_call is None:
        return rep

    reg = _resolve_registry(registry, rep)
    _check_identity(src, stem, reg, rep)
    params, why = src.params()
    _check_params(src, params, why, rep)
    facts = _file_facts(src)
    _check_verbs(src, facts, reg, rep)
    _check_results(src, rep)
    if contribution:
        _check_policy_source(code, src, rep)
    if run_smoke:
        if s03_ok:
            _smoke(code, filename, src, facts, rep)
        else:
            rep.skipped.append("X01：拒绝名单没过，不执行这段代码")
    else:
        # 只记一条说明，不算「跑过」：checks_run 里没有 X01
        rep.findings.append(Finding("X01", INFO, "未做冒烟执行（run_smoke=False）"))
    return rep


def _check_identity(src: _PySource, stem: str, reg, rep: ComplianceReport) -> None:
    cls = src.skill_cls
    assert cls is not None
    # S01
    rep.ran("S01")
    sl_node = src.meta.get("safety_level")
    if sl_node is None:
        rep.add("S01", FAIL, "SkillMetadata 没有显式写 safety_level —— 缺省值是 CONFIRM，漏写不报错，"
                             "只会悄悄落在确认档", src.meta_call.lineno if src.meta_call else None)
    else:
        v = _ev(sl_node, src.names)
        if isinstance(v, _EnumRef) and v.owner == "SafetyLevel" and v.member in ("AUTO", "CONFIRM", "DANGEROUS"):
            rep.extra["safety_level"] = v.member.lower()
        else:
            rep.add("S01", FAIL, f"safety_level 必须写成 SafetyLevel.AUTO / CONFIRM / DANGEROUS，"
                                 f"现在是 {_unparse(sl_node)}", sl_node.lineno)
    ver = src.value("version")
    if isinstance(ver, str):
        rep.extra["version"] = ver
    # S04
    rep.ran("S04")
    name = src.value("name")
    if not isinstance(name, str) or not name:
        rep.add("S04", FAIL, "metadata 的 name 必须是字符串字面量（或模块级字符串常量）", src.line_of("name"))
        return
    rep.skill_name = name
    err = _name_error(name)
    if err:
        rep.add("S04", FAIL, f"名字不合法：{err}", src.line_of("name"))
    if cls.name != name:
        rep.add("S04", FAIL, f"类名 {cls.name} 与 metadata.name {name} 不一致", cls.lineno)
    if stem == "skill":
        rep.add("S04", INFO, "没给安装名（filename 取缺省值）：安装时文件必须命名为 "
                             f"{name}.py —— enabled.json 按文件名启用，注册按 metadata.name")
    elif stem != name:
        rep.add("S04", FAIL, f"安装文件名 {stem}.py 与 metadata.name {name} 不一致 —— "
                             "enabled.json 按文件名启用、注册表按 metadata.name 注册，两者不同时"
                             "启用的是一个名字、出现的是另一个")
    # S05
    rep.ran("S05")
    if reg is None:
        rep.skipped.append("S05：没有可用的注册表，撞名未核对")
        return
    try:
        if reg.has(name):
            origin = _forge()._origin_of(reg, name)
            rep.add("S05", FAIL, f"名字 {name} 已被一个已注册技能占用（来源：{origin}）——自定义技能最后加载，"
                                 "同名同版本会顶掉原来那个，注册表只打一行警告")
    except Exception as exc:  # noqa: BLE001
        rep.add("S05", WARN, f"撞名检查读注册表失败（{exc}）")
    for p in _forge()._name_collision_problems(name, _NoSkills()):
        rep.add("S05", FAIL, p)


def _global_row(name: str, unit: str) -> tuple[str, str, str, str] | None:
    from mast.core.safety import _GLOBAL_CHECKS
    n, u = name.lower(), unit.lower()
    for row in _GLOBAL_CHECKS:
        if row[0] in n and row[1] in u:
            return row
    return None


def _check_params(src: _PySource, params: list[dict[str, Any]], why: str, rep: ComplianceReport) -> None:
    from mast.core.safety import _GLOBAL_CHECKS
    from mast.core.si_quantity import needs_strict_prefix

    rep.ran("P01", "P02", "P03", "P04")
    if why:
        rep.add("P01", FAIL, why, src.line_of("parameters"))
    for p in params:
        name, line = str(p.get("name") or "?"), p.get("_line")
        unit, typ = p.get("unit"), p.get("type")
        if unit is _UNRES or not isinstance(unit if unit is not None else "", str):
            rep.add("P01", FAIL, f"参数 {name}：unit 不能静态求值", line)
            continue
        unit = (unit or "").strip()
        lo, hi, default = p.get("min_value"), p.get("max_value"), p.get("default")
        numeric = typ in ("float", "int")
        if unit:
            if lo is None or hi is None:
                rep.add("P01", FAIL, f"带量纲参数 {name} [{unit}] 缺 min_value / max_value —— "
                                     "参数包络只对写了边界的参数生效", line)
                lo = hi = None
            elif lo is _UNRES or hi is _UNRES or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in (lo, hi)):
                rep.add("P01", FAIL, f"带量纲参数 {name} [{unit}] 的 min / max 不能静态求值"
                                     "（写成数字字面量或模块级数字常量）", line)
                lo = hi = None
            elif lo > hi:
                rep.add("P01", FAIL, f"参数 {name}：min_value {lo} > max_value {hi}", line)
            if lo is not None and hi is not None and isinstance(default, (int, float)) \
                    and not isinstance(default, bool) and not (lo <= default <= hi):
                rep.add("P01", FAIL, f"参数 {name}：默认值 {default} 不在 [{lo}, {hi}] 内", line)
            row = _global_row(name, unit)
            if row:
                rep.add("P01", INFO, f"参数 {name} [{unit}] 命中全局包络 _GLOBAL_CHECKS 行 {row!r}："
                                     f"运行时还要落在 SafetyLimits.{row[2]} … {row[3]} 之内", line)
            else:
                rep.add("P01", INFO, f"参数 {name} [{unit}] 不命中任何全局包络行，只受自身 [min, max] 约束", line)
            # P02
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and needs_strict_prefix(lo, hi):
                desc = p.get("description")
                text = desc if isinstance(desc, str) else ""
                hits = _EXPONENT.findall(text)
                if hits:
                    rep.add("P02", FAIL, f"参数 {name} 的量程整段远离 1，解析器强制要求 SI 前缀、会拒绝指数写法，"
                                         f"而描述里在教指数写法 {hits} —— 改成 '50n' 这类写法", line)
        elif numeric:
            pats = [r for r in _GLOBAL_CHECKS if r[0] in name.lower()]
            if pats:
                rep.add("P01", WARN, f"参数 {name} 的名字命中全局包络行 {pats[0]!r}，但没写 unit —— "
                                     "全局包络要「名字 + 单位」同时命中才生效，它现在绕过了这一层", line)
            elif typ == "float" and _UNIT_SUFFIX.search(name.lower()):
                rep.add("P01", WARN, f"参数 {name} 的名字像带量纲（单位后缀），却没写 unit", line)
    # P03
    pcs = src.value("preconditions")
    if pcs is _UNRES or (pcs is not None and not isinstance(pcs, list)):
        rep.add("P03", FAIL, "preconditions 要写成字符串字面量列表", src.line_of("preconditions"))
    elif pcs:
        from mast.core.preconditions import precondition_recognized
        for pc in pcs:
            if not isinstance(pc, str) or not precondition_recognized(pc):
                rep.add("P03", FAIL, f"前置条件 {pc!r} 不在词表里 —— BaseSkill.check_preconditions 会报"
                                     "「Cannot verify」，这个技能每次都会被拒", src.line_of("preconditions"))
    caps = src.value("capabilities")
    if caps is _UNRES:
        rep.add("P04", FAIL, "capabilities 不能静态求值（写成 frozenset({\"bias_pulse\"}) 这样的字面量）",
                src.line_of("capabilities"))
        rep.extra["capabilities"] = _UNRES
    else:
        rep.extra["capabilities"] = sorted(str(c) for c in (caps or []))


def _bypass_imports(tree: ast.AST) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for m in mods:
            if any(m == b or m.startswith(b + ".") for b in _BYPASS_IMPORTS):
                out.append((node.lineno, m))
                break
    return out


def _file_facts(src: _PySource) -> _Facts:
    exec_def = src.methods.get("execute")
    if src.skill_cls is not None:
        for b in src.skill_cls.body:
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b.name == "execute":
                exec_def = b
    seed = set(_params_of(exec_def)[1:2]) if exec_def is not None else set()
    ctx = _context_names([src.tree], seed, src.funcs, src.methods, src.classes)
    return _collect_facts([src.tree], ctx, set(src.funcs) | set(src.classes), set(src.methods))


def _check_verbs(src: _PySource, facts: _Facts, reg, rep: ComplianceReport) -> None:
    rep.ran("V01", "V02", "V03", "V04")
    for ln, txt in facts.dynamic_verbs:
        rep.add("V01", FAIL, f"动词藏在变量里：{txt} —— 本仓每一个安全工具（中止策略、安全审计、命令名"
                             "核对）都靠字面量找 Nanonis 调用。改成两个字面量分支，或持有字面量的 thunk", ln)
    # V02
    verbs = sorted(facts.verbs)
    rep.verbs = verbs
    if verbs:
        lib = _nanonis_methods()
        if lib is None:
            rep.add("V02", WARN, "nanonis_spm 没装：命令名存在性**没有核对**（这一项不算通过）")
            rep.skipped.append("V02：nanonis_spm 未安装，命令名未核对")
        else:
            for v in verbs:
                if v not in lib:
                    rep.add("V02", FAIL, f"Nanonis 命令 {v!r} 在 nanonis_spm 里不存在，也没被 nanonis_patch 补上 "
                                         "—— 真机上会以 Method not found 失败", facts.verbs[v][0])
    # 子技能
    subs: dict[str, SkillFootprint] = {}
    for s in sorted(facts.runs):
        if reg is None:
            rep.skipped.append(f"V03：没有注册表，子技能 {s} 的足迹未核对")
            continue
        if not reg.has(s):
            rep.add("V03", FAIL, f"context.run 调用的子技能 {s!r} 不在注册表里", facts.runs[s][0])
            continue
        sf = skill_footprint(reg, s)
        subs[s] = sf
        if sf.basis == "declared":
            rep.add("V03", INFO, f"子技能 {s} 的足迹按其 category 推断为 {sf.effective}（静态看不透：{sf.reasons[:1]}）")
    rep.sub_skills = sorted(facts.runs)
    # V03
    reasons = [f"L{ln} 动词是变量 {txt}" for ln, txt in facts.dynamic_verbs]
    reasons += [f"L{ln} 子技能名是变量 {txt}" for ln, txt in facts.dynamic_runs]
    reasons += [f"L{ln} {txt}" for ln, txt in facts.escapes]
    reasons += [f"L{ln} import 了 {mod} —— 它能绕开执行上下文碰到仪器或网络" for ln, mod in _bypass_imports(src.tree)]
    own = _own_footprint(facts.verbs, bool(facts.state_lines))
    fp = UNKNOWN if reasons else _combine([own] + [sf.effective for sf in subs.values()])
    rep.footprint = fp
    if fp == UNKNOWN:
        if not reasons:
            reasons = [f"子技能 {s} 的足迹无法判断" for s, sf in subs.items() if sf.effective == UNKNOWN] \
                or ["子技能的足迹无法判断"]
        rep.add("V03", FAIL, "足迹无法静态分类（硬件只经执行上下文可达，而这里看不透它去了哪）："
                             + "；".join(reasons[:4]))
    cat = src.value("category")
    cat_name = cat.member if isinstance(cat, _EnumRef) and cat.owner == "SkillCategory" else ""
    if not cat_name:
        if "category" not in src.meta:
            rep.add("V03", WARN, "没写 category —— 缺省是 READ", src.meta_call.lineno if src.meta_call else None)
            cat_name = "READ"
        else:
            rep.add("V03", FAIL, f"category 必须写成 SkillCategory.X 字面量，现在是 {_unparse(src.meta.get('category'))}",
                    src.line_of("category"))
    rep.extra["category"] = cat_name.lower()
    if fp in _FP_RANK and cat_name:
        if cat_name == "ANALYSIS" and fp != "pure-analysis":
            rep.add("V03", FAIL, f"category=ANALYSIS 的定义是「不碰硬件」，而这段代码的足迹是 {fp}"
                                 f"（动词 {verbs or '—'}，子技能 {sorted(subs) or '—'}）", src.line_of("category"))
        elif cat_name == "READ" and fp == "hardware-write":
            writes = [v for v in verbs if not _is_read_verb(v)]
            rep.add("V03", FAIL, f"category=READ，但这段代码会写硬件（{writes or sorted(subs)}）",
                    src.line_of("category"))
        elif cat_name == "READ" and fp == "pure-analysis":
            rep.add("V03", WARN, "category=READ，但没有任何硬件读取 —— 纯计算请用 ANALYSIS", src.line_of("category"))
        elif cat_name == "WRITE" and fp != "hardware-write":
            rep.add("V03", INFO, f"category=WRITE，而静态足迹是 {fp}（声明偏保守，不拦）", src.line_of("category"))
        elif cat_name == "COMPOSITE" and not facts.runs:
            rep.add("V03", INFO, "category=COMPOSITE，但没有 context.run 子技能调用", src.line_of("category"))
    # V04（启发式）
    reads = {v for v in verbs if _is_read_verb(v)}
    for w in (v for v in verbs if not _is_read_verb(v)):
        prefix = w.split("_", 1)[0] + "_"
        if not any(r.startswith(prefix) for r in reads):
            rep.add("V04", WARN, f"写了 {w} 却没有同模块的回读（{prefix}…Get）—— 写进去的值没被读回来确认",
                    facts.verbs[w][0])
    # P04
    need: dict[str, str] = {}
    for v, cap in _CAPABILITY_VERBS.items():
        if v in facts.verbs:
            need.setdefault(cap, f"Nanonis 命令 {v}")
    if reg is not None:
        for s in subs:
            try:
                caps = set(getattr(reg._get_metadata(reg.get(s)), "capabilities", None) or ())
            except Exception:  # noqa: BLE001
                caps = set()
            for cap in caps & set(_CAPABILITY_VERBS.values()):
                need.setdefault(cap, f"子技能 {s}")
    declared = rep.extra.get("capabilities")
    if need and declared is not _UNRES:
        missing = sorted(set(need) - set(declared or ()))
        for cap in missing:
            rep.add("P04", FAIL, f"这个技能会{'发脉冲' if cap == 'bias_pulse' else '修针'}（来自 {need[cap]}），"
                                 f"却没声明能力标签 {cap!r} —— SAFE / SEMI 模式闸只认能力标签，没有它，"
                                 "SAFE 模式拦不住这个技能", src.line_of("capabilities"))


def _check_results(src: _PySource, rep: ComplianceReport) -> None:
    rep.ran("R01", "R02")
    valid = set(_field_names("SkillResult"))
    for node in ast.walk(src.tree):
        if isinstance(node, ast.Call) and _callee_name(node.func) == "SkillResult":
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in valid:
                    rep.add("R01", FAIL, f"SkillResult 没有字段 {kw.arg!r}（合法字段：{sorted(valid)}）"
                                         "—— 调用那一刻抛 TypeError，而注册时发现不了", node.lineno)
            _raw_envelope_in(node, rep)
    # 形状一：一句话的 return getattr(x, "return_value", …)
    for node in ast.walk(src.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [n for n in node.body if not isinstance(n, ast.Expr)]
            if len(body) == 1 and isinstance(body[0], ast.Return):
                v = body[0].value
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "getattr" \
                        and len(v.args) >= 2 and isinstance(v.args[1], ast.Constant) \
                        and v.args[1].value == "return_value":
                    rep.add("R02", FAIL, f"{node.name}() 把整个回包信封当读数交出去了 —— 用 "
                                         "mast.io.nanonis_files.decode_reply 剥开", node.lineno)
        # 形状二："raw": str(…)
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "raw" and isinstance(v, ast.Call) \
                        and isinstance(v.func, ast.Name) and v.func.id == "str":
                    rep.add("R02", FAIL, "把整个回包字符串化塞进 data（\"raw\": str(…)）—— 要原始回包看 "
                                         "nanonis_calls", node.lineno)


def _raw_envelope_in(call: ast.Call, rep: ComplianceReport) -> None:
    """形状三：SkillResult(...) 的参数里原样放着 ``x.return_value``。

    从 ``.return_value`` 往上走到这个 ``SkillResult(...)``：途中经过任何函数调用
    （``decode_reply(rec.return_value)``）或下标（``rec.return_value[2][0]``）都算「剥过」，
    不报；一路只经过容器与关键字的，才是把整个三段信封当数据交出去。
    """
    parents: dict[int, ast.AST] = {}
    for p in ast.walk(call):
        for c in ast.iter_child_nodes(p):
            parents[id(c)] = p
    for n in ast.walk(call):
        if not (isinstance(n, ast.Attribute) and n.attr == "return_value"):
            continue
        cur, processed = parents.get(id(n)), False
        while cur is not None and cur is not call:
            if isinstance(cur, (ast.Call, ast.Subscript)):
                processed = True
                break
            cur = parents.get(id(cur))
        if not processed:
            rep.add("R02", FAIL, "SkillResult 里原样放了 .return_value（Nanonis 的三段信封）—— 用 "
                                 "mast.io.nanonis_files.decode_reply 剥开再交", n.lineno)


def _check_policy_source(code: str, src: _PySource | None, rep: ComplianceReport, *, label: str = "源码") -> None:
    rep.ran("M03")
    if src is not None:
        cit = src.meta.get("citations")
        if cit is not None:
            v = _ev(cit, src.names)
            if v is _UNRES or v:
                rep.add("M03", FAIL, "metadata 带 citations —— contrib 不收论文移植（方法来自已发表工作的技能）",
                        getattr(cit, "lineno", None))
    for i, line in enumerate(code.splitlines(), 1):
        m = _DOI.search(line) or _ARXIV.search(line)
        if m:
            rep.add("M03", FAIL, f"{label}里出现文献标识 {m.group()[:40]!r} —— contrib 不收论文移植", i)
            continue
        s = _SOFT_PORT.search(line)
        if s:
            rep.add("M03", WARN, f"{label}里出现 {s.group()!r} —— 若这是论文 / 他人代码的复现，contrib 不收", i)


# ─────────────────────────────────────────────────────────────────────────
# 冒烟执行
# ─────────────────────────────────────────────────────────────────────────

class _SmokeState:
    def snapshot(self):
        from mast.core.types import HardwareState
        return HardwareState()


class _SmokeCtx:
    """冒烟用执行上下文：每个 safe_call 回一个形状正确、数值为零的回包。

    故意不用 MagicMock：MagicMock 的 ``.error`` 是真值，所有技能都会走错误分支，
    主路径一次都不会被执行 —— 而主路径正是要冒烟的那部分。
    """

    run_id = "compliance-smoke"

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self.calls: list[str] = []
        self.runs: list[str] = []
        self.state = _SmokeState()

    def safe_call(self, method_name, *args, **kwargs):
        from mast.core.types import NanonisCallRecord
        self.calls.append(str(method_name))
        return NanonisCallRecord(method=str(method_name), args=args, return_value=("", b"", [0.0] * 8))

    urgent_call = safe_call

    def run(self, skill_name, params=None, **kwargs):
        from mast.core.types import SkillResult
        self.runs.append(str(skill_name))
        return SkillResult(skill_name=str(skill_name), success=True, data={})

    def check_abort(self) -> bool:
        # 跑得太久就叫停：问的是「调下去会不会炸」，不是「跑完要多久」
        return (time.monotonic() - self._t0) > 2.0

    def abort_reason(self) -> str:
        return ""

    def narrate(self, kind, **data) -> None:
        return None

    def emit_progress(self, *args, **kwargs) -> None:
        return None


def _smoke_params(meta) -> dict:
    params: dict = {}
    for p in getattr(meta, "parameters", None) or []:
        if not p.required:
            if p.default is not None:
                params[p.name] = p.default
            continue
        if p.allowed_values:
            params[p.name] = p.allowed_values[0]
        elif p.type == "int":
            params[p.name] = int(p.min_value if p.min_value is not None else 0)
        elif p.type == "float":
            lo = p.min_value if p.min_value is not None else 0.0
            hi = p.max_value if p.max_value is not None else 1.0
            params[p.name] = float(lo) if lo > 0 else float(min(1e-9, hi))
        elif p.type == "bool":
            params[p.name] = False
        else:
            params[p.name] = "0"
    return params


_SMOKE_SEQ = [0]


def _smoke(code: str, filename: str, src: _PySource, facts: _Facts, rep: ComplianceReport) -> None:
    rep.ran("X01")
    from mast.core.types import SkillResult

    _SMOKE_SEQ[0] += 1
    mod_name = f"_mast_compliance_smoke_{_SMOKE_SEQ[0]}"
    out: dict[str, Any] = {}

    def target() -> None:
        mod = types.ModuleType(mod_name)
        mod.__file__ = filename
        sys.modules[mod_name] = mod
        stage = "导入模块"
        try:
            exec(compile(code, filename, "exec"), mod.__dict__)  # noqa: S102 — 拒绝名单通过后、调用方明确要求时才到这里
            stage = "找技能类"
            cls = mod.__dict__[src.skill_cls.name]  # type: ignore[union-attr]
            stage = "实例化"
            inst = cls()
            stage = "metadata()"
            meta = inst.metadata()
            params = _smoke_params(meta)
            ctx = _SmokeCtx()
            stage = "execute()"
            res = inst.execute(ctx, params)
            out.update(ok=True, res=res, meta=meta, params=params, ctx=ctx)
        except BaseException as exc:  # noqa: BLE001 — 冒烟要看到的正是这个
            out.update(ok=False, stage=stage, exc=exc,
                       tb="".join(traceback.format_exception_only(type(exc), exc)).strip())
        finally:
            sys.modules.pop(mod_name, None)

    t = threading.Thread(target=target, name="skill-compliance-smoke", daemon=True)
    t.start()
    t.join(_SMOKE_TIMEOUT_S)
    if t.is_alive():
        rep.add("X01", FAIL, f"冒烟执行 {_SMOKE_TIMEOUT_S:.0f} s 没有返回 —— 会等待的技能必须响应 check_abort()")
        return
    if not out.get("ok"):
        rep.add("X01", FAIL, f"冒烟执行在「{out.get('stage')}」抛异常：{out.get('tb')}")
        return
    res, meta, ctx = out["res"], out["meta"], out["ctx"]
    if getattr(meta, "name", None) != rep.skill_name and rep.skill_name:
        rep.add("X01", FAIL, f"运行时 metadata().name = {getattr(meta, 'name', None)!r}，与静态读到的 "
                             f"{rep.skill_name!r} 不一致")
    if not isinstance(res, SkillResult):
        rep.add("X01", FAIL, f"execute() 返回了 {type(res).__name__}，不是 SkillResult")
        return
    if not res.skill_name:
        rep.add("X01", FAIL, "返回的 SkillResult 没有 skill_name")
    rep.add("X01", INFO, f"冒烟执行：success={res.success}，safe_call {len(ctx.calls)} 次，"
                         f"子技能 {len(ctx.runs)} 次（参数 {out['params']}）")
    extra_verbs = sorted(set(ctx.calls) - set(facts.verbs))
    extra_runs = sorted(set(ctx.runs) - set(facts.runs))
    if extra_verbs or extra_runs:
        rep.add("V03", FAIL, f"运行时出现了静态分析没看到的调用（动词 {extra_verbs or '—'}，子技能 "
                             f"{extra_runs or '—'}）—— 足迹结论不可信")


# ─────────────────────────────────────────────────────────────────────────
# check_spec
# ─────────────────────────────────────────────────────────────────────────

def check_spec(spec: dict, *, registry=None) -> ComplianceReport:
    """一份 CompositeSpec 的合规报告。子步必须全部是官方来源（内置 / 官方组合 / 论文包）。

    为什么「只能引用官方子步」是硬规则：启动时的加载顺序是 发现内置 → spec → agent 工具
    → 自定义技能 → 覆盖层，spec 注册那一刻，自定义技能与 agent 工具都还不在注册表里，
    引用它们的 spec 会被加载器以「引用了不存在的技能」拒掉 —— 每次启动都拒，只留一行日志。
    """
    name = str((spec or {}).get("name") or "") if isinstance(spec, dict) else ""
    rep = ComplianceReport(target=f"spec:{name or '?'}", kind="spec", skill_name=name)
    rep.ran("C01")
    if not isinstance(spec, dict):
        rep.add("C01", FAIL, "spec 要是一个 JSON 对象")
        return rep
    reg = _resolve_registry(registry, rep)
    # S01
    rep.ran("S01")
    sl = spec.get("safety_level")
    if sl is None:
        rep.add("S01", FAIL, "spec 没有显式写 safety_level —— 缺省会落成 confirm；要写出来，审阅才看得见")
    elif str(sl).lower() not in _SAFETY_NAMES:
        rep.add("S01", FAIL, f"safety_level 必须是 auto / confirm / dangerous，现在是 {sl!r}")
    # S04
    rep.ran("S04")
    if not name:
        rep.add("S04", FAIL, "spec 没有 name")
    else:
        err = _name_error(name)
        if err:
            rep.add("S04", FAIL, f"名字不合法：{err}")
    # S05
    rep.ran("S05")
    collision: set[str] = set()
    if name and reg is not None:
        try:
            if reg.has(name):
                origin = _forge()._origin_of(reg, name)
                rep.add("S05", FAIL, f"名字 {name} 已被一个已注册技能占用（来源：{origin}）")
            collision = set(_forge()._name_collision_problems(name, reg))
        except Exception as exc:  # noqa: BLE001
            rep.add("S05", WARN, f"撞名检查读注册表失败（{exc}）")
        for p in _forge()._name_collision_problems(name, _NoSkills()):
            rep.add("S05", FAIL, p)
    elif reg is None:
        rep.skipped.append("S05：没有可用的注册表，撞名未核对")
    if reg is None:
        rep.add("C01", FAIL, "没有可用的注册表 —— spec 的子步无从校验")
        return rep
    # C01 —— 设计期全量校验（与 agent 轨 draft_composite / save_composite 同一个函数）
    try:
        result = _forge()._validate(spec, reg)
    except Exception as exc:  # noqa: BLE001
        rep.add("C01", FAIL, f"spec 校验内核不可用（{type(exc).__name__}: {exc}）")
        return rep
    for p in result.get("problems") or []:
        p = str(p)
        if p in collision:
            continue                    # 撞名已在 S05 报过
        if p.startswith("警告："):
            rep.add("C01", WARN, p)
        else:
            rep.add("C01", FAIL, p)
    for st in result.get("steps") or []:
        for w in st.get("warnings") or []:
            rep.add("C01", WARN, f"步骤 {st.get('id')!r}：{w}")
    steps = sorted(_forge()._step_skills(spec.get("nodes")))
    rep.sub_skills = steps
    official = _forge()._OFFICIAL_ORIGINS
    for s in steps:
        if not reg.has(s):
            continue                    # 「不存在」已由校验内核报过
        origin = _forge()._origin_of(reg, s)
        if origin not in official:
            rep.add("C01", FAIL, f"子步 {s} 的来源是 {origin}，不是官方技能 —— 启动时 spec 先于它加载，"
                                 "这份 spec 会在每次启动时被拒绝注册")
    # V03
    rep.ran("V03")
    subs = {s: skill_footprint(reg, s) for s in steps if reg.has(s)}
    for s, sf in subs.items():
        if sf.basis == "declared":
            rep.add("V03", INFO, f"子步 {s} 的足迹按其 category 推断为 {sf.effective}")
        elif sf.basis == "none":
            rep.add("V03", FAIL, f"子步 {s} 的足迹无法判断：{'；'.join(sf.reasons[:2])}")
    fp = _combine(sf.effective for sf in subs.values()) if subs else UNKNOWN
    rep.footprint = fp
    rep.verbs = sorted({v for sf in subs.values() for v in sf.verbs})
    if fp == UNKNOWN:
        rep.add("V03", FAIL, "足迹无法分类（没有可判断的子步）")
    # 生效安全级（声明只能收紧，继承自子步最高级）
    try:
        from mast.skills.composite.interpreter import _inherited_safety_level
        from mast.skills.composite.spec import CompositeSpec
        eff = _inherited_safety_level(CompositeSpec.from_dict(spec), reg)
        rep.extra["safety_level"] = str(spec.get("safety_level") or "").lower()
        rep.extra["effective_safety_level"] = str(getattr(eff, "value", eff)).lower()
    except Exception as exc:  # noqa: BLE001
        rep.add("C01", WARN, f"生效安全级推不出来（{exc}）")
    return rep


# ─────────────────────────────────────────────────────────────────────────
# check_contrib_dir
# ─────────────────────────────────────────────────────────────────────────

_IGNORED_ENTRIES = {"__pycache__", ".pytest_cache", ".DS_Store"}


def check_contrib_dir(path, *, registry=None, run_smoke: bool = True) -> ComplianceReport:
    """``contrib/skills/<名>/`` 的合规报告：manifest（M01–M03）+ 代码或 spec 的全部判据。

    目录约定：``manifest.json``；``skill.py``（Python 技能）或 ``spec.json``（组合 spec），
    二选一；``test_*.py``（至少一个）；可选 ``README.md``。安装时只拷 ``skill.py`` 一个文件
    （改名为 ``<名>.py``），所以目录里不能有它要 import 的其它模块。
    """
    d = Path(path)
    rep = ComplianceReport(target=d.as_posix(), kind="contrib")
    rep.ran("M01", "M02", "M03")
    if not d.is_dir():
        rep.add("M01", FAIL, f"{d} 不是目录")
        return rep
    entries = sorted(p for p in d.iterdir() if p.name not in _IGNORED_ENTRIES)
    manifest = _check_manifest(d, rep)
    kind = manifest.get("kind") if isinstance(manifest.get("kind"), str) else ""
    py, js = d / "skill.py", d / "spec.json"
    if not kind:
        kind = "python" if py.is_file() else ("spec" if js.is_file() else "")
    # 目录内容
    if kind == "python" and not py.is_file():
        rep.add("M02", FAIL, "manifest 说 kind=python，但目录里没有 skill.py")
    if kind == "spec" and not js.is_file():
        rep.add("M02", FAIL, "manifest 说 kind=spec，但目录里没有 spec.json")
    if py.is_file() and js.is_file():
        rep.add("M02", FAIL, "skill.py 与 spec.json 只能二选一")
    tests = [p.name for p in entries if p.is_file() and p.name.startswith("test_") and p.suffix == ".py"]
    for p in entries:
        if p.is_dir():
            rep.add("M02", FAIL, f"目录里有子目录 {p.name}/ —— 安装只拷 skill.py / spec.json，子目录不会跟过去")
        elif p.suffix == ".py" and p.name not in ("skill.py", "conftest.py") and p.name not in tests:
            rep.add("M02", FAIL, f"多余的模块 {p.name} —— 安装时只拷 skill.py，它 import 不到这个文件")
        elif p.name == "conftest.py":
            rep.add("M02", FAIL, "技能目录里不放 conftest.py（contrib/conftest.py 已经管路径与 nanonis_spm 替身）")
    if not tests:
        rep.add("M02", FAIL, "没有测试文件（test_*.py）—— 最低验证级别是 unit-tested")
    listed = manifest.get("tests")
    if isinstance(listed, list):
        for t in listed:
            if not isinstance(t, str) or not (d / t).is_file():
                rep.add("M02", FAIL, f"manifest.tests 里的 {t!r} 在目录里不存在")
    m_name = manifest.get("name")
    if isinstance(m_name, str) and m_name != d.name:
        rep.add("M02", FAIL, f"manifest.name {m_name!r} 与目录名 {d.name!r} 不一致")
    # 代码 / spec 本体
    sub: ComplianceReport | None = None
    text = ""
    if kind == "python" and py.is_file():
        text = py.read_text(encoding="utf-8")
        sub = check_python_source(text, filename=f"{d.name}.py", registry=registry,
                                  run_smoke=run_smoke, contribution=True)
        if re.search(r"^\s*(?:from|import)\s+contrib\b", text, re.M):
            rep.add("M02", FAIL, "skill.py import 了 contrib 下的东西 —— 装到 config/custom_skills/ 之后找不到它")
    elif kind == "spec" and js.is_file():
        text = js.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            rep.add("C01", FAIL, f"spec.json 不是合法 JSON：{exc}", exc.lineno)
            data = None
        if data is not None:
            sub = check_spec(data, registry=registry)
            _check_policy_source(text, None, sub, label="spec.json ")
    if sub is not None:
        rep.merge(sub)
        rep.target = d.as_posix()
        rep.kind = "contrib"
    rep.skill_name = rep.skill_name or d.name
    readme = d / "README.md"
    if readme.is_file():
        _check_policy_source(readme.read_text(encoding="utf-8"), None, rep, label="README.md ")
    # manifest 与代码一致
    if sub is not None:
        real_name = sub.skill_name
        if isinstance(m_name, str) and real_name and m_name != real_name:
            rep.add("M02", FAIL, f"manifest.name {m_name!r} 与技能自己的名字 {real_name!r} 不一致")
        m_sl = manifest.get("safety_level")
        real_sl = sub.extra.get("effective_safety_level") or sub.extra.get("safety_level")
        if isinstance(m_sl, str) and real_sl and m_sl != real_sl:
            what = "生效安全级（继承自子步）" if kind == "spec" else "代码里的 safety_level"
            rep.add("M02", FAIL, f"manifest.safety_level={m_sl!r}，而{what}是 {real_sl!r}")
        m_ver = manifest.get("version")
        if kind == "python" and isinstance(m_ver, str) and sub.extra.get("version") \
                and m_ver != sub.extra["version"]:
            rep.add("M02", FAIL, f"manifest.version {m_ver!r} 与 metadata.version {sub.extra['version']!r} 不一致")
        m_fp = manifest.get("footprint")
        rep.ran("V03")
        if isinstance(m_fp, str) and m_fp in FOOTPRINTS:
            if sub.footprint == UNKNOWN:
                rep.add("V03", FAIL, f"manifest 声明足迹 {m_fp}，但代码的足迹无法静态核实")
            elif sub.footprint != m_fp:
                rep.add("V03", FAIL, f"manifest 声明足迹 {m_fp}，而代码的足迹是 {sub.footprint}")
    return rep


def _check_manifest(d: Path, rep: ComplianceReport) -> dict:
    p = d / "manifest.json"
    if not p.is_file():
        rep.add("M01", FAIL, "缺 manifest.json")
        return {}
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        rep.add("M01", FAIL, f"manifest.json 不是合法 JSON：{exc}", exc.lineno)
        return {}
    if not isinstance(m, dict):
        rep.add("M01", FAIL, "manifest.json 要是一个 JSON 对象")
        return {}
    for k, required in MANIFEST_KEYS.items():
        if required and k not in m:
            rep.add("M01", FAIL, f"manifest 缺必填键 {k!r}")
    for k in m:
        if k not in MANIFEST_KEYS:
            rep.add("M01", WARN, f"manifest 里有未知键 {k!r}（拼错了？）")

    def bad(key: str, why: str) -> None:
        rep.add("M01", FAIL, f"manifest.{key}：{why}")

    if "schema" in m and m["schema"] != 1:
        bad("schema", "只认 1")
    for k in ("name", "summary"):
        if k in m and (not isinstance(m[k], str) or not m[k].strip()):
            bad(k, "要是非空字符串")
    if "summary_zh" in m and not isinstance(m["summary_zh"], str):
        bad("summary_zh", "要是字符串")
    if "kind" in m and m["kind"] not in _MANIFEST_KINDS:
        bad("kind", f"只能是 {_MANIFEST_KINDS}")
    if "version" in m and not (isinstance(m["version"], str) and _SEMVER.match(m["version"])):
        bad("version", "要是 X.Y.Z 形式的字符串")
    if "safety_level" in m and m["safety_level"] not in _SAFETY_NAMES:
        bad("safety_level", f"只能是 {_SAFETY_NAMES}")
    if "footprint" in m and m["footprint"] not in FOOTPRINTS:
        bad("footprint", f"只能是 {FOOTPRINTS}")
    if "authors" in m:
        a = m["authors"]
        if not isinstance(a, list) or not a or not all(
                isinstance(x, dict) and isinstance(x.get("name"), str) and x["name"].strip() for x in a):
            bad("authors", "要是非空列表，每项是带非空 name 的对象")
    if "license" in m and m["license"] != "MIT":
        bad("license", "只收 MIT（入站许可 = 出站许可）")
    if "verification" in m and m["verification"] not in _MANIFEST_VERIFICATION:
        bad("verification", f"只能是 {_MANIFEST_VERIFICATION}（维护者上机验证 = 毕业，不在 contrib 里）")
    if m.get("verification") == "contributor-hardware":
        hn = m.get("hardware_notes")
        if not isinstance(hn, str) or not hn.strip():
            bad("hardware_notes", "verification=contributor-hardware 时必填：在什么仪器 / 控制器版本上、怎么验的")
    if "tests" in m and not (isinstance(m["tests"], list) and m["tests"]
                             and all(isinstance(t, str) for t in m["tests"])):
        bad("tests", "要是非空的文件名列表")
    if "policy" in m:
        pol = m["policy"]
        if not isinstance(pol, dict):
            bad("policy", "要是对象")
        else:
            for k in _MANIFEST_POLICY_KEYS:
                if k not in pol:
                    bad("policy", f"缺 {k!r}")
                elif pol[k] is not True:
                    rep.add("M03", FAIL, {
                        "original_work": "policy.original_work 不是 true —— contrib 不收论文移植或他人代码的移植",
                        "no_machine_specific_defaults": "policy.no_machine_specific_defaults 不是 true —— "
                                                        "默认值里不能有某台仪器的标定值 / 增益 / 量程",
                        "accepts_inbound_license": "policy.accepts_inbound_license 不是 true —— 投稿按 MIT 入站，"
                                                   "且可能被收进闭源 / 商业发行版",
                    }[k])
    return m
