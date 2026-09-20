"""Builder backend API (P1) — skills catalog + composite CRUD + favorites.

Pure functions + a route list builder, mirroring ``agents_api.py``:
``app.py`` inserts the routes AFTER ``launch()`` (Gradio rebuilds its FastAPI
app inside launch — routes inserted earlier are silently lost) and EVERY
route is wrapped with ``mast.webui.route_auth.authed_route`` (修复项 invariant:
hand-inserted routes bypass Gradio's own login_check).

Catalog design (RFC §4): one contract for today's ~252 skills and a future
library of thousands — a lightweight full index (client-side fuzzy search)
plus on-demand full cards; the query params (q/category/tag/source/safety/
level/page) are in the contract from day one so the backend can later swap
to SQLite FTS5 with no frontend change. P1 serves everything from an
in-memory cache built off the live registry (admin overrides applied).
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── wiring (set once by app.py at startup, same pattern as composite_panel) ──

_registry = None          # live SkillRegistry
_catalog_lock = threading.Lock()
_catalog_cache: dict | None = None   # {"index": [...], "cards": {name: card}}


def set_live_registry(registry) -> None:
    global _registry
    _registry = registry


def invalidate_catalog() -> None:
    """Drop the cache — next GET rebuilds (call after hot-(un)register)."""
    global _catalog_cache
    with _catalog_lock:
        _catalog_cache = None


# ── domain / source / zh-sidecar helpers ────────────────────────────────────

def _domain_by_skill() -> dict[str, str]:
    """Invert encyclopedia.DOMAINS (curated, Chinese names) once."""
    try:
        from mast.webui.encyclopedia import DOMAINS
        out: dict[str, str] = {}
        for d in DOMAINS:
            for s in d.get("skills", ()):
                out.setdefault(s, d.get("name", "其他"))
        return out
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("DOMAINS load failed: %s", exc)
        return {}


def _skill_source(cls) -> str:
    """builtin / composite / paper / custom / user_composite / agent_tool / overlay.

    **委托到唯一真源** ``skills.overlay.provenance.classify_origin``（2026-08-20）。
    这里原本有一份自己的判定；两处各写一遍的话，UI 上的来源徽章和 provenance 表
    迟早各说各话，而那种不一致没有任何测试在看。
    """
    from mast.skills.overlay.provenance import classify_origin

    return classify_origin(cls)

_SOURCE_ZH = {"builtin": "内置", "composite": "组合", "paper": "论文",
              "custom": "自定义", "user_composite": "用户组合",
              "agent_tool": "Agent 工具", "overlay": "覆盖", "other": "其他"}


def _zh_sidecar() -> dict[str, dict]:
    """Chinese names/descriptions {name: {zh, description_zh}}.

    The packaged default ships in ``mast/gui/skill_zh.json`` (bundled into the
    frozen build via the ``Tree(MASTv2/mast)`` datas in mast2.spec, so it must
    live INSIDE the mast package — not repo-root config/, which is not packaged).
    An optional ``config/skill_zh.json`` under the user/project root overrides or
    extends it, so operators can hand-tune names without editing the package."""
    from pathlib import Path
    out: dict[str, dict] = {}
    # 1) packaged default (ships with every install)
    try:
        p = Path(__file__).resolve().parent / "skill_zh.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                out.update(d)
    except Exception as exc:  # pragma: no cover
        logger.debug("packaged skill_zh load failed: %s", exc)
    # 2) optional user/project override
    try:
        from mast._runtime_paths import project_root
        p = project_root() / "config" / "skill_zh.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                out.update(d)
    except Exception as exc:  # pragma: no cover
        logger.debug("skill_zh override load failed: %s", exc)
    return out


def _skill_extra(name: str) -> dict:
    """SKILL_EXTRA + admin guidance override merge (same as skill_guidance)."""
    try:
        from mast.knowledge.skill_guidance import SKILL_EXTRA
        extra = dict(SKILL_EXTRA.get(name) or {})
    except Exception:  # pragma: no cover
        extra = {}
    try:
        from mast.admin.override_store import ConfigOverrideRegistry, deep_merge
        ovr = ConfigOverrideRegistry.get().get_guidance_override(name)
        if ovr:
            extra = deep_merge(extra, ovr)
    except Exception:
        pass
    return extra


# ── catalog build ────────────────────────────────────────────────────────────

def _param_dict(p) -> dict:
    return {
        "name": p.name, "type": p.type, "description": p.description or "",
        "unit": getattr(p, "unit", None), "required": bool(p.required),
        "default": p.default,
        "min": getattr(p, "min_value", None), "max": getattr(p, "max_value", None),
        "allowed_values": getattr(p, "allowed_values", None),
    }


def build_catalog() -> dict:
    """Compute {"index": [...], "cards": {name: card}} from the live registry.

    Reads via registry.list_skills() so admin overrides are applied. ~250
    entries → a few ms; rebuilt lazily after invalidate_catalog()."""
    reg = _registry
    if reg is None:
        return {"index": [], "cards": {}}
    domains = _domain_by_skill()
    zh = _zh_sidecar()
    index: list[dict] = []
    cards: dict[str, dict] = {}
    for m in sorted(reg.list_skills(), key=lambda s: s.name):
        try:
            cls = reg.get(m.name)
        except KeyError:  # pragma: no cover — racing unregister
            continue
        source = _skill_source(cls)
        sl = getattr(m.safety_level, "value", str(m.safety_level))
        zh_e = zh.get(m.name) or {}
        if source == "agent_tool":
            try:
                from mast.skills.composite.tool_skills import AGENT_TOOL_DOMAINS
                domain = AGENT_TOOL_DOMAINS.get(
                    getattr(cls, "_AGENT_TOOL_SOURCE", ""), "Agent 工具")
            except Exception:  # pragma: no cover
                domain = "Agent 工具"
        else:
            domain = domains.get(m.name, "其他")
        entry = {
            "name": m.name,
            "zh": zh_e.get("zh", ""),
            "category": getattr(m.category, "name", str(m.category)),
            "safety": str(sl).lower(),
            "level": int(getattr(m, "composition_level", 0) or 0),
            "source": source,
            "source_zh": _SOURCE_ZH.get(source, source),
            "tags": list(m.tags or []),
            "domain": domain,
        }
        index.append(entry)
        card = {
            **entry,
            "version": m.version,
            "description": m.description or "",
            "description_zh": zh_e.get("description_zh", ""),
            "parameters": [_param_dict(p) for p in (m.parameters or [])],
            "preconditions": list(m.preconditions or []),
            "postconditions": list(m.postconditions or []),
            "estimated_duration_s": m.estimated_duration_s,
            "rollback_skill": m.rollback_skill,
            "extra": _skill_extra(m.name),
        }
        # P4: 用户组合的声明式输出签名进卡片（"工作流即技能"的对外接口）。
        if source == "user_composite":
            try:
                card["outputs"] = list(cls()._spec.outputs or [])
            except Exception:  # pragma: no cover
                pass
        cards[m.name] = card
    return {"index": index, "cards": cards}


def get_catalog() -> dict:
    """Cached catalog (build on first use / after invalidation)."""
    global _catalog_cache
    with _catalog_lock:
        if _catalog_cache is None:
            _catalog_cache = build_catalog()
        return _catalog_cache


def warm_catalog() -> None:
    """Prebuild off the event loop (daemon thread at startup, never blocks)."""
    try:
        get_catalog()
    except Exception as exc:  # pragma: no cover
        logger.warning("warm_catalog failed: %s", exc)


def filter_index(index: list[dict], *, q: str = "", category: str = "",
                 tag: str = "", source: str = "", safety: str = "",
                 level: str = "", domain: str = "") -> list[dict]:
    """In-memory filtering — the same params later map onto SQLite FTS5."""
    ql = q.strip().lower()
    out = []
    for e in index:
        if category and e["category"] != category:
            continue
        if source and e["source"] != source:
            continue
        if safety and e["safety"] != safety:
            continue
        if domain and e["domain"] != domain:
            continue
        if tag and tag not in e["tags"]:
            continue
        if level not in ("", None) and str(e["level"]) != str(level):
            continue
        if ql:
            hay = " ".join([e["name"].lower(), e["zh"], e["domain"],
                            " ".join(e["tags"])]).lower()
            if ql not in hay:
                continue
        out.append(e)
    return out


# ── composite validate (per-step param lint, RFC §5) ────────────────────────

_MAX_TREE_DEPTH = 50   # 纵深防御：恶意深嵌套 spec 不许打穿递归（review F6）


# 子节点容器 —— **委托到 spec.py 的两张表**(_CHILD_LIST_KEYS/_CHILD_MAP_KEYS）。
#
# 2026-08-25:这三个遍历原来各自手写 if/loop/llm/human/agent 五种,**漏了 try**。
# 于是藏在 ``try`` 的 body / finally 里的 step 对整个设计期 lint 是隐形的:未知
# 技能、超包络的字面量、z-approach 硬错,一条都不会报——而运行期它照样打到硬件
# 上。``spec.walk_nodes`` 早就是「遍历一棵 spec 树」的唯一真源(loader 的缺失技能
# 检查已经用它),这里补上同一条。
def _child_branches(n: dict):
    """Yield every child node-list of *n* (six containers, try included)."""
    try:
        from mast.skills.composite.spec import _CHILD_LIST_KEYS, _CHILD_MAP_KEYS
    except Exception:  # pragma: no cover — defensive
        _CHILD_LIST_KEYS = {"if": ("then", "else"), "loop": ("body",),
                            "try": ("body", "finally"), "agent": ("on_error",),
                            "llm": ("on_error",)}
        _CHILD_MAP_KEYS = {"human": ("routes",), "llm": ("routes",)}
    t = n.get("type")
    for key in _CHILD_LIST_KEYS.get(t, ()):
        yield n.get(key) or []
    for key in _CHILD_MAP_KEYS.get(t, ()):
        branches = n.get(key)
        if isinstance(branches, dict):
            for sub in branches.values():
                yield sub or []


def _walk_steps(nodes, out: list[dict], depth: int = 0) -> bool:
    """Collect step nodes. Returns False when the tree exceeds the depth cap."""
    if depth > _MAX_TREE_DEPTH:
        return False
    ok = True
    for n in nodes or []:
        if not isinstance(n, dict):
            continue
        if n.get("type") == "step":
            out.append(n)
        for branch in _child_branches(n):
            ok = _walk_steps(branch, out, depth + 1) and ok
    return ok


def _collect_names(nodes, out: set, depth: int = 0) -> None:
    """node id ∪ loop var ∪ set var —— $expr 可见名字的图侧来源。"""
    if depth > _MAX_TREE_DEPTH:
        return
    for n in nodes or []:
        if not isinstance(n, dict):
            continue
        if n.get("id"):
            out.add(n["id"])
        t = n.get("type")
        if t == "loop" and n.get("var"):
            out.add(n["var"])
        elif t == "set" and n.get("var"):
            out.add(n["var"])
        for branch in _child_branches(n):
            _collect_names(branch, out, depth + 1)


_IDENT_RE = None


def _expr_idents(expr: str) -> set:
    """表达式中的裸标识符（不含带引号的字符串键、属性访问本就被禁）。"""
    global _IDENT_RE
    import re
    if _IDENT_RE is None:
        _IDENT_RE = re.compile(r"(?<!['\"\w])([A-Za-z_一-鿿]"
                               r"[A-Za-z0-9_一-鿿]*)")
    return set(_IDENT_RE.findall(str(expr)))


def _node_exprs(n: dict) -> list:
    """非 step 参数之外的表达式字段（与前端 exprStrings 对齐）。"""
    out = []
    t = n.get("type")
    if t == "if" and n.get("cond"):
        out.append(("cond", n["cond"]))
    elif t == "loop":
        for k in ("count", "cond", "iterable"):
            if n.get(k):
                out.append((k, str(n[k])))
    elif t == "set" and n.get("value") is not None:
        out.append(("value", str(n["value"])))
    elif t in ("llm", "human", "agent"):
        for k, v in (n.get("inputs") or {}).items():
            if isinstance(v, dict) and "$expr" in v:
                out.append((f"inputs.{k}", str(v["$expr"])))
    return out


_SAFETY_RANK = {"auto": 0, "confirm": 1, "dangerous": 2}


def validate_spec_payload(spec_dict: dict, *, registry=None) -> dict:
    """Full design-time validation report for the editor.

    Returns {"ok", "problems": [spec-level], "steps": [{id, skill, errors,
    warnings}]}. Per-step lint: unknown skill, unknown/missing params, bounds
    on literal numerics, coarse-Z-approach hard error, DANGEROUS warning,
    declared safety_level below leaf max (P4 preview, warning only).

    ``registry`` (2026-08-25): 显式传入要用的 SkillRegistry。``None`` 回落模块全局
    ``_registry``(GUI/API 的接线方式,行为不变)。agent 侧的技能工坊持有的是被
    注入的那个注册表 —— 校验和执行必须问同一个对象,否则会出现「校验说没有、执行
    时有」这种查起来最费劲的不一致。
    """
    from mast.skills.composite.spec import CompositeSpec

    try:
        spec = CompositeSpec.from_dict(spec_dict)
    except Exception as exc:
        return {"ok": False, "problems": [f"spec 解析失败：{exc}"], "steps": []}
    problems = spec.validate()

    reg = registry if registry is not None else _registry
    steps: list[dict] = []
    step_nodes: list[dict] = []
    if not _walk_steps(spec_dict.get("nodes"), step_nodes):
        return {"ok": False,
                "problems": [f"节点树嵌套超过 {_MAX_TREE_DEPTH} 层"], "steps": []}
    max_leaf_rank = 0

    # $expr 标识符解析检查（review F4：节点 id 重命名后悬空引用此前只在运行期
    # safe_eval 才炸）。已知名字 = 节点 id ∪ 工作流参数 ∪ 循环/set 变量 ∪
    # safe_eval 白名单函数 ∪ {last, True, False, None}。警告级（不拦保存）。
    known = set()
    _collect_names(spec_dict.get("nodes"), known)
    for p in spec_dict.get("params") or []:
        if isinstance(p, dict) and p.get("name"):
            known.add(p["name"])
    try:
        from mast.skills.composite.spec import _SAFE_FUNCS
        known |= set(_SAFE_FUNCS)
    except Exception:  # pragma: no cover
        known |= {"len", "abs", "min", "max", "round", "int", "float", "str",
                  "bool", "sum", "sorted", "any", "all", "range", "list",
                  "dict", "tuple", "set"}
    known |= {"last", "True", "False", "None", "and", "or", "not", "in", "if",
              "else", "for"}

    guard = None
    if reg is not None:
        try:
            from mast.config import SafetyLimits
            from mast.core.safety import SafetyGuard
            guard = SafetyGuard(SafetyLimits())
        except Exception as exc:  # pragma: no cover
            logger.warning("validate: SafetyGuard unavailable: %s", exc)

    for n in step_nodes:
        sid = n.get("id") or "?"
        skill = n.get("skill") or ""
        errors: list[str] = []
        warnings: list[str] = []
        params = n.get("params") or {}
        literal = {k: v for k, v in params.items()
                   if not (isinstance(v, dict) and "$expr" in v)}
        dynamic = set(params) - set(literal)

        if reg is None or not reg.has(skill):
            errors.append(f"技能 {skill!r} 不存在于注册表")
            steps.append({"id": sid, "skill": skill,
                          "errors": errors, "warnings": warnings})
            continue
        # P4: 钉住的版本必须真实存在（运行期 registry.get fail-closed，
        # 设计期先抓出来）。
        pinned = n.get("skill_version")
        if pinned:
            try:
                reg.get(skill, str(pinned))
            except KeyError:
                errors.append(f"钉住的版本 {skill}@{pinned} 不在注册表"
                              "（解除钉住或换可用版本）")
        try:
            meta = reg._get_metadata(reg.get(skill))
        except Exception as exc:
            errors.append(f"技能元数据不可读：{exc}")
            steps.append({"id": sid, "skill": skill,
                          "errors": errors, "warnings": warnings})
            continue

        sl = str(getattr(meta.safety_level, "value", meta.safety_level)).lower()
        max_leaf_rank = max(max_leaf_rank, _SAFETY_RANK.get(sl, 1))

        # Coarse Z approach: hard design-time error — the runtime gate
        # (ExecutionContext.run, 修复项) rejects it anyway; surface it here.
        from mast.core.safety import is_coarse_sample_approach
        if is_coarse_sample_approach(skill, literal):
            errors.append("开环 Z 粗逼近（z-approach）不可在工作流内执行——"
                          "必须由人工经审批流程执行")
        elif skill == "MotorMove" and "direction" in dynamic:
            errors.append("MotorMove 的 direction 不可动态提供（$expr）——"
                          "方向必须是设计期字面量")
        if sl == "dangerous":
            warnings.append("DANGEROUS 技能在工作流（agent 路径）中会被运行期"
                            "审批门拒绝，需人工执行")

        spec_params = {p.name: p for p in (meta.parameters or [])}
        for k in params:
            if k not in spec_params:
                errors.append(f"未知参数 {k!r}")
        for pname, p in spec_params.items():
            if p.required and pname not in params and p.default is None:
                errors.append(f"缺少必填参数 {pname!r}")
        if guard is not None and literal:
            try:
                violations = guard.check_parameter_bounds(meta, literal)
                # ``check_parameter_bounds`` 末尾还带了一份**必填参数在不在**的
                # 检查（safety.py «Required parameter … is missing»），而这里只
                # 能把**字面量**喂给它 —— 用 ``$expr`` 动态提供的必填参数在它眼里
                # 一律缺席。症状很刁钻：一个步骤只要**同时**有字面量和 $expr 参数
                # 就报「缺少必填参数」（全 $expr 时 ``if literal`` 短路，反而不报），
                # 而那是一条**硬错**，直接把保存拦下来。
                #
                # 必填与否这件事上面已经用**完整** params（字面量 ∪ 动态）判过了
                # （「缺少必填参数」那一条），所以这里要的只是**边界**。逐条算出它
                # 会为动态参数生成的那句话并剔除 —— 精确匹配而不是模糊过滤，措辞
                # 一旦漂移这个过滤就自然失效（回到今天的行为，不会更糟），而
                # ``test_expr_supplied_required_params_do_not_read_as_missing``
                # 会红。
                spurious = {
                    f"Required parameter '{p.name}' is missing"
                    for p in (meta.parameters or [])
                    if getattr(p, "required", False) and p.name in dynamic
                }
                errors.extend(v for v in violations if v not in spurious)
            except Exception as exc:  # pragma: no cover
                warnings.append(f"边界预检失败：{exc}")
        for pname in dynamic:
            p = spec_params.get(pname)
            if p is not None and p.allowed_values:
                warnings.append(f"参数 {pname!r} 为枚举型却由 $expr 动态提供——"
                                "运行期才校验，建议改为字面量")
        # 悬空引用（review F4）：$expr 引用的标识符必须解析得到。警告级。
        for pname, v in (params or {}).items():
            if isinstance(v, dict) and "$expr" in v:
                for tok in _expr_idents(v["$expr"]) - known:
                    warnings.append(
                        f"参数 {pname!r} 的表达式引用了未知名字 {tok!r}"
                        "（节点 id 改名了？）——运行期会报 unknown variable")
        steps.append({"id": sid, "skill": skill,
                      "errors": errors, "warnings": warnings})

    # if/loop/set 的表达式同样查悬空引用（全局 problems，警告级）。
    def _lint_exprs(nodes, depth=0):
        if depth > _MAX_TREE_DEPTH:
            return
        for n in nodes or []:
            if not isinstance(n, dict):
                continue
            for field, ex in _node_exprs(n):
                for tok in _expr_idents(ex) - known:
                    problems.append(
                        f"警告：节点 {n.get('id', '?')!r} 的 {field} 引用了"
                        f"未知名字 {tok!r}——运行期会报 unknown variable")
            for branch in _child_branches(n):
                _lint_exprs(branch, depth + 1)
    _lint_exprs(spec_dict.get("nodes"))

    declared_rank = _SAFETY_RANK.get(str(spec.safety_level).lower(), 1)
    if declared_rank < max_leaf_rank:
        # P4 硬化：折叠/保存不得洗白审批级别（X_safety review 第 3 条）。
        # 只在保存/校验路径强制——既有存量 spec 启动加载仍是警告级
        # （loader 不经此处），避免升级即打坏老工作流。
        problems.append(
            f"声明的 safety_level={spec.safety_level!r} 低于子技能最高级"
            f"（{[k for k, v in _SAFETY_RANK.items() if v == max_leaf_rank][0]}）"
            "——effective level 必须 ≥ max(叶子)，请上调 safety_level")

    # 以「警告：」开头的为 advisory（叶子-max 提示、悬空引用）——不拦保存。
    hard_problems = [p for p in problems if not p.startswith("警告：")]
    ok = not hard_problems and all(not s["errors"] for s in steps)
    return {"ok": ok, "problems": problems, "steps": steps}


# ── resolved-skills snapshot (sync-ready record, RFC §7) ────────────────────

def resolved_skills_snapshot(spec_dict: dict) -> dict[str, str]:
    """{skill_name: registry version} for every step at save time — the data
    basis for P4 version pinning / drift warnings."""
    reg = _registry
    if reg is None:
        return {}
    out: dict[str, str] = {}
    step_nodes: list[dict] = []
    _walk_steps(spec_dict.get("nodes"), step_nodes)
    for n in step_nodes:
        skill = n.get("skill") or ""
        if skill and skill not in out and reg.has(skill):
            try:
                out[skill] = reg._get_metadata(reg.get(skill)).version
            except Exception:  # pragma: no cover
                pass
    return out


def save_extra_meta() -> dict:
    """Machine/author stamps for the sync-ready record.

    ⚠️ 这两个值**会离开本机**:技能上传时它们跟着 manifest 走,推送服务器把
    它们写进 inbox 索引并打进日志(见 ``mast.update.server``)。也就是说,分享
    一个技能就等于把**登录用户名与主机名**一起分享出去 —— 而这两样在多数站点
    都是真实姓名或课题组标识。

    默认仍然采集(审核方要知道技能来自哪台机器)。不希望外发的站点设
    ``MAST_ANON_AUTHOR=1``,两个字段一起留空 —— 上传照常,只是不带身份。
    """
    if os.environ.get("MAST_ANON_AUTHOR", "").strip().lower() in ("1", "true", "yes"):
        return {"_author": "", "_machine": ""}
    try:
        author = getpass.getuser()
    except Exception:  # pragma: no cover
        author = ""
    try:
        machine = socket.gethostname()
    except Exception:  # pragma: no cover
        machine = ""
    return {"_author": author, "_machine": machine}


# ── builder agent: NL → CompositeSpec（P2-E，主轨「数据而非代码」） ──────────

_GEN_SYSTEM = """你是 MAST 技能构建 agent。把用户的自然语言需求转换成一个 CompositeSpec JSON（声明式 STM 工作流）。

只输出一个 JSON 对象，不要任何解释文字。格式：
{"name": "<字母/数字/_/-/中文，≤81>", "description": "<一句话>", "safety_level": "confirm",
 "params": [{"name": "...", "type": "int|float|str|bool", "default": ..., "description": "...", "required": false}],
 "nodes": [<节点列表>], "tags": [...]}

节点类型（顺序即执行序）：
- step: {"type":"step","id":"<唯一标识符>","skill":"<目录中的技能名>","params":{"参数名": 字面量 或 {"$expr":"表达式"}}}
- if:   {"type":"if","id":"...","cond":"<布尔表达式>","then":[...],"else":[...]}
- loop: {"type":"loop","id":"...","mode":"repeat|foreach|while","count":"<表达式>"(repeat)
         ,"iterable":"<表达式>","var":"item"(foreach),"cond":"<表达式>"(while),"max_iter":<=100,"body":[...]}
- set:  {"type":"set","id":"...","var":"变量名","value":"<表达式字符串>"}
- llm:  {"type":"llm","id":"...","mode":"route","responsibility":"<一句话职责>",
         "inputs":{"名":{"$expr":"..."}},"routes":{"分支名":[...子节点...]},
         "route_descriptions":{"分支名":"描述"},"escape":"<必须是 routes 之一，作为不确定时的安全出口>"}

硬规则：
1. step.skill 必须严格来自下方技能目录，参数名必须匹配；
2. 绝不使用 MotorMove 的 z-approach 方向（物理危险，会被拒绝）；
3. 触达硬件的 loop 必须设 max_iter ≤ 100；
4. 表达式（cond/count/value/$expr）是受限 Python 表达式：可引用工作流参数名、
   上游节点 id（其结果 dict，如 q['quality']）、set 变量；禁属性访问/import；
5. 模糊判断（图像好坏/下一步策略）用 llm 节点而不是写死阈值；llm 必须有 escape 分支；
6. 节点 id 全局唯一、简短、表义。

技能目录（名(参数) — 描述）：
{catalog}
"""


def _gen_model_factory():
    """Patchable in tests. Heavy import lazy."""
    from mast.agents._shared.models import make_chat_model
    return make_chat_model("orchestrator", max_tokens=8000, temperature=0.2)


def _skill_digest(limit: int = 400) -> str:
    cards = get_catalog()["cards"]
    lines = []
    for name, c in list(cards.items())[:limit]:
        ps = ",".join(p["name"] for p in c.get("parameters", []))
        desc = (c.get("description") or "")[:70].replace("\n", " ")
        lines.append(f"{name}({ps}) — {desc}")
    return "\n".join(lines)


def _extract_json_obj(text: str) -> dict | None:
    dec = json.JSONDecoder()
    for i, ch in enumerate(str(text)):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(str(text)[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "nodes" in obj:
            return obj
    return None


def generate_spec_sync(prompt: str, current_spec: dict | None = None,
                       *, model=None, max_attempts: int = 2) -> dict:
    """NL → spec，含「生成 → 静态校验 → 问题回喂修复」环（≤max_attempts 轮）。

    产物是数据（CompositeSpec JSON），不是代码——零 exec、零 import；安全
    问题在结构上不存在（设计期 lint + 运行期 ctx.run 门照常把关）。不落盘：
    前端把结果载入画布当草稿，保存与否由用户决定。"""
    try:
        mdl = model if model is not None else _gen_model_factory()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"模型不可用：{exc}"}
    sys_msg = _GEN_SYSTEM.replace("{catalog}", _skill_digest())
    user = f"用户需求：{prompt}"
    if current_spec:
        user += ("\n\n当前画布上的工作流（按需求修改它而不是从零开始）：\n"
                 + json.dumps(current_spec, ensure_ascii=False))
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": user}]
    last_report = None
    for attempt in range(max_attempts):
        try:
            resp = mdl.invoke(msgs)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"生成调用失败：{exc}"}
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content
                               if isinstance(b, dict) and b.get("type") == "text")
        spec = _extract_json_obj(str(content))
        if spec is None:
            msgs.append({"role": "assistant", "content": str(content)[:4000]})
            msgs.append({"role": "user",
                         "content": "你没有输出合法的 spec JSON。只输出一个 JSON 对象。"})
            continue
        report = validate_spec_payload(spec)
        last_report = report
        if report["ok"]:
            return {"spec": spec, "report": report, "attempts": attempt + 1}
        # 校验问题回喂，修复一轮
        issues = list(report.get("problems") or [])
        for s in report.get("steps") or []:
            issues += [f"[{s['id']}] {e}" for e in s["errors"]]
        msgs.append({"role": "assistant", "content": json.dumps(spec, ensure_ascii=False)[:6000]})
        msgs.append({"role": "user",
                     "content": "你的 spec 未通过校验，修复以下问题后重新输出完整 JSON：\n- "
                                + "\n- ".join(issues[:20])})
    return {"error": "生成的 spec 未通过校验", "report": last_report}


# ── 分享到实验室库（P5：manifest-only，永不发送 .py） ────────────────────────

def share_to_lab_sync(name: str) -> dict:
    """把一个已保存的工作流 manifest（含 sync-ready 字段）POST 到实验室
    push server 的 /skills/upload。显式 publish 动作而非 save 副作用（R5
    降级共识）；本地为真源，失败不影响任何本地状态。"""
    from mast.webui.composite_panel import composite_store
    from mast.skills.composite.version_store import VersionStoreError
    store = composite_store()
    try:
        store.load(name)                      # 名字合法性 + 存在性
    except VersionStoreError as exc:
        return {"error": str(exc)}
    try:
        raw_path = store._root / f"{name}.json"   # 原始 JSON 含 _author/_sha256
        manifest = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"读取 manifest 失败：{exc}"}
    try:
        from mast._runtime_paths import project_root
        from mast.update.client import (
            _read_token, _require_https, read_server_url,
        )
        root = project_root()
        url = read_server_url(root)
        token = _read_token(root)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"读取实验室服务器配置失败：{exc}"}
    if not url or not token:
        return {"error": "未配置实验室更新服务器（server_url/token）——"
                         "在启动器的推送设置中配置后重试"}
    tls_err = _require_https(url)
    if tls_err:
        return {"error": tls_err}
    try:
        import httpx

        from mast import __version__ as ver
        r = httpx.post(url.rstrip("/") + "/skills/upload",
                       headers={"Authorization": f"Bearer {token}"},
                       json={"manifest": manifest, "client_version": ver},
                       timeout=30)
        if r.status_code == 200:
            body = r.json()
            return {"ok": True, "id": body.get("id"),
                    "status": body.get("status", "pending_review")}
        return {"error": f"服务器拒绝（HTTP {r.status_code}）：{r.text[:200]}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"上传失败：{exc}"}


# ── favorites (per-machine, RFC §6) ─────────────────────────────────────────

_FAV_LOCK = threading.Lock()
_MAX_FAVORITES = 500


def _favorites_path() -> Path:
    from mast._runtime_paths import project_root
    return project_root() / "config" / "builder_favorites.json"


def load_favorites() -> list[str]:
    try:
        p = _favorites_path()
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            favs = d.get("favorites")
            if isinstance(favs, list):
                return [str(x) for x in favs[:_MAX_FAVORITES]]
    except Exception as exc:  # pragma: no cover
        logger.warning("favorites load failed: %s", exc)
    return []


def save_favorites(favorites: list) -> list[str]:
    favs = [str(x)[:120] for x in favorites if isinstance(x, str)][:_MAX_FAVORITES]
    p = _favorites_path()
    with _FAV_LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp",
                                   prefix=p.name + ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"favorites": favs}, f, ensure_ascii=False, indent=1)
            os.replace(tmp, str(p))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return favs


# ── route handlers ───────────────────────────────────────────────────────────

async def _read_json(request) -> dict:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


def build_routes():
    """Return the Starlette routes (all session-guarded). Insert AFTER launch()."""
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from mast.webui.composite_panel import _hot_register, _hot_unregister, composite_store
    from mast.webui.route_auth import authed_route
    from mast.skills.composite.spec import CompositeSpec
    from mast.skills.composite.version_store import (
        VersionConflictError, VersionStoreError,
    )

    def _catalog(request):
        qp = request.query_params
        cat = get_catalog()
        idx = filter_index(
            cat["index"], q=qp.get("q", ""), category=qp.get("category", ""),
            tag=qp.get("tag", ""), source=qp.get("source", ""),
            safety=qp.get("safety", ""), level=qp.get("level", ""),
            domain=qp.get("domain", ""))
        try:
            page = max(1, int(qp.get("page", "1")))
            size = min(50_000, max(1, int(qp.get("page_size", "50000"))))
        except ValueError:
            page, size = 1, 50_000
        total = len(idx)
        items = idx[(page - 1) * size: page * size]
        return JSONResponse({"total": total, "page": page, "skills": items})

    def _catalog_card(request):
        name = request.path_params.get("name", "")
        card = get_catalog()["cards"].get(name)
        if card is None:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return JSONResponse(card)

    def _composites_list(request):
        return JSONResponse({"composites": composite_store().list_specs()})

    def _composite_get(request):
        name = request.path_params.get("name", "")
        store = composite_store()
        try:
            spec = store.load(name)
        except VersionStoreError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse({"spec": spec.to_dict(),
                             "versions": store.list_versions(name)})

    # P1 review F1（critical）：async handler 在事件循环上做文件 I/O + 拿
    # version_store 的 per-root RLock 会冻死整个 loop（该锁与 Gradio 工作线程
    # 共用）。读完 body 后阻塞段一律 run_in_threadpool —— 与 route_auth 保留
    # sync handler 线程池语义是同一条纪律。
    from starlette.concurrency import run_in_threadpool

    def _save_sync(name, body):
        spec_dict = body.get("spec")
        if not isinstance(spec_dict, dict):
            return JSONResponse({"error": "missing_spec"}, status_code=400)
        if spec_dict.get("name") != name:
            return JSONResponse(
                {"error": "name_mismatch",
                 "detail": "URL 中的名字与 spec.name 不一致"}, status_code=400)
        report = validate_spec_payload(spec_dict)
        if not report["ok"]:
            return JSONResponse({"error": "invalid_spec", **report},
                                status_code=422)
        try:
            spec = CompositeSpec.from_dict(spec_dict)
        except Exception as exc:
            return JSONResponse({"error": "invalid_spec",
                                 "problems": [str(exc)]}, status_code=422)
        base_version = body.get("base_version")
        extra = save_extra_meta()
        extra["_resolved_skills"] = resolved_skills_snapshot(spec_dict)
        try:
            saved = composite_store().save(
                spec, base_version=(int(base_version)
                                    if base_version is not None else None),
                extra_meta=extra)
        except VersionConflictError as exc:
            stored = 0
            try:
                stored = composite_store().load(name).version
            except Exception:
                pass
            return JSONResponse({"error": "version_conflict",
                                 "detail": str(exc),
                                 "stored_version": stored}, status_code=409)
        except VersionStoreError as exc:
            return JSONResponse({"error": "invalid_spec",
                                 "problems": [str(exc)]}, status_code=422)
        msg = _hot_register(saved.name)
        invalidate_catalog()
        return JSONResponse({"ok": True, "version": saved.version,
                             "hot_registered": "已热注册" in msg,
                             "message": msg.strip(),
                             "report": report})

    async def _save(request):
        name = request.path_params.get("name", "")
        body = await _read_json(request)
        return await run_in_threadpool(_save_sync, name, body)

    async def _validate(request):
        body = await _read_json(request)
        spec_dict = body.get("spec")
        if not isinstance(spec_dict, dict):
            return JSONResponse({"error": "missing_spec"}, status_code=400)
        report = await run_in_threadpool(validate_spec_payload, spec_dict)
        return JSONResponse(report)

    def _restore_sync(name, version):
        try:
            saved = composite_store().restore(name, version)
        except VersionStoreError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        msg = _hot_register(saved.name)
        invalidate_catalog()
        return JSONResponse({"ok": True, "version": saved.version,
                             "hot_registered": "已热注册" in msg,
                             "message": msg.strip()})

    async def _restore(request):
        name = request.path_params.get("name", "")
        body = await _read_json(request)
        try:
            version = int(body.get("version"))
        except (TypeError, ValueError):
            return JSONResponse({"error": "missing_version"}, status_code=400)
        return await run_in_threadpool(_restore_sync, name, version)

    def _delete(request):
        name = request.path_params.get("name", "")
        store = composite_store()
        try:  # review F2：穿越名等异常不许变 500 预言机
            if not store.exists(name):
                return JSONResponse({"error": "not_found"}, status_code=404)
            store.delete(name)
        except VersionStoreError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        msg = _hot_unregister(name)
        invalidate_catalog()
        return JSONResponse({"ok": True, "message": msg.strip()})

    def _diff(request):
        name = request.path_params.get("name", "")
        qp = request.query_params
        try:
            v1, v2 = int(qp.get("v1")), int(qp.get("v2"))
        except (TypeError, ValueError):
            return JSONResponse({"error": "missing_versions"}, status_code=400)
        try:
            return JSONResponse(composite_store().diff(name, v1, v2))
        except VersionStoreError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    async def _generate(request):
        body = await _read_json(request)
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return JSONResponse({"error": "missing_prompt"}, status_code=400)
        cur = body.get("current_spec")
        result = await run_in_threadpool(
            generate_spec_sync, prompt,
            cur if isinstance(cur, dict) else None)
        status = 200 if "spec" in result else 422
        return JSONResponse(result, status_code=status)

    async def _share(request):
        name = request.path_params.get("name", "")
        result = await run_in_threadpool(share_to_lab_sync, name)
        return JSONResponse(result, status_code=200 if result.get("ok") else 502)

    def _decisions(request):
        """决策日志查看（P4）：graduation 训练底料的出口。倒序最近 N 条，
        可按 workflow/mechanism 过滤。"""
        qp = request.query_params
        try:
            limit = min(500, max(1, int(qp.get("limit", "100"))))
        except ValueError:
            limit = 100
        wf = qp.get("workflow", "")
        mech = qp.get("mechanism", "")
        out: list[dict] = []
        try:
            from mast.skills.composite.llm_node import decision_log_path
            p = decision_log_path()
            if p.exists():
                lines = p.read_text(encoding="utf-8").strip().splitlines()
                for line in reversed(lines):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if wf and rec.get("workflow") != wf:
                        continue
                    if mech and rec.get("mechanism") != mech:
                        continue
                    out.append(rec)
                    if len(out) >= limit:
                        break
        except Exception as exc:  # pragma: no cover
            return JSONResponse({"decisions": [], "error": str(exc)})
        return JSONResponse({"decisions": out, "total_returned": len(out)})

    def _personas(request):
        try:
            from mast.skills.composite.persona import list_personas
            return JSONResponse({"personas": list_personas()})
        except Exception as exc:  # pragma: no cover — defensive
            return JSONResponse({"personas": [], "error": str(exc)})

    def _favorites_get(request):
        return JSONResponse({"favorites": load_favorites()})

    async def _favorites_post(request):
        body = await _read_json(request)
        favs = body.get("favorites")
        if not isinstance(favs, list):
            return JSONResponse({"error": "missing_favorites"}, status_code=400)
        try:
            return JSONResponse({"ok": True, "favorites": save_favorites(favs)})
        except Exception as exc:  # pragma: no cover
            return JSONResponse({"error": str(exc)}, status_code=500)

    a = authed_route
    return [
        Route("/skills/catalog", a(_catalog), methods=["GET"]),
        Route("/skills/catalog/{name}", a(_catalog_card), methods=["GET"]),
        Route("/composites", a(_composites_list), methods=["GET"]),
        Route("/composites/validate", a(_validate), methods=["POST"]),
        Route("/composites/{name}", a(_composite_get), methods=["GET"]),
        Route("/composites/{name}", a(_save), methods=["POST"]),
        Route("/composites/{name}", a(_delete), methods=["DELETE"]),
        Route("/composites/{name}/restore", a(_restore), methods=["POST"]),
        Route("/composites/{name}/diff", a(_diff), methods=["GET"]),
        Route("/builder/favorites", a(_favorites_get), methods=["GET"]),
        Route("/builder/favorites", a(_favorites_post), methods=["POST"]),
        Route("/builder/personas", a(_personas), methods=["GET"]),
        Route("/builder/generate", a(_generate), methods=["POST"]),
        Route("/builder/decisions", a(_decisions), methods=["GET"]),
        Route("/builder/share/{name}", a(_share), methods=["POST"]),
    ]


__all__ = [
    "set_live_registry", "invalidate_catalog", "get_catalog", "warm_catalog",
    "build_catalog", "filter_index", "validate_spec_payload",
    "resolved_skills_snapshot", "save_extra_meta",
    "load_favorites", "save_favorites", "build_routes",
]
