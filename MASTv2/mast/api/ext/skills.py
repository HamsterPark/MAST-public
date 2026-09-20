"""技能检索 / 技能卡 / 组合技能起草与保存 / Python 技能提议。

## 为什么按「动作」检索

技能是按它**治什么病**命名的（``ForgeAuTip`` = 修一根金针尖），而使用者按**要什么
结果**说话（「打几发 10 V 脉冲、让碘掉到金上」）—— 同一串动作，两个名字，完全不搭。
拿名字去搜会搜不到，然后自己一条条手打，而手打的版本没有技能里的那些护栏（同点
预算、幅值包络、时间预算）。所以检索除了名字 / 描述 / 标签 / 参数名，还按**技能会
发的 Nanonis 命令**建倒排：搜 ``BiasPulse`` 能搜到每一个会打脉冲的技能。

命令集与足迹是**静态**读出来的，出自 ``mast.skills.compliance.skill_footprint``（投稿校验器
用的同一份分析）：技能源码里 ``safe_call("Verb", …)`` / ``urgent_call`` 的字面量，``context.run``
的子技能递归，声明式组合技能取子步的并集。看不透就标 ``verbs_unknown`` —— 不报空集，
空集会被读成「这个技能不碰仪器」。

## 造技能：与进程内 agent 同一份裁决

``/composites/draft`` 与 ``/composites`` 走技能工坊的同一套判据与保存实现
（``skill_forge_tools._validate`` / ``save_composite_impl``）：纯别名硬拒、触硬件的
循环必须写迭代上限、不改人做的技能、CAS。``/skills/proposals`` 走
``propose_python_skill_impl``：落盘待人审，**不注册、不执行**。
"""

from __future__ import annotations

import logging
import statistics
import threading
from typing import Any

from fastapi import APIRouter, Query, Request

from mast.api import direct_exec
from mast.api.ext.common import ExtError, caller_of, registry_of, runtime_of
from mast.api.ext.schemas import CompositeBody, ProposalBody

logger = logging.getLogger(__name__)

router = APIRouter(tags=["skills"])

class SkillIndex:
    """注册表之上的检索索引：每个技能的 Nanonis 命令、子技能与足迹。

    **全部取自 ``mast.skills.compliance.skill_footprint``** —— 投稿校验器（``scripts/skill_check.py``）
    用的同一份静态分析。技能卡与校验器对同一个技能不会给出两种结论；那份分析也处理了这里
    以前自己写的那份没处理的情形（执行上下文被交出去、动词是变量、``execute`` 来自别的模块），
    看不透时说 ``unknown`` 并写明原因，``footprint`` 再按 category 保守取声明值。

    注册表里的类集合变了就重建（键 = 名字 + 类对象身份：同名组合技能存了新版本就是一个新类）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._rows: dict[str, dict] = {}

    def rows(self, registry: Any) -> dict[str, dict]:
        try:
            metas = list(registry.list_skills())
        except Exception as exc:  # noqa: BLE001
            raise ExtError(503, "not_wired", f"技能注册表读不到：{exc}") from exc
        classes: dict[str, Any] = {}
        for m in metas:
            try:
                classes[str(m.name)] = registry.get(str(m.name))
            except Exception:  # noqa: BLE001
                classes[str(m.name)] = None
        # 只看名字与数量的键会让技能卡一直给旧版本的子步与命令。
        key = (id(registry), frozenset((n, id(c)) for n, c in classes.items()))
        with self._lock:
            if key == self._key and self._rows:
                return self._rows
            from mast.skills.compliance import skill_footprint

            rows: dict[str, dict] = {}
            for m in metas:
                name = str(m.name)
                fp = skill_footprint(registry, name)
                rows[name] = {
                    "meta": m, "cls": classes.get(name),
                    "verbs": list(fp.verbs),
                    "verbs_unknown": not fp.verbs_known,
                    "sub_skills": [s for s in fp.sub_skills if s in classes],
                    "footprint": fp.effective,
                    "footprint_basis": fp.basis,
                    "footprint_reasons": list(fp.reasons[:5]),
                }
            self._rows, self._key = rows, key
            return rows


_INDEX = SkillIndex()


def footprint(row: dict) -> str:
    """``pure-analysis`` / ``hardware-read-only`` / ``hardware-write`` / ``unknown``（取自
    ``skill_footprint(...).effective``：静态看得透用静态结论，看不透按 category 保守声明）。"""
    return str(row.get("footprint") or "unknown")


def _intent_map() -> dict[str, set[str]]:
    """意图词 → 技能名（``webui.encyclopedia.INTENT_MAPPING``，静态、可能陈旧 ——
    用的时候对活注册表过滤）。"""
    out: dict[str, set[str]] = {}
    try:
        import re

        from mast.webui.encyclopedia import INTENT_MAPPING

        for m in INTENT_MAPPING:
            skills = {s for s in re.split(r"[^\w]+", str(m.get("skill") or "")) if s}
            for kw in re.split(r"\s*[/,，、]\s*", str(m.get("keywords") or "")):
                kw = kw.strip().lower()
                if kw:
                    out.setdefault(kw, set()).update(skills)
    except Exception:  # noqa: BLE001
        pass
    return out


def _origin(registry: Any, name: str) -> tuple[str, bool, str]:
    """(给人读的来源标签, 是否官方, 语言中立的来源代号)。取不到 ⇒ ``other``，**不是**官方。"""
    try:
        from mast.agents._shared.skill_forge_tools import (
            _OFFICIAL_ORIGINS,
            _ORIGIN_ZH,
            _origin_of,
        )

        o = _origin_of(registry, name)
        return _ORIGIN_ZH.get(o, o), o in _OFFICIAL_ORIGINS, o
    except Exception:  # noqa: BLE001
        return "other", False, "other"


def _enum_value(v: Any) -> str:
    return str(getattr(v, "value", v) or "").lower()


@router.get("/skills/search")
def search_skills(request: Request, q: str = Query(..., min_length=1, max_length=200),
                  limit: int = Query(20, ge=1, le=100)):
    """按动作找技能：名字、描述、标签、参数名、意图词表，以及**技能会发的 Nanonis 命令**。

    ``matched_on`` 说明每一条是凭什么命中的（``verb:Bias_Pulse`` 表示它会发这条命令）。
    官方来源排在前面 —— 官方技能的前置条件、回滚与参数包络是实机验证过的。
    """
    registry = registry_of(request)
    if registry is None:
        raise ExtError(503, "not_wired", "技能注册表未接线", missing=["skill_registry"])
    rows = _INDEX.rows(registry)
    import re

    terms = [t for t in re.split(r"[\s,，]+", q.strip().lower()) if t]
    intents = _intent_map()
    scored: list[tuple[int, bool, str, list[str]]] = []
    for name, row in rows.items():
        meta = row["meta"]
        lname = name.lower()
        desc = str(getattr(meta, "description", "") or "").lower()
        tags = [str(t).lower() for t in (getattr(meta, "tags", None) or ())]
        pnames = [str(p.name).lower() for p in (getattr(meta, "parameters", None) or ())]
        score, why = 0, []
        for t in terms:
            if t == lname:
                score += 100
                why.append("name")
            elif t in lname:
                score += 40
                why.append("name")
            hit_v = [v for v in row["verbs"] if t in v.lower()]
            if hit_v:
                score += 30
                why.append(f"verb:{hit_v[0]}")
            if any(name in intents.get(k, ()) for k in intents if t in k or k in t):
                score += 35
                why.append("intent")
            if any(t in tg for tg in tags):
                score += 20
                why.append("tag")
            if any(t in pn for pn in pnames):
                score += 10
                why.append("param")
            if t in desc:
                score += 15
                why.append("description")
        if score:
            _, official, _code = _origin(registry, name)
            scored.append((score, official, name, list(dict.fromkeys(why))))
    scored.sort(key=lambda x: (-x[0], not x[1], x[2]))
    results = []
    for score, official, name, why in scored[:limit]:
        row = rows[name]
        meta = row["meta"]
        origin, _, origin_code = _origin(registry, name)
        results.append({
            "name": name, "score": score, "origin": origin, "origin_code": origin_code,
            "official": official,
            "safety_level": _enum_value(getattr(meta, "safety_level", "")),
            "category": _enum_value(getattr(meta, "category", "")),
            "footprint": footprint(row), "matched_on": why,
            "description": str(getattr(meta, "description", "") or "")[:200],
        })
    return {"query": q, "count": len(results), "total_matched": len(scored), "results": results}


def _measured(request: Request, name: str) -> tuple[dict | None, str]:
    rt = runtime_of(request)
    repos = getattr(rt, "_v2_repos", None)
    if repos is None:
        return None, "v2 记录未接线，没有这台仪器上的实测耗时"
    try:
        rows = repos.actions._fetchall(
            "SELECT duration_ms FROM actions WHERE action_type = ? AND status = 'succeeded' "
            "AND duration_ms IS NOT NULL ORDER BY hlc DESC LIMIT 200", (name,))
    except Exception as exc:  # noqa: BLE001
        return None, f"读不到实测耗时：{exc}"
    vals = sorted(float(r["duration_ms"]) / 1000.0 for r in rows if r.get("duration_ms") is not None)
    if not vals:
        return None, "这台仪器上还没有这个技能的成功记录"
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    return {"n": len(vals), "p50_s": round(statistics.median(vals), 2),
            "p95_s": round(p95, 2), "max_s": round(vals[-1], 2)}, ""


@router.get("/skills/{name}")
def skill_card(name: str, request: Request):
    """技能卡：参数（类型 / 单位 / 必填 / 默认 / 范围 / 取值集）、前置条件、能力标签、
    安全级、来源，以及 Nanonis 命令、足迹分类、是否取仪器令牌、是否要样品、哪些参数
    接受 ``"5n"`` 写法、本机是否关闭、这台仪器上的实测耗时。"""
    registry = registry_of(request)
    if registry is None:
        raise ExtError(503, "not_wired", "技能注册表未接线", missing=["skill_registry"])
    rows = _INDEX.rows(registry)
    row = rows.get(name)
    if row is None:
        near = sorted(n for n in rows if name.lower() in n.lower())[:8]
        raise ExtError(404, "unknown_skill", f"没有叫 {name!r} 的技能；按动作搜用 /skills/search",
                       did_you_mean=near)
    meta = row["meta"]
    try:
        eff = registry._get_metadata(row["cls"]) if row["cls"] is not None else meta
    except Exception:  # noqa: BLE001
        eff = meta
    from mast.agents._shared.skill_forge_tools import _card, _origin_of

    origin_code = _origin_of(registry, name)
    card = _card(eff, origin_code)
    try:
        from mast.core.instrument_lock import needs_token

        token = bool(needs_token(eff, name))
    except Exception:  # noqa: BLE001
        token = None
    try:
        from mast.core.sample_gate import requires_sample

        needs_sample = bool(requires_sample(eff, name))
    except Exception:  # noqa: BLE001
        needs_sample = None
    try:
        from mast.agents._shared.skill_adapter import _si_params

        si = {k: ("prefix_required" if v else "prefix_optional") for k, v in _si_params(eff).items()}
    except Exception:  # noqa: BLE001
        si = {}
    face = {"disabled": False, "reason": ""}
    try:
        from mast.skills.tool_face import compute

        f = compute()
        if name in f.hardware:
            face = {"disabled": True, "reason": "硬件模块关闭（这台仪器没有这个硬件）"}
        elif name in f.advanced:
            face = {"disabled": True, "reason": "高级能力未授予"}
        elif name in f.unsubscribed:
            face = {"disabled": False, "reason": "未订阅（只影响内部 agent 的工具表，不影响执行）"}
    except Exception as exc:  # noqa: BLE001
        face = {"disabled": None, "reason": f"关闭名单读不到：{exc}"}
    measured, note = _measured(request, name)
    card.update({
        "origin_code": origin_code,
        "category": _enum_value(getattr(eff, "category", "")),
        "tags": list(getattr(eff, "tags", None) or []),
        "composition_level": getattr(eff, "composition_level", None),
        "footprint": footprint(row),
        "footprint_basis": row.get("footprint_basis"),
        "footprint_reasons": row.get("footprint_reasons") or [],
        "verbs": row["verbs"], "verbs_unknown": row["verbs_unknown"],
        "sub_skills": row["sub_skills"],
        "takes_instrument_token": token,
        "requires_sample": needs_sample,
        "si_params": si,
        "tool_face": face,
        "duration": {"estimated_s": getattr(eff, "estimated_duration_s", None),
                     "measured": measured, "note": note},
    })
    return direct_exec.jsonable(card)


# ─────────────────────────────────────────────────────────────────────
# 造技能
# ─────────────────────────────────────────────────────────────────────

@router.post("/composites/draft")
def draft_composite(body: CompositeBody, request: Request):
    """校验一份组合技能草稿（**不落盘、不注册**），问题整批返回。``spec: "?"`` 返回格式说明。"""
    from mast.agents._shared.skill_forge_tools import (
        SPEC_SYNTAX,
        _default_store,
        _validate,
        official_overlap_hints,
    )

    spec = body.spec
    if isinstance(spec, str):
        if spec.strip() in ("", "?", "？", "help"):
            return {"ok": False, "syntax": SPEC_SYNTAX, "problems": ["(没给 spec —— 上面是格式说明)"]}
        import json

        try:
            spec = json.loads(spec)
        except ValueError as exc:
            return {"ok": False, "syntax": SPEC_SYNTAX, "problems": [f"spec 不是合法 JSON：{exc}"]}
    if not isinstance(spec, dict):
        return {"ok": False, "syntax": SPEC_SYNTAX, "problems": ["spec 要是一个 JSON 对象"]}
    registry = registry_of(request)
    if registry is None:
        raise ExtError(503, "not_wired", "技能注册表未接线", missing=["skill_registry"])
    store = _default_store()
    rep = _validate(spec, registry, store=store)
    rep["hints"] = official_overlap_hints(spec, registry, store=store)
    if not rep["ok"]:
        rep["syntax"] = SPEC_SYNTAX
    return direct_exec.jsonable(rep)


@router.post("/composites")
def save_composite(body: CompositeBody, request: Request):
    """保存组合技能并热注册（署名 ``ext:<名字>``）。之后它就是一个普通技能，经 /jobs 执行，
    每个子步都过全部安全闸。改自己之前存过的同名技能要带 ``base_version``；人做的技能不改。"""
    from mast.agents._shared.skill_forge_tools import save_composite_impl

    registry = registry_of(request)
    if registry is None:
        raise ExtError(503, "not_wired", "技能注册表未接线", missing=["skill_registry"])
    caller = caller_of(request)
    rep = save_composite_impl(registry, caller.agent_id, body.spec, body.base_version,
                              origin="ext_gateway")
    rep = dict(rep)
    if rep.get("ok"):
        rep["message"] = (f"{rep.get('name')} v{rep.get('version')} 已保存并热注册。用 POST /jobs "
                          "按名字执行它；在交接报告里说明你造了什么、为什么。")
    return direct_exec.jsonable(rep)


@router.post("/skills/proposals")
def propose_skill(body: ProposalBody, request: Request):
    """提议一个新的原子技能（Python 草稿）—— 组合表达不了时才用。

    落盘到 ``config/custom_skills/<name>.py`` 待人审，**不注册、不执行**；要用，操作员审过
    代码、把名字加进 ``enabled.json``、重启。回执里附一份合规检查报告（与投稿校验器同一
    套判据），可以照着改。"""
    from mast.agents._shared.skill_forge_tools import propose_python_skill_impl

    registry = registry_of(request)
    if registry is None:
        raise ExtError(503, "not_wired", "技能注册表未接线", missing=["skill_registry"])
    caller = caller_of(request)
    rep = dict(propose_python_skill_impl(registry, caller.agent_id, body.name.strip(),
                                         body.code, body.rationale))
    try:
        from mast.skills.compliance import check_python_source

        rep["compliance"] = check_python_source(
            body.code, filename=f"{body.name.strip()}.py", registry=registry).to_dict()
    except Exception as exc:  # noqa: BLE001 — 合规报告是附带的，拿不到不挡提议
        rep["compliance"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return direct_exec.jsonable(rep)


__all__ = ["SkillIndex", "footprint", "router"]
