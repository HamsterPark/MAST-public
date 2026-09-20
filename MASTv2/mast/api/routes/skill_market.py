"""技能市场 + 订阅列表的 HTTP 入口。

全体技能是**市场**；用户维护自己的**订阅列表**，那份列表才是 agent 日常的装载面。
出厂是「全订阅」，所以这个功能上线当天工具面一个字都不变（见
``mast.skills.subscription`` 的 absent=全订阅 一节）。

为什么前缀是 ``/skill-market`` 而不是 ``/skills/market``
=======================================================
``skills_ext.py`` 有 ``GET /skills/{name}`` —— 它会把 ``/skills/market`` 当成
``name="market"`` 捕获掉，返回 **200 加一个错的 handler**。本仓踩过两次路由遮蔽，
``skill_overlay.py`` 就是因此换的前缀。靠「让本模块先注册」也能绕开，但那是把正确性
挂在注册顺序上 —— 一个别人可以随手改、而且改的人不会知道这条依赖的东西。

THIN RELAY ONLY（房规）
======================
每个 handler 惰性 import 核心，在 try/except 里，缺失或失败时返回
``degraded=True`` 的有效响应 —— **绝不 500**。

一条比「别 500」更重要的纪律
============================
响应里不能只有 ``ok``。改订阅只改了一个 holder，而 agent 的工具表在**建图时冻结**：

* 有任务在跑 ⇒ 重建排队 —— 存下来了，没生效；
* 重建被调度了但失败 —— 存下来了，没生效；
* 根本没有活的 runtime（独立 API 模式）—— 存下来了，下次启动才生效。

三条路径都不报错。所以写响应统一带 ``rebuild_note``（人话，UI 逐字显示）、
``agent_path_pending``、``fingerprint_matches``（三态）。

订阅不是安全机制
================
未订阅的技能仍在 SkillRegistry 里：手动执行、composite 子步、conduct 的
``ctx.run`` 照常够得到它。这里的写端点因此**不过 admin PIN** —— PIN 的威胁模型是
「一只人手误改了包络或授能」（见 ``admin_pin.py``），而订阅改错的代价是工具面少
几个技能、一键 reset 就回来。要推翻这个判断需要回答：订阅列表能让哪一个此前做不到
的危险动作变得能做？答案是一个都没有。
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Query, Request

from mast.api.schemas_skill_market import (
    AuditEntry,
    AuditResponse,
    ImportReport,
    ImportRequest,
    ImportResponse,
    LabFetchResponse,
    LabIndexEntry,
    LabIndexResponse,
    LabPublishRequest,
    LabPublishResponse,
    ManifestEntry,
    ManifestMissing,
    MarketCatalogResponse,
    MarketEntry,
    MarketStatusResponse,
    Recommendation,
    RecommendationListResponse,
    RecommendationResolveRequest,
    RecommendationResolveResponse,
    SubscriptionManifest,
    SubscriptionWriteRequest,
    SubscriptionWriteResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["skill-market"])


# ─────────────────────────────────────────────────────────────────────────────
# 取核心（全部惰性、全部可缺席）
# ─────────────────────────────────────────────────────────────────────────────

def _runtime(request: Request):
    return getattr(getattr(request.app.state, "ctx", None), "live_app", None)


def _registry(request: Request):
    ctx = getattr(request.app.state, "ctx", None)
    reg = getattr(ctx, "skill_registry", None)
    if reg is not None:
        return reg
    rt = _runtime(request)
    return getattr(rt, "_registry", None) if rt is not None else None


def _catalog_index(request: Request) -> list[dict]:
    """市场目录（= builder palette 那份，已应用 admin 覆写）。"""
    if _registry(request) is None:
        return []
    from mast.webui.builder_api import get_catalog
    return list(get_catalog().get("index") or [])


def _market_names(request: Request) -> set[str]:
    """市场全集 —— 订阅门算补集要用它。

    优先走注册表（那是真源）；目录缓存只是它的一个视图。
    """
    reg = _registry(request)
    if reg is not None:
        try:
            return {m.name for m in reg.list_skills()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("市场全集取不到：%s", exc)
    return {str(e.get("name") or "") for e in _catalog_index(request)} - {""}


def _fingerprints(registry) -> dict:
    """生效探针 —— 与 ``skill_overlay._fingerprints`` 同源。

    刻意 import 它而不是抄一份：两处各写一遍，迟早只有一处是对的（本仓把这件事
    叫做「一个记号两种结构」）。
    """
    try:
        from mast.api.routes.skill_overlay import _fingerprints as _fp
        return _fp(registry)
    except Exception as exc:  # noqa: BLE001
        logger.debug("生效探针不可用：%s", exc)
        return {"fingerprint_matches": None}


def _refresh(request: Request, reason: str) -> dict:
    """把「装载面变了」推到三条消费者链上，并如实回答「生效了没有」。

    走 ``refresh_after_skill_change`` 而不是只调 ``_request_composite_rebuild``：
    技能**集合**虽然没变，但工具**面**变了，而 ``/agents/tools`` 与 builder palette
    的缓存都按面缓存 —— 不失效的话界面会继续显示退订前那张表。
    """
    out = {"note": "", "pending": None}
    rt = _runtime(request)
    if rt is None:
        out["note"] = "（没有活的运行时 —— 已存盘，下次启动生效）"
        out["pending"] = True
        return out
    try:
        from mast.admin.reload_wiring import ORCH_PENDING_QUEUED, refresh_after_skill_change

        outcome = refresh_after_skill_change(f"subscription: {reason}")
        out["pending"] = outcome.agent_path_pending
        if outcome.orchestrator == ORCH_PENDING_QUEUED:
            out["note"] = "（任务运行中：已排队，当前任务结束后自动生效）"
        elif outcome.agent_path_pending:
            out["note"] = f"（agent 侧尚未跟上：{outcome.describe()}）"
        else:
            out["note"] = "（agent 工具表已重建）"
    except Exception as exc:  # noqa: BLE001
        logger.warning("订阅变更后刷新失败：%s", exc, exc_info=True)
        out["note"] = f"（刷新失败：{exc} —— 重启才保险）"
        out["pending"] = True
    return out


def _count(subscribed: "frozenset[str] | None", market: set) -> int:
    """订阅面上有几个技能。

    ``subscribed or market`` 在这里是**错的**：退订到一个不剩时 ``subscribed`` 是
    空集（falsy），``or`` 会滑到 market 去，于是「我退订了一切」在界面上显示成
    「全部 459 个都订阅着」。空集是一个答案，不是「没有答案」。
    """
    if subscribed is None:          # 未定制 = 全订阅
        return len(market)
    return len(subscribed | (sub_mandatory() & market))


def sub_mandatory() -> frozenset:
    from mast.skills.subscription import MANDATORY_SKILLS
    return MANDATORY_SKILLS


def _apply(request: Request, mutate, *, reason: str) -> dict:
    """所有订阅写的唯一收口：改 holder → 刷新 → 收集诚实字段。

    ``mutate`` 是一个只做 holder 变更、返回 holder 结果 dict 的可调用对象。
    """
    res = dict(mutate() or {})
    if not res.get("ok", True):
        return {**res, "rebuild_note": "", "agent_path_pending": None,
                "fingerprint_matches": None}
    refreshed = _refresh(request, reason)
    fp = _fingerprints(_registry(request))
    return {
        **res,
        "rebuild_note": ("已保存" + (refreshed["note"] or "")),
        "agent_path_pending": refreshed["pending"],
        "fingerprint_matches": fp.get("fingerprint_matches"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /skill-market/catalog
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/skill-market/catalog", response_model=MarketCatalogResponse)
def market_catalog(
    request: Request,
    q: str = Query(default=""),
    category: str = Query(default=""),
    tag: str = Query(default=""),
    source: str = Query(default=""),
    safety: str = Query(default=""),
    level: str = Query(default=""),
    domain: str = Query(default=""),
    subscribed: str = Query(default="", description='"1" 只看已订阅 / "0" 只看未订阅 / "" 全部'),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50000, ge=1, le=50000),
) -> MarketCatalogResponse:
    """市场目录 —— builder palette 那份，外加 ``subscribed`` / ``mandatory`` 两列。

    ``subscribed`` **不进目录缓存**：订阅的变化频率远高于技能集合，进了缓存就得
    每次订阅写都把整份目录作废（把一个便宜操作变贵）。它在这里现 join，一次
    frozenset 查询。
    """
    try:
        from mast.skills import subscription as sub
        from mast.webui.builder_api import filter_index
    except Exception as exc:  # noqa: BLE001
        return MarketCatalogResponse(page=page, degraded=True, reason=str(exc))

    if _registry(request) is None:
        return MarketCatalogResponse(page=page, degraded=True,
                                     reason="没有活的技能注册表（独立 API 模式）")
    try:
        idx = filter_index(_catalog_index(request), q=q, category=category, tag=tag,
                           source=source, safety=safety, level=level, domain=domain)
        subs = sub.subscribed_names()
        pending_by_skill = {r["skill"]: r["id"] for r in sub.pending_recommendations()}

        def _is_sub(name: str) -> bool:
            return True if subs is None else (name in subs or name in sub.MANDATORY_SKILLS)

        if subscribed in ("0", "1"):
            want = subscribed == "1"
            idx = [e for e in idx if _is_sub(str(e.get("name") or "")) is want]

        total = len(idx)
        items = idx[(page - 1) * page_size: page * page_size]
        rows = []
        for e in items:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "")
            rows.append(MarketEntry(
                name=name,
                zh=str(e.get("zh") or ""),
                category=str(e.get("category") or ""),
                safety=str(e.get("safety") or ""),
                level=int(e.get("level", 0) or 0),
                source=str(e.get("source") or "other"),
                source_zh=str(e.get("source_zh") or ""),
                tags=list(e.get("tags") or []),
                domain=str(e.get("domain") or "其他"),
                subscribed=_is_sub(name),
                mandatory=name in sub.MANDATORY_SKILLS,
                pending_rec_id=pending_by_skill.get(name, ""),
            ))
        return MarketCatalogResponse(total=total, page=page, skills=rows,
                                     customised=sub.is_customised())
    except Exception as exc:  # noqa: BLE001
        logger.warning("市场目录失败：%s", exc, exc_info=True)
        return MarketCatalogResponse(page=page, degraded=True, reason=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# GET /skill-market/status
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/skill-market/status", response_model=MarketStatusResponse)
def market_status(request: Request) -> MarketStatusResponse:
    """订阅现状 + **agent 那边跟上了没有**。"""
    try:
        from mast.skills import subscription as sub
    except Exception as exc:  # noqa: BLE001
        return MarketStatusResponse(degraded=True, reason=str(exc))
    try:
        st = sub.state(_market_names(request) or None)
        fp = _fingerprints(_registry(request))
        rt = _runtime(request)
        pending = None
        try:
            if rt is not None:
                pending = bool(rt.pending_agent_rebuild())
        except Exception:  # noqa: BLE001
            pending = None      # 判断不了就说判断不了
        return MarketStatusResponse(
            customised=st["customised"],
            subscribed_count=st["subscribed_count"],
            market_total=st["market_total"],
            mandatory=st["mandatory"],
            missing_entries=st["missing_entries"],
            unreadable=st["unreadable"],
            pending_count=st["pending_count"],
            store_path=st["store_path"],
            agent_path_pending=pending,
            fingerprint_matches=fp.get("fingerprint_matches"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("市场状态失败：%s", exc, exc_info=True)
        return MarketStatusResponse(degraded=True, reason=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# POST /skill-market/subscription
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/skill-market/subscription", response_model=SubscriptionWriteResponse)
def write_subscription(request: Request,
                       body: SubscriptionWriteRequest) -> SubscriptionWriteResponse:
    """增量改订阅。**没有整体替换** —— 理由见 schemas 模块 docstring。"""
    try:
        from mast.skills import subscription as sub
    except Exception as exc:  # noqa: BLE001
        return SubscriptionWriteResponse(ok=False, degraded=True, reason=str(exc))

    market = _market_names(request)
    if not market:
        return SubscriptionWriteResponse(
            ok=False, degraded=True,
            reason="没有活的技能注册表 —— 不知道市场全集就不能定制订阅"
                   "（否则第一次定制会把没列出来的全部退订）")

    # 注册表不认识的名字：照收（技能集合是动态的），但要说。
    unknown = sorted({*body.subscribe, *body.unsubscribe} - market)

    def _mutate():
        skipped: list[str] = []
        changed: list[str] = []
        materialised = False
        ok = True
        reason = ""
        if body.subscribe:
            r = sub.subscribe(body.subscribe, all_names=market, via=sub.VIA_UI)
            ok = ok and r.get("ok", False)
            reason = reason or r.get("reason", "")
            changed += r.get("changed") or []
            materialised = materialised or bool(r.get("materialised"))
        if body.unsubscribe:
            r = sub.unsubscribe(body.unsubscribe, all_names=market, via=sub.VIA_UI)
            ok = ok and r.get("ok", False)
            reason = reason or r.get("reason", "")
            changed += r.get("changed") or []
            skipped += r.get("skipped_mandatory") or []
            materialised = materialised or bool(r.get("materialised"))
        return {"ok": ok, "reason": reason, "changed": sorted(set(changed)),
                "skipped_mandatory": sorted(set(skipped)),
                "materialised": materialised,
                "customised": sub.is_customised(),
                "subscribed_count": _count(sub.subscribed_names(), market)}

    res = _apply(request, _mutate, reason="ui")
    note = res.get("rebuild_note", "")
    if res.get("skipped_mandatory"):
        note += ("；未退订必装技能 " + "、".join(res["skipped_mandatory"])
                 + "（它们保证工具面上永远有「停下来 / 退针」可用）")
    return SubscriptionWriteResponse(
        ok=bool(res.get("ok", True)), reason=str(res.get("reason") or ""),
        customised=bool(res.get("customised")),
        materialised=bool(res.get("materialised")),
        changed=list(res.get("changed") or []),
        skipped_mandatory=list(res.get("skipped_mandatory") or []),
        unknown=unknown,
        subscribed_count=int(res.get("subscribed_count") or 0),
        rebuild_note=note,
        agent_path_pending=res.get("agent_path_pending"),
        fingerprint_matches=res.get("fingerprint_matches"),
    )


@router.post("/skill-market/subscription/reset",
             response_model=SubscriptionWriteResponse)
def reset_subscription(request: Request) -> SubscriptionWriteResponse:
    """回到出厂态：全订阅。"""
    try:
        from mast.skills import subscription as sub
    except Exception as exc:  # noqa: BLE001
        return SubscriptionWriteResponse(ok=False, degraded=True, reason=str(exc))
    res = _apply(request, lambda: sub.reset_to_default(), reason="reset")
    return SubscriptionWriteResponse(
        ok=bool(res.get("ok", True)), customised=False,
        subscribed_count=len(_market_names(request)),
        rebuild_note=res.get("rebuild_note", ""),
        agent_path_pending=res.get("agent_path_pending"),
        fingerprint_matches=res.get("fingerprint_matches"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 推荐
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/skill-market/audit", response_model=AuditResponse)
def market_audit(limit: int = Query(default=50, ge=1, le=500)) -> AuditResponse:
    """订阅面的变更史：什么时候、动了什么、经哪条路。

    ``reset`` 会清掉 entries，但**不清 audit** —— 「他什么时候把定制推倒重来的」
    正是事后最想知道的一件事。
    """
    try:
        from mast.skills import subscription as sub

        return AuditResponse(entries=[AuditEntry(**a) for a in sub.audit_tail(limit)])
    except Exception as exc:  # noqa: BLE001
        return AuditResponse(degraded=True, reason=str(exc))


@router.get("/skill-market/recommendations", response_model=RecommendationListResponse)
def list_recommendations() -> RecommendationListResponse:
    try:
        from mast.skills import subscription as sub

        return RecommendationListResponse(
            pending=[Recommendation(**r) for r in sub.pending_recommendations()],
            resolved=[Recommendation(**r) for r in sub.resolved_recommendations()],
        )
    except Exception as exc:  # noqa: BLE001
        return RecommendationListResponse(degraded=True, reason=str(exc))


@router.post("/skill-market/recommendations/{rec_id}/resolve",
             response_model=RecommendationResolveResponse)
def resolve_recommendation(request: Request, rec_id: str,
                           body: RecommendationResolveRequest
                           ) -> RecommendationResolveResponse:
    """接受 / 拒绝一条 agent 推荐。**这是推荐唯一能变成订阅的地方。**

    agent 手上没有写订阅的工具（见 ``agents/_shared/market_tools.py``）—— 它只能
    把想法放进 pending，改自己的工具面是人面上的动作。
    """
    try:
        from mast.skills import subscription as sub
    except Exception as exc:  # noqa: BLE001
        return RecommendationResolveResponse(ok=False, degraded=True, reason=str(exc))

    market = _market_names(request)
    res = _apply(request,
                 lambda: sub.resolve_recommendation(rec_id, body.accept,
                                                    all_names=market or None),
                 reason=f"recommendation:{rec_id}")
    rec = res.get("recommendation")
    return RecommendationResolveResponse(
        ok=bool(res.get("ok", False)), reason=str(res.get("reason") or ""),
        recommendation=Recommendation(**rec) if isinstance(rec, dict) else None,
        already_subscribed_by_default=bool(res.get("already_subscribed_by_default")),
        customised=sub.is_customised(),
        changed=[res["skill"]] if res.get("ok") and body.accept and res.get("skill") else [],
        subscribed_count=_count(sub.subscribed_names(), market),
        rebuild_note=res.get("rebuild_note", ""),
        agent_path_pending=res.get("agent_path_pending"),
        fingerprint_matches=res.get("fingerprint_matches"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 分享：导出 / 导入
# ─────────────────────────────────────────────────────────────────────────────

def _composite_spec(name: str) -> dict | None:
    """user_composite 的**原始** spec JSON（内嵌进 manifest 的那份）。

    读盘上那份而不是 ``store.load()`` 的对象：``CompositeSpec.from_dict`` 会丢掉
    全部下划线键（``_author`` / ``_content_sha256`` / ``_saved_at``），而那几个正是
    收到这份 manifest 的人要用来判断「这是谁的、动过没有」的东西。读法与
    ``builder_api.share_to_lab_sync`` 一致。
    """
    try:
        import json

        from mast.webui.composite_panel import composite_store
        store = composite_store()
        store.load(name)                       # 名字合法性 + 存在性
        raw = store._root / f"{name}.json"
        return json.loads(raw.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("取 %s 的 spec 失败：%s", name, exc)
        return None


@router.get("/skill-market/export", response_model=SubscriptionManifest)
def export_subscription(request: Request) -> SubscriptionManifest:
    """把订阅列表导出成一份可以发给别人的 manifest。

    **代码不随单走**：只有 ``user_composite`` 内嵌 spec（那是纯数据），其余来源
    只写名字。这条与 push server ``/skills/upload`` 的红线是同一条 —— 一份能
    自动执行别人 .py 的分享格式就是一条 RCE 通道。
    """
    try:
        import mast
        from mast.skills import subscription as sub
        from mast.skills.overlay.provenance import classify_origin
    except Exception as exc:  # noqa: BLE001
        logger.warning("导出失败：%s", exc)
        return SubscriptionManifest()

    reg = _registry(request)
    subs = sub.subscribed_names()
    names = sorted(subs) if subs is not None else sorted(_market_names(request))
    entries: list[ManifestEntry] = []
    for name in names:
        source, version = "other", ""
        if reg is not None:
            try:
                cls = reg.get(name)
                source = classify_origin(cls)
                version = str(getattr(cls().metadata(), "version", "") or "")
            except Exception:  # noqa: BLE001 — 名字还在清单里但技能没了，照样导出
                source = "absent"
        spec = _composite_spec(name) if source == "user_composite" else None
        entries.append(ManifestEntry(name=name, source=source, version=version,
                                     spec=spec))
    return SubscriptionManifest(
        exported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        machine=_machine_label(),
        app_version=str(getattr(mast, "__version__", "") or ""),
        customised=sub.is_customised(),
        entries=entries,
    )


def _machine_label() -> str:
    try:
        import socket
        return socket.gethostname()
    except Exception:  # noqa: BLE001
        return ""


@router.post("/skill-market/import", response_model=ImportResponse)
def import_subscription(request: Request, body: ImportRequest) -> ImportResponse:
    """导入一份别人的订阅列表。

    顺序：内嵌 composite 先走**既有的** builder 校验/保存/热注册 → 名字匹配 →
    应用订阅 → **一次**刷新。校验一行都不在这里重实现。
    """
    try:
        from mast.skills import subscription as sub
    except Exception as exc:  # noqa: BLE001
        return ImportResponse(ok=False, degraded=True, reason=str(exc))

    man = body.manifest if isinstance(body.manifest, dict) else {}
    if str(man.get("kind") or "") != "mast-skill-subscription":
        return ImportResponse(ok=False, reason="这不是一份订阅列表（kind 不对）")
    raw_entries = man.get("entries")
    if not isinstance(raw_entries, list):
        return ImportResponse(ok=False, reason="manifest 里没有 entries 列表")
    if body.mode not in ("replace", "merge"):
        return ImportResponse(ok=False, reason=f"未知的 mode：{body.mode}")

    parsed: list[dict] = [e for e in raw_entries if isinstance(e, dict) and e.get("name")]
    report = ImportReport()

    # ① 内嵌 composite —— 走既有保存路径（dry_run 时跳过，只报告）
    if not body.dry_run:
        for e in parsed:
            spec = e.get("spec")
            if not isinstance(spec, dict):
                continue
            name = str(e.get("name"))
            try:
                from mast.api.routes.builder import _save_spec
                res = _save_spec(name, spec, base_version=None, require_new=False)
                if getattr(res, "ok", False):
                    report.composites_saved.append(name)
                else:
                    report.composites_failed.append(
                        {"name": name, "error": str(getattr(res, "error", "") or ""),
                         "problems": list(getattr(res, "problems", []) or [])})
            except Exception as exc:  # noqa: BLE001
                report.composites_failed.append({"name": name, "error": str(exc)})

    # ② 名字匹配（composite 存完之后再算，刚落地的那些才算得上「有」）
    market = _market_names(request)
    if not market:
        return ImportResponse(ok=False, degraded=True,
                              reason="没有活的技能注册表 —— 无法比对这份清单")
    wanted: list[str] = []
    for e in parsed:
        name = str(e.get("name"))
        if name in market:
            report.matched.append(name)
            wanted.append(name)
        else:
            src = str(e.get("source") or "")
            report.missing.append(ManifestMissing(
                name=name, source=src, hint=_missing_hint(src)))
    # 括号是必须的：`-` 比 `&` 紧，不写括号读起来像另一个意思。
    # 意思是「市场里有、但这份清单没列的必装项」—— 导入会强制把它们加回来。
    report.mandatory_added = sorted(sub.MANDATORY_SKILLS & (market - set(wanted)))

    if body.dry_run:
        return ImportResponse(ok=True, dry_run=True, report=report,
                              customised=sub.is_customised(),
                              subscribed_count=len(wanted))

    # ③ 应用 + ④ 一次刷新
    def _mutate():
        target = set(wanted) | (sub.MANDATORY_SKILLS & market)
        if body.mode == "merge":
            current = sub.subscribed_names()
            target |= (current if current is not None else market)
        return sub.set_subscribed(target, via=sub.VIA_IMPORT)

    res = _apply(request, _mutate, reason="import")
    return ImportResponse(
        ok=bool(res.get("ok", True)), report=report, dry_run=False,
        customised=True, changed=sorted(wanted),
        subscribed_count=len(res.get("subscribed") if res.get("subscribed") is not None
                             else wanted),
        rebuild_note=res.get("rebuild_note", ""),
        agent_path_pending=res.get("agent_path_pending"),
        fingerprint_matches=res.get("fingerprint_matches"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 分享（实验室中心索引）—— 二期
# ─────────────────────────────────────────────────────────────────────────────

def _lab_server() -> tuple[str, str, str]:
    """``(url, token, 出错原因)``。配置读法与 ``builder_api.share_to_lab_sync`` 一致。"""
    try:
        from mast._runtime_paths import project_root
        from mast.update.client import _read_token, _require_https, read_server_url

        root = project_root()
        url = read_server_url(root)
        token = _read_token(root)
    except Exception as exc:  # noqa: BLE001
        return "", "", f"读取实验室服务器配置失败：{exc}"
    if not url or not token:
        return "", "", ("未配置实验室服务器（server_url / token）—— "
                        "在启动器的推送设置里配置后重试")
    tls = _require_https(url)
    if tls:
        return "", "", tls
    return url.rstrip("/"), token, ""


@router.post("/skill-market/share/publish", response_model=LabPublishResponse)
def publish_subscription(request: Request,
                         body: LabPublishRequest) -> LabPublishResponse:
    """把当前订阅列表发布到实验室中心索引。

    发的是 :func:`export_subscription` 产出的**同一份** manifest —— 不另拼一份，
    否则「导出给别人的」和「发布到中心的」迟早不是一个东西。
    """
    url, token, why = _lab_server()
    if why:
        return LabPublishResponse(ok=False, reason=why)
    man = export_subscription(request).model_dump()
    try:
        import httpx

        import mast
        r = httpx.post(f"{url}/subscriptions/upload",
                       headers={"Authorization": f"Bearer {token}"},
                       json={"manifest": man, "label": body.label, "note": body.note,
                             "client_version": str(getattr(mast, "__version__", ""))},
                       timeout=30)
    except Exception as exc:  # noqa: BLE001
        return LabPublishResponse(ok=False, reason=f"发布失败：{exc}")
    if r.status_code != 200:
        # 服务端的拒绝理由原样带回来 —— 「代码不随单走」那条红线的报文就在里面，
        # 压成一句「发布失败」会让人下次还这么发。
        detail = ""
        try:
            detail = str(r.json().get("detail") or "")
        except Exception:  # noqa: BLE001
            detail = r.text[:200]
        return LabPublishResponse(ok=False, reason=f"服务器拒绝（{r.status_code}）：{detail}")
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    return LabPublishResponse(ok=True, id=str(data.get("id") or ""),
                              status=str(data.get("status") or ""),
                              skill_count=len(man.get("entries") or []))


@router.get("/skill-market/lab-index", response_model=LabIndexResponse)
def lab_index() -> LabIndexResponse:
    """浏览中心索引（只有摘要，没有条目本身）。"""
    url, token, why = _lab_server()
    if why:
        return LabIndexResponse(degraded=True, reason=why)
    try:
        import httpx
        r = httpx.get(f"{url}/subscriptions/index",
                      headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code != 200:
            return LabIndexResponse(degraded=True,
                                    reason=f"服务器返回 {r.status_code}")
        rows = r.json().get("subscriptions") or []
    except Exception as exc:  # noqa: BLE001
        return LabIndexResponse(degraded=True, reason=f"读取失败：{exc}")
    return LabIndexResponse(subscriptions=[LabIndexEntry(**{
        k: v for k, v in row.items() if k in LabIndexEntry.model_fields
    }) for row in rows if isinstance(row, dict)])


@router.get("/skill-market/lab-fetch/{sub_id}", response_model=LabFetchResponse)
def lab_fetch(sub_id: str) -> LabFetchResponse:
    """把中心索引里的一份 manifest 取回本机（**只取回，不导入**）。

    取回与导入分成两步是刻意的：导入会换掉他的工作面，那要他自己按一次
    —— 与覆盖层「写清单是一次编辑，让它生效是一次决定」同一条。

    失败走 ``ok`` / ``reason``，**不折叠进 manifest 的某个字段**。
    """
    url, token, why = _lab_server()
    if why:
        return LabFetchResponse(reason=why)
    try:
        import httpx
        r = httpx.get(f"{url}/subscriptions/download/{sub_id}",
                      headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code != 200:
            return LabFetchResponse(reason=f"服务器返回 {r.status_code}")
        raw = r.json()
    except Exception as exc:  # noqa: BLE001
        return LabFetchResponse(reason=f"读取失败：{exc}")
    if not isinstance(raw, dict) or raw.get("kind") != "mast-skill-subscription":
        return LabFetchResponse(reason="取回的不是一份订阅列表")
    return LabFetchResponse(
        ok=True,
        manifest=SubscriptionManifest(**{k: v for k, v in raw.items()
                                         if k in SubscriptionManifest.model_fields}))


def _missing_hint(source: str) -> str:
    """这个名字在本机没有 —— 告诉他下一步该干什么，而不是只说「没有」。"""
    return {
        "user_composite": "对方的用户组合技能，但 manifest 里没有内嵌 spec —— 请他重新导出",
        "overlay": "来自技能覆盖层：需要先在本机装上对应的覆盖条目或签名技能包",
        "custom": "来自用户自建 .py：代码不随订阅单走，需要单独获取并显式启用",
        "agent_tool": "来自某个 agent 的工具桥接：本机该 agent 版本可能没有它",
    }.get(source, "本机注册表里没有这个技能")
