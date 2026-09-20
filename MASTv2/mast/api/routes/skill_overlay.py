"""Skill 覆盖层的 HTTP 入口 —— UI 上那个「重新加载技能」按钮的后端。

THIN RELAY ONLY（房规）：每个 handler 惰性 import 核心
（``mast.skills.overlay.*`` + ``CoreRuntime.reload_overlay_skills``），在
try/except 里，缺失或失败时返回 ``degraded=True`` 的有效响应 —— **绝不 500**。
校验/安全逻辑一行都不在这里重实现。

一条比「别 500」更重要的纪律
============================
响应里**不能只有一个 ok**。这个功能的头号事故形态是「以为生效其实没有」，而它
有三条独立的失败路径，全都不会报错：

* 重载排队了（任务在跑）—— 注册表都没动；
* 注册表换了，但 agent 图还没重建 —— 模型手上仍是旧工具；
* 图重建被调度了但失败了。

所以每个响应都带 ``agent_path_pending`` 和 ``fingerprint_matches``，UI 按它们
显示，而不是按 ``ok``。``fingerprint_matches`` 为 ``None`` 表示**判断不了**
（进程刚起、还没建过工具表）—— 不是 False。这条纪律和
``admin/reload_wiring.py:36-40`` 那句「不知道就报 None，别报 False」是同一条。

为什么路径是 ``/skill-overlay`` 而不是 ``/skills/overlay``
=========================================================
``skills_ext.py:121`` 有 ``GET /skills/{name}`` —— 它会把 ``/skills/overlay``
当成 ``name="overlay"`` 捕获掉，返回 **200 加一个错的 handler**。本仓踩过两次
路由遮蔽，两次都是这个症状（见 ``api/app.py`` 里 conduct 那段注释）。

可以靠「让本模块先注册」绕开，但那是把正确性挂在**注册顺序**上 —— 一个别人可以
随手改、而且改的人不会知道这条依赖的东西。换个不重叠的前缀，这条依赖就不存在了。

PIN 门控
========
启用/停用和重载都过 admin PIN。**没设 PIN 时放行** —— 这是台实验台仪器，把用户
锁在自己的显微镜外面，比 PIN 想防的那件事更糟（同 ``admin_pin.py`` 的威胁模型：
它防的是一只人手，不是模型 —— 模型根本没有通往这个 API 的路）。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from mast.api.schemas_skill_overlay import (
    EjectableListResponse,
    EjectableModule,
    EjectImportSite,
    EjectRequest,
    EjectResponse,
    OverlayDriftInfo,
    OverlayEntryInfo,
    OverlayEntryRequest,
    OverlayEntryResponse,
    OverlayReloadRequest,
    OverlayReloadResponse,
    OverlayRestoreResponse,
    OverlaySkillInfo,
    OverlayStatusResponse,
    SkillPackActionResponse,
    SkillPackAutoEnableRequest,
    SkillPackFetchRequest,
    SkillPackFetchResponse,
    SkillPackInfo,
    SkillPackListResponse,
    SkillPackRemoveRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["skill-overlay"])


def _runtime(request: Request):
    return getattr(getattr(request.app.state, "ctx", None), "live_app", None)


def _registry(request: Request):
    rt = _runtime(request)
    return getattr(rt, "_registry", None) if rt is not None else None


def _check_pin(pin: str) -> tuple[bool, str]:
    """``(放行吗, 拒绝理由)``。没设 PIN = 放行（见模块 docstring）。"""
    try:
        from mast.api.admin_pin import pin_is_set, reason_text, verify_pin

        if not pin_is_set():
            return True, ""
        ok, reason = verify_pin(pin)
        return ok, ("" if ok else reason_text(reason))
    except Exception as exc:  # noqa: BLE001
        # 门读不出来**不能**当成放行 —— 那正好把守卫变成摆设。
        logger.warning("admin PIN 校验失败：%s", exc, exc_info=True)
        return False, "管理员 PIN 校验不可用，本次拒绝（这是保守失败，不是故障）"


def _fingerprints(registry) -> dict:
    """生效探针：注册表侧 vs agent 侧。"""
    out = {"fingerprint_matches": None, "registry_fingerprint": "",
           "wrapped_fingerprint": "", "wrapped_at": 0.0,
           "agent_overlaid": [], "registry_overlaid": []}
    if registry is None:
        return out
    try:
        from mast.agents.instrument_control.tools import (
            LAST_WRAP, expected_wrap_fingerprint,
        )

        exp, exp_ovl = expected_wrap_fingerprint(registry)
        out["registry_fingerprint"] = exp
        out["registry_overlaid"] = exp_ovl
        out["wrapped_fingerprint"] = LAST_WRAP.get("fingerprint") or ""
        out["wrapped_at"] = float(LAST_WRAP.get("at") or 0.0)
        out["agent_overlaid"] = list(LAST_WRAP.get("overlaid") or [])
        if out["wrapped_fingerprint"]:
            out["fingerprint_matches"] = (
                out["wrapped_fingerprint"] == out["registry_fingerprint"])
        # 没建过工具表 ⇒ 保持 None（判断不了），**不是** False
    except Exception as exc:  # noqa: BLE001
        logger.debug("生效探针取值失败：%s", exc)
    return out


@router.get("/skill-overlay", response_model=OverlayStatusResponse)
def overlay_status(request: Request) -> OverlayStatusResponse:
    """覆盖层现状：清单、生效了哪些、以及 **agent 那边跟上了没有**。"""
    try:
        from mast.skills.overlay import loader, manifest as M, paths as P
    except Exception as exc:  # noqa: BLE001
        return OverlayStatusResponse(degraded=True, reason=f"覆盖层不可用：{exc}")

    try:
        man = M.load()
        st = loader.manager().status()
        applied = st.get("applied") or {}
        entries: list[OverlayEntryInfo] = []
        for e in man.entries:
            ok_path, why = P.is_valid_rel(e.path)
            a = applied.get(e.path) or {}
            entries.append(OverlayEntryInfo(
                path=e.path, enabled=e.enabled,
                exists=P.entry_path(e.path).is_file(),
                overlay_of=P.overlay_of(e.path) or "",
                applied=bool(a), skills=list(a.get("names") or []),
                sha256=str(a.get("sha256") or ""),
                valid_path=ok_path, path_error=why,
            ))
        tracked = {e.path for e in man.entries}
        untracked = [f for f in M.discover_files() if f not in tracked]

        registry = _registry(request)
        skills: list[OverlaySkillInfo] = []
        if registry is not None:
            for name in sorted({n for a in applied.values()
                                for n in (a.get("names") or [])}):
                p = registry.provenance(name)
                skills.append(OverlaySkillInfo(
                    name=name, origin=p.origin, module=p.module,
                    source_path=p.source_path, short_sha=p.short_sha,
                    displaced_module=p.displaced_module,
                    signature=p.signature, described=p.describe()))

        last = loader.manager().last_report()
        rt = _runtime(request)
        pending = False
        try:
            pending = bool(rt.pending_agent_rebuild()) if rt is not None else False
        except Exception:  # noqa: BLE001
            pending = False

        return OverlayStatusResponse(
            overlay_dir=str(P.overlay_dir()),
            entries=entries, untracked=untracked, overlaid_skills=skills,
            last_reload=(last.describe() if last else ""),
            baseline_drift=(last.baseline_drift if last else []),
            pending_rebuild=pending,
            **_fingerprints(registry),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("覆盖层状态读取失败：%s", exc, exc_info=True)
        return OverlayStatusResponse(degraded=True, reason=str(exc))


@router.post("/skill-overlay/reload", response_model=OverlayReloadResponse)
def overlay_reload(request: Request,
                   body: OverlayReloadRequest) -> OverlayReloadResponse:
    """重新加载覆盖层，并把三条消费者链都推一遍。"""
    ok_pin, why = _check_pin(body.pin)
    if not ok_pin:
        return OverlayReloadResponse(ok=False, reason=why)

    rt = _runtime(request)
    if rt is None or not hasattr(rt, "reload_overlay_skills"):
        return OverlayReloadResponse(
            degraded=True,
            reason="没有活的运行时（独立 API 模式）—— 覆盖层要在主服务里重载。")
    try:
        res = rt.reload_overlay_skills(reason=body.reason or "ui")
    except Exception as exc:  # noqa: BLE001
        logger.warning("覆盖层重载失败：%s", exc, exc_info=True)
        return OverlayReloadResponse(degraded=True, reason=str(exc))

    refresh = res.get("refresh") or {}
    fp = _fingerprints(_registry(request))
    return OverlayReloadResponse(
        ok=not res.get("failed"),
        status=str(res.get("status") or ""),
        summary=str(res.get("summary") or ""),
        applied=list(res.get("applied") or []),
        failed=list(res.get("failed") or []),
        restored=list(res.get("restored") or []),
        baseline_drift=list(res.get("baseline_drift") or []),
        # 排队时 agent 侧必然没跟上 —— 明说，别让 ok=True 盖过去。
        agent_path_pending=(True if refresh.get("queued")
                            else refresh.get("agent_path_pending")),
        refresh=str(refresh.get("described") or
                    ("已排队：当前任务结束后自动生效" if refresh.get("queued") else "")),
        fingerprint_matches=fp["fingerprint_matches"],
    )


@router.post("/skill-overlay/entry", response_model=OverlayEntryResponse)
def overlay_set_entry(request: Request,
                      body: OverlayEntryRequest) -> OverlayEntryResponse:
    """启用/停用一个条目（只写清单，**不**自动重载）。

    刻意分成两步：写清单是一次编辑，让它生效是一次**决定**。合成一步的话，
    用户勾一个复选框就换掉了正在跑的仪器上的技能。
    """
    ok_pin, why = _check_pin(body.pin)
    if not ok_pin:
        return OverlayEntryResponse(ok=False, reason=why)
    try:
        from mast.skills.overlay import manifest as M, paths as P
    except Exception as exc:  # noqa: BLE001
        return OverlayEntryResponse(degraded=True, reason=f"覆盖层不可用：{exc}")

    ok_path, path_why = P.is_valid_rel(body.path)
    if not ok_path:
        return OverlayEntryResponse(ok=False, reason=path_why)
    try:
        man = M.load()
        if man.unreadable:
            return OverlayEntryResponse(
                ok=False,
                reason=f"清单读不出来（{man.unreadable}）—— 先修好文件，"
                       "否则这次写入会把已有条目全抹掉。")
        rel = P.normalise_rel(body.path)
        man.upsert(M.Entry(path=rel, enabled=body.enabled,
                           allow_removals=list(body.allow_removals),
                           note=body.note))
        M.save(man)
        e = man.get(rel)
        return OverlayEntryResponse(ok=True, entry=OverlayEntryInfo(
            path=rel, enabled=bool(e.enabled),
            exists=P.entry_path(rel).is_file(),
            overlay_of=P.overlay_of(rel) or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("覆盖层条目写入失败：%s", exc, exc_info=True)
        return OverlayEntryResponse(degraded=True, reason=str(exc))


@router.post("/skill-overlay/restore-all", response_model=OverlayRestoreResponse)
def overlay_restore_all(request: Request,
                        body: OverlayReloadRequest) -> OverlayRestoreResponse:
    """急救按钮：不管位移表乱成什么样，硬恢复到启动时的基线。

    存在的理由是「回滚本身出了问题」时还有一条出路 —— 一个只能靠自己内部状态
    恢复的系统，在那个状态坏掉时就没救了。
    """
    ok_pin, why = _check_pin(body.pin)
    if not ok_pin:
        return OverlayRestoreResponse(ok=False, reason=why)

    registry = _registry(request)
    if registry is None:
        return OverlayRestoreResponse(degraded=True, reason="没有活的技能注册表")
    try:
        from mast.admin import reload_wiring as rw
        from mast.skills.overlay import loader

        restored = loader.manager().restore_all(registry)
        outcome = rw.refresh_after_skill_change("overlay restore-all")
        return OverlayRestoreResponse(
            ok=True, restored=restored,
            summary=(f"已恢复 {len(restored)} 个技能到启动基线。"
                     + outcome.describe()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("覆盖层恢复失败：%s", exc, exc_info=True)
        return OverlayRestoreResponse(degraded=True, reason=str(exc))


# ---------------------------------------------------------------------------
# 导出到覆盖层
# ---------------------------------------------------------------------------

@router.get("/skill-overlay/ejectable", response_model=EjectableListResponse)
def overlay_ejectable(request: Request,
                      q: str = "") -> EjectableListResponse:
    """哪些内置技能模块可以导出。``q`` 按模块名或技能类名过滤。

    没有 ``q`` 时**不做类分析**（170 个模块逐个递归解析基类要好几秒，而列表页
    只需要模块名）。给了 ``q`` 才对**筛剩下的**那些做分析 —— 这样搜 "SetBias"
    也能找到 ``builtins/bias.py``，代价只在真的搜的时候付。
    """
    del request
    try:
        from mast.pyexec import _srcfiles
        from mast.skills.overlay import eject as E

        root = _srcfiles.source_root()
        mods = E.list_ejectable(source_root=root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("列举可导出模块失败：%s", exc, exc_info=True)
        return EjectableListResponse(degraded=True, reason=str(exc))

    needle = (q or "").strip().lower()
    out: list[EjectableModule] = []
    for d in mods:
        name_hit = not needle or needle in d["dotted"].lower()
        classes: list[str] = []
        undec: list[str] = []
        if needle:
            # 搜的时候才分析 —— 搜 "SetBias" 要能找到它所在的模块
            try:
                src = _srcfiles.resolve(d["dotted"])
                if src is not None:
                    sk, unk = E._skill_classes(src.read_bytes(), d["dotted"],
                                               source_root=root)
                    classes, undec = sorted(sk), sorted(unk)
            except Exception:  # noqa: BLE001
                pass
            if not name_hit and not any(needle in c.lower() for c in classes):
                continue
        out.append(EjectableModule(dotted=d["dotted"], rel=d["rel"],
                                   n_bytes=d["n_bytes"], already=d["already"],
                                   skill_classes=classes, undecidable=undec))
    return EjectableListResponse(source_root=str(root), modules=out)


@router.post("/skill-overlay/eject", response_model=EjectResponse)
def overlay_eject(request: Request, body: EjectRequest) -> EjectResponse:
    """把一个内置模块的源码字节取出来放进覆盖层。**不启用它。**

    过 admin PIN：这一步会往数据目录里写文件，和启用/重载同一道门。
    """
    del request
    ok, why = _check_pin(body.pin)
    if not ok:
        return EjectResponse(reason=why, dotted=body.dotted)
    try:
        from mast.skills.overlay import eject as E

        r = E.eject(body.dotted, overwrite=bool(body.overwrite))
    except Exception as exc:  # noqa: BLE001
        logger.warning("导出 %s 失败：%s", body.dotted, exc, exc_info=True)
        return EjectResponse(degraded=True, reason=str(exc), dotted=body.dotted)
    return EjectResponse(
        ok=r.ok, reason=r.reason, dotted=r.dotted, rel=r.rel, path=r.path,
        sha256=r.sha256, n_bytes=r.n_bytes, warning=r.warning,
        skill_classes=list(r.skill_classes), undecidable=list(r.undecidable),
        n_skill_binds=r.n_skill_binds,
        importers=[EjectImportSite(**s.as_dict()) for s in r.importers])


@router.get("/skill-overlay/drift", response_model=list[OverlayDriftInfo])
def overlay_drift(request: Request) -> list[OverlayDriftInfo]:
    """每个覆盖层条目基于的内置版，之后变过没有。

    永不抛：读不出来就报 ``drifted=None``（判断不了），不报 ``False``。
    一个「全都没漂」的空回答，正好在升级之后最该提醒的那一刻什么都不说。
    """
    del request
    try:
        from mast.skills.overlay import eject as E, manifest as M

        rels = sorted({e.path for e in M.load().entries} | set(M.discover_files()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("列举覆盖层条目失败：%s", exc, exc_info=True)
        return []
    out: list[OverlayDriftInfo] = []
    for rel in rels:
        try:
            d = E.drift(rel)
        except Exception as exc:  # noqa: BLE001
            out.append(OverlayDriftInfo(rel=rel, known=False, drifted=None,
                                        reason="复核失败：" + str(exc)))
            continue
        out.append(OverlayDriftInfo(**d.as_dict()))
    return out


# ---------------------------------------------------------------------------
# 签名技能包
# ---------------------------------------------------------------------------

@router.get("/skill-overlay/packs", response_model=SkillPackListResponse)
def overlay_packs(request: Request) -> SkillPackListResponse:
    """本机装了哪些签名包，以及它们**现在**还验不验得过。"""
    del request
    try:
        from mast.skills.overlay import manifest as M
        from mast.update import skillpack_client as SC

        rows = SC.installed_packs()
        man = M.load()
    except Exception as exc:  # noqa: BLE001
        logger.warning("列举技能包失败：%s", exc, exc_info=True)
        return SkillPackListResponse(degraded=True, reason=str(exc))
    return SkillPackListResponse(
        packs=[SkillPackInfo(**r) for r in rows],
        # 清单读不出来时**不要**报 True —— 那会让界面显示「自动启用：开」，
        # 而实际上装包那一步会因为读不出清单而拒绝启用。
        auto_enable=bool(man.auto_enable_packs) and not man.unreadable,
        reason=man.unreadable)


@router.post("/skill-overlay/packs/fetch", response_model=SkillPackFetchResponse)
def overlay_pack_fetch(request: Request,
                       body: SkillPackFetchRequest) -> SkillPackFetchResponse:
    """从推送服务器拉一个**已签名**的技能包并装上。

    这条路的后端（``skillpack_client.fetch_and_install``）2026-08-20 就写好了，但
    **一直没有 HTTP 入口** —— 全仓零调用方，界面上只能看已装的包，装新包得有人
    到机器前面跑脚本。这个端点补的就是那一段。

    信任链一个字都不放松：验签 → 逐文件 sha256 → 落盘后每次加载再复核，全在
    ``fetch_and_install`` 里，这里一行都不重实现。没有烘焙公钥或没装
    ``cryptography`` 时它自己 fail-closed（一个包都不装），那正是我们要的。

    **装 ≠ 生效**：默认不重载（``reload_now=False``），响应里 ``needs_reload``
    说清楚还差一步。要顺手重载就带 ``reload_now``，那时响应会带三态生效字段。
    """
    ok_pin, why = _check_pin(body.pin)
    if not ok_pin:
        return SkillPackFetchResponse(reason=why)

    pack_id = str(body.pack_id or "").strip()
    if not pack_id:
        return SkillPackFetchResponse(reason="要拉哪个包？给一个 pack_id。")

    try:
        from mast._runtime_paths import project_root
        from mast.update.client import _read_token, read_server_url
        from mast.update import skillpack_client as SC

        root = project_root()
        url = read_server_url(root)
        token = _read_token(root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取推送服务器配置失败：%s", exc, exc_info=True)
        return SkillPackFetchResponse(degraded=True, reason=str(exc))
    if not url or not token:
        return SkillPackFetchResponse(
            reason="未配置推送服务器（server_url / token）—— 在启动器的推送设置里配置后重试")

    try:
        res = SC.fetch_and_install(url, token, pack_id)
    except Exception as exc:  # noqa: BLE001 — fetch_and_install 声称永不抛，兜一层
        logger.warning("拉取技能包 %s 失败：%s", pack_id, exc, exc_info=True)
        return SkillPackFetchResponse(degraded=True, reason=str(exc), pack_id=pack_id)

    out = SkillPackFetchResponse(
        ok=bool(res.ok),
        reason="" if res.ok else "；".join(res.reasons or ("原因不明",)),
        summary=res.describe(),
        needs_reload=bool(res.needs_reload),
        pack_id=res.pack_id or pack_id,
        version=res.version,
        installed=list(res.installed),
        enabled=list(res.enabled),
        shadowed=list(res.shadowed),
        replaced_version=res.replaced_version,
    )
    if not (res.ok and body.reload_now):
        return out

    rt = _runtime(request)
    if rt is None or not hasattr(rt, "reload_overlay_skills"):
        out.summary += "（没有活的运行时 —— 包已装上，重启后生效）"
        out.agent_path_pending = True
        return out
    try:
        rl = rt.reload_overlay_skills(reason=f"pack fetch {pack_id}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("拉包后重载失败：%s", exc, exc_info=True)
        out.summary += f"（包已装上，但重载失败：{exc}）"
        out.agent_path_pending = True
        return out
    refresh = rl.get("refresh") or {}
    out.reloaded = True
    out.needs_reload = False
    out.summary += "；" + str(rl.get("summary") or "已重载")
    out.agent_path_pending = (True if refresh.get("queued")
                              else refresh.get("agent_path_pending"))
    out.fingerprint_matches = _fingerprints(_registry(request))["fingerprint_matches"]
    return out


@router.post("/skill-overlay/packs/remove", response_model=SkillPackActionResponse)
def overlay_pack_remove(request: Request,
                        body: SkillPackRemoveRequest) -> SkillPackActionResponse:
    """删掉一个签名包，并把它在清单里的条目一起摘掉。"""
    del request
    ok, why = _check_pin(body.pin)
    if not ok:
        return SkillPackActionResponse(reason=why)
    try:
        from mast.update import skillpack_client as SC

        done, msg = SC.remove_pack(body.pack_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("移除技能包 %s 失败：%s", body.pack_id, exc, exc_info=True)
        return SkillPackActionResponse(degraded=True, reason=str(exc))
    return SkillPackActionResponse(ok=done, reason="" if done else msg,
                                   summary=msg if done else "",
                                   needs_reload=done)


@router.post("/skill-overlay/packs/auto-enable",
             response_model=SkillPackActionResponse)
def overlay_pack_auto_enable(
        request: Request,
        body: SkillPackAutoEnableRequest) -> SkillPackActionResponse:
    """开/关「签名包装进来就自动启用」。

    关掉是给正在跑长实验的机器用的 —— 包照装，什么时候换由用户定。
    """
    del request
    ok, why = _check_pin(body.pin)
    if not ok:
        return SkillPackActionResponse(reason=why)
    try:
        from mast.skills.overlay import manifest as M

        man = M.load()
        if man.unreadable:
            # 「读不到」当成「空的」写回去，会抹掉用户已启用的全部条目。
            return SkillPackActionResponse(
                reason=f"覆盖层清单读不出来（{man.unreadable}）—— 本次不写入。")
        man.auto_enable_packs = bool(body.enabled)
        M.save(man)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写 auto_enable_packs 失败：%s", exc, exc_info=True)
        return SkillPackActionResponse(degraded=True, reason=str(exc))
    return SkillPackActionResponse(
        ok=True,
        summary=("签名包装进来会自动启用（仍需一次重载才生效）"
                 if body.enabled else
                 "签名包装进来「不会」自动启用 —— 要手工在上面的条目表里启用"))
