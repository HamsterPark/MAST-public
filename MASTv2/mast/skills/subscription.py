"""技能订阅列表 —— 用户自己的那张装载面清单。

为什么要有这一层
================
``hardware_modules`` 开头那段论证在这里同样成立，而且更尖锐：**工具表是模型每一
回合都要读一遍的菜单**。instrument_control 现在挂着几百个技能，其中绝大多数在任何
一台具体机器上、任何一个具体课题里都用不到。菜单越长，真正该用的那几个越难被选中。

硬件模块门回答的是「这台机器有没有这个硬件」，高级能力门回答的是「这个动作能不能
绕过一层保护」。两个都不回答第三个问题：**这位用户这段时间在做什么**。订阅列表
回答的就是它。

订阅**不是**安全机制
====================
这一条要一直大声说，因为把安全逻辑往这道门上挂是最自然、也最错的下一步。

* 未订阅的技能**仍在 SkillRegistry 里**。手动执行（``POST /api/skills/{name}/
  execute``）、composite 的子步、conduct 的 ``ctx.run`` 三条路都经
  :meth:`mast.core.execution_context.ExecutionContext.run` 收口，而那里**没有**
  名单式判断 —— 它们照常能调到未订阅的技能，这是设计，不是漏洞。
* 真正的安全地板在更下面两层：``execution_context._ABORT_SAFE_WRITES``（Nanonis
  verb 级）与 watchdog 的紧急端口 SafeRetract。订阅门被绕过、被关掉、被导入一份
  乱七八糟的清单，那两层一个字都不会松。
* 2026-08-20 conduct 层的拍板是「工具面全开 + 服务端包络裁决」：把关靠**包络**，
  不靠「不给工具」。订阅列表服从同一条哲学 —— 它整理的是**便利**，裁决的是别人。

所以这个模块的失败方向与 ``hardware_modules`` **相反**：那边读不懂就 fail-closed
（回默认全关），因为一个不存在的硬件的技能只会失败；这边读不懂就 **fail-open 回全
订阅**，因为「订阅文件坏了 ⇒ agent 一夜之间失去全部技能」是个真事故，而它换来的
所谓保护并不存在。读不出来会被记成 ``unreadable`` 并一路报到界面上 —— fail-open
不等于装作没事（见 ``overlay/manifest.py`` 的同一条纪律）。

absent = 全订阅
===============
``customised=False``（文件不存在，或从没定制过）时 :func:`unloaded_skill_names`
恒返回空集 ⇒ 这道门是恒真门 ⇒ 工具面与这个功能不存在时**逐位相同**。上线当天零
行为变化，这是可验证的（测试比对 wrap 指纹），不是一句承诺。

第一次显式定制会 **materialise**：把**当时**的市场全集写进 ``entries``、置
``customised=True``。此后新出现的技能（discover / overlay / 桥接 / agent 自建）
不在 entries 里 ⇒ **只进市场，不进订阅** —— 「新技能默认不自动装载」这条需求
不是靠一个开关实现的，是 materialise 的副产品。

谁对账 entries
==============
entries 里的名字**不剔**未知项（与 ``hardware_modules.set_enabled`` 丢 unknown 的
做法相反，理由：技能集合是动态的 —— overlay 卸载、pack 移除、custom 停用都会让一个
名字暂时消失，替用户把它从清单里删掉等于替他改单，而他下次装回来就会发现自己的
订阅被人动过）。差集计算时它们天然无效果；:func:`state` 的 ``missing_entries``
把它们诚实地列出来，界面上是一个徽标 —— **那就是这张名单的对账者**。

接线（读端四处，共用同一个函数）
================================
``build_instrument_skill_tools`` / ``expected_wrap_fingerprint``（两者都在
``agents/instrument_control/tools.py``）/ ``webui.agents_api._compute_agent_tools``
/ 市场与 palette 路由。四处调同一个 :func:`unloaded_skill_names`，不各写一遍。
万一还是漏了一处：wrap 指纹探针覆盖全集，漏改会立刻表现为
``fingerprint_matches=False``，不会静默。

写端只有一条路：``api/routes/skill_market.py::_apply_subscription_change``，它写完
必调 ``reload_wiring.refresh_after_skill_change`` —— **持久化 ≠ 生效**，工具表在
建图时冻结（同 ``hardware_modules`` 那段教训）。
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: 落盘文件名（在 ``<project_root>/config/`` 下）。``config/`` 在 OTA 的
#: ``DEFAULT_EXCLUDE_TOP`` 里，所以升级永远不会覆盖用户的订阅。
STORE_FILENAME = "skill_subscription.json"

#: pending 推荐与 audit 的上限（环形，丢最旧）。一个跑飞的 agent 不该把这个文件
#: 刷成几十兆 —— 同 wishlist 的 MAX_ITEMS 理由。
MAX_PENDING = 50
MAX_AUDIT = 500

# ─────────────────────────────────────────────────────────────────────────────
# 必装豁免集
# ─────────────────────────────────────────────────────────────────────────────
#: 无论订阅列表怎么写，工具面上永远留着的动词。
#:
#: **这不是安全机制**（安全在 ``_ABORT_SAFE_WRITES`` 和 watchdog，比技能层低两层），
#: 而是**工具面可操作性保证**：不管用户怎么删，模型手上永远有「停下来 / 退针」
#: 可用。判据是「abort 之后仍然允许的那一族动词」，不是「危险」——所以这里没有
#: SetBias，也没有任何采集类的 Stop*（停的是采集，不是运动）。
#:
#: 它**只豁免订阅门，不豁免硬件/高级能力门**：给一台没有的硬件保留工具位，正是
#: ``hardware_modules`` 开头反对的事。``StopNanonisScript`` 同时受高级能力门管，
#: 那道门关着它就仍然不出现 —— 这是对的。
#:
#: 谁对账：``test_skill_subscription.py`` 断言每个名字都在 discover 后的注册表里
#: （拼错 ⇒ 豁免了空气 ⇒ 当场红），界面上是一枚「必装」徽标（用户侧对账点）。
#: 名单只住这一处，前端与导入端都从 API 的 ``mandatory`` 字段读，不复制。
MANDATORY_SKILLS: frozenset[str] = frozenset({
    "SafeRetract",         # 急停退针本体
    "WithdrawTip",         # 常规退针
    "StopScan",            # 停扫
    "StopMotor",           # 停粗动
    "StopAutoApproach",    # 停自动进针
    "StopNanonisScript",   # 停脚本（abort 表里最重要的一条 Script_Stop）
})

#: audit 里 ``via`` 的取值不是闭集（``recommendation:<id>`` 带 id），但这几个是
#: 固定词，写成常量免得两处拼得不一样。
VIA_UI = "ui"
VIA_IMPORT = "import"
VIA_RESET = "reset"
VIA_MATERIALISE = "materialise"

REC_PENDING = "pending"
REC_ACCEPTED = "accepted"
REC_REJECTED = "rejected"


# ─────────────────────────────────────────────────────────────────────────────
# 状态
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _State:
    """一份不可变快照。读者拿到的永远是完整的一份，不会看到写到一半的。"""

    customised: bool = False
    entries: frozenset[str] = frozenset()
    pending: tuple[dict, ...] = ()
    audit: tuple[dict, ...] = ()
    #: 读不出来时的原因（``""`` = 读得出来）。**与「文件不存在」不是一回事**：
    #: 不存在 = 还没人定制过（正常）；读不出来 = 有人配了但我们没看懂（要说）。
    unreadable: str = ""

    def to_doc(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "customised": bool(self.customised),
            "entries": sorted(self.entries),
            "pending": [dict(p) for p in self.pending],
            "audit": [dict(a) for a in self.audit],
        }


_lock = threading.Lock()
_state: _State | None = None          # None = 还没从盘上读过
_store_path_cache: Path | None = None


def store_path() -> Path:
    """订阅文件的位置。**惰性**解析。

    绝不写成模块级常量、也绝不用私有的 ``parents[N]`` 走法 —— 那正是 wishlist 那次
    「每一个自以为重定向了的测试其实都在写用户的真文件」的成因
    （``tests/v2/conftest.py::_real_wishlist_is_read_only`` 记着那 25 行垃圾）。
    """
    global _store_path_cache
    if _store_path_cache is None:
        from mast._runtime_paths import project_root
        _store_path_cache = project_root() / "config" / STORE_FILENAME
    return _store_path_cache


def reset_default_store() -> None:
    """丢掉路径与状态缓存（改了 ``MAST2_PROJECT_ROOT`` 之后必须调）。"""
    global _store_path_cache, _state
    with _lock:
        _store_path_cache = None
        _state = None


#: 测试里想要一个干净 holder 时用；生产路径不调。
reset_for_tests = reset_default_store


def _parse(raw) -> _State:
    """把盘上的 JSON 变成状态。看不懂的部分丢掉，看得懂的留下。"""
    if not isinstance(raw, dict):
        raise ValueError(f"顶层不是对象而是 {type(raw).__name__}")
    entries = raw.get("entries")
    if entries is None:
        entries = []
    if not isinstance(entries, (list, tuple, set, frozenset)):
        raise ValueError(f"entries 不是列表而是 {type(entries).__name__}")
    names = frozenset(str(x) for x in entries if isinstance(x, str) and str(x).strip())

    pending: list[dict] = []
    for it in (raw.get("pending") or []):
        if isinstance(it, dict) and it.get("skill"):
            pending.append({
                "id": str(it.get("id") or _new_rec_id()),
                "skill": str(it.get("skill")),
                "by_agent": str(it.get("by_agent") or ""),
                "reason": str(it.get("reason") or ""),
                "conversation_id": str(it.get("conversation_id") or ""),
                "at": str(it.get("at") or ""),
                "status": str(it.get("status") or REC_PENDING),
                "resolved_at": it.get("resolved_at") or None,
            })
    audit: list[dict] = []
    for it in (raw.get("audit") or []):
        if isinstance(it, dict):
            audit.append({
                "at": str(it.get("at") or ""),
                "action": str(it.get("action") or ""),
                "skills": [str(x) for x in (it.get("skills") or [])],
                "via": str(it.get("via") or ""),
            })
    return _State(
        customised=bool(raw.get("customised")),
        entries=names,
        pending=tuple(pending[-MAX_PENDING:]),
        audit=tuple(audit[-MAX_AUDIT:]),
    )


def _load_locked() -> _State:
    """从盘上读一份。**调用者必须已持 _lock。**"""
    p = store_path()
    if not p.is_file():
        return _State()          # 不存在 = 还没定制过（正常，零回归默认）
    try:
        return _parse(json.loads(p.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        # fail-open：回到「全订阅」，而不是「全退订」。理由见模块 docstring ——
        # 订阅不是保护，往关的方向兜没有任何好处，只会让 agent 一夜之间失能。
        logger.error(
            "订阅文件读不出来（%s）：%s —— 本次按**全订阅**处理（不是空订阅），"
            "界面会显示这条原因；修好文件后重启或重载即可", p, exc)
        return _State(unreadable=f"{type(exc).__name__}: {exc}")


def _ensure() -> _State:
    global _state
    st = _state
    if st is not None:
        return st
    with _lock:
        if _state is None:
            _state = _load_locked()
        return _state


def _save_locked(st: _State) -> None:
    """原子落盘。**调用者必须已持 _lock。**

    复用覆盖层那份 ``atomic_write_json``（``.part`` + ``os.replace``）——
    半个文件是本仓反复被咬的形状，两处各写一遍迟早只有一处是对的。
    """
    try:
        from mast.skills.overlay.manifest import atomic_write_json
        atomic_write_json(store_path(), st.to_doc())
        return
    except Exception as exc:  # noqa: BLE001 — 覆盖层不可用时不能连订阅都保存不了
        logger.debug("复用 overlay.atomic_write_json 失败（%s），用本地等价实现", exc)
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(json.dumps(st.to_doc(), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, p)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _new_rec_id() -> str:
    return "rec-" + secrets.token_hex(3)


def _with_audit(st: _State, action: str, skills: Iterable[str], via: str) -> _State:
    entry = {"at": _now(), "action": action,
             "skills": sorted({str(s) for s in skills}), "via": str(via or "")}
    return replace(st, audit=(st.audit + (entry,))[-MAX_AUDIT:])


def _commit(st: _State) -> _State:
    """装上新状态并落盘。**调用者必须已持 _lock。**"""
    global _state
    _save_locked(st)
    _state = st
    return st


# ─────────────────────────────────────────────────────────────────────────────
# 读
# ─────────────────────────────────────────────────────────────────────────────

def is_customised() -> bool:
    """用户显式定制过吗？``False`` = 全订阅（出厂态）。"""
    return _ensure().customised


def unreadable_reason() -> str:
    """订阅文件读不出来的原因；``""`` = 读得出来（含「文件不存在」）。"""
    return _ensure().unreadable


def subscribed_names() -> frozenset[str] | None:
    """显式订阅的名字集；``None`` = 未定制（= 全订阅，不是空集）。

    返回 ``None`` 而不是「全集」是刻意的：这个模块**不知道**全集是什么（它不持有
    registry），而把「未定制」和「订阅了恰好这些」混成同一个返回值，正是
    ``unknown_is_not_an_answer`` 那一族事故的形状。
    """
    st = _ensure()
    return st.entries if st.customised else None


def unloaded_skill_names(all_names: Iterable[str] | None = None) -> frozenset[str]:
    """**不该出现在 agent 工具面上**的技能名。

    这是四个读端共用的唯一一个函数（IC 装配 / 生效探针 / UI 镜像 / palette）。

    * 未定制 ⇒ 空集（恒真门，零回归）；
    * 已定制 ⇒ ``全集 − entries − MANDATORY_SKILLS``；
    * ``all_names`` 给不出来 ⇒ 空集 + 一行 debug（fail-open：这个模块宁可少滤，
      也不肯在不知道全集的情况下瞎滤）。

    **每次现算，不缓存补集**：全集会随 overlay 重载/热注册变化，缓存一份补集就等于
    多一个「新技能注册了但门没跟上」的说谎窗口，而这个差集只有几百个元素。
    """
    st = _ensure()
    if not st.customised:
        return frozenset()
    if all_names is None:
        logger.debug("unloaded_skill_names: 没有全集可比，按不过滤处理")
        return frozenset()
    universe = {str(n) for n in all_names}
    return frozenset(universe - st.entries - MANDATORY_SKILLS)


def is_subscribed(name: str) -> bool:
    """单个技能在不在装载面上（必装项恒为 True）。"""
    st = _ensure()
    if not st.customised:
        return True
    return name in st.entries or name in MANDATORY_SKILLS


def pending_recommendations() -> list[dict]:
    """还没被裁决的推荐。"""
    return [dict(p) for p in _ensure().pending if p.get("status") == REC_PENDING]


def resolved_recommendations(limit: int = 20) -> list[dict]:
    """已裁决的尾巴（留痕可见 —— 拒绝也是一条记录，不是删除）。"""
    out = [dict(p) for p in _ensure().pending if p.get("status") != REC_PENDING]
    return out[-limit:]


def audit_tail(limit: int = 20) -> list[dict]:
    return [dict(a) for a in _ensure().audit[-limit:]]


def state(all_names: Iterable[str] | None = None) -> dict:
    """界面视图。``all_names`` 给了才能算市场总数与失联条目。"""
    st = _ensure()
    universe = {str(n) for n in all_names} if all_names is not None else None
    missing: list[str] = []
    if universe is not None and st.customised:
        missing = sorted(st.entries - universe)
    if universe is None:
        subscribed_count = len(st.entries) if st.customised else 0
    elif st.customised:
        subscribed_count = len((st.entries & universe) | (MANDATORY_SKILLS & universe))
    else:
        subscribed_count = len(universe)
    return {
        "customised": st.customised,
        "subscribed_count": subscribed_count,
        "market_total": len(universe) if universe is not None else 0,
        "mandatory": sorted(MANDATORY_SKILLS),
        "missing_entries": missing,
        "unreadable": st.unreadable,
        "pending_count": len(pending_recommendations()),
        "store_path": str(store_path()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 写
# ─────────────────────────────────────────────────────────────────────────────

def _materialised(st: _State, all_names: Iterable[str], via: str) -> _State:
    """未定制 → 定制：把**当时的**市场全集写进 entries。

    这一步是「新技能默认只进市场不进订阅」的全部实现：此后新注册的名字不在
    entries 里，差集自然把它们排除掉。
    """
    if st.customised:
        return st
    st = replace(st, customised=True,
                 entries=frozenset(str(n) for n in all_names))
    return _with_audit(st, VIA_MATERIALISE, (), via or VIA_UI)


def set_subscribed(names: Iterable[str], *, via: str = VIA_UI) -> dict:
    """整体写（导入 replace 用）。不经过 materialise —— 它自己就是一次定制。"""
    wanted = frozenset(str(n) for n in names)
    with _lock:
        st = _state if _state is not None else _load_locked()
        st = replace(st, customised=True, entries=wanted)
        st = _with_audit(st, "set", wanted, via)
        _commit(st)
    return {"ok": True, "customised": True, "subscribed": sorted(wanted)}


def subscribe(names: Iterable[str], *, all_names: Iterable[str] | None = None,
              via: str = VIA_UI) -> dict:
    """把这些名字加进订阅面。"""
    return _mutate(names, add=True, all_names=all_names, via=via)


def unsubscribe(names: Iterable[str], *, all_names: Iterable[str] | None = None,
                via: str = VIA_UI) -> dict:
    """把这些名字从订阅面拿掉（必装项跳过并如实报告）。"""
    return _mutate(names, add=False, all_names=all_names, via=via)


def _mutate(names: Iterable[str], *, add: bool,
            all_names: Iterable[str] | None, via: str) -> dict:
    wanted = [str(n) for n in names if str(n).strip()]
    skipped_mandatory = sorted({n for n in wanted if n in MANDATORY_SKILLS}) if not add else []
    effective = [n for n in wanted if add or n not in MANDATORY_SKILLS]
    with _lock:
        st = _state if _state is not None else _load_locked()
        materialised = False
        if not st.customised:
            if all_names is None:
                # 不知道全集就 materialise，等于把没列出来的全部退订 —— 宁可拒绝。
                return {"ok": False, "changed": [],
                        "skipped_mandatory": skipped_mandatory,
                        "reason": "首次定制需要知道市场全集（内部错误：调用方没传 all_names）"}
            st = _materialised(st, all_names, via)
            materialised = True
        before = st.entries
        new = (before | set(effective)) if add else (before - set(effective))
        changed = sorted(new ^ before)
        if changed:
            st = replace(st, entries=new)
            st = _with_audit(st, "subscribe" if add else "unsubscribe", changed, via)
        if changed or materialised:
            # materialise 本身就是一次改动 —— 即使这次增删没净变化也要落盘，
            # 否则下一次进程重启又回到「未定制」，而 audit 里已经写了 materialise。
            _commit(st)
        cust = st.customised
        count = len(st.entries)
    return {"ok": True, "customised": cust, "changed": changed,
            "materialised": materialised,
            "skipped_mandatory": skipped_mandatory, "subscribed_count": count}


def reset_to_default(*, via: str = VIA_RESET) -> dict:
    """回到出厂态：全订阅。``customised=False``，entries 清空。"""
    with _lock:
        st = _state if _state is not None else _load_locked()
        st = replace(st, customised=False, entries=frozenset())
        st = _with_audit(st, "reset", (), via)
        _commit(st)
    return {"ok": True, "customised": False}


# ─────────────────────────────────────────────────────────────────────────────
# 推荐（agent 只能写到这里，写不进订阅面）
# ─────────────────────────────────────────────────────────────────────────────

def add_recommendation(skill: str, *, by_agent: str = "", reason: str = "",
                       conversation_id: str = "") -> dict:
    """记一条待确认的推荐。同一技能已有未决推荐时**幂等**返回既有那条。

    agent 能到达的写路径只有这一条 —— 它不能改自己的工具面，那是人面上的动作。
    """
    skill = str(skill or "").strip()
    if not skill:
        return {"ok": False, "reason": "技能名为空"}
    with _lock:
        st = _state if _state is not None else _load_locked()
        for p in st.pending:
            if p.get("skill") == skill and p.get("status") == REC_PENDING:
                return {"ok": True, "duplicate": True, "recommendation": dict(p)}
        rec = {"id": _new_rec_id(), "skill": skill, "by_agent": str(by_agent or ""),
               "reason": str(reason or ""), "conversation_id": str(conversation_id or ""),
               "at": _now(), "status": REC_PENDING, "resolved_at": None}
        st = replace(st, pending=(st.pending + (rec,))[-MAX_PENDING:])
        _commit(st)
    return {"ok": True, "duplicate": False, "recommendation": dict(rec)}


def resolve_recommendation(rec_id: str, accept: bool, *,
                           all_names: Iterable[str] | None = None) -> dict:
    """裁决一条推荐。接受 ⇒ 顺带订阅；拒绝 ⇒ 只留痕，什么都不改。

    **未定制态接受不 materialise**：默认已经全订阅，这条推荐的技能本就在面上；
    这时候把用户悄悄转成明确名单，audit 里看起来会像是他主动定制过。
    """
    rec_id = str(rec_id or "")
    with _lock:
        st = _state if _state is not None else _load_locked()
        found = None
        for p in st.pending:
            if p.get("id") == rec_id:
                found = p
                break
        if found is None:
            return {"ok": False, "reason": f"没有这条推荐：{rec_id}"}
        if found.get("status") != REC_PENDING:
            return {"ok": False, "reason": f"这条推荐已经是 {found.get('status')}",
                    "recommendation": dict(found)}
        skill = str(found.get("skill") or "")
        new_status = REC_ACCEPTED if accept else REC_REJECTED
        updated = {**found, "status": new_status, "resolved_at": _now()}
        st = replace(st, pending=tuple(updated if p.get("id") == rec_id else p
                                       for p in st.pending))
        st = _with_audit(st, new_status, [skill], f"recommendation:{rec_id}")
        already = not st.customised
        if accept and st.customised:
            st = replace(st, entries=st.entries | {skill})
        _commit(st)
    return {"ok": True, "recommendation": updated, "skill": skill,
            "already_subscribed_by_default": bool(accept and already)}


__all__ = [
    "SCHEMA_VERSION", "STORE_FILENAME", "MANDATORY_SKILLS",
    "MAX_PENDING", "MAX_AUDIT",
    "VIA_UI", "VIA_IMPORT", "VIA_RESET", "VIA_MATERIALISE",
    "REC_PENDING", "REC_ACCEPTED", "REC_REJECTED",
    "store_path", "reset_default_store", "reset_for_tests",
    "is_customised", "unreadable_reason", "subscribed_names",
    "unloaded_skill_names", "is_subscribed", "state",
    "pending_recommendations", "resolved_recommendations", "audit_tail",
    "set_subscribed", "subscribe", "unsubscribe", "reset_to_default",
    "add_recommendation", "resolve_recommendation",
]
