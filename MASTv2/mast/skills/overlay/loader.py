"""覆盖层的加载与热重载 —— 声明式，不是增量式。

``reload_skills()`` 每次都从盘上重新算出「现在应该是什么样」，再把注册表推到那个
状态。增量式（记住上次做了什么、这次只做差量）会在任何一次异常退出之后留下一个
谁也说不清的中间态。

两阶段
======
**阶段 A 全部先解析、后应用**：每个文件各自 ``ast.parse`` → 在临时模块里
``exec`` → 收集 ``BaseSkill`` 子类 → 算三集合 → 跑「只能收紧」比对。
任何一个文件失败，**只跳过它自己**并记账，其余继续（同
``custom_loader.py:88-89`` 的纪律）。

**阶段 B 才动注册表**：先 disable 所有「本轮不该在了」的，再 apply 所有该在的。
顺序固定 ⇒ 结果可重现。

回滚：位移表为主，基线快照为辅
==============================
``_displaced`` 在 register 之前抓走**被顶掉的整个版本 dict**。停用时先核对
「现在注册在这个名字上的还是不是我放的那个」，是才回填。

为什么位移表优于纯快照：快照有**拍摄时机**问题（覆盖层可以盖内置/composite/custom
任意一类，哪个时刻拍都不对）；快照丢版本维度（``_skills[name]`` 是
``{version: class}``）；而且快照回答不了「我顶掉的是**谁**」——那是 provenance 的
一部分。基线快照仍然留着，但只做两件事：「全部恢复内置」的急救按钮，和每次重载
后的不变式自检。

任务运行中：**整体挂起**
========================
``ExecutionContext.run`` 在**每个子步骤**都现查注册表
（``core/execution_context.py:366``），而 agent 侧 ``wrap_skill`` 建图时就把类钉死
了（``skill_adapter.py:635,659``）。所以任务运行中换注册表，会让一个跑到一半的
composite 后半段换成新代码 —— 比「晚几分钟生效」危险得多，而且完全不可复盘。

因此这里**不动注册表**，只把「待重载」记下来，任务结束时由
``CoreRuntime.drain_pending_agent_rebuild`` 一并做掉。这才真正兑现「正在跑的任务
从头到尾用它启动时那一版」。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from mast.skills.overlay import checks, manifest as M, paths as P, provenance as PV

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NOCHANGE = "nochange"
STATUS_BUSY = "busy"
STATUS_QUEUED = "queued"


@dataclass
class EntryResult:
    rel: str
    ok: bool = False
    reason: str = ""
    replaced: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    sha256: str = ""

    def describe(self) -> str:
        if not self.ok:
            return f"{self.rel}：{self.reason}"
        bits = []
        if self.replaced:
            bits.append(f"覆盖 {len(self.replaced)}")
        if self.added:
            bits.append(f"新增 {len(self.added)}")
        if self.dropped:
            bits.append(f"移除 {len(self.dropped)}")
        return f"{self.rel}：{'、'.join(bits) or '无技能'}（sha {self.sha256[:8]}）"


@dataclass
class ReloadReport:
    status: str = STATUS_OK
    applied: list[EntryResult] = field(default_factory=list)
    failed: list[EntryResult] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    baseline_drift: list[str] = field(default_factory=list)
    note: str = ""
    at: float = field(default_factory=time.time)

    @property
    def changed(self) -> bool:
        return bool(self.applied or self.disabled or self.restored)

    def describe(self) -> str:
        if self.status == STATUS_BUSY:
            return "已有一次重载在进行中，本次未执行。"
        if self.status == STATUS_QUEUED:
            return self.note or "任务运行中，已排队。"
        if self.status == STATUS_NOCHANGE:
            return "覆盖层没有变化（文件内容与上次加载完全一致）。"
        lines = []
        if self.applied:
            lines.append(f"已生效 {len(self.applied)} 个覆盖模块：")
            lines += [f"  · {r.describe()}" for r in self.applied]
        if self.restored:
            lines.append(f"已恢复内置版 {len(self.restored)} 个："
                         + "、".join(sorted(self.restored)))
        if self.failed:
            lines.append(f"被拒绝 {len(self.failed)} 个：")
            lines += [f"  · {r.describe()}" for r in self.failed]
        if self.baseline_drift:
            lines.append("⚠ 基线自检发现下列技能不再是内置版且不由覆盖层解释："
                         + "、".join(sorted(self.baseline_drift)))
        return "\n".join(lines) or "覆盖层为空。"


@dataclass
class _Applied:
    rel: str
    names: list[str]
    module: str
    sha256: str
    #: 我**具体注册了哪个版本** —— 按 name 存一个版本号。
    #: 少了它，多版本共存时回滚会判错：内置有 1.0.0 和 2.0.0、覆盖只换了 1.0.0，
    #: 而「所有版本都是我的吗」这个问题的答案是 False，于是回滚被自己拒掉。
    versions: dict = field(default_factory=dict)
    #: 声明移除的（``allow_removals``）—— 回滚时要把它们放回去。
    removed: list[str] = field(default_factory=list)


class OverlayManager:
    """进程内唯一的覆盖层状态。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._busy = False
        self._applied: dict[str, _Applied] = {}
        self._displaced: dict[str, dict[str, type]] = {}
        self._baseline: dict[str, dict[str, type]] | None = None
        self._last: ReloadReport | None = None

    # ── 基线 ─────────────────────────────────────────────────────────
    def capture_baseline(self, registry) -> None:
        """在**覆盖层加载之前**拍一份基线快照。只拍一次。"""
        with self._lock:
            if self._baseline is None:
                self._baseline = registry.snapshot_names()
                logger.debug("覆盖层基线快照：%d 个技能", len(self._baseline))

    def restore_all(self, registry) -> list[str]:
        """急救按钮：不管位移表乱成什么样，硬恢复到基线。"""
        with self._lock:
            if self._baseline is None:
                return []
            restored = []
            for name, versions in self._baseline.items():
                cur = registry._skills.get(name)
                if cur != versions:
                    registry._skills[name] = dict(versions)
                    registry._provenance.pop(name, None)
                    restored.append(name)
            # 基线里没有、现在有的（覆盖层新增的）一并撤掉
            for name in [n for n in list(registry._skills)
                         if n not in self._baseline]:
                registry.unregister(name)
                restored.append(name)
            self._applied.clear()
            self._displaced.clear()
            return restored

    def last_report(self) -> ReloadReport | None:
        with self._lock:
            return self._last

    def status(self) -> dict:
        with self._lock:
            return {
                "applied": {rel: {"names": a.names, "module": a.module,
                                  "sha256": a.sha256}
                            for rel, a in self._applied.items()},
                "busy": self._busy,
                "baseline_size": len(self._baseline or {}),
            }

    # ── 加载一个模块 ─────────────────────────────────────────────────
    @staticmethod
    def _load_module(rel: str, source: str):
        """在**独立命名空间**里加载。返回 ``(module, 错误)``。"""
        path = P.entry_path(rel)
        mod_name = P.module_name(rel)
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            return None, f"{rel}：建不出模块 spec"
        mod = importlib.util.module_from_spec(spec)
        # 先进 sys.modules 再 exec：dataclass / typing / inspect.getsource /
        # traceback 都靠它。custom_loader 现在没这么做，是个既有小坑，这里不重复。
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException as exc:  # noqa: BLE001 — 用户代码，什么都可能抛
            sys.modules.pop(mod_name, None)
            return None, f"{rel}：加载时抛异常 {type(exc).__name__}: {exc}"
        return mod, ""

    @staticmethod
    def _skill_classes(mod) -> tuple[list[type], list[str]]:
        """``(可注册的类, 说不清的那些)``。

        ⚠️ 「实例化失败」**不能**静默跳过。跳过之后这个类就从名字集合里消失，
        表现成「这个模块不再提供 X」—— 用户会去找自己删了什么，而真正的原因是
        ``__init__`` 抛了异常。一次「读不到」被折叠成了一个具体的值。
        """
        from mast.skills.base import BaseSkill

        out: list[type] = []
        problems: list[str] = []
        for attr in vars(mod).values():
            if not (isinstance(attr, type) and issubclass(attr, BaseSkill)
                    and attr is not BaseSkill
                    and attr.__module__ == mod.__name__):
                continue
            try:
                attr()          # 必须无参可实例化，否则注册不了
            except Exception as exc:   # noqa: BLE001
                problems.append(
                    f"{attr.__name__} 无法无参实例化（{type(exc).__name__}: {exc}）"
                    " —— 技能类的 __init__ 不能要参数")
                continue
            out.append(attr)
        return out, problems

    @staticmethod
    def _names_of(classes) -> tuple[dict[str, type], list[str]]:
        """``({技能名: 类}, 说不清的那些)``。

        同上：``metadata()`` 抛异常时**必须说出来**。静默跳过会让一个
        「metadata 里有 bug」表现成「这个模块什么技能都没有」，两者的修法完全不同。
        """
        from mast.core.registry import SkillRegistry

        out: dict[str, type] = {}
        problems: list[str] = []
        for cls in classes:
            try:
                meta = SkillRegistry._get_metadata_raw(cls)
            except Exception as exc:  # noqa: BLE001
                problems.append(
                    f"{cls.__name__}.metadata() 抛了 {type(exc).__name__}: {exc}"
                    " —— 不是「没有这个技能」，是它的元数据算不出来")
                continue
            out[meta.name] = cls
        return out, problems

    @staticmethod
    def _baseline_names(registry, overlay_of: str | None) -> dict[str, object]:
        """被覆盖的那个内置模块**实际贡献**了哪些技能名 → 它们的 raw metadata。"""
        if not overlay_of:
            return {}
        from mast.core.registry import SkillRegistry

        out: dict[str, object] = {}
        with registry._lock:
            items = list(registry._skills.items())
        for name, versions in items:
            for cls in versions.values():
                if getattr(cls, "__module__", "") == overlay_of:
                    try:
                        out[name] = SkillRegistry._get_metadata_raw(cls)
                    except Exception:  # noqa: BLE001
                        out[name] = None
                    break
        return out


#: 进程内单例。
_manager = OverlayManager()


def manager() -> OverlayManager:
    return _manager


def reset_for_tests() -> None:
    global _manager
    _manager = OverlayManager()


def _prepare(registry, entry: M.Entry) -> tuple[EntryResult, dict | None]:
    """阶段 A：解析 + 校验一个条目。**不动注册表。**"""
    rel = entry.path
    res = EntryResult(rel=rel)

    ok, why = P.is_valid_rel(rel)
    if not ok:
        res.reason = why
        return res, None

    path = P.entry_path(rel)
    if not path.is_file():
        res.reason = "文件不存在（清单里启用了，盘上没有）"
        return res, None

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        res.reason = f"读不出来：{exc}"
        return res, None

    syn = checks.check_parses(source, rel)
    if not syn.ok:
        res.reason = syn.error
        return res, None

    res.sha256 = PV.file_sha256(path)
    mod, err = OverlayManager._load_module(rel, source)
    if mod is None:
        res.reason = err
        return res, None

    classes, cls_problems = OverlayManager._skill_classes(mod)
    new_map, meta_problems = OverlayManager._names_of(classes)
    broken = cls_problems + meta_problems

    if not new_map:
        sys.modules.pop(P.module_name(rel), None)
        # 「一个技能都没算出来」有两种完全不同的原因，修法也完全不同：
        # 文件里真的没有技能类 vs. 有但它们的 __init__/metadata() 抛异常。
        # 不分开说，用户会去找自己删了什么。
        res.reason = ("；".join(broken) if broken else
                      "这个文件里没有可注册的技能"
                      "（没有 __module__ 指向本文件的 BaseSkill 子类）")
        return res, None

    overlay_of = P.overlay_of(rel)
    base_map = OverlayManager._baseline_names(registry, overlay_of)

    sets = checks.diff_sets(set(base_map), set(new_map))
    # 算出来的类里有坏的，即便别的能用也要说 —— 不然那几个技能会**静默缺席**，
    # 而 dropped 检查会把原因归到「你删了它」。
    problems = list(broken)
    problems += checks.check_no_silent_removal(sets, entry.allow_removals)

    from mast.core.registry import SkillRegistry
    for name, cls in new_map.items():
        base_meta = base_map.get(name)
        if base_meta is None:
            continue
        try:
            new_meta = SkillRegistry._get_metadata_raw(cls)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: 元数据读不出来（{exc}）")
            continue
        rep = checks.check_not_relaxed(new_meta, base_meta)
        problems.extend(rep.problems)

    if problems:
        sys.modules.pop(P.module_name(rel), None)
        res.reason = "；".join(problems)
        return res, None

    res.ok = True
    res.replaced = sorted(sets.replaced)
    res.added = sorted(sets.added)
    res.dropped = sorted(sets.dropped)
    return res, {"module": mod, "classes": new_map, "overlay_of": overlay_of,
                 "entry": entry}


def _apply(registry, res: EntryResult, prepared: dict) -> None:
    """阶段 B：把一个已通过校验的模块推进注册表。"""
    mgr = _manager
    names: list[str] = []
    versions: dict[str, str] = {}
    removed: list[str] = []

    # 声明移除的（allow_removals ∩ dropped）**真的移除**。
    # 报文里写的是「确实要移除的话…」，那就得真移除 —— 否则用户写了声明、
    # 看着技能还在，只会得出「这个开关没用」。
    entry = prepared["entry"]
    for name in sorted(set(res.dropped) & set(entry.allow_removals or ())):
        with registry._lock:
            existing = registry._skills.get(name)
            if existing and name not in mgr._displaced:
                mgr._displaced[name] = dict(existing)
        if registry.unregister(name):
            removed.append(name)

    for name, cls in prepared["classes"].items():
        with registry._lock:
            existing = registry._skills.get(name)
            if existing and name not in mgr._displaced:
                # 先抓走整个版本 dict —— 停用时要原样放回去
                mgr._displaced[name] = dict(existing)
        displaced_mod = ""
        prev = mgr._displaced.get(name) or {}
        if prev:
            displaced_mod = getattr(next(iter(prev.values())), "__module__", "")
        from mast.core.registry import SkillRegistry
        try:
            version = SkillRegistry._get_metadata_raw(cls).version
        except Exception:  # noqa: BLE001
            version = ""
        registry.register(cls, provenance=PV.for_overlay(
            name, cls, version=version, source_path=res.rel,
            sha256=res.sha256, displaced_module=displaced_mod,
            pack_id=P.pack_id_of(res.rel),
            signature=_signature_for(res.rel)))
        names.append(name)
        versions[name] = version
    mgr._applied[res.rel] = _Applied(
        rel=res.rel, names=names, module=prepared["module"].__name__,
        sha256=res.sha256, versions=versions, removed=removed)


def _signature_for(rel: str) -> str:
    """这个条目的签名状态。

    ``_packs/`` 下的条目走到这里时**一定**已经过了 :func:`_recheck_packs`
    （复核不过的在阶段 A 之前就被剔掉了），所以查不到公钥说明有人改了加载顺序 ——
    那种情况报 ``unverified``，不是一个看起来正常的值。「验不了」不能长得像
    「验过了」。
    """
    if not P.is_pack_entry(rel):
        return PV.SIG_LOCAL
    pub = _PACK_PUBKEYS.get(P.pack_id_of(rel), "")
    return f"verified:{pub}" if pub else "unverified"


def _disable(registry, rel: str) -> list[str]:
    """停用一个覆盖模块，把它顶掉的东西**放回去**。"""
    mgr = _manager
    app = mgr._applied.pop(rel, None)
    if app is None:
        return []
    restored: list[str] = []
    for name in app.names:
        ver = app.versions.get(name, "")
        with registry._lock:
            cur = registry._skills.get(name) or {}
            # ⚠️ 只看**我注册的那个版本**还是不是我的。
            # 用「所有版本都是我的吗」当判据会在多版本共存时判错：内置有 1.0.0 和
            # 2.0.0、覆盖只换了 1.0.0，那个问题的答案是 False，于是回滚被自己拒掉，
            # 覆盖版永远留在注册表里 —— 而日志说的是「拒绝回滚以免覆盖第三方」，
            # 指向一个根本不存在的第三方。
            mine_cls = cur.get(ver) if ver else None
            taken_by = (getattr(mine_cls, "__module__", "")
                        if mine_cls is not None else "")
            if cur and taken_by != app.module:
                logger.warning(
                    "停用覆盖 %s 时发现 %s v%s 已经不是我放的那个类"
                    "（现在来自 %s）—— 拒绝回滚它，以免覆盖第三方的注册",
                    rel, name, ver or "?", taken_by or "（已不在注册表）")
                continue
            prev = mgr._displaced.pop(name, None)
            if prev:
                registry._skills[name] = dict(prev)
                registry._provenance.pop(name, None)
            else:
                registry._skills.pop(name, None)
                registry._provenance.pop(name, None)
            restored.append(name)

    # 声明移除的那些也要放回去 —— 停用覆盖 = 完全回到覆盖之前的样子。
    for name in app.removed:
        prev = mgr._displaced.pop(name, None)
        if prev:
            with registry._lock:
                registry._skills[name] = dict(prev)
                registry._provenance.pop(name, None)
            restored.append(name)

    sys.modules.pop(app.module, None)
    return restored


#: 本轮被 pack 复核挡掉的条目 —— 由 reload_skills 收进报告，报告完清空。
_PACK_REJECTS: list = []

#: 本轮复核通过的包 → 签名者公钥前 8 位。
#: provenance 的 signature 字段要说的是「**谁签的**」，不是「来自哪个包」——
#: 后者已经有独立的 pack_id 字段。第一版写的是 pack_id[:8]，那是同一个信息
#: 换个地方再说一遍，而真正要紧的「哪把钥匙签的」一个字都没说。
_PACK_PUBKEYS: dict = {}


def _recheck_packs(pack_ids: list[str]) -> dict[str, str]:
    """``{pack_id: 拒绝理由}``。验得过的不出现在结果里，公钥记进 :data:`_PACK_PUBKEYS`。

    读不出来也算拒绝：一个 `_packs/<id>/` 目录**存在**却回答不了「你是谁签的」，
    正是「手放文件冒充签名版」那条路的形状。
    """
    from mast.update import skillpack as SPK

    out: dict[str, str] = {}
    _PACK_PUBKEYS.clear()
    for pid in pack_ids:
        if not pid:
            continue
        d = P.packs_dir() / pid
        try:
            v = SPK.verify_installed(d)
        except Exception as exc:                     # noqa: BLE001
            out[pid] = f"复核时出错：{type(exc).__name__}: {exc}"
            continue
        if not v.ok:
            out[pid] = "；".join(v.reasons) or "复核不过"
        else:
            _PACK_PUBKEYS[pid] = v.pubkey8
    return out


def reload_skills(registry, *, reason: str = "manual",
                  task_busy: bool = False) -> ReloadReport:
    """把覆盖层推到「盘上现在说的那个状态」。

    ``task_busy=True`` 时**什么都不做**并返回 ``queued`` —— 见模块 docstring。
    """
    mgr = _manager
    if task_busy:
        return ReloadReport(
            status=STATUS_QUEUED,
            note="任务运行中，覆盖层重载已排队 —— 当前任务结束后自动生效"
                 "（跑到一半的流程从头到尾用它启动时那一版）。")

    with mgr._lock:
        if mgr._busy:
            return ReloadReport(status=STATUS_BUSY)
        mgr._busy = True
    try:
        mgr.capture_baseline(registry)
        man = M.load()
        if man.unreadable:
            rep = ReloadReport(status=STATUS_OK)
            rep.note = (f"覆盖层清单读不出来（{man.unreadable}）——"
                        "本次不做任何改动，已生效的覆盖保持原样。")
            logger.error("覆盖层重载中止：%s", rep.note)
            with mgr._lock:
                mgr._last = rep
            return rep

        wanted = {e.path: e for e in man.enabled()}

        # ── 签名包：加载前复核 ──────────────────────────────────────────
        # 覆盖层会把 _packs/ 下的东西标成 `verified:<id>`。那个标记必须有凭据，
        # 否则它就是一句谎话 —— 手工往 _packs/ 里放一个文件，就冒充了签名版。
        #
        # 下载时验过一次**不够**：那次和现在之间隔着一段任何人都能写的时间。
        # 所以每次加载都拿 .pack.json 里的 sha256 重新对一遍，对不上**整包**
        # 的条目全部拒绝（不是只拒被改的那个 —— 半个包的语义没人说得清）。
        bad_packs = _recheck_packs(sorted({P.pack_id_of(r) for r in wanted
                                           if P.is_pack_entry(r)}))
        for pid, why in bad_packs.items():
            for rel in [r for r in list(wanted)
                        if P.is_pack_entry(r) and P.pack_id_of(r) == pid]:
                wanted.pop(rel, None)
                rep_pre = EntryResult(rel=rel, ok=False, reason=why)
                _PACK_REJECTS.append(rep_pre)
            logger.error("签名包 %s 复核不过，本轮它的条目全部不加载：%s", pid, why)

        # 快速返回：启用集合与 sha 都没变
        same = (set(wanted) == set(mgr._applied) and all(
            PV.file_sha256(P.entry_path(rel)) == mgr._applied[rel].sha256
            for rel in wanted))
        if same and mgr._applied:
            rep = ReloadReport(status=STATUS_NOCHANGE)
            with mgr._lock:
                mgr._last = rep
            return rep

        # 阶段 A
        prepared: dict[str, tuple[EntryResult, dict]] = {}
        rep = ReloadReport()
        # 被 pack 复核挡掉的先进 failed —— 它们必须出现在报告里，不能只在日志里。
        rep.failed.extend(_PACK_REJECTS)
        _PACK_REJECTS.clear()
        for rel, entry in sorted(wanted.items()):
            res, data = _prepare(registry, entry)
            if res.ok and data is not None:
                prepared[rel] = (res, data)
            else:
                rep.failed.append(res)
                logger.warning("覆盖层拒绝 %s：%s", rel, res.reason)

        # 阶段 B —— 先撤后上，顺序固定
        for rel in sorted(set(mgr._applied) - set(prepared)):
            rep.restored.extend(_disable(registry, rel))
            rep.disabled.append(rel)
        for rel in sorted(prepared):
            res, data = prepared[rel]
            if rel in mgr._applied:
                _disable(registry, rel)      # 先撤旧版再上新版
            _apply(registry, res, data)
            rep.applied.append(res)

        # 不变式自检：没被覆盖的名字必须还是基线里那个类**对象**
        rep.baseline_drift = _baseline_drift(registry)
        if rep.baseline_drift:
            logger.error("覆盖层基线自检：%s 不再是内置版且不由覆盖层解释 ——"
                         "回滚可能没回干净", "、".join(rep.baseline_drift))

        logger.info("覆盖层重载（%s）：生效 %d、恢复 %d、拒绝 %d",
                    reason, len(rep.applied), len(rep.restored), len(rep.failed))
        with mgr._lock:
            mgr._last = rep
        return rep
    finally:
        with mgr._lock:
            mgr._busy = False


def _baseline_drift(registry) -> list[str]:
    mgr = _manager
    if mgr._baseline is None:
        return []
    covered = {n for a in mgr._applied.values() for n in a.names}
    drift = []
    with registry._lock:
        for name, versions in mgr._baseline.items():
            if name in covered:
                continue
            cur = registry._skills.get(name)
            if cur is None:
                drift.append(name)
                continue
            for v, cls in versions.items():
                if cur.get(v) is not cls:
                    drift.append(name)
                    break
    return sorted(drift)


def apply_overlays(registry, *, reason: str = "startup") -> ReloadReport:
    """启动时调用一次。与热重载走**同一条路** —— 两条路会各自漂。"""
    return reload_skills(registry, reason=reason)


__all__ = ["OverlayManager", "ReloadReport", "STATUS_BUSY", "STATUS_NOCHANGE",
           "STATUS_OK", "STATUS_QUEUED", "apply_overlays", "manager",
           "reload_skills", "reset_for_tests"]
