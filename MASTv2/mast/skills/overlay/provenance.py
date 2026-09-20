"""每个技能的**来历** —— 一个只能被观测、不能被声明的事实。

为什么不放进 ``SkillMetadata``
==============================
``SkillMetadata`` 是**技能作者对接口与安全画像的声明**；provenance 是**运行时对
加载事实的记录**。四条理由，第三条最硬：

1. **语义归属不同**。放进 metadata 就得由 ``SkillRegistry._get_metadata()`` 注入，
   而那段有一条明写的、载重的契约（覆写层失败仍回落基线，但必须可见、绝不静默，
   ``core/registry.py:353-357``）。往一条安全关键路径上加可变注入，换来的只是省
   几个 join。
2. **缓存粒度不对**。``_metadata_cache`` 按**类**缓存；provenance 是按**注册事件**
   的（``loaded_at`` / ``sha256`` / 顶掉了谁）。同一个类重复注册要有不同的
   ``loaded_at``，按类缓存表达不了。
3. **metadata 是可被声明的**。``admin/override_store.py`` 按字段名白名单 merge
   ——``skill_overrides.json`` 是一个用户可写的文件。而 provenance 的**全部价值
   在于它不可被声明**：一个能被写进配置的「来历」什么也证明不了。
4. **表达不了「顶掉了谁」**。位移关系是双向的，单个类的 metadata 里没有它的位置。

代价（诚实记下）：metadata 会自动流进每个已有的序列化点，side table 要每个消费点
显式 join。但那恰恰是想要的 —— 每个 join 点都是一次「这里要不要露出来」的决定。
而且最便宜的那个判据是白送的：``skill_adapter.py:1245`` 已经把
``tool.metadata["skill_source"] = cls.__module__`` 盖上去了，用独立命名空间
``mast.skills._overlay.*`` 之后它自动说实话。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

#: 覆盖层加载出来的模块都在这个命名空间下 —— 判 origin 只看它。
OVERLAY_NAMESPACE = "mast.skills._overlay"

#: 本机手放的文件（未签名）；签名包解出来的写 ``verified:<pubkey8>``。
SIG_LOCAL = "local_unsigned"
SIG_NONE = "n/a"


@dataclass(frozen=True)
class SkillProvenance:
    """一个技能此刻**从哪来**。"""

    name: str
    version: str = ""
    origin: str = "other"          # overlay / builtin / composite / paper / custom / …
    module: str = ""
    source_path: str = ""          # 覆盖层才有：相对 <data_root> 的 posix 路径
    sha256: str = ""               # 覆盖层才有：文件全文
    loaded_at: float = 0.0
    pack_id: str = ""              # 签名包 id；本机手改 = ""
    signature: str = SIG_NONE
    displaced_module: str = ""     # 覆盖时被顶掉的那个类的 __module__

    @property
    def is_overlay(self) -> bool:
        return self.origin == "overlay"

    @property
    def short_sha(self) -> str:
        return self.sha256[:8] if self.sha256 else ""

    def describe(self) -> str:
        """给日志和 UI 的一行话。"""
        if not self.is_overlay:
            return f"{self.origin}（{self.module}）"
        bits = [f"覆盖层 {self.source_path}"]
        if self.short_sha:
            bits.append(f"sha {self.short_sha}")
        if self.displaced_module:
            bits.append(f"顶掉 {self.displaced_module}")
        if self.pack_id:
            bits.append(f"包 {self.pack_id}")
        bits.append("已签名" if self.signature.startswith("verified") else "本机未签名")
        return "，".join(bits)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "version": self.version, "origin": self.origin,
            "module": self.module, "source_path": self.source_path,
            "sha256": self.sha256, "short_sha": self.short_sha,
            "loaded_at": self.loaded_at, "pack_id": self.pack_id,
            "signature": self.signature,
            "displaced_module": self.displaced_module,
            "is_overlay": self.is_overlay,
        }


def classify_origin(cls) -> str:
    """一个技能类的来源分类。

    **唯一真源** —— ``webui/builder_api._skill_source()`` 现在委托到这里。
    两处各写一遍的话，UI 上的徽章和 provenance 表迟早各说各话，而那种不一致
    没有任何测试在看。
    """
    if getattr(cls, "_AGENT_TOOL_SOURCE", None):
        return "agent_tool"
    mod = getattr(cls, "__module__", "") or ""
    # 覆盖层优先判：一个覆盖版的 SpecComposite 首先是「覆盖」。
    if OVERLAY_NAMESPACE in mod:
        return "overlay"
    try:
        from mast.skills.composite.interpreter import SpecComposite
        if isinstance(cls, type) and issubclass(cls, SpecComposite):
            return "user_composite"
    except Exception:  # pragma: no cover — defensive
        pass
    if ".skills.builtins" in mod:
        return "builtin"
    if ".skills.composite" in mod:
        return "composite"
    if ".skills.paper" in mod:
        return "paper"
    if ".skills.custom" in mod:
        return "custom"
    return "other"


def synthesize(name: str, cls, *, version: str = "") -> SkillProvenance:
    """没有显式盖章时，按类现场合成一条。

    ``SkillRegistry.provenance()`` **从不返回 None** —— 「不知道来历」和「来自内置」
    是两件不同的事，但对调用方来说都得有一个可渲染的答案；把 None 丢给 UI 只会
    在每个消费点长出一个 ``or "未知"``。
    """
    return SkillProvenance(
        name=name, version=version or "",
        origin=classify_origin(cls),
        module=getattr(cls, "__module__", "") or "",
    )


def file_sha256(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def for_overlay(name: str, cls, *, version: str, source_path: str,
                sha256: str = "", displaced_module: str = "",
                pack_id: str = "", signature: str = SIG_LOCAL) -> SkillProvenance:
    return SkillProvenance(
        name=name, version=version, origin="overlay",
        module=getattr(cls, "__module__", "") or "",
        source_path=source_path, sha256=sha256, loaded_at=time.time(),
        pack_id=pack_id, signature=signature,
        displaced_module=displaced_module,
    )


def fingerprint(triples) -> str:
    """``[(技能名, 模块, 版本), …]`` → 一个稳定的指纹。

    **生效探针的核心。** 注册表说「已覆盖」和 agent **真的在跑覆盖版**是两件事，
    中间隔着一次图重建 —— 而重建可能被任务占用推迟、可能失败、可能压根没接线。

    两侧各自收集三元组、调**同一个**函数：
      · 注册表侧 = 现在注册着什么；
      · agent 侧 = ``build_instrument_skill_tools`` 上次实际 wrap 了什么。
    不相等 ⇒ 工具表还没跟上，UI 必须这么说。

    这比「记得去看日志」强的地方在于它**不会因为遗漏而说谎**：指纹覆盖全集，
    漏掉任何一个技能都会让两边对不上。
    """
    h = hashlib.sha256()
    for triple in sorted(triples):
        # ``repr`` 而不是拼字符串：技能名或模块名里出现分隔符时，拼接会让两个
        # 不同的输入映射到同一个指纹 —— 而那恰好会把「工具表没跟上」报成「跟上了」，
        # 正是这个探针唯一不能犯的错。
        h.update(repr(tuple(str(x) for x in triple)).encode("utf-8"))
    return h.hexdigest()


def registry_triples(registry) -> list[tuple[str, str, str]]:
    """注册表现在注册着什么。"""
    out = []
    with registry._lock:
        items = list(registry._skills.items())
    for name, versions in items:
        if not versions:
            continue
        from mast.core.registry import _parse_version
        latest = sorted(versions.keys(), key=_parse_version)[-1]
        cls = versions[latest]
        out.append((name, getattr(cls, "__module__", "") or "", latest))
    return out


__all__ = ["OVERLAY_NAMESPACE", "SIG_LOCAL", "SIG_NONE", "SkillProvenance",
           "classify_origin", "file_sha256", "fingerprint", "for_overlay",
           "registry_triples", "synthesize"]
