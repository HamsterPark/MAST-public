"""SkillRegistry — discovers, registers, and manages skill modules.

P-vendored from v1 mast/core/registry.py 2026-04-23. Adjustments for v2:
  - Default discover packages still target `mast.skills.*` (resolved via v2
    PYTHONPATH; subpackages live in MASTv2/mast/skills/)
  - `to_tool_definitions()` retained for any code that wants raw Claude
    tool_use format (e.g., experiment_design agent's skill catalog tool)
  - `describe_skills()` is **new in v2** — returns Markdown descriptions for
    LLM context-building (Experiment Design agent will use it for skill lookup)

Phase 4 wires up: at MAST module init, instantiate registry and call
`registry.discover()` once. Skills are then accessible by name across all 6
agents.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import sys
import threading
from typing import Any

from mast.core.types import ParameterSpec, SafetyLevel, SkillMetadata

logger = logging.getLogger(__name__)

# v1 ParameterSpec.type → JSON Schema type
_TYPE_MAP: dict[str, str] = {
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
}

# v2 default discover packages — same names as v1, resolved via PYTHONPATH.
_DEFAULT_DISCOVER_PACKAGES = [
    "mast.skills.builtins",
    "mast.skills.composite",
    "mast.skills.paper",
]


class SkillRegistry:
    """Discovers, registers, and manages skill modules."""

    def __init__(self):
        self._skills: dict[str, dict[str, type]] = {}  # name -> {version -> class}
        self._metadata_cache: dict[type, SkillMetadata] = {}
        # 每个技能**从哪来**（按 name，最新版本胜）。刻意**不**放进 SkillMetadata：
        # metadata 是作者的声明、而且能被 skill_overrides.json 覆写；provenance 的
        # 全部价值在于它只能被观测、不能被声明。详见 skills/overlay/provenance.py。
        self._provenance: dict[str, object] = {}
        # 修复项 review fix: register/unregister became runtime operations
        # (composite hot-(un)register from GUI worker threads) while
        # list_skills() is iterated concurrently from Starlette threads
        # (/agents/snapshot), agent tools (describe_skills) and admin tabs —
        # an unguarded dict mutation mid-iteration raises RuntimeError
        # ("dictionary changed size during iteration", empirically reproduced
        # in review). RLock: register/unregister/get/list all nest freely.
        self._lock = threading.RLock()

    # ── Registration ─────────────────────────────────────────────────

    def register(self, skill_class: type, *, provenance=None) -> None:
        """Register a BaseSkill subclass. Extracts metadata for name+version key.

        ``provenance`` (kw-only, 2026-08-20) 记这个技能**从哪来**。不传的话
        :meth:`provenance` 会按 ``__module__`` 现场合成一条 —— 所以老调用方
        （discover / spec loader / custom loader / tool bridge）一行都不用改。
        """
        meta = self._get_metadata(skill_class)
        with self._lock:
            self._metadata_cache[skill_class] = meta
            if meta.name not in self._skills:
                self._skills[meta.name] = {}
            # 审查 [#144]: a (name, version) collision previously
            # overwrote the prior class SILENTLY — two distinct skills sharing a
            # name (e.g. a copy-paste error, or a paper skill re-using a builtin
            # name) would leave only the last-discovered one reachable with no
            # trace. Warn on a genuine conflict (different class), but stay
            # quiet when the SAME class is re-registered (idempotent discover()
            # re-runs are legitimate).
            existing = self._skills[meta.name].get(meta.version)
            # 覆盖层是**有意**换掉同名同版本的类 —— 那不是冲突，是它的全部工作。
            # 照 warn 会让每次覆盖都产生一条看起来像错误的日志，真正的冲突反而
            # 被淹掉（「狼来了」是这类守卫最常见的死法）。
            _intentional = getattr(provenance, "origin", "") == "overlay"
            if existing is not None and existing is not skill_class and _intentional:
                logger.info(
                    "覆盖层替换 %s v%s：%s.%s ← %s.%s",
                    meta.name, meta.version,
                    skill_class.__module__, skill_class.__qualname__,
                    existing.__module__, existing.__qualname__,
                )
            elif existing is not None and existing is not skill_class:
                logger.warning(
                    "Skill name+version collision: %s v%s — %s.%s is overwriting %s.%s",
                    meta.name, meta.version,
                    skill_class.__module__, skill_class.__qualname__,
                    existing.__module__, existing.__qualname__,
                )
            self._skills[meta.name][meta.version] = skill_class
            if provenance is not None:
                self._provenance[meta.name] = provenance
            elif meta.name in self._provenance:
                # 换了实现却留着旧来历，比没有来历更糟 —— 那会指向一个已经不在
                # 这个名字上的文件。丢掉，让 provenance() 现场合成。
                prev = self._provenance.get(meta.name)
                if getattr(prev, "module", None) != getattr(skill_class, "__module__", ""):
                    self._provenance.pop(meta.name, None)
        logger.info("Registered skill: %s v%s", meta.name, meta.version)

    def get(self, name: str, version: str | None = None) -> type:
        """Get skill class by name. If version is None, return latest by semver."""
        with self._lock:
            if name not in self._skills:
                raise KeyError(f"Skill '{name}' not found in registry")
            versions = self._skills[name]
            if version is not None:
                if version not in versions:
                    raise KeyError(f"Skill '{name}' version '{version}' not found")
                return versions[version]
            latest = sorted(versions.keys(), key=_parse_version)[-1]
            return versions[latest]

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._skills

    def unregister(self, name: str, *, only_subclass_of: type | None = None) -> bool:
        """Remove ALL versions of *name*. Returns True if it was registered.

        修复项 (2026-06-11): composite hot-unregister — deleting a composite in
        the GUI must also drop it from the LIVE registry, otherwise the
        deleted skill stays callable (and listed to agents) until restart.

        ``only_subclass_of`` guards against name collisions: when set, the
        unregister is REFUSED unless every registered version is a subclass of
        that type (e.g. a composite named "SetBias" must never deregister the
        builtin SetBias)."""
        with self._lock:
            versions = self._skills.get(name)
            if not versions:
                return False
            if only_subclass_of is not None and not all(
                    isinstance(cls, type) and issubclass(cls, only_subclass_of)
                    for cls in versions.values()):
                logger.warning(
                    "unregister(%r) refused: registered class(es) are not all %s "
                    "(name collision with a non-matching skill)",
                    name, only_subclass_of.__name__)
                return False
            self._skills.pop(name, None)
            self._provenance.pop(name, None)
            for cls in versions.values():
                self._metadata_cache.pop(cls, None)
        logger.info("Unregistered skill: %s (%d version(s))", name, len(versions))
        return True

    def provenance(self, name: str):
        """这个技能此刻从哪来。**从不返回 None。**

        没有显式盖章的（内置 discover、spec loader、custom loader、tool 桥接）
        按 ``__module__`` 现场合成 —— 「不知道来历」和「来自内置」是两件不同的
        事，但对调用方来说都得有一个可渲染的答案；返回 None 只会在每个消费点
        长出一个 ``or "未知"``。
        """
        from mast.skills.overlay.provenance import synthesize

        with self._lock:
            got = self._provenance.get(name)
            if got is not None:
                return got
            versions = self._skills.get(name)
        if not versions:
            from mast.skills.overlay.provenance import SkillProvenance
            return SkillProvenance(name=name, origin="absent")
        latest = sorted(versions.keys(), key=_parse_version)[-1]
        return synthesize(name, versions[latest], version=latest)

    def snapshot_names(self) -> dict[str, dict[str, type]]:
        """当前 name → {version → class} 的浅快照。

        覆盖层用它做两件事：「全部恢复内置」的急救按钮，和每次重载后的不变式
        自检（没被覆盖的名字必须还 ``is`` 原来那个类对象）。
        """
        with self._lock:
            return {n: dict(v) for n, v in self._skills.items()}

    def list_skills(self) -> list[SkillMetadata]:
        """List metadata for all registered skills (latest version of each)."""
        result: list[SkillMetadata] = []
        with self._lock:
            for versions in self._skills.values():
                # Latest version
                latest_v = sorted(versions.keys(), key=_parse_version)[-1]
                cls = versions[latest_v]
                cached = self._metadata_cache.get(cls)
                result.append(cached if cached is not None else self._get_metadata(cls))
        return result

    # ── Auto-discovery ───────────────────────────────────────────────

    def discover(self, *packages: str) -> int:
        """Auto-discover BaseSkill subclasses in given packages.

        Returns count of newly registered skills. Walks packages recursively.
        """
        if not packages:
            packages = tuple(_DEFAULT_DISCOVER_PACKAGES)

        count = 0
        for package_name in packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                logger.debug("Package '%s' not importable: %s", package_name, e)
                continue

            if not hasattr(package, "__path__"):
                count += self._scan_module(package)
                continue

            seen: set[str] = set()
            for _importer, modname, _ispkg in pkgutil.walk_packages(
                package.__path__, prefix=package_name + "."
            ):
                try:
                    mod = importlib.import_module(modname)
                    seen.add(modname)
                    count += self._scan_module(mod)
                except Exception as e:
                    logger.warning("Failed to import %s: %s", modname, e)

            # Frozen (PyInstaller) fallback: pkgutil.walk_packages enumerates NOTHING
            # for a frozen package (the FrozenImporter exposes no filesystem walk), so
            # the loop above registers 0 skills in the PACKAGED app — the empty
            # describe_skills / no-tools bug. But each skill package's __init__ eagerly
            # imports its submodules, so they ARE loaded in sys.modules — scan those
            # for anything the walk missed. No-op in dev (walk already covered them).
            prefix = package_name + "."
            for modname, mod in list(sys.modules.items()):
                if mod is not None and modname.startswith(prefix) and modname not in seen:
                    seen.add(modname)
                    count += self._scan_module(mod)
        return count

    # ── Output formats ───────────────────────────────────────────────

    def to_tool_definitions(self) -> list[dict]:
        """v1-compat: Claude tool_use raw format. Used only if a caller needs it.

        v2 agents do NOT use this — they wrap each skill as a LangChain @tool via
        `mast.agents._shared.skill_adapter.wrap_skill`. This method is kept for
        Experiment-Design's optional "skill catalog browse" function.
        """
        tools: list[dict] = []
        for meta in self.list_skills():
            properties: dict[str, Any] = {}
            required: list[str] = []
            for param in meta.parameters:
                prop: dict[str, Any] = {
                    "type": _TYPE_MAP.get(param.type, "string"),
                }
                if param.description:
                    prop["description"] = param.description
                if param.unit:
                    prop["description"] = prop.get("description", "") + f" (unit: {param.unit})"
                if param.min_value is not None:
                    prop["minimum"] = param.min_value
                if param.max_value is not None:
                    prop["maximum"] = param.max_value
                if param.allowed_values is not None:
                    prop["enum"] = param.allowed_values
                if param.default is not None:
                    prop["default"] = param.default
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)
            tools.append({
                "name": meta.name,
                "description": meta.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        return tools

    def describe_skills(
        self,
        category: str | None = None,
        safety_level: SafetyLevel | None = None,
        tag: str | None = None,
    ) -> str:
        """**v2-new**: human-readable Markdown listing for Experiment-Design agent.

        Filtered by optional category / safety_level / tag. Used as a tool
        return-value when XD agent asks "what skills are available?".
        """
        lines: list[str] = ["# Available skills\n"]
        skills = sorted(self.list_skills(), key=lambda m: m.name)

        for m in skills:
            if category is not None and m.category != category:
                continue
            if safety_level is not None and m.safety_level != safety_level:
                continue
            if tag is not None and tag not in m.tags:
                continue

            sl = m.safety_level.name if hasattr(m.safety_level, "name") else str(m.safety_level)
            cat = m.category.name if hasattr(m.category, "name") else str(m.category)
            lines.append(f"## {m.name} (v{m.version}) — {sl} / {cat}")
            if m.description:
                lines.append(m.description)
            if m.parameters:
                params_str = ", ".join(
                    f"{p.name}: {p.type}" + (f" [{p.unit}]" if p.unit else "")
                    + ("?" if not p.required else "")
                    for p in m.parameters
                )
                lines.append(f"- **params**: {params_str}")
            if m.preconditions:
                lines.append(f"- **preconditions**: {', '.join(m.preconditions)}")
            if m.tags:
                lines.append(f"- **tags**: {', '.join(m.tags)}")
            lines.append("")  # blank line between skills

        return "\n".join(lines)

    # ── Internal helpers ─────────────────────────────────────────────

    def _scan_module(self, mod) -> int:
        """Scan a module for BaseSkill subclasses and register them."""
        count = 0
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                self._is_base_skill(obj)
                and obj.__module__ == mod.__name__
                and not inspect.isabstract(obj)
            ):
                # Skip parametrized base/helper classes that the auto-scanner
                # can't instantiate. register() reads metadata by calling
                # ``obj()`` with no args; a class whose __init__ requires a
                # positional arg (e.g. the generic ``SpecComposite(spec)``
                # interpreter base — its registrable form is the no-arg
                # ``_Bound`` subclass produced by make_spec_skill, registered
                # explicitly via register_spec) is NOT a discoverable skill and
                # would otherwise emit a spurious "Failed to register" warning
                # on every scan. (2026-06-08: surfaced during full v2 testing.)
                if not self._instantiable_no_args(obj):
                    logger.debug("Skipping non-discoverable skill class %s "
                                 "(__init__ requires args)", _name)
                    continue
                try:
                    self.register(obj)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to register %s: %s", _name, e)
        return count

    @staticmethod
    def _instantiable_no_args(cls: type) -> bool:
        """True if ``cls()`` is callable with no positional args.

        The auto-scanner registers a skill by instantiating it with no args to
        read its metadata; a class whose ``__init__`` has a required positional
        parameter is a parametrized base/helper, not a discoverable skill.
        Conservative: if the signature can't be introspected, return True so the
        existing register()/try-except path still handles it.
        """
        try:
            sig = inspect.signature(cls.__init__)
        except (ValueError, TypeError):
            return True
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is inspect.Parameter.empty:
                return False
        return True

    @staticmethod
    def _is_base_skill(cls: type) -> bool:
        """Check if cls inherits BaseSkill (without importing v1 BaseSkill directly)."""
        for parent in inspect.getmro(cls):
            if parent.__name__ == "BaseSkill" and parent is not cls:
                return True
        return False

    @staticmethod
    def _get_metadata_raw(skill_class: type) -> SkillMetadata:
        """Metadata AS DECLARED BY THE SKILL — admin overrides NOT applied.

        Split out of ``_get_metadata`` (2026-08-19, skill-overlay 修复项). The
        overlay's "may only tighten, never relax" check has to compare
        raw-against-raw: if it compared against override-MERGED baselines, an
        admin override that *lowers* a builtin's safety_level would become the
        yardstick, and an overlay could then legally stop at that lowered
        level — laundering the approval gate through two layers that each
        looked reasonable on its own.

        Use this ONLY when judging the skill author's declaration. Everything
        that needs the EFFECTIVE envelope (gates, tool schemas, the UI) wants
        ``_get_metadata``.
        """
        if not hasattr(skill_class, "metadata"):
            raise ValueError(
                f"Skill class '{skill_class.__name__}' has no 'metadata' attribute"
            )
        meta = skill_class.metadata
        if callable(meta):
            try:
                meta = meta()  # classmethod or @property?
            except TypeError:
                # Bound method on instance — need to instantiate
                meta = skill_class().metadata()
        return meta

    @staticmethod
    def _get_metadata(skill_class: type) -> SkillMetadata:
        """Extract SkillMetadata from a skill class, applying admin overrides."""
        meta = SkillRegistry._get_metadata_raw(skill_class)
        # Apply admin override. 修复项 review: this is now a LOAD-BEARING path
        # for the approval gates (ctx.run + executor both judge safety_level
        # through here) — an override-layer failure still falls back to the
        # baseline metadata (fail-open w.r.t. an admin DANGEROUS upgrade), but
        # it must at least be VISIBLE, never silent.
        try:
            from mast.admin.override_store import (
                ConfigOverrideRegistry,
                apply_skill_metadata_override,
            )
            ovr = ConfigOverrideRegistry.get().get_skill_override(meta.name)
            if ovr:
                meta = apply_skill_metadata_override(meta, ovr)
        except Exception as exc:
            logger.warning(
                "admin override apply failed for skill %r — using BASELINE "
                "metadata (an admin safety_level upgrade may not be in "
                "effect): %s", getattr(meta, "name", "?"), exc)
        return meta


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse semver string to tuple for sorting."""
    parts: list[int] = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


# ── "which skills can this machine actually call?" — ONE implementation ──────

def _names_from(registry) -> set[str]:
    """Skill NAMES out of whatever ``registry.list_skills()`` hands back.

    ``SkillRegistry.list_skills()`` returns ``list[SkillMetadata]`` — **objects,
    not strings**. Reading the name means reading ``.name``; a registry that
    already lists plain names is accepted too. Nothing else is: a value that is
    neither is DROPPED rather than coerced.

    That last rule is the whole point of this helper. The previous version of
    this scan did ``{str(n) for n in registry.list_skills()}``, which turns each
    dataclass into ``"SkillMetadata(name='ScanAt', version=…)"``. The set came
    back **non-empty and 100% wrong**, so every caller's membership test failed
    and every caller's "did I get anything?" guard passed. On the rig
    (2026-08-04) that reported all 19 required skills missing on a rig with 417
    registered — including one that had been called successfully minutes before.
    """
    if registry is None:
        return set()
    try:
        entries = registry.list_skills()
    except Exception as exc:  # noqa: BLE001
        logger.warning("registered_skill_names: list_skills() failed: %s", exc)
        return set()
    names: set[str] = set()
    for entry in entries:
        name = getattr(entry, "name", None)      # SkillMetadata → its name
        if not isinstance(name, str) and isinstance(entry, str):
            name = entry                          # already a bare name
        if isinstance(name, str) and name:
            names.add(name)
    return names


def registered_skill_names(registry: "SkillRegistry | None" = None) -> set[str]:
    """The skills that can ACTUALLY be dispatched on this machine, by name.

    The single answer to a question three self-checks used to answer two ways.
    ``TipConditioningSelfCheck`` and ``TipForgeSelfCheck`` shared one
    implementation (the broken one — forge imported conditioning's helper);
    ``ScanIntelSelfCheck`` had its own, and it was right. On the rig
    2026-08-04 the two answers were live at the same moment on the same rig:
    "417 registered (complete)" and "missing 19". When one question has several
    implementations, the wrong one does not raise its hand — it just disagrees
    with the others somewhere nobody is comparing.

    Two different questions hide here, and this answers them in priority order:

    1. **"Is it callable right now?"** — the live registry the ``ExecutionContext``
       dispatches through (``ctx._registry``). This is the authority: a skill is
       callable iff it is in there, because that is the dict ``ctx.run()`` looks
       in.
    2. **"Is it in this build at all?"** — asked whenever the live answer is
       unavailable (a bare context, a unit test, a CLI probe, a registry that
       raises). Answered by a fresh ``discover()``, which is also what catches
       the frozen-build failure this family of checks was written for:
       PyInstaller's ``walk_packages`` walks nothing, so a skill whose module the
       package ``__init__`` forgot to import silently ceases to exist in the
       packaged app.

    Falling through to (2) rather than reporting "nothing installed" is a
    deliberate choice about WHICH error to make. The two are not symmetric: a
    false *present* fails loudly at the point of use (``ctx.run`` raises and
    names the skill), while a false *missing* stops the agent before it starts
    and tells it a falsehood on the way out — the blocking self-checks that call
    this do not block mechanically, they block by SAYING the workflow cannot
    run. That is the failure being fixed here, so ambiguity resolves away from
    it.

    Never raises: a self-check that dies while checking is worse than one that
    reports an empty set, and the caller can tell the two apart by the size.
    """
    names = _names_from(registry)
    if names:
        return names
    try:
        fresh = SkillRegistry()
        fresh.discover()
        return _names_from(fresh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("registered_skill_names: discover() failed: %s", exc)
        return set()


__all__ = ["SkillRegistry", "registered_skill_names"]
