"""Custom-skill loader — 从用户数据目录加载显式启用的 .py 技能（P5-A）。

目录：``project_root()/config/custom_skills/``（升级/OTA 不会动它）。

安全模型（缺一不加载）：
  1. **explicit allowlist** —— 只加载 ``enabled.json`` 的 ``{"enabled": [名]}``
     列表里的文件。把 .py 拷进目录**不会**被执行：启用是用户的显式动作
     （等价于 skill_author 的 allow_exec 人审门，且对手工拷入的文件同样生效）；
  2. **AST 复检** —— 加载前重跑 skill_author 的静态拒绝名单（deny-list，
     防呆不防恶意——真正的边界仍是 allowlist + 本地文件系统权限）；
  3. **best-effort 隔离** —— 单个技能加载失败只记日志，绝不影响启动。

注意：这是「例外轨」基础设施（R3 主轨是 CompositeSpec 数据）。网络分发
.py 仍被冻结（P5 服务器端只收 manifest——Ed25519 签名链是解冻硬前提）。
"""

from __future__ import annotations

import importlib.util
import json
import logging

logger = logging.getLogger(__name__)


def custom_skills_dir():
    from mast._runtime_paths import project_root
    return project_root() / "config" / "custom_skills"


def _enabled_names() -> list[str]:
    p = custom_skills_dir() / "enabled.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        names = d.get("enabled")
        if isinstance(names, list):
            return [str(n) for n in names if isinstance(n, str)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("custom_skills/enabled.json unreadable: %s", exc)
    return []


def load_custom_skills(registry) -> list[str]:
    """Import + register every ENABLED custom skill. Returns loaded names."""
    d = custom_skills_dir()
    loaded: list[str] = []
    for name in _enabled_names():
        if not name.isidentifier() or name.startswith("__"):
            logger.warning("custom skill %r: bad name — skipped", name)
            continue
        f = d / f"{name}.py"
        if not f.exists():
            logger.warning("custom skill %r enabled but %s missing", name, f)
            continue
        try:
            code = f.read_text(encoding="utf-8")
            # AST 复检（与 skill_author 同一 deny-list；手工拷入的文件也过）
            try:
                from mast.llm.skill_author import SkillAuthor
                violations = SkillAuthor._ast_safety_check(
                    SkillAuthor.__new__(SkillAuthor), code)
            except Exception:  # pragma: no cover — checker 不可用时保守拒绝
                violations = ["AST checker unavailable"]
            if violations:
                logger.warning("custom skill %r failed AST check — NOT "
                               "loaded:\n  %s", name, "\n  ".join(violations))
                continue
            spec = importlib.util.spec_from_file_location(
                f"mast.skills.custom.{name}", str(f))
            if spec is None or spec.loader is None:
                raise ImportError("cannot create module spec")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            from mast.skills.base import BaseSkill
            n_before = len(loaded)
            for attr in vars(mod).values():
                if (isinstance(attr, type) and issubclass(attr, BaseSkill)
                        and attr is not BaseSkill
                        and attr.__module__ == mod.__name__):
                    registry.register(attr)
                    loaded.append(name)
                    break
            if len(loaded) == n_before:
                logger.warning("custom skill %r: no BaseSkill subclass found",
                               name)
        except Exception as exc:  # noqa: BLE001 — 单个失败不毁启动
            logger.warning("custom skill %r load failed: %s", name, exc)
    if loaded:
        logger.info("Loaded %d custom skill(s): %s", len(loaded),
                    ", ".join(loaded))
    return loaded


__all__ = ["custom_skills_dir", "load_custom_skills"]
