"""SkillAuthor: LLM generates new skill code at runtime."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import re
from pathlib import Path
from typing import Any

from mast.core.registry import SkillRegistry
from mast.core.types import SafetyLevel
from mast.llm.client import ClaudeClient
from mast.llm.prompts.skill_author_system import SKILL_AUTHOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Modules that generated skills must never import
_FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "sys", "shutil", "pathlib",
    "socket", "ctypes", "signal", "multiprocessing",
    "importlib", "builtins", "code", "codeop", "compileall",
})

# Built-in functions that generated code must not call
_FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "compile", "__import__",
    "globals", "locals", "getattr", "setattr", "delattr",
    "vars", "dir",
    "breakpoint", "exit", "quit",
})

# A generated skill's *name* must be a plain Python identifier. It is used both
# as the on-disk filename and the dotted module name, so anything else (path
# separators, '..', leading dot, dunders) is a path-traversal / code-overwrite
# vector and is rejected outright.
_VALID_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# P5 (2026-06-12): custom skills moved OUT of the frozen package directory —
# the old _pkg_root()/skills/custom location was wiped by every upgrade,
# fought the OTA delta whitelist, and was never in the registry discover list
# (skills silently vanished on restart). User-authored code is user data:
# it lives under the data root, and loading is gated by an explicit
# enabled.json allowlist (see mast.skills.custom_loader).
from mast._runtime_paths import project_root as _proj_root
_CUSTOM_SKILLS_DIR = _proj_root() / "config" / "custom_skills"


def _validate_skill_name(name: str) -> None:
    """Reject any name that is not a bare Python identifier.

    Guards against ``name='../../core/registry'`` (overwrite real code),
    ``name='..\\..\\evil'`` (escape the custom dir), dunder names, and dotted
    module-path injection — all of which the original code interpolated
    straight into the output path and dotted module name with no validation.
    """
    if not isinstance(name, str) or not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"Invalid skill name {name!r}: must match {_VALID_NAME_RE.pattern} "
            f"(a plain identifier, no path separators / dots / dunders)."
        )
    if name.startswith("__") or name.endswith("__"):
        raise ValueError(f"Invalid skill name {name!r}: dunder names are not allowed.")


class SkillAuthor:
    """LLM generates new skill code at runtime."""

    def __init__(self, client: ClaudeClient, registry: SkillRegistry):
        self._client = client
        self._registry = registry

    def create_skill(
        self, description: str, name: str, *, allow_exec: bool = False
    ) -> str:
        """Generate a new skill from natural language description.

        1. Validate ``name`` is a plain identifier (no path traversal).
        2. Build prompt with BaseSkill template + existing skills as examples
        3. Claude generates Python code
        4. AST safety check (best-effort static guard; NOT a sandbox).
        5. Write to mast/skills/custom/{name}.py
        6. ONLY if ``allow_exec=True`` (explicit human confirmation):
           import + exec_module + register the skill.

        SECURITY: step 6 runs arbitrary LLM-generated Python *in-process* with
        full interpreter privileges. The AST check is a deny-list and is
        defeatable (e.g. via attribute-chain escapes), so it is defence-in-depth
        only — never a sandbox. Therefore exec is gated behind ``allow_exec``,
        which a caller must set only after a human has reviewed the returned
        code. With the default ``allow_exec=False`` the code is written to disk
        for review but is NOT executed or registered.

        Returns the generated code (always), so a caller can show it for review
        before re-invoking with ``allow_exec=True``.
        """
        # Reject path-traversal / module-injection names BEFORE any path use.
        _validate_skill_name(name)

        # Build the generation prompt
        prompt = self._build_generation_prompt(description, name)

        try:
            raw_response = self._client.single_turn(
                prompt=prompt,
                system=SKILL_AUTHOR_SYSTEM_PROMPT,
            )
        except Exception as e:
            raise RuntimeError(f"LLM code generation failed: {e}") from e

        # Extract Python code from response (may be wrapped in markdown fences)
        code = self._extract_code(raw_response)

        # AST safety check
        violations = self._ast_safety_check(code)
        if violations:
            raise ValueError(
                f"Generated code failed safety check:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )
        # Note: _ast_safety_check already calls ast.parse and checks SyntaxError

        # Write to custom skills directory
        _CUSTOM_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        skill_file = _CUSTOM_SKILLS_DIR / f"{name}.py"
        # Defence-in-depth: even with a validated name, confirm the resolved
        # path stays inside the custom skills dir before writing.
        if skill_file.resolve().parent != _CUSTOM_SKILLS_DIR.resolve():
            raise ValueError(
                f"Refusing to write skill outside custom dir: {skill_file}"
            )
        skill_file.write_text(code, encoding="utf-8")
        logger.info("Wrote generated skill to %s", skill_file)

        if not allow_exec:
            logger.warning(
                "Generated skill written to %s but NOT executed/registered "
                "(allow_exec=False). Review the code, then re-invoke with "
                "allow_exec=True to register it.",
                skill_file,
            )
            return code

        # Import and register — runs arbitrary LLM code in-process. Gated above.
        try:
            module_name = f"mast.skills.custom.{name}"
            spec = importlib.util.spec_from_file_location(module_name, str(skill_file))
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec for {skill_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find and register BaseSkill subclasses in the module
            registered = 0
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and hasattr(obj, "metadata")
                    and obj.__module__ == module_name
                    and attr_name != "BaseSkill"
                ):
                    self._registry.register(obj)
                    registered += 1
                    logger.info("Registered generated skill: %s", attr_name)

            if registered == 0:
                logger.warning("No BaseSkill subclasses found in generated code")

        except Exception as e:
            logger.error("Failed to import/register generated skill: %s", e)
            raise RuntimeError(
                f"Generated code written to {skill_file} but import failed: {e}"
            ) from e

        return code

    def _ast_safety_check(self, code: str) -> list[str]:
        """Check generated code AST for dangerous patterns.

        NOTE: this is a best-effort DENY-LIST, NOT a sandbox. A determined
        adversary can still craft escapes; it only raises the bar. The real
        gate is the ``allow_exec`` human-confirmation in ``create_skill``.

        Forbidden:
        - import os/subprocess/sys/shutil and other dangerous modules
        - eval(), exec(), compile(), __import__()
        - globals(), locals(), getattr(), setattr(), delattr(), vars(), dir()
        - open() with write modes
        - breakpoint(), exit(), quit()
        - ANY dunder attribute access / name (blocks the classic
          ``().__class__.__bases__[0].__subclasses__()`` sandbox escape and
          ``obj.__dict__`` / ``__globals__`` / ``__builtins__`` reach-arounds)

        Returns list of violation messages (empty means safe).
        """
        violations: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Code has syntax errors: {e}"]

        for node in ast.walk(tree):
            # Block dunder attribute access (.__class__, .__bases__, .__globals__…)
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    violations.append(
                        f"Forbidden dunder attribute access: '.{node.attr}' "
                        f"(line {node.lineno})"
                    )
            # Block bare dunder name references (e.g. __builtins__, __loader__)
            elif isinstance(node, ast.Name):
                if node.id.startswith("__") and node.id.endswith("__"):
                    violations.append(
                        f"Forbidden dunder name: '{node.id}' "
                        f"(line {node.lineno})"
                    )

            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split(".")[0]
                    if top_module in _FORBIDDEN_MODULES:
                        violations.append(
                            f"Forbidden import: '{alias.name}' "
                            f"(line {node.lineno})"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    top_module = node.module.split(".")[0]
                    if top_module in _FORBIDDEN_MODULES:
                        violations.append(
                            f"Forbidden import from: '{node.module}' "
                            f"(line {node.lineno})"
                        )

            # Check function calls
            elif isinstance(node, ast.Call):
                func_name = _get_call_name(node.func)
                if func_name in _FORBIDDEN_CALLS:
                    violations.append(
                        f"Forbidden call: '{func_name}()' "
                        f"(line {node.lineno})"
                    )

                # Check open() with write mode
                if func_name == "open" and len(node.args) >= 2:
                    mode_arg = node.args[1]
                    if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                        if any(c in mode_arg.value for c in "wax"):
                            violations.append(
                                f"Forbidden: open() with write mode '{mode_arg.value}' "
                                f"(line {node.lineno})"
                            )
                # Also check open() with mode keyword
                if func_name == "open":
                    for kw in node.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            if isinstance(kw.value.value, str) and any(
                                c in kw.value.value for c in "wax"
                            ):
                                violations.append(
                                    f"Forbidden: open() with write mode "
                                    f"'{kw.value.value}' (line {node.lineno})"
                                )

        return violations

    def _build_generation_prompt(self, description: str, name: str) -> str:
        """Build the prompt for skill code generation."""
        # Gather existing skill names as reference
        existing_skills = self._registry.list_skills()
        skill_names = [m.name for m in existing_skills]

        return (
            f"Generate a MAST skill with the following specification:\n\n"
            f"- Skill name: {name}\n"
            f"- Description: {description}\n"
            f"- Safety level: CONFIRM (default for generated skills)\n\n"
            f"Existing registered skills for reference: {skill_names}\n\n"
            f"Generate ONLY the Python code. No explanations or markdown fences.\n"
            f"The code must define exactly one class that inherits from BaseSkill "
            f"and implements metadata() and execute() methods."
        )

    @staticmethod
    def _extract_code(response: str) -> str:
        """Extract Python code from LLM response, stripping markdown fences if present."""
        # Try to extract from ```python ... ``` blocks
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        # If no fences, assume the entire response is code
        return response.strip()


def _get_call_name(node: ast.expr) -> str:
    """Extract the function name from a Call node's func attribute."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
