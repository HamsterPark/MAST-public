"""Validation framework for the MAST Admin interface."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ValidationError:
    """A single validation error."""
    field: str
    message: str


@dataclass
class ValidationResult:
    """Aggregated validation result."""
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def add(self, error: ValidationError | None) -> None:
        """Add an error if it is not ``None``."""
        if error is not None:
            self.errors.append(error)

    def extend(self, errors: list[ValidationError]) -> None:
        """Add multiple errors."""
        self.errors.extend(errors)

    def to_markdown(self) -> str:
        """Format errors as a Markdown bullet list."""
        if not self.errors:
            return ""
        lines = [f"- **{e.field}**: {e.message}" for e in self.errors]
        return "\n".join(lines)


# ── Validator functions ──────────────────────────────────────────────

def validate_min_lt_max(
    min_val: float | None,
    max_val: float | None,
    field: str,
) -> ValidationError | None:
    """Check that *min_val* < *max_val* when both are given."""
    if min_val is not None and max_val is not None and min_val >= max_val:
        return ValidationError(
            field, f"最小值 ({min_val:g}) 必须小于最大值 ({max_val:g})"
        )
    return None


def validate_json_string(text: str, field: str) -> ValidationError | None:
    """Check that *text* is valid JSON."""
    if not text:
        return None
    try:
        json.loads(text)
        return None
    except json.JSONDecodeError as exc:
        return ValidationError(field, f"JSON 格式无效: {exc}")


def validate_json_array(text: str, field: str) -> ValidationError | None:
    """Check that *text* is a valid JSON array."""
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return ValidationError(field, "必须是 JSON 数组")
        return None
    except json.JSONDecodeError as exc:
        return ValidationError(field, f"JSON 格式无效: {exc}")


def validate_not_empty(val: str | None, field: str) -> ValidationError | None:
    """Check that *val* is not empty or whitespace."""
    if not val or (isinstance(val, str) and not val.strip()):
        return ValidationError(field, "不能为空")
    return None


def validate_in_set(
    val: str, valid: set[str], field: str,
) -> ValidationError | None:
    """Check that *val* is a member of *valid*."""
    if val and val not in valid:
        return ValidationError(field, f"无效值: {val}")
    return None


def validate_skill_names(
    names: list[str], valid_names: set[str],
) -> list[ValidationError]:
    """Return errors for skill names not in the registry."""
    return [
        ValidationError(name, "技能未注册")
        for name in names if name and name not in valid_names
    ]
