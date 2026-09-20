"""Shared parsing utilities for the MAST Admin interface."""

from __future__ import annotations

from typing import Any


def parse_float(text: str | float | None, field_name: str) -> float:
    """Parse a user-entered float, raising ``ValueError`` with a readable message."""
    if text is None or (isinstance(text, str) and text.strip() == ""):
        raise ValueError(f"'{field_name}' 不能为空")
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ValueError(f"'{field_name}' 不是有效数字: {text}")


def parse_optional_float(s: str) -> float | None:
    """Parse a string to float, returning ``None`` for empty or invalid."""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def coerce_value(s: str, type_str: str) -> Any:
    """Coerce a string value to the appropriate Python type."""
    if not s:
        return None
    if type_str == "float":
        try:
            return float(s)
        except ValueError:
            return s
    if type_str == "int":
        try:
            return int(s)
        except ValueError:
            return s
    if type_str == "bool":
        return s.lower() in ("true", "1", "yes")
    return s


def dataframe_to_rows(table_data: Any) -> list[list[Any]]:
    """Normalize a ``gr.Dataframe`` handler value into a plain ``list[list]``.

    Gradio passes the value of an ``interactive`` ``gr.Dataframe`` to a handler
    as one of several shapes depending on version/edit state:

    - a ``pandas.DataFrame`` (most common in Gradio 6.x),
    - a dict ``{"headers": [...], "data": [[...], ...]}`` (component payload),
    - a plain ``list[list]`` (initial value / some code paths),
    - ``None`` (empty).

    Iterating a ``DataFrame`` yields its *column names*, not rows, and testing
    its truth value raises ``ValueError`` — both silently corrupt overrides or
    crash the save button. This helper always returns row-major ``list[list]``.
    """
    if table_data is None:
        return []
    # pandas.DataFrame — use .values.tolist() to get rows (NOT iteration,
    # which yields column names).
    values = getattr(table_data, "values", None)
    if values is not None and hasattr(values, "tolist"):
        return [list(r) for r in values.tolist()]
    # Component payload dict: {"headers": [...], "data": [[...], ...]}
    if isinstance(table_data, dict):
        data = table_data.get("data", [])
        return [list(r) for r in data] if data else []
    # Plain list of rows.
    if isinstance(table_data, (list, tuple)):
        return [list(r) if isinstance(r, (list, tuple)) else [r] for r in table_data]
    return []


def parse_comma_list(text: str) -> list[str]:
    """Split comma-separated text into trimmed, non-empty strings."""
    return [s.strip() for s in text.split(",") if s.strip()]


def parse_newline_list(text: str) -> list[str]:
    """Split newline-separated text into trimmed, non-empty strings."""
    return [line.strip() for line in (text or "").split("\n") if line.strip()]


def fmt_sci(value: float, unit: str) -> str:
    """Format a value with unit in human-friendly scientific notation."""
    abs_val = abs(value)
    if abs_val == 0:
        return f"0 {unit}"

    si_prefixes = [
        (1e-15, "f"),
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "\u03bc"),
        (1e-3, "m"),
        (1, ""),
        (1e3, "k"),
        (1e6, "M"),
    ]

    for scale, prefix in si_prefixes:
        if abs_val < scale * 1000:
            scaled = value / scale
            if scaled == int(scaled):
                return f"{int(scaled)} {prefix}{unit}"
            return f"{scaled:g} {prefix}{unit}"

    return f"{value:g} {unit}"
