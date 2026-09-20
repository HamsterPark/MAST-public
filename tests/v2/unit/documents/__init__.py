"""Tests for mast.documents.

This package marker is required: ``tests/v2/memory/test_store.py`` already exists,
and without ``__init__.py`` here pytest's rootdir-relative import puts both files
in the top-level ``test_store`` module namespace and aborts collection with an
"import file mismatch". ``tests/v2/unit`` and its sibling packages carry the same
marker for the same reason.
"""
