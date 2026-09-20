"""Force MASTv2/ to the head of sys.path before any mast.* imports in this dir.

Without this, pytest's rootdir-based sys.path injection puts repo-root first,
which makes `import mast` resolve to the v1 tree under ``mast/`` instead of
``MASTv2/mast/``. The parent ``tests/v2/conftest.py`` already attempts this,
but the prepend can be lost when pytest re-orders sys.path during collection.
"""
from __future__ import annotations

import sys
from pathlib import Path

# tests/v2/unit/logging/v2/conftest.py: parents[5] is the repo root.
_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")

while _MASTV2_ROOT in sys.path:
    sys.path.remove(_MASTV2_ROOT)
sys.path.insert(0, _MASTV2_ROOT)

for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]
