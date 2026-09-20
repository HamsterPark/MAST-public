"""Top-level test conftest: mock nanonis_spm so tests run without hardware."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# pyarrow must load before the langchain/torch stack (see mast/__init__.py):
# otherwise a full-suite run access-violates the moment any test builds a
# pandas DataFrame after an agent-graph test has imported langchain.agents.
try:
    import pyarrow  # noqa: F401
except Exception:
    pass

# Mock nanonis_spm before any mast imports pull it in
if "nanonis_spm" not in sys.modules:
    nanonis_mock = MagicMock()
    sys.modules["nanonis_spm"] = nanonis_mock
    sys.modules["nanonis_spm.Nanonis"] = nanonis_mock.Nanonis
