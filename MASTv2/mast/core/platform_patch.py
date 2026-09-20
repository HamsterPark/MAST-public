"""
Workaround for Python 3.14 + Windows: platform._wmi_query() hangs.

On some Windows 11 machines, `platform.machine()` calls `_wmi_query()`
which blocks indefinitely via WMI COM. The original trigger was
`gradio.tunneling` reading `platform.machine()` at module level; Gradio
itself is gone from the shipped app (removed in the TS rewrite), so this
patch now only matters to whatever still pulls in such a module.

Fix: cache the result of `platform.machine()` using `os.environ` lookup
(PROCESSOR_ARCHITECTURE), then monkey-patch `platform.machine` to return
the cached value without touching WMI.

Import this module before any import that reads `platform.machine()` at
module level.
"""

import os
import platform

# Get machine architecture from environment (fast, no WMI)
_MACHINE = os.environ.get("PROCESSOR_ARCHITECTURE", "").lower()
_MACHINE_MAP = {"amd64": "AMD64", "x86": "x86", "arm64": "ARM64"}
_CACHED_MACHINE = _MACHINE_MAP.get(_MACHINE, _MACHINE or "AMD64")

# Also cache platform.system() in case it has similar issues
_CACHED_SYSTEM = "Windows" if os.name == "nt" else platform.system()

_original_machine = platform.machine
_original_system = platform.system


def _fast_machine():
    return _CACHED_MACHINE


def _fast_system():
    return _CACHED_SYSTEM


platform.machine = _fast_machine
platform.system = _fast_system
