"""Run the same setup mast2.spec does, before PyInstaller's Analysis.

Why this exists: PyInstaller's Analysis subprocess puts the SPEC DIRECTORY on
``sys.path``, so a bare ``import mast`` can resolve to the wrong tree. mast2.spec
strips REPO_ROOT, prepends ``MASTv2/``, wipes stale ``mast.*`` from
``sys.modules`` and then pre-imports the packages that must cache as v2. This
script reproduces that without paying for a full PyInstaller run.

The pre-import list is read from the packaging specification so the checker
and the build use the same module names. Two checks prevent drift:

* the module list is no longer duplicated here — it is PARSED OUT OF
  ``mast2.spec``, the file whose behaviour this script is supposed to mirror; and
* ``tests/v2/unit/test_spec_preimport_parity.py`` pins that every name in that
  list still resolves (``importlib.util.find_spec``), so the next rename fails a
  test instead of rotting in a script.

Exit code 0 = the v2 tree resolves and every pre-imported module is real.
"""

import importlib
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MASTV2_ROOT = os.path.join(REPO_ROOT, "MASTv2")
SPEC_PATH = os.path.join(REPO_ROOT, "mast2.spec")

# The spec's force-import block: `for _modname in (\n  "mast.x", "mast.y", ...\n):`
_PREIMPORT_BLOCK = re.compile(
    r"for\s+_modname\s+in\s+\((?P<body>.*?)\):", re.DOTALL
)


def spec_preimport_modules(spec_path: str = SPEC_PATH) -> list[str]:
    """The module names mast2.spec force-imports, read from the spec itself.

    Parsed rather than imported: the spec references ``SPEC``, which only
    PyInstaller injects. Raises if the block cannot be found — a silent empty
    list would turn this checker into a no-op, which is the failure mode the
    module docstring is about.
    """
    with open(spec_path, encoding="utf-8") as fh:
        source = fh.read()
    match = _PREIMPORT_BLOCK.search(source)
    if match is None:
        raise SystemExit(
            f"Could not find the `for _modname in (...)` pre-import block in "
            f"{spec_path}. If the spec was restructured, update this parser — "
            f"do not leave it returning nothing."
        )
    names = re.findall(r'"([^"]+)"', match.group("body"))
    if not names:
        raise SystemExit(f"Pre-import block in {spec_path} parsed to zero modules.")
    return names


def main() -> int:
    # Same sys.path surgery as the spec: REPO_ROOT out, MASTv2 first.
    sys.path[:] = [
        p for p in sys.path if os.path.abspath(p) != os.path.abspath(REPO_ROOT)
    ]
    if MASTV2_ROOT not in sys.path:
        sys.path.insert(0, MASTV2_ROOT)

    print("sys.path[:5]:", sys.path[:5])

    import mast

    print("mast.__file__:", mast.__file__)
    if "MASTv2" not in (mast.__file__ or "").replace("\\", "/"):
        print(f"WRONG TREE: {mast.__file__}", file=sys.stderr)
        return 1

    modules = spec_preimport_modules()
    print(f"spec pre-imports {len(modules)} modules")
    failed: list[tuple[str, str]] = []
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — report all, then fail once
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        path = (getattr(mod, "__file__", "") or "").replace("\\", "/")
        # Namespace packages have no __file__; only check the ones that do.
        if path and "MASTv2" not in path:
            failed.append((name, f"resolved to the wrong tree: {path}"))

    if failed:
        print(f"\n{len(failed)} module(s) failed:", file=sys.stderr)
        for name, why in failed:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
