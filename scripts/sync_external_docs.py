"""Copy the bilingual external guide into the Claude Code plugin.

    docs/external/{zh,en}/*.md  ->  integrations/claude-code/skills/mast-operator/references/{zh,en}/

The copy is byte for byte: the plugin is installed straight from the repository,
so the skill's references must be exactly the files the guide publishes. Files
under ``references/`` without a source are orphans and are deleted.

    python scripts/sync_external_docs.py           # sync
    python scripts/sync_external_docs.py --check   # compare only, write nothing

Exit codes: 0 in sync (or synced), 1 out of sync (``--check``), 2 a source
folder is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "external"
DST = ROOT / "integrations" / "claude-code" / "skills" / "mast-operator" / "references"
LANGS = ("zh", "en")


def plan(src: Path, dst: Path) -> tuple[list[tuple[Path, Path]], list[Path], list[str], int]:
    """``(copies, orphans, problems, expected_count)``; ``copies`` holds (source, target)."""
    problems: list[str] = []
    expected: dict[Path, Path] = {}
    for lang in LANGS:
        folder = src / lang
        if not folder.is_dir():
            problems.append(f"missing source folder {_rel(folder)}")
            continue
        for f in sorted(folder.glob("*.md")):
            if f.is_file():
                expected[dst / lang / f.name] = f
    copies = [(s, d) for d, s in sorted(expected.items())
              if not d.is_file() or d.read_bytes() != s.read_bytes()]
    orphans = sorted(p for p in dst.rglob("*") if p.is_file() and p not in expected) \
        if dst.is_dir() else []
    return copies, orphans, problems, len(expected)


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _prune_empty_dirs(dst: Path) -> None:
    if not dst.is_dir():
        return
    for d in sorted((p for p in dst.rglob("*") if p.is_dir()), key=lambda p: len(p.parts),
                    reverse=True):
        if d.parent == dst and d.name in LANGS:
            continue
        if not any(d.iterdir()):
            d.rmdir()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="compare only; exit 1 when the references are out of sync")
    ap.add_argument("--src", type=Path, default=SRC, help="guide root (default docs/external)")
    ap.add_argument("--dst", type=Path, default=DST, help="plugin references folder")
    args = ap.parse_args(argv)

    copies, orphans, problems, count = plan(args.src, args.dst)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2

    if args.check:
        for s, d in copies:
            print(f"out of sync: {_rel(d)} (source {_rel(s)})")
        for o in orphans:
            print(f"orphan: {_rel(o)}")
        if copies or orphans:
            print("fix: python scripts/sync_external_docs.py", file=sys.stderr)
            return 1
        print(f"in sync: {count} files")
        return 0

    for s, d in copies:
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_bytes(s.read_bytes())
        print(f"copied {_rel(s)} -> {_rel(d)}")
    for o in orphans:
        o.unlink()
        print(f"deleted orphan {_rel(o)}")
    _prune_empty_dirs(args.dst)
    print(f"done: {count} files, {len(copies)} copied, {len(orphans)} orphans deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
