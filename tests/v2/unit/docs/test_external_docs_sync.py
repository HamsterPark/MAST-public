"""The plugin's copy of the external guide is the guide, byte for byte.

``docs/external/{zh,en}/`` is the source; the Claude Code plugin ships a copy in
``integrations/claude-code/skills/mast-operator/references/`` because a plugin
is installed as a directory and cannot reach outside it. A copy that drifts is
worse than no copy: the agent reads the stale one. ``scripts/sync_external_docs.py``
makes the copy; these tests check it independently of that script.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (same bootstrap as test_wrap_skill_minimal) ──
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import re
import subprocess
import urllib.parse

SRC = _REPO / "docs" / "external"
DST = _REPO / "integrations" / "claude-code" / "skills" / "mast-operator" / "references"
SCRIPT = _REPO / "scripts" / "sync_external_docs.py"
LANGS = ("zh", "en")

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\s*\)")
_REF_DEF = re.compile(r"^ {0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
_AUTOLINK = re.compile(r"<([a-zA-Z][a-zA-Z0-9+.-]*:[^>\s]+)>")
_BARE_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>()\[\]`'\"，。；、）]+")


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return p.as_posix()


def _links(text: str) -> list[str]:
    """Link targets outside code: inline links, reference definitions, autolinks and
    bare URLs (GitHub renders those as links too)."""
    out: list[str] = []
    fence = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        line = _INLINE_CODE.sub(" ", line)
        ref = _REF_DEF.match(line)
        if ref:
            out.append(ref.group(1))
            continue
        for rx in (_INLINE_LINK, _AUTOLINK):
            out.extend(rx.findall(line))
            line = rx.sub(" ", line)
        out.extend(u.rstrip(".,;:!?") for u in _BARE_URL.findall(line))
    return out


def _expected_files() -> dict[str, set[str]]:
    assert SRC.is_dir(), "docs/external does not exist yet: the bilingual guide is not written"
    return {lang: {p.name for p in (SRC / lang).glob("*.md") if p.is_file()} for lang in LANGS}


def test_sync_script_check_passes():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True,
                          text=True, timeout=60, cwd=str(_REPO))
    assert proc.returncode == 0, (
        f"sync_external_docs.py --check exited {proc.returncode}:\n{proc.stdout}{proc.stderr}\n"
        "Fix: python scripts/sync_external_docs.py")


def test_references_are_byte_copies_of_the_guide_without_orphans():
    expected = _expected_files()
    problems: list[str] = []
    for lang in LANGS:
        assert expected[lang], f"docs/external/{lang} has no .md files"
        for name in sorted(expected[lang]):
            src, dst = SRC / lang / name, DST / lang / name
            if not dst.is_file():
                problems.append(f"missing copy {_rel(dst)}")
            elif dst.read_bytes() != src.read_bytes():
                problems.append(f"{_rel(dst)} differs from {_rel(src)}")
    if DST.is_dir():
        for p in sorted(DST.rglob("*")):
            if p.is_file():
                rel = p.relative_to(DST).parts
                if len(rel) != 2 or rel[0] not in LANGS or rel[1] not in expected[rel[0]]:
                    problems.append(f"orphan {_rel(p)}")
    assert not problems, "\n".join(problems + ["Fix: python scripts/sync_external_docs.py"])


def test_reference_links_stay_inside_the_guide_or_use_https():
    files = {p.resolve() for lang in LANGS for p in (DST / lang).glob("*.md") if p.is_file()}
    assert files, f"{_rel(DST)} has no guide files; run python scripts/sync_external_docs.py"
    bad: list[str] = []
    for f in sorted(files):
        for target in _links(f.read_text(encoding="utf-8")):
            if target.startswith("https://") or target.startswith("#"):
                continue
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("/"):
                bad.append(f"{_rel(f)}: {target} (links must be https:// or relative inside the "
                           "guide; put local addresses in `inline code`)")
                continue
            path = urllib.parse.unquote(target.split("#", 1)[0].split("?", 1)[0])
            if (f.parent / path).resolve() not in files:
                bad.append(f"{_rel(f)}: {target} (not a file of the guide)")
    assert not bad, "\n".join(bad)


def test_the_sync_script_copies_prunes_and_checks(tmp_path):
    """The script itself, on a scratch tree: copy, orphan removal, --check exit codes."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    for lang in LANGS:
        (src / lang).mkdir(parents=True)
        # CRLF and non-ASCII on purpose: the copy must not normalize anything
        (src / lang / "README.md").write_bytes(f"# {lang}\r\n扫描隧道显微镜\n".encode("utf-8"))
    (dst / "zh").mkdir(parents=True)
    (dst / "zh" / "old.md").write_text("stale", encoding="utf-8")
    (dst / "stray").mkdir()
    (dst / "stray" / "x.txt").write_text("stray", encoding="utf-8")

    def run(*extra):
        return subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst),
                               *extra], capture_output=True, text=True, timeout=60)

    before = run("--check")
    assert before.returncode == 1 and "orphan" in before.stdout and "out of sync" in before.stdout
    assert (dst / "zh" / "old.md").exists(), "--check must not write"
    assert run().returncode == 0
    for lang in LANGS:
        assert (dst / lang / "README.md").read_bytes() == (src / lang / "README.md").read_bytes()
    assert not (dst / "zh" / "old.md").exists() and not (dst / "stray").exists()
    assert run("--check").returncode == 0
    (src / "en" / "README.md").write_bytes(b"# en changed\n")
    assert run("--check").returncode == 1
    (src / "zh" / "README.md").unlink()
    (src / "zh").rmdir()
    assert run("--check").returncode == 2
