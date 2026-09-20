"""Every document for outside readers exists in Chinese and English, with the same shape.

A translation that silently drops a section, a table or a warning is the usual
way the two languages drift apart. The prose may differ; the skeleton may not:

* the same files in ``docs/external/zh`` and ``docs/external/en``;
* per pair: the same number of headings at each level, the same number of
  tables, the fenced code blocks identical byte for byte and in order, the same
  relative link targets (``zh`` and ``en`` path segments and ``X.zh.md`` /
  ``X.md`` mapped onto each other) and the same external links.

The same check covers the paired root-level files (``X.md`` in English next to
``X.zh.md`` in Chinese) wherever they exist.
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
from collections import Counter

GUIDE = _REPO / "docs" / "external"
MIN_GUIDE_FILES = 8

#: (English, Chinese) pairs outside the guide. A location that does not exist is
#: skipped; a location with only one of the two files is a failure.
ROOT_PAIRS = [
    *((f"scripts/release/public_overlay/{n}.md", f"scripts/release/public_overlay/{n}.zh.md")
      for n in ("AGENTS", "CONTRIBUTING", "SECURITY")),
    *((f"{n}.md", f"{n}.zh.md") for n in ("AGENTS", "CONTRIBUTING", "SECURITY")),
    ("contrib/README.md", "contrib/README.zh.md"),
    ("integrations/claude-code/README.md", "integrations/claude-code/README.zh.md"),
]

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]|$)")
_TABLE_RULE = re.compile(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?")
_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\s*\)")
_REF_DEF = re.compile(r"^ {0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
_AUTOLINK = re.compile(r"<([a-zA-Z][a-zA-Z0-9+.-]*:[^>\s]+)>")
_BARE_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>()\[\]`'\"，。；、）]+")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


class Skeleton:
    """The parts of a markdown file that a translation must keep."""

    def __init__(self, text: str):
        self.headings: Counter = Counter()
        self.tables = 0
        self.code_blocks: list[str] = []
        self.relative: list[str] = []
        self.external: list[str] = []
        fence, block = None, []
        for line in text.splitlines(keepends=True):
            bare = line.rstrip("\r\n")
            m = _FENCE.match(bare)
            if fence:
                block.append(line)
                if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) \
                        and not bare.strip().strip(fence[0]):
                    self.code_blocks.append("".join(block))
                    fence, block = None, []
                continue
            if m:
                fence, block = m.group(1), [line]
                continue
            h = _HEADING.match(bare)
            if h:
                self.headings[len(h.group(1))] += 1
            if "|" in bare and _TABLE_RULE.fullmatch(bare.strip()):
                self.tables += 1
            self._links(_INLINE_CODE.sub(" ", bare))
        if fence:
            self.code_blocks.append("".join(block) + "<unclosed fence>")

    def _links(self, line: str) -> None:
        targets: list[str] = []
        ref = _REF_DEF.match(line)
        if ref:
            targets.append(ref.group(1))
            line = ""
        for rx in (_INLINE_LINK, _AUTOLINK):
            targets.extend(rx.findall(line))
            line = rx.sub(" ", line)
        targets.extend(u.rstrip(".,;:!?") for u in _BARE_URL.findall(line))
        for t in targets:
            if _SCHEME.match(t):
                self.external.append(t)
            else:
                norm = _normalize(t)
                if norm:
                    self.relative.append(norm)


def _normalize(target: str) -> str:
    """zh and en path segments, and X.zh.md versus X.md, map onto one form."""
    path = target.split("#", 1)[0].split("?", 1)[0]
    if not path:
        return ""                                   # same-page anchor: headings differ by language
    parts = ["{lang}" if p in ("zh", "en") else p for p in path.split("/")]
    return re.sub(r"\.zh\.md$", ".md", "/".join(parts))


def _label(p: Path) -> str:
    try:
        return p.resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return p.as_posix()


def _compare(en_path: Path, zh_path: Path) -> list[str]:
    en = Skeleton(en_path.read_text(encoding="utf-8"))
    zh = Skeleton(zh_path.read_text(encoding="utf-8"))
    pair = f"{_label(en_path)} <-> {_label(zh_path)}"
    problems: list[str] = []
    if en.headings != zh.headings:
        problems.append(f"{pair}: headings per level differ (en {dict(sorted(en.headings.items()))}, "
                        f"zh {dict(sorted(zh.headings.items()))})")
    if en.tables != zh.tables:
        problems.append(f"{pair}: {en.tables} table(s) in en, {zh.tables} in zh")
    if en.code_blocks != zh.code_blocks:
        if len(en.code_blocks) != len(zh.code_blocks):
            problems.append(f"{pair}: {len(en.code_blocks)} code block(s) in en, "
                            f"{len(zh.code_blocks)} in zh")
        else:
            first = next(i for i, (a, b) in enumerate(zip(en.code_blocks, zh.code_blocks)) if a != b)
            problems.append(f"{pair}: code block {first + 1} differs (code blocks must be "
                            "identical in both languages):\n  en: "
                            f"{en.code_blocks[first][:200]!r}\n  zh: {zh.code_blocks[first][:200]!r}")
    if sorted(en.relative) != sorted(zh.relative):
        diff = sorted((Counter(en.relative) - Counter(zh.relative)).elements())
        rdiff = sorted((Counter(zh.relative) - Counter(en.relative)).elements())
        problems.append(f"{pair}: relative links differ (only en: {diff}, only zh: {rdiff})")
    if sorted(en.external) != sorted(zh.external):
        diff = sorted((Counter(en.external) - Counter(zh.external)).elements())
        rdiff = sorted((Counter(zh.external) - Counter(en.external)).elements())
        problems.append(f"{pair}: external links differ (only en: {diff}, only zh: {rdiff})")
    return problems


def _guide_files(lang: str) -> set[str]:
    folder = GUIDE / lang
    return {p.name for p in folder.glob("*.md") if p.is_file()} if folder.is_dir() else set()


def test_the_guide_has_the_same_files_in_both_languages():
    zh, en = _guide_files("zh"), _guide_files("en")
    assert len(zh) >= MIN_GUIDE_FILES and len(en) >= MIN_GUIDE_FILES, (
        f"docs/external/zh has {len(zh)} and docs/external/en has {len(en)} guide files; "
        f"expected at least {MIN_GUIDE_FILES} each")
    assert zh == en, f"only in zh: {sorted(zh - en)}; only in en: {sorted(en - zh)}"


def test_each_guide_pair_has_the_same_skeleton():
    names = sorted(_guide_files("zh") & _guide_files("en"))
    assert len(names) >= MIN_GUIDE_FILES, f"only {len(names)} guide pairs to compare"
    problems = [p for n in names for p in _compare(GUIDE / "en" / n, GUIDE / "zh" / n)]
    assert not problems, "\n".join(problems)


def test_paired_root_level_files_have_the_same_skeleton():
    problems: list[str] = []
    checked = 0
    for en_rel, zh_rel in ROOT_PAIRS:
        en_path, zh_path = _REPO / en_rel, _REPO / zh_rel
        if not en_path.exists() and not zh_path.exists():
            continue
        if en_path.exists() != zh_path.exists():
            missing = zh_rel if en_path.exists() else en_rel
            problems.append(f"{missing} is missing: every outside-facing document comes in both "
                            "languages")
            continue
        checked += 1
        problems.extend(_compare(en_path, zh_path))
    assert checked >= 1, "no paired root-level document found; the check would pass vacuously"
    assert not problems, "\n".join(problems)


def test_the_skeleton_sees_what_a_translation_can_lose():
    """Guard for the guard: each kind of drift is caught on a tiny example."""
    base = ("# T\n\n## A\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```text\nrun it\n```\n\n"
            "[next](02-next.md) [lang](../zh/01.md) [site](https://example.org/x)\n")
    same = Skeleton(base)
    assert dict(same.headings) == {1: 1, 2: 1} and same.tables == 1
    assert same.code_blocks == ["```text\nrun it\n```\n"]
    assert same.relative == ["02-next.md", "../{lang}/01.md"]
    assert same.external == ["https://example.org/x"]
    for broken in (base.replace("## A", "A"), base.replace("|---|---|", ""),
                   base.replace("run it", "运行"), base.replace("02-next.md", "03-other.md"),
                   base.replace("https://example.org/x", "https://example.org/y")):
        other = Skeleton(broken)
        assert (dict(other.headings), other.tables, other.code_blocks, other.relative,
                other.external) != (dict(same.headings), same.tables, same.code_blocks,
                                    same.relative, same.external)
