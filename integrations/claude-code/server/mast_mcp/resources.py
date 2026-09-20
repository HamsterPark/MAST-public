"""The operator guide as MCP resources: ``mast://guide/{lang}/{file}``.

The files live in the plugin at ``skills/mast-operator/references/{zh,en}/``
(copied there byte for byte from ``docs/external/`` by
``scripts/sync_external_docs.py``). A missing or empty directory simply lists
nothing. ``MAST_GUIDE_DIR`` points elsewhere (used by the tests).
"""
from __future__ import annotations

import os
import re

LANGS = ("en", "zh")
URI_PREFIX = "mast://guide/"
_FILE_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.md$")
_HEADING_RX = re.compile(r"^#\s+(.+?)\s*#*\s*$")


def default_guide_dir() -> str:
    # <plugin>/server/mast_mcp/resources.py -> <plugin>/skills/mast-operator/references
    plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(plugin_root, "skills", "mast-operator", "references")


def _title_of(path: str, fallback: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                m = _HEADING_RX.match(line.strip())
                if m:
                    return m.group(1)
    except (OSError, UnicodeDecodeError):
        pass
    return fallback


class GuideResources:
    def __init__(self, root: str | None = None):
        self.root = os.path.abspath(root or os.environ.get("MAST_GUIDE_DIR") or default_guide_dir())

    def _files(self, lang: str) -> list[str]:
        folder = os.path.join(self.root, lang)
        try:
            names = os.listdir(folder)
        except OSError:
            return []
        return sorted(n for n in names
                      if _FILE_RX.match(n) and os.path.isfile(os.path.join(folder, n)))

    def list_resources(self) -> list[dict]:
        out = []
        for lang in LANGS:
            for name in self._files(lang):
                path = os.path.join(self.root, lang, name)
                label = "English" if lang == "en" else "中文"
                out.append({
                    "uri": f"{URI_PREFIX}{lang}/{name}",
                    "name": f"guide-{lang}-{name[:-3]}",
                    "title": f"{_title_of(path, name)} ({label})",
                    "description": f"MAST operator guide, {label}: {name}",
                    "mimeType": "text/markdown",
                    "size": os.path.getsize(path),
                })
        return out

    def list_templates(self) -> list[dict]:
        return [{
            "uriTemplate": URI_PREFIX + "{lang}/{file}",
            "name": "mast-guide",
            "title": "MAST operator guide",
            "description": ("Bilingual guide for agents operating MAST. lang is 'en' or 'zh'; "
                            "start with README.md."),
            "mimeType": "text/markdown",
        }]

    def read_resource(self, uri: str) -> dict:
        """The contents item for ``uri``; ``KeyError`` if it is not a guide file."""
        if not uri.startswith(URI_PREFIX):
            raise KeyError(uri)
        rest = uri[len(URI_PREFIX):]
        lang, _, name = rest.partition("/")
        if lang not in LANGS or not _FILE_RX.match(name) or name not in self._files(lang):
            raise KeyError(uri)
        path = os.path.join(self.root, lang, name)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError):
            raise KeyError(uri) from None
        return {"uri": uri, "mimeType": "text/markdown", "text": text}
