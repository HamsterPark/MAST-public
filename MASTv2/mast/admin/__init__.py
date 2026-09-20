"""mast.admin — config override JSON registry.

v2's safety_mw.py reads the override JSON files through this lightweight
ConfigOverrideRegistry, so edits made in the admin UI take effect on v2
agents at the next reload.

(The v1 tree this package was originally vendored from was archived out of
the repo on 2026-06-01; the sibling modules here — html_builders / parsers /
reload_wiring / validators — arrived after that.)
"""
