"""MAST gui-level BACKEND helpers (TS-rewrite cutover).

The Gradio UI (app.py + panels) has been DELETED — the UI is now the TypeScript
SPA served by the FastAPI service (mast.api). What remains under this package are
gradio-free backend helpers that the API reuses (builder_api, agents_api,
records_api, settings_store, encyclopedia, route_auth, composite_panel,
dashboard, literature_panel, wishlist_panel, tip_shape_records). These will be
relocated out of ``mast.webui`` in a follow-up; nothing here imports gradio.
"""

from __future__ import annotations
