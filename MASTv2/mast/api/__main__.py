"""Dev runner: ``.venv-v2-py313/Scripts/python.exe -m mast.api``

Boots the read-only API standalone on 127.0.0.1:7870 (clear of Gradio 7862/7863
and OTA 8766). Use this during Phase 2/3 to develop the frontend against real
config/settings data without the full Gradio stack.

Env:
  MAST_API_HOST   (default 127.0.0.1)
  MAST_API_PORT   (default 7870)
  MAST_API_RELOAD (set to 1 for autoreload)
"""

from __future__ import annotations

import logging
import os


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    host = os.environ.get("MAST_API_HOST", "127.0.0.1")
    port = int(os.environ.get("MAST_API_PORT", "7870"))
    reload = os.environ.get("MAST_API_RELOAD", "") in ("1", "true", "True")
    # Live mode wires the real core (registry + experiment store) so the API
    # serves real data — this is what the cutover launcher uses. Default OFF so
    # pure frontend dev stays fast + degraded. `reload` forces factory/standalone.
    live = os.environ.get("MAST_API_LIVE", "") in ("1", "true", "True")

    if live and not reload:
        from mast.api.app import create_app
        from mast.api.bootstrap import build_live_context

        app = create_app(context=build_live_context())
        uvicorn.run(app, host=host, port=port)
    else:
        # factory=True → uvicorn calls create_app() itself (and can reload it).
        uvicorn.run("mast.api.app:create_app", host=host, port=port, reload=reload, factory=True)


if __name__ == "__main__":
    main()
