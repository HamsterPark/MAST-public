"""Start the MAST MCP server on stdio.

Claude Code starts it through the plugin's ``.mcp.json``; by hand it is just::

    python run_server.py

Settings come from environment variables (see ``mast_mcp/config.py``):
MAST_URL, MAST_USER, MAST_PASSWORD, MAST_VERIFY_TLS, MAST_ALLOW_REMOTE,
MAST_ACTOR, MAST_FETCH_DIR. MAST_MCP_LOG sets the stderr log level (INFO).

This file stays importable by old Pythons on purpose, so that an interpreter
that is too old prints a clear message instead of a SyntaxError.
"""
import logging
import os
import sys

# MAST's bundled runtime is an embeddable CPython whose ._pth file switches on
# isolated path mode: the script's own directory is NOT on sys.path there.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main():
    if sys.version_info < (3, 10):
        sys.stderr.write("mast-mcp: Python 3.10 or newer is required (this is %s). Point the "
                         "plugin option 'python' at MAST's bundled runtime "
                         "(MASTv2/pyruntime/python.exe) or at a newer Python.\n"
                         % sys.version.split()[0])
        return 2

    # The protocol owns the real stdout. Anything else that prints (a stray
    # print(), a noisy library) goes to stderr instead of corrupting it.
    proto_out = sys.stdout.buffer
    sys.stdout = sys.stderr

    level = os.environ.get("MAST_MCP_LOG", "INFO").strip().upper() or "INFO"
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, level, logging.INFO),
                        format="mast-mcp %(levelname)s %(name)s: %(message)s")

    from mast_mcp import SERVER_NAME, SERVER_TITLE, __version__
    from mast_mcp.config import load_config
    from mast_mcp.resources import GuideResources
    from mast_mcp.stdio_rpc import serve
    from mast_mcp.tools import INSTRUCTIONS, Toolbox

    config = load_config()
    log = logging.getLogger("mast_mcp")
    log.info("starting %s %s with %s", SERVER_NAME, __version__, config.describe())
    if config.problem:
        log.warning("tools will refuse to run: %s", config.problem)

    return serve(
        server_info={"name": SERVER_NAME, "title": SERVER_TITLE, "version": __version__},
        instructions=INSTRUCTIONS,
        tools=Toolbox(config),
        resources=GuideResources(),
        instream=sys.stdin.buffer,
        outstream=proto_out,
    )


if __name__ == "__main__":
    sys.exit(main())
