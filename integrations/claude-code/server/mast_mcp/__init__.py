"""MAST MCP server: lets Claude Code (or any MCP client) drive MAST through its
external agent API (``/api/ext/v1``).

Standard library only, on purpose: the server must start on MAST's bundled
embeddable Python, on the development venv, or on any Python >= 3.10, without
installing a single package.

Layout:

* ``stdio_rpc``  MCP over stdio (JSON-RPC framing and method dispatch). The only
                 module that knows the wire protocol; swapping in an official
                 SDK later means rewriting this file and nothing else.
* ``config``     environment -> settings, including the loopback-only rule.
* ``client``     HTTP to MAST, with every failure turned into a sentence.
* ``tools``      the ``mast_*`` tools.
* ``resources``  the bilingual operator guide as ``mast://guide/{lang}/{file}``.
"""

__version__ = "0.1.0"

SERVER_NAME = "mast"
SERVER_TITLE = "MAST (scanning tunneling microscope)"
