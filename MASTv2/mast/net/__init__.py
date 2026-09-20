"""Networking helpers for MAST.

Currently: Tailscale detection + cross-network remote-access URL derivation
(:mod:`mast.net.tailscale`). Kept separate from :mod:`mast.core` so the launcher
(which is not part of the ``mast.core`` runtime) can import it without pulling in
the heavy runtime graph.
"""
