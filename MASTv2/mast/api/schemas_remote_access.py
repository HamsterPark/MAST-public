"""Typed response for GET /api/remote-access (设置 → 远程访问 panel).

Mirrors :mod:`mast.net.tailscale` for the wire. Kept out of the big shared
``schemas.py`` (like schemas_optics / schemas_settings_admin_write) so the
remote-access feature stays self-contained.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TailscaleInfo(BaseModel):
    """Live Tailscale node status (from ``tailscale status --json``)."""
    installed: bool = Field(description="tailscale CLI/daemon found on this machine")
    running: bool = Field(description="daemon BackendState == Running")
    ready: bool = Field(description="installed + running + holding a Tailscale IP")
    backend_state: str = Field(default="", description="Running / Stopped / NeedsLogin / …")
    device_name: str = Field(default="", description="this node's short hostname")
    magic_dns: str = Field(default="", description="MagicDNS FQDN (empty if MagicDNS off)")
    tailnet: str = Field(default="", description="tailnet name, e.g. name.ts.net")
    ips: list[str] = Field(default_factory=list, description="Tailscale IPs (v4 first)")
    hint: str = Field(default="", description="Chinese next-step when not ready")


class RemoteUrl(BaseModel):
    """One address to open on the OTHER computer."""
    label: str
    host: str
    url: str


class RemoteAccessResponse(BaseModel):
    lan_enabled: bool = Field(description="service bound to 0.0.0.0 (remote reachable)")
    scheme: str = Field(description="http | https actually serving")
    port: int = Field(description="port the service is serving on")
    lan_ip: str = Field(default="", description="this machine's local-network IPv4")
    tailscale: TailscaleInfo
    urls: list[RemoteUrl] = Field(
        default_factory=list, description="addresses to open on another computer")
    note: str = Field(default="", description="one-line human guidance for the panel")


__all__ = ["TailscaleInfo", "RemoteUrl", "RemoteAccessResponse"]
