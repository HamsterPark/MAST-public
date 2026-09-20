"""Cloud sync: upload/download skills to/from a shared registry (stub)."""

from __future__ import annotations


class CloudSync:
    """Upload/download skills to/from shared registry. Stub for future implementation."""

    def __init__(self, registry_url: str = ""):
        self._url = registry_url

    def upload_skill(self, skill_path: str, metadata: dict) -> bool:
        """Upload a custom skill to the shared registry. Returns success."""
        raise NotImplementedError("Cloud sync not yet implemented")

    def download_skill(self, skill_name: str, version: str = "latest") -> str | None:
        """Download a skill from shared registry. Returns local path or None."""
        raise NotImplementedError("Cloud sync not yet implemented")

    def list_remote_skills(self) -> list[dict]:
        """List skills available in the remote registry."""
        raise NotImplementedError("Cloud sync not yet implemented")
