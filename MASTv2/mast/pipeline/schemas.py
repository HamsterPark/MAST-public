"""Pydantic message models for orchestrator <-> MAST communication."""

from __future__ import annotations

from pydantic import BaseModel


class MissionRequest(BaseModel):
    """Incoming instruction from orchestrator."""

    instruction: str
    experiment_name: str = ""
    priority: str = "normal"  # "normal", "urgent"
    metadata: dict = {}


class MissionResponse(BaseModel):
    """Response back to orchestrator."""

    success: bool
    results: list[dict] = []  # List of SkillResult as dicts
    summary: str = ""
    experiment_id: str = ""
    error: str = ""


class SkillRequest(BaseModel):
    """Direct skill execution request."""

    skill_name: str
    parameters: dict = {}
    approval_source: str = "auto"


class StatusResponse(BaseModel):
    """System status response."""

    connected: bool
    instrument_state: dict = {}
    active_experiment: str | None = None
    available_skills: list[str] = []
    environment: dict = {}  # sensor_name -> SensorReading as dict
