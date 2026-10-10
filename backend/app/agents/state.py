"""Typed state shared by every FactoryMind agent.

The state is the single source of truth for an investigation. Agents read from
it and append to it; no agent reads another agent's private reasoning and no
agent can write a measurement that was not produced by the ML layer. That is
what prevents fabricated sensor readings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

Engine = Literal["llm", "rule_based", "hybrid"]
AgentStatus = Literal["ok", "warning", "skipped", "blocked"]


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


def _extend_unique(left: list[Any], right: list[Any]) -> list[Any]:
    seen = {repr(item) for item in left}
    return left + [item for item in right if repr(item) not in seen]


class InvestigationState(TypedDict, total=False):
    """Graph state.

    ``measurements`` is populated only by the ML layer and is treated as
    read-only for every downstream agent.
    """

    request_id: str
    machine_id: str | None
    machine_name: str | None
    triggered_alert_id: str | None
    engine: Engine

    # Raw operator-supplied telemetry (never invented by an agent).
    readings: list[dict[str, Any]]

    # ML outputs, injected by the service layer before the graph runs.
    measurements: dict[str, Any]

    # Per-agent outputs.
    trace: Annotated[list[dict[str, Any]], _append]
    reports: Annotated[list[dict[str, Any]], _append]
    alerts: Annotated[list[dict[str, Any]], _append]
    citations: Annotated[list[dict[str, Any]], _extend_unique]
    root_causes: Annotated[list[dict[str, Any]], _append]
    recommendations: Annotated[list[dict[str, Any]], _append]
    unsupported_claims: Annotated[list[dict[str, Any]], _append]
    knowledge_answers: Annotated[list[dict[str, Any]], _append]

    # Loop control.
    step: int
    max_steps: int
    max_steps_reached: bool
    agents_ran: Annotated[list[str], _append]
    skipped_agents: Annotated[list[str], _append]

    started_at: str
    completed_at: str | None
    verdict: str | None
    requires_human_approval: bool


class AgentReport(BaseModel):
    agent: str
    role: str
    status: AgentStatus
    engine: Engine
    summary: str
    findings: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = None
    caveats: list[str] = Field(default_factory=list)


class StepTrace(BaseModel):
    step: int
    agent: str
    action: str
    summary: str
    status: AgentStatus
    engine: Engine = "rule_based"
    duration_ms: float | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


def new_state(
    request_id: str,
    max_steps: int = 24,
    engine: Engine = "rule_based",
    machine_id: str | None = None,
    machine_name: str | None = None,
    readings: list[dict[str, Any]] | None = None,
    measurements: dict[str, Any] | None = None,
    triggered_alert_id: str | None = None,
) -> InvestigationState:
    return InvestigationState(
        request_id=request_id,
        machine_id=machine_id,
        machine_name=machine_name,
        triggered_alert_id=triggered_alert_id,
        engine=engine,
        readings=readings or [],
        measurements=measurements or {},
        trace=[],
        reports=[],
        alerts=[],
        citations=[],
        root_causes=[],
        recommendations=[],
        unsupported_claims=[],
        knowledge_answers=[],
        step=0,
        max_steps=max_steps,
        max_steps_reached=False,
        agents_ran=[],
        skipped_agents=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        verdict=None,
        requires_human_approval=True,
    )
