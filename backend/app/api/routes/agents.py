"""Agentic investigation endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import ValidationError

from app.agents import graph as agent_graph
from app.api.deps import DbSession
from app.core.logging_config import get_logger
from app.schemas.api import AgentInvestigationRequest, InvestigationResponse
from app.services import investigation_service

logger = get_logger(__name__)
router = APIRouter(prefix="/agents", tags=["agents"])


@router.post(
    "/investigate",
    response_model=InvestigationResponse,
    summary="Run the six-agent investigation workflow",
)
def investigate(
    session: DbSession, payload: AgentInvestigationRequest
) -> InvestigationResponse:
    """Runs monitoring -> diagnosis -> RUL -> documentation -> planning -> verification.

    Every action the workflow proposes is created as a ``pending_approval``
    recommendation. Nothing is executed automatically.
    """
    body = payload.model_dump()
    if body.get("reading") is not None:
        try:
            body["reading"] = payload.reading.model_dump()
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

    result = investigation_service.run_investigation(session, body, persist=True)
    if result.get("error") == "no_reading":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "no_reading",
                "message": result["message"],
            },
        )

    record = investigation_service.persist_investigation(session, result)
    result["investigation_id"] = record.id
    result.pop("_state", None)
    result.pop("persisted", None)
    logger.info(
        "Investigation %s verdict=%s steps=%d recommendations=%d",
        record.id, result["verdict"], result["steps_executed"], len(result["recommendations"]),
    )

    return InvestigationResponse(
        investigation_id=result["investigation_id"],
        status=result["status"],
        engine=result["engine"],
        llm_enabled=result["llm_enabled"],
        steps_executed=result["steps_executed"],
        max_steps_reached=result["max_steps_reached"],
        trace=result["trace"],
        reports=result["reports"],
        root_causes=result["root_causes"],
        citations=result["citations"],
        unsupported_claims=result["unsupported_claims"],
        recommendations=[
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "description": r.get("description"),
                "priority": r.get("priority"),
                "action_type": r.get("action_type"),
                "rationale": r.get("rationale"),
                "confidence": r.get("confidence"),
                "estimated_downtime_min": r.get("estimated_downtime_min"),
                "evidence": r.get("evidence"),
                "citations": r.get("citations"),
                "status": r.get("status", "pending_approval"),
                "requires_human_approval": True,
            }
            for r in result["recommendations"]
        ],
        verdict=result["verdict"],
        requires_human_approval=True,
        alerts_created=result["alerts_created"],
        duration_ms=result["duration_ms"],
        started_at=result["started_at"],
        completed_at=result["completed_at"],
    )


@router.get("/catalog", summary="Agent catalogue and workflow topology")
def catalog() -> dict[str, Any]:
    return {
        "agents": agent_graph.agent_catalog(),
        "graph": agent_graph.graph_description(),
        "guarantees": [
            "The graph is acyclic, so no agent can re-enter another agent.",
            "Each node enforces a step budget; exceeding it aborts the run and records "
            "max_steps_reached rather than retrying.",
            "Agents receive measurements as read-only input, so no agent can fabricate a "
            "sensor reading.",
            "Documentation citations are stripped if the cited passage was not in the "
            "retrieved context.",
            "The verification agent always sets requires_human_approval to true.",
        ],
    }


@router.get("/investigations", summary="Recent investigations")
def recent(session: DbSession, limit: int = Query(default=25, ge=1, le=100)) -> dict[str, Any]:
    records = investigation_service.recent_investigations(session, limit=limit)
    return {
        "items": [
            {
                "id": r.id,
                "machine_id": r.machine_id,
                "created_at": r.created_at,
                "completed_at": r.completed_at,
                "status": r.status,
                "engine": r.engine,
                "verdict": r.verdict,
                "steps_executed": r.steps_executed,
                "max_steps_reached": r.max_steps_reached,
                "root_cause_count": len(r.root_causes or []),
                "citation_count": len(r.citations or []),
                "unsupported_claim_count": len(r.unsupported_claims or []),
                "summary": r.final_summary,
            }
            for r in records
        ],
        "total": len(records),
    }


@router.get("/investigations/{investigation_id}", summary="Full investigation record")
def detail(investigation_id: str, session: DbSession) -> dict[str, Any]:
    record = investigation_service.get_investigation(session, investigation_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investigation '{investigation_id}' not found",
        )
    return {
        "investigation_id": record.id,
        "machine_id": record.machine_id,
        "created_at": record.created_at,
        "completed_at": record.completed_at,
        "status": record.status,
        "engine": record.engine,
        "verdict": record.verdict,
        "steps_executed": record.steps_executed,
        "max_steps_reached": record.max_steps_reached,
        "trace": record.trace or [],
        "root_causes": record.root_causes or [],
        "citations": record.citations or [],
        "unsupported_claims": record.unsupported_claims or [],
        "requires_human_approval": record.requires_human_approval,
        "summary": record.final_summary,
    }
