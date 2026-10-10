"""Maintenance workflow, alerts, analytics and dashboard endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import ValidationError
from sqlalchemy import select

from app.api.deps import DbSession, InvalidTransition, invalid_transition
from app.core.logging_config import get_logger
from app.models import Alert, MaintenanceHistory
from app.schemas.api import (
    AlertRead,
    AnalyticsResponse,
    ApprovalRequest,
    CompletionRequest,
    DashboardSummary,
    MachineComparison,
    ModelMetricsList,
    RecommendationRead,
)
from app.services import dashboard_service, maintenance_service, twin_service

logger = get_logger(__name__)
router = APIRouter(tags=["operations"])


# ---------------------------------------------------------------------------
# Dashboard & analytics
# ---------------------------------------------------------------------------
@router.get("/dashboard/summary", response_model=DashboardSummary, summary="Executive dashboard")
def dashboard_summary(session: DbSession) -> Any:
    return dashboard_service.dashboard_summary(session)


@router.get("/analytics", response_model=AnalyticsResponse, summary="Analytics and reports")
def analytics(session: DbSession) -> Any:
    return dashboard_service.analytics(session)


@router.get("/models/metrics", response_model=ModelMetricsList, summary="Model evaluation metrics")
def model_metrics(session: DbSession) -> Any:
    return dashboard_service.model_metrics(session)


@router.get("/dashboard/comparison", response_model=MachineComparison, summary="Machine comparison")
def comparison(session: DbSession) -> Any:
    machines, _ = dashboard_service.list_machines(session, limit=200)
    items = []
    for machine in machines:
        health = dashboard_service.machine_health(session, machine.id)
        reading = dashboard_service.latest_reading(session, machine.id)
        items.append(
            {
                "id": machine.id,
                "asset_id": machine.asset_id,
                "name": machine.name,
                "machine_type": machine.machine_type,
                "location": machine.location,
                **health,
                "latest_reading": (
                    {
                        "air_temperature_k": reading.air_temperature_k,
                        "process_temperature_k": reading.process_temperature_k,
                        "rotational_speed_rpm": reading.rotational_speed_rpm,
                        "torque_nm": reading.torque_nm,
                        "tool_wear_min": reading.tool_wear_min,
                        "recorded_at": reading.recorded_at.isoformat() if reading.recorded_at else None,
                    }
                    if reading else None
                ),
            }
        )
    return MachineComparison(items=items, total=len(items))


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------
@router.get(
    "/maintenance/recommendations",
    response_model=list[RecommendationRead],
    summary="List maintenance recommendations",
)
def recommendations(
    session: DbSession,
    status_filter: str | None = Query(default=None, alias="status"),
    machine_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> Any:
    return maintenance_service.list_recommendations(
        session, status=status_filter, machine_id=machine_id, limit=limit
    )


@router.get("/maintenance/workflow", summary="Approval workflow definition")
def workflow() -> dict[str, Any]:
    return maintenance_service.workflow_description()


@router.get("/maintenance/history", summary="Maintenance history")
def history(
    session: DbSession,
    machine_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    return [
        {
            "id": record.id,
            "machine_id": record.machine_id,
            "recommendation_id": record.recommendation_id,
            "event_type": record.event_type,
            "performed_by": record.performed_by,
            "notes": record.notes,
            "downtime_min": record.downtime_min,
            "cost": record.cost,
            "performed_at": record.performed_at,
        }
        for record in maintenance_service.history(session, machine_id=machine_id, limit=limit)
    ]


def _load_recommendation(session: DbSession, recommendation_id: str) -> Any:
    from app.models import MaintenanceRecommendation

    record = session.get(MaintenanceRecommendation, recommendation_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recommendation '{recommendation_id}' not found",
        )
    return record


@router.post(
    "/maintenance/{recommendation_id}/approve",
    response_model=RecommendationRead,
    summary="Approve a recommendation (human approval gate)",
)
def approve(
    recommendation_id: str, session: DbSession, payload: ApprovalRequest
) -> Any:
    record = _load_recommendation(session, recommendation_id)
    try:
        maintenance_service.approve(
            session, record, approver=payload.approver, comment=payload.comment,
            downtime=payload.override_downtime_min,
        )
    except InvalidTransition as exc:
        raise invalid_transition(exc)
    return RecommendationRead.model_validate(record)


@router.post(
    "/maintenance/{recommendation_id}/reject",
    response_model=RecommendationRead,
    summary="Reject a recommendation",
)
def reject(recommendation_id: str, session: DbSession, payload: ApprovalRequest) -> Any:
    record = _load_recommendation(session, recommendation_id)
    try:
        maintenance_service.reject(session, record, approver=payload.approver, comment=payload.comment)
    except InvalidTransition as exc:
        raise invalid_transition(exc)
    return RecommendationRead.model_validate(record)


@router.post(
    "/maintenance/{recommendation_id}/start",
    response_model=RecommendationRead,
    summary="Mark approved work as in progress",
)
def start_work(recommendation_id: str, session: DbSession, payload: ApprovalRequest) -> Any:
    record = _load_recommendation(session, recommendation_id)
    try:
        maintenance_service.start(session, record, performed_by=payload.approver, notes=payload.comment)
    except InvalidTransition as exc:
        raise invalid_transition(exc)
    return RecommendationRead.model_validate(record)


@router.post(
    "/maintenance/{recommendation_id}/complete",
    response_model=RecommendationRead,
    summary="Complete maintenance work",
    responses={409: {"description": "The recommendation was never approved."}},
)
def complete(recommendation_id: str, session: DbSession, payload: CompletionRequest) -> Any:
    """Marks work complete. Refused unless the recommendation was approved first."""
    record = _load_recommendation(session, recommendation_id)
    try:
        maintenance_service.complete(
            session, record, performed_by=payload.performed_by, notes=payload.notes,
            downtime=payload.actual_downtime_min, cost=payload.actual_cost,
        )
    except InvalidTransition as exc:
        raise invalid_transition(exc)
    return RecommendationRead.model_validate(record)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
@router.get("/alerts", response_model=list[AlertRead], summary="List alerts")
def alerts(
    session: DbSession,
    severity: str | None = None,
    unresolved_only: bool = Query(default=False),
    machine_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> Any:
    statement = select(Alert)
    if severity:
        statement = statement.where(Alert.severity == severity)
    if unresolved_only:
        statement = statement.where(Alert.is_resolved.is_(False))
    if machine_id:
        statement = statement.where(Alert.machine_id == machine_id)
    rows = session.scalars(statement.order_by(Alert.created_at.desc()).limit(limit))
    return [AlertRead.model_validate(row) for row in rows]


@router.post(
    "/alerts/{alert_id}/resolve", response_model=AlertRead, summary="Mark an alert resolved"
)
def resolve_alert(alert_id: str, session: DbSession) -> Any:
    record = session.get(Alert, alert_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    record.is_resolved = True
    session.add(record)
    session.flush()
    return AlertRead.model_validate(record)


# ---------------------------------------------------------------------------
# Digital twin
# ---------------------------------------------------------------------------
@router.post("/twin/simulate", summary="Simulate a hypothetical operating point")
def simulate(payload: dict[str, Any]) -> dict[str, Any]:
    """Runs hypothetical parameters through the real models.

    The response is explicitly flagged ``is_simulation: true``. These are model
    outputs on operator-supplied inputs, not machine measurements.
    """
    try:
        validated = twin_simulation_schema(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return twin_service.simulate(validated)


def twin_simulation_schema(payload: dict[str, Any]) -> dict[str, Any]:
    from pydantic import BaseModel, Field

    class TwinSimulation(BaseModel):
        machine_id: str | None = None
        machine_type: str | None = None
        air_temperature_k: float = Field(default=300.5, gt=200, le=400)
        process_temperature_k: float = Field(default=310.6, gt=200, le=400)
        rotational_speed_rpm: float = Field(default=1500.0, ge=0, le=30000)
        torque_nm: float = Field(default=40.0, ge=0, le=200)
        tool_wear_min: float = Field(default=0.0, ge=0, le=500)

    return TwinSimulation(**payload).model_dump()


@router.get("/twin/{machine_id}/snapshot", summary="Measured state for one machine")
def snapshot(machine_id: str, session: DbSession) -> dict[str, Any]:
    if dashboard_service.get_machine(session, machine_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Machine '{machine_id}' not found"
        )
    return twin_service.health_snapshot(session, machine_id)


# ---------------------------------------------------------------------------
# Settings / datasets
# ---------------------------------------------------------------------------
@router.get("/settings", summary="Effective runtime configuration (secrets redacted)")
def runtime_settings() -> dict[str, Any]:
    from app.core.config import settings
    from app.ml import datasets as dataset_registry

    return {
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
            "api_prefix": settings.api_prefix,
            "log_level": settings.log_level,
        },
        "database": {
            "dialect": "sqlite" if settings.is_sqlite else "postgresql",
            "echo": settings.db_echo,
            "url_redacted": settings.database_url.split("@")[-1],
        },
        "llm": {
            "enabled": settings.llm_enabled,
            "active": settings.llm_active,
            "model": settings.openai_model if settings.llm_active else None,
            "api_key_present": bool(settings.openai_api_key),
            "note": "API keys are read from the environment only and are never returned.",
        },
        "rag": {
            "embedding_model": settings.embedding_model,
            "top_k": settings.rag_top_k,
            "chunk_size": settings.rag_chunk_size,
            "chunk_overlap": settings.rag_chunk_overlap,
        },
        "thresholds": {
            "anomaly": settings.anomaly_threshold,
            "risk_watch": settings.risk_watch_threshold,
            "risk_high": settings.risk_high_threshold,
            "risk_critical": settings.risk_critical_threshold,
        },
        "computer_vision": {
            "image_size": settings.vision_image_size,
            "patch_size": settings.vision_patch_size,
            "threshold": settings.vision_threshold,
        },
        "ml_runtime": {"torch_device": settings.torch_device, "random_seed": settings.random_seed},
        "datasets": {k: v.to_dict() for k, v in dataset_registry.all_statuses().items()},
        "paths": {
            "datasets_dir": str(settings.datasets_dir),
            "models_dir": str(settings.models_dir),
            "storage_dir": str(settings.storage_dir),
        },
    }


@router.get("/datasets", summary="Dataset availability and provenance")
def dataset_status() -> dict[str, Any]:
    from app.ml import datasets as dataset_registry

    return {
        "statuses": {k: v.to_dict() for k, v in dataset_registry.all_statuses().items()},
        "manifest": dataset_registry.manifest(),
        "policy": (
            "Each dataset has its own loader, preprocessing pipeline and models. FactoryMind "
            "never merges datasets and never synthesises data to fill a gap."
        ),
    }


_ = MaintenanceHistory
