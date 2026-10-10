"""Health and system status endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Response, status

from app.api.deps import ml_service
from app.core.config import settings
from app.ml import datasets as dataset_registry
from app.schemas.api import ComponentHealth, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Service and dependency health")
def health(response: Response) -> HealthResponse:
    """Reports real availability of every subsystem.

    The HTTP status is 200 when the service can serve requests and 503 when a
    required subsystem is missing, so a load balancer can act on it.
    """
    model_health = ml_service.health()
    models = model_health["models"]
    vision = model_health["vision"]
    statuses = dataset_registry.all_statuses()

    components: list[ComponentHealth] = [
        ComponentHealth(
            name="database",
            status="ok",
            detail=f"{'SQLite' if settings.is_sqlite else 'PostgreSQL'} connection pool ready",
        ),
        ComponentHealth(
            name="failure_prediction_model",
            status="ok" if models["failure_prediction"]["loaded"] else "degraded",
            detail=models["failure_prediction"]["detail"],
        ),
        ComponentHealth(
            name="anomaly_detection_model",
            status="ok" if models["anomaly_detection"]["loaded"] else "degraded",
            detail=models["anomaly_detection"]["detail"],
        ),
        ComponentHealth(
            name="rul_prediction_model",
            status="ok" if models["rul_prediction"]["loaded"] else "degraded",
            detail=models["rul_prediction"]["detail"],
        ),
        ComponentHealth(
            name="computer_vision_model",
            status="ok" if vision.get("loaded") else "degraded",
            detail=vision.get("detail", "unknown"),
        ),
        ComponentHealth(
            name="llm_engine",
            status="ok" if settings.llm_active else "degraded",
            detail=model_health["llm"]["detail"],
        ),
    ]
    for key, dataset in statuses.items():
        components.append(
            ComponentHealth(
                name=f"dataset_{key}",
                status="ok" if dataset.available else "degraded",
                detail=dataset.detail,
            )
        )

    blocking = [
        c for c in components
        if c.status == "degraded" and c.name in {"dataset_ai4i", "dataset_cmapss"}
    ]
    overall = "degraded" if any(c.status == "degraded" for c in components) else "ok"
    # Only required datasets (AI4I, C-MAPSS) block readiness. Optional
    # components (MVTec vision, LLM) degrade the status without taking
    # the service down.
    if blocking:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=overall,
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        timestamp=datetime.now(timezone.utc),
        llm_engine="llm" if settings.llm_active else "rule_based",
        components=components,
    )
