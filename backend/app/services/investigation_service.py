"""Investigation orchestration and persistence.

This is the only place that couples the ML layer to the agent graph. The ML
layer produces the ``measurements`` dictionary; the agents get read-only access
to it. Nothing downstream can add a sensor value that the models did not emit.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents import graph as agent_graph
from app.core.logging_config import get_logger
from app.models import (
    AgentInvestigation,
    Alert,
    MaintenanceRecommendation,
)
from app.services import ml_service

logger = get_logger(__name__)

SEVERITY_ORDER = {"info": 0, "low": 1, "warning": 2, "high": 3, "critical": 4}


def collect_measurements(
    reading: dict[str, Any],
    include_rul: bool = True,
    rul_engine_unit: str = "engine_unit_1",
    rul_subset: str = "FD001",
) -> dict[str, Any]:
    """Run every available model against one reading.

    Each model is optional. A missing model yields a ``model_trained: False``
    entry rather than an exception, so an investigation can still run on
    partial infrastructure and says exactly what is missing.
    """
    measurements: dict[str, Any] = {
        "failure": _safe(lambda: ml_service.failure_model().predict(reading)),
        "anomaly": _safe(lambda: ml_service.anomaly_model().predict(reading)),
        "rul": None,
    }

    if include_rul:
        def _rul() -> dict[str, Any]:
            from app.ml.preprocessing import cmapss

            subset = rul_subset if rul_subset in cmapss.available_subsets() else "FD001"
            frame = cmapss.sequence_for_unit(subset, rul_engine_unit)
            sensors = ml_service.rul_model().sensors
            window = frame[sensors].to_numpy(dtype=float)
            return ml_service.rul_model().predict_window(
                window, subset=subset, engine_unit=rul_engine_unit
            )

        measurements["rul"] = _safe(_rul)
    return measurements


def _safe(fn: Any) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model inference unavailable: %s: %s", type(exc).__name__, exc)
        return {
            "model_trained": False,
            "detail": str(exc),
            "shap_note": "",
            "explanation": "",
            "features": [],
            "feature_deviation": [],
        }


def run_investigation(
    session: Session,
    payload: dict[str, Any],
    persist: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()

    machine_id = payload.get("machine_id")
    machine_name = None
    if machine_id:
        from app.models import Machine

        machine = session.get(Machine, machine_id)
        machine_name = machine.name if machine else None

    reading_payload = payload.get("reading")
    if reading_payload is None:
        reading_payload = _latest_reading_payload(session, machine_id)
    if reading_payload is None:
        return {
            "error": "no_reading",
            "message": (
                "An investigation needs at least one sensor reading. Supply `reading` in "
                "the request, or pass a `machine_id` that already has stored readings."
            ),
        }

    # Explicit caller-supplied overrides are treated as prior observations, not
    # as model output, and are clearly separated in the trace.
    overrides = {
        key: payload[key]
        for key in ("failure_probability", "anomaly_score", "rul_cycles")
        if payload.get(key) is not None
    }
    if overrides.get("failure_probability") is not None and not reading_payload:
        reading_payload = reading_payload or {}

    measurements = collect_measurements(reading_payload)
    if "failure_probability" in overrides:
        measurements["failure"] = {
            **measurements.get("failure", {}),
            "failure_probability": float(overrides["failure_probability"]),
            "risk_category": _band(float(overrides["failure_probability"])),
            "explanation": "Value supplied by the caller, not produced by the trained model.",
            "source": "caller_override",
        }
    if "anomaly_score" in overrides:
        measurements["anomaly"] = {
            **measurements.get("anomaly", {}),
            "anomaly_score": float(overrides["anomaly_score"]),
            "is_anomaly": float(overrides["anomaly_score"]) >= 0.62,
            "source": "caller_override",
        }
    if "rul_cycles" in overrides:
        measurements["rul"] = {
            **(measurements.get("rul") or {}),
            "model_trained": True,
            "predicted_rul": float(overrides["rul_cycles"]),
            "lower_bound": float(overrides["rul_cycles"]) * 0.85,
            "upper_bound": float(overrides["rul_cycles"]) * 1.15,
            "uncertainty_pct": 15.0,
            "engine_unit": "caller_override",
            "model_family": "caller_supplied",
            "model_version": "caller_override",
            "degradation_curve": [],
            "confidence_note": "Supplied by the caller, not produced by a trained model.",
            "source": "caller_override",
        }

    readings = [{**reading_payload}]
    if machine_id:
        readings = _recent_readings(session, machine_id, limit=8) or readings

    state = agent_graph.run_pipeline(
        machine_id=machine_id,
        machine_name=machine_name,
        readings=readings,
        measurements=measurements,
        max_steps=int(payload.get("max_steps", 24)),
        triggered_alert_id=payload.get("alert_id"),
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 2)

    alerts_created: list[Alert] = []
    recommendations: list[MaintenanceRecommendation] = []

    if persist:
        alerts_created = _persist_alerts(session, state)
        if payload.get("create_recommendations", True):
            recommendations = _persist_recommendations(session, state)
        session.flush()

    started_at = _parse_iso(state.get("started_at"))
    completed_at = _parse_iso(state.get("completed_at"))

    return {
        "investigation_id": None,
        "persisted": persist,
        "status": "error" if state.get("verdict") == "error" else "completed",
        "engine": state.get("engine"),
        "llm_enabled": ml_service.settings.llm_active,
        "steps_executed": state.get("step", 0),
        "max_steps_reached": bool(state.get("max_steps_reached")),
        "trace": state.get("trace", []),
        "reports": state.get("reports", []),
        "root_causes": state.get("root_causes", []),
        "citations": state.get("citations", []),
        "unsupported_claims": state.get("unsupported_claims", []),
        "knowledge_answers": state.get("knowledge_answers", []),
        "recommendations": [
            {
                **(r if not hasattr(r, "id") else _rec_dict(r)),
                "status": r.status if hasattr(r, "status") else "pending_approval",
            }
            for r in (recommendations or state.get("recommendations", []))
        ],
        "verdict": state.get("verdict"),
        "requires_human_approval": True,
        "alerts_created": [alert.id for alert in alerts_created],
        "machine_id": machine_id,
        "machine_name": machine_name,
        "readings_used": readings,
        "measurements_summary": {
            key: {
                "model_trained": bool((value or {}).get("model_trained")),
                "model_version": (value or {}).get("model_version"),
                "detail": (value or {}).get("detail"),
            }
            for key, value in measurements.items()
        },
        "duration_ms": duration_ms,
        "started_at": started_at or datetime.now(timezone.utc),
        "completed_at": completed_at or datetime.now(timezone.utc),
        "_state": state,
    }


def _rec_dict(record: MaintenanceRecommendation) -> dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "description": record.description,
        "priority": record.priority,
        "action_type": record.action_type,
        "rationale": record.rationale,
        "confidence": record.confidence,
        "evidence": record.evidence,
        "citations": record.citations,
        "estimated_downtime_min": record.estimated_downtime_min,
    }


def _persist_alerts(session: Session, state: dict[str, Any]) -> list[Alert]:
    created: list[Alert] = []
    existing = {
        alert.id for alert in session.scalars(
            select(Alert).where(Alert.machine_id == state.get("machine_id"))
        )
    } if state.get("machine_id") else set()
    del existing

    for payload in state.get("alerts", []):
        alert = Alert(
            machine_id=state.get("machine_id"),
            alert_type=payload.get("alert_type", "investigation"),
            severity=payload.get("severity", "info"),
            title=payload.get("title", "Agent alert"),
            message=payload.get("message", ""),
            source=payload.get("source", "agentic_workflow"),
            payload=payload.get("payload"),
        )
        session.add(alert)
        created.append(alert)
    return created


def _persist_recommendations(
    session: Session, state: dict[str, Any]
) -> list[MaintenanceRecommendation]:
    created: list[MaintenanceRecommendation] = []
    for payload in state.get("recommendations", []):
        record = MaintenanceRecommendation(
            machine_id=state.get("machine_id"),
            title=payload.get("title", "Maintenance action"),
            description=payload.get("description", ""),
            priority=payload.get("priority", "medium"),
            action_type=payload.get("action_type", "inspection"),
            estimated_downtime_min=payload.get("estimated_downtime_min"),
            rationale=payload.get("rationale"),
            confidence=payload.get("confidence"),
            evidence=payload.get("evidence"),
            citations=payload.get("citations"),
            status="pending_approval",
            requires_human_approval=True,
        )
        session.add(record)
        created.append(record)
    session.flush()
    return created


def persist_investigation(session: Session, result: dict[str, Any]) -> AgentInvestigation:
    """Write the investigation record and link it to its recommendations."""
    state = result.get("_state", {})
    record = AgentInvestigation(
        machine_id=result.get("machine_id"),
        created_at=result.get("started_at"),
        completed_at=result.get("completed_at"),
        status=result.get("status", "completed"),
        engine=result.get("engine", "rule_based"),
        triggering_alert_id=state.get("triggered_alert_id"),
        trace=result.get("trace", []),
        final_summary=_summary_text(result),
        root_causes=result.get("root_causes", []),
        citations=result.get("citations", []),
        unsupported_claims=result.get("unsupported_claims", []),
        verdict=result.get("verdict"),
        requires_human_approval=True,
        steps_executed=result.get("steps_executed", 0),
        max_steps_reached=result.get("max_steps_reached", False),
    )
    session.add(record)
    session.flush()

    if result.get("recommendations"):
        ids = [r["id"] for r in result["recommendations"] if isinstance(r, dict) and "id" in r]
        if ids:
            from sqlalchemy import update

            session.execute(
                update(MaintenanceRecommendation)
                .where(MaintenanceRecommendation.id.in_(ids))
                .values(investigation_id=record.id)
            )
    return record


def _summary_text(result: dict[str, Any]) -> str:
    verification = next(
        (r for r in result.get("reports", []) if r.get("agent") == "verification"), None
    )
    if verification:
        return verification["summary"]
    causes = result.get("root_causes") or []
    if causes:
        return f"Leading hypothesis: {causes[0].get('hypothesis')}"
    return "No actionable findings were produced."


def _band(probability: float) -> str:
    from app.ml.failure_prediction.inference import risk_category

    return risk_category(probability)


def _latest_reading_payload(session: Session, machine_id: str | None) -> dict[str, Any] | None:
    if not machine_id:
        return None
    from app.services.dashboard_service import latest_reading

    reading = latest_reading(session, machine_id)
    if reading is None:
        return None
    return {
        "machine_type": None,
        "air_temperature_k": reading.air_temperature_k,
        "process_temperature_k": reading.process_temperature_k,
        "rotational_speed_rpm": reading.rotational_speed_rpm,
        "torque_nm": reading.torque_nm,
        "tool_wear_min": reading.tool_wear_min,
    }


def _recent_readings(session: Session, machine_id: str, limit: int = 8) -> list[dict[str, Any]]:
    from app.models import SensorReading

    rows = session.scalars(
        select(SensorReading)
        .where(SensorReading.machine_id == machine_id)
        .order_by(SensorReading.recorded_at.desc())
        .limit(limit)
    ).all()
    return [
        {
            "air_temperature_k": row.air_temperature_k,
            "process_temperature_k": row.process_temperature_k,
            "rotational_speed_rpm": row.rotational_speed_rpm,
            "torque_nm": row.torque_nm,
            "tool_wear_min": row.tool_wear_min,
            "recorded_at": row.recorded_at.isoformat() if row.recorded_at else None,
        }
        for row in reversed(list(rows))
    ]


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def recent_investigations(session: Session, limit: int = 25) -> list[AgentInvestigation]:
    return list(
        session.scalars(
            select(AgentInvestigation).order_by(AgentInvestigation.created_at.desc()).limit(limit)
        )
    )


def get_investigation(session: Session, investigation_id: str) -> AgentInvestigation | None:
    return session.get(AgentInvestigation, investigation_id)
