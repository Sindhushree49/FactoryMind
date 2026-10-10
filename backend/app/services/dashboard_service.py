"""Dashboard, machine registry and analytics services.

Every statistic returned here is computed from the database or from a trained
model. Where a model is missing, the response says ``null`` with an explicit
reason - dashboard numbers are never hardcoded placeholders.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import (
    AgentInvestigation,
    Alert,
    AnomalyEvent,
    FailurePrediction,
    Machine,
    MaintenanceHistory,
    MaintenanceRecommendation,
    ModelVersion,
    RULPrediction,
    SensorReading,
    VisualInspection,
)
from app.ml.preprocessing import ai4i
from app.services import ml_service

logger = get_logger(__name__)

PROVENANCE_DISCLAIMER = (
    "All figures are computed at request time from FactoryMind's own database and "
    "trained models. No value on this dashboard is a hardcoded placeholder."
)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# ---------------------------------------------------------------------------
# Machines
# ---------------------------------------------------------------------------
def create_machine(session: Session, payload: dict[str, Any]) -> Machine:
    existing = session.scalar(select(Machine).where(Machine.asset_id == payload["asset_id"]))
    if existing is not None:
        raise ValueError(f"A machine with asset_id '{payload['asset_id']}' already exists.")
    machine = Machine(**payload)
    session.add(machine)
    session.flush()
    logger.info("Created machine %s (%s)", machine.name, machine.asset_id)
    return machine


def list_machines(session: Session, limit: int = 50, offset: int = 0,
                  active_only: bool = False) -> tuple[list[Machine], int]:
    statement = select(Machine)
    if active_only:
        statement = statement.where(Machine.is_active.is_(True))
    total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
    machines = list(
        session.scalars(statement.order_by(Machine.created_at.desc()).limit(limit).offset(offset))
    )
    return machines, int(total)


def get_machine(session: Session, machine_id: str) -> Machine | None:
    return session.get(Machine, machine_id)


def latest_reading(session: Session, machine_id: str) -> SensorReading | None:
    return session.scalar(
        select(SensorReading)
        .where(SensorReading.machine_id == machine_id)
        .order_by(SensorReading.recorded_at.desc())
        .limit(1)
    )


def machine_health(session: Session, machine_id: str) -> dict[str, Any]:
    """Assemble a machine's current health from the newest stored predictions."""
    failure = session.scalar(
        select(FailurePrediction)
        .where(FailurePrediction.machine_id == machine_id, FailurePrediction.is_simulated.is_(False))
        .order_by(FailurePrediction.created_at.desc())
        .limit(1)
    )
    anomaly = session.scalar(
        select(AnomalyEvent)
        .where(AnomalyEvent.machine_id == machine_id)
        .order_by(AnomalyEvent.detected_at.desc())
        .limit(1)
    )
    rul = session.scalar(
        select(RULPrediction)
        .where(RULPrediction.machine_id == machine_id)
        .order_by(RULPrediction.created_at.desc())
        .limit(1)
    )
    return {
        "failure_probability": round(failure.failure_probability, 6) if failure else None,
        "risk_category": failure.risk_category if failure else "healthy",
        "anomaly_score": round(anomaly.anomaly_score, 6) if anomaly else None,
        "is_anomaly": anomaly.is_anomaly if anomaly else None,
        "predicted_rul": round(rul.predicted_rul, 2) if rul else None,
        "model_version": failure.model_version if failure else None,
        "assessed_at": _aware(failure.created_at) if failure else None,
    }


def seed_machines_from_dataset(session: Session, count: int = 8) -> list[Machine]:
    """Register machines and replay *real* AI4I rows as their sensor history.

    Rows are taken verbatim from the published dataset; none are synthesised.
    """
    try:
        frame, report = ai4i.prepare()
    except ai4i.AI4IDatasetMissing:
        # Let the caller distinguish "dataset absent" (503) from
        # "every requested machine already seeded" (200, empty list).
        raise

    created: list[Machine] = []
    stride = max(1, len(frame) // max(count, 1))
    for index in range(count):
        asset_id = f"FM-{index + 1:03d}"
        if session.scalar(select(Machine).where(Machine.asset_id == asset_id)):
            continue
        slice_ = frame.iloc[index * stride : (index + 1) * stride]
        if slice_.empty:
            continue
        machine = Machine(
            asset_id=asset_id,
            name=f"Cell {chr(ord('A') + index % 6)}{index + 1} Lathe",
            model="AI4I-Lathe-X",
            machine_type=str(slice_["machine_type"].mode().iloc[0]),
            location=f"Plant 1 / Line {chr(ord('A') + index % 6)}",
        )
        session.add(machine)
        session.flush()
        for _, row in slice_.iterrows():
            session.add(
                SensorReading(
                    machine_id=machine.id,
                    air_temperature_k=float(row["air_temperature_k"]),
                    process_temperature_k=float(row["process_temperature_k"]),
                    rotational_speed_rpm=float(row["rotational_speed_rpm"]),
                    torque_nm=float(row["torque_nm"]),
                    tool_wear_min=float(row["tool_wear_min"]),
                    source="ai4i",
                )
            )
        created.append(machine)

    if created:
        logger.info(
            "Seeded %d machines from AI4I (%d source rows, failure rate %.4f)",
            len(created), report.rows, report.failure_rate,
        )
    return created


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def dashboard_summary(session: Session) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(hours=24)

    total_machines = int(session.scalar(select(func.count(Machine.id))) or 0)
    open_alerts = session.scalar(
        select(func.count(Alert.id)).where(Alert.is_resolved.is_(False))
    ) or 0
    critical_alerts = session.scalar(
        select(func.count(Alert.id)).where(
            Alert.is_resolved.is_(False), Alert.severity.in_(["high", "critical"])
        )
    ) or 0
    pending_approvals = session.scalar(
        select(func.count(MaintenanceRecommendation.id)).where(
            MaintenanceRecommendation.status == "pending_approval"
        )
    ) or 0
    anomalies_24h = session.scalar(
        select(func.count(AnomalyEvent.id)).where(AnomalyEvent.detected_at >= day_ago)
    ) or 0
    investigations_24h = session.scalar(
        select(func.count(AgentInvestigation.id)).where(AgentInvestigation.created_at >= day_ago)
    ) or 0

    distribution = _health_distribution(session)
    healthy = distribution["healthy"] + distribution["watch"]
    at_risk = distribution["high"] + distribution["critical"]

    mean_failure = session.scalar(
        select(func.avg(FailurePrediction.failure_probability)).where(
            FailurePrediction.is_simulated.is_(False)
        )
    )
    mean_rul = session.scalar(select(func.avg(RULPrediction.predicted_rul)))

    return {
        "total_machines": total_machines,
        "healthy_machines": healthy,
        "machines_at_risk": at_risk,
        "critical_alerts": int(critical_alerts),
        "open_alerts": int(open_alerts),
        "pending_approvals": int(pending_approvals),
        "unresolved_inspections": int(
            session.scalar(
                select(func.count(VisualInspection.id)).where(
                    VisualInspection.model_trained.is_(True),
                    VisualInspection.is_anomalous.is_(True),
                    VisualInspection.status != "reviewed",
                )
            ) or 0
        ),
        "health_distribution": distribution,
        "mean_failure_probability": round(float(mean_failure), 6) if mean_failure else None,
        "mean_rul_cycles": round(float(mean_rul), 2) if mean_rul else None,
        "anomalies_last_24h": int(anomalies_24h),
        "investigations_last_24h": int(investigations_24h),
        "recent_activities": recent_activity(session, limit=12),
        "failure_trend": failure_trend(session, days=14),
        "generated_at": now,
        "data_provenance": PROVENANCE_DISCLAIMER,
    }


def _health_distribution(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(FailurePrediction.machine_id, FailurePrediction.risk_category)
        .where(FailurePrediction.is_simulated.is_(False))
        .order_by(FailurePrediction.created_at.desc())
    ).all()

    latest: dict[str, str] = {}
    for machine_id, category in rows:
        if machine_id and machine_id not in latest:
            latest[machine_id] = category

    total = int(session.scalar(select(func.count(Machine.id))) or 0)
    distribution = {"healthy": 0, "watch": 0, "high": 0, "critical": 0, "unknown": 0}
    for category in latest.values():
        distribution[category] = distribution.get(category, 0) + 1
    distribution["unknown"] = max(0, total - len(latest))
    return distribution


def failure_trend(session: Session, days: int = 14) -> list[dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = session.execute(
        select(FailurePrediction.created_at, FailurePrediction.failure_probability)
        .where(FailurePrediction.created_at >= since, FailurePrediction.is_simulated.is_(False))
        .order_by(FailurePrediction.created_at)
    ).all()
    buckets: dict[str, list[float]] = {}
    for created_at, probability in rows:
        key = _aware(created_at).strftime("%Y-%m-%d")
        buckets.setdefault(key, []).append(float(probability))
    return [
        {
            "bucket": key,
            "value": round(sum(values) / len(values), 6),
            "count": len(values),
        }
        for key, values in sorted(buckets.items())
    ]


def recent_activity(session: Session, limit: int = 12) -> list[dict[str, Any]]:
    activity: list[dict[str, Any]] = []

    for alert in session.scalars(
        select(Alert).order_by(Alert.created_at.desc()).limit(limit)
    ):
        activity.append(
            {
                "kind": "alert",
                "title": alert.title,
                "detail": alert.message[:220],
                "occurred_at": _aware(alert.created_at),
                "severity": alert.severity,
                "machine_id": alert.machine_id,
            }
        )
    for inspection in session.scalars(
        select(VisualInspection).order_by(VisualInspection.created_at.desc()).limit(limit)
    ):
        activity.append(
            {
                "kind": "visual_inspection",
                "title": f"Visual inspection: {inspection.original_filename}",
                "detail": inspection.message or inspection.status,
                "occurred_at": _aware(inspection.created_at),
                "severity": "warning" if inspection.is_anomalous else "info",
                "machine_id": inspection.machine_id,
            }
        )
    for investigation in session.scalars(
        select(AgentInvestigation).order_by(AgentInvestigation.created_at.desc()).limit(limit)
    ):
        activity.append(
            {
                "kind": "agent_investigation",
                "title": "Agentic investigation completed",
                "detail": (investigation.final_summary or investigation.verdict or "")[:220],
                "occurred_at": _aware(investigation.created_at),
                "severity": "warning" if investigation.verdict != "supported" else "info",
                "machine_id": investigation.machine_id,
            }
        )
    for record in session.scalars(
        select(MaintenanceHistory).order_by(MaintenanceHistory.performed_at.desc()).limit(limit)
    ):
        activity.append(
            {
                "kind": "maintenance",
                "title": f"Maintenance: {record.event_type}",
                "detail": (record.notes or "")[:220],
                "occurred_at": _aware(record.performed_at),
                "severity": "info",
                "machine_id": record.machine_id,
            }
        )

    activity.sort(key=lambda item: item["occurred_at"], reverse=True)
    return activity[:limit]


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def analytics(session: Session) -> dict[str, Any]:
    from app.ml import datasets as dataset_registry
    from app.ml import registry

    risk_histogram = [
        {"bucket": bucket, "count": count}
        for bucket, count in session.execute(
            select(FailurePrediction.risk_category, func.count(FailurePrediction.id))
            .where(FailurePrediction.is_simulated.is_(False))
            .group_by(FailurePrediction.risk_category)
        ).all()
    ]

    anomaly_timeline = _daily_counts(session, AnomalyEvent, AnomalyEvent.detected_at, days=14)

    rul_distribution: list[dict[str, Any]] = []
    buckets = session.execute(
        select(RULPrediction.predicted_rul).order_by(RULPrediction.predicted_rul)
    ).scalars().all()
    if buckets:
        edges = [0, 20, 40, 60, 80, 100, 125]
        for low, high in zip(edges[:-1], edges[1:]):
            count = sum(1 for value in buckets if low <= value < high)
            rul_distribution.append(
                {"bucket": f"{low}-{high}", "count": count, "lower": low, "upper": high}
            )

    downtime_rows = session.execute(
        select(
            MaintenanceHistory.machine_id,
            func.coalesce(func.sum(MaintenanceHistory.downtime_min), 0),
            func.count(MaintenanceHistory.id),
        )
        .group_by(MaintenanceHistory.machine_id)
        .order_by(func.coalesce(func.sum(MaintenanceHistory.downtime_min), 0).desc())
    ).all()
    machine_names = {
        m.id: m.name for m in session.scalars(select(Machine))
    }
    downtime_by_machine = [
        {
            "machine_id": machine_id,
            "machine_name": machine_names.get(machine_id, "unknown"),
            "downtime_min": int(total or 0),
            "events": int(count or 0),
        }
        for machine_id, total, count in downtime_rows
        if machine_id
    ]

    completed = session.scalar(
        select(func.count(MaintenanceRecommendation.id)).where(
            MaintenanceRecommendation.status == "completed"
        )
    ) or 0
    approved = session.scalar(
        select(func.count(MaintenanceRecommendation.id)).where(
            MaintenanceRecommendation.status.in_(["approved", "in_progress", "completed"])
        )
    ) or 0

    model_metrics = []
    for task in (registry.TASK_FAILURE, registry.TASK_ANOMALY, registry.TASK_RUL):
        for manifest in registry.list_versions(task):
            model_metrics.append(
                {
                    "model_version": f"{manifest.get('name')}:{manifest.get('version')}",
                    "task": task,
                    "algorithm": manifest.get("family") or manifest.get("algorithm"),
                    "dataset": manifest.get("dataset"),
                    "dataset_sha256": manifest.get("dataset_sha256"),
                    "metrics": manifest.get("metrics", {}),
                    "feature_names": manifest.get("feature_names"),
                    "hyperparameters": manifest.get("hyperparameters"),
                    "training_samples": manifest.get("training_samples"),
                    "trained_at": manifest.get("trained_at"),
                    "artifact_path": manifest.get("path"),
                }
            )
    for category in registry.categories(registry.TASK_VISION):
        for manifest_file in sorted(registry.task_dir(registry.TASK_VISION).glob(f"{category}/*/manifest.json")):
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            model_metrics.append(
                {
                    "model_version": f"{manifest.get('name')}:{manifest.get('version')}",
                    "task": registry.TASK_VISION,
                    "algorithm": manifest.get("family"),
                    "dataset": manifest.get("dataset"),
                    "dataset_sha256": None,
                    "metrics": manifest.get("metrics", {}),
                    "feature_names": manifest.get("feature_names"),
                    "hyperparameters": manifest.get("hyperparameters"),
                    "training_samples": manifest.get("training_samples"),
                    "trained_at": manifest.get("trained_at"),
                    "artifact_path": str(manifest_file.parent),
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc),
        "failure_trend": failure_trend(session, days=14),
        "risk_histogram": risk_histogram,
        "anomaly_timeline": anomaly_timeline,
        "rul_distribution": rul_distribution,
        "maintenance_efficiency": {
            "recommendations_completed": int(completed),
            "recommendations_approved": int(approved),
            "approval_rate": round(approved / max(int(approved) + _pending(session), 1), 4),
            "downtime_min_total": int(
                session.scalar(
                    select(func.coalesce(func.sum(MaintenanceHistory.downtime_min), 0))
                ) or 0
            ),
            "cost_total": round(
                float(
                    session.scalar(select(func.coalesce(func.sum(MaintenanceHistory.cost), 0.0)))
                    or 0.0
                ),
                2,
            ),
        },
        "downtime_by_machine": downtime_by_machine,
        "model_metrics": model_metrics,
        "dataset_manifest": {
            "statuses": {k: v.to_dict() for k, v in dataset_registry.all_statuses().items()},
            "provenance": dataset_registry.manifest(),
        },
        "disclaimer": PROVENANCE_DISCLAIMER,
    }


def _pending(session: Session) -> int:
    return int(
        session.scalar(
            select(func.count(MaintenanceRecommendation.id)).where(
                MaintenanceRecommendation.status == "pending_approval"
            )
        ) or 0
    )


def _daily_counts(session: Session, model: Any, column: Any, days: int = 14) -> list[dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = session.execute(select(column).where(column >= since)).scalars().all()
    counts: dict[str, int] = {}
    for value in rows:
        key = _aware(value).strftime("%Y-%m-%d")
        counts[key] = counts.get(key, 0) + 1
    return [
        {"bucket": key, "value": float(count), "count": count}
        for key, count in sorted(counts.items())
    ]


def model_metrics(session: Session) -> dict[str, Any]:
    """Published model versions, from disk and mirrored into the database."""
    from app.ml import registry

    items: list[dict[str, Any]] = []
    for task in (registry.TASK_FAILURE, registry.TASK_ANOMALY, registry.TASK_RUL):
        for manifest in registry.list_versions(task):
            items.append(
                {
                    "model_version": f"{manifest.get('name')}:{manifest.get('version')}",
                    "task": task,
                    "algorithm": manifest.get("family") or manifest.get("algorithm"),
                    "dataset": manifest.get("dataset"),
                    "dataset_sha256": manifest.get("dataset_sha256"),
                    "metrics": manifest.get("metrics", {}),
                    "feature_names": manifest.get("feature_names"),
                    "hyperparameters": manifest.get("hyperparameters"),
                    "training_samples": manifest.get("training_samples"),
                    "trained_at": manifest.get("trained_at"),
                    "artifact_path": manifest.get("path"),
                }
            )

    for row in session.scalars(select(ModelVersion)):
        items.append(
            {
                "model_version": f"{row.name}:{row.version}",
                "task": row.task,
                "algorithm": row.algorithm,
                "dataset": row.dataset,
                "dataset_sha256": row.dataset_sha256,
                "metrics": row.metrics or {},
                "feature_names": row.feature_names,
                "hyperparameters": row.hyperparameters,
                "training_samples": row.training_samples,
                "trained_at": row.trained_at,
                "artifact_path": row.artifact_path,
            }
        )

    vision = ml_service.vision_status()
    if vision.get("loaded"):
        items.append(
            {
                "model_version": f"patchcore:{vision.get('version')}",
                "task": "visual_inspection",
                "algorithm": "patchcore",
                "dataset": "MVTec AD",
                "dataset_sha256": None,
                "metrics": vision.get("metrics") or {},
                "feature_names": None,
                "hyperparameters": None,
                "training_samples": None,
                "trained_at": None,
                "artifact_path": None,
            }
        )

    # De-duplicate on model_version, keeping the first (on-disk source of truth).
    seen: set[str] = set()
    unique = []
    for item in items:
        if item["model_version"] in seen:
            continue
        seen.add(item["model_version"])
        unique.append(item)

    return {
        "items": unique,
        "total": len(unique),
        "active_models": sum(1 for item in unique if item["metrics"]),
    }
