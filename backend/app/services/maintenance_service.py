"""Maintenance recommendation lifecycle.

State machine - every transition is explicit and recorded in history:

    pending_approval -> approved -> in_progress -> completed
                       -> rejected
                       -> cancelled

No agent, service or API can move a recommendation straight to ``completed``;
that requires an approval first. This is the mechanical guarantee behind
"human approval required".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Alert, MaintenanceHistory, MaintenanceRecommendation

logger = get_logger(__name__)

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "pending_approval": {"approved", "rejected", "cancelled"},
    "approved": {"in_progress", "completed", "cancelled"},
    "in_progress": {"completed", "cancelled"},
    "rejected": set(),
    "completed": set(),
    "cancelled": set(),
}


class InvalidTransition(ValueError):
    """Raised when an approval-gated transition is attempted out of order."""


def list_recommendations(
    session: Session,
    status: str | None = None,
    machine_id: str | None = None,
    limit: int = 100,
) -> list[MaintenanceRecommendation]:
    statement = select(MaintenanceRecommendation)
    if status:
        statement = statement.where(MaintenanceRecommendation.status == status)
    if machine_id:
        statement = statement.where(MaintenanceRecommendation.machine_id == machine_id)
    return list(
        session.scalars(
            statement.order_by(MaintenanceRecommendation.created_at.desc()).limit(limit)
        )
    )


def _transition(
    session: Session,
    recommendation: MaintenanceRecommendation,
    new_status: str,
    performed_by: str,
    notes: str | None,
    downtime: int | None,
    cost: float | None,
) -> MaintenanceRecommendation:
    current = recommendation.status
    if new_status not in ALLOWED_TRANSITIONS.get(current, set()):
        raise InvalidTransition(
            f"Cannot move a recommendation from '{current}' to '{new_status}'. "
            f"Allowed from '{current}': {sorted(ALLOWED_TRANSITIONS.get(current, set())) or 'none (terminal state)'}"
        )
    recommendation.status = new_status
    session.add(recommendation)

    session.add(
        MaintenanceHistory(
            machine_id=recommendation.machine_id,
            recommendation_id=recommendation.id,
            event_type=new_status,
            performed_by=performed_by,
            notes=notes,
            downtime_min=downtime if downtime is not None else recommendation.estimated_downtime_min,
            cost=cost,
            performed_at=datetime.now(timezone.utc),
        )
    )

    if new_status == "approved":
        recommendation.approved_by = performed_by
        recommendation.approved_at = datetime.now(timezone.utc)
    elif new_status == "completed":
        recommendation.completed_by = performed_by
        recommendation.completed_at = datetime.now(timezone.utc)

    if new_status in {"approved", "in_progress", "completed"}:
        session.add(
            Alert(
                machine_id=recommendation.machine_id,
                alert_type="maintenance_update",
                severity="info",
                title=f"Maintenance {new_status.replace('_', ' ')}: {recommendation.title}",
                message=f"{performed_by} moved recommendation {recommendation.id} to {new_status}.",
                source="maintenance_workflow",
                payload={"recommendation_id": recommendation.id, "status": new_status},
            )
        )
    session.flush()
    logger.info("Recommendation %s: %s -> %s by %s", recommendation.id, current, new_status, performed_by)
    return recommendation


def approve(
    session: Session,
    recommendation: MaintenanceRecommendation,
    approver: str,
    comment: str | None = None,
    downtime: int | None = None,
) -> MaintenanceRecommendation:
    return _transition(
        session, recommendation, "approved", approver, comment, downtime, None
    )


def reject(
    session: Session,
    recommendation: MaintenanceRecommendation,
    approver: str,
    comment: str | None = None,
) -> MaintenanceRecommendation:
    return _transition(session, recommendation, "rejected", approver, comment, None, None)


def complete(
    session: Session,
    recommendation: MaintenanceRecommendation,
    performed_by: str,
    notes: str | None = None,
    downtime: int | None = None,
    cost: float | None = None,
) -> MaintenanceRecommendation:
    """Complete a recommendation. Requires prior approval."""
    if recommendation.status == "pending_approval":
        raise InvalidTransition(
            "This recommendation has not been approved. A qualified engineer must approve "
            "it via POST /api/maintenance/{id}/approve before the work can be marked complete."
        )
    return _transition(session, recommendation, "completed", performed_by, notes, downtime, cost)


def start(
    session: Session,
    recommendation: MaintenanceRecommendation,
    performed_by: str,
    notes: str | None = None,
) -> MaintenanceRecommendation:
    return _transition(session, recommendation, "in_progress", performed_by, notes, None, None)


def history(
    session: Session, machine_id: str | None = None, limit: int = 100
) -> list[MaintenanceHistory]:
    statement = select(MaintenanceHistory)
    if machine_id:
        statement = statement.where(MaintenanceHistory.machine_id == machine_id)
    return list(
        session.scalars(
            statement.order_by(MaintenanceHistory.performed_at.desc()).limit(limit)
        )
    )


def workflow_description() -> dict[str, Any]:
    return {
        "states": sorted(ALLOWED_TRANSITIONS),
        "transitions": {k: sorted(v) for k, v in ALLOWED_TRANSITIONS.items()},
        "human_approval_required": True,
        "guarantee": (
            "A recommendation cannot reach 'completed' without first passing through "
            "'approved'. The transition table is enforced server-side; no agent or client "
            "can bypass it."
        ),
    }
