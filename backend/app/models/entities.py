"""SQLAlchemy ORM models - one module per bounded area, all re-exported."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, utcnow


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="engineer", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Machine(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "machines"

    asset_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    model: Mapped[str] = mapped_column(String(120), default="AI4I-Lathe-X", nullable=False)
    machine_type: Mapped[str] = mapped_column(String(64), default="lathe", nullable=False)
    location: Mapped[str] = mapped_column(String(160), default="Plant 1 / Line A", nullable=False)
    rated_rpm: Mapped[float] = mapped_column(Float, default=2000.0)
    rated_torque_nm: Mapped[float] = mapped_column(Float, default=40.0)
    installed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    readings: Mapped[list["SensorReading"]] = relationship(
        back_populates="machine", cascade="all, delete-orphan", lazy="selectin"
    )


class SensorReading(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "sensor_readings"
    __table_args__ = (Index("ix_sensor_readings_machine_recorded", "machine_id", "recorded_at"),)

    machine_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("machines.id", ondelete="CASCADE"), index=True, nullable=False
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True, nullable=False
    )
    air_temperature_k: Mapped[float] = mapped_column(Float, nullable=False)
    process_temperature_k: Mapped[float] = mapped_column(Float, nullable=False)
    rotational_speed_rpm: Mapped[float] = mapped_column(Float, nullable=False)
    torque_nm: Mapped[float] = mapped_column(Float, nullable=False)
    tool_wear_min: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="ai4i", nullable=False)

    machine: Mapped[Machine] = relationship(back_populates="readings")


class FailurePrediction(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "failure_predictions"
    __table_args__ = (Index("ix_failure_predictions_machine_created", "machine_id", "created_at"),)

    machine_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("machines.id", ondelete="SET NULL"), index=True
    )
    reading_id: Mapped[str | None] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_category: Mapped[str] = mapped_column(String(24), nullable=False)
    predicted_failure_type: Mapped[str | None] = mapped_column(String(64))
    predicted_failure_type_confidence: Mapped[float | None] = mapped_column(Float)
    shap_values: Mapped[dict | None] = mapped_column(JSON)
    top_features: Mapped[list | None] = mapped_column(JSON)
    explanation: Mapped[str | None] = mapped_column(Text)
    input_payload: Mapped[dict | None] = mapped_column(JSON)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class AnomalyEvent(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "anomaly_events"

    machine_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("machines.id", ondelete="SET NULL"), index=True
    )
    reading_id: Mapped[str | None] = mapped_column(String(32), index=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    severity: Mapped[str] = mapped_column(String(24), default="normal", nullable=False)
    contributing_features: Mapped[list | None] = mapped_column(JSON)
    raw_reading: Mapped[dict | None] = mapped_column(JSON)


class RULPrediction(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "rul_predictions"

    machine_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("machines.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    engine_unit: Mapped[str] = mapped_column(String(64), nullable=False)
    cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_family: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_subset: Mapped[str] = mapped_column(String(16), default="FD001", nullable=False)
    predicted_rul: Mapped[float] = mapped_column(Float, nullable=False)
    lower_bound: Mapped[float] = mapped_column(Float, nullable=False)
    upper_bound: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty_pct: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_note: Mapped[str | None] = mapped_column(Text)
    degradation_curve: Mapped[list | None] = mapped_column(JSON)


class VisualInspection(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "visual_inspections"

    machine_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("machines.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(64))
    model_trained: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_anomalous: Mapped[bool | None] = mapped_column(Boolean)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    num_anomalous_patches: Mapped[int | None] = mapped_column(Integer)
    localisation_pct: Mapped[float | None] = mapped_column(Float)
    heatmap_path: Mapped[str | None] = mapped_column(String(512))
    overlay_path: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False)
    message: Mapped[str | None] = mapped_column(Text)


class AgentInvestigation(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "agent_investigations"

    machine_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("machines.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)
    engine: Mapped[str] = mapped_column(String(32), default="rule_based", nullable=False)
    triggering_alert_id: Mapped[str | None] = mapped_column(String(32), index=True)
    trace: Mapped[list | None] = mapped_column(JSON)
    final_summary: Mapped[str | None] = mapped_column(Text)
    root_causes: Mapped[list | None] = mapped_column(JSON)
    citations: Mapped[list | None] = mapped_column(JSON)
    unsupported_claims: Mapped[list | None] = mapped_column(JSON)
    verdict: Mapped[str | None] = mapped_column(String(32))
    requires_human_approval: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    steps_executed: Mapped[int] = mapped_column(Integer, default=0)
    max_steps_reached: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class MaintenanceRecommendation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "maintenance_recommendations"

    machine_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("machines.id", ondelete="SET NULL"), index=True
    )
    investigation_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("agent_investigations.id", ondelete="SET NULL"), index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(24), default="medium", nullable=False)
    action_type: Mapped[str] = mapped_column(String(48), default="inspection", nullable=False)
    estimated_downtime_min: Mapped[int | None] = mapped_column(Integer)
    estimated_cost: Mapped[float | None] = mapped_column(Float)
    evidence: Mapped[list | None] = mapped_column(JSON)
    citations: Mapped[list | None] = mapped_column(JSON)
    rationale: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="pending_approval", index=True)
    approved_by: Mapped[str | None] = mapped_column(String(255))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_by: Mapped[str | None] = mapped_column(String(255))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    requires_human_approval: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class MaintenanceHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "maintenance_history"

    machine_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("machines.id", ondelete="SET NULL"), index=True
    )
    recommendation_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("maintenance_recommendations.id", ondelete="SET NULL"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    performed_by: Mapped[str] = mapped_column(String(160), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    downtime_min: Mapped[int | None] = mapped_column(Integer)
    cost: Mapped[float | None] = mapped_column(Float)
    performed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True, nullable=False
    )


class KnowledgeDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "knowledge_documents"

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    char_count: Mapped[int | None] = mapped_column(Integer)
    embedding_model: Mapped[str | None] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(32), default="ingested", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)


class ModelVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)

    name: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    task: Mapped[str] = mapped_column(String(64), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(80), nullable=False)
    dataset: Mapped[str] = mapped_column(String(160), nullable=False)
    dataset_sha256: Mapped[str | None] = mapped_column(String(64))
    metrics: Mapped[dict | None] = mapped_column(JSON)
    feature_names: Mapped[list | None] = mapped_column(JSON)
    hyperparameters: Mapped[dict | None] = mapped_column(JSON)
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    training_samples: Mapped[int | None] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Alert(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "alerts"

    machine_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("machines.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    alert_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(24), default="info", nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(48), default="agentic_workflow", nullable=False)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSON)


__all__ = [
    "AgentInvestigation",
    "Alert",
    "AnomalyEvent",
    "FailurePrediction",
    "KnowledgeDocument",
    "Machine",
    "MaintenanceHistory",
    "MaintenanceRecommendation",
    "ModelVersion",
    "RULPrediction",
    "SensorReading",
    "User",
    "VisualInspection",
]
