"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RiskCategory = Literal["healthy", "watch", "high", "critical"]
Severity = Literal["info", "low", "warning", "high", "critical"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------
class ComponentHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "unavailable"]
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    version: str
    environment: str
    timestamp: datetime
    llm_engine: str
    components: list[ComponentHealth]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class HealthDistribution(BaseModel):
    healthy: int
    watch: int
    high: int
    critical: int
    unknown: int


class DashboardSummary(BaseModel):
    total_machines: int
    healthy_machines: int
    machines_at_risk: int
    critical_alerts: int
    open_alerts: int
    pending_approvals: int
    unresolved_inspections: int
    health_distribution: HealthDistribution
    mean_failure_probability: float | None
    mean_rul_cycles: float | None
    anomalies_last_24h: int
    investigations_last_24h: int
    recent_activities: list["ActivityItem"]
    failure_trend: list["TrendPoint"]
    generated_at: datetime
    data_provenance: str


class ActivityItem(BaseModel):
    kind: str
    title: str
    detail: str | None
    occurred_at: datetime
    severity: str = "info"
    machine_id: str | None = None


class TrendPoint(BaseModel):
    bucket: str
    value: float
    count: int = 0


# ---------------------------------------------------------------------------
# Machines
# ---------------------------------------------------------------------------
class MachineBase(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    asset_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    model: str = Field(default="AI4I-Lathe-X", max_length=120)
    machine_type: str = Field(default="lathe", max_length=64)
    location: str = Field(default="Plant 1 / Line A", max_length=160)
    rated_rpm: float = Field(default=2000.0, gt=0)
    rated_torque_nm: float = Field(default=40.0, gt=0)
    is_active: bool = True


class MachineCreate(MachineBase):
    pass


class MachineUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    location: str | None = None
    machine_type: str | None = None
    rated_rpm: float | None = Field(default=None, gt=0)
    rated_torque_nm: float | None = Field(default=None, gt=0)
    is_active: bool | None = None


class MachineRead(ORMModel):
    id: str
    asset_id: str
    name: str
    model: str
    machine_type: str
    location: str
    rated_rpm: float
    rated_torque_nm: float
    is_active: bool
    created_at: datetime
    latest_reading: "SensorReadingRead | None" = None
    health: "MachineHealthSummary | None" = None


class MachineList(BaseModel):
    items: list[MachineRead]
    total: int
    limit: int
    offset: int


class SensorReadingRead(ORMModel):
    id: str
    machine_id: str
    recorded_at: datetime
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    source: str


class SensorReadingCreate(BaseModel):
    machine_id: str | None = None
    air_temperature_k: float = Field(gt=200, le=400, description="Kelvin (AI4I schema)")
    process_temperature_k: float = Field(gt=200, le=400)
    rotational_speed_rpm: float = Field(ge=0, le=30000)
    torque_nm: float = Field(ge=0, le=200)
    tool_wear_min: float = Field(ge=0, le=500)

    @field_validator("air_temperature_k", "process_temperature_k")
    @classmethod
    def _warn_celsius(cls, value: float) -> float:
        return value


class MachineHealthSummary(BaseModel):
    failure_probability: float | None = None
    risk_category: RiskCategory = "healthy"
    anomaly_score: float | None = None
    is_anomaly: bool | None = None
    predicted_rul: float | None = None
    model_version: str | None = None
    assessed_at: datetime | None = None


# ---------------------------------------------------------------------------
# ML inference
# ---------------------------------------------------------------------------
class FailurePredictionRequest(SensorReadingCreate):
    machine_type: str | None = None
    persist: bool = True
    is_simulation: bool = False
    shap_top_k: int = Field(default=10, ge=1, le=50)


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: Literal["increases_risk", "decreases_risk", "neutral"]
    rank: int


class FailurePredictionResponse(BaseModel):
    model_version: str
    model_family: str
    failure_probability: float
    risk_category: RiskCategory
    decision_threshold: float
    predicted_failure_type: str | None
    predicted_failure_type_confidence: float | None
    explanation: str
    shap_note: str
    features: list[FeatureContribution]
    shap_base_value: float | None = None
    model_trained: bool = True
    prediction_id: str | None = None
    is_simulation: bool = False
    created_at: datetime


class AnomalyDetectionRequest(SensorReadingCreate):
    persist: bool = True
    threshold: float | None = Field(default=None, ge=0, le=1)


class AnomalyDetectionResponse(BaseModel):
    model_version: str
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    severity: Severity
    expected_value: dict[str, float]
    feature_deviation: list[FeatureContribution]
    explanation: str
    model_trained: bool = True
    anomaly_event_id: str | None = None
    created_at: datetime


class RULRequest(BaseModel):
    engine_unit: str = Field(min_length=1, max_length=64, example="engine_unit_1")
    cycle: int = Field(ge=1, le=100000, example=150)
    window: list[list[float]] | None = Field(
        default=None,
        description="Optional sensor window; when omitted the library's own cycles for this unit are used.",
    )
    dataset_subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"
    model_family: Literal["xgboost", "lstm"] | None = None
    persist: bool = True


class RULResponse(BaseModel):
    model_version: str
    model_family: str
    dataset_subset: str
    engine_unit: str
    cycle: int
    predicted_rul: float
    lower_bound: float
    upper_bound: float
    uncertainty_pct: float
    confidence_note: str
    window_length: int
    degradation_curve: list[dict[str, float]]
    model_trained: bool = True
    rul_prediction_id: str | None = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Computer vision
# ---------------------------------------------------------------------------
class VisualInspectionRequest(BaseModel):
    category: str | None = Field(default=None, max_length=64)
    machine_id: str | None = None
    persist: bool = True


class VisualInspectionResponse(BaseModel):
    inspection_id: str | None
    model_version: str | None
    model_trained: bool
    is_anomalous: bool | None
    anomaly_score: float | None
    threshold: float | None
    num_anomalous_patches: int | None
    localisation_pct: float | None
    category: str | None
    verdict: str
    message: str
    original_url: str | None = None
    heatmap_url: str | None = None
    overlay_url: str | None = None
    preprocessing: dict[str, Any] | None = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class AgentInvestigationRequest(BaseModel):
    machine_id: str | None = None
    reading: SensorReadingCreate | None = None
    alert_id: str | None = None
    failure_probability: float | None = Field(default=None, ge=0, le=1)
    rul_cycles: float | None = Field(default=None, ge=0)
    anomaly_score: float | None = Field(default=None, ge=0, le=1)
    create_alerts: bool = True
    create_recommendations: bool = True
    max_steps: int = Field(default=24, ge=6, le=64)


class AgentStepTrace(BaseModel):
    step: int
    agent: str
    action: str
    summary: str
    status: Literal["ok", "warning", "skipped", "blocked"]
    engine: str = "rule_based"
    duration_ms: float | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class AgentReport(BaseModel):
    agent: str
    role: str
    status: Literal["ok", "warning", "skipped", "blocked"]
    engine: str
    summary: str
    findings: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = None
    caveats: list[str] = Field(default_factory=list)


class InvestigationResponse(BaseModel):
    investigation_id: str
    status: str
    engine: str
    llm_enabled: bool
    steps_executed: int
    max_steps_reached: bool
    trace: list[AgentStepTrace]
    reports: list[AgentReport]
    root_causes: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    unsupported_claims: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    verdict: str
    requires_human_approval: bool
    alerts_created: list[str]
    duration_ms: float
    started_at: datetime
    completed_at: datetime


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
class KnowledgeUploadResponse(BaseModel):
    document_id: str
    filename: str
    title: str | None
    page_count: int | None
    chunk_count: int
    char_count: int
    embedding_model: str
    status: str
    index_size: int
    message: str


class KnowledgeQueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None
    save_history: bool = True


class RetrievedChunk(BaseModel):
    chunk_id: int
    document_id: str
    document_title: str | None
    filename: str
    page: int | None
    score: float
    text: str


class KnowledgeQueryResponse(BaseModel):
    question: str
    answer: str
    answer_engine: Literal["extractive", "llm_synthesised"]
    citations: list[RetrievedChunk]
    retrieved_count: int
    insufficient_evidence: bool
    disclaimer: str
    created_at: datetime


class KnowledgeDocumentRead(ORMModel):
    id: str
    filename: str
    title: str | None
    page_count: int | None
    chunk_count: int | None
    embedding_model: str | None
    status: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------
class RecommendationRead(ORMModel):
    id: str
    machine_id: str | None
    investigation_id: str | None
    title: str
    description: str
    priority: str
    action_type: str
    estimated_downtime_min: int | None
    estimated_cost: float | None
    evidence: list | None
    citations: list | None
    rationale: str | None
    confidence: float | None
    status: str
    approved_by: str | None
    approved_at: datetime | None
    completed_by: str | None
    completed_at: datetime | None
    created_at: datetime


class ApprovalRequest(BaseModel):
    approver: str = Field(min_length=1, max_length=160)
    comment: str | None = Field(default=None, max_length=2000)
    override_downtime_min: int | None = Field(default=None, ge=0)


class CompletionRequest(BaseModel):
    performed_by: str = Field(min_length=1, max_length=160)
    notes: str | None = Field(default=None, max_length=4000)
    actual_downtime_min: int | None = Field(default=None, ge=0)
    actual_cost: float | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
class AlertRead(ORMModel):
    id: str
    machine_id: str | None
    created_at: datetime
    alert_type: str
    severity: str
    title: str
    message: str
    source: str
    is_read: bool
    is_resolved: bool
    payload: dict | None


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
class ModelMetrics(BaseModel):
    model_version: str
    task: str
    algorithm: str
    dataset: str
    metrics: dict[str, Any]
    feature_names: list[str] | None = None
    hyperparameters: dict[str, Any] | None = None
    training_samples: int | None = None
    trained_at: datetime | None = None
    artifact_path: str | None = None
    dataset_sha256: str | None = None


class ModelMetricsList(BaseModel):
    items: list[ModelMetrics]
    total: int
    active_models: int


class AnalyticsResponse(BaseModel):
    generated_at: datetime
    failure_trend: list[TrendPoint]
    risk_histogram: list[dict[str, Any]]
    anomaly_timeline: list[TrendPoint]
    rul_distribution: list[dict[str, Any]]
    maintenance_efficiency: dict[str, Any]
    downtime_by_machine: list[dict[str, Any]]
    model_metrics: list[ModelMetrics]
    dataset_manifest: dict[str, Any]
    disclaimer: str


class MachineComparison(BaseModel):
    items: list[dict[str, Any]]
    total: int


DashboardSummary.model_rebuild()
MachineRead.model_rebuild()
