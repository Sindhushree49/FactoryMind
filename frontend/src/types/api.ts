/**
 * Typed API contract mirroring backend/app/schemas/api.py.
 * Field names match the backend exactly so responses need no reshaping.
 */

export type RiskCategory = 'healthy' | 'watch' | 'high' | 'critical'
export type Severity = 'info' | 'low' | 'warning' | 'high' | 'critical'
export type AgentStatus = 'ok' | 'warning' | 'skipped' | 'blocked'

export interface ComponentHealth {
  name: string
  status: 'ok' | 'degraded' | 'unavailable'
  detail: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  app: string
  version: string
  environment: string
  timestamp: string
  llm_engine: 'llm' | 'rule_based'
  components: ComponentHealth[]
}

export interface ActivityItem {
  kind: string
  title: string
  detail: string | null
  occurred_at: string
  severity: string
  machine_id: string | null
}

export interface TrendPoint {
  bucket: string
  value: number
  count: number
}

export interface HealthDistribution {
  healthy: number
  watch: number
  high: number
  critical: number
  unknown: number
}

export interface DashboardSummary {
  total_machines: number
  healthy_machines: number
  machines_at_risk: number
  critical_alerts: number
  open_alerts: number
  pending_approvals: number
  unresolved_inspections: number
  health_distribution: HealthDistribution
  mean_failure_probability: number | null
  mean_rul_cycles: number | null
  anomalies_last_24h: number
  investigations_last_24h: number
  recent_activities: ActivityItem[]
  failure_trend: TrendPoint[]
  generated_at: string
  data_provenance: string
}

export interface SensorReading {
  id?: string
  machine_id?: string
  recorded_at?: string
  air_temperature_k: number
  process_temperature_k: number
  rotational_speed_rpm: number
  torque_nm: number
  tool_wear_min: number
  source?: string
}

export interface MachineHealthSummary {
  failure_probability: number | null
  risk_category: RiskCategory
  anomaly_score: number | null
  is_anomaly: boolean | null
  predicted_rul: number | null
  model_version: string | null
  assessed_at: string | null
}

export interface Machine {
  id: string
  asset_id: string
  name: string
  model: string
  machine_type: string
  location: string
  rated_rpm: number
  rated_torque_nm: number
  is_active: boolean
  created_at: string
  latest_reading: SensorReading | null
  health: MachineHealthSummary | null
}

export interface MachineList {
  items: Machine[]
  total: number
  limit: number
  offset: number
}

export interface FeatureContribution {
  feature: string
  value: number
  contribution: number
  direction: 'increases_risk' | 'decreases_risk' | 'neutral'
  rank: number
  z_score?: number
  expected_mean?: number
  expected_range?: [number, number]
  out_of_training_range?: boolean
}

export interface FailurePredictionRequest extends SensorReading {
  machine_type?: string | null
  persist?: boolean
  is_simulation?: boolean
  shap_top_k?: number
}

export interface FailurePredictionResponse {
  model_version: string
  model_family: string
  failure_probability: number
  risk_category: RiskCategory
  decision_threshold: number
  predicted_failure_type: string | null
  predicted_failure_type_confidence: number | null
  explanation: string
  shap_note: string
  features: FeatureContribution[]
  shap_base_value: number | null
  model_trained: boolean
  prediction_id: string | null
  is_simulation: boolean
  created_at: string
}

export interface AnomalyDetectionRequest extends SensorReading {
  persist?: boolean
  threshold?: number | null
}

export interface AnomalyDetectionResponse {
  model_version: string
  anomaly_score: number
  threshold: number
  is_anomaly: boolean
  severity: Severity
  expected_value: Record<string, number>
  feature_deviation: FeatureContribution[]
  explanation: string
  model_trained: boolean
  anomaly_event_id: string | null
  created_at: string
}

export interface RULRequest {
  engine_unit: string
  cycle: number
  window?: number[][] | null
  dataset_subset?: 'FD001' | 'FD002' | 'FD003' | 'FD004'
  model_family?: 'xgboost' | 'lstm' | null
  persist?: boolean
}

export interface DegradationPoint {
  step: number
  cycle_offset: number
  rul: number
  health_signal: number
}

export interface RULResponse {
  model_version: string
  model_family: string
  dataset_subset: string
  engine_unit: string
  cycle: number
  predicted_rul: number
  lower_bound: number
  upper_bound: number
  uncertainty_pct: number
  confidence_note: string
  window_length: number
  degradation_curve: DegradationPoint[]
  model_trained: boolean
  rul_prediction_id: string | null
  created_at: string
}

export interface VisualInspectionResponse {
  inspection_id: string | null
  model_version: string | null
  model_trained: boolean
  is_anomalous: boolean | null
  anomaly_score: number | null
  threshold: number | null
  num_anomalous_patches: number | null
  localisation_pct: number | null
  category: string | null
  verdict: string
  message: string
  original_url: string | null
  heatmap_url: string | null
  overlay_url: string | null
  preprocessing: Record<string, unknown> | null
  created_at: string
}

export interface AgentStepTrace {
  step: number
  agent: string
  action: string
  summary: string
  status: AgentStatus
  engine: string
  duration_ms: number | null
  evidence: Record<string, unknown>
}

export interface AgentReport {
  agent: string
  role: string
  status: AgentStatus
  engine: string
  summary: string
  findings: Record<string, unknown>
  citations: unknown[]
  confidence: number | null
  caveats: string[]
}

export interface RootCause {
  hypothesis: string
  mechanism: string
  confidence: number
  evidence: Record<string, unknown>
  evidence_basis: string
  source: string
  rule_id?: string
  causality_note: string
}

export interface Recommendation {
  id: string
  title: string
  description: string
  priority: string
  action_type: string
  rationale?: string | null
  confidence?: number | null
  evidence?: unknown[] | null
  citations?: unknown[] | null
  estimated_downtime_min?: number | null
  status: string
  requires_human_approval?: boolean
  approved_by?: string | null
  approved_at?: string | null
  completed_by?: string | null
  completed_at?: string | null
  created_at: string
  machine_id?: string | null
  investigation_id?: string | null
}

export interface InvestigationRequest {
  machine_id?: string | null
  reading?: SensorReading | null
  alert_id?: string | null
  failure_probability?: number | null
  anomaly_score?: number | null
  rul_cycles?: number | null
  create_alerts?: boolean
  create_recommendations?: boolean
  max_steps?: number
}

export interface UnsupportedClaim {
  claim: string
  reason: string
  required_evidence: string
}

export interface InvestigationResponse {
  investigation_id: string
  status: string
  engine: string
  llm_enabled: boolean
  steps_executed: number
  max_steps_reached: boolean
  trace: AgentStepTrace[]
  reports: AgentReport[]
  root_causes: RootCause[]
  citations: Citation[]
  unsupported_claims: UnsupportedClaim[]
  recommendations: Recommendation[]
  verdict: string
  requires_human_approval: boolean
  alerts_created: string[]
  duration_ms: number
  started_at: string
  completed_at: string
}

export interface Citation {
  index: number
  document_id: string
  document_title: string | null
  filename: string
  page: number | null
  reference: string
  score: number
}

export interface KnowledgeUploadResponse {
  document_id: string
  filename: string
  title: string | null
  page_count: number | null
  chunk_count: number
  char_count: number
  embedding_model: string
  status: string
  index_size: number
  message: string
}

export interface RetrievedChunk {
  chunk_id: number
  document_id: string
  document_title: string | null
  filename: string
  page: number | null
  score: number
  text: string
}

export interface KnowledgeQueryResponse {
  question: string
  answer: string
  answer_engine: 'extractive' | 'llm_synthesised'
  citations: RetrievedChunk[]
  retrieved_count: number
  insufficient_evidence: boolean
  disclaimer: string
  created_at: string
}

export interface KnowledgeDocument {
  id: string
  filename: string
  title: string | null
  page_count: number | null
  chunk_count: number | null
  embedding_model: string | null
  status: string
  created_at: string
}

export interface IndexHealth {
  chunks: number
  documents: number
  embedding_model: string
  embedding_available: boolean
  backend: string
  dimension: number | null
  index_path: string
  status?: string
  detail?: string
}

export interface Alert {
  id: string
  machine_id: string | null
  created_at: string
  alert_type: string
  severity: Severity
  title: string
  message: string
  source: string
  is_read: boolean
  is_resolved: boolean
  payload: Record<string, unknown> | null
}

export interface ModelMetrics {
  model_version: string
  task: string
  algorithm: string
  dataset: string
  dataset_sha256?: string | null
  metrics: Record<string, unknown>
  feature_names?: string[] | null
  hyperparameters?: Record<string, unknown> | null
  training_samples?: number | null
  trained_at?: string | null
  artifact_path?: string | null
}

export interface AnalyticsResponse {
  generated_at: string
  failure_trend: TrendPoint[]
  risk_histogram: { bucket: string; count: number }[]
  anomaly_timeline: TrendPoint[]
  rul_distribution: { bucket: string; count: number; lower: number; upper: number }[]
  maintenance_efficiency: Record<string, number>
  downtime_by_machine: {
    machine_id: string
    machine_name: string
    downtime_min: number
    events: number
  }[]
  model_metrics: ModelMetrics[]
  dataset_manifest: {
    statuses: Record<string, DatasetStatus>
    provenance: Record<string, unknown>
  }
  disclaimer: string
}

export interface DatasetStatus {
  key: string
  title: string
  path: string
  available: boolean
  detail: string
  file_count: number
  bytes: number
  files: string[]
}

export interface TwinSimulationResponse {
  machine_id: string | null
  is_simulation: true
  disclaimer: string
  scenario: Record<string, number | string | boolean>
  baseline: Record<string, number | string>
  deltas_from_baseline: Record<string, number>
  derived_features: Record<string, number>
  failure: FailurePredictionResponse | null
  anomaly: AnomalyDetectionResponse | null
  comparison: {
    failure_probability: { baseline: number; scenario: number; delta: number }
    risk_category: { baseline: RiskCategory; scenario: RiskCategory }
  } | null
  unavailable_reason?: string
}

export interface TwinSnapshot {
  machine_id: string
  is_simulation: false
  measured: boolean
  health: MachineHealthSummary
  readings: Record<string, number | string | null> | null
  note: string | null
}

export interface AgentCatalog {
  agents: { key: string; label: string; description: string }[]
  graph: Record<string, unknown>
  guarantees: string[]
}

export interface RuntimeSettings {
  app: Record<string, string>
  database: Record<string, unknown>
  llm: Record<string, unknown>
  rag: Record<string, unknown>
  thresholds: Record<string, number>
  computer_vision: Record<string, number>
  ml_runtime: Record<string, unknown>
  datasets: Record<string, DatasetStatus>
  paths: Record<string, string>
}
