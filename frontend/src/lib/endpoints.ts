/** Typed endpoint wrappers. One function per backend route. */

import api, { storageUrl } from '@/lib/api'
import type {
  AgentCatalog,
  Alert,
  AnalyticsResponse,
  Citation,
  DashboardSummary,
  DatasetStatus,
  HealthResponse,
  IndexHealth,
  InvestigationRequest,
  InvestigationResponse,
  KnowledgeDocument,
  KnowledgeQueryResponse,
  KnowledgeUploadResponse,
  Machine,
  MachineList,
  ModelMetrics,
  Recommendation,
  RULRequest,
  RULResponse,
  RuntimeSettings,
  SensorReading,
  TwinSimulationResponse,
  TwinSnapshot,
  VisualInspectionResponse,
  AnomalyDetectionRequest,
  AnomalyDetectionResponse,
  FailurePredictionRequest,
  FailurePredictionResponse,
} from '@/types/api'

/* -- system ---------------------------------------------------------------- */
export const getHealth = () => api.get<HealthResponse>('/health').then((r) => r.data)

export const getSettings = () => api.get<RuntimeSettings>('/settings').then((r) => r.data)

export const getDatasets = () =>
  api
    .get<{ statuses: Record<string, DatasetStatus>; manifest: Record<string, unknown> }>('/datasets')
    .then((r) => r.data)

/* -- dashboard & machines --------------------------------------------------- */
export const getDashboard = () => api.get<DashboardSummary>('/dashboard/summary').then((r) => r.data)

export const getMachines = (limit = 50) =>
  api.get<MachineList>('/machines', { params: { limit } }).then((r) => r.data)

export const getMachine = (id: string) => api.get<Machine>(`/machines/${id}`).then((r) => r.data)

export const seedMachines = (count = 8) =>
  api.post<{ created: string[]; count: number; source: string }>('/machines/seed', { count }).then((r) => r.data)

export const addReading = (machineId: string, reading: SensorReading) =>
  api
    .post<Machine>(`/machines/${machineId}/readings`, reading)
    .then((r) => r.data)

export const getMachineReadings = (machineId: string, limit = 200) =>
  api
    .get<{ readings: SensorReading[] }>(`/machines/${machineId}/readings`, { params: { limit } })
    .then((r) => r.data)

/* -- ML -------------------------------------------------------------------- */
export const predictFailure = (payload: FailurePredictionRequest) =>
  api.post<FailurePredictionResponse>('/predict/failure', payload).then((r) => r.data)

export const detectAnomaly = (payload: AnomalyDetectionRequest) =>
  api.post<AnomalyDetectionResponse>('/detect/anomaly', payload).then((r) => r.data)

export const predictRul = (payload: RULRequest) =>
  api.post<RULResponse>('/predict/rul', payload).then((r) => r.data)

export const getModelMetrics = () =>
  api
    .get<{ items: ModelMetrics[]; total: number; active_models: number }>('/models/metrics')
    .then((r) => r.data)

export const getAnalytics = () => api.get<AnalyticsResponse>('/analytics').then((r) => r.data)

/* -- vision ---------------------------------------------------------------- */
export const inspectImage = (file: File, category?: string, machineId?: string) => {
  const form = new FormData()
  form.append('file', file)
  if (category) form.append('category', category)
  if (machineId) form.append('machine_id', machineId)
  return api
    .post<VisualInspectionResponse>('/vision/inspect', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data)
}

export const getVisionCategories = () =>
  api
    .get<{
      trained_categories: string[]
      loaded: boolean
      detail: string
      dataset: string
      dataset_path: string
      model: string
      expected_categories: string[]
    }>('/vision/categories')
    .then((r) => r.data)

/* -- agents ---------------------------------------------------------------- */
export const investigate = (payload: InvestigationRequest) =>
  api.post<InvestigationResponse>('/agents/investigate', payload).then((r) => r.data)

export const getAgentCatalog = () => api.get<AgentCatalog>('/agents/catalog').then((r) => r.data)

export const getInvestigations = (limit = 25) =>
  api
    .get<{ items: Record<string, unknown>[]; total: number }>('/agents/investigations', {
      params: { limit },
    })
    .then((r) => r.data)

/* -- knowledge ------------------------------------------------------------- */
export const uploadDocument = (file: File, title?: string) => {
  const form = new FormData()
  form.append('file', file)
  if (title) form.append('title', title)
  return api
    .post<KnowledgeUploadResponse>('/knowledge/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data)
}

export const queryKnowledge = (question: string, topK = 5) =>
  api
    .post<KnowledgeQueryResponse>('/knowledge/query', { question, top_k: topK })
    .then((r) => r.data)

export const getDocuments = () =>
  api.get<KnowledgeDocument[]>('/knowledge/documents').then((r) => r.data)

export const deleteDocument = (id: string) => api.delete(`/knowledge/documents/${id}`).then(() => true)

export const getIndexHealth = () => api.get<IndexHealth>('/knowledge/status').then((r) => r.data)

/* -- maintenance & alerts -------------------------------------------------- */
export const getRecommendations = (status?: string) =>
  api
    .get<Recommendation[]>('/maintenance/recommendations', { params: status ? { status } : {} })
    .then((r) => r.data)

export const approveRecommendation = (id: string, approver: string, comment?: string) =>
  api
    .post<Recommendation>(`/maintenance/${id}/approve`, { approver, comment })
    .then((r) => r.data)

export const rejectRecommendation = (id: string, approver: string, comment?: string) =>
  api
    .post<Recommendation>(`/maintenance/${id}/reject`, { approver, comment })
    .then((r) => r.data)

export const completeRecommendation = (id: string, performedBy: string, notes?: string) =>
  api
    .post<Recommendation>(`/maintenance/${id}/complete`, { performed_by: performedBy, notes })
    .then((r) => r.data)

export const getMaintenanceHistory = (machineId?: string) =>
  api
    .get<Record<string, unknown>[]>('/maintenance/history', {
      params: machineId ? { machine_id: machineId } : {},
    })
    .then((r) => r.data)

export const getAlerts = (unresolvedOnly = false) =>
  api
    .get<Alert[]>('/alerts', { params: unresolvedOnly ? { unresolved_only: true } : {} })
    .then((r) => r.data)

export const resolveAlert = (id: string) =>
  api.post<Alert>(`/alerts/${id}/resolve`).then((r) => r.data)

/* -- digital twin ---------------------------------------------------------- */
export const simulateTwin = (payload: Record<string, number | string | null>) =>
  api.post<TwinSimulationResponse>('/twin/simulate', payload).then((r) => r.data)

export const getTwinSnapshot = (machineId: string) =>
  api.get<TwinSnapshot>(`/twin/${machineId}/snapshot`).then((r) => r.data)

/* -- helpers --------------------------------------------------------------- */
export const citationLabel = (citation: Citation | { filename: string; page?: number | null }) => {
  const page = 'page' in citation ? citation.page : null
  return `${citation.filename}${page ? ` p.${page}` : ''}`
}

export { storageUrl }
