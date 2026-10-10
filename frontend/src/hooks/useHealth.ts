import { getHealth } from '@/lib/endpoints'
import { useAsync } from './useApi'
import type { HealthResponse } from '@/types/api'

/** Poll the health endpoint so the sidebar stays honest about degraded subsystems. */
export function useHealth(intervalMs = 30_000): {
  data: HealthResponse | null
  error: unknown
  loading: boolean
  refetch: () => void
} {
  return useAsync<HealthResponse>(() => getHealth(), [], intervalMs)
}
