/**
 * Axios client for the FactoryMind API.
 *
 * Errors are normalised into an `ApiError` carrying the backend's structured
 * detail (error code, message, remediation command) so the UI can show the
 * actual remedy instead of "something went wrong".
 */

import axios, { AxiosError, AxiosInstance } from 'axios'

const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  status: number
  code: string
  remediation?: string
  details?: unknown

  constructor(message: string, status: number, code = 'error', remediation?: string, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.remediation = remediation
    this.details = details
  }
}

function normalise(error: AxiosError): ApiError {
  const status = error.response?.status ?? 0
  const data = error.response?.data
  if (data && typeof data === 'object') {
    const record = data as Record<string, unknown>
    const detail =
      typeof record.detail === 'object' && record.detail !== null
        ? (record.detail as Record<string, unknown>)
        : record
    return new ApiError(
      (detail.message as string) ??
        (detail.error as string) ??
        error.message ??
        'Request failed',
      status,
      (detail.error as string) ?? 'http_error',
      detail.remediation as string | undefined,
      detail.details ?? record,
    )
  }
  if (error.code === 'ERR_NETWORK') {
    return new ApiError(
      'Cannot reach the FactoryMind API. Start the backend with: uvicorn app.main:app --reload',
      0,
      'network_error',
      'cd backend && uvicorn app.main:app --reload --port 8000',
    )
  }
  return new ApiError(error.message || 'Request failed', status, 'request_error')
}

export const api: AxiosInstance = axios.create({
  baseURL: `${BASE}/api`,
  timeout: 120_000,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => Promise.reject(normalise(error)),
)

/** Absolute URL for a backend-served static file. */
export function storageUrl(path: string | null | undefined): string | null {
  if (!path) return null
  if (path.startsWith('http') || path.startsWith('data:')) return path
  return `${BASE}${path}`
}

export default api
