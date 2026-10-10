/**
 * Minimal data-fetching hooks.
 *
 * Deliberately dependency-free (no react-query) so the install stays lean, while
 * still giving loading / error / refetch states and cancellation on unmount.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError } from '@/lib/api'

export interface AsyncState<T> {
  data: T | null
  error: unknown
  loading: boolean
  refetch: () => void
}

export function useAsync<T>(
  fn: () => Promise<T>,
  deps: unknown[] = [],
  intervalMs = 0,
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)
  const active = useRef(true)
  const fnRef = useRef(fn)
  fnRef.current = fn

  useEffect(() => {
    active.current = true

    const run = (showSpinner: boolean) => {
      if (showSpinner) setLoading(true)
      fnRef
        .current()
        .then((result) => {
          if (active.current) {
            setData(result)
            setError(null)
          }
        })
        .catch((err: unknown) => {
          if (active.current) setError(err)
        })
        .finally(() => {
          if (active.current) setLoading(false)
        })
    }

    run(true)
    let timer: ReturnType<typeof setInterval> | null = null
    if (intervalMs > 0) {
      timer = setInterval(() => run(false), intervalMs)
    }
    return () => {
      active.current = false
      if (timer) clearInterval(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const refetch = useCallback(() => setNonce((n) => n + 1), [])
  return { data, error, loading, refetch }
}

/** Tracks an async action (form submit, file upload) with its own status. */
export function useAction<Args extends unknown[], R>(
  action: (...args: Args) => Promise<R>,
): {
  run: (...args: Args) => Promise<R | null>
  pending: boolean
  error: unknown
  result: R | null
  reset: () => void
} {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [result, setResult] = useState<R | null>(null)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  const run = useCallback(
    async (...args: Args) => {
      setPending(true)
      setError(null)
      try {
        const value = await action(...args)
        if (alive.current) setResult(value)
        return value
      } catch (err) {
        if (alive.current) setError(err)
        return null
      } finally {
        if (alive.current) setPending(false)
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  const reset = useCallback(() => {
    setError(null)
    setResult(null)
  }, [])

  return { run, pending, error, result, reset }
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return 'Unknown error'
}
