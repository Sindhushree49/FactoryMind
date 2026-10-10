import { CheckCircle2, History, ShieldCheck, ThumbsDown, Wrench } from 'lucide-react'
import { useState } from 'react'

import { Badge, EmptyState, ErrorBanner, Loading, Panel, StatCard } from '@/components/ui'
import { useAction, useAsync } from '@/hooks/useApi'
import {
  approveRecommendation,
  completeRecommendation,
  getMaintenanceHistory,
  getRecommendations,
  rejectRecommendation,
} from '@/lib/endpoints'
import type { Recommendation } from '@/types/api'

const FILTERS = [
  { key: '', label: 'All' },
  { key: 'pending_approval', label: 'Pending approval' },
  { key: 'approved', label: 'Approved' },
  { key: 'in_progress', label: 'In progress' },
  { key: 'completed', label: 'Completed' },
  { key: 'rejected', label: 'Rejected' },
]

/** Mirrors ALLOWED_TRANSITIONS in backend/app/services/maintenance_service.py. */
const NEXT_STATES: Record<string, string[]> = {
  pending_approval: ['approved', 'rejected'],
  approved: ['completed'],
  in_progress: ['completed'],
}

export default function MaintenancePage() {
  const [filter, setFilter] = useState('')
  const [operator, setOperator] = useState('')
  const [comment, setComment] = useState('')

  const recommendations = useAsync(() => getRecommendations(filter || undefined), [filter])
  const history = useAsync(() => getMaintenanceHistory(), [])

  const approve = useAction<[string, string, string?], Recommendation>(approveRecommendation)
  const reject = useAction<[string, string, string?], Recommendation>(rejectRecommendation)
  const complete = useAction<[string, string, string?], Recommendation>(completeRecommendation)

  const refresh = () => {
    recommendations.refetch()
    history.refetch()
  }

  const handle = (
    kind: 'approve' | 'reject' | 'complete',
    id: string,
  ): Promise<Recommendation | null> => {
    const who = operator.trim()
    if (!who) return Promise.resolve(null)
    const note = comment.trim() || undefined
    const action = kind === 'approve' ? approve : kind === 'reject' ? reject : complete
    return action.run(id, who, note).then((result) => {
      if (result) {
        setComment('')
        refresh()
      }
      return result
    })
  }

  const pending = approve.pending || reject.pending || complete.pending
  const actionError = approve.error ?? reject.error ?? complete.error
  const items = recommendations.data ?? []

  const counts = {
    total: items.length,
    pending: items.filter((item) => item.status === 'pending_approval').length,
    approved: items.filter((item) => item.status === 'approved').length,
    completed: items.filter((item) => item.status === 'completed').length,
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">Maintenance History</h1>
        <p className="mt-1 text-[13px] text-slate-400">
          The human approval gate. A recommendation cannot reach{' '}
          <span className="mono text-slate-300">completed</span> without first passing through{' '}
          <span className="mono text-slate-300">approved</span> — the transition table is enforced
          server-side and out-of-order moves are refused with HTTP 409.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Recommendations" value={counts.total} />
        <StatCard label="Awaiting approval" value={counts.pending} tone={counts.pending ? 'warn' : 'good'} />
        <StatCard label="Approved, not finished" value={counts.approved} tone="accent" />
        <StatCard label="Completed" value={counts.completed} tone="good" icon={<CheckCircle2 className="h-4 w-4" />} />
      </div>

      <Panel title="Sign-off" subtitle="Recorded against every transition you make">
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="label" htmlFor="operator">
              Your name (required to act)
            </label>
            <input
              id="operator"
              type="text"
              value={operator}
              onChange={(event) => setOperator(event.target.value)}
              className="input"
              placeholder="Qualified engineer"
            />
          </div>
          <div>
            <label className="label" htmlFor="comment">
              Comment (optional)
            </label>
            <input
              id="comment"
              type="text"
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              className="input"
              placeholder="Why this decision was taken"
            />
          </div>
        </div>
        {!operator.trim() && (
          <p className="mt-3 text-[11px] text-amber-300/80">
            Approval actions stay disabled until a name is supplied, because the backend stores it as
            the accountable party.
          </p>
        )}
      </Panel>

      <div className="flex flex-wrap gap-2">
        {FILTERS.map((item) => (
          <button
            key={item.key || 'all'}
            onClick={() => setFilter(item.key)}
            className={filter === item.key ? 'btn-primary' : 'btn-secondary'}
          >
            {item.label}
          </button>
        ))}
      </div>

      {Boolean(actionError) && <ErrorBanner error={actionError} />}
      {Boolean(recommendations.error) && (
        <ErrorBanner error={recommendations.error} onRetry={recommendations.refetch} />
      )}

      <Panel
        title="Approval queue"
        subtitle={
          filter ? `Filtered by ${filter.replace(/_/g, ' ')}` : 'Every recommendation, newest first'
        }
        actions={
          <button className="btn-ghost" onClick={recommendations.refetch}>
            Refresh
          </button>
        }
        dense
      >
        {recommendations.loading ? (
          <Loading label="Loading recommendations" />
        ) : !items.length ? (
          <EmptyState
            title="No recommendations"
            detail="Run an agentic investigation to generate maintenance proposals."
          />
        ) : (
          <div className="space-y-3 p-1">
            {items.map((rec) => (
              <RecommendationCard
                key={rec.id}
                rec={rec}
                operator={operator.trim()}
                pending={pending}
                onAct={(kind) => handle(kind, rec.id)}
              />
            ))}
          </div>
        )}
      </Panel>

      <Panel title="Maintenance log" subtitle="Every recorded transition, newest first" dense>
        {history.loading ? (
          <Loading label="Loading history" />
        ) : !history.data?.length ? (
          <EmptyState
            title="No maintenance events recorded"
            detail="Approving or completing work writes an immutable history entry."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[12px]">
              <thead>
                <tr className="border-b border-white/[0.08]">
                  {['Performed at', 'Event', 'Performed by', 'Downtime', 'Cost', 'Notes'].map(
                    (header) => (
                      <th
                        key={header}
                        className="px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500"
                      >
                        {header}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {history.data.map((record, index) => (
                  <tr key={index} className="border-b border-white/[0.04] last:border-0">
                    <td className="mono px-3 py-2 text-[10px] text-slate-500">
                      {asTime(record.performed_at)}
                    </td>
                    <td className="px-3 py-2">
                      <Badge value={asText(record.event_type)} />
                    </td>
                    <td className="px-3 py-2 text-slate-300">{asText(record.performed_by)}</td>
                    <td className="mono px-3 py-2 text-slate-400">{asNumber(record.downtime_min)}</td>
                    <td className="mono px-3 py-2 text-slate-400">{asNumber(record.cost)}</td>
                    <td className="max-w-md truncate px-3 py-2 text-slate-400">
                      {asText(record.notes, '—')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <div className="flex items-start gap-2 rounded-lg border border-risk-healthy/25 bg-risk-healthy/[0.06] p-3 text-[11.5px] leading-relaxed text-emerald-200">
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" />
        No agent, service or API route can move a recommendation straight to completed. Every
        transition below is a deliberate, attributed action.
      </div>
    </div>
  )
}

function RecommendationCard({
  rec,
  operator,
  pending,
  onAct,
}: {
  rec: Recommendation
  operator: string
  pending: boolean
  onAct: (kind: 'approve' | 'reject' | 'complete') => Promise<Recommendation | null>
}) {
  const next = NEXT_STATES[rec.status] ?? []

  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3.5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-[13.5px] font-semibold text-slate-100">{rec.title}</p>
          <p className="mt-1 text-[12px] leading-relaxed text-slate-400">{rec.description}</p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Badge value={rec.priority} />
          <Badge value={rec.status} />
        </div>
      </div>

      {rec.rationale && (
        <p className="mt-2 text-[11.5px] leading-relaxed text-slate-500">{rec.rationale}</p>
      )}

      <div className="mono mt-2.5 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-slate-600">
        <span>action {rec.action_type}</span>
        {rec.confidence !== undefined && rec.confidence !== null && (
          <span>confidence {rec.confidence.toFixed(2)}</span>
        )}
        {rec.estimated_downtime_min !== undefined && rec.estimated_downtime_min !== null && (
          <span>est. downtime {rec.estimated_downtime_min} min</span>
        )}
        <span>raised {new Date(rec.created_at).toLocaleString()}</span>
        {rec.approved_by && <span>approved by {rec.approved_by}</span>}
        {rec.completed_by && <span>completed by {rec.completed_by}</span>}
      </div>

      {rec.machine_id && (
        <a
          href="/machines"
          className="mt-2 inline-block text-[11px] font-semibold text-cyan-brand hover:underline"
        >
          {rec.machine_id} →
        </a>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-white/[0.06] pt-3">
        {next.includes('approved') && (
          <button
            className="btn-primary"
            disabled={pending || !operator}
            onClick={() => void onAct('approve')}
          >
            <ShieldCheck className="h-4 w-4" /> Approve
          </button>
        )}
        {next.includes('rejected') && (
          <button
            className="btn-danger"
            disabled={pending || !operator}
            onClick={() => void onAct('reject')}
          >
            <ThumbsDown className="h-4 w-4" /> Reject
          </button>
        )}
        {next.includes('completed') && (
          <button
            className="btn-secondary"
            disabled={pending || !operator}
            onClick={() => void onAct('complete')}
          >
            <Wrench className="h-4 w-4" /> Mark completed
          </button>
        )}
        {next.length === 0 && (
          <span className="flex items-center gap-1.5 text-[11px] text-slate-600">
            <History className="h-3.5 w-3.5" /> Terminal state — no further transitions permitted.
          </span>
        )}
      </div>
    </div>
  )
}

function asText(value: unknown, fallback = '—'): string {
  return value === null || value === undefined || value === '' ? fallback : String(value)
}

function asNumber(value: unknown): string {
  return typeof value === 'number' ? value.toLocaleString() : '—'
}

function asTime(value: unknown): string {
  const text = asText(value)
  const parsed = new Date(text)
  return Number.isNaN(parsed.getTime()) ? text : parsed.toLocaleString()
}
