import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from 'lucide-react'
import type { ReactNode } from 'react'

import { ApiError } from '@/lib/api'

/* -------------------------------------------------------------------------- */
/* Panel                                                                      */
/* -------------------------------------------------------------------------- */
export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = '',
  dense = false,
}: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  dense?: boolean
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-white/[0.06] px-5 py-4">
          <div className="min-w-0">
            {title && <h2 className="section-title truncate">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[12px] text-slate-400">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={dense ? 'p-3' : 'p-5'}>{children}</div>
    </section>
  )
}

/* -------------------------------------------------------------------------- */
/* StatCard                                                                   */
/* -------------------------------------------------------------------------- */
export function StatCard({
  label,
  value,
  unit,
  hint,
  icon,
  tone = 'neutral',
  trend,
}: {
  label: string
  value: ReactNode
  unit?: string
  hint?: ReactNode
  icon?: ReactNode
  tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'accent'
  trend?: { value: number; label: string }
}) {
  const tones: Record<string, string> = {
    neutral: 'text-slate-50',
    good: 'text-risk-healthy',
    warn: 'text-risk-watch',
    bad: 'text-risk-critical',
    accent: 'text-cyan-brand',
  }
  return (
    <div className="card card-hover p-4">
      <div className="flex items-start justify-between gap-3">
        <span className="stat-label">{label}</span>
        {icon && <span className="text-slate-600">{icon}</span>}
      </div>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className={`text-2xl font-bold tabular-nums ${tones[tone]}`}>{value}</span>
        {unit && <span className="text-[12px] text-slate-500">{unit}</span>}
      </div>
      {hint && <div className="mt-1 text-[11px] leading-snug text-slate-500">{hint}</div>}
      {trend && (
        <div className="mt-2 text-[11px] text-slate-400">
          <span className={trend.value >= 0 ? 'text-risk-high' : 'text-risk-healthy'}>
            {trend.value >= 0 ? '+' : ''}
            {trend.value.toFixed(3)}
          </span>{' '}
          {trend.label}
        </div>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Badges                                                                     */
/* -------------------------------------------------------------------------- */
const RISK_CLASS: Record<string, string> = {
  healthy: 'badge-healthy',
  watch: 'badge-watch',
  high: 'badge-high',
  critical: 'badge-critical',
  info: 'badge-info',
  warning: 'badge-watch',
  low: 'badge-info',
  ok: 'badge-healthy',
  supported: 'badge-healthy',
  conditional: 'badge-watch',
  insufficient_evidence: 'badge-high',
  blocked: 'badge-critical',
  skipped: 'badge-info',
  error: 'badge-critical',
  pending_approval: 'badge-watch',
  approved: 'badge-info',
  in_progress: 'badge-info',
  completed: 'badge-healthy',
  rejected: 'badge-critical',
  cancelled: 'badge-info',
  model_not_trained: 'badge-high',
  category_mismatch: 'badge-high',
  defect_detected: 'badge-critical',
  within_tolerance: 'badge-healthy',
}

export function Badge({ value, label }: { value: string; label?: string }) {
  const cls = RISK_CLASS[value] ?? 'badge-info'
  return <span className={cls}>{label ?? value.replace(/_/g, ' ')}</span>
}

export function RiskBadge({ value }: { value: string }) {
  return <Badge value={value} label={value.toUpperCase()} />
}

/* -------------------------------------------------------------------------- */
/* Status messages                                                            */
/* -------------------------------------------------------------------------- */
export function ErrorBanner({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  if (!error) return null
  const api = error instanceof ApiError ? error : null
  const message = api?.message ?? (error as Error)?.message ?? 'Unknown error'
  const remediation = api?.remediation

  return (
    <div className="rounded-xl border border-risk-critical/30 bg-risk-critical/[0.08] p-4">
      <div className="flex items-start gap-3">
        <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-risk-critical" />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-risk-critical">
            {api ? `${api.code} (HTTP ${api.status})` : 'Request failed'}
          </p>
          <p className="mt-1 text-[13px] leading-relaxed text-slate-300">{message}</p>
          {remediation && (
            <pre className="mono mt-2 overflow-x-auto rounded-md border border-white/10 bg-navy-950/70 p-2.5 text-[11px] text-cyan-300">
              {remediation}
            </pre>
          )}
          {onRetry && (
            <button type="button" onClick={onRetry} className="btn-secondary mt-3">
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export function Notice({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'warn' | 'good'
  title?: ReactNode
  children: ReactNode
}) {
  const map = {
    info: { cls: 'border-azure-brand/25 bg-azure-brand/[0.07] text-azure-100', Icon: Info },
    warn: { cls: 'border-amber-500/30 bg-amber-500/[0.07] text-amber-200', Icon: AlertTriangle },
    good: { cls: 'border-risk-healthy/30 bg-risk-healthy/[0.07] text-emerald-200', Icon: CheckCircle2 },
  }
  const { cls, Icon } = map[tone]
  return (
    <div className={`rounded-xl border p-4 ${cls}`}>
      <div className="flex items-start gap-3">
        <Icon className="mt-0.5 h-4 w-4 shrink-0" />
        <div className="min-w-0 flex-1 text-[12px] leading-relaxed">
          {title && <p className="mb-1 text-[13px] font-semibold">{title}</p>}
          {children}
        </div>
      </div>
    </div>
  )
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-3 py-12 text-slate-500">
      <Loader2 className="h-5 w-5 animate-spin text-cyan-brand" />
      <span className="text-sm">{label}...</span>
    </div>
  )
}

export function EmptyState({ title, detail, action }: { title: string; detail?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-white/10 px-6 py-12 text-center">
      <p className="text-sm font-semibold text-slate-300">{title}</p>
      {detail && <p className="max-w-md text-[12px] leading-relaxed text-slate-500">{detail}</p>}
      {action}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Table                                                                      */
/* -------------------------------------------------------------------------- */
export function Table({
  columns,
  rows,
  empty = 'No rows',
}: {
  columns: { key: string; header: ReactNode; align?: 'left' | 'right' | 'center'; width?: string }[]
  rows: Record<string, ReactNode>[]
  empty?: string
}) {
  if (!rows.length) return <EmptyState title={empty} />
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-[12px]">
        <thead>
          <tr className="border-b border-white/[0.08]">
            {columns.map((col) => (
              <th
                key={col.key}
                style={{ width: col.width }}
                className={`px-3 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500 ${
                  col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : ''
                }`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={index}
              className="border-b border-white/[0.04] transition-colors last:border-0 hover:bg-white/[0.02]"
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={`px-3 py-2.5 text-slate-300 ${
                    col.align === 'right' ? 'text-right tabular-nums' : col.align === 'center' ? 'text-center' : ''
                  }`}
                >
                  {row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Gauge                                                                      */
/* -------------------------------------------------------------------------- */
export function Gauge({
  value,
  label,
  size = 132,
  tone = 'auto',
}: {
  value: number
  label?: string
  size?: number
  tone?: 'auto' | 'accent'
}) {
  const pct = Math.max(0, Math.min(1, value))
  const stroke = 9
  const radius = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius
  const colour =
    tone === 'accent'
      ? '#22d3ee'
      : pct >= 0.82
        ? '#ef4444'
        : pct >= 0.6
          ? '#f97316'
          : pct >= 0.3
            ? '#eab308'
            : '#10b981'

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#1d242f" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={colour}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - pct)}
          style={{ transition: 'stroke-dashoffset 700ms cubic-bezier(0.4,0,0.2,1), stroke 300ms' }}
        />
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dominantBaseline="central"
          fill="#f1f5f9"
          fontSize={size * 0.24}
          fontWeight={700}
          transform={`rotate(90 ${size / 2} ${size / 2})`}
        >
          {(pct * 100).toFixed(0)}%
        </text>
      </svg>
      {label && <span className="mt-1 text-[11px] uppercase tracking-wider text-slate-500">{label}</span>}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Bar                                                                        */
/* -------------------------------------------------------------------------- */
export function Meter({
  value,
  max = 1,
  colour = '#22d3ee',
  height = 6,
}: {
  value: number
  max?: number
  colour?: string
  height?: number
}) {
  const pct = Math.max(0, Math.min(100, (value / (max || 1)) * 100))
  return (
    <div className="w-full overflow-hidden rounded-full bg-charcoal-600" style={{ height }}>
      <div
        className="h-full rounded-full"
        style={{ width: `${pct}%`, background: colour, transition: 'width 500ms ease' }}
      />
    </div>
  )
}
