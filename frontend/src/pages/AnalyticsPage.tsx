import { BarChart3, Download, RefreshCw } from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { EmptyState, ErrorBanner, Loading, Meter, Notice, Panel, StatCard, Table } from '@/components/ui'
import { useAsync } from '@/hooks/useApi'
import { getAnalytics, getModelMetrics } from '@/lib/endpoints'
import type { AnalyticsResponse } from '@/types/api'

const RISK_COLOURS: Record<string, string> = {
  healthy: '#10b981',
  watch: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
  unknown: '#64748b',
}

const TOOLTIP_STYLE = {
  backgroundColor: '#161c25',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 8,
  fontSize: 12,
  color: '#e2e8f0',
}

export default function AnalyticsPage() {
  const analytics = useAsync<AnalyticsResponse>(() => getAnalytics(), [])
  const metrics = useAsync(() => getModelMetrics(), [])

  if (analytics.loading && !analytics.data) return <Loading label="Computing analytics" />
  if (!analytics.data) return <ErrorBanner error={analytics.error} onRetry={analytics.refetch} />

  const data = analytics.data
  const efficiency = data.maintenance_efficiency
  const trend = data.failure_trend.map((point) => ({
    bucket: point.bucket.slice(5),
    probability: point.value,
    predictions: point.count,
  }))
  const anomalies = data.anomaly_timeline.map((point) => ({
    bucket: point.bucket.slice(5),
    events: point.count,
  }))

  const metricRows = data.model_metrics.map((model) => ({
    version: model.model_version,
    task: model.task,
    algorithm: model.algorithm,
    dataset: model.dataset,
    samples: model.training_samples ?? '—',
    metrics: Object.entries(model.metrics)
      .slice(0, 4)
      .map(([key, value]) => `${key} ${formatMetric(value)}`)
      .join(' · ') || '—',
    trained: model.trained_at ? new Date(model.trained_at).toLocaleString() : '—',
  }))

  const datasetRows = Object.values(data.dataset_manifest.statuses).map((status) => ({
    dataset: status.title,
    key: status.key,
    available: status.available ? 'available' : 'missing',
    files: `${status.file_count} · ${(status.bytes / 1048576).toFixed(1)} MB`,
    detail: status.detail,
  }))

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-50">Analytics &amp; Reports</h1>
          <p className="mt-1 text-[13px] text-slate-400">
            Fleet trends, maintenance efficiency, model evaluation metrics and dataset provenance.
            Generated {new Date(data.generated_at).toLocaleString()}.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn-secondary" onClick={analytics.refetch}>
            <RefreshCw className="h-4 w-4" /> Refresh
          </button>
          <button
            className="btn-secondary"
            onClick={() => {
              const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
              const url = URL.createObjectURL(blob)
              const link = document.createElement('a')
              link.href = url
              link.download = `factorymind-analytics-${Date.now()}.json`
              link.click()
              URL.revokeObjectURL(url)
            }}
          >
            <Download className="h-4 w-4" /> Export
          </button>
        </div>
      </div>

      {Boolean(analytics.error) && <ErrorBanner error={analytics.error} onRetry={analytics.refetch} />}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Recommendations completed"
          value={efficiency.recommendations_completed ?? 0}
          tone="good"
        />
        <StatCard
          label="Approval rate"
          value={`${((efficiency.approval_rate ?? 0) * 100).toFixed(0)}%`}
          hint={`${efficiency.recommendations_approved ?? 0} approved of all raised`}
          tone="accent"
          icon={<BarChart3 className="h-4 w-4" />}
        />
        <StatCard label="Recorded downtime" value={efficiency.downtime_min_total ?? 0} unit="min" />
        <StatCard label="Recorded cost" value={(efficiency.cost_total ?? 0).toFixed(2)} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Failure risk trend" subtitle="Mean stored failure probability per day">
          {trend.length === 0 ? (
            <EmptyState
              title="No prediction history"
              detail="Run predictions to populate the 14-day series."
            />
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="bucket" stroke="#64748b" fontSize={10} tickLine={false} />
                <YAxis
                  stroke="#64748b"
                  fontSize={10}
                  tickLine={false}
                  domain={[0, 'auto']}
                  tickFormatter={(v: number) => v.toFixed(2)}
                />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Line
                  type="monotone"
                  dataKey="probability"
                  stroke="#22d3ee"
                  strokeWidth={2}
                  dot={{ r: 2.5, fill: '#22d3ee' }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Panel>

        <Panel title="Risk histogram" subtitle="Stored predictions by risk band">
          {data.risk_histogram.length === 0 ? (
            <EmptyState title="No stored predictions" />
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={data.risk_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="bucket" stroke="#64748b" fontSize={10} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={10} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {data.risk_histogram.map((entry) => (
                    <Cell key={entry.bucket} fill={RISK_COLOURS[entry.bucket] ?? '#64748b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Panel>

        <Panel title="Anomaly timeline" subtitle="Detected anomaly events per day">
          {anomalies.length === 0 ? (
            <EmptyState title="No anomaly events stored" />
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={anomalies}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="bucket" stroke="#64748b" fontSize={10} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={10} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Bar dataKey="events" fill="#f97316" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Remaining-life distribution" subtitle="Stored RUL predictions in cycle bands">
          {data.rul_distribution.length === 0 ? (
            <EmptyState
              title="No RUL predictions"
              detail="Run a remaining-useful-life prediction to populate this histogram."
            />
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={data.rul_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="bucket" stroke="#64748b" fontSize={10} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={10} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Panel>

        <Panel title="Downtime by machine" subtitle="Sum of recorded downtime minutes">
          {data.downtime_by_machine.length === 0 ? (
            <EmptyState
              title="No downtime recorded"
              detail="Completing approved maintenance work writes downtime into the ledger."
            />
          ) : (
            <ul className="space-y-3">
              {data.downtime_by_machine.map((row) => {
                const worst = Math.max(
                  ...data.downtime_by_machine.map((item) => item.downtime_min),
                  1,
                )
                return (
                  <li key={row.machine_id}>
                    <div className="flex items-center justify-between gap-3 text-[12px]">
                      <span className="truncate text-slate-300">{row.machine_name}</span>
                      <span className="tabular-nums text-slate-400">
                        {row.downtime_min} min · {row.events} event{row.events === 1 ? '' : 's'}
                      </span>
                    </div>
                    <div className="mt-1.5">
                      <Meter value={row.downtime_min} max={worst} colour="#22d3ee" />
                    </div>
                  </li>
                )
              })}
            </ul>
          )}
        </Panel>
      </div>

      <Panel
        title="Model evaluation metrics"
        subtitle={
          metrics.data
            ? `${metrics.data.active_models} active of ${metrics.data.total} registered versions`
            : 'Loading registry'
        }
        dense
      >
        {metrics.loading && !metrics.data ? (
          <Loading label="Loading model registry" />
        ) : metricRows.length === 0 ? (
          <EmptyState
            title="No trained models registered"
            detail="Training scripts write a manifest per version; none has been run yet."
          />
        ) : (
          <Table
            columns={[
              { key: 'version', header: 'Version' },
              { key: 'task', header: 'Task' },
              { key: 'algorithm', header: 'Algorithm' },
              { key: 'dataset', header: 'Dataset' },
              { key: 'samples', header: 'Samples', align: 'right' },
              { key: 'metrics', header: 'Headline metrics' },
              { key: 'trained', header: 'Trained' },
            ]}
            rows={metricRows}
          />
        )}
      </Panel>

      <Panel title="Dataset provenance" subtitle="Availability of every registered dataset" dense>
        {datasetRows.length === 0 ? (
          <EmptyState title="No datasets registered" />
        ) : (
          <Table
            columns={[
              { key: 'dataset', header: 'Dataset' },
              { key: 'key', header: 'Key' },
              { key: 'available', header: 'Status' },
              { key: 'files', header: 'Files · size', align: 'right' },
              { key: 'detail', header: 'Detail' },
            ]}
            rows={datasetRows}
          />
        )}
      </Panel>

      <Notice tone="info" title="What these numbers mean">
        <p>{data.disclaimer}</p>
      </Notice>
    </div>
  )
}

function formatMetric(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4)
  if (value === null || value === undefined) return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
