import {
  Activity,
  AlertTriangle,
  Bot,
  Boxes,
  Cpu,
  Download,
  RefreshCw,
  ShieldCheck,
  TrendingUp,
  Wrench,
} from 'lucide-react'
import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { ErrorBanner, Loading, Meter, Panel, StatCard, EmptyState, Notice } from '@/components/ui'
import { useAsync } from '@/hooks/useApi'
import { getDashboard, getAlerts, getMachines, seedMachines } from '@/lib/endpoints'
import type { DashboardSummary, Machine } from '@/types/api'

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

export default function DashboardPage() {
  const dashboard = useAsync<DashboardSummary>(() => getDashboard(), [])
  const machines = useAsync(() => getMachines(50), [])
  const alerts = useAsync(() => getAlerts(false), [])

  if (dashboard.loading && !dashboard.data) return <Loading label="Loading executive dashboard" />

  const data = dashboard.data
  if (!data) return <ErrorBanner error={dashboard.error} onRetry={dashboard.refetch} />

  const distribution = data.health_distribution
  const distributionData = Object.entries(distribution).map(([key, value]) => ({
    name: key,
    value,
    fill: RISK_COLOURS[key] ?? '#64748b',
  }))

  const trend = data.failure_trend.map((point) => ({
    bucket: point.bucket.slice(5),
    probability: point.value,
    predictions: point.count,
  }))

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-50">Executive Dashboard</h1>
          <p className="mt-1 text-[13px] text-slate-400">
            Fleet health, failure risk and agent activity. Generated{' '}
            {new Date(data.generated_at).toLocaleString()}.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn-secondary" onClick={dashboard.refetch}>
            <RefreshCw className="h-4 w-4" /> Refresh
          </button>
          <button
            className="btn-secondary"
            onClick={() => {
              const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
              const url = URL.createObjectURL(blob)
              const link = document.createElement('a')
              link.href = url
              link.download = `factorymind-dashboard-${Date.now()}.json`
              link.click()
              URL.revokeObjectURL(url)
            }}
          >
            <Download className="h-4 w-4" /> Export
          </button>
        </div>
      </div>

      {Boolean(dashboard.error) && <ErrorBanner error={dashboard.error} onRetry={dashboard.refetch} />}

      {data.total_machines === 0 && (
        <Notice tone="warn" title="No machines registered">
          <p>
            The dashboard is empty because no machines exist yet. Seed machines from real AI4I
            telemetry — the readings are verbatim dataset rows, nothing is synthesised.
          </p>
          <button
            className="btn-primary mt-3"
            disabled={machines.loading}
            onClick={() => seedMachines(8).then(dashboard.refetch)}
          >
            <Boxes className="h-4 w-4" /> Seed 8 machines from AI4I
          </button>
        </Notice>
      )}

      {/* KPI row */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Total machines"
          value={data.total_machines}
          icon={<Cpu className="h-4 w-4" />}
          hint={`${machines.data?.total ?? 0} returned by the registry`}
        />
        <StatCard
          label="Healthy machines"
          value={data.healthy_machines}
          tone="good"
          icon={<ShieldCheck className="h-4 w-4" />}
          hint={`${distribution.healthy} nominal · ${distribution.watch} watch`}
        />
        <StatCard
          label="Machines at risk"
          value={data.machines_at_risk}
          tone={data.machines_at_risk > 0 ? 'bad' : 'good'}
          icon={<AlertTriangle className="h-4 w-4" />}
          hint={`${distribution.high} high · ${distribution.critical} critical`}
        />
        <StatCard
          label="Critical alerts"
          value={data.critical_alerts}
          tone={data.critical_alerts > 0 ? 'bad' : 'good'}
          icon={<AlertTriangle className="h-4 w-4" />}
          hint={`${data.open_alerts} unresolved · ${data.pending_approvals} awaiting approval`}
        />
      </div>

      {/* Second row */}
      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Health distribution" subtitle="Latest stored prediction per machine">
          {distributionData.every((d) => d.value === 0) ? (
            <EmptyState
              title="No predictions stored yet"
              detail="Run a failure prediction from the Predictive Maintenance page to populate this chart."
            />
          ) : (
            <div className="flex items-center gap-4">
              <ResponsiveContainer width="52%" height={190}>
                <PieChart>
                  <Pie
                    data={distributionData.filter((d) => d.value > 0)}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={48}
                    outerRadius={76}
                    paddingAngle={3}
                    stroke="none"
                  >
                    {distributionData
                      .filter((d) => d.value > 0)
                      .map((entry) => (
                        <Cell key={entry.name} fill={entry.fill} />
                      ))}
                  </Pie>
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                </PieChart>
              </ResponsiveContainer>
              <ul className="flex-1 space-y-2">
                {distributionData.map((item) => (
                  <li key={item.name} className="flex items-center justify-between text-[12px]">
                    <span className="flex items-center gap-2 text-slate-400">
                      <span
                        className="h-2.5 w-2.5 rounded-sm"
                        style={{ background: item.fill }}
                      />
                      {item.name}
                    </span>
                    <span className="tabular-nums font-semibold text-slate-200">{item.value}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Panel>

        <Panel
          title="Failure risk trend"
          subtitle="Mean failure probability of stored predictions per day"
        >
          {trend.length === 0 ? (
            <EmptyState
              title="No prediction history"
              detail="Submit sensor readings to build the trend series."
            />
          ) : (
            <ResponsiveContainer width="100%" height={190}>
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

        <Panel title="Operational pulse" subtitle="24-hour activity">
          <ul className="space-y-3.5">
            <PulseRow
              icon={<Activity className="h-4 w-4" />}
              label="Anomalies detected"
              value={data.anomalies_last_24h}
              colour="#f97316"
            />
            <PulseRow
              icon={<Bot className="h-4 w-4" />}
              label="Agent investigations"
              value={data.investigations_last_24h}
              colour="#22d3ee"
            />
            <PulseRow
              icon={<Wrench className="h-4 w-4" />}
              label="Pending approvals"
              value={data.pending_approvals}
              colour="#eab308"
            />
            <PulseRow
              icon={<TrendingUp className="h-4 w-4" />}
              label="Unresolved inspections"
              value={data.unresolved_inspections}
              colour="#ef4444"
            />
          </ul>
          <div className="mt-5 space-y-2.5 border-t border-white/[0.06] pt-4">
            <div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Mean failure probability</span>
                <span className="tabular-nums text-slate-200">
                  {data.mean_failure_probability === null
                    ? 'n/a'
                    : data.mean_failure_probability.toFixed(4)}
                </span>
              </div>
              <div className="mt-1.5">
                <Meter
                  value={data.mean_failure_probability ?? 0}
                  colour="#ef4444"
                />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Mean remaining useful life</span>
                <span className="tabular-nums text-slate-200">
                  {data.mean_rul_cycles === null ? 'n/a' : `${data.mean_rul_cycles.toFixed(0)} cycles`}
                </span>
              </div>
              <div className="mt-1.5">
                <Meter value={data.mean_rul_cycles ?? 0} max={125} colour="#10b981" />
              </div>
            </div>
          </div>
        </Panel>
      </div>

      {/* Machines + activity */}
      <div className="grid gap-4 lg:grid-cols-5">
        <Panel
          className="lg:col-span-3"
          title="Machine fleet"
          subtitle={`${machines.data?.total ?? 0} registered assets`}
          actions={<Link to="/machines">View all</Link>}
        >
          {machines.loading ? (
            <Loading label="Loading machines" />
          ) : !machines.data?.items.length ? (
            <EmptyState
              title="No machines registered"
              detail="Use POST /api/machines or seed from AI4I telemetry."
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-[12px]">
                <thead>
                  <tr className="border-b border-white/[0.08]">
                    <th className="px-2 py-2 text-[11px] uppercase tracking-wider text-slate-500">Asset</th>
                    <th className="px-2 py-2 text-[11px] uppercase tracking-wider text-slate-500">Type</th>
                    <th className="px-2 py-2 text-right text-[11px] uppercase tracking-wider text-slate-500">Fail prob</th>
                    <th className="px-2 py-2 text-[11px] uppercase tracking-wider text-slate-500">RUL</th>
                    <th className="px-2 py-2 text-[11px] uppercase tracking-wider text-slate-500">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {machines.data.items.slice(0, 10).map((machine) => (
                    <MachineRow key={machine.id} machine={machine} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <Panel
          className="lg:col-span-2"
          title="Recent activity"
          subtitle="Alerts, inspections, investigations and maintenance"
        >
          {data.recent_activities.length === 0 ? (
            <EmptyState
              title="No activity recorded"
              detail="Run an investigation or submit telemetry to generate events."
            />
          ) : (
            <ul className="max-h-[380px] space-y-3 overflow-y-auto pr-1">
              {data.recent_activities.map((item, index) => (
                <li key={index} className="flex gap-3 border-l-2 border-white/[0.08] pl-3">
                  <span
                    className="mt-1 h-2 w-2 shrink-0 rounded-full"
                    style={{ background: RISK_COLOURS[item.severity] ?? '#64748b' }}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[12px] font-semibold text-slate-200">{item.title}</p>
                    {item.detail && (
                      <p className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-slate-500">
                        {item.detail}
                      </p>
                    )}
                    <p className="mono mt-1 text-[10px] text-slate-600">
                      {new Date(item.occurred_at).toLocaleString()}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <p className="text-[11px] leading-relaxed text-slate-600">{data.data_provenance}</p>

      {alerts.data && alerts.data.length > 0 && (
        <Panel title="Alert log" subtitle={`${alerts.data.length} most recent alerts`} dense>
          <AlertTable alerts={alerts.data} />
        </Panel>
      )}
    </div>
  )
}

function MachineRow({ machine }: { machine: Machine }) {
  const health = machine.health
  const probability = health?.failure_probability
  return (
    <tr className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.02]">
      <td className="px-2 py-2.5">
        <p className="font-medium text-slate-200">{machine.name}</p>
        <p className="mono text-[10px] text-slate-600">{machine.asset_id}</p>
      </td>
      <td className="px-2 py-2.5 text-slate-400">{machine.machine_type}</td>
      <td className="px-2 py-2.5 text-right tabular-nums text-slate-300">
        {probability === null || probability === undefined ? '—' : probability.toFixed(3)}
      </td>
      <td className="px-2 py-2.5 text-right tabular-nums text-slate-300">
        {health?.predicted_rul ? health.predicted_rul.toFixed(0) : '—'}
      </td>
      <td className="px-2 py-2.5">
        <RiskPill category={health?.risk_category ?? 'unknown'} />
      </td>
    </tr>
  )
}

function RiskPill({ category }: { category: string }) {
  const colour = RISK_COLOURS[category] ?? '#64748b'
  return (
    <span
      className="badge"
      style={{ background: `${colour}1f`, color: colour, boxShadow: `inset 0 0 0 1px ${colour}4d` }}
    >
      {category}
    </span>
  )
}

function PulseRow({
  icon,
  label,
  value,
  colour,
}: {
  icon: React.ReactNode
  label: string
  value: number
  colour: string
}) {
  return (
    <li className="flex items-center justify-between gap-3">
      <span className="flex items-center gap-2 text-[12px] text-slate-400">
        <span style={{ color: colour }}>{icon}</span>
        {label}
      </span>
      <span className="tabular-nums text-[15px] font-semibold text-slate-100">{value}</span>
    </li>
  )
}

function AlertTable({ alerts }: { alerts: import('@/types/api').Alert[] }) {
  const rows = useMemo(
    () =>
      alerts.slice(0, 20).map((alert) => ({
        time: new Date(alert.created_at).toLocaleString(),
        type: alert.alert_type,
        severity: (
          <span
            className="badge"
            style={{
              background: `${RISK_COLOURS[alert.severity] ?? '#64748b'}1f`,
              color: RISK_COLOURS[alert.severity] ?? '#64748b',
            }}
          >
            {alert.severity}
          </span>
        ),
        title: alert.title,
        source: <span className="mono text-[10px] text-slate-500">{alert.source}</span>,
        state: alert.is_resolved ? 'resolved' : 'open',
      })),
    [alerts],
  )
  return (
    <table className="w-full text-left text-[12px]">
      <thead>
        <tr className="border-b border-white/[0.08]">
          {['Time', 'Type', 'Severity', 'Title', 'Source', 'State'].map((header) => (
            <th key={header} className="px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500">
              {header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={index} className="border-b border-white/[0.04] last:border-0">
            <td className="mono px-3 py-2 text-[10px] text-slate-500">{row.time}</td>
            <td className="px-3 py-2 text-slate-400">{row.type}</td>
            <td className="px-3 py-2">{row.severity}</td>
            <td className="px-3 py-2 text-slate-200">{row.title}</td>
            <td className="px-3 py-2">{row.source}</td>
            <td className="px-3 py-2 text-slate-400">{row.state}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/* Local link helper so the table header can navigate without a router import cycle. */
function Link({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <a href={to} className="text-[12px] font-semibold text-cyan-brand hover:underline">
      {children}
    </a>
  )
}

/* Recharts chart used on the analytics page re-exports these primitives. */
export { BarChart, Bar, Legend }
