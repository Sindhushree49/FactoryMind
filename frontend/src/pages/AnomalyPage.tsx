import { Activity, Radar } from 'lucide-react'
import { useState } from 'react'
import {
  CartesianGrid,
  PolarAngleAxis,
  PolarGrid,
  Radar as RadarChart,
  PolarRadiusAxis,
  ResponsiveContainer,
  RadarChart as RechartsRadar,
  Tooltip,
} from 'recharts'

import { Badge, ErrorBanner, Gauge as GaugeDial, Loading, Notice, Panel, StatCard } from '@/components/ui'
import { useAction, useAsync } from '@/hooks/useApi'
import { detectAnomaly, getAlerts } from '@/lib/endpoints'
import type { AnomalyDetectionRequest, AnomalyDetectionResponse, SensorReading } from '@/types/api'

const DEFAULTS = {
  air_temperature_k: 300.5,
  process_temperature_k: 310.6,
  rotational_speed_rpm: 1500,
  torque_nm: 40,
  tool_wear_min: 0,
}

const TOOLTIP_STYLE = {
  backgroundColor: '#161c25',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 8,
  fontSize: 12,
  color: '#e2e8f0',
}

export default function AnomalyPage() {
  const [form, setForm] = useState<Record<string, number>>(DEFAULTS)
  const [threshold, setThreshold] = useState<number | null>(null)
  const action = useAction<[AnomalyDetectionRequest], AnomalyDetectionResponse>(
    detectAnomaly,
  )
  const alerts = useAsync(() => getAlerts(false), [])
  const result = action.result

  const submit = () =>
    action.run({ ...(form as unknown as SensorReading), persist: true, threshold })

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">Anomaly Detection</h1>
        <p className="mt-1 text-[13px] text-slate-400">
          Isolation Forest scoring against the AI4I operating envelope, with per-sensor deviation in
          standard deviations.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-5">
        <Panel className="lg:col-span-2" title="Reading to score" subtitle="AI4I sensor schema">
          <div className="space-y-3">
            {Object.entries(DEFAULTS).map(([key]) => (
              <div key={key}>
                <label className="label" htmlFor={key}>
                  {key.replace(/_/g, ' ')}
                </label>
                <input
                  id={key}
                  type="number"
                  value={form[key]}
                  step="0.1"
                  onChange={(event) => setForm((prev) => ({ ...prev, [key]: Number(event.target.value) }))}
                  className="input font-mono"
                />
              </div>
            ))}
            <div>
              <label className="label" htmlFor="threshold">
                Threshold override (blank = model default)
              </label>
              <input
                id="threshold"
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={threshold ?? ''}
                onChange={(event) => setThreshold(event.target.value === '' ? null : Number(event.target.value))}
                className="input font-mono"
                placeholder="model default"
              />
            </div>
            <button className="btn-primary w-full" onClick={submit} disabled={action.pending}>
              <Activity className="h-4 w-4" />
              {action.pending ? 'Scoring...' : 'Score reading'}
            </button>
          </div>
        </Panel>

        <div className="space-y-5 lg:col-span-3">
          {action.pending && <Loading label="Scoring with Isolation Forest" />}
          {Boolean(action.error) && <ErrorBanner error={action.error} onRetry={submit} />}

          {!action.pending && !result && (
            <Panel title="Awaiting a reading">
              <Notice tone="info" title="No score yet">
                <p>
                  Submit a reading to obtain an anomaly score, the calibrated threshold, and the
                  per-sensor deviation that produced the score.
                </p>
              </Notice>
            </Panel>
          )}

          {result && (
            <>
              <Panel
                title="Anomaly score"
                subtitle={result.model_version}
                actions={<Badge value={result.is_anomaly ? 'high' : 'healthy'} label={result.severity.toUpperCase()} />}
              >
                <div className="flex flex-wrap items-center gap-6">
                  <GaugeDial value={result.anomaly_score} label="anomaly score" />
                  <dl className="min-w-[200px] flex-1 space-y-2 text-[12px]">
                    <Row label="Threshold" value={result.threshold.toFixed(3)} />
                    <Row label="Flagged as anomaly" value={String(result.is_anomaly)} />
                    <Row label="Severity" value={result.severity} />
                    <Row label="Event stored" value={result.anomaly_event_id ?? 'no'} />
                  </dl>
                </div>
                <p className="mt-4 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-[12px] leading-relaxed text-slate-300">
                  {result.explanation}
                </p>
              </Panel>

              <Panel title="Sensor deviation" subtitle="Observed value vs AI4I training distribution">
                <ResponsiveContainer width="100%" height={280}>
                  <RechartsRadar data={result.feature_deviation} outerRadius="72%">
                    <PolarGrid stroke="rgba(255,255,255,0.08)" />
                    <PolarAngleAxis dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 9 }} />
                    <RadarChart
                      name="Deviation"
                      dataKey="contribution"
                      stroke="#22d3ee"
                      fill="#22d3ee"
                      fillOpacity={0.28}
                    />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                  </RechartsRadar>
                </ResponsiveContainer>

                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-left text-[12px]">
                    <thead>
                      <tr className="border-b border-white/[0.08]">
                        {['Sensor', 'Value', 'Expected mean', 'Training range', 'z-score', 'Out of range'].map(
                          (header) => (
                            <th
                              key={header}
                              className="px-2 py-2 text-[11px] uppercase tracking-wider text-slate-500"
                            >
                              {header}
                            </th>
                          ),
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {result.feature_deviation.map((item) => (
                        <tr key={item.feature} className="border-b border-white/[0.04] last:border-0">
                          <td className="px-2 py-2 text-slate-300">{item.feature}</td>
                          <td className="mono px-2 py-2 text-slate-200">{item.value}</td>
                          <td className="mono px-2 py-2 text-slate-400">
                            {item.expected_mean?.toFixed(2) ?? '—'}
                          </td>
                          <td className="mono px-2 py-2 text-slate-400">
                            {item.expected_range
                              ? `[${item.expected_range[0].toFixed(1)}, ${item.expected_range[1].toFixed(1)}]`
                              : '—'}
                          </td>
                          <td
                            className="mono px-2 py-2"
                            style={{
                              color:
                                Math.abs(item.z_score ?? 0) > 3
                                  ? '#ef4444'
                                  : Math.abs(item.z_score ?? 0) > 2
                                    ? '#f97316'
                                    : '#94a3b8',
                            }}
                          >
                            {item.z_score?.toFixed(2) ?? '—'}
                          </td>
                          <td className="px-2 py-2">
                            {item.out_of_training_range ? (
                              <span className="badge-critical">outside</span>
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          )}
        </div>
      </div>

      <Panel title="Anomaly event log" subtitle="Most recent stored anomaly events" dense>
        {alerts.loading ? (
          <Loading />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[12px]">
              <thead>
                <tr className="border-b border-white/[0.08]">
                  {['Time', 'Type', 'Severity', 'Title', 'Source', 'Resolved'].map((header) => (
                    <th key={header} className="px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(alerts.data ?? []).slice(0, 25).map((alert) => (
                  <tr key={alert.id} className="border-b border-white/[0.04] last:border-0">
                    <td className="mono px-3 py-2 text-[10px] text-slate-500">
                      {new Date(alert.created_at).toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-slate-400">{alert.alert_type}</td>
                    <td className="px-3 py-2">
                      <Badge value={alert.severity} label={alert.severity} />
                    </td>
                    <td className="px-3 py-2 text-slate-200">{alert.title}</td>
                    <td className="mono px-3 py-2 text-[10px] text-slate-500">{alert.source}</td>
                    <td className="px-3 py-2 text-slate-400">{alert.is_resolved ? 'yes' : 'no'}</td>
                  </tr>
                ))}
                {!alerts.data?.length && (
                  <tr>
                    <td colSpan={6} className="px-3 py-8 text-center text-slate-500">
                      No alerts recorded yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label="Detection method" value="Isolation Forest" hint="Unsupervised, 300 trees" />
        <StatCard
          label="Training data"
          value="AI4I 2020"
          hint="6 000 fitting rows, 25% held out for calibration"
        />
        <StatCard
          label="Label source"
          value="None"
          tone="warn"
          hint="AI4I ships no anomaly labels, so the threshold is calibrated on held-out good readings"
        />
      </div>

      <div className="flex items-center gap-2 text-[11px] text-slate-600">
        <Radar className="h-3.5 w-3.5" />
        Anomaly score is the normalised distance from the learned good-operating manifold, not a
        failure probability.
      </div>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/[0.04] pb-1.5">
      <dt className="text-slate-500">{label}</dt>
      <dd className="mono text-slate-200">{value}</dd>
    </div>
  )
}

export { CartesianGrid }
