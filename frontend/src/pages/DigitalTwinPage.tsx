import { Boxes, FlaskConical, Play, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { Badge, ErrorBanner, Loading, Notice, Panel, StatCard } from '@/components/ui'
import { useAction, useAsync } from '@/hooks/useApi'
import { getMachine, getMachines, getTwinSnapshot, simulateTwin } from '@/lib/endpoints'
import type { TwinSimulationResponse, TwinSnapshot } from '@/types/api'

const FIELDS = [
  { key: 'air_temperature_k', label: 'Air temperature', unit: 'K', min: 250, max: 340, step: 0.5 },
  { key: 'process_temperature_k', label: 'Process temperature', unit: 'K', min: 250, max: 370, step: 0.5 },
  { key: 'rotational_speed_rpm', label: 'Rotational speed', unit: 'rpm', min: 0, max: 6000, step: 25 },
  { key: 'torque_nm', label: 'Torque', unit: 'N·m', min: 0, max: 60, step: 0.5 },
  { key: 'tool_wear_min', label: 'Tool wear', unit: 'min', min: 0, max: 260, step: 2 },
] as const

const TOOLTIP_STYLE = {
  backgroundColor: '#161c25',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 8,
  fontSize: 12,
  color: '#e2e8f0',
}

export default function DigitalTwinPage() {
  const machines = useAsync(() => getMachines(50), [])
  const [machineId, setMachineId] = useState('')
  const [form, setForm] = useState<Record<string, number>>({
    air_temperature_k: 300.5,
    process_temperature_k: 310.6,
    rotational_speed_rpm: 1500,
    torque_nm: 40,
    tool_wear_min: 0,
  })

  const simulate = useAction<[Record<string, number | string | null>], TwinSimulationResponse>(
    simulateTwin,
  )
  const snapshot = useAsync<TwinSnapshot | null>(
    () => (machineId ? getTwinSnapshot(machineId) : Promise.resolve(null)),
    [machineId],
  )
  const result = simulate.result

  const run = () => simulate.run({ ...form, machine_id: machineId || null })

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">Digital Twin</h1>
        <p className="mt-1 text-[13px] text-slate-400">
          Measured machine state and hypothetical operating scenarios. Simulated results are always
          separated from real telemetry.
        </p>
      </div>

      <div className="prose-note flex items-start gap-2">
        <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
        <span>
          Scenario outputs are real model evaluations of <em>operator-supplied</em> parameters.
          They are not measurements and must never be recorded as machine telemetry.
        </span>
      </div>

      <div className="grid gap-5 lg:grid-cols-5">
        {/* Controls */}
        <Panel className="lg:col-span-2" title="Scenario" subtitle="Hypothetical operating point">
          <div className="space-y-4">
            <div>
              <label className="label" htmlFor="machine">
                Asset (optional)
              </label>
              <select
                id="machine"
                value={machineId}
                onChange={(event) => setMachineId(event.target.value)}
                className="input"
              >
                <option value="">No specific asset</option>
                {(machines.data?.items ?? []).map((machine) => (
                  <option key={machine.id} value={machine.id}>
                    {machine.asset_id} — {machine.name}
                  </option>
                ))}
              </select>
            </div>

            {FIELDS.map((field) => (
              <div key={field.key}>
                <div className="flex items-baseline justify-between">
                  <label className="label mb-0" htmlFor={field.key}>
                    {field.label}
                  </label>
                  <span className="mono text-[11px] text-cyan-brand">
                    {Number(form[field.key]).toFixed(1)} {field.unit}
                  </span>
                </div>
                <input
                  id={field.key}
                  type="range"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={form[field.key]}
                  onChange={(event) => setForm((prev) => ({ ...prev, [field.key]: Number(event.target.value) }))}
                  className="mt-1.5 w-full accent-cyan-400"
                />
              </div>
            ))}

            <button className="btn-primary w-full" onClick={run} disabled={simulate.pending}>
              <FlaskConical className="h-4 w-4" />
              {simulate.pending ? 'Simulating...' : 'Simulate scenario'}
            </button>
          </div>
        </Panel>

        <div className="space-y-5 lg:col-span-3">
          {/* Measured state */}
          <Panel
            title="Measured state"
            subtitle={machineId ? 'Real stored telemetry — not simulated' : 'Select an asset'}
            actions={
              machineId ? <Badge value="ok" label="MEASURED" /> : <Badge value="info" label="NO ASSET" />
            }
          >
            {!machineId && (
              <Notice tone="info">
                <p>Choose an asset to display its last recorded sensor reading and stored verdicts.</p>
              </Notice>
            )}
            {machineId && snapshot.loading && <Loading label="Loading snapshot" />}
            {Boolean(snapshot.error) && <ErrorBanner error={snapshot.error} />}
            {snapshot.data?.readings && (
              <div className="grid gap-4 md:grid-cols-2">
                <dl className="space-y-2 text-[12px]">
                  <Row label="Air temperature" value={`${snapshot.data.readings.air_temperature_k} K`} />
                  <Row
                    label="Process temperature"
                    value={`${snapshot.data.readings.process_temperature_k} K`}
                  />
                  <Row
                    label="Rotational speed"
                    value={`${snapshot.data.readings.rotational_speed_rpm} rpm`}
                  />
                  <Row label="Torque" value={`${snapshot.data.readings.torque_nm} N·m`} />
                  <Row label="Tool wear" value={`${snapshot.data.readings.tool_wear_min} min`} />
                  <Row label="Source" value={String(snapshot.data.readings.source ?? '—')} />
                </dl>
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <StatCard
                      label="Failure probability"
                      value={
                        snapshot.data.health.failure_probability === null
                          ? '—'
                          : snapshot.data.health.failure_probability.toFixed(3)
                      }
                    />
                    <StatCard label="Risk" value={snapshot.data.health.risk_category} />
                    <StatCard
                      label="Anomaly score"
                      value={
                        snapshot.data.health.anomaly_score === null
                          ? '—'
                          : snapshot.data.health.anomaly_score.toFixed(3)
                      }
                    />
                    <StatCard
                      label="RUL"
                      value={snapshot.data.health.predicted_rul ?? '—'}
                      unit={snapshot.data.health.predicted_rul ? 'cyc' : undefined}
                    />
                  </div>
                  <MachineCurve machineId={machineId} />
                </div>
              </div>
            )}
          </Panel>

          {/* Simulation result */}
          {simulate.pending && <Loading label="Running scenario through the trained models" />}
          {Boolean(simulate.error) && <ErrorBanner error={simulate.error} onRetry={run} />}

          {result && (
            <Panel
              title="Simulation result"
              subtitle={result.machine_id ?? 'scenario — no specific asset'}
              actions={<Badge value="critical" label="SIMULATED" />}
            >
              {!result.failure && !result.anomaly && (
                <Notice tone="warn" title="No trained model available">
                  <p>{result.unavailable_reason}</p>
                </Notice>
              )}

              {result.comparison && (
                <div className="grid gap-4 sm:grid-cols-3">
                  <StatCard
                    label="Scenario failure probability"
                    value={result.comparison.failure_probability.scenario.toFixed(4)}
                    tone={
                      result.comparison.failure_probability.scenario > 0.6 ? 'bad' : 'good'
                    }
                    trend={{
                      value: result.comparison.failure_probability.delta,
                      label: 'vs AI4I baseline duty',
                    }}
                  />
                  <StatCard
                    label="Baseline probability"
                    value={result.comparison.failure_probability.baseline.toFixed(4)}
                    hint="at 300.5 K / 1500 rpm / 40 N·m"
                  />
                  <StatCard
                    label="Risk band"
                    value={result.comparison.risk_category.scenario}
                    hint={`baseline: ${result.comparison.risk_category.baseline}`}
                  />
                </div>
              )}

              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div>
                  <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                    Parameter deltas from baseline
                  </p>
                  <dl className="space-y-1.5 text-[12px]">
                    {Object.entries(result.deltas_from_baseline).map(([key, value]) => (
                      <div key={key} className="flex justify-between gap-3 border-b border-white/[0.04] pb-1">
                        <dt className="text-slate-500">{key.replace(/_/g, ' ')}</dt>
                        <dd
                          className="mono"
                          style={{ color: value > 0 ? '#f97316' : value < 0 ? '#3b82f6' : '#94a3b8' }}
                        >
                          {value > 0 ? '+' : ''}
                          {value.toFixed(2)}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
                <div>
                  <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                    Derived features
                  </p>
                  <dl className="space-y-1.5 text-[12px]">
                    {Object.entries(result.derived_features).map(([key, value]) => (
                      <div key={key} className="flex justify-between gap-3 border-b border-white/[0.04] pb-1">
                        <dt className="truncate text-slate-500">{key.replace(/_/g, ' ')}</dt>
                        <dd className="mono text-slate-300">{value.toFixed(3)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </div>

              {result.failure && (
                <div className="mt-4">
                  <p className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                    <Boxes className="h-3.5 w-3.5" /> Top SHAP drivers in this scenario
                  </p>
                  <ul className="space-y-1.5">
                    {result.failure.features.slice(0, 5).map((feature) => (
                      <li key={feature.feature} className="flex items-center justify-between gap-3 text-[11.5px]">
                        <span className="text-slate-300">{feature.feature}</span>
                        <span
                          className="mono"
                          style={{ color: feature.contribution >= 0 ? '#ef4444' : '#10b981' }}
                        >
                          {feature.contribution >= 0 ? '+' : ''}
                          {feature.contribution.toFixed(4)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <p className="mt-4 text-[11px] leading-relaxed text-amber-300/80">{result.disclaimer}</p>
            </Panel>
          )}
        </div>
      </div>
    </div>
  )
}

/** Failure probability of the asset's stored prediction history, as measured. */
function MachineCurve({ machineId }: { machineId: string }) {
  const machine = useAsync(() => getMachine(machineId), [machineId])
  const health = machine.data?.health
  if (!health) return null
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
        Current verdict
      </p>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-charcoal-600">
        <div
          className="h-full rounded-full bg-cyan-brand"
          style={{ width: `${(health.failure_probability ?? 0) * 100}%` }}
        />
      </div>
      <p className="mono mt-1.5 text-[10px] text-slate-500">
        {health.failure_probability === null
          ? 'no stored prediction — run one from Predictive Maintenance'
          : `p(failure) = ${health.failure_probability.toFixed(4)} · ${health.risk_category}`}
      </p>
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

export { Play, ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, TOOLTIP_STYLE }
