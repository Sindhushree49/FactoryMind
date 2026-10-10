import { AlertTriangle, Info, Zap } from 'lucide-react'
import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { ErrorBanner, Gauge as GaugeDial, Loading, Notice, Panel } from '@/components/ui'
import { useAction } from '@/hooks/useApi'
import { predictFailure } from '@/lib/endpoints'
import type { FailurePredictionRequest, FailurePredictionResponse } from '@/types/api'

/** AI4I nominal operating point. Temperatures are Kelvin, as published. */
const DEFAULTS = {
  air_temperature_k: 300.5,
  process_temperature_k: 310.6,
  rotational_speed_rpm: 1500,
  torque_nm: 40,
  tool_wear_min: 0,
  machine_type: 'L',
}

const FIELDS = [
  { key: 'air_temperature_k', label: 'Air temperature', unit: 'K', min: 250, max: 330, step: 0.1 },
  { key: 'process_temperature_k', label: 'Process temperature', unit: 'K', min: 250, max: 360, step: 0.1 },
  { key: 'rotational_speed_rpm', label: 'Rotational speed', unit: 'rpm', min: 0, max: 30000, step: 10 },
  { key: 'torque_nm', label: 'Torque', unit: 'N·m', min: 0, max: 60, step: 0.5 },
  { key: 'tool_wear_min', label: 'Tool wear', unit: 'min', min: 0, max: 260, step: 1 },
] as const

const TOOLTIP_STYLE = {
  backgroundColor: '#161c25',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 8,
  fontSize: 12,
  color: '#e2e8f0',
}

export default function PredictivePage() {
  const [form, setForm] = useState<Record<string, number | string>>(DEFAULTS)
  const [persist, setPersist] = useState(true)
  const action = useAction<[FailurePredictionRequest], FailurePredictionResponse>(predictFailure)
  const result = action.result

  const submit = () =>
    action.run({
      air_temperature_k: Number(form.air_temperature_k),
      process_temperature_k: Number(form.process_temperature_k),
      rotational_speed_rpm: Number(form.rotational_speed_rpm),
      torque_nm: Number(form.torque_nm),
      tool_wear_min: Number(form.tool_wear_min),
      machine_type: String(form.machine_type),
      persist,
      shap_top_k: 10,
    })

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">Predictive Maintenance</h1>
        <p className="mt-1 text-[13px] text-slate-400">
          Failure probability and SHAP attribution from the AI4I-trained classifier. Enter
          telemetry to score it.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-5">
        {/* Input */}
        <Panel className="lg:col-span-2" title="Sensor input" subtitle="AI4I 2020 schema">
          <div className="space-y-4">
            {FIELDS.map((field) => (
              <div key={field.key}>
                <div className="flex items-baseline justify-between">
                  <label className="label mb-0" htmlFor={field.key}>
                    {field.label}
                  </label>
                  <span className="mono text-[11px] text-cyan-brand">
                    {Number(form[field.key]).toFixed(field.step < 1 ? 1 : 0)} {field.unit}
                  </span>
                </div>
                <input
                  id={field.key}
                  type="range"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={Number(form[field.key])}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, [field.key]: Number(event.target.value) }))
                  }
                  className="mt-1.5 w-full accent-cyan-400"
                />
                <input
                  type="number"
                  value={Number(form[field.key])}
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, [field.key]: Number(event.target.value) }))
                  }
                  className="input mt-1.5 font-mono"
                />
              </div>
            ))}

            <div>
              <label className="label" htmlFor="machine_type">
                Machine type
              </label>
              <select
                id="machine_type"
                value={String(form.machine_type)}
                onChange={(event) => setForm((prev) => ({ ...prev, machine_type: event.target.value }))}
                className="input"
              >
                <option value="L">L — Low (60%)</option>
                <option value="M">M — Medium (30%)</option>
                <option value="H">H — High (10%)</option>
              </select>
            </div>

            <label className="flex items-center gap-2 text-[12px] text-slate-400">
              <input
                type="checkbox"
                checked={persist}
                onChange={(event) => setPersist(event.target.checked)}
                className="accent-cyan-400"
              />
              Store this prediction
            </label>

            <div className="flex gap-2">
              <button className="btn-primary flex-1" onClick={submit} disabled={action.pending}>
                <Zap className="h-4 w-4" />
                {action.pending ? 'Predicting...' : 'Predict failure'}
              </button>
              <button className="btn-secondary" onClick={() => { setForm(DEFAULTS); action.reset() }}>
                Reset
              </button>
            </div>
          </div>
        </Panel>

        {/* Result */}
        <div className="space-y-5 lg:col-span-3">
          {action.pending && <Loading label="Running model and SHAP explainer" />}

          {Boolean(action.error) && <ErrorBanner error={action.error} onRetry={submit} />}

          {!action.pending && !result && (
            <Panel title="Awaiting input">
              <Notice tone="info" title="No prediction yet">
                <p>
                  Adjust the sensor values and press <strong>Predict failure</strong>. The
                  response includes the model's SHAP attribution for each feature.
                </p>
                <p className="mt-2">
                  <Info className="mr-1 inline h-3 w-3" />
                  Nominal AI4I duty is 300 K air, 310 K process, 1500 rpm, 40 N·m, zero wear.
                </p>
              </Notice>
            </Panel>
          )}

          {result && (
            <>
              <Panel
                title="Prediction"
                subtitle={`${result.model_family} · ${result.model_version}`}
                actions={<RiskBadge value={result.risk_category} />}
              >
                <div className="flex flex-wrap items-center gap-6">
                  <GaugeDial value={result.failure_probability} label="failure probability" />
                  <dl className="min-w-[220px] flex-1 space-y-2 text-[12px]">
                    <Row label="Decision threshold" value={result.decision_threshold.toFixed(2)} />
                    <Row
                      label="Predicted failure mode"
                      value={
                        result.predicted_failure_type
                          ? `${result.predicted_failure_type} (${(
                              (result.predicted_failure_type_confidence ?? 0) * 100
                            ).toFixed(1)}%)`
                          : 'not modelled'
                      }
                    />
                    <Row label="SHAP base value" value={result.shap_base_value?.toFixed(4) ?? '—'} />
                    <Row label="Prediction stored" value={result.prediction_id ?? 'no'} />
                  </dl>
                </div>
                <p className="mt-4 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-[12px] leading-relaxed text-slate-300">
                  {result.explanation}
                </p>
              </Panel>

              <Panel title="SHAP feature attribution" subtitle="Contribution to this prediction">
                <div className="space-y-2.5">
                  {result.features.map((feature) => (
                    <ContributionBar
                      key={feature.feature}
                      feature={feature.feature}
                      value={feature.value}
                      contribution={feature.contribution}
                      direction={feature.direction}
                    />
                  ))}
                </div>
                <div className="prose-note mt-4">
                  <AlertTriangle className="mr-1 inline h-3.5 w-3.5" />
                  {result.shap_note}
                </div>
              </Panel>
            </>
          )}
        </div>
      </div>

      {result && (
        <Panel title="Attribution chart" subtitle="Signed SHAP contribution per feature">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={result.features.map((f) => ({ name: f.feature, contribution: f.contribution }))}
              layout="vertical"
              margin={{ left: 24, right: 24 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis type="number" stroke="#64748b" fontSize={10} tickLine={false} />
              <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={10} width={150} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => v.toFixed(4)} />
              <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                {result.features.map((f) => (
                  <Cell
                    key={f.feature}
                    fill={f.contribution >= 0 ? '#ef4444' : '#10b981'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      )}
    </div>
  )
}

function ContributionBar({
  feature,
  value,
  contribution,
  direction,
}: {
  feature: string
  value: number
  contribution: number
  direction: string
}) {
  const magnitude = Math.abs(contribution)
  const scale = 0.5
  const width = Math.min(100, (magnitude / scale) * 100)
  const colour = contribution >= 0 ? '#ef4444' : '#10b981'
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-[11px]">
        <span className="flex items-center gap-1.5 font-medium text-slate-300">
          {feature}
          <span
            className="badge"
            style={{
              background: direction === 'increases_risk' ? '#ef44441f' : '#10b9811f',
              color: contribution >= 0 ? '#ef4444' : '#10b981',
            }}
          >
            {direction === 'increases_risk' ? '↑ risk' : direction === 'decreases_risk' ? '↓ risk' : 'neutral'}
          </span>
        </span>
        <span className="mono text-slate-500">
          value {value.toFixed(2)} · SHAP {contribution >= 0 ? '+' : ''}
          {contribution.toFixed(4)}
        </span>
      </div>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-charcoal-600">
        <div className="h-full rounded-full" style={{ width: `${width}%`, background: colour }} />
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

function RiskBadge({ value }: { value: string }) {
  const colour =
    value === 'critical' ? '#ef4444' : value === 'high' ? '#f97316' : value === 'watch' ? '#eab308' : '#10b981'
  return (
    <span
      className="badge"
      style={{ background: `${colour}1f`, color: colour, boxShadow: `inset 0 0 0 1px ${colour}4d` }}
    >
      {value.toUpperCase()}
    </span>
  )
}

export { BarChart }
