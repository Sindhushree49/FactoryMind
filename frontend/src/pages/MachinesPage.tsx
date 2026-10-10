import { Boxes, Database } from 'lucide-react'
import { useState } from 'react'

import { ErrorBanner, Loading, Notice, Panel, StatCard, Table, EmptyState } from '@/components/ui'
import { useAction, useAsync } from '@/hooks/useApi'
import { getMachines, seedMachines } from '@/lib/endpoints'
import type { Machine } from '@/types/api'

const RISK_COLOURS: Record<string, string> = {
  healthy: '#10b981',
  watch: '#eab308',
  high: '#f97316',
  critical: '#ef4443',
  unknown: '#64748b',
}

export default function MachinesPage() {
  const machines = useAsync(() => getMachines(100), [])
  const seed = useAction(seedMachines)
  const [selected, setSelected] = useState<Machine | null>(null)

  if (machines.loading && !machines.data) return <Loading label="Loading machine registry" />
  if (!machines.data) return <ErrorBanner error={machines.error} onRetry={machines.refetch} />

  const items = machines.data.items
  const active = items.filter((m) => m.is_active).length
  const withRisk = items.filter((m) => m.health && m.health.risk_category !== 'healthy').length
  const withAnomaly = items.filter((m) => m.health?.is_anomaly).length

  const rows = items.map((machine) => ({
    asset: (
      <div>
        <p className="font-medium text-slate-200">{machine.name}</p>
        <p className="mono text-[10px] text-slate-600">{machine.asset_id}</p>
      </div>
    ),
    type: machine.machine_type,
    location: machine.location,
    air_k: machine.latest_reading ? machine.latest_reading.air_temperature_k.toFixed(1) : '—',
    process_k: machine.latest_reading ? machine.latest_reading.process_temperature_k.toFixed(1) : '—',
    rpm: machine.latest_reading ? Math.round(machine.latest_reading.rotational_speed_rpm) : '—',
    torque: machine.latest_reading ? machine.latest_reading.torque_nm.toFixed(1) : '—',
    wear: machine.latest_reading ? machine.latest_reading.tool_wear_min.toFixed(0) : '—',
    probability:
      machine.health?.failure_probability === null || machine.health?.failure_probability === undefined
        ? '—'
        : machine.health.failure_probability.toFixed(3),
    rul: machine.health?.predicted_rul ? machine.health.predicted_rul.toFixed(0) : '—',
    risk: <RiskPill category={machine.health?.risk_category ?? 'unknown'} />,
    actions: (
      <button className="btn-ghost px-2 py-1 text-[11px]" onClick={() => setSelected(machine)}>
        Inspect
      </button>
    ),
  }))

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-50">Machine Intelligence</h1>
          <p className="mt-1 text-[13px] text-slate-400">
            Registered assets with their latest measured telemetry and stored model verdicts.
          </p>
        </div>
        <button
          className="btn-secondary"
          disabled={seed.pending}
          onClick={() => seed.run(8).then(() => machines.refetch())}
        >
          <Database className="h-4 w-4" />
          {seed.pending ? 'Seeding...' : 'Seed from AI4I'}
        </button>
      </div>

      {Boolean(seed.error) && <ErrorBanner error={seed.error} />}
      {Boolean(machines.error) && <ErrorBanner error={machines.error} onRetry={machines.refetch} />}

      {items.length === 0 && (
        <Notice tone="warn" title="Registry is empty">
          <p>
            Seed machines to replay verbatim AI4I telemetry as their sensor history. No reading is
            synthesised.
          </p>
        </Notice>
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Registered assets" value={items.length} />
        <StatCard label="Active" value={active} tone="good" />
        <StatCard label="Not healthy" value={withRisk} tone={withRisk ? 'warn' : 'good'} />
        <StatCard label="Anomalous reading" value={withAnomaly} tone={withAnomaly ? 'bad' : 'good'} />
      </div>

      <Panel title="Fleet telemetry" subtitle="Temperatures in Kelvin, matching the AI4I schema" dense>
        <Table
          empty="No machines registered"
          columns={[
            { key: 'asset', header: 'Asset' },
            { key: 'type', header: 'Type' },
            { key: 'location', header: 'Location' },
            { key: 'air_k', header: 'Air K', align: 'right' },
            { key: 'process_k', header: 'Process K', align: 'right' },
            { key: 'rpm', header: 'RPM', align: 'right' },
            { key: 'torque', header: 'Torque Nm', align: 'right' },
            { key: 'wear', header: 'Wear min', align: 'right' },
            { key: 'probability', header: 'Fail prob', align: 'right' },
            { key: 'rul', header: 'RUL', align: 'right' },
            { key: 'risk', header: 'Risk' },
            { key: 'actions', header: '', align: 'right' },
          ]}
          rows={rows}
        />
      </Panel>

      {selected && (
        <Panel
          title={selected.name}
          subtitle={`${selected.asset_id} · ${selected.model} · registered ${new Date(
            selected.created_at,
          ).toLocaleString()}`}
          actions={
            <button className="btn-ghost" onClick={() => setSelected(null)}>
              Close
            </button>
          }
        >
          <MachineDetail machine={selected} />
        </Panel>
      )}
    </div>
  )
}

function MachineDetail({ machine }: { machine: Machine }) {
  const health = machine.health
  const reading = machine.latest_reading
  if (!health) return <EmptyState title="No health record for this machine" />

  return (
    <div className="grid gap-5 md:grid-cols-2">
      <div className="space-y-3">
        <h3 className="text-[13px] font-semibold uppercase tracking-wider text-slate-400">
          Model verdicts
        </h3>
        <dl className="space-y-2 text-[12px]">
          <Row label="Failure probability" value={fmt(health.failure_probability)} />
          <Row label="Risk category" value={health.risk_category} />
          <Row label="Anomaly score" value={fmt(health.anomaly_score)} />
          <Row label="Anomalous" value={health.is_anomaly === null ? '—' : String(health.is_anomaly)} />
          <Row label="Remaining useful life" value={health.predicted_rul ? `${health.predicted_rul} cycles` : '—'} />
          <Row label="Model version" value={health.model_version ?? 'no model trained'} />
          <Row label="Assessed at" value={health.assessed_at ? new Date(health.assessed_at).toLocaleString() : '—'} />
        </dl>
      </div>
      <div className="space-y-3">
        <h3 className="text-[13px] font-semibold uppercase tracking-wider text-slate-400">
          Latest measured reading
        </h3>
        {reading ? (
          <dl className="space-y-2 text-[12px]">
            <Row label="Recorded at" value={new Date(reading.recorded_at ?? '').toLocaleString()} />
            <Row label="Air temperature" value={`${reading.air_temperature_k.toFixed(1)} K`} />
            <Row label="Process temperature" value={`${reading.process_temperature_k.toFixed(1)} K`} />
            <Row label="Rotational speed" value={`${Math.round(reading.rotational_speed_rpm)} rpm`} />
            <Row label="Torque" value={`${reading.torque_nm.toFixed(1)} Nm`} />
            <Row label="Tool wear" value={`${reading.tool_wear_min.toFixed(0)} min`} />
            <Row label="Source" value={reading.source ?? '—'} />
          </dl>
        ) : (
          <EmptyState title="No stored readings" detail="Submit telemetry for this machine." />
        )}
        <div className="flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-[11px] text-slate-500">
          <Boxes className="h-3.5 w-3.5 shrink-0" />
          Run a what-if against this asset in the Digital Twin page.
        </div>
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

function fmt(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toFixed(4)
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
