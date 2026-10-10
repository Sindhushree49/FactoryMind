import { Cpu, Database, FolderTree, RefreshCw, Server, Sliders, Sparkles } from 'lucide-react'

import { Badge, EmptyState, ErrorBanner, Loading, Notice, Panel, StatCard } from '@/components/ui'
import { useAsync } from '@/hooks/useApi'
import { getDatasets, getSettings } from '@/lib/endpoints'
import type { DatasetStatus } from '@/types/api'

export default function SettingsPage() {
  const settings = useAsync(() => getSettings(), [])
  const datasets = useAsync(() => getDatasets(), [])

  if (settings.loading && !settings.data) return <Loading label="Reading runtime configuration" />
  if (!settings.data) return <ErrorBanner error={settings.error} onRetry={settings.refetch} />

  const config = settings.data
  const statuses = config.datasets ?? datasets.data?.statuses ?? {}

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-50">Settings</h1>
          <p className="mt-1 text-[13px] text-slate-400">
            The effective runtime configuration, as the running backend reports it. Secrets are
            redacted server-side and are never returned by the API.
          </p>
        </div>
        <button className="btn-secondary" onClick={settings.refetch}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </button>
      </div>

      {Boolean(settings.error) && <ErrorBanner error={settings.error} onRetry={settings.refetch} />}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Application"
          value={config.app.version ?? '—'}
          hint={config.app.environment}
          icon={<Server className="h-4 w-4" />}
        />
        <StatCard
          label="LLM engine"
          value={<Badge value={config.llm.active ? 'ok' : 'skipped'} label={config.llm.active ? 'ACTIVE' : 'RULE-BASED'} />}
          hint={typeof config.llm.model === 'string' ? config.llm.model : 'no model configured'}
          tone={config.llm.active ? 'good' : 'neutral'}
          icon={<Sparkles className="h-4 w-4" />}
        />
        <StatCard
          label="Embedding model"
          value={render(config.rag.embedding_model)}
          hint={`top_k ${render(config.rag.top_k)} · ${render(config.rag.chunk_size)}/${render(config.rag.chunk_overlap)} chunk`}
          icon={<Cpu className="h-4 w-4" />}
        />
        <StatCard
          label="Datasets available"
          value={Object.values(statuses).filter((status) => status.available).length}
          unit={`of ${Object.keys(statuses).length}`}
          tone="accent"
          icon={<Database className="h-4 w-4" />}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Application" subtitle="Process identity and logging">
          <Entries record={config.app} />
        </Panel>

        <Panel title="Database" subtitle="Connection details with credentials stripped">
          <Entries record={config.database} />
        </Panel>

        <Panel
          title="LLM provider"
          subtitle="Read from the environment only"
          actions={<Badge value={config.llm.enabled ? 'ok' : 'skipped'} label={config.llm.enabled ? 'ENABLED' : 'DISABLED'} />}
        >
          <Entries record={config.llm} />
        </Panel>

        <Panel title="Retrieval" subtitle="Chunking and embedding configuration">
          <Entries record={config.rag} />
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="Decision thresholds"
          subtitle="The exact cut-offs every risk verdict is derived from"
          actions={<Sliders className="h-4 w-4 text-slate-600" />}
        >
          <Entries record={config.thresholds} />
        </Panel>

        <Panel title="Computer vision" subtitle="PatchCore preprocessing and score threshold">
          <Entries record={config.computer_vision} />
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="ML runtime" subtitle="Device selection and determinism">
          <Entries record={config.ml_runtime} />
        </Panel>

        <Panel
          title="Filesystem paths"
          subtitle="Resolved absolute locations"
          actions={<FolderTree className="h-4 w-4 text-slate-600" />}
        >
          <Entries record={config.paths} />
        </Panel>
      </div>

      <Panel title="Datasets" subtitle="Registry status reported by the backend" dense>
        {Object.keys(statuses).length === 0 ? (
          <EmptyState title="No datasets registered" />
        ) : (
          <div className="grid gap-3 p-1 md:grid-cols-2">
            {Object.values(statuses).map((status) => (
              <DatasetCard key={status.key} status={status} />
            ))}
          </div>
        )}
      </Panel>

      <Notice tone="info" title="These values are read-only here">
        <p>
          Configuration is environment-driven. Edit the deployment environment and restart the
          backend rather than changing values in this screen, so the running process and the
          reported configuration cannot drift apart.
        </p>
      </Notice>
    </div>
  )
}

function DatasetCard({ status }: { status: DatasetStatus }) {
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[13px] font-semibold text-slate-100">{status.title}</p>
        <Badge value={status.available ? 'ok' : 'blocked'} label={status.available ? 'AVAILABLE' : 'MISSING'} />
      </div>
      <p className="mono mt-1 text-[10px] text-slate-600">{status.key}</p>
      <p className="mt-2 text-[11.5px] leading-relaxed text-slate-400">{status.detail}</p>
      <p className="mono mt-2 text-[10px] text-slate-600">
        {status.file_count} file{status.file_count === 1 ? '' : 's'} · {(status.bytes / 1048576).toFixed(1)} MB
      </p>
      {status.path && (
        <p className="mono mt-1 break-all text-[10px] text-slate-700">{status.path}</p>
      )}
    </div>
  )
}

function Entries({ record }: { record: Record<string, unknown> }) {
  const entries = Object.entries(record)
  if (entries.length === 0) {
    return <p className="text-[12px] text-slate-600">Nothing reported.</p>
  }
  return (
    <dl className="space-y-1.5">
      {entries.map(([key, value]) => (
        <div
          key={key}
          className="flex items-start justify-between gap-3 border-b border-white/[0.04] pb-1.5 text-[12px]"
        >
          <dt className="text-slate-500">{key.replace(/_/g, ' ')}</dt>
          <dd className="mono max-w-[60%] break-words text-right text-slate-200">
            {render(value)}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function render(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
