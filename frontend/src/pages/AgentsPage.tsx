import { AlertTriangle, Bot, CheckCircle2, Play, ShieldCheck } from 'lucide-react'
import { useState } from 'react'
import { motion } from 'framer-motion'

import { Badge, ErrorBanner, Loading, Notice, Panel, StatCard } from '@/components/ui'
import { useAction, useAsync } from '@/hooks/useApi'
import { getAgentCatalog, getInvestigations, investigate } from '@/lib/endpoints'
import type { InvestigationRequest, InvestigationResponse } from '@/types/api'

/**
 * Only the AI4I sensor fields are sent. `AgentInvestigationRequest` on the backend
 * accepts no `machine_type`, so the workflow always runs the models on the raw
 * reading — adding the field here would be discarded by the API.
 */
const DEFAULTS = {
  air_temperature_k: 318,
  process_temperature_k: 336,
  rotational_speed_rpm: 820,
  torque_nm: 54,
  tool_wear_min: 218,
}

const STATUS_COLOUR: Record<string, string> = {
  ok: '#10b981',
  warning: '#eab308',
  skipped: '#64748b',
  blocked: '#ef4444',
}

export default function AgentsPage() {
  const catalog = useAsync(() => getAgentCatalog(), [])
  const history = useAsync(() => getInvestigations(15), [])
  const [form, setForm] = useState<Record<string, number | string>>(DEFAULTS)
  const action = useAction<[InvestigationRequest], InvestigationResponse>(investigate)
  const result = action.result

  const submit = () =>
    action.run({
      reading: {
        air_temperature_k: Number(form.air_temperature_k),
        process_temperature_k: Number(form.process_temperature_k),
        rotational_speed_rpm: Number(form.rotational_speed_rpm),
        torque_nm: Number(form.torque_nm),
        tool_wear_min: Number(form.tool_wear_min),
      },
      create_alerts: true,
      create_recommendations: true,
    })

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-50">AI Investigation Center</h1>
          <p className="mt-1 text-[13px] text-slate-400">
            Six agents run as an acyclic LangGraph workflow. Every step is traced, every claim is
            sourced, and nothing is executed without human approval.
          </p>
        </div>
        <Badge
          value={catalog.data?.graph?.llm_active ? 'ok' : 'info'}
          label={catalog.data?.graph?.llm_active ? 'LLM ENGINE' : 'RULE-BASED ENGINE'}
        />
      </div>

      <Panel title="Agent workflow" subtitle={String(catalog.data?.graph?.backend ?? 'loading')} dense>
        <div className="flex flex-wrap items-center gap-2 p-2">
          {(catalog.data?.agents ?? []).map((agent, index) => (
            <div key={agent.key} className="flex items-center gap-2">
              <div className="rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-2">
                <p className="flex items-center gap-1.5 text-[12px] font-semibold text-slate-200">
                  <Bot className="h-3.5 w-3.5 text-cyan-brand" />
                  {agent.label}
                </p>
                <p className="mt-0.5 max-w-[190px] text-[10.5px] leading-snug text-slate-500">
                  {agent.description}
                </p>
              </div>
              {index < (catalog.data?.agents?.length ?? 0) - 1 && (
                <span className="text-slate-700">→</span>
              )}
            </div>
          ))}
        </div>
        {catalog.data && (
          <div className="border-t border-white/[0.06] p-4">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Safety guarantees
            </p>
            <ul className="grid gap-1.5 md:grid-cols-2">
              {catalog.data.guarantees.map((guarantee, index) => (
                <li key={index} className="flex items-start gap-1.5 text-[11.5px] leading-snug text-slate-400">
                  <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0 text-risk-healthy" />
                  {guarantee}
                </li>
              ))}
            </ul>
          </div>
        )}
      </Panel>

      <div className="grid gap-5 lg:grid-cols-4">
        <Panel className="lg:col-span-1" title="Trigger" subtitle="Submit a degraded reading">
          <div className="space-y-3">
            {Object.entries(DEFAULTS).map(([key]) => (
              <div key={key}>
                <label className="label" htmlFor={key}>
                  {key.replace(/_/g, ' ')}
                </label>
                <input
                  id={key}
                  type="number"
                  step="0.1"
                  value={Number(form[key])}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, [key]: Number(event.target.value) }))
                  }
                  className="input font-mono"
                />
              </div>
            ))}
            <button className="btn-primary w-full" onClick={submit} disabled={action.pending}>
              <Play className="h-4 w-4" />
              {action.pending ? 'Investigating...' : 'Run investigation'}
            </button>
          </div>
        </Panel>

        <div className="space-y-5 lg:col-span-3">
          {action.pending && <Loading label="Running the agent graph" />}
          {Boolean(action.error) && <ErrorBanner error={action.error} onRetry={submit} />}

          {!action.pending && !result && (
            <Panel title="Awaiting a trigger">
              <Notice tone="info" title="No investigation run">
                <p>
                  The default reading is deliberately abnormal (high temperature, low speed, high
                  torque and near end-of-life tool wear) so the workflow produces a full trace.
                </p>
              </Notice>
            </Panel>
          )}

          {result && (
            <>
              <div className="grid gap-4 sm:grid-cols-4">
                <StatCard label="Verdict" value={result.verdict} tone={result.verdict === 'supported' ? 'good' : 'warn'} />
                <StatCard label="Steps executed" value={result.steps_executed} />
                <StatCard label="Duration" value={result.duration_ms.toFixed(0)} unit="ms" />
                <StatCard
                  label="Recommendations"
                  value={result.recommendations.length}
                  hint="all awaiting approval"
                />
              </div>

              <Panel title="Execution trace" subtitle={`${result.steps_executed} steps · ${result.engine}`}>
                <ol className="space-y-2.5">
                  {result.trace.map((step, index) => (
                    <motion.li
                      key={`${step.step}-${step.agent}-${index}`}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05, duration: 0.25 }}
                      className="flex items-start gap-3"
                    >
                      <span
                        className="mt-1 grid h-6 w-6 shrink-0 place-items-center rounded-full text-[10px] font-bold"
                        style={{
                          background: `${STATUS_COLOUR[step.status] ?? '#64748b'}22`,
                          color: STATUS_COLOUR[step.status] ?? '#64748b',
                        }}
                      >
                        {step.step}
                      </span>
                      <div className="min-w-0 flex-1 border-l border-white/[0.06] pb-1 pl-3">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-[12px] font-semibold capitalize text-slate-200">
                            {step.agent}
                          </span>
                          <span className="mono text-[10px] text-slate-600">{step.action}</span>
                          <Badge value={step.status} />
                          {step.duration_ms !== null && (
                            <span className="mono ml-auto text-[10px] text-slate-600">
                              {step.duration_ms}ms
                            </span>
                          )}
                        </div>
                        <p className="mt-1 text-[11.5px] leading-snug text-slate-400">{step.summary}</p>
                      </div>
                    </motion.li>
                  ))}
                </ol>
              </Panel>

              {result.root_causes.length > 0 && (
                <Panel title="Root-cause candidates" subtitle="Hypotheses requiring physical confirmation">
                  <div className="space-y-3">
                    {result.root_causes.map((cause, index) => (
                      <div key={index} className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <p className="text-[13px] font-semibold text-slate-100">{cause.hypothesis}</p>
                          <span className="badge-info">{(cause.confidence * 100).toFixed(0)}% confidence</span>
                        </div>
                        <p className="mt-1.5 text-[12px] leading-relaxed text-slate-400">
                          {cause.mechanism}
                        </p>
                        <p className="mt-2 flex items-start gap-1.5 text-[11px] leading-snug text-amber-300/80">
                          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
                          {cause.causality_note}
                        </p>
                        <p className="mono mt-1.5 text-[10px] text-slate-600">
                          basis: {cause.evidence_basis} · source: {cause.source}
                        </p>
                      </div>
                    ))}
                  </div>
                </Panel>
              )}

              {result.recommendations.length > 0 && (
                <Panel
                  title="Maintenance recommendations"
                  subtitle="Pending human approval — nothing is executed automatically"
                >
                  <div className="space-y-3">
                    {result.recommendations.map((rec) => (
                      <div key={rec.id} className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <p className="text-[13px] font-semibold text-slate-100">{rec.title}</p>
                          <div className="flex items-center gap-2">
                            <Badge value={rec.status} />
                            <span className="badge-info">{rec.priority}</span>
                          </div>
                        </div>
                        <p className="mt-1.5 text-[12px] leading-relaxed text-slate-400">
                          {rec.description}
                        </p>
                        {rec.rationale && (
                          <p className="mt-2 text-[11px] leading-snug text-slate-500">{rec.rationale}</p>
                        )}
                        <div className="mt-2 flex items-center gap-3 border-t border-white/[0.06] pt-2">
                          <span className="mono text-[10px] text-slate-600">id {rec.id}</span>
                          <a href="/maintenance" className="ml-auto text-[11px] font-semibold text-cyan-brand hover:underline">
                            Go to approval queue →
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                </Panel>
              )}

              {result.unsupported_claims.length > 0 && (
                <Panel title="Unsupported claims" subtitle="Flagged by the verification agent">
                  <ul className="space-y-2">
                    {result.unsupported_claims.map((claim, index) => (
                      <li key={index} className="rounded-lg border border-risk-high/25 bg-risk-high/[0.06] p-3">
                        <p className="text-[12px] font-semibold text-risk-high">{claim.claim}</p>
                        <p className="mt-1 text-[11.5px] leading-snug text-slate-400">{claim.reason}</p>
                        <p className="mt-1 text-[11px] leading-snug text-slate-500">
                          Needs: {claim.required_evidence}
                        </p>
                      </li>
                    ))}
                  </ul>
                </Panel>
              )}

              {result.citations.length > 0 && (
                <Panel title="Document citations" subtitle="Every citation resolved to an indexed passage">
                  <ul className="space-y-1.5">
                    {result.citations.map((citation, index) => (
                      <li key={index} className="flex items-center justify-between gap-3 text-[11.5px]">
                        <span className="truncate text-slate-300">[{citation.index}] {citation.reference}</span>
                        <span className="mono shrink-0 text-[10px] text-slate-600">
                          score {citation.score.toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </Panel>
              )}

              <div className="flex items-center gap-2 rounded-lg border border-risk-healthy/25 bg-risk-healthy/[0.06] p-3 text-[11.5px] text-emerald-200">
                <ShieldCheck className="h-4 w-4 shrink-0" />
                Human approval is mandatory. Approve or reject these actions on the Maintenance
                History page.
              </div>
            </>
          )}
        </div>
      </div>

      <Panel title="Investigation history" subtitle="Most recent runs" dense>
        {history.loading ? (
          <Loading />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[12px]">
              <thead>
                <tr className="border-b border-white/[0.08]">
                  {['Started', 'Verdict', 'Engine', 'Steps', 'Causes', 'Citations', 'Summary'].map((header) => (
                    <th key={header} className="px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(history.data?.items ?? []).map((item, index) => (
                  <tr key={index} className="border-b border-white/[0.04] last:border-0">
                    <td className="mono px-3 py-2 text-[10px] text-slate-500">
                      {new Date(String(item.created_at)).toLocaleString()}
                    </td>
                    <td className="px-3 py-2">
                      <Badge value={String(item.verdict ?? 'unknown')} />
                    </td>
                    <td className="mono px-3 py-2 text-[10px] text-slate-500">{String(item.engine)}</td>
                    <td className="px-3 py-2 tabular-nums text-slate-300">{String(item.steps_executed)}</td>
                    <td className="px-3 py-2 tabular-nums text-slate-300">{String(item.root_cause_count)}</td>
                    <td className="px-3 py-2 tabular-nums text-slate-300">{String(item.citation_count)}</td>
                    <td className="max-w-md truncate px-3 py-2 text-slate-400">{String(item.summary ?? '')}</td>
                  </tr>
                ))}
                {!history.data?.items.length && (
                  <tr>
                    <td colSpan={7} className="px-3 py-8 text-center text-slate-500">
                      No investigations recorded yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  )
}
