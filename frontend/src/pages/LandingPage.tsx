import {
  Activity,
  ArrowRight,
  Bot,
  Brain,
  Camera,
  CheckCircle2,
  Cpu,
  Database,
  FileText,
  Gauge,
  Layers,
  ShieldCheck,
  Workflow,
} from 'lucide-react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

import { useHealth } from '@/hooks/useHealth'

const PIPELINE = [
  { label: 'Data collection', icon: Database },
  { label: 'Preprocessing', icon: Layers },
  { label: 'Failure prediction', icon: Gauge },
  { label: 'Anomaly detection', icon: Activity },
  { label: 'RUL estimation', icon: Cpu },
  { label: 'Visual inspection', icon: Camera },
  { label: 'Root-cause investigation', icon: Bot },
  { label: 'Maintenance planning', icon: Workflow },
  { label: 'Verification', icon: ShieldCheck },
  { label: 'Human approval', icon: CheckCircle2 },
]

const AGENTS = [
  { name: 'Monitoring', desc: 'Detects abnormal telemetry and raises graded alerts.' },
  { name: 'Diagnosis', desc: 'Attributes risk using SHAP plus a documented sensor-pattern rule base.' },
  { name: 'RUL', desc: 'Estimates remaining life and states its own uncertainty.' },
  { name: 'Documentation', desc: 'Retrieves procedures from indexed manuals with citations.' },
  { name: 'Planning', desc: 'Converts supported hypotheses into prioritised actions.' },
  { name: 'Verification', desc: 'Flags unsupported claims and holds for human approval.' },
]

const DATASETS = [
  {
    name: 'AI4I 2020',
    role: 'Failure prediction & failure-mode classification',
    licence: 'CC BY 4.0 · UCI dataset 601',
  },
  {
    name: 'NASA C-MAPSS',
    role: 'Remaining Useful Life regression (split by engine unit)',
    licence: 'US Government public domain',
  },
  {
    name: 'MVTec AD',
    role: 'Unsupervised visual anomaly detection & localisation',
    licence: 'CC BY-NC-SA 4.0 · manual registration',
  },
]

const DIFFERENTIATORS = [
  {
    title: 'No invented numbers',
    body: 'An untrained model returns 503 with the exact training command. The UI labels untrained and simulated states explicitly rather than showing a plausible-looking placeholder.',
  },
  {
    title: 'Attribution, not causation',
    body: 'SHAP values are reported as model behaviour, and every explanation carries that qualifier. Root causes are hypotheses requiring physical confirmation.',
  },
  {
    title: 'Bounded agents',
    body: 'The investigation graph is acyclic, each node enforces a step budget, and agents get read-only access to measurements — so no agent can loop or fabricate a sensor reading.',
  },
  {
    title: 'Approval is enforced server-side',
    body: 'A recommendation cannot reach "completed" without first passing through "approved". The state machine rejects out-of-order transitions with HTTP 409.',
  },
]

export default function LandingPage() {
  const { data: health } = useHealth(60_000)
  const trained = health?.components.filter((c) => c.status === 'ok').length ?? 0
  const total = health?.components.length ?? 0

  return (
    <div className="min-h-screen">
      {/* Nav */}
      <header className="sticky top-0 z-40 border-b border-white/[0.06] bg-navy-950/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg bg-gradient-to-br from-cyan-400 to-azure-brand">
              <Brain className="h-5 w-5 text-navy-950" />
            </div>
            <div>
              <p className="text-[15px] font-bold tracking-tight text-slate-50">FactoryMind</p>
              <p className="text-[10px] uppercase tracking-widest text-slate-500">
                Agentic Predictive Maintenance
              </p>
            </div>
          </div>
          <nav className="hidden items-center gap-6 text-[13px] text-slate-400 md:flex">
            <a href="#pipeline" className="transition hover:text-slate-100">Pipeline</a>
            <a href="#agents" className="transition hover:text-slate-100">Agents</a>
            <a href="#datasets" className="transition hover:text-slate-100">Datasets</a>
            <a href="#integrity" className="transition hover:text-slate-100">Integrity</a>
            <Link to="/dashboard" className="btn-primary">
              Open platform <ArrowRight className="h-4 w-4" />
            </Link>
          </nav>
          <Link to="/dashboard" className="btn-primary md:hidden">
            Open
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="grid-bg absolute inset-0 opacity-70" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-navy-950/40 to-navy-950" />
        <div className="relative mx-auto max-w-7xl px-6 py-20 sm:py-28">
          <motion.div
            initial={{ opacity: 0, y: 22 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55 }}
            className="max-w-3xl"
          >
            <span className="badge-info">
              <Activity className="h-3 w-3" />
              {health
                ? `${trained}/${total} subsystems healthy · ${health.llm_engine === 'llm' ? 'LLM engine active' : 'rule-based agent engine'}`
                : 'Connecting to backend...'}
            </span>
            <h1 className="mt-6 text-4xl font-extrabold leading-[1.1] tracking-tight text-slate-50 sm:text-6xl">
              Industrial failures are
              <br />
              <span className="text-gradient">predictable before they happen.</span>
            </h1>
            <p className="mt-6 max-w-2xl text-[15px] leading-relaxed text-slate-400">
              FactoryMind combines gradient-boosted failure prediction, Isolation Forest anomaly
              detection, LSTM and XGBoost remaining-life regression, PatchCore visual inspection,
              SHAP attribution and a six-agent LangGraph investigation workflow — over three real
              public research datasets.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link to="/dashboard" className="btn-primary px-5 py-2.5">
                Executive dashboard <ArrowRight className="h-4 w-4" />
              </Link>
              <Link to="/agents" className="btn-secondary px-5 py-2.5">
                Run an investigation
              </Link>
              <a href="/docs" className="btn-ghost px-5 py-2.5">
                API reference
              </a>
            </div>
            <p className="mt-6 max-w-xl text-[11px] leading-relaxed text-slate-600">
              Every prediction on this platform is produced by a trained model or a documented
              rule. Nothing is a placeholder. When a model is missing, the interface says so and
              prints the command that produces it.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Pipeline */}
      <section id="pipeline" className="border-t border-white/[0.06] bg-navy-900/40">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <h2 className="section-title">End-to-end pipeline</h2>
          <p className="mt-1.5 text-[13px] text-slate-400">
            Ten stages from telemetry to a human-approved maintenance action. Each stage is a
            separately testable module with its own metrics.
          </p>
          <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {PIPELINE.map((stage, index) => (
              <motion.div
                key={stage.label}
                initial={{ opacity: 0, y: 14 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: index * 0.03 }}
                className="card card-hover relative p-4"
              >
                <div className="flex items-center gap-2.5">
                  <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-cyan-brand/10 text-cyan-brand">
                    <stage.icon className="h-4 w-4" />
                  </span>
                  <span className="mono text-slate-600">{String(index + 1).padStart(2, '0')}</span>
                </div>
                <p className="mt-3 text-[13px] font-semibold text-slate-200">{stage.label}</p>
                {index < PIPELINE.length - 1 && (
                  <ArrowRight className="absolute -right-2.5 top-1/2 hidden h-3.5 w-3.5 -translate-y-1/2 text-slate-700 lg:block" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Agents */}
      <section id="agents" className="border-t border-white/[0.06]">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="section-title">Six specialised agents</h2>
              <p className="mt-1.5 max-w-2xl text-[13px] text-slate-400">
                Orchestrated by LangGraph in a strictly acyclic sequence. Each agent reads model
                output, writes a trace entry, and states its own confidence.
              </p>
            </div>
            <Link to="/agents" className="btn-secondary">
              Open investigation center
            </Link>
          </div>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {AGENTS.map((agent, index) => (
              <motion.div
                key={agent.name}
                initial={{ opacity: 0, y: 14 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className="card card-hover p-5"
              >
                <div className="flex items-center gap-2">
                  <Bot className="h-4 w-4 text-cyan-brand" />
                  <span className="mono text-slate-600">{String(index + 1).padStart(2, '0')}</span>
                </div>
                <h3 className="mt-3 text-[15px] font-semibold text-slate-100">{agent.name} Agent</h3>
                <p className="mt-1.5 text-[12.5px] leading-relaxed text-slate-400">{agent.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Datasets */}
      <section id="datasets" className="border-t border-white/[0.06] bg-navy-900/40">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <h2 className="section-title">Real research datasets, kept separate</h2>
          <p className="mt-1.5 max-w-3xl text-[13px] text-slate-400">
            Each dataset has its own loader, preprocessing pipeline and model family. They are
            never merged, and missing data is reported rather than synthesised.
          </p>
          <div className="mt-8 grid gap-4 lg:grid-cols-3">
            {DATASETS.map((dataset) => (
              <div key={dataset.name} className="card p-5">
                <div className="flex items-center gap-2.5">
                  <Database className="h-4 w-4 text-cyan-brand" />
                  <h3 className="text-[15px] font-semibold text-slate-100">{dataset.name}</h3>
                </div>
                <p className="mt-3 text-[12.5px] leading-relaxed text-slate-400">{dataset.role}</p>
                <p className="mono mt-4 border-t border-white/[0.06] pt-3 text-[11px] text-slate-500">
                  {dataset.licence}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Integrity */}
      <section id="integrity" className="border-t border-white/[0.06]">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <h2 className="section-title">Engineering integrity</h2>
          <p className="mt-1.5 max-w-3xl text-[13px] text-slate-400">
            The hard part of a portfolio AI project is refusing to fake the parts that do not work.
            These are the mechanisms that enforce that here.
          </p>
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            {DIFFERENTIATORS.map((item) => (
              <div key={item.title} className="card p-5">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-risk-healthy" />
                  <h3 className="text-[14px] font-semibold text-slate-100">{item.title}</h3>
                </div>
                <p className="mt-2.5 text-[12.5px] leading-relaxed text-slate-400">{item.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[0.06] bg-navy-900/60">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-8 text-[12px] text-slate-500">
          <div className="flex items-center gap-2">
            <FileText className="h-3.5 w-3.5" />
            <span>FactoryMind — agentic AI for predictive maintenance</span>
          </div>
          <div className="flex gap-5">
            <Link to="/settings" className="hover:text-slate-300">Settings</Link>
            <Link to="/analytics" className="hover:text-slate-300">Analytics</Link>
            <a href="/docs" className="hover:text-slate-300">API docs</a>
          </div>
        </div>
      </footer>
    </div>
  )
}
