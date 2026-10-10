import {
  Activity,
  Bot,
  Boxes,
  Brain,
  Camera,
  Cpu,
  FileText,
  Gauge,
  LayoutDashboard,
  Menu,
  Settings as SettingsIcon,
  Wrench,
  X,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'

import { useHealth } from '@/hooks/useHealth'
import { Badge } from '@/components/ui'

const NAV = [
  { to: '/dashboard', label: 'Executive Dashboard', icon: LayoutDashboard, group: 'Overview' },
  { to: '/machines', label: 'Machine Intelligence', icon: Cpu, group: 'Overview' },
  { to: '/predictive', label: 'Predictive Maintenance', icon: Gauge, group: 'Models' },
  { to: '/anomaly', label: 'Anomaly Detection', icon: Activity, group: 'Models' },
  { to: '/vision', label: 'Visual Inspection', icon: Camera, group: 'Models' },
  { to: '/twin', label: 'Digital Twin', icon: Boxes, group: 'Models' },
  { to: '/agents', label: 'AI Investigation Center', icon: Bot, group: 'Agentic' },
  { to: '/knowledge', label: 'Knowledge Base', icon: FileText, group: 'Agentic' },
  { to: '/maintenance', label: 'Maintenance History', icon: Wrench, group: 'Operations' },
  { to: '/analytics', label: 'Analytics & Reports', icon: Brain, group: 'Operations' },
  { to: '/settings', label: 'Settings', icon: SettingsIcon, group: 'Operations' },
]

export default function AppShell() {
  const [open, setOpen] = useState(false)
  const location = useLocation()
  const { data: health } = useHealth()

  useEffect(() => {
    setOpen(false)
  }, [location.pathname])

  const groups = NAV.reduce<Record<string, typeof NAV>>((acc, item) => {
    ;(acc[item.group] ??= []).push(item)
    return acc
  }, {})

  const online = health?.status === 'ok'
  const degraded = health?.components.filter((c) => c.status !== 'ok') ?? []

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-[268px] flex-col border-r border-white/[0.06] bg-navy-900/95 backdrop-blur-xl transition-transform duration-300 lg:static lg:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center gap-3 border-b border-white/[0.06] px-5 py-4">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-gradient-to-br from-cyan-400 to-azure-brand">
            <Brain className="h-5 w-5 text-navy-950" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-[15px] font-bold tracking-tight text-slate-50">FactoryMind</p>
            <p className="truncate text-[10px] uppercase tracking-widest text-slate-500">
              Industrial AI Platform
            </p>
          </div>
          <button className="btn-ghost lg:hidden" onClick={() => setOpen(false)} aria-label="Close menu">
            <X className="h-4 w-4" />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 py-4">
          {Object.entries(groups).map(([group, items]) => (
            <div key={group} className="mb-5">
              <p className="mb-2 px-3 text-[10px] font-bold uppercase tracking-widest text-slate-600">
                {group}
              </p>
              <ul className="space-y-0.5">
                {items.map(({ to, label, icon: Icon }) => (
                  <li key={to}>
                    <NavLink
                      to={to}
                      className={({ isActive }) =>
                        `group flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium transition-all ${
                          isActive
                            ? 'bg-cyan-brand/10 text-cyan-300 shadow-[inset_2px_0_0_0_#22d3ee]'
                            : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-100'
                        }`
                      }
                    >
                      <Icon className="h-4 w-4 shrink-0" />
                      <span className="truncate">{label}</span>
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <div className="border-t border-white/[0.06] p-3">
          <div className="rounded-lg bg-white/[0.03] p-3">
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-2 text-[11px] font-semibold text-slate-300">
                <span
                  className={`h-2 w-2 rounded-full ${
                    online ? 'bg-risk-healthy' : degraded.length ? 'bg-risk-watch' : 'bg-risk-critical'
                  } ${degraded.length ? 'animate-pulseRing' : ''}`}
                />
                {online ? 'All systems nominal' : degraded.length ? 'Degraded' : 'API unreachable'}
              </span>
            </div>
            <p className="mt-1.5 text-[10px] leading-snug text-slate-500">
              {health
                ? `v${health.version} · ${health.environment} · ${health.llm_engine === 'llm' ? 'LLM engine' : 'rule-based'}`
                : 'Connecting to the API...'}
            </p>
            {degraded.length > 0 && (
              <ul className="mt-2 space-y-1">
                {degraded.slice(0, 3).map((component) => (
                  <li key={component.name} className="text-[10px] leading-snug text-slate-500">
                    <span className="text-slate-400">{component.name.replace(/_/g, ' ')}</span>:{' '}
                    {component.detail.slice(0, 70)}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </aside>

      {/* Mobile overlay */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-navy-950/70 backdrop-blur-sm lg:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Content */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-white/[0.06] bg-navy-950/80 px-4 py-3 backdrop-blur-xl lg:hidden">
          <button className="btn-ghost" onClick={() => setOpen(true)} aria-label="Open menu">
            <Menu className="h-5 w-5" />
          </button>
          <span className="text-sm font-semibold text-slate-100">FactoryMind</span>
          {health && <Badge value={health.status} label={health.status.toUpperCase()} />}
        </header>
        <main className="min-w-0 flex-1 px-4 py-5 sm:px-6 lg:px-8">
          <Outlet />
        </main>
        <footer className="border-t border-white/[0.06] px-6 py-4 text-[11px] text-slate-600">
          FactoryMind — every figure is computed from the backend's own database and trained
          models. Simulated and untrained states are always labelled.
        </footer>
      </div>
    </div>
  )
}
