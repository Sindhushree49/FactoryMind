import { Route, Routes } from 'react-router-dom'

import AppShell from '@/components/AppShell'
import {
  AgentsPage,
  AnalyticsPage,
  AnomalyPage,
  DashboardPage,
  DigitalTwinPage,
  KnowledgePage,
  LandingPage,
  MachinesPage,
  MaintenancePage,
  PredictivePage,
  SettingsPage,
  VisionPage,
} from '@/pages'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route element={<AppShell />}>
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/machines" element={<MachinesPage />} />
        <Route path="/predictive" element={<PredictivePage />} />
        <Route path="/anomaly" element={<AnomalyPage />} />
        <Route path="/vision" element={<VisionPage />} />
        <Route path="/agents" element={<AgentsPage />} />
        <Route path="/twin" element={<DigitalTwinPage />} />
        <Route path="/knowledge" element={<KnowledgePage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/maintenance" element={<MaintenancePage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<DashboardPage />} />
      </Route>
    </Routes>
  )
}
