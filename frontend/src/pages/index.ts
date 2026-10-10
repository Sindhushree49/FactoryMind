/**
 * Barrel for the route components declared in App.tsx.
 *
 * Every page is a default export, so each one is re-exported under the name the
 * router imports. Keeping the mapping in one place means a route and its
 * component can never drift apart silently.
 */

export { default as AgentsPage } from './AgentsPage'
export { default as AnalyticsPage } from './AnalyticsPage'
export { default as AnomalyPage } from './AnomalyPage'
export { default as DashboardPage } from './DashboardPage'
export { default as DigitalTwinPage } from './DigitalTwinPage'
export { default as KnowledgePage } from './KnowledgePage'
export { default as LandingPage } from './LandingPage'
export { default as MachinesPage } from './MachinesPage'
export { default as MaintenancePage } from './MaintenancePage'
export { default as PredictivePage } from './PredictivePage'
export { default as SettingsPage } from './SettingsPage'
export { default as VisionPage } from './VisionPage'
