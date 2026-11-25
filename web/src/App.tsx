import { lazy, Suspense, useEffect, useState, useSyncExternalStore } from 'react'
import { Activity, ArrowLeft, ArrowRight, BookOpen, Boxes, ChevronRight, FlaskConical, Gauge, GitBranch, Menu, Play, RotateCcw, Shield, X } from 'lucide-react'
import type { CaseTab } from './Dashboard'
import { incidents, initialCase } from './data'
import type { Case, Incident, Status } from './data'
import { Explainability, Methodology, Roadmap } from './Methods'
import Overview from './Overview'
import Research from './Research'
import { Badge } from './ui'

const Dashboard = lazy(() => import('./Dashboard'))

const routes = [
  { id: 'overview', name: 'Overview', icon: Gauge },
  { id: 'soc', name: 'SOC Dashboard', icon: Activity },
  { id: 'research', name: 'ML Research Lab', icon: FlaskConical },
  { id: 'explainability', name: 'Explainability Center', icon: Boxes },
  { id: 'methodology', name: 'Dataset & Methodology', icon: BookOpen },
  { id: 'roadmap', name: 'Project Roadmap', icon: GitBranch },
] as const

type Route = typeof routes[number]['id']
const getRoute = (): Route => routes.find(route => '#' + route.id === window.location.hash)?.id ?? 'overview'
const subscribe = (callback: () => void) => {
  window.addEventListener('hashchange', callback)
  return () => window.removeEventListener('hashchange', callback)
}
const guided = [
  { title: 'Observe a suspicious flow', help: 'Find “Unusual outbound transfer” in the synthetic queue, then open the case.', route: 'soc' },
  { title: 'Inspect its attributes', help: 'Check the flow endpoints, connection state and byte counts in the investigation drawer.', route: 'soc', tab: 'attributes' },
  { title: 'Review illustrative evidence', help: 'Read the scripted observation and consider the plausible benign explanation.', route: 'soc', tab: 'evidence' },
  { title: 'Investigate the incident', help: 'Acknowledge the alert, then mark it under investigation before continuing.', route: 'soc', tab: 'workflow' },
  { title: 'Record an analyst decision', help: 'Add your assessment as a note, then mark the case reviewed. You can export it locally.', route: 'soc', tab: 'workflow' },
  { title: 'Compare measured research', help: 'Compare Stage A with Stage B and select a model to inspect its confusion matrix.', route: 'research' },
  { title: 'Explore the future explanation workflow', help: 'Switch between illustrative error cases and local or global explanations.', route: 'explainability' },
] satisfies Array<{ title: string; help: string; route: Route; tab?: CaseTab }>

const focusLater = (id: string) => window.requestAnimationFrame(() => document.getElementById(id)?.focus())
const sessionTime = () => new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })

export default function App() {
  const route = useSyncExternalStore(subscribe, getRoute)
  const current = routes.find(item => item.id === route)!
  const [menuOpen, setMenuOpen] = useState(false)
  const [step, setStep] = useState<number | null>(null)
  const [finished, setFinished] = useState(false)
  const [active, setActive] = useState<Incident>()
  const [tab, setTab] = useState<CaseTab>('attributes')
  const [cases, setCases] = useState<Record<string, Case>>({})
  const exampleCase = cases[incidents[0].id]
  const currentStep = step === null ? undefined : guided[step]
  const isGuide = step !== null || finished
  const progressed = exampleCase?.status === 'Under investigation' || exampleCase?.status === 'Reviewed'
  const needed = step === 3 && !progressed ? 'Mark the case under investigation to continue.'
    : step === 4 && !(exampleCase?.status === 'Reviewed' && exampleCase.events.some(event => event.note)) ? 'Add a note and mark the case reviewed to continue.'
    : undefined

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' })
    document.title = current.name + ' · CET433 SOC Lab'
  }, [current.name])

  useEffect(() => {
    function escape(event: KeyboardEvent) {
      if (event.key !== 'Escape') return
      if (menuOpen) {
        setMenuOpen(false)
        focusLater('menu-toggle')
      } else if (isGuide) {
        setStep(null)
        setFinished(false)
        focusLater('start-tour')
      } else if (route === 'soc' && active) {
        setActive(undefined)
        focusLater('main-content')
      }
    }
    window.addEventListener('keydown', escape)
    return () => window.removeEventListener('keydown', escape)
  }, [active, isGuide, menuOpen, route])

  function goTo(next: number) {
    const target = guided[next]
    setStep(next)
    setFinished(false)
    setActive(target.tab ? incidents[0] : undefined)
    if (target.tab) setTab(target.tab)
    if (route !== target.route) window.location.hash = target.route
    focusLater('guide-title')
    if (route === 'soc' && target.tab) document.getElementById('case-heading')?.scrollIntoView({ block: 'nearest' })
  }

  function launch() {
    setMenuOpen(false)
    goTo(0)
  }

  function exit() {
    setStep(null)
    setFinished(false)
    focusLater('start-tour')
  }

  function select(incident: Incident) {
    if (incident.id === incidents[0].id && step === 0) {
      goTo(1)
      return
    }
    setActive(incident)
    setTab('attributes')
    focusLater('case-heading')
  }

  function modify(id: string, event: { label: string; note?: string }, status?: Status) {
    const incident = incidents.find(item => item.id === id)!
    setCases(previous => {
      const entry = previous[id] ?? initialCase(incident)
      return { ...previous, [id]: { status: status ?? entry.status, events: [...entry.events, { ...event, time: sessionTime() }] } }
    })
  }

  const eventFor = (status: Status) => status === 'Reviewed' ? 'Case reviewed'
    : status === 'Under investigation' && active && cases[active.id]?.status === 'Reviewed' ? 'Case reopened' : status

  const links = routes.map(({ id, name, icon: Icon }) => <a key={id} href={'#' + id} aria-current={route === id ? 'page' : undefined} onClick={() => {
    setMenuOpen(false); setStep(null); setFinished(false); focusLater('main-content')
  }}><Icon size={18} strokeWidth={1.8} aria-hidden="true"/><span>{name}</span>{route === id && <ChevronRight size={14} className="nav-chevron" aria-hidden="true"/>}</a>)

  return <div className={isGuide ? 'shell guide-open' : 'shell'}>
    <a className="skip-link" href="#main-content" onClick={event => { event.preventDefault(); focusLater('main-content') }}>Skip to main content</a>
    <aside className={'sidebar' + (menuOpen ? ' mobile-open' : '')}>
      <a className="brand" href="#overview" onClick={() => { setMenuOpen(false); setStep(null); setFinished(false) }}>
        <span className="brand-mark"><Shield size={24} strokeWidth={1.7} aria-hidden="true"/></span><span><strong>CET433 <b>SOC Lab</b></strong><small>Explainable Network IDS</small></span>
      </a>
      <div className="nav-caption">WORKSPACE</div>
      <nav aria-label="Workspace">{links}</nav>
      <div className="sidebar-foot"><span className="side-project-icon"><FlaskConical size={17} aria-hidden="true"/></span><div><strong>Research prototype</strong><span>Phase 1 evidence · Phase 2 preview</span></div></div>
    </aside>
    <div className="shell-main">
      <div className="topbar">
        <button id="menu-toggle" type="button" className="icon-button mobile-toggle" aria-label={menuOpen ? 'Close navigation' : 'Open navigation'} aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X size={22}/> : <Menu size={22}/>}</button>
        <div className="breadcrumb"><span>CET433 Lab</span><ChevronRight size={14} aria-hidden="true"/><strong>{current.name}</strong></div>
        <div className="top-actions"><Badge tone="cyan">Academic demo</Badge><button id="start-tour" className="button secondary compact" onClick={launch}><Play size={14} aria-hidden="true"/><span>Guided demo</span></button></div>
      </div>
      <main id="main-content" tabIndex={-1} className="content" key={route}>
        {route === 'overview' && <Overview launch={launch}/>}
        {route === 'soc' && <Suspense fallback={<p role="status">Loading the SOC dashboard…</p>}><Dashboard cases={cases} active={active} tab={tab} select={select} chooseTab={setTab}
          close={() => { setActive(undefined); focusLater('main-content') }}
          update={(id, status) => modify(id, { label: eventFor(status) }, status)}
          addNote={(id, note) => modify(id, { label: 'Analyst note added', note })}/></Suspense>}
        {route === 'research' && <Research/>}
        {route === 'explainability' && <Explainability/>}
        {route === 'methodology' && <Methodology/>}
        {route === 'roadmap' && <Roadmap/>}
      </main>
      {isGuide && <section className="guide" aria-label="Guided demonstration">
        <div className="guide-copy"><span className="eyebrow">{finished ? 'Walkthrough complete' : 'GUIDED DEMO · ' + (step! + 1) + ' / ' + guided.length}</span>
          <h2 id="guide-title" tabIndex={-1}>{finished ? 'You completed the SOC experience' : currentStep?.title}</h2>
          <p>{finished ? 'Replay the walkthrough or continue exploring the workspace.' : needed ?? currentStep?.help}</p>
          {step !== null && <progress aria-label="Guided demo progress" value={step + 1} max={guided.length}/>}
        </div>
        <div className="guide-actions">
          <button className="button plain compact" onClick={exit}>Exit demo</button>
          {finished ? <button className="button primary compact" onClick={launch}><RotateCcw size={16} aria-hidden="true"/> Replay</button> : <>
            <button className="button secondary compact" disabled={step === 0} onClick={() => goTo(step! - 1)}><ArrowLeft size={16} aria-hidden="true"/> Back</button>
            {step === guided.length - 1
              ? <button className="button primary compact" onClick={() => { setStep(null); setFinished(true); focusLater('guide-title') }}>Finish demo</button>
              : <button className="button primary compact" disabled={!!needed} onClick={() => goTo(step! + 1)}>Next <ArrowRight size={16} aria-hidden="true"/></button>}
          </>}
        </div>
      </section>}
    </div>
  </div>
}
