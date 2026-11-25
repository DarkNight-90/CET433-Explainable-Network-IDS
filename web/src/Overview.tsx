import { Activity, ArrowRight, CircleCheck, Database, Fingerprint, FlaskConical, Network, Play, ScanLine, Shield, ShieldCheck, TriangleAlert } from 'lucide-react'
import { team, phase1 } from './data'
import { Badge, External, Panel } from './ui'

function NetworkVisual() {
  return <div className="network-visual" aria-hidden="true">
    <div className="network-cross horizontal"/><div className="network-cross vertical"/>
    <div className="radar-ring ring-one"/><div className="radar-ring ring-two"/><div className="radar-ring ring-three"/>
    <div className="network-link link-one"/><div className="network-link link-two"/><div className="network-link link-three"/><div className="network-link link-four"/>
    <div className="network-node node-one"><Network size={18}/></div>
    <div className="network-node node-two"><Activity size={19}/></div>
    <div className="network-node node-three"><Fingerprint size={18}/></div>
    <div className="network-node node-four"><TriangleAlert size={17}/></div>
    <div className="network-core"><ShieldCheck strokeWidth={1.35} size={49}/></div>
  </div>
}

const metrics = [
  { label: 'Official training flows', value: '175,341', hint: 'UNSW-NB15 training partition', icon: Database, tone: 'cyan' },
  { label: 'Internal validation flows', value: '35,068', hint: 'Frozen M02.2 partition', icon: ScanLine, tone: 'blue' },
  { label: 'Baseline attack F1', value: '95.14%', hint: 'Logistic Regression · threshold 0.5', icon: Activity, tone: 'violet' },
  { label: 'Shared exact-feature groups', value: '0', hint: 'Internal fit / validation boundary', icon: ShieldCheck, tone: 'green' },
]

export default function Overview({ launch }: { launch: () => void }) {
  return <>
    <section className="hero">
      <NetworkVisual/>
      <div className="hero-copy">
        <div className="hero-label"><Shield size={15} aria-hidden="true"/> CET433 · Academic SOC demonstration</div>
        <h1>Network defense,<br/><span>explained.</span></h1>
        <p>Investigate a suspicious network flow, record an analyst decision, and explore the measured research behind an explainable intrusion detection system.</p>
        <div className="flex flex-wrap gap-3">
          <button className="button primary" onClick={launch}><Play size={16} fill="currentColor" aria-hidden="true"/> Launch SOC Experience <ArrowRight size={17} aria-hidden="true"/></button>
          <a className="button secondary" href="#research"><FlaskConical size={17} aria-hidden="true"/> Explore research</a>
        </div>
      </div>
    </section>

    <section className="evidence-section" aria-labelledby="evidence-title">
      <div className="section-heading"><h2 id="evidence-title">Phase 1 at a glance</h2><Badge tone="cyan"><CircleCheck size={14} aria-hidden="true"/> Research foundation complete</Badge></div>
      <div className="stats-grid">
        {metrics.map(({ label, value, hint, icon: Icon, tone }) => <div className="stat panel" key={label}>
          <span className={'tile-icon tone-' + tone}><Icon size={19} aria-hidden="true"/></span>
          <span className="stat-label">{label}</span><strong>{value}</strong><span className="muted">{hint}</span>
        </div>)}
      </div>
    </section>

    <div className="overview-bottom">
      <Panel title="The analyst’s problem" description="For enterprise SOC teams and managed security service providers">
        <div className="overview-points">
          <div><span className="tile-icon tone-amber"><TriangleAlert size={17} aria-hidden="true"/></span><div><h3>High alert volume</h3><p>False alarms consume triage time. Missed attacks can delay response. The study weights those errors at 5 and 120 analyst-equivalent minutes respectively.</p></div></div>
          <div><span className="tile-icon tone-cyan"><Fingerprint size={17} aria-hidden="true"/></span><div><h3>Decision support analysts can question</h3><p>A classifier can prioritize suspicious flows. Explanations could help analysts examine the evidence, consider benign context, and document a decision.</p></div></div>
        </div>
      </Panel>
      <Panel title="Project team" description="CET433 · Network & Cyber Security">
        <ul className="team-list">{team.map(({ name, id, initials }) => <li key={id}><span className="avatar" aria-hidden="true">{initials}</span><div><strong>{name}</strong><span>{id}</span></div></li>)}</ul>
      </Panel>
    </div>
    <footer className="page-footer"><span>Academic decision-support prototype</span><External href={phase1}>Phase 1 evidence bundle</External></footer>
  </>
}
