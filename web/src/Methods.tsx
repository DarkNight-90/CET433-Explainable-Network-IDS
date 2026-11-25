import { useState } from 'react'
import { ArrowDownRight, Beaker, CircleCheck, Database, Fingerprint, GitBranch, Info, ListChecks } from 'lucide-react'
import { phase1, provider, references, repository } from './data'
import { Badge, External, PageHeader, Panel } from './ui'

const cases = {
  fp: { title: 'False-positive review', truth: 'Normal → Flagged', description: 'A scheduled backup appears unusual because its volume and duration resemble suspicious transfers. The analyst would check an approved job record before closing the alert.', values: [['Bytes sent', .72], ['Burst volume', .49], ['Known service context', -.38], ['Session duration', .24]] as const },
  fn: { title: 'False-negative review', truth: 'Attack → Missed', description: 'A scripted harmful flow resembles routine HTTPS in the example. An analyst would compare the flow with independent context and consider which useful signals were absent.', values: [['Common service', -.71], ['Low byte volume', -.49], ['Repeated attempts', .31], ['Short duration', -.22]] as const },
}
const globalExample = [['Bytes sent', .69], ['Service', .54], ['Packet count', .44], ['Duration', .33]] as const

export function Explainability() {
  const [view, setView] = useState<'local' | 'global'>('local')
  const [scenario, setScenario] = useState<'fp' | 'fn'>('fp')
  const example = cases[scenario]
  const values = view === 'global' ? globalExample : example.values
  return <>
    <PageHeader tag="Phase 2 design" title="Explainability Center" description="Explore how analysts could examine a model’s evidence and assess where a decision went wrong."/>
    <div className="callout xai-notice"><Beaker size={21} aria-hidden="true"/><strong>Explainability research in progress — example attributions are illustrative.</strong></div>
    <div className="lab-grid xai-grid">
      <Panel title="Attribution example" description="Hand-authored values in arbitrary demonstration units.">
        <div className="segmented" aria-label="Explanation scope"><button aria-pressed={view === 'local'} onClick={() => setView('local')}>Local explanation</button><button aria-pressed={view === 'global'} onClick={() => setView('global')}>Global summary</button></div>
        {view === 'local' && <div className="segmented" aria-label="Illustrative error case"><button aria-pressed={scenario === 'fp'} onClick={() => setScenario('fp')}>False positive</button><button aria-pressed={scenario === 'fn'} onClick={() => setScenario('fn')}>False negative</button></div>}
        <p className="muted">{view === 'global' ? 'Example aggregate magnitude across a reference population.' : 'Positive points toward Attack. Negative points toward Normal.'}</p>
        <ul className="attribution-list">{values.map(([label, value]) => <li key={label}><span>{label}</span><div className={'attribution-track ' + (view === 'local' ? 'diverging' : '')} aria-hidden="true"><span className={value < 0 ? 'negative' : 'positive'} style={{ width: (Math.abs(value) * (view === 'local' ? 50 : 100)) + '%' }}/></div><strong>{view === 'local' && value > 0 ? '+' : ''}{value.toFixed(2)}</strong></li>)}</ul>
      </Panel>
      <Panel title={view === 'global' ? 'Reading a global summary' : example.title} description={view === 'local' ? example.truth : 'Population-level patterns'}>
        <p>{view === 'global' ? 'A global summary combines attribution magnitudes across many rows. It helps identify recurring influences. A value for one flow can still differ in size or direction.' : example.description}</p>
        <div className="callout"><Info size={20} aria-hidden="true"/><div><strong>Interpret with context</strong><p>Feature associations can reflect correlations, feature transformations or background choices. Attribution does not establish network intent or causality.</p></div></div>
      </Panel>
    </div>
    <div className="overview-bottom">
      <Panel title="SHAP"><p>Attribute a prediction to feature contributions relative to a reference. Planned validation covers the output scale, positive class, one-hot feature mapping and additive consistency.</p></Panel>
      <Panel title="LIME"><p>Fit a local surrogate around an example to approximate the decision nearby. Planned checks cover neighborhood plausibility, fidelity and stability across settings.</p></Panel>
    </div>
  </>
}

export function Methodology() {
  return <>
    <PageHeader tag="Research methods" title="Dataset & Methodology" description="UNSW-NB15 flow data, frozen exact-feature groups and fit-only preprocessing."/>
    <div className="stats-grid">{[
      ['175,341', 'Official training observations'], ['82,332', 'Official test observations · provider count'], ['140,273', 'M02.2 internal fit'], ['35,068', 'M02.2 internal validation'],
    ].map(([count, label]) => <div className="panel stat methodology-stat" key={label}><Database size={20} className="cyan" aria-hidden="true"/><strong>{count}</strong><span>{label}</span></div>)}</div>
    <div className="overview-bottom">
      <Panel title="Published training composition" description="68.06% Attack · 31.94% Normal · 45 original columns">
        <figure><img className="evidence-image" src="/evidence/training-labels.png" alt="Official training CSV class distributions. 119,341 Attack rows and 56,000 Normal rows. Generic and Exploits are the most common attack families."/><figcaption>Original Phase 1 training-partition EDA figure</figcaption></figure>
      </Panel>
      <Panel title="Exact-feature leakage controls" description="Frozen M02.2 positional assignments">
        <dl className="method-findings">
          <div><dt>Extra exact model-input vectors</dt><dd>74,301</dd></div>
          <div><dt>Conflicting binary / family label groups</dt><dd>229 / 1,772</dd></div>
          <div><dt>Fit / validation exact group overlap</dt><dd>0</dd></div>
          <div><dt>Frozen CV folds with zero overlap</dt><dd>5 of 5</dd></div>
        </dl>
        <p>Each exact group keeps its source rows and labels. Near-duplicates and related sessions can still span groups. The supplied partitions do not establish chronological generalization.</p>
      </Panel>
    </div>
    <Panel title="Preprocessing inside each fitted pipeline" description="Exclude id and both targets. The remaining 42 inputs contain 39 numeric and three categorical fields.">
      <div className="pipeline-steps"><div><strong>Freeze groups</strong><span>Split before learned transforms</span></div><ArrowDownRight size={19} aria-hidden="true"/><div><strong>Numeric branch</strong><span>Median → clip below zero → log1p → standardize</span></div><ArrowDownRight size={19} aria-hidden="true"/><div><strong>Categorical branch</strong><span>One-hot encode with fit-side rare-level pooling</span></div></div>
      <p>Seed 42. Balanced class weights. Each CV fit learns preprocessing only from its fitting fold.</p>
      <details><summary>View original Phase 1 pipeline diagram</summary><img className="evidence-image pipeline-image" src="/evidence/preprocessing.png" alt="Frozen internal fit and validation branches feed numeric and categorical preprocessing. Only fitting rows learn weights and transform parameters."/></details>
    </Panel>
    <Panel title="Dataset provider and citations">
      <p>UNSW grants perpetual free use for academic research with five required citations. Commercial use requires agreement with the authors. The demo contains aggregated evidence and synthetic incident examples.</p>
      <div className="flex flex-wrap gap-4"><External href={provider}>Official UNSW-NB15 dataset</External><External href={phase1}>Phase 1 reports</External></div>
      <ol className="reference-list">{references.map(([title, link]) => <li key={link}><External href={link}>{title}</External></li>)}</ol>
    </Panel>
  </>
}

const roadmap = [
  { title: 'Phase 1 research foundation', status: 'Complete', tone: 'green', icon: CircleCheck, description: 'SOC business case, training-partition EDA, frozen split checks and the Logistic Regression baseline.' },
  { title: 'Stage A and Stage B comparisons', status: 'Completed · team-reported', tone: 'cyan', icon: ListChecks, description: 'Fixed models and bounded tree optimization. Run exports and threshold-wide verification remain pending.' },
  { title: 'M03.3 independent inference audit', status: 'Implemented · execution pending', tone: 'violet', icon: Fingerprint, description: 'Authenticate the original Stage A artifacts and reproduce inference-only validation diagnostics.' },
  { title: 'Multiclass classification', status: 'Planned', tone: 'neutral', icon: GitBranch, description: 'Evaluate Normal plus nine families with macro-F1, per-class recall and support.' },
  { title: 'SHAP / LIME validation', status: 'Planned', tone: 'neutral', icon: Beaker, description: 'Verify global summaries, local fidelity and actual false-positive and false-negative cases.' },
  { title: 'Robustness testing', status: 'Planned', tone: 'neutral', icon: ListChecks, description: 'Design a defensive evaluation under distribution shift and record changes in error rates.' },
  { title: 'Model API and authenticated loading', status: 'Planned', tone: 'neutral', icon: Fingerprint, description: 'Control artifact custody, validate request schemas and measure serving latency.' },
  { title: 'Final official held-out evaluation', status: 'Pending', tone: 'amber', icon: Database, description: 'Freeze the model and operating policy before scoring; report aggregate and exact-feature-unseen strata.' },
]

export function Roadmap() {
  return <>
    <PageHeader tag="Milestones" title="Project Roadmap" description="From the completed benchmark foundation to a validated analyst-facing research prototype."/>
    <ol className="roadmap">{roadmap.map(({ title, status, tone, icon: Icon, description }) => <li className="panel" key={title}>
      <span className={'tile-icon tone-' + tone}><Icon size={20} aria-hidden="true"/></span><div><h2>{title}</h2><p>{description}</p></div><Badge tone={tone}>{status}</Badge>
    </li>)}</ol>
    <footer className="page-footer"><span>Phase 2 outcomes require their own evidence.</span><External href={repository}>Repository and methodology</External></footer>
  </>
}
