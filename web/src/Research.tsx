import { useState } from 'react'
import { CircleCheck, FlaskConical, Info } from 'lucide-react'
import { cost, number, percent, phase1, precision, results } from './data'
import type { Matrix, Result } from './data'
import { Badge, External, PageHeader, Panel } from './ui'

const metrics = [
  { id: 'f1', label: 'F1' },
  { id: 'recall', label: 'Attack recall' },
  { id: 'precision', label: 'Attack precision' },
  { id: 'fpr', label: 'FPR' },
] as const
type Metric = typeof metrics[number]['id']

const read = (row: Result, metric: Metric) => metric === 'precision' ? precision(row) : row[metric]

function Confusion({ matrix }: { matrix: Matrix }) {
  const rows = [
    { actual: 'Normal', cells: [['TN', matrix.tn], ['FP', matrix.fp]] as const },
    { actual: 'Attack', cells: [['FN', matrix.fn], ['TP', matrix.tp]] as const },
  ]
  return <table className="matrix">
    <caption>Counts and percentages within each actual class</caption>
    <thead><tr><th scope="col">Actual / predicted</th><th scope="col">Normal</th><th scope="col">Attack</th></tr></thead>
    <tbody>{rows.map(({ actual, cells }) => <tr key={actual}><th scope="row">{actual}</th>
      {cells.map(([label, value]) => <td className={label === 'TN' || label === 'TP' ? 'correct' : 'error'} key={label}><span>{label}</span><strong>{number(value)}</strong><small>{percent(value / (cells[0][1] + cells[1][1]))}</small></td>)}
    </tr>)}</tbody>
  </table>
}

export default function Research() {
  const [stage, setStage] = useState<'All' | 'A' | 'B'>('All')
  const [metric, setMetric] = useState<Metric>('f1')
  const [modelId, setModelId] = useState('a-lr')
  const visible = results.filter(row => stage === 'All' || row.stage === stage)
  const selected = results.find(row => row.id === modelId) ?? results[0]
  const costs = results.flatMap(row => row.matrix ? [{ label: (row.stage === 'B' ? 'Tuned ' : '') + row.model, value: cost(row.matrix) }] : [])
  return <>
    <PageHeader tag="Research evidence" title="ML Research Lab" description="Measured binary results on 35,068 frozen internal validation flows at threshold 0.5. The official test partition remains unopened for model evaluation.">
      <Badge tone="cyan"><FlaskConical size={15} aria-hidden="true"/> Internal validation</Badge>
    </PageHeader>
    <div className="research-context">
      <div className="panel context-card"><Badge tone="blue">Stage A</Badge><h2>Fixed baselines</h2><p>Logistic Regression, Random Forest and LightGBM use the same frozen validation population.</p></div>
      <div className="panel context-card"><Badge tone="violet">Stage B</Badge><h2>52 completed fits</h2><p>The team reports 50 fold fits and two final refits. Mean CV F1 selected settings within each tree family.</p></div>
      <div className="panel context-card"><Badge tone="amber">Operating targets</Badge><h2>Recall ≥95% · FPR &lt;1%</h2><p>Tuned Random Forest leads by validation F1 at 0.5. Both tuned models exceed the FPR limit.</p></div>
    </div>
    <div className="lab-grid">
      <Panel title="Model comparison" description="Choose a stage and metric. Bars use a 0–100% scale.">
        <div className="research-controls">
          <div className="segmented" aria-label="Compare stages">
            {(['All', 'A', 'B'] as const).map(value => <button key={value} aria-pressed={stage === value} onClick={() => setStage(value)}>{value === 'All' ? 'Both stages' : 'Stage ' + value}</button>)}
          </div>
          <label className="field-inline">Metric <select value={metric} onChange={event => setMetric(event.target.value as Metric)}>{metrics.map(item => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label>
        </div>
        <ul className="bar-list compare-bars">{visible.map(row => {
          const value = read(row, metric)
          const active = row.id === selected.id
          return <li key={row.id}>
            <button className={active ? 'compare-row active' : 'compare-row'} aria-pressed={active} onClick={() => setModelId(row.id)}>
              <span className="compare-name"><Badge tone={row.stage === 'A' ? 'blue' : 'violet'}>{row.stage}</Badge>{row.model}</span><strong>{value === undefined ? 'Unavailable' : percent(value, 3)}</strong>
            </button>
            {value !== undefined && <progress aria-label={row.model + ' Stage ' + row.stage + ' ' + metric} value={value} max={1} className={row.stage === 'B' ? 'violet-bar' : ''}/>}
          </li>
        })}</ul>
      </Panel>
      <Panel title="Confusion matrix" description={selected.model + ' · Stage ' + selected.stage}>
        {selected.matrix ? <Confusion matrix={selected.matrix}/> : <p className="empty">The Stage A tree confusion counts are unavailable in the reviewed report. Select Logistic Regression or a Stage B model.</p>}
        <div className="model-values">
          {[...metrics, { id: 'auc' as const, label: 'Reported ROC-AUC' }].map(item => {
            const value = item.id === 'auc' ? selected.auc : read(selected, item.id)
            return <div key={item.id}><span>{item.label}</span><strong>{value === undefined ? 'Unavailable' : value.toFixed(6)}</strong></div>
          })}
        </div>
      </Panel>
    </div>
    <Panel title="Estimated error cost" description="Analyst-equivalent minutes on internal validation. Policy weights model error burden as 5 × FP + 120 × FN.">
      <ul className="bar-list cost-bars">{costs.map(({ label, value }) => <li key={label}><div><span>{label}</span><strong>{number(value)} min</strong></div><progress value={value} max={Math.max(...costs.map(item => item.value))} aria-label={label + ' estimated cost'}/></li>)}</ul>
    </Panel>
    <div className="callout research-note"><Info size={20} aria-hidden="true"/><div><strong>Evidence provenance</strong><p>Confusion-derived metrics and costs were checked against supplied counts. Stage A tree summaries, ROC-AUC scalars, fit totals and threshold-wide infeasibility come from team reports. Original score vectors and run exports were unavailable for independent replication. The lab displays no synthetic ROC points.</p><External href={phase1.replace('/tree/', '/blob/') + '/baseline.md'}>Read the Phase 1 baseline report</External></div></div>
    <p className="integrity-line"><CircleCheck size={16} aria-hidden="true"/> Cohort supports from frozen M02.2: 11,200 Normal · 23,868 Attack</p>
  </>
}
