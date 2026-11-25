import { useState } from 'react'
import { Activity, ArrowRight, Check, CircleDot, Clock3, Download, ListFilter, Search, ShieldAlert, X } from 'lucide-react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { incidents, initialCase, number, statuses, trend } from './data'
import type { Case, Incident, Status } from './data'
import { Badge, PageHeader, Panel } from './ui'

export type CaseTab = 'attributes' | 'evidence' | 'workflow'
const severityTone = { Critical: 'red', High: 'amber', Medium: 'blue', Low: 'neutral' }
const statusTone = { New: 'neutral', Acknowledged: 'blue', 'Under investigation': 'amber', Reviewed: 'green' }
const labels = { attributes: 'Flow attributes', evidence: 'Evidence', workflow: 'Workflow' }

function exportCase(incident: Incident, entry: Case) {
  const data = JSON.stringify({
    provenance: 'Illustrative case — not live model inference',
    exportedAt: new Date().toISOString(), incident, investigation: entry,
  }, null, 2)
  const url = URL.createObjectURL(new Blob([data], { type: 'application/json' }))
  const link = document.createElement('a')
  link.href = url
  link.download = incident.id + '-investigation.json'
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function Drawer({ incident, entry, tab, chooseTab, close, update, addNote }: {
  incident: Incident; entry: Case; tab: CaseTab
  chooseTab: (tab: CaseTab) => void; close: () => void
  update: (status: Status) => void; addNote: (note: string) => void
}) {
  const [note, setNote] = useState('')
  const values = [
    ['Flow ID', incident.id], ['Protocol', incident.protocol], ['Service', incident.service], ['Connection state', incident.connection],
    ['Duration', incident.duration], ['Packets', number(incident.packets)], ['Bytes sent', incident.sent], ['Bytes received', incident.received],
  ]
  const actions: Array<{ label: string; status: Status }> = []
  if (entry.status === 'New') actions.push({ label: 'Acknowledge alert', status: 'Acknowledged' })
  if (entry.status === 'Acknowledged') actions.push({ label: 'Mark under investigation', status: 'Under investigation' })
  if (entry.status === 'Under investigation') actions.push({ label: 'Mark reviewed', status: 'Reviewed' })
  if (entry.status === 'Reviewed') actions.push({ label: 'Reopen incident', status: 'Under investigation' })
  return <aside className="drawer panel" aria-labelledby="case-heading">
    <div className="drawer-head">
      <div><p className="eyebrow">{incident.id}</p><h2 tabIndex={-1} id="case-heading">{incident.title}</h2></div>
      <button className="icon-button" aria-label="Close investigation" onClick={close}><X size={19}/></button>
    </div>
    <div className="drawer-body">
      <Badge tone="amber">Illustrative case — not live model inference</Badge>
      <div className="flex flex-wrap gap-2 mt-3"><Badge tone={severityTone[incident.severity]}>{incident.severity}</Badge><Badge tone={statusTone[entry.status]}>{entry.status}</Badge></div>
      <div className="segmented" aria-label="Investigation views">
        {(Object.keys(labels) as CaseTab[]).map(key => <button key={key} aria-pressed={tab === key} onClick={() => chooseTab(key)}>{labels[key]}</button>)}
      </div>
      {tab === 'attributes' && <>
        <div className="address-pair"><div><span>Source</span><code>{incident.source}</code></div><ArrowRight size={17} aria-hidden="true"/><div><span>Destination</span><code>{incident.destination}</code></div></div>
        <dl className="attributes">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
        <p className="muted">Demonstration label <strong>{incident.family}</strong></p>
      </>}
      {tab === 'evidence' && <>
        <h3>Scenario evidence</h3><p>{incident.evidence}</p>
        <div className="callout"><CircleDot size={19} aria-hidden="true"/><div><strong>Context to verify</strong><p>{incident.context}</p></div></div>
        <p className="muted">Scripted scenario observations. No model scores or attributions were used.</p>
      </>}
      {tab === 'workflow' && <>
        <p className="muted" role="status">Current status: {entry.status}</p>
        <div className="case-actions">{actions.map(action => <button key={action.label} className="button primary compact" onClick={() => update(action.status)}>{action.label}</button>)}</div>
        <form className="note-form" onSubmit={event => {
          event.preventDefault()
          if (note.trim()) {
            addNote(note.trim())
            setNote('')
          }
        }}>
          <label htmlFor="analyst-note">Analyst note</label>
          <textarea id="analyst-note" rows={3} maxLength={500} value={note} onChange={event => setNote(event.target.value)} placeholder="Record what you checked and your assessment."/>
          <div className="flex items-center justify-between gap-2"><span className="muted">{note.length}/500</span><button className="button secondary compact" disabled={!note.trim()}>Add note</button></div>
        </form>
        <h3>Investigation timeline</h3>
        <ol className="timeline">{[...entry.events].reverse().map((item, index) => <li key={index}><Clock3 size={14} aria-hidden="true"/><div><strong>{item.label}</strong><span>{item.time}</span>{item.note && <p>{item.note}</p>}</div></li>)}</ol>
      </>}
    </div>
    <div className="drawer-footer"><button className="button secondary compact" onClick={() => exportCase(incident, entry)}><Download size={16} aria-hidden="true"/> Export investigation</button><span>Session only</span></div>
  </aside>
}


export default function Dashboard({ cases, active, tab, select, chooseTab, close, update, addNote }: {
  cases: Record<string, Case>; active: Incident | undefined; tab: CaseTab
  select: (incident: Incident) => void; chooseTab: (tab: CaseTab) => void; close: () => void
  update: (id: string, status: Status) => void; addNote: (id: string, note: string) => void
}) {
  const [query, setQuery] = useState('')
  const [severity, setSeverity] = useState('All severities')
  const [status, setStatus] = useState('All statuses')
  const caseFor = (incident: Incident) => cases[incident.id] ?? initialCase(incident)
  const visible = incidents.filter(incident => (
    (severity === 'All severities' || severity === incident.severity) &&
    (status === 'All statuses' || status === caseFor(incident).status) &&
    [incident.id, incident.title, incident.source, incident.destination, incident.family].some(value => value.toLowerCase().includes(query.trim().toLowerCase()))
  ))
  const counts = statuses.map(state => [state, incidents.filter(incident => caseFor(incident).status === state).length] as const)
  const families = [...new Set(incidents.map(incident => incident.family))].map(family => ({ family, count: incidents.filter(incident => incident.family === family).length })).sort((a, b) => b.count - a.count)
  return <>
    <PageHeader tag="Operations" title="SOC Dashboard" description="Investigate scripted flow alerts and record analyst decisions for this session.">
      <Badge tone="amber"><Activity size={15} aria-hidden="true"/> Synthetic demonstration data</Badge>
    </PageHeader>
    <div className="status-grid">{counts.map(([state, count]) => <div className="panel counter" key={state}><Badge tone={statusTone[state]}>{state}</Badge><strong>{count}</strong></div>)}</div>
    <div className={'workspace ' + (active ? 'with-drawer' : '')}>
      <div className="workspace-main">
        <Panel title="Incident queue" description="Predefined cases using documentation-range addresses." accessory={<ShieldAlert size={20} className="muted" aria-hidden="true"/>}>
          <div className="filters">
            <label className="search-box"><Search size={17} aria-hidden="true"/><span className="sr-only">Search incidents</span><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search alerts, flow ID or address"/></label>
            <label className="filter-select"><span className="sr-only">Filter by severity</span><select value={severity} onChange={event => setSeverity(event.target.value)}>{['All severities', 'Critical', 'High', 'Medium', 'Low'].map(value => <option key={value}>{value}</option>)}</select></label>
            <label className="filter-select"><ListFilter size={16} aria-hidden="true"/><span className="sr-only">Filter by status</span><select value={status} onChange={event => setStatus(event.target.value)}>{['All statuses', ...statuses].map(value => <option key={value}>{value}</option>)}</select></label>
          </div>
          <p className="sr-only" role="status">{visible.length} incidents shown</p>
          {visible.length === 0 && <p className="empty">No cases match these filters.</p>}
          <ul className="incident-list">{visible.map(incident => <li key={incident.id}>
            <button className="incident-row" aria-pressed={active?.id === incident.id} onClick={() => select(incident)}>
              <span className={'severity-line tone-' + severityTone[incident.severity]} aria-hidden="true"/>
              <div className="incident-name"><strong>{incident.title}</strong><span>{incident.id} · {incident.time}</span></div>
              <Badge tone={severityTone[incident.severity]}>{incident.severity}</Badge>
              <span className="incident-status">{caseFor(incident).status === 'Reviewed' && <Check size={15} aria-hidden="true"/>}{caseFor(incident).status}</span>
              <ArrowRight size={16} className="incident-arrow" aria-hidden="true"/>
            </button>
          </li>)}</ul>
        </Panel>
        <div className="dashboard-charts">
          <Panel title="Traffic pattern" description="Scripted intervals · synthetic flow counts">
            <div className="chart-frame" aria-hidden="true">
              <ResponsiveContainer><AreaChart accessibilityLayer={false} data={trend} margin={{ right: 8, top: 10, left: -22, bottom: 0 }}>
                <CartesianGrid stroke="#253248" vertical={false}/><XAxis dataKey="time" tick={{ fill: '#9caec7', fontSize: 11 }} tickLine={false}/><YAxis allowDecimals={false} tick={{ fill: '#9caec7', fontSize: 11 }} tickLine={false}/>
                <Tooltip contentStyle={{ background: '#111b2b', borderColor: '#344660' }} labelStyle={{ color: '#e7f0ff' }} itemStyle={{ color: '#e7f0ff' }}/>
                <Area name="Scripted flows" dataKey="flows" stroke="#40d9ef" fill="#17344a" strokeWidth={2} isAnimationActive={false}/>
                <Area name="Flagged examples" dataKey="flagged" stroke="#bca5fc" fill="transparent" strokeWidth={2} isAnimationActive={false}/>
              </AreaChart></ResponsiveContainer>
            </div>
            <details className="chart-data"><summary>Traffic data table</summary><table><thead><tr><th>Time</th><th>Flows</th><th>Flagged</th></tr></thead><tbody>{trend.map(row => <tr key={row.time}><th scope="row">{row.time}</th><td>{row.flows}</td><td>{row.flagged}</td></tr>)}</tbody></table></details>
          </Panel>
          <Panel title="Attack categories" description="Labels assigned to the synthetic queue">
            <ul className="bar-list">{families.map(({ family, count }) => <li key={family}><div><span>{family}</span><strong>{count}</strong></div><progress value={count} max={incidents.length} aria-label={family}/></li>)}</ul>
          </Panel>
        </div>
      </div>
      {active && <Drawer key={active.id} incident={active} entry={caseFor(active)} tab={tab} chooseTab={chooseTab} close={close} update={next => update(active.id, next)} addNote={note => addNote(active.id, note)}/>}
    </div>
  </>
}
