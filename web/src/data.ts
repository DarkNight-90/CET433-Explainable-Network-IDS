export const number = (value: number) => value.toLocaleString('en-US')
export const percent = (value: number, digits = 2) => (value * 100).toFixed(digits) + '%'

export const repository = 'https://github.com/DarkNight-90/CET433-Explainable-Network-IDS'
export const phase1 = repository + '/tree/main/deliverables/phase1'
export const provider = 'https://research.unsw.edu.au/projects/unsw-nb15-dataset'

export const team = [
  { name: 'Mohamed Hesham', id: '230105474', initials: 'MH' },
  { name: 'Radwa Tarek', id: '230103871', initials: 'RT' },
  { name: 'Roaa Tarek', id: '230103862', initials: 'RT' },
]

export type Matrix = { tp: number; fp: number; tn: number; fn: number }
export type Result = {
  id: string
  stage: 'A' | 'B'
  model: string
  precision?: number
  recall: number
  f1: number
  fpr: number
  auc?: number
  matrix?: Matrix
}

export const results: Result[] = [
  { id: 'a-lr', stage: 'A', model: 'Logistic Regression', precision: .962089, recall: .940967, f1: .951411, fpr: .079018, auc: .986691,
    matrix: { tp: 22459, fp: 885, tn: 10315, fn: 1409 } },
  { id: 'a-rf', stage: 'A', model: 'Random Forest', recall: .946665, f1: .961305, fpr: .048750 },
  { id: 'a-lgbm', stage: 'A', model: 'LightGBM', recall: .944151, f1: .961452, fpr: .042321 },
  { id: 'b-rf', stage: 'B', model: 'Random Forest', recall: .971300, f1: .968379, fpr: .074018, auc: .992963,
    matrix: { tp: 23183, fp: 829, tn: 10371, fn: 685 } },
  { id: 'b-lgbm', stage: 'B', model: 'LightGBM', recall: .955003, f1: .966011, fpr: .047321, auc: .993947,
    matrix: { tp: 22794, fp: 530, tn: 10670, fn: 1074 } },
]
export const cost = ({ fp, fn }: Matrix) => 5 * fp + 120 * fn
export const precision = (result: Result) => {
  if (result.precision !== undefined) return result.precision
  const { matrix } = result
  return matrix ? matrix.tp / (matrix.tp + matrix.fp) : undefined
}

export const statuses = ['New', 'Acknowledged', 'Under investigation', 'Reviewed'] as const
export type Status = typeof statuses[number]
export type Severity = 'Critical' | 'High' | 'Medium' | 'Low'
export type Incident = {
  id: string; title: string; severity: Severity; family: string; time: string
  source: string; destination: string; protocol: string; service: string; connection: string
  duration: string; packets: number; sent: string; received: string
  evidence: string; context: string
}
export const incidents: Incident[] = [
  { id: 'FLOW-1042', title: 'Unusual outbound transfer', severity: 'Critical', family: 'Exploits', time: '09:42',
    source: '192.0.2.24', destination: '203.0.113.18', protocol: 'TCP', service: 'HTTPS', connection: 'FIN',
    duration: '18.4 s', packets: 1840, sent: '12.8 MB', received: '84 KB',
    evidence: 'This scripted flow sends significantly more data than it receives. The scenario places that asymmetry outside its fictional workstation profile.',
    context: 'A scheduled backup could explain the same pattern. Compare the timing with approved transfer activity before drawing a conclusion.' },
  { id: 'FLOW-1041', title: 'Broad service discovery', severity: 'High', family: 'Reconnaissance', time: '09:37',
    source: '198.51.100.32', destination: '192.0.2.81', protocol: 'TCP', service: 'Unknown', connection: 'INT',
    duration: '1.4 s', packets: 82, sent: '4.6 KB', received: '0 B',
    evidence: 'The scenario groups brief, one-way connection attempts that were never established.',
    context: 'An authorized inventory process can produce similar traffic. Confirm whether an inventory window was approved.' },
  { id: 'FLOW-1040', title: 'Request volume burst', severity: 'High', family: 'DoS', time: '09:21',
    source: '203.0.113.52', destination: '198.51.100.41', protocol: 'UDP', service: 'DNS', connection: 'CON',
    duration: '4.0 s', packets: 2404, sent: '188 KB', received: '35 KB',
    evidence: 'The synthetic case marks a brief concentration of requests for review.',
    context: 'A traffic replay could also look like a burst. Verify expected traffic and service health in the fictional scenario.' },
  { id: 'FLOW-1039', title: 'Unusual service payload size', severity: 'Medium', family: 'Fuzzers', time: '09:12',
    source: '192.0.2.45', destination: '198.51.100.72', protocol: 'TCP', service: 'HTTP', connection: 'FIN',
    duration: '2.1 s', packets: 119, sent: '94 KB', received: '21 KB',
    evidence: 'The example assigns an atypical payload size to an otherwise ordinary service.',
    context: 'Large requests can be legitimate. Check the service role and context before escalating.' },
  { id: 'FLOW-1038', title: 'Repeated resolver requests', severity: 'Medium', family: 'Generic', time: '09:04',
    source: '198.51.100.19', destination: '192.0.2.53', protocol: 'UDP', service: 'DNS', connection: 'CON',
    duration: '7.2 s', packets: 297, sent: '31 KB', received: '29 KB',
    evidence: 'The scenario presents repeated resolver activity as an anomaly cue.',
    context: 'A retry loop or stale cache may explain the repetition. Consider known resolver issues.' },
  { id: 'FLOW-1037', title: 'Short inbound session', severity: 'Low', family: 'Reconnaissance', time: '08:48',
    source: '203.0.113.70', destination: '192.0.2.17', protocol: 'TCP', service: 'HTTPS', connection: 'FIN',
    duration: '0.7 s', packets: 12, sent: '5.8 KB', received: '8.2 KB',
    evidence: 'This example is a completed short session marked for a low-priority check.',
    context: 'A health probe is a plausible explanation. Look for a matching maintenance record.' },
]

export type Event = { time: string; label: string; note?: string }
export type Case = { status: Status; events: Event[] }
export function initialCase(incident: Incident): Case {
  return { status: 'New', events: [{ time: incident.time + ' · scenario', label: 'Scripted flow added to queue' }] }
}

export const trend = [
  { time: '08:00', flows: 43, flagged: 9 },
  { time: '08:20', flows: 61, flagged: 15 },
  { time: '08:40', flows: 52, flagged: 11 },
  { time: '09:00', flows: 85, flagged: 24 },
  { time: '09:20', flows: 74, flagged: 18 },
  { time: '09:40', flows: 106, flagged: 31 },
  { time: '10:00', flows: 92, flagged: 21 },
]

export const references = [
  ['UNSW-NB15: A comprehensive data set for network intrusion detection systems', 'https://ieeexplore.ieee.org/document/7348942'],
  ['The evaluation of Network Anomaly Detection Systems', 'https://doi.org/10.1080/19393555.2015.1125974'],
  ['Novel geometric area analysis technique for anomaly detection', 'https://ieeexplore.ieee.org/document/7948715'],
  ['Big data analytics for intrusion detection system', 'https://doi.org/10.1007/978-3-319-59439-2_5'],
  ['NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems', 'https://arxiv.org/abs/2011.09144'],
]
