import { strict as assert } from 'node:assert'
import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { dirname, extname, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'
import axe from 'axe-core'
import ts from 'typescript'

const project = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const root = resolve(project, 'dist')
const source = await readFile(resolve(project, 'src/data.ts'), 'utf8')
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } })
const { cost, incidents, precision, results } = await import('data:text/javascript,' + encodeURIComponent(outputText))
const { headers: [rule] } = JSON.parse(await readFile(resolve(project, 'vercel.json'), 'utf8'))
const contentTypes = { '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html; charset=utf-8',
  '.woff2': 'font/woff2', '.png': 'image/png' }

const expected = {
  'a-lr': [.962089, .940967, .951411, .079018, 173505],
  'b-rf': [.965476, .971300, .968379, .074018, 86345],
  'b-lgbm': [.977277, .955003, .966011, .047321, 131530],
}
for (const result of results.filter(item => item.matrix)) {
  const { tp, fp, tn, fn } = result.matrix
  assert.equal(tp + fp + tn + fn, 35068)
  assert.equal(tn + fp, 11200)
  const derived = [precision(result), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn), fp / (fp + tn)]
  const rounded = derived.map(value => Number(value.toFixed(6)))
  assert.deepEqual([...rounded, cost(result.matrix)], expected[result.id])
  assert.deepEqual([result.recall, result.f1, result.fpr], rounded.slice(1))
}
for (const incident of incidents) {
  for (const ip of [incident.source, incident.destination]) {
    assert.match(ip, /^(?:192\.0\.2|198\.51\.100|203\.0\.113)\.\d{1,3}$/)
  }
}

let server
let browser
try {
  let base = process.env.SOC_DEMO_URL
  if (!base) {
    server = createServer(async (request, response) => {
      try {
        const pathname = new URL(request.url, 'http://localhost').pathname
        const path = resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname))
        if (!path.startsWith(root + sep)) throw new Error('outside build')
        const body = await readFile(path)
        for (const { key, value } of rule.headers) response.setHeader(key, value)
        response.writeHead(200, { 'Content-Type': contentTypes[extname(path)] ?? 'application/octet-stream' })
        response.end(body)
      } catch {
        response.writeHead(404).end()
      }
    })
    await new Promise((done, reject) => server.listen(0, '127.0.0.1', done).once('error', reject))
    base = 'http://127.0.0.1:' + server.address().port
  }

  browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
    headless: true,
    args: ['--disable-background-networking'],
  })
  const browserErrors = []
  for (const width of [1440, 820, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, reducedMotion: 'reduce' })
    page.on('pageerror', error => browserErrors.push(String(error)))
    page.on('console', message => {
      if (message.type() === 'error') browserErrors.push(message.text())
    })
    page.on('requestfailed', request => {
      if (request.failure()?.errorText !== 'net::ERR_ABORTED') browserErrors.push(request.url() + ' ' + request.failure()?.errorText)
    })
    await page.goto(base)
    await page.getByRole('heading', { level: 1, name: 'Network defense, explained.' }).waitFor()
    assert.equal(await page.evaluate(() => globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches), true)
    const checkWidth = async () => assert.equal(await page.evaluate(() => globalThis.document.documentElement.scrollWidth > globalThis.innerWidth), false, 'page overflow ' + page.url() + ' @' + width)
    const checkA11y = async () => {
      await page.evaluate(axe.source)
      const issues = await page.evaluate(async () => (await globalThis.axe.run({ runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa'] } })).violations)
      assert.deepEqual(issues.map(issue => [issue.id, issue.nodes.map(node => node.target)]), [], 'accessibility ' + page.url() + ' @' + width)
    }
    await checkWidth()
    if (process.env.SOC_SCREENSHOTS && (width === 1440 || width === 390)) {
      await page.screenshot({ path: resolve(process.env.SOC_SCREENSHOTS, 'cet433-overview-' + width + '.png'), fullPage: true })
    }

    if (width === 1440 || width === 390) {
      await checkA11y()
      await page.getByRole('button', { name: 'Launch SOC Experience' }).focus()
      await page.keyboard.press('Enter')
      const guide = page.getByRole('region', { name: 'Guided demonstration' })
      await guide.getByRole('heading', { name: 'Observe a suspicious flow' }).waitFor()
      await page.getByRole('button', { name: /Unusual outbound transfer/ }).click()
      await page.getByText('Illustrative case — not live model inference', { exact: true }).waitFor()
      assert.equal(await page.getByText('192.0.2.24', { exact: true }).count(), 1)
      await guide.getByRole('button', { name: 'Back', exact: true }).click()
      await guide.getByRole('button', { name: 'Next', exact: true }).click()
      await guide.getByRole('button', { name: 'Next', exact: true }).click()
      await page.getByRole('heading', { name: 'Scenario evidence' }).waitFor()
      await guide.getByRole('button', { name: 'Next', exact: true }).click()
      assert.equal(await guide.getByRole('button', { name: 'Next', exact: true }).isDisabled(), true)
      await page.getByRole('button', { name: 'Acknowledge alert' }).click()
      await page.getByRole('button', { name: 'Mark under investigation' }).click()
      await guide.getByRole('button', { name: 'Next', exact: true }).click()
      await page.getByRole('textbox', { name: 'Analyst note' }).fill('Checked planned backup window. Review recorded locally.')
      await page.getByRole('button', { name: 'Add note', exact: true }).click()
      await page.getByRole('button', { name: 'Mark reviewed' }).click()
      await page.getByText('Checked planned backup window. Review recorded locally.', { exact: true }).waitFor()
      const downloadPromise = page.waitForEvent('download')
      await page.getByRole('button', { name: 'Export investigation' }).click()
      const download = await downloadPromise
      assert.equal(download.suggestedFilename(), 'FLOW-1042-investigation.json')
      const exported = JSON.parse(await readFile(await download.path(), 'utf8'))
      assert.equal(exported.investigation.status, 'Reviewed')
      assert.match(exported.investigation.events.at(-2).note, /backup window/)
      await checkWidth()
      await checkA11y()
      await guide.getByRole('button', { name: 'Next', exact: true }).click()
      await page.getByRole('heading', { level: 1, name: 'ML Research Lab' }).waitFor()
      await page.getByRole('button', { name: 'Stage B', exact: true }).click()
      await page.getByRole('combobox', { name: 'Metric' }).selectOption('recall')
      await page.getByRole('button', { name: /Random Forest/ }).click()
      await page.getByText('23,183', { exact: true }).waitFor()
      await page.getByText('86,345 min', { exact: true }).waitFor()
      await checkWidth()
      await checkA11y()
      await guide.getByRole('button', { name: 'Next', exact: true }).click()
      await page.getByRole('button', { name: 'False negative', exact: true }).click()
      await page.getByRole('heading', { name: 'False-negative review' }).waitFor()
      await page.getByRole('button', { name: 'Global summary' }).click()
      await page.getByRole('heading', { name: 'Reading a global summary' }).waitFor()
      await guide.getByRole('button', { name: 'Finish demo' }).click()
      await guide.getByRole('heading', { name: 'You completed the SOC experience' }).waitFor()
      await checkA11y()
      await guide.getByRole('button', { name: 'Replay', exact: true }).click()
      await guide.getByRole('button', { name: 'Exit demo', exact: true }).click()
      assert.equal(await guide.count(), 0)
      await page.getByRole('textbox', { name: 'Search incidents' }).fill('1042')
      await page.getByRole('combobox', { name: 'Filter by severity' }).selectOption('Low')
      await page.getByText('No cases match these filters.').waitFor()
      await page.getByRole('combobox', { name: 'Filter by severity' }).selectOption('Critical')
      await page.getByRole('combobox', { name: 'Filter by status' }).selectOption('Reviewed')
      await page.getByRole('button', { name: /Unusual outbound transfer/ }).click()
      await page.getByRole('button', { name: 'Workflow', exact: true }).click()
      await page.getByRole('button', { name: 'Reopen incident' }).click()
      await page.getByText('Case reopened', { exact: true }).waitFor()
      await page.keyboard.press('Escape')
      assert.equal(await page.getByRole('complementary', { name: 'Unusual outbound transfer' }).count(), 0)
    }

    for (const [id, heading] of [
      ['soc', 'SOC Dashboard'], ['research', 'ML Research Lab'], ['explainability', 'Explainability Center'],
      ['methodology', 'Dataset & Methodology'], ['roadmap', 'Project Roadmap'], ['overview', 'Network defense, explained.'],
    ]) {
      await page.goto(base + '/#' + id)
      await page.getByRole('heading', { level: 1, name: heading }).waitFor()
      await checkWidth()
      if (width === 1440 || width === 390) await checkA11y()
    }
    if (width <= 900) {
      await page.getByRole('button', { name: 'Open navigation' }).click()
      await page.getByRole('navigation', { name: 'Workspace' }).getByRole('link', { name: 'Project Roadmap' }).click()
      await page.getByRole('heading', { level: 1, name: 'Project Roadmap' }).waitFor()
      await page.getByRole('button', { name: 'Open navigation' }).waitFor()
      await page.goBack()
      await page.getByRole('heading', { level: 1, name: 'Network defense, explained.' }).waitFor()
    }
    await page.close()
    console.log('Chromium routing and layout passed at ' + width + ' px')
  }
  assert.deepEqual(browserErrors, [])
  console.log('Metric checks, guided workflow, export, accessibility, and console checks passed.')
} finally {
  if (browser) await browser.close()
  if (server) server.close()
}
