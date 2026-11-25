# CET433 SOC Lab

A standalone React, TypeScript and Vite demonstration. Tailwind utilities and CSS provide the shell and responsive layouts. Lucide supplies icons, Recharts renders the scripted traffic chart, and locally hosted variable fonts support offline presentation after loading the page assets. Fragment routes support direct links and browser history without a server rewrite. The chart workspace is loaded on demand.

## Run locally

Use Node 24 and npm. From the repository root:

```sh
cd web
npm ci
npm run dev
```

Open the printed local address. Select **Launch SOC Experience**, open the scripted outbound transfer, review its attributes and evidence, acknowledge it, start an investigation, add a note and mark the case reviewed. The walkthrough then opens the research lab and illustrative explainability examples. Export writes a JSON file to the browser's download location. Notes and decisions remain in this tab's session state and reset on reload.

## Evidence and privacy

- [Phase 1 evidence](../deliverables/phase1/README.md) and [project metrics](../README.md) are the sources for the published aggregates. The app recomputes costs and labels the AUC and optimization provenance. It contains no score vectors and displays no ROC curves.
- The two images under `public/evidence/` are byte-identical copies of the original Phase 1 training-label chart and preprocessing diagram. They are copied into the standalone Vercel build so it can run with `web` as its only root.
- Incidents and traffic trends are scripted. Their endpoint addresses come from the three RFC 5737 documentation blocks. Feature attribution values are hand-authored examples. There is no live inference, traffic capture, telemetry, account system, or backend.
- `vercel.json` configures response headers. Its policy restricts scripts, fonts, and images to the deployed origin, disables outbound script connections and embedding, and allows inline styles for chart sizing and bars. The demo does not use cookies or browser storage. The locally hosted [Inter](public/licenses/inter.txt) and [Manrope](public/licenses/manrope.txt) font licenses accompany the build.

## Checks

```sh
npm run typecheck
npm run lint
npm run build
npm audit --audit-level=low
npm run test:browser
```

The browser test serves `dist` with the Vercel response headers. It uses Chrome at `/usr/bin/google-chrome`; set `CHROME_PATH` for a different installation. It checks 1440, 820, 390, and 320 px layouts and routes, plus a full desktop and mobile guided tour, investigation controls, JSON export, reduced motion, axe checks and console errors. To test a reachable preview using the same workflow, set `SOC_DEMO_URL` to the preview origin before running the browser command.

## Preview deployment

Use a **new** Vercel project for this GitHub repository. For Git integration, use root directory `web`, framework `Vite`, build command `npm run build` and output directory `dist`. The website branch is the preview source. Skip any setup path that would automatically publish it to production. No environment variables are needed.

For an authenticated CLI preview without Git auto-deployment, use a clean copy of the committed `web/` directory as the project root so local datasets and research artifacts are outside the upload context. Confirm the project is new and target preview explicitly. Review the build logs, check the deployed headers and run the browser suite against the preview origin. Production requires a separate approval after preview review.
