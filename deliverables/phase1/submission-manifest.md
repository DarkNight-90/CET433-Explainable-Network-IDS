# Phase 1 readiness and submission manifest

## Rubric checklist

| Criterion | Marks available | Evidence and review status |
| --- | ---: | --- |
| Problem and market case | 2 | **Complete for discussion.** Enterprise SOC/MSSP, alert triage problem, explainability objective, cost assumptions, operating targets and benchmark limitations. [Business case](business-case.md) |
| Data and preprocessing pipeline | 3 | **Complete for discussion.** Training-only EDA, scoped tables, academic terms and citations, pipeline architecture and implementation mapping, frozen membership and five CV boundary checks. [EDA](eda.md) · [Preprocessing](preprocessing.md) · [Terms](dataset-ethics.md) |
| Baseline model and metrics | 3 | **Complete for discussion from supplied aggregate.** Logistic Regression, confusion matrix, both class recalls, precision, F1, FPR and policy cost. **Manual source review** remains for the original Stage A export and independent score-based AUC verification. [Baseline](baseline.md) · [Tasks](tasks.md) |

The numbers are the rubric weights. The course team determines the awarded marks.

## Submission inventory

| Status | Artifacts and scope |
| --- | --- |
| Complete | [Phase 1 index and safe demo](README.md), [EDA report](eda.md), [profile](profile.tsv), [missingness](missing.tsv), [full distributions](distributions.tsv), [numeric descriptions](numeric-summary.tsv), [frozen split audit](split-verification.json), [PNG and PDF figures](figures/), [architecture source](preprocessing.dot), [generator](generate_evidence.py) |
| Complete | [Pipeline and leakage documentation](preprocessing.md), [task definitions](tasks.md), [baseline report](baseline.md) and [source transcription](baseline.json), [business case](business-case.md), [UNSW terms and references](dataset-ethics.md) |
| Ready for team review | [Report draft](../../REPORT_TEMPLATE.md) and [eight-slide outline](../../PRESENTATION_OUTLINE.md). Rehearse the ten-minute talk and confirm names and any course-required submission format. The [November 26 discussion deck](phase1-discussion.pptx) accompanies the outline. The original starter deck is preserved. |
| Missing locally | Completed Stage A/Stage B run manifests and metric exports. The local Stage A/B ZIPs contain source snapshots. If accessible, cross-check the supplied baseline against the original metrics export. AUC needs per-row validation scores to be independently verified. The baseline report contains no Stage B result or threshold-sweep claim. |
| Outside verified results | The historical proposal names KNN in Phase 1. This handover specifies Logistic Regression, and no verified KNN result was available. A historical numerical row-random overlap comparison was also unavailable. Multiclass metrics, attack-family recall, SHAP evidence, official test robustness and temporal generalization remain unevidenced here. |

## Verification record

- Revalidated the official training CSV's checksum, schema and support against the local manifest. Verified both frozen assignment digests and all six group boundaries without generating new memberships.
- Recomputed confusion-derived metrics and policy cost and matched both row supports to the frozen validation manifest. ROC-AUC remains source-reported. No claim of joint target attainment is supported by the supplied baseline.
- Ran the standalone demo and replayed aggregate evidence with runtime checks for official-test/model-artifact reads and estimator fitting. Checked figure labels, generated tables and report consistency. The official test partition stayed unopened and no trained artifact was loaded during this work.
- Compared pre- and post-work hashes of the training CSV, frozen assignments, source snapshots, original notebooks, stage code and other existing experiment inputs. Existing experimental files were preserved.
- Ten existing no-fit preflight/acceptance tests passed. The experimental notebooks and controlled runners were left untouched. No Git commits or pushes were performed.

**Discussion decision:** GO for a Phase 1 benchmark and methods defense. Present the missed targets, evidence provenance and remaining experiments as limitations. The evidence does not establish deployment readiness.
