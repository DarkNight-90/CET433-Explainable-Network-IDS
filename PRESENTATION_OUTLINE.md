# CET433 Phase 1 discussion
## Eight slides · Ten minutes · November 26, 2026

| Slide | Time | Show | Speaker notes |
| --- | --- | --- | --- |
| 1. Network intrusion detection for SOC analysts | 0:40 | Course, project and team | Introduce the enterprise SOC or MSSP and explain that Phase 1 establishes a benchmark and safe evaluation protocol. |
| 2. Alert triage and error costs | 1:10 | Recall ≥95%, FPR <1%, cost = 5 × FP + 120 × FN | Explain alert workload, the consequences of missed attacks and how feature-level explanations could support human review. The weights are assumptions about error burden. |
| 3. UNSW-NB15 and class imbalance | 1:20 | [Label distributions](deliverables/phase1/figures/labels.pdf) | Official training CSV has 175,341 rows and 45 columns. Explain the Attack majority at 68.06%, the rich and sparse families, and why production prevalence may differ. Mention UNSW's academic research terms. |
| 4. Frozen split and leakage | 1:20 | [Boundary table](deliverables/phase1/preprocessing.md#frozen-m022-membership) | 74,301 extra feature-vector occurrences make row-level leakage plausible. Cover 140,273 fit and 35,068 validation rows, the five frozen folds and all six zero-overlap checks. Exact-feature groups do not establish chronological separation. |
| 5. Pipeline and tasks | 1:20 | [Pipeline diagram](deliverables/phase1/figures/preprocessing.pdf) | Identify the three excluded columns and explain median imputation, zero clipping, log1p, numeric scaling and categorical pooling. Describe fold-local fitting, balanced Logistic Regression and the prospective ten-class task. |
| 6. Binary baseline | 1:30 | [Confusion matrix](deliverables/phase1/figures/confusion-matrix.pdf) and [metrics](deliverables/phase1/baseline.md) | For M02.2 internal validation at 0.5, explain 94.10% recall, 7.90% FPR and 0.951411 F1. Point out 885 false positives and 1,409 false negatives. Show how these produce 173,505 analyst-equivalent minutes. Both operating targets are missed. |
| 7. Evidence demo | 1:30 | [Read-only instructions](deliverables/phase1/README.md#read-only-discussion-demo) | Show the verification output and the local PNG or PDF confusion matrix. Trace the row sums to frozen validation supports. AUC is supplied and requires ranked scores for independent verification. |
| 8. Conclusions and limitations | 1:10 | Verified Phase 1 findings and next evaluation milestones | State what exact grouping guarantees. Explain remaining multiclass work, chronology, representativeness and the need to validate explanations and robustness. Keep production-readiness and analyst-savings claims out of the conclusion. |

Open the figures and run the read-only verifier before the session. Use the generated split summary and figures as the offline demo if the original training file or Python environment is unavailable. Bring the original Stage A metrics export if available for confirming provenance.
