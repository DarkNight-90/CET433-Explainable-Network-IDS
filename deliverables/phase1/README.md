# CET433 Phase 1 discussion bundle

**November 26, 2026** · Network Intrusion Detection with Explainable AI for SOC Analysts

Start with the [rubric and submission manifest](submission-manifest.md), [Phase 1 report draft](../../REPORT_TEMPLATE.md) and [eight-slide speaker outline](../../PRESENTATION_OUTLINE.md). The report and outline use the project's existing document locations. The [November 26 discussion deck](phase1-discussion.pptx) accompanies the evidence. The starter deck and experimental notebooks are preserved.

| Evidence | Location |
| --- | --- |
| Training-partition EDA and charts | [EDA report](eda.md) |
| Preprocessing and leakage prevention | [Pipeline and frozen split](preprocessing.md) |
| Binary and multiclass task definitions | [Tasks](tasks.md) |
| Baseline and confusion matrix | [Stage A evaluation](baseline.md) |
| Business case and objectives | [SOC use case](business-case.md) |
| Dataset terms, citations and ethics | [UNSW-NB15 references](dataset-ethics.md) |

## Read-only discussion demo

1. Open [labels.pdf](figures/labels.pdf), [preprocessing.pdf](figures/preprocessing.pdf) and [confusion-matrix.pdf](figures/confusion-matrix.pdf) in advance. The baseline figure annotates counts and percentages by actual class.
2. From the repository root, run the following command. Use Python 3.11+ with NumPy, pandas and scikit-learn already available. Verification was run locally on Python 3.14.10, NumPy 2.4.6, pandas 2.3.3 and scikit-learn 1.9.0.

   ```sh
   python -B deliverables/phase1/generate_evidence.py
   ```

3. Point out the verified training shape (175,341 × 45), the 140,273/35,068 row split, and all six zero-overlap checks. Compare the script's metrics and cost with the confusion matrix. Explain that AUC was reported from the supplied aggregate.
4. For an offline demo without Python or the CSV, open [split-verification.json](split-verification.json), [baseline.json](baseline.json) and the same PDFs. The saved split JSON includes the original assignment digests and frozen supports.

The command reads the local `data/official/UNSW_NB15_training-set.csv`, dataset and split manifests, and `artifacts/generated/m02.2-split-assignments.json`. Use the Phase 1 command above. The experimental notebook's general preflight can read both published partitions. The Phase 1 demo has no need for the official test CSV, estimator artifacts, notebook execution, or a dashboard.

## Reproduce the evidence files

Use the same checkout, with the checksum-matching training CSV and frozen assignments in their existing paths. With Matplotlib available, generate an independent set of aggregate tables and figures into a fresh output directory. This example uses temporary storage.

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cet433-phase1-mpl python -B deliverables/phase1/generate_evidence.py --output /tmp/cet433-phase1-evidence
```

The generator emits the profile, per-column missingness, frequencies, numeric statistics, frozen split audit and four figure pairs. The architecture diagram source is [preprocessing.dot](preprocessing.dot) and the checked-in PNG and PDF were rendered with Graphviz. A literal `-` in the distribution table is a dataset category.
