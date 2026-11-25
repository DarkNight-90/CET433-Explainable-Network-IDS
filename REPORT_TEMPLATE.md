# CET433 Phase 1 report draft
## Network Intrusion Detection with Explainable AI for SOC Analysts

November 26, 2026 · Fall 2026 · Mohamed Hesham Abbas, Radwa Tarek and Roaa Tarek

### Abstract

This project studies flow-level intrusion detection for security operations teams facing high alert volume and limited review capacity. Phase 1 defines binary detection and ten-class attack-family identification using UNSW-NB15. The public training CSV contains 175,341 observations with 42 model inputs. Exact input groups were kept together in a frozen split and five internal CV folds. The supplied Stage A Logistic Regression baseline on 35,068 internal validation rows achieved attack precision 0.962089, recall 0.940967 and F1 0.951411 at threshold 0.5. The FPR of 0.079018 and recall miss the frozen targets. Repeated inputs, conflicting labels, limited rare-class support and the absence of chronological evidence restrict the conclusions. Feature-level explanations remain a project objective.

### Problem and market case

An enterprise SOC or MSSP needs to prioritize suspicious traffic without overwhelming analysts. A learned flow classifier can rank suspicious observations. A validated feature-level explanation could help analysts investigate and review the resulting alerts. The operating targets are recall ≥95% and FPR <1% jointly. The predefined cost policy assigns five analyst minutes to a false positive and 120 analyst-equivalent minutes to a false negative. This policy prioritizes missed-attack reduction subject to false-alarm constraints. [Business case and objectives](deliverables/phase1/business-case.md)

### Data and methodology

UNSW-NB15 combines normal activities and synthetic attacks captured in a cyber range. The official training CSV has 45 columns and 175,341 rows. It includes 56,000 Normal and 119,341 Attack rows. There were zero missing values and 74,301 extra exact feature-vector occurrences. We verified 140,273 M02.2 fit rows and 35,068 internal validation rows from frozen assignments. All five CV boundaries and the fit/validation boundary have zero shared exact input groups. Repeated groups and original labels are retained.

The pipeline excludes the row ID and both target columns. It applies fit-side median imputation, zero lower clipping, log1p and standardization to numeric features. It encodes protocol, service and state with fit-side dense one-hot encoding and rare-level pooling. The binary baseline is balanced Logistic Regression with seed 42. Every fit uses a single combined preprocessing and classification pipeline. [EDA](deliverables/phase1/eda.md) · [Methodology](deliverables/phase1/preprocessing.md)

### Task definitions and verified baseline evidence

Binary detection predicts Normal versus Attack. The proposed ten-class task predicts Normal or one of nine attack families, with macro-F1, per-class recall and a labeled confusion matrix specified for later evaluation. No completed multiclass experiment is evidenced here.

| M02.2 internal validation at threshold 0.5 | Value |
| --- | ---: |
| True positives / false positives | 22,459 / 885 |
| True negatives / false negatives | 10,315 / 1,409 |
| Attack precision / recall / F1 | 0.962089 / 0.940967 / 0.951411 |
| Normal recall / FPR | 0.920982 / 0.079018 |
| Reported ROC-AUC | 0.986691 |
| Estimated policy cost | 173,505 analyst-equivalent minutes |

The counts and confusion-derived metrics are consistent with the supplied Stage A aggregate and M02.2 supports. ROC-AUC is transcribed because the validation scores and completed run export were unavailable locally. Accuracy of 93.46% obscures the 885 false alarms and 1,409 missed attacks and the different costs of those errors. The baseline misses both targets and no joint attainment is established by the available Phase 1 evidence. [Task definitions](deliverables/phase1/tasks.md) · [Baseline calculations and confusion matrix](deliverables/phase1/baseline.md)

### Limitations and ethics

The benchmark prevalence and synthetic cyber-range behavior can differ from live enterprise networks. Exact grouping cannot exclude near-duplicates or shared sessions and does not establish temporal generalization. The public dataset is used for academic research under UNSW's terms, with no unauthorized traffic collection. This draft makes no production-readiness claim. Multiclass performance, attack-family recall, measured analyst savings, SHAP results and robustness results remain outside the verified Phase 1 evidence. [Terms, limitations and bibliography](deliverables/phase1/dataset-ethics.md)

### References

Dataset source and academic research terms. [UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

1. Moustafa, N., and Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). IEEE Military Communications and Information Systems Conference (MilCIS). [IEEE](https://ieeexplore.ieee.org/document/7348942)
2. Moustafa, N., and Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset. Information Security Journal: A Global Perspective. [Publisher](https://doi.org/10.1080/19393555.2015.1125974)
3. Moustafa, N., et al. (2017). Novel geometric area analysis technique for anomaly detection using trapezoidal area estimation on large-scale networks. IEEE Transactions on Big Data, early access. [IEEE](https://ieeexplore.ieee.org/document/7948715)
4. Moustafa, N., Creech, G., and Slay, J. (2017). Big data analytics for intrusion detection system: Statistical decision-making using finite Dirichlet mixture models. In Data Analytics and Decision Support for Cybersecurity, pp. 127–156. Springer. [Publisher](https://doi.org/10.1007/978-3-319-59439-2_5)
5. Sarhan, M., Layeghy, S., Moustafa, N., and Portmann, M. NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems. In Big Data Technologies and Applications, BDTA 2020 and WiCON 2020 proceedings, p. 117. Springer. [Paper](https://arxiv.org/abs/2011.09144)
