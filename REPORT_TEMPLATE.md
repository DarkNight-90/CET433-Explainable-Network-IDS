# CET433 Project 2 Report — Explainable Network IDS for SOC Analysts

> Complete this report only after running the notebook. Replace every `[INSERT ...]` field with measured results. Do not invent metrics.

## 1. Abstract
[INSERT 150–200 word summary: problem, dataset, methods, best measured result, limitations.]

## 2. Problem and Market Case
**Customer:** Company SOC / MSSP.  
**Problem:** High alert volume and missed attacks make network monitoring expensive.  
**Proposed value:** Classify network flows and provide feature-level explanations to help analysts prioritize alerts.  
**Success criteria:** Recall ≥ 0.95 and FPR < 0.01 (report honestly whether achieved).  
**Business assumptions:** False-positive handling = 5 analyst minutes; missed-attack handling = 120 analyst-equivalent minutes. Justify or replace these assumptions.

## 3. Dataset and Licence
Dataset: UNSW-NB15 official training/testing partition files.  
Source URL: [INSERT verified source URL].  
Licence/terms: [INSERT verified licence or terms].  
Rows/features/classes: [INSERT from notebook output].  
Privacy and safe-use statement: public dataset; no unauthorized network capture or attack activity.

## 4. Exploratory Data Analysis
Include class balance, attack-category counts, missing values, infinities, duplicates, and key observations. Insert notebook figures and describe their implications.

## 5. Cleaning and Leakage-Safe Preprocessing
Describe dropping `id`, replacing ±infinity with NaN, removing exact duplicates before splitting, keeping `label` and `attack_cat` as targets only, and fitting imputers/scalers/encoders inside each model pipeline on training folds only. State the fixed random seed and split proportions.

## 6. Models and Experimental Setup
Describe the binary and multiclass tasks. List baseline models and at least three tuned algorithms, their search spaces, cross-validation strategy, and training-only tuning. Explain why each model is appropriate.

## 7. Results and Evaluation
Insert the model comparison table with precision, recall, F1, FPR, ROC-AUC, latency, and CV F1. Include confusion matrices and per-class recall. Discuss the best model and measured gain over baseline.

## 8. Threshold and Business Metric
Report selected threshold, confusion matrix counts, cost assumptions, estimated analyst hours saved on validation data and scaled estimate per 100,000 flows. Explain uncertainty and assumptions.

## 9. Explainability
Insert global SHAP summary. Discuss one actual false negative and one actual false positive using their top SHAP features. If a class has no such error at the chosen threshold, state that and explain the fallback case honestly.

## 10. Robustness and Generalization
Report validation vs official held-out test results and per-class test recall. Discuss distribution shift. **Important:** UNSW-NB15 partition files used here do not provide a reliable timestamp column; this comparison is not a true temporal split. Add a genuine chronological experiment only after obtaining timestamped data.

## 11. Prototype and Deployment
Describe Streamlit dashboard workflow: CSV upload, feature validation, model scoring, alert table, downloadable results, and per-flow SHAP reasons. Add screenshots from a successful local run.

## 12. Limitations, Ethics, and Privacy
Discuss false positives/negatives, dataset age/representativeness, possible drift, model bias across traffic classes, human review, privacy minimization, and no automated punitive action based solely on model output.

## 13. Cost and Deployment Considerations
Discuss analyst time assumptions, compute requirements, model latency, monitoring/retraining, secure storage, access control, and operational costs.

## 14. Conclusion and Future Work
Summarize measured performance, whether success criteria were met, remaining temporal-split limitation, and next steps.

## References
[INSERT dataset citation, library documentation, SHAP reference, and any other sources actually used.]

## Appendix
- Hyperparameter search settings
- Additional confusion matrices
- Individual contribution table and Git commit evidence
