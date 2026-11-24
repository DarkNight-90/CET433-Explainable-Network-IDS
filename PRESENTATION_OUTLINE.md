# CET433 Project 2 — 10-minute presentation outline

1. **Title and team (0:30)** — project title, members, course.
2. **Problem and customer (1:00)** — SOC/MSSP alert fatigue, missed attacks, business need.
3. **Dataset (1:00)** — UNSW-NB15 source, licence, shape, labels, class imbalance.
4. **Pipeline (1:00)** — cleaning, split-before-fit, leakage prevention, preprocessing pipeline.
5. **Models (1:00)** — baseline and three tuned models; cross-validation on training data only.
6. **Results (1:30)** — comparison table, confusion matrix, precision/recall/F1/FPR, per-class recall, latency.
7. **Threshold and business value (1:00)** — selected threshold, cost assumptions, estimated analyst hours saved.
8. **Explainability (1:00)** — SHAP global view, one false negative and one false positive.
9. **Robustness and limitations (0:45)** — official held-out test results; disclose that no true temporal split is possible from supplied partition files without real timestamps.
10. **Live demo and conclusion (1:15)** — upload flow CSV, show alert and explanation, summarize measured result and future work.

## Defense preparation
Each member should be ready to explain their own code, design decisions, metrics, data-cleaning choices, leakage checks, limitations, and contribution history. Replace this outline with actual measured figures and screenshots after execution.
