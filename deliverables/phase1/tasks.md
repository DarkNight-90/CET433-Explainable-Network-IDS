# Task definitions

Both tasks use the 42 flow-level model inputs described in [preprocessing](preprocessing.md). The M02.2 internal fit partition has 140,273 rows and the M02.2 internal validation partition has 35,068 rows, both from the official UNSW-NB15 training CSV.

| Task | Target and output | Phase 1 status |
| --- | --- | --- |
| Binary detection | Predict Normal (label 0) or Attack (label 1). Use the Attack probability for thresholding. | Stage A Logistic Regression baseline at 0.5. Internal validation precision 0.962089, recall 0.940967 and F1 0.951411. [Evaluation](baseline.md) |
| Multiclass classification | Predict one of Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode or Worms from `attack_cat`. | Task and protocol defined. Notebook prototype cells exist outside the controlled Stage A/B execution. A completed multiclass experiment and metrics have not been verified. |

A prospective multiclass experiment would use the frozen fit folds to choose settings, fit preprocessing only on each fold's fit side, refit the chosen configuration on all internal fit rows and score internal validation. Use the same ten labels in every output. Report unweighted macro-F1 as the average of the ten class F1 values, and each class's recall as correctly predicted class rows divided by its actual support. Report raw counts and a row-normalized 10 × 10 confusion matrix with actual labels on rows and predicted labels on columns. Include support and disclose undefined metrics and their averaging treatment. A single Worms error changes recall by 1/26 on internal validation. Binary attack recall pools all nine families and supplies no family-level measurement here.
