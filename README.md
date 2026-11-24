# CET433 Project 2 — Explainable Network IDS

Use `CET433_Project2_IDS_M01R.ipynb` for new runs. The original
`CET433_Project2_IDS_required.ipynb` is preserved as supplied and contains the
original self-writing bootstrap. The proposal, report template, and slide deck
are included as project material.

## Environment

The project targets **CPython 3.12**. `requirements.txt` pins the direct runtime
and notebook kernel dependencies. For a repeatable environment, resolve the
transitive dependencies once on the intended platform and retain the
hash-pinned lock with the run:

```sh
python3.12 -m venv .venv
uv pip compile --python-version 3.12 --generate-hashes requirements.txt -o requirements.lock
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m ipykernel install --user --name cet433 --display-name "CET433 (Python 3.12)"
```

The checkout has direct pins only. The lock command needs registry access and
was blocked by PyPI DNS resolution during remediation. The full Python 3.12
stack has yet to be installed and tested together. Select the registered kernel
in your notebook frontend. For a hosted notebook, upload the project files
together, use a Python 3.12 runtime with the dependencies prepared separately,
and set `CET433_PROJECT_ROOT` to the project directory if the kernel starts
elsewhere. The notebook does not install packages.

## Notebook and artifacts

Obtain `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` separately
and put them beside the M01R notebook. Start a fresh kernel for this notebook.
The notebook fits preprocessing inside each model pipeline and keeps the
official test partition separate from model and threshold selection.

Labeled training inputs require binary `label` values and nonblank
`attack_cat` labels consistent with `Normal` versus attack. Flow tables require
rows and unique, nonblank CSV headers. Inference selects features in artifact
order, discards extra columns, and rejects missing columns and incompatible
values. `proto`, `service`, and `state` are categorical. The other model
features must be numeric or parseable as numbers. Missing features remain
missing for pipeline handling, and numeric infinities are converted to missing.

A complete run writes these outputs under the project directory:

- `models/ids_model.joblib` with the fitted binary pipeline, feature schema,
  decision threshold, model name, artifact version, Python version, and model
  package versions.
- `data_sample/unseen_flows_sample.csv` with an official test sample.
- `results.json` with the run's evaluation summary.

The dashboard requires the current artifact metadata. Use artifacts from your
own trusted run, since loading joblib can execute code. The
`ids_utils.log_clip` serialization symbol is preserved. Older artifact
dictionaries without metadata need to be re-exported under the current
contract, and package versions should match the training environment.

## Dashboard and tests

Run from the project directory:

```sh
.venv/bin/python -m streamlit run app/soc_dashboard.py
.venv/bin/python -B -m unittest discover -s tests -v
```

You can also pass the absolute dashboard script path from elsewhere. The model
location is resolved from the project files. The dashboard reports a clear
error if the model is absent or its contract is invalid. With a generated model,
upload an unseen flow CSV to see real scores, download the results, and inspect
SHAP contributions for an individual row.

The bounded tests use tiny tables and an in-memory transformer serialization
check. They need no dataset, trained model, or Streamlit server.

## Remaining scientific and submission work

- Record the exact dataset source and usage terms, input checksums, and split
  membership. Check identical feature vectors, conflicting labels, and related
  flows across CV folds, validation, and the official test partition.
- Obtain verified timestamps and compare random and chronological splits using
  a matched protocol. The official UNSW-NB15 holdout is the existing
  generalization check.
- Review joint feasibility for recall ≥95% and FPR <1%, the test threshold sweep,
  rare-class support, and the analyst-time cost assumptions. M01R preserves the
  existing threshold-selection and evaluation methods.
- Verify the SHAP class mapping and output scale for the selected model, and
  correct misleading fallback case titles before reporting results.
- Validate strict JSON export for missing comparison metrics. Complete the
  authorized full run, dashboard smoke test, report, slides, and contribution
  evidence using measured results.
