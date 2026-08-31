# Validation summary

Validation is reproducible through `scripts/run_demo.py` and `pytest`. In addition to the final chronological holdout, three expanding-window rolling-origin folds fit on older cohorts, calibrate on the latest portion of each development window, and evaluate the following origination window. Each fold reports AUC/KS/Gini/PR-AUC/Brier plus approval and approved-loan default rates at the policy PD ceiling. Tests assert that training ends before testing begins and that outcome, recovery, approval, date, latent simulation, and audit fields are absent from the model allowlist.

Reviewers should inspect temporal boundaries, censoring, leakage allowlists, calibration on a later cohort, segment results, recovery sample size, policy sensitivity, stress assumptions, and decision replay. Generated metrics live in `artifacts/demo_report.json`; no static result is asserted before execution. Independent validation on representative real data remains mandatory.
