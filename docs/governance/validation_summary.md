# Validation summary

Validation is reproducible through `scripts/run_demo.py` and `pytest`. Reviewers should inspect temporal boundaries, censoring, leakage allowlists, calibration on a later cohort, segment results, recovery sample size, policy sensitivity, stress assumptions, and decision replay. Generated metrics live in `artifacts/demo_report.json`; no static result is asserted before execution. Independent validation on representative real data remains mandatory.

