# Monitoring and change policy

Daily operational monitoring covers API errors/latency/volume, idempotency failures, score and decision distributions. Monthly cohort monitoring covers feature missingness, PSI, grade mix, approval/refer/decline rates, and expected loss. Delayed outcome monitoring begins only as cohorts mature and covers calibration, AUC/KS, defaults, realized loss, roll rates, and vintage curves.

Persistent breaches prompt investigation and controlled challenger development; they do not automatically retrain or promote. Changes to target, features, calibration, grade cutoffs, policies, pricing assumptions, or infrastructure require versioning, tests, backtest, fairness/stability review, approval, rollback plan, and an audit entry. Models are not retrained per application.

