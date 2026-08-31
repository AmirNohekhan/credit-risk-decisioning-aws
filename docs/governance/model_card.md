# PD model card

## Purpose and population

Application-time ranking and calibrated 12-month PD estimation for synthetic unsecured personal-loan applicants. Intended for engineering demonstration and policy simulation; not intended for real lending, regulatory capital, collections, fraud, or other credit products.

## Target and sample

Default and censoring follow `docs/modeling/default_definition.md`. Training reflects only loans booked by a simulated previous policy, exposing acceptance bias. Splits are chronological 60/20/20 train/calibration/test. Audit attributes, outcomes, approval decisions, dates after observation, recoveries, and protected traits are excluded.

## Methods

Champion benchmark: imputation, one-hot encoding, scaling, and class-weighted logistic regression. Challenger: regularized histogram gradient boosting; probabilities are sigmoid-calibrated on a later validation cohort. Evaluation reports ROC-AUC, PR-AUC, KS, Gini, and Brier. Promotion should additionally require segment stability, calibration, EL backtest, policy economics, fairness review, explanation quality, and independent validation.

LGD is a bounded random-forest regression among defaults with observed synthetic recoveries. It is evaluated on out-of-time test defaults using MAE, RMSE, mean actual/predicted LGD, and a training-mean LGD benchmark. Small default samples must be disclosed and prevent model fitting below the configured minimum. EAD is a documented installment-balance approximation. EL is PD × LGD × EAD. These assumptions are simplified and product-specific.

## Risks and monitoring

Synthetic validity, selection bias, missing-data shift, calibration drift, proxy discrimination, policy feedback loops, limited-default uncertainty, and weak recovery data are material risks. Monitor immediate scores, grades, approvals, missingness and PSI; after maturation monitor default rate, AUC/KS, calibration, realized loss and vintage performance. PSI informs investigation but never automatically triggers retraining. Review when new cohorts mature, drift persists, performance degrades, or policy/target changes.

Reject-inference analysis compares an approved-only logistic model with inverse-probability weighting based on the historical acceptance propensity. Population evaluation uses synthetic counterfactual outcomes solely because the simulator knows them. Real declined-applicant outcomes remain unknown, and weighting depends on correct propensity specification and adequate overlap; the method is a sensitivity analysis, not recovery of truth.
