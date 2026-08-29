# Architecture decision records

1. **Personal loans:** focused, fully funded installment exposure; alternatives (cards, mortgages) need different behavior and EAD.
2. **12-month PD:** actionable and observable horizon; longer horizons increase censoring.
3. **Logistic champion:** transparent, stable benchmark; not assumed superior.
4. **Boosted challenger:** nonlinear interactions with regularization; requires explanation and stability checks.
5. **Calibration:** underwriting and EL need probabilities, not rankings alone.
6. **Separate policy/model:** allows policy changes, exceptions, and governance without retraining.
7. **Reject inference:** prior approvals select observed outcomes; methods are sensitivity analyses, not truth recovery.
8. **PD/LGD/EAD:** separates default frequency, severity, and exposure; enables EL and stress analysis.
9. **SageMaker:** managed training/registry/endpoints; local interfaces prevent cloud lock-in for development.
10. **Online and batch:** application latency differs from portfolio monitoring throughput.
11. **Versioned decisions:** model, feature, and policy pins are required for replay and audit.

