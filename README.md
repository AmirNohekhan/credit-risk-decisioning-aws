# Credit Risk Modeling & Loan Decisioning Platform

A lender must decide whether extending credit creates an acceptable risk–return trade-off. This local-first platform estimates 12-month probability of default (PD), loss given default (LGD), exposure at default (EAD), and expected loss (EL), then combines those estimates with pricing and versioned credit policy to produce auditable **APPROVE / REFER / DECLINE** decisions for unsecured personal loans.

```text
Application → application-time features → calibrated PD / LGD / EAD
            → Expected Loss → pricing/economics → credit policy → Decision + reasons
```

The model score and policy decision are deliberately separate. Protected characteristics are never model inputs. Results are synthetic demonstrations, not lending advice or claims of regulatory compliance.

## What is implemented

- Reproducible applicants, prior-policy approvals, monthly amortization/performance, delinquency, charge-off, recovery, and maturity-aware 12-month labels.
- Chronological train/validation/test splits; interpretable logistic champion and gradient-boosting challenger with held-out sigmoid calibration.
- PD discrimination/calibration metrics, modeled LGD among defaults, installment-loan EAD, EL, grades, risk-based pricing, and expected contribution.
- Configurable policy rules, linked reason codes, reject-selection diagnostics/IPW utility, audit-only fairness analysis, PSI, vintages, stress tests, and a policy frontier.
- FastAPI scoring, decision, retrieval, simulation, and stress endpoints; JSON local audit store; AWS adapter architecture and Terraform.

## Credit-risk semantics

Default is the first charge-off or 90+ days past due within 12 months after origination. Features are frozen at application time. A non-default label is assigned only after the full window has matured; an earlier default can be labeled immediately. EAD is estimated outstanding principal for a fully funded installment loan, not a revolving-credit conversion factor. See [default definition](docs/modeling/default_definition.md) and [model card](docs/governance/model_card.md).

## Local quick start

```bash
python -m venv .venv
# activate the environment, then:
python -m pip install -e ".[dev]"
python scripts/run_demo.py
pytest
uvicorn credit_risk.serving:app --reload
```

Or use `make setup`, `make data`, `make demo`, `make test`, and `make serve`. The demo writes genuine run outputs to `artifacts/demo_report.json`, `policy_frontier.csv`, and `vintage_analysis.csv`; this README intentionally contains no fabricated metrics.

Example request:

```bash
curl -X POST http://localhost:8000/v1/applications/decision -H "content-type: application/json" -d '{"application_id":"app_847201","applicant_id":"cust_23181","requested_amount":25000,"term_months":36,"annual_income":92000,"employment_length_years":6,"housing_status":"mortgage","credit_history_years":11,"loan_purpose":"debt_consolidation","bureau":{"credit_score":704,"revolving_utilization":0.41,"open_accounts":8,"delinquencies_24m":0,"hard_inquiries_6m":1,"total_debt":38600}}'
```

## AWS production architecture

```text
Client → API Gateway → Lambda orchestration → SageMaker endpoint (model only)
                                      ├── policy/pricing configuration
                                      └── DynamoDB decision metadata

EventBridge → Step Functions → validate → feature build → train/calibrate
                         → backtest/fairness/stability gate → Model Registry
S3: raw/processed/features/models/predictions/monitoring/backtests
CloudWatch: latency/errors/volume plus links to analytical monitoring artifacts
```

SageMaker owns training, registry, online model scoring, and Batch Transform; Lambda validates and applies policy. DynamoDB is operational metadata only. S3 analytical data should be Parquet partitioned by product and origination date. At scale, use distributed processing, endpoint autoscaling, version-pinned policy/model references, and scheduled matured-cohort monitoring. Immediate monitoring covers score/decision/missingness drift; outcome metrics wait for maturation.

## Deployment and cost

Build and push the image to ECR, create the SageMaker model/endpoint configuration, package Lambda, then:

```bash
cd infrastructure/terraform/environments/dev
terraform init
terraform plan -var='artifact_bucket_name=globally-unique-name'
terraform apply
# remove paid resources when finished
terraform destroy
```

AWS credentials are not required locally. A continuously provisioned SageMaker endpoint is the main development cost; use serverless inference where suitable or delete endpoints. S3, DynamoDB on-demand, Lambda, API Gateway, logs, NAT gateways, ECR storage, KMS, and Step Functions also incur usage charges. Production requires private networking, least privilege, encryption, retention controls, PII-safe logs, formal validation, compliance/legal review, and independent infrastructure security review.

## Limitations

Synthetic relationships cannot establish real-world performance, fairness, or pricing. Reject inference does not reveal counterfactual outcomes; IPW depends on correct acceptance propensities and overlap. The compact challenger uses scikit-learn histogram boosting rather than a distributed tuner. Reason codes are deterministic policy/risk-factor codes, not a claim that they meet any jurisdiction's adverse-action rules. Terraform is a secure deployable skeleton and needs organization-specific VPC, CI identity, alarms, model artifact, and Lambda package inputs.

Architecture decisions are captured in [ADRs](docs/decisions/README.md); monitoring, governance, security, data provenance, scale, and change controls are under `docs/`.

