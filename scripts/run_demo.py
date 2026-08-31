from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credit_risk.analytics import (
    fairness_audit,
    inverse_probability_weights,
    population_stability_index,
    reject_inference_comparison,
    vintage_analysis,
)
from credit_risk.decisioning import DecisionEngine, Policy, policy_simulation, stress_portfolio
from credit_risk.modeling import save_registry_manifest, train_models
from credit_risk.schemas import Application, Bureau
from credit_risk.simulation import generate_applications, matured_booked, simulate_performance


def sample_app(row) -> Application:
    def val(name, default):
        v = getattr(row, name)
        return default if pd.isna(v) else v

    return Application(
        application_id=str(row.application_id),
        applicant_id=str(row.applicant_id),
        requested_amount=float(row.requested_amount),
        term_months=int(row.term_months),
        annual_income=float(row.annual_income),
        employment_length_years=float(val("employment_length_years", 5)),
        housing_status=str(row.housing_status),
        credit_history_years=float(row.credit_history_years),
        loan_purpose=str(row.loan_purpose),
        bureau=Bureau(
            credit_score=int(val("credit_score", 620)),
            revolving_utilization=float(val("revolving_utilization", 0.5)),
            open_accounts=int(val("open_accounts", 4)),
            delinquencies_24m=int(row.delinquencies_24m),
            hard_inquiries_6m=int(row.hard_inquiries_6m),
            total_debt=float(row.total_debt),
        ),
    )


def main(n: int = 2500):
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    apps = generate_applications(n)
    loans, performance = simulate_performance(apps)
    booked = matured_booked(loans)
    bundle, model_report = train_models(booked)
    bundle.save(str(out / "model.joblib"))
    save_registry_manifest(str(out / "model_registry.json"), bundle, model_report)
    # Keep the interactive demo quick; model metrics still use the full chronological test cohort.
    engine = DecisionEngine(bundle, Policy.from_yaml(ROOT / "configs" / "policy.yaml"))
    test = booked.sort_values("application_date").tail(60).copy()
    decisions = [engine.decide(sample_app(r)) for r in test.itertuples()]
    test["decision"] = [d.decision for d in decisions]
    test["pd_12m"] = [d.pd_12m for d in decisions]
    test["expected_loss"] = [d.expected_loss for d in decisions]
    policies = {
        name: policy_simulation(
            DecisionEngine(bundle, Policy(version=name, max_pd=maxpd, refer_pd=refer)),
            [sample_app(r) for r in test.itertuples()],
        )
        for name, maxpd, refer in [
            ("conservative", 0.10, 0.065),
            ("balanced", 0.18, 0.10),
            ("growth", 0.25, 0.15),
        ]
    }
    frontier = []
    for threshold in [0.07, 0.10, 0.13, 0.16, 0.20, 0.25]:
        result = policy_simulation(
            DecisionEngine(bundle, Policy(max_pd=threshold, refer_pd=max(0.04, threshold - 0.06))),
            [sample_app(r) for r in test.itertuples()],
        )
        result["max_pd"] = threshold
        frontier.append(result)
    fairness = fairness_audit(test).to_dict("records")
    vintage = vintage_analysis(loans, performance)
    vintage.to_csv(out / "vintage_analysis.csv", index=False)
    pd.DataFrame(frontier).to_csv(out / "policy_frontier.csv", index=False)
    stress = stress_portfolio(engine, [sample_app(r) for r in test.itertuples()], 1.35, 0.08, 0.02)
    report = {
        "synthetic_population": {
            "applications": len(apps),
            "historically_approved": int(loans.approved_historically.sum()),
            "matured_booked": len(booked),
            "defaults": int(booked.default_12m.sum()),
        },
        "models": model_report,
        "portfolio": {
            "approval_rate": float((test.decision == "APPROVE").mean()),
            "observed_default_rate": float(test.default_12m.mean()),
            "expected_loss": float(test.expected_loss.sum()),
        },
        "policies": policies,
        "stress": stress,
        "fairness_audit": fairness,
        "selection_bias": {
            "approved_rate": float(loans.approved_historically.mean()),
            "population_default_rate": float(loans.default_12m.mean()),
            "booked_default_rate": float(booked.default_12m.mean()),
            "ipw_effective_weight": float(inverse_probability_weights(loans).sum()),
        },
        "reject_inference": reject_inference_comparison(loans),
        "stability": {
            "credit_score_psi": population_stability_index(
                booked.iloc[: len(booked) // 2].credit_score.fillna(620),
                booked.iloc[len(booked) // 2 :].credit_score.fillna(620),
            )
        },
        "example_decision": decisions[0].model_dump(mode="json"),
    }
    (out / "demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
