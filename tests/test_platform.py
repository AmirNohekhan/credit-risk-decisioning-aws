from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from credit_risk.analytics import population_stability_index
from credit_risk.decisioning import DecisionEngine, ead_estimate, risk_grade
from credit_risk.features import monthly_payment
from credit_risk.modeling import ModelBundle, temporal_split
from credit_risk.schemas import Application, Bureau
from credit_risk.serving import create_app
from credit_risk.simulation import generate_applications, matured_booked, simulate_performance


class Stub:
    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 0.96), np.full(len(x), 0.04)])


class LGD:
    def predict(self, x):
        return np.full(len(x), 0.6)


@pytest.fixture
def application():
    return Application(
        application_id="app_test",
        applicant_id="cust_test",
        requested_amount=25000,
        term_months=36,
        annual_income=92000,
        employment_length_years=6,
        housing_status="mortgage",
        credit_history_years=11,
        loan_purpose="debt_consolidation",
        bureau=Bureau(
            credit_score=704,
            revolving_utilization=0.41,
            open_accounts=8,
            delinquencies_24m=0,
            hard_inquiries_6m=1,
            total_debt=38600,
        ),
    )


def test_amortization_and_ead():
    assert monthly_payment(12000, 0.12, 12) == pytest.approx(1066.19, rel=0.01)
    assert ead_estimate(10000, 36) == 8800


def test_maturity_and_temporal_split():
    loans, perf = simulate_performance(generate_applications(300), as_of="2025-12-31")
    mature = matured_booked(loans)
    assert len(perf) > 0 and mature.default_12m.notna().all()
    a, b, c = temporal_split(mature)
    assert a.application_date.max() <= b.application_date.min() <= c.application_date.min()


def test_decision_and_api(application, tmp_path: Path):
    bundle = ModelBundle(Stub(), LGD())
    engine = DecisionEngine(bundle)
    d = engine.decide(application)
    assert d.expected_loss == pytest.approx(0.04 * 0.6 * 22000, abs=0.1)
    assert d.decision == "APPROVE"
    assert d.reason_codes
    client = TestClient(create_app(bundle, tmp_path / "decisions.json"))
    assert client.get("/health").status_code == 200
    response = client.post("/v1/applications/decision", json=application.model_dump(mode="json"))
    assert response.status_code == 200
    assert client.get("/v1/applications/app_test").json()["model_version"] == "pd-gb-cal-v1"


def test_grades_and_psi():
    assert [risk_grade(x) for x in [0.01, 0.04, 0.08, 0.15, 0.3]] == list("ABCDE")
    assert population_stability_index(np.arange(100), np.arange(100)) == pytest.approx(0)


def test_validation_rejects_implausible():
    with pytest.raises(ValueError):
        Application(
            application_id="app",
            applicant_id="cust",
            requested_amount=100000,
            term_months=36,
            annual_income=20000,
            housing_status="rent",
            credit_history_years=2,
            bureau=Bureau(),
        )
