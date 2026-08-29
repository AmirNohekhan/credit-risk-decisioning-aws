from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schemas import Application

NUMERIC_FEATURES = [
    "requested_amount",
    "term_months",
    "annual_income",
    "employment_length_years",
    "credit_history_years",
    "credit_score",
    "revolving_utilization",
    "open_accounts",
    "delinquencies_24m",
    "hard_inquiries_6m",
    "total_debt",
    "debt_to_income",
    "loan_to_income",
    "payment_to_income",
    "thin_file",
]
CATEGORICAL_FEATURES = ["housing_status", "loan_purpose"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 12
    return principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months)


def application_features(app: Application) -> dict[str, Any]:
    b = app.bureau
    score = b.credit_score if b.credit_score is not None else 620
    debt = b.total_debt if b.total_debt is not None else app.annual_income * 0.3
    thin = int(app.credit_history_years < 2 or (b.open_accounts or 0) < 2 or b.credit_score is None)
    payment = monthly_payment(app.requested_amount, 0.15, app.term_months)
    return {
        "requested_amount": app.requested_amount,
        "term_months": app.term_months,
        "annual_income": app.annual_income,
        "employment_length_years": app.employment_length_years,
        "credit_history_years": app.credit_history_years,
        "housing_status": app.housing_status,
        "loan_purpose": app.loan_purpose,
        "credit_score": score,
        "revolving_utilization": b.revolving_utilization,
        "open_accounts": b.open_accounts,
        "delinquencies_24m": b.delinquencies_24m,
        "hard_inquiries_6m": b.hard_inquiries_6m,
        "total_debt": debt,
        "debt_to_income": debt / app.annual_income,
        "loan_to_income": app.requested_amount / app.annual_income,
        "payment_to_income": payment * 12 / app.annual_income,
        "thin_file": thin,
    }


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["debt_to_income"] = out.total_debt / out.annual_income
    out["loan_to_income"] = out.requested_amount / out.annual_income
    rates = np.full(len(out), 0.15)
    monthly = out.requested_amount * (rates / 12) / (1 - (1 + rates / 12) ** (-out.term_months))
    out["payment_to_income"] = monthly * 12 / out.annual_income
    out["thin_file"] = (
        (out.credit_history_years < 2) | (out.open_accounts.fillna(0) < 2) | out.credit_score.isna()
    ).astype(int)
    return out
