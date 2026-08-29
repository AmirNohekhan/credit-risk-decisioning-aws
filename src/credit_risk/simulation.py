from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features, monthly_payment

PURPOSES = ["debt_consolidation", "home_improvement", "major_purchase", "medical", "other"]
HOUSING = ["rent", "mortgage", "own", "other"]


def generate_applications(n: int = 3000, seed: int = 42, start: str = "2021-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(start) + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    income = np.exp(rng.normal(np.log(70000), 0.5, n)).clip(18000, 350000)
    history = rng.gamma(3, 3, n).clip(0, 40)
    score = (610 + history * 5 + rng.normal(0, 55, n)).clip(300, 850)
    util = rng.beta(2, 3, n).clip(0, 1.4)
    debt = income * rng.beta(2.2, 4, n) * 0.9
    frame = pd.DataFrame(
        {
            "application_id": [f"app_{i:07d}" for i in range(n)],
            "applicant_id": [f"cust_{i:07d}" for i in range(n)],
            "application_date": dates,
            "annual_income": income.round(2),
            "employment_length_years": rng.uniform(0, 20, n),
            "housing_status": rng.choice(HOUSING, n, p=[0.47, 0.35, 0.14, 0.04]),
            "credit_history_years": history,
            "requested_amount": rng.uniform(2000, 40000, n).round(2),
            "term_months": rng.choice([24, 36, 48, 60], n, p=[0.1, 0.5, 0.15, 0.25]),
            "loan_purpose": rng.choice(PURPOSES, n, p=[0.5, 0.13, 0.12, 0.1, 0.15]),
            "credit_score": score.round(),
            "revolving_utilization": util,
            "open_accounts": rng.poisson(7, n),
            "delinquencies_24m": rng.poisson(0.3, n),
            "hard_inquiries_6m": rng.poisson(1.2, n),
            "total_debt": debt.round(2),
            "audit_group": rng.choice(["group_a", "group_b"], n),
        }
    )
    thin = rng.random(n) < 0.12
    frame.loc[thin, "credit_history_years"] = rng.uniform(0, 2, thin.sum())
    frame.loc[thin & (rng.random(n) < 0.55), ["credit_score", "open_accounts"]] = np.nan
    frame.loc[rng.random(n) < 0.08, "employment_length_years"] = np.nan
    frame.loc[rng.random(n) < 0.04, "revolving_utilization"] = np.nan
    return build_features(frame).sort_values("application_date").reset_index(drop=True)


def historical_policy(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame.credit_score.fillna(600) >= 570)
        & (frame.debt_to_income <= 0.65)
        & (frame.delinquencies_24m <= 3)
        & (frame.annual_income >= 18000)
    )


def simulate_performance(  # type: ignore[misc]
    apps: pd.DataFrame, seed: int = 43, as_of: str = "2025-12-31"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    loans = apps.copy()
    loans["approved_historically"] = historical_policy(loans)
    x = (
        -3.4
        + 0.020 * (650 - loans.credit_score.fillna(600))
        + 2.1 * loans.revolving_utilization.fillna(0.6)
        + 2.5 * loans.debt_to_income
        + 0.45 * loans.delinquencies_24m
        + 0.16 * loans.hard_inquiries_6m
        - 0.035 * loans.credit_history_years
        + 0.6 * loans.thin_file
    )
    loans["latent_pd_12m"] = 1 / (1 + np.exp(-x))
    loans["latent_default_12m"] = rng.random(len(loans)) < loans.latent_pd_12m
    loans["matured_12m"] = (pd.Timestamp(as_of) - loans.application_date).dt.days >= 365
    loans["default_12m"] = np.where(loans.matured_12m, loans.latent_default_12m.astype(int), np.nan)
    rows = []
    for loan in loans[loans.approved_historically].itertuples():
        balance = float(loan.requested_amount)
        apr = 0.08 + 0.20 * float(loan.latent_pd_12m)
        pmt = monthly_payment(balance, apr, int(loan.term_months))
        default_month = int(rng.integers(3, 13)) if loan.latent_default_12m else 999
        for month in range(1, min(int(loan.term_months), 18) + 1):
            if loan.application_date + pd.DateOffset(months=month) > pd.Timestamp(as_of):
                break
            interest = balance * apr / 12
            scheduled = min(pmt, balance + interest)
            dpd = 0
            if month == default_month:
                dpd = 90
            actual = 0.0 if dpd >= 90 else scheduled
            principal = max(0, actual - interest)
            balance = max(0, balance - principal)
            recovery = float(balance * rng.beta(2, 5)) if dpd >= 90 else 0.0
            rows.append(
                {
                    "application_id": loan.application_id,
                    "month": month,
                    "scheduled_payment": scheduled,
                    "actual_payment": actual,
                    "days_past_due": dpd,
                    "balance": balance,
                    "charge_off": dpd >= 90,
                    "recovery_amount": recovery,
                    "prepayment": False,
                }
            )
            if dpd >= 90 or balance <= 0:
                break
    perf = pd.DataFrame(rows)
    if not perf.empty:
        defaults = perf[perf.charge_off].set_index("application_id")
        loans["ead_realized"] = loans.application_id.map(defaults.balance)
        loans["recovery_realized"] = loans.application_id.map(defaults.recovery_amount)
        loans["lgd_realized"] = (1 - loans.recovery_realized / loans.ead_realized).clip(0, 1)
    return loans, perf


def matured_booked(loans: pd.DataFrame) -> pd.DataFrame:
    return loans[
        loans.approved_historically & (loans.matured_12m | (loans.default_12m == 1))
    ].copy()
