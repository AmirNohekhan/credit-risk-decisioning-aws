from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected, actual, bins: int = 10) -> float:
    e = np.asarray(expected, float)
    a = np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ep = np.histogram(e, edges)[0] / len(e)
    ap = np.histogram(a, edges)[0] / len(a)
    ep = np.clip(ep, 1e-6, None)
    ap = np.clip(ap, 1e-6, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def fairness_audit(df: pd.DataFrame, group_col="audit_group") -> pd.DataFrame:
    return (
        df.groupby(group_col, observed=True)
        .agg(
            count=(group_col, "size"),
            approval_rate=("decision", lambda x: (x == "APPROVE").mean()),
            average_pd=("pd_12m", "mean"),
            observed_default_rate=("default_12m", "mean"),
        )
        .reset_index()
    )


def vintage_analysis(loans: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    base = loans[loans.approved_historically].copy()
    base["vintage"] = base.application_date.dt.to_period("Q").astype(str)
    rows = []
    for vintage, g in base.groupby("vintage"):
        ids = set(g.application_id)
        p = performance[performance.application_id.isin(ids)]
        rows.append(
            {
                "vintage": vintage,
                "loans": len(g),
                "3m_default_rate": p[p.month <= 3].groupby("application_id").charge_off.max().sum()
                / max(len(g), 1),
                "6m_default_rate": p[p.month <= 6].groupby("application_id").charge_off.max().sum()
                / max(len(g), 1),
                "12m_default_rate": p[p.month <= 12]
                .groupby("application_id")
                .charge_off.max()
                .sum()
                / max(len(g), 1),
            }
        )
    return pd.DataFrame(rows)


def inverse_probability_weights(apps: pd.DataFrame) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    x = apps[["credit_score", "debt_to_income", "loan_to_income"]].fillna(
        apps[["credit_score", "debt_to_income", "loan_to_income"]].median()
    )
    m = LogisticRegression().fit(x, apps.approved_historically)
    prop = m.predict_proba(x)[:, 1]
    return np.where(apps.approved_historically, 1 / np.clip(prop, 0.05, 1), 0)
