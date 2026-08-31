from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from .features import application_features
from .modeling import ModelBundle
from .schemas import Application, Decision, RiskScore

GRADE_BOUNDS = [(0.025, "A"), (0.055, "B"), (0.10, "C"), (0.18, "D"), (1.0, "E")]


def risk_grade(pd_: float) -> str:
    return next(g for bound, g in GRADE_BOUNDS if pd_ <= bound)


def ead_estimate(amount: float, term: int) -> float:
    return amount * (0.88 if term <= 36 else 0.92)


def reason_codes(f: dict, pd_: float) -> list[str]:
    adverse = []
    if (f["revolving_utilization"] or 0) > 0.65:
        adverse.append("HIGH_REVOLVING_UTILIZATION")
    if float(f["debt_to_income"]) > 0.45:
        adverse.append("HIGH_DEBT_TO_INCOME")
    if int(f["delinquencies_24m"] or 0) > 0:
        adverse.append("RECENT_DELINQUENCY")
    if float(f["credit_history_years"]) < 3:
        adverse.append("LIMITED_CREDIT_HISTORY")
    if int(f["hard_inquiries_6m"] or 0) >= 3:
        adverse.append("MULTIPLE_RECENT_INQUIRIES")
    if float(f["loan_to_income"]) > 0.4:
        adverse.append("HIGH_LOAN_TO_INCOME")
    if not adverse:
        adverse = [
            "STRONG_PAYMENT_HISTORY"
            if int(f["delinquencies_24m"] or 0) == 0
            else "MODERATE_CREDIT_RISK"
        ]
    return adverse[:4]


@dataclass
class Policy:
    version: str = "policy-v1"
    max_pd: float = 0.18
    refer_pd: float = 0.10
    max_dti: float = 0.55
    min_income: float = 18_000
    severe_delinquencies: int = 3
    funding_cost: float = 0.05
    operating_cost: float = 0.02
    target_margin: float = 0.035
    apr_floor: float = 0.07
    apr_cap: float = 0.36

    @classmethod
    def from_yaml(cls, path: str | Path) -> Policy:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        pricing = config.get("pricing", {})
        return cls(
            version=config["version"],
            max_pd=float(config["max_pd"]),
            refer_pd=float(config["refer_pd"]),
            max_dti=float(config["max_dti"]),
            min_income=float(config["minimum_verified_income"]),
            severe_delinquencies=int(config["severe_delinquencies_24m"]),
            funding_cost=float(pricing["funding_cost"]),
            operating_cost=float(pricing["operating_cost"]),
            target_margin=float(pricing["target_margin"]),
            apr_floor=float(pricing["apr_floor"]),
            apr_cap=float(pricing["apr_cap"]),
        )


class DecisionEngine:
    def __init__(self, bundle: ModelBundle, policy: Policy | None = None):
        self.bundle = bundle
        self.policy = policy or Policy()

    def score(self, app: Application) -> RiskScore:
        f = application_features(app)
        frame = pd.DataFrame([f])
        pd_ = float(self.bundle.predict_pd(frame)[0])
        lgd = float(self.bundle.predict_lgd(frame)[0])
        ead = ead_estimate(app.requested_amount, app.term_months)
        el = pd_ * lgd * ead
        return RiskScore(
            application_id=app.application_id,
            pd_12m=round(pd_, 6),
            lgd=round(lgd, 6),
            ead=round(ead, 2),
            expected_loss=round(el, 2),
            risk_grade=risk_grade(pd_),
            model_version=self.bundle.model_version,
            feature_version=self.bundle.feature_version,
        )

    def decide(self, app: Application) -> Decision:
        s = self.score(app)
        f = application_features(app)
        p = self.policy
        annual_el = s.expected_loss / app.requested_amount
        apr = float(
            np.clip(
                p.funding_cost + p.operating_cost + p.target_margin + annual_el,
                p.apr_floor,
                p.apr_cap,
            )
        )
        expected_interest = app.requested_amount * apr * (app.term_months / 12) * 0.55
        funding = app.requested_amount * p.funding_cost * (app.term_months / 12) * 0.55
        net = (
            expected_interest
            + s.ead * 0.01
            - s.expected_loss
            - funding
            - app.requested_amount * p.operating_cost
        )
        hard = (
            app.annual_income < p.min_income
            or float(f["debt_to_income"]) > p.max_dti
            or int(f["delinquencies_24m"] or 0) >= p.severe_delinquencies
        )
        decision = cast(
            Literal["APPROVE", "REFER", "DECLINE"],
            "DECLINE"
            if hard or s.pd_12m > p.max_pd or net < 0
            else ("REFER" if s.pd_12m > p.refer_pd or int(f["thin_file"]) else "APPROVE"),
        )
        return Decision(
            **s.model_dump(),
            decision_id=f"dec_{uuid4().hex[:16]}",
            decision=decision,
            recommended_apr=round(apr * 100, 2),
            expected_net_contribution=round(net, 2),
            reason_codes=reason_codes(f, s.pd_12m),
            policy_version=p.version,
        )


def policy_simulation(engine: DecisionEngine, apps: list[Application]) -> dict:
    ds = [engine.decide(a) for a in apps]
    n = max(len(ds), 1)
    booked = [d for d in ds if d.decision == "APPROVE"]
    return {
        "applications": len(ds),
        "approval_rate": sum(d.decision == "APPROVE" for d in ds) / n,
        "refer_rate": sum(d.decision == "REFER" for d in ds) / n,
        "decline_rate": sum(d.decision == "DECLINE" for d in ds) / n,
        "booked_volume": round(sum(d.ead for d in booked), 2),
        "expected_loss": round(sum(d.expected_loss for d in booked), 2),
        "expected_net_contribution": round(sum(d.expected_net_contribution for d in booked), 2),
    }
