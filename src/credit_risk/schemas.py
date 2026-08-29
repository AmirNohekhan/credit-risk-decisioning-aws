from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Bureau(BaseModel):
    credit_score: int | None = Field(None, ge=300, le=850)
    revolving_utilization: float | None = Field(None, ge=0, le=2)
    open_accounts: int | None = Field(None, ge=0, le=100)
    delinquencies_24m: int | None = Field(None, ge=0, le=50)
    hard_inquiries_6m: int | None = Field(None, ge=0, le=30)
    total_debt: float | None = Field(None, ge=0)


class Application(BaseModel):
    application_id: str = Field(min_length=3, max_length=80)
    applicant_id: str = Field(min_length=3, max_length=80)
    requested_amount: float = Field(ge=1_000, le=100_000)
    term_months: Literal[24, 36, 48, 60]
    annual_income: float = Field(ge=1_000, le=5_000_000)
    employment_length_years: float | None = Field(None, ge=0, le=60)
    housing_status: Literal["rent", "mortgage", "own", "other"]
    credit_history_years: float = Field(ge=0, le=70)
    loan_purpose: Literal[
        "debt_consolidation", "home_improvement", "major_purchase", "medical", "other"
    ] = "other"
    bureau: Bureau

    @model_validator(mode="after")
    def plausible(self):
        if self.requested_amount > self.annual_income * 2:
            raise ValueError("requested amount cannot exceed twice annual income")
        return self


class RiskScore(BaseModel):
    application_id: str
    pd_12m: float
    lgd: float
    ead: float
    expected_loss: float
    risk_grade: str
    model_version: str
    feature_version: str


class Decision(RiskScore):
    decision_id: str
    decision_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    decision: Literal["APPROVE", "REFER", "DECLINE"]
    recommended_apr: float
    expected_net_contribution: float
    reason_codes: list[str]
    policy_version: str


class PolicyRequest(BaseModel):
    applications: list[Application]
    max_pd: float = Field(0.18, gt=0, lt=1)
    refer_pd: float = Field(0.10, gt=0, lt=1)
    max_dti: float = Field(0.55, gt=0, lt=2)


class StressRequest(BaseModel):
    applications: list[Application]
    pd_multiplier: float = Field(1.35, ge=1, le=5)
    lgd_addon: float = Field(0.08, ge=0, le=0.5)
    funding_cost_addon: float = Field(0.02, ge=0, le=0.2)
