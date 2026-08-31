from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .decisioning import DecisionEngine, Policy, policy_simulation, stress_portfolio
from .modeling import ModelBundle
from .schemas import Application, PolicyRequest, StressRequest
from .storage import JsonDecisionStore


def create_app(bundle: ModelBundle | None = None, store_path: Path | None = None) -> FastAPI:
    if bundle is None:
        path = Path(os.getenv("CREDIT_RISK_MODEL", "artifacts/model.joblib"))
        if not path.exists():
            raise RuntimeError(
                "Model artifact missing; run `make train` or `python scripts/run_demo.py`"
            )
        bundle = ModelBundle.load(str(path))
    policy_path = Path(os.getenv("CREDIT_RISK_POLICY", "configs/policy.yaml"))
    policy = Policy.from_yaml(policy_path) if policy_path.exists() else Policy()
    engine = DecisionEngine(bundle, policy)
    store = JsonDecisionStore(store_path or Path("artifacts/decisions.json"))
    api = FastAPI(title="Credit Risk Decisioning API", version="1.0")

    @api.get("/health")
    def health():
        return {
            "status": "healthy",
            "model_version": bundle.model_version,
            "policy_version": engine.policy.version,
        }

    @api.post("/v1/applications/score")
    def score(app: Application):
        return engine.score(app)

    @api.post("/v1/applications/decision")
    def decision(app: Application):
        d = engine.decide(app)
        store.put(d)
        return d

    @api.get("/v1/applications/{application_id}")
    def retrieve(application_id: str):
        record = store.get(application_id)
        if not record:
            raise HTTPException(404, "decision not found")
        return record

    @api.post("/v1/policy/simulate")
    def simulate(req: PolicyRequest):
        return policy_simulation(
            DecisionEngine(
                bundle,
                Policy(
                    **{
                        **engine.policy.__dict__,
                        "max_pd": req.max_pd,
                        "refer_pd": req.refer_pd,
                        "max_dti": req.max_dti,
                    }
                ),
            ),
            req.applications,
        )

    @api.post("/v1/stress-test")
    def stress(req: StressRequest):
        return stress_portfolio(
            engine,
            req.applications,
            req.pd_multiplier,
            req.lgd_addon,
            req.funding_cost_addon,
        )

    return api


app = None
try:
    app = create_app()
except RuntimeError:
    pass
