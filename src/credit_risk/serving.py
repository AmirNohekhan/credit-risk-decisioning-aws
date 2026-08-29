from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .decisioning import DecisionEngine, Policy, policy_simulation
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
    engine = DecisionEngine(bundle)
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
                bundle, Policy(max_pd=req.max_pd, refer_pd=req.refer_pd, max_dti=req.max_dti)
            ),
            req.applications,
        )

    @api.post("/v1/stress-test")
    def stress(req: StressRequest):
        base = [engine.decide(a) for a in req.applications]
        stressed = []
        for d in base:
            pd_ = min(0.999, d.pd_12m * req.pd_multiplier)
            lgd = min(0.99, d.lgd + req.lgd_addon)
            stressed.append(pd_ * lgd * d.ead)
        return {
            "scenario": "simulated_downturn",
            "applications": len(base),
            "base_expected_loss": sum(d.expected_loss for d in base),
            "stressed_expected_loss": sum(stressed),
            "assumptions": req.model_dump(exclude={"applications"}),
        }

    return api


app = None
try:
    app = create_app()
except RuntimeError:
    pass
