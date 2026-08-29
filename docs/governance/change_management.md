# Change management

Every production decision pins model, feature, and policy versions. Candidate promotion requires code review, CI, immutable artifacts, model metadata, chronological validation, calibration and stability gates, policy backtest, fairness review, security approval, and rollback readiness. Emergency policy changes remain separate from model releases. DynamoDB records metadata needed to reproduce a decision; sensitive raw attributes belong in governed systems, not logs.

