import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credit_risk.simulation import generate_applications, simulate_performance

out = ROOT / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
loans, performance = simulate_performance(generate_applications())
loans.to_csv(out / "loans.csv", index=False)
performance.to_csv(out / "monthly_performance.csv", index=False)
print(f"wrote {len(loans)} applications and {len(performance)} monthly records")
