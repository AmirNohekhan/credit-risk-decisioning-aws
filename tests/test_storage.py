from types import SimpleNamespace

from credit_risk.storage import JsonDecisionStore


def _decision(app_id, outcome="APPROVE"):
    return SimpleNamespace(
        application_id=app_id,
        model_dump=lambda mode="json": {"application_id": app_id, "decision": outcome},
    )


def test_store_round_trip_and_overwrite(tmp_path):
    store = JsonDecisionStore(tmp_path / "nested" / "decisions.json")
    assert store.get("a") is None
    store.put(_decision("a"))
    store.put(_decision("a", "DECLINE"))
    assert store.get("a")["decision"] == "DECLINE"


def test_store_leaves_no_temp_file(tmp_path):
    path = tmp_path / "decisions.json"
    JsonDecisionStore(path).put(_decision("a"))
    assert [p.name for p in tmp_path.iterdir()] == ["decisions.json"]
