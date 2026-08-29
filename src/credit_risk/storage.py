import json
from pathlib import Path

from .schemas import Decision


class JsonDecisionStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, d: Decision):
        records = self._read()
        records[d.application_id] = d.model_dump(mode="json")
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def get(self, application_id: str):
        return self._read().get(application_id)

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
