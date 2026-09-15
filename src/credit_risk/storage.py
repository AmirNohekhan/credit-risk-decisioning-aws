import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .schemas import Decision


class JsonDecisionStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, d: Decision) -> None:
        records = self._read()
        records[d.application_id] = d.model_dump(mode="json")
        self._write(records)

    def get(self, application_id: str) -> dict[str, Any] | None:
        return self._read().get(application_id)

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def _write(self, records: dict[str, Any]) -> None:
        # Write to a temp file and rename so a crash mid-write can't truncate
        # or corrupt the store; os.replace is atomic on POSIX and Windows.
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(records, tmp_file, indent=2)
            os.replace(tmp_name, self.path)
        except BaseException:
            os.unlink(tmp_name)
            raise
