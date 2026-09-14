import json
import os
import tempfile
from pathlib import Path

from .schemas import Decision


class JsonDecisionStore:
    """A JSON-backed decision record store, used as an audit trail.

    Writes are staged to a temporary file in the same directory and then
    atomically renamed into place, so a crash or power loss mid-write can
    never leave ``path`` holding a truncated or partially written file that
    would make every previously stored decision unreadable.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, d: Decision):
        records = self._read()
        records[d.application_id] = d.model_dump(mode="json")
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            Path(tmp_name).unlink(missing_ok=True)

    def get(self, application_id: str):
        return self._read().get(application_id)

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
