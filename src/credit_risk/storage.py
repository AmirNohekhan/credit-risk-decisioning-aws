import contextlib
import json
import os
import tempfile
from pathlib import Path

from .schemas import Decision


class JsonDecisionStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, d: Decision):
        records = self._read()
        records[d.application_id] = d.model_dump(mode="json")
        self._write(records)

    def get(self, application_id: str):
        return self._read().get(application_id)

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def _write(self, records: dict) -> None:
        """Write the store via a temp file plus atomic rename.

        This is an audit trail of lending decisions, so a crash or disk-full
        error partway through a plain write must not leave decisions.json
        truncated or invalid -- that would make every subsequent read of the
        store (including for unrelated application_ids) raise and lose
        previously recorded decisions.
        """
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2)
            os.replace(tmp_name, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp_name)
            raise
