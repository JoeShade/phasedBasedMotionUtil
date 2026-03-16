"""This file owns JSONL diagnostics logging so worker-side events are preserved on disk without flooding the IPC channel."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class JsonlLogger:
    """This logger writes one structured JSON object per line for later diagnostics bundle assembly."""

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        level: str,
        event_type: str,
        job_id: str,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event_type": event_type,
            "job_id": job_id,
            "stage": stage,
            "message": message,
            "payload": payload or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
