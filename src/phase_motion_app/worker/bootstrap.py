"""This file keeps the shell-side worker launch contract lightweight so the PyQt process can spawn a render worker without importing the heavy render implementation up front."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phase_motion_app.core.render_job import RenderRequest


@dataclass(frozen=True)
class RenderWorkerConfig:
    """This model contains the shell endpoint and authoritative render request for one worker process."""

    host: str
    port: int
    session_token: str
    job_id: str
    role: str
    request: RenderRequest


def render_worker_entry(config: RenderWorkerConfig, cancel_event: Any) -> None:
    """Import and run the heavy render worker only inside the spawned child process."""

    from phase_motion_app.worker.render import render_worker_process_main

    render_worker_process_main(config, cancel_event)
