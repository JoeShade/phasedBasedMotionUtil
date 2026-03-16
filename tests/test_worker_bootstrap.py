"""This file tests the lightweight worker bootstrap so the PyQt shell does not import the heavy render implementation merely to prepare a spawn target."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_worker_bootstrap_import_does_not_load_render_module() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import phase_motion_app.worker.bootstrap\n"
                "print('phase_motion_app.worker.render' in sys.modules)\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    assert completed.stdout.strip() == "False"
