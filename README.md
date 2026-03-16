# Phase-based Motion Amplification Desktop Utility

This repository contains a Python + PyQt6 desktop application for offline-only phase-based motion amplification. The implementation follows `systemDesign.md` as the primary design source.

Current implemented slices:

- project packaging and test configuration
- sidecar domain models and schema validation
- safe sidecar reload that restores only reusable `intent`
- mandatory pre-flight and storage/finalization policy logic
- worker IPC, watchdog, diagnostics bundle writing, and spawn-worker integration
- real render worker using ffmpeg/ffprobe subprocesses and phase-derived amplification
- fast source probe wiring in the PyQt shell
- drift-check and exclusion-zone editor with source-frame geometry and frame extraction
- shell-side render supervision, dry-run validation, terminal cleanup controls, and last-used settings persistence

How to run tests:

```powershell
python -m pip install -e .[dev]
python -m pytest
```

How to start the app:

```powershell
python -m pip install -e .
phase-motion-app
```

Persisted convenience state is stored under `~/.phase_motion_app/` by default. Final MP4 and sidecar files are still written to the selected output folder.

Known deviations from the design are tracked in [docs/deviations.md](docs/deviations.md).
