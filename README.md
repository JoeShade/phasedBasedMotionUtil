# Phase-based Motion Amplification Desktop Utility

This repository contains a Python + PyQt6 desktop application for offline-only phase-based motion amplification. The implementation follows `systemDesign.md` as the primary design source.

Current implemented slices:

- project packaging and test configuration
- sidecar domain models and schema validation
- safe sidecar reload that restores only reusable `intent`
- mandatory pre-flight and storage/finalization policy logic
- worker IPC, watchdog, diagnostics bundle writing, and spawn-worker integration
- real render worker using ffmpeg/ffprobe subprocesses and phase-derived amplification
- bounded decode -> compute -> encode worker pipeline with policy-driven queue depth and helper concurrency
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

Resource-policy presets now affect actual worker execution behavior:
- `Conservative`: smallest chunk caps and shallowest queues
- `Balanced`: moderate helper fan-out with bounded queueing
- `Aggressive`: larger chunk caps, deeper bounded queues, and higher warp/motion helper counts when cores allow

Developer perf-smoke:

```powershell
python tools/perf_smoke_phase_pipeline.py --frames 32 --width 192 --height 192 --repetitions 3
```

That script profiles synthetic phase chunks and prints the applied scheduler plan plus motion-estimation, warp, process-chunk, and analysis-handoff timings for the selected resource policies.

Known deviations from the design are tracked in [docs/deviations.md](docs/deviations.md).
