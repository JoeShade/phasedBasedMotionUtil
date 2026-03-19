# Phase-based Motion Amplification Desktop Utility

This repository contains the current Phase-based Motion Amplification desktop application. It is a PyQt6 shell around a testable core domain layer plus a separate spawned worker process for heavy render execution.

The supported product shape is intentionally narrow:

- offline processing of recorded video
- phase-based amplification only
- one active render at a time
- MP4 video output with audio stripped
- static source-space mask zones and one optional quantitative-analysis ROI

`systemDesign.md` is the primary design source of truth. `docs/architecture-notes.md` adds short implementation notes, and `AGENTS.md` is the contributor/agent workflow guide for this repository.

## Implemented workflow

1. Select a recorded video source.
2. Run shell-side probe, SHA-256 fingerprinting, and first/last-frame extraction.
3. Review drift, define static mask zones, and optionally define one quantitative-analysis ROI.
4. Run shell-side dry-run pre-flight.
5. Launch one spawned worker that reruns authoritative pre-flight and performs the render.
6. Finalize a paired MP4 plus JSON sidecar and write diagnostics artifacts.

## Current capabilities

- shell-side source probing, source fingerprinting, stale-source detection, and first-frame preview
- drift review using first/last decoded frames
- static include/exclude mask zones with feathering
- tied processing/output resolution with one codec-safe effective render size
- optional quantitative analysis with auto or manual band behavior and render-time artifact export
- shell-side plus worker-side pre-flight checks
- loopback IPC, watchdog supervision, and explicit terminal-state classification
- bounded worker pipeline with normalization, decode, phase processing, optional background analysis handoff, encode, validation, and paired finalization
- optional CuPy-backed acceleration for dense warp, render-path resize, and FFT-based local motion estimation, with explicit CPU fallback when the backend is unavailable or disabled
- diagnostics bundle writing, retained-evidence cleanup, and convenience settings persistence

## Non-goals

- live capture or live preview
- full amplified preview playback in the shell
- batch queues or concurrent renders
- moving masks or tracked ROIs
- a separate operator-controlled output resize distinct from processing resolution
- analysis-only runs without a render

## Install and run

Install from a source checkout:

```powershell
python -m pip install -e .
```

Launch the installed entrypoint:

```powershell
phase-motion-app
```

Or launch directly from a Windows source checkout:

```powershell
.\run.bat
```

Install developer dependencies:

```powershell
python -m pip install -e .[dev]
```

The core runtime dependencies are `PyQt6`, `numpy`, `jsonschema`, `psutil`, and `static-ffmpeg`.

The app resolves `ffmpeg` and `ffprobe` through `static-ffmpeg` by default. If you need explicit overrides, set both `PHASE_MOTION_FFMPEG` and `PHASE_MOTION_FFPROBE` together. Partial override configuration is treated as an error.

Optional hardware acceleration uses CuPy and is not required for the default install path. Install a CuPy wheel that matches the local CUDA runtime, for example:

```powershell
python -m pip install cupy-cuda12x nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
```

On Windows, the app registers the packaged NVIDIA runtime and NVRTC paths before importing CuPy so optional PyPI-installed GPU support can activate without a separately installed CUDA toolkit.

## Tests

Run targeted tests while iterating:

```powershell
python -m pytest tests/test_settings_store.py
```

Run the full suite before finalizing:

```powershell
python -m pytest
```

## Repository layout

- `src/phase_motion_app/app`: PyQt shell, dialogs, shell-side validation, and worker supervision
- `src/phase_motion_app/core`: testable domain logic, models, sidecars, diagnostics, media helpers, storage rules, and numeric processing
- `src/phase_motion_app/worker`: spawned worker bootstrap plus the real render worker
- `tests`: regression suite covering core logic, Qt shell behavior, supervision, and worker integration
- `tools`: developer-only scripts such as the synthetic performance smoke

## Runtime data and sample inputs

When launched from a source checkout, the app defaults to repo-local runtime directories so development stays self-contained:

- `input/`
- `output/`
- `temp/`
- `diagnostics/`

The tracked `input/` clips are development fixtures for manual checks. `output/`, `temp/`, `diagnostics/`, `scratch/`, and similar runtime artifacts are intentionally ignored or reduced to `.gitkeep` placeholders where appropriate.

Convenience state is stored separately under `~/.phase_motion_app/settings.json` unless a test or caller overrides the state path.

## Supporting docs

- [systemDesign.md](systemDesign.md)
- [AGENTS.md](AGENTS.md)
- [docs/architecture-notes.md](docs/architecture-notes.md)
- [docs/deviations.md](docs/deviations.md)
