# Phase-based Motion Amplification Desktop Utility

This repository contains the current Python desktop implementation of the Phase-based Motion Amplification Utility. It is a PyQt6 application for offline processing of recorded video with a separate worker process for probe, validation, amplification, quantitative analysis, encoding, diagnostics, and output finalization.

`systemDesign.md` is the primary design and architecture source of truth. `docs/architecture-notes.md` adds short implementation notes, and `docs/deviations.md` is reserved for temporary code/design mismatches.

## Current capabilities

- single-source, single-render desktop workflow
- fast source probe, canonical source fingerprinting, and first-frame preview
- drift review plus static source-space mask zones
- optional quantitative-analysis ROI with render-time artifact export
- mandatory shell-side and worker-side pre-flight checks
- separate render worker with loopback IPC, watchdog supervision, and explicit terminal-state rules
- bounded worker pipeline with decode, phase processing, optional background analysis handoff, encode, staged validation, and paired MP4/JSON finalization
- diagnostics bundle writing, retained-failure cleanup, and last-used convenience settings

## Constraints and non-goals

- offline processing only
- phase-based amplification only
- one active render at a time
- MP4 output only, audio stripped
- downscale-only output policy
- no live mode
- no full amplified preview mode
- no batch queue
- no analysis-only mode

## Install and run

Runtime install:

```powershell
python -m pip install -e .
phase-motion-app
```

Windows source-checkout launcher:

```powershell
.\run.bat
```

Developer install:

```powershell
python -m pip install -e .[dev]
```

The project depends on `PyQt6`, `numpy`, `jsonschema`, `psutil`, and `static-ffmpeg`. The worker resolves `ffmpeg` and `ffprobe` through `static-ffmpeg` unless `PHASE_MOTION_FFMPEG` and `PHASE_MOTION_FFPROBE` are both set explicitly.

## Tests

Run the full suite:

```powershell
python -m pytest
```

During development, run targeted tests for touched modules first and the full suite before finalizing.

## Repository layout

- `src/phase_motion_app/app`: PyQt shell, dialogs, shell-side validation, worker supervision, and UI state management
- `src/phase_motion_app/core`: testable domain logic, models, pre-flight, sidecars, masking, diagnostics, storage, watchdog, and media helpers
- `src/phase_motion_app/worker`: spawned worker bootstrap plus the render worker implementation
- `tests`: unit and integration-style tests
- `tools`: developer-only scripts such as the synthetic pipeline perf smoke

## Runtime paths

When launched from a source checkout, the app uses repo-local `input/`, `output/`, `temp/`, and `diagnostics/` directories by default so development runs stay self-contained. User-level convenience state still lives under `~/.phase_motion_app/settings.json` unless a test or caller overrides the state path.

Diagnostics are written under the configured diagnostics root, and successful exports are committed only when both the final MP4 and matching JSON sidecar are in place.

## Supporting docs

- [systemDesign.md](systemDesign.md)
- [docs/architecture-notes.md](docs/architecture-notes.md)
- [docs/deviations.md](docs/deviations.md)
- [PARALLELIZATION_ANALYSIS.md](PARALLELIZATION_ANALYSIS.md)
