# Architecture Notes

This file records short implementation notes that supplement, but do not replace, [systemDesign.md](../systemDesign.md).

## Layer boundaries

- `phase_motion_app.app`: PyQt shell, dialogs, shell-side validation, and render supervision
- `phase_motion_app.core`: testable domain logic, models, media helpers, storage rules, diagnostics, sidecars, and numeric processing
- `phase_motion_app.worker`: spawned process entrypoints and the real render worker

If logic can be exercised without Qt and without a child process, it should usually live in `core`.

## Current shape of the render path

- the shell probes, fingerprints, and prepares the request
- the shell launches exactly one spawned worker per render
- the worker performs authoritative pre-flight
- the worker runs a bounded two-pass render pipeline with one codec-safe effective render resolution for both processing and final encode
- optional CuPy acceleration is selected in `core`, not in the Qt layer; Core Settings show capability state and the Pre-flight Report carries explicit GPU-active or CPU-fallback status
- on Windows, `core.acceleration` registers packaged NVIDIA CUDA runtime and NVRTC paths before importing CuPy so optional PyPI-installed GPU support can activate without a separately installed toolkit
- when GPU acceleration is active, scheduler chunk sizing is clamped against free device memory; analysis-enabled downscale runs also clamp chunk size when the richer analysis domain would otherwise make batches too large, and background quantitative-analysis handoff queues small host-side sub-batches to avoid retaining extra device-resident chunks or requiring one large host copy
- the hot accelerated kernels are render-path resize, dense warp/remap, and FFT-based local motion estimation
- quantitative analysis reuses the same motion-estimation backend instead of owning a separate CPU-only variant
- staged output is validated before paired finalization

## Coordinate and reproducibility rules

- mask zones and analysis ROI are stored in source-frame coordinates
- sidecars reload only reusable `intent`
- convenience app state is stored separately from reproducible export metadata and is written through a same-directory temp-file replace

## Toolchain resolution

- `core.toolchain` resolves packaged `ffmpeg` and `ffprobe` by default
- explicit tool overrides are all-or-nothing; set both `PHASE_MOTION_FFMPEG` and `PHASE_MOTION_FFPROBE` together or leave both unset

## Diagnostics

- diagnostics bundle writing happens in the worker
- retention cleanup is budget-based and oldest-first
- watchdog and IPC classification stay in the shell

## Developer references

- [docs/deviations.md](deviations.md) for temporary design mismatches
