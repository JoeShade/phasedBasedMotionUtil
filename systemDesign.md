# System Design: Phase-based Motion Amplification Desktop Utility

## 1. Purpose

This document describes the current repository design for the Phase-based Motion Amplification Desktop Utility.

The product is an offline PyQt6 desktop application for engineering review of recorded video. It is intentionally narrow:

- phase-based amplification only
- offline processing only
- one active render at a time
- fixed-camera workflow with static mask geometry
- bounded, supervised worker execution rather than in-process rendering

This document should describe what the repository actually implements today. If code and design diverge, update one of them in the same change.

## 2. Product Scope

### 2.1 Implemented workflow

The current application supports this operator flow:

1. Choose a recorded video source.
2. Run a fast source probe and a canonical SHA-256 fingerprint in the shell.
3. Review the first/last-frame drift editor and define static mask zones in source-frame coordinates.
4. Optionally define one quantitative-analysis ROI and analysis-band behavior.
5. Run shell-side dry-run pre-flight.
6. Launch a spawned worker that reruns authoritative pre-flight and performs the render.
7. Finalize a paired MP4 plus JSON sidecar and write diagnostics artifacts.

### 2.2 Major capabilities

- source probe through `ffprobe`
- source fingerprinting and stale-source detection
- drift acknowledgement workflow
- static include/exclude mask zones with feathering
- one codec-safe effective render resolution used for both phase processing and final encode
- optional CuPy-backed hardware acceleration for dense warp, render-path resize, and FFT-based local motion estimation with explicit CPU fallback
- resource-policy-driven scheduler selection
- mandatory scratch/output/RAM pre-flight admission
- worker IPC handshake, message validation, and watchdog classification
- phase-based amplification in a separate worker process
- render-time quantitative-analysis artifact export
- diagnostics bundle writing and retained-evidence cleanup
- safe sidecar validation and reusable-intent reload

### 2.3 Explicit non-goals

The current repository does not implement:

- live capture or live preview
- full amplified preview playback in the shell
- batch queues or concurrent renders
- arbitrary video editing
- tracked masks or moving ROIs
- upscaled output
- a separate operator-controlled output resize distinct from the processing resolution
- cloud execution, remote telemetry, or distributed processing
- analysis-only runs without a render

## 3. Repository Architecture

The repository is split into three runtime layers plus tests.

### 3.1 `phase_motion_app.app`

The `app` package owns PyQt shell behavior:

- window construction and UI state
- source selection
- source probe and fingerprint worker threads
- drift/mask/ROI dialogs
- shell-side dry-run pre-flight
- render launch and supervision
- display of progress, warnings, terminal outcomes, and cleanup actions
- the Core Settings hardware-acceleration control plus capability messaging, with final GPU-active or CPU-fallback reporting in the pre-flight report
- convenience persistence that is explicitly separate from reproducible sidecar intent

The shell must not perform the heavy render pipeline. Long-running render work belongs in the worker.

### 3.2 `phase_motion_app.core`

The `core` package owns repository-wide rules and models that are testable without Qt:

- shared domain models
- sidecar schema validation and reload boundaries
- pre-flight logic and scheduler selection
- masking and drift logic
- media helper abstractions around `ffmpeg`/`ffprobe`
- optional acceleration capability detection and CPU/GPU backend selection
- storage/finalization policy
- diagnostics and retention policy
- watchdog and IPC validation
- quantitative-analysis logic
- phase-processing engine code

If logic can be unit tested without Qt and without a spawned worker, it should normally live here.

### 3.3 `phase_motion_app.worker`

The `worker` package owns the spawned child-process entrypoints:

- lightweight bootstrap contract used by the shell
- render worker process implementation
- test-only worker scaffold used to exercise IPC and watchdog behavior without invoking the real render engine

Heavy imports are intentionally delayed until the spawned child starts so the shell remains lightweight.

### 3.4 `tests`

The `tests` tree is the authoritative regression net for:

- pure core logic
- shell state behavior
- worker supervision and IPC
- render-worker integration with synthetic inputs

The suite is mostly unit and integration style. Real-media fixture coverage is intentionally limited to keep the repository small and deterministic.

## 4. Runtime Design

### 4.1 Source loading and authoritative identity

When a source is chosen, the shell starts three lightweight background tasks:

- fast `ffprobe` metadata collection
- SHA-256 fingerprinting
- first/last-frame extraction for drift review

The shell tracks a cheap snapshot of source path, size, and modification time. If the selected file changes or disappears, authoritative readiness is cleared and probe/fingerprint/drift review are restarted or invalidated as needed.

### 4.2 Drift and masking

Mask zones and the optional analysis ROI are defined in source-frame coordinates. The same geometry is serialized into sidecars and later scaled into the effective render domain inside worker/core code.

The drift editor compares first and last frames. When drift exceeds the warning threshold, render remains blocked until the operator acknowledges the reviewed source state.

### 4.3 Pre-flight

Pre-flight is mandatory in two places:

- shell-side dry-run pre-flight for operator feedback before launch
- worker-side authoritative pre-flight against the live render request

Current pre-flight gates include:

- output container policy
- processing/output resolution equality enforcement
- even output dimensions for the current encoder path
- requested hardware-acceleration availability reporting with warning-plus-fallback behavior when CuPy/CUDA cannot be used
- GPU-aware chunk sizing that clamps the optional accelerated path against currently available device memory before phase processing begins
- frequency-band sanity versus Nyquist
- source normalization warnings
- color/rotation/display-transform blockers
- scratch, output-volume, retention-budget, and RAM admission
- resource-policy-driven scheduler selection

### 4.4 Worker process isolation

The render worker runs in a separate spawned process. The shell and worker communicate over a loopback socket using newline-delimited JSON plus a shared cancellation event.

The shell validates:

- `hello` / `hello_ack` handshake
- session token and job binding
- strict sequence ordering
- terminal-message rules
- process liveness and exit-code agreement

The watchdog treats missing heartbeats, missing progress, and terminal-state disagreement as separate failure classes.

### 4.5 Render pipeline

The current worker pipeline is a bounded two-pass design:

1. Probe and normalize the source when cadence or pixel geometry require it.
2. Run a bounded reference decode pass to build the phase-processing reference state.
3. Run a bounded decode -> compute -> encode pipeline for the render pass.
4. Optionally hand chunks to background quantitative-analysis accumulation through a bounded queue.
5. Validate the staged MP4.
6. Commit the final MP4 and JSON sidecar as a pair.

The pipeline is intentionally deterministic. One codec-safe effective render resolution is used for both phase processing and the final MP4 output, so the normal render path does not perform a second processing-to-output resize. When the optional CuPy backend is available and the operator enables it, dense warp, render-path resize, and FFT-based local motion estimation run on the GPU while CPU fallback remains authoritative. The scheduler clamps GPU chunk sizing against available device memory, clamps analysis-enabled downscale runs when the richer analysis domain would otherwise make chunks too large, and background quantitative-analysis handoff stores bounded host-side sub-batches so optional analysis does not retain extra device-resident frame batches or require one large host copy.

### 4.6 Output finalization

The repository follows a paired-output rule:

- a visible MP4 is not considered complete until the matching sidecar is committed
- staged MP4 validation must pass before final rename
- lone-MP4 failure paths are quarantined or explicitly marked incomplete

This keeps successful output claims aligned with reproducible metadata.

## 5. Data Model Boundaries

### 5.1 Sidecar domains

The sidecar model has three top-level domains:

- `intent`: reproducible operator intent that can be reloaded later
- `observed_environment`: machine- and runtime-contingent facts recorded for review only
- `results`: run outputs, warnings, artifact paths, and pre-flight evidence

Only reusable intent is loaded back into the shell. Observed environment and prior results are never silently treated as future defaults.

### 5.2 Convenience persistence

Separate from sidecars, the shell stores last-used settings and machine-local preferences in `~/.phase_motion_app/settings.json` by default. This state exists for convenience only and is not part of the reproducible export contract.

### 5.3 Runtime paths

When running from a source checkout, the app defaults to repo-local runtime directories:

- `input/`
- `output/`
- `temp/`
- `diagnostics/`

Diagnostics are written under the configured diagnostics root. The convenience settings file remains under `~/.phase_motion_app/` unless explicitly overridden.

## 6. Quantitative Analysis

Quantitative analysis is part of the implemented repository, not a future design stub.

Current design rules:

- analysis runs only alongside a render
- analysis uses the same configured frequency range as the render
- one optional ROI is supported
- whole-frame-minus-mask fallback is supported when no ROI is drawn
- artifact export is derived from the internal motion-analysis path, not from the encoded MP4
- when hardware acceleration is active, analysis reuses the same local motion-estimation backend rather than reintroducing a separate CPU-only kernel path

Current exported analysis behavior includes ROI spectra, quality scoring, auto/manual band handling, and heatmap-oriented artifacts driven by the same render-time motion data used for amplification.

## 7. Scheduling and Resource Policies

The user-facing resource policies map to real runtime scheduler inputs:

- `conservative`
- `balanced`
- `aggressive`

The scheduler chooses bounded chunk size, helper counts, queue depth, and analysis handoff behavior from the current machine budget. The repository intentionally avoids unbounded buffering, nested worker-process trees, and speculative out-of-order execution.

## 8. Diagnostics and Retention

Every run can produce diagnostics material, and failure paths are expected to leave reviewable evidence when possible.

Current repository rules:

- diagnostics are written by the worker
- diagnostics bundle generation is capped by configured size policy
- retained evidence is measured separately from active scratch admission
- oldest-first purge planning is used to bring retained evidence back under budget

## 9. Testing Expectations

This repository relies on targeted unit and integration-style tests as the primary regression mechanism.

Expectations for repository changes:

- add or update tests with substantive behavior changes
- prefer regression tests for bug fixes
- run targeted tests while iterating
- run the full suite before finalizing

## 10. Temporary Deviations

Temporary code/design mismatches, when they exist, should be tracked in `docs/deviations.md`.

As of this document revision, `docs/deviations.md` should remain short and only list live, intentional divergences. Historical review notes do not belong there.
