# Overview

This review treated `systemDesign.md` as the architectural source of truth and audited the repository as an engineering tool rather than a demo application.

The codebase started with strong coverage around IPC, watchdog behavior, sidecar validation, and storage/finalization, but it had several material mismatches against the design contract. The most significant gap was memory handling: the worker still decoded and processed the full clip in memory even though the design required bounded chunk execution.

That gap is now closed. The render worker uses a bounded two-pass pipeline, shell and worker preflight both derive the same chunk plan from measured RAM, and long clips are now admitted when a safe batch plan exists. The full suite passes with `160` tests.

# Review Scope

- Reviewed `systemDesign.md` first and used it as the primary architecture contract.
- Audited the project layout under:
  - `src/phase_motion_app/app`
  - `src/phase_motion_app/core`
  - `src/phase_motion_app/worker`
  - `tests`
- Focused extra attention on:
  - shell vs worker separation
  - spawn-only assumptions
  - IPC protocol/session handling
  - watchdog and terminal-state classification
  - `Ready` vs `Fingerprint Pending`
  - stale-source invalidation
  - pre-flight admission checks
  - sidecar schema and reload boundaries
  - staged MP4 validation and paired-output finalization
  - normalization of VFR and non-square-pixel inputs
  - bounded-memory render execution

# Design Mismatches vs systemDesign.md

## Fixed in this review

- The shell UI advanced from `Pre-flight Check` to `Rendering` too early.
  - Worker-side `preflight` activity could move the shell into `Rendering` before the worker actually entered rendering.

- A failed worker bootstrap left the shell stuck in `Pre-flight Check`.
  - Supervisor/start failures did not restore the UI to `Ready`.

- The shell imported the heavy worker implementation during GUI import.
  - `app/main_window.py` imported `worker/render.py` directly, which leaked render-engine imports into the shell process.

- The typed sidecar model dropped `results.output_details`.
  - Schema validation allowed those fields, but the model layer discarded them.

- Staged MP4 validation was stricter than the design contract.
  - The implementation required both frame-count and duration evidence instead of accepting either valid CFR-consistency path.

- Rotated-input ingest was not design-faithful.
  - Probe geometry and raw decode geometry diverged when rotation metadata was present.

- The worker still materialized the full decoded clip and large derived arrays in memory.
  - This was the largest architecture mismatch against Sections 12.5, 16, 18, and 33.7.
  - The worker now uses a bounded two-pass decode/process/encode pipeline instead of a full-clip in-memory path.

- Shell-side and worker-side RAM admission math still modeled the old full-clip engine.
  - Long clips could be rejected purely because of frame count even when a safe bounded chunk plan should have been allowed.
  - Both paths now derive the same bounded scheduler plan from measured RAM and selected resource policy.

## Still mismatched or only partially resolved

- Probe-to-preflight enforcement is still incomplete for some source-format conditions.
  - `SourceMetadata` supports rotation, display-transform, contradictory-color, and decode-format blockers.
  - The current ffprobe-to-`SourceMetadata` path still hardcodes some of those values instead of deriving them all authoritatively.

- The normalized-source path still stages a full derived working file when cadence or pixel geometry must be regularized.
  - This is design-faithful, but it still shifts some large-input cost from RAM to scratch storage and I/O.

# Bugs and Weaknesses Found

- `MainWindow._apply_render_state()` promoted `PREFLIGHT_CHECK` to `RENDERING` on worker preflight snapshots.
- `MainWindow._start_render()` could leave the controller in `PREFLIGHT_CHECK` after a bootstrap/start failure.
- `JobResults.from_dict()` and `JobResults.to_dict()` did not preserve `results.output_details`.
- `validate_staged_mp4()` rejected outputs when one valid CFR-consistency path was present but the other was missing.
- `app/main_window.py` imported the heavy render worker module during GUI import instead of using a lightweight bootstrap layer.
- Raw frame extraction and long-lived rawvideo decode let `ffmpeg` autorotate some inputs, which corrupted previews and would also destabilize render decode for rotated sources.
- Probe metadata was not carrying rotation or sample-aspect-ratio facts into preflight classification.
- The shell and worker both lacked a deterministic normalized-source path for VFR and non-square-pixel inputs.
- The drift editor did not expose an explicit crosshair cursor when rectangle creation was active.
- Last-frame shell preview extraction could fail on normalized VFR sources because the probe-side estimated frame count could land one frame past the actual `fps=` filter output boundary.
- The worker decoded the full clip into RAM before phase processing and then allocated additional full-clip processing arrays.
- Preflight RAM admission mirrored that full-clip assumption, so long clips were blocked even when bounded chunk execution was feasible.

# Tests Run

- Baseline before the deep review work: `pytest -q`
  - Result: `127 passed`
- Targeted regression passes during the review:
  - `pytest -q tests/test_job_state.py tests/test_main_window.py tests/test_storage.py tests/test_sidecar.py tests/test_worker_bootstrap.py tests/test_render_worker.py`
  - `pytest -q tests/test_ffprobe.py tests/test_media_tools.py tests/test_preflight.py tests/test_main_window.py`
  - `pytest -q tests/test_phase_engine.py tests/test_preflight.py tests/test_render_worker.py`
  - `pytest -q tests/test_main_window.py`
- Final full pass: `pytest -q`
  - Result: `160 passed in 5.60s`

# Tests Added

25 tests were added across the review.

- State-machine regressions for `Ready`, `Pre-flight Check`, and launch-failure recovery
- Sidecar preservation/schema coverage for `output_details` and normalization metadata
- Staged MP4 duration-only CFR validation coverage
- Worker bootstrap boundary coverage
- Rotation and sample-aspect-ratio probe coverage
- Autorotation/preview decode regressions and normalized VFR last-frame extraction coverage
- Normalization-plan and ffmpeg normalization-filter coverage
- Crosshair cursor coverage for rectangle creation
- Worker-side normalization metadata coverage
- Streaming phase-engine coverage for chunked static and motion clips
- Bounded scheduler coverage for long clips and tighter RAM budgets
- Worker diagnostics coverage for the new two-pass bounded pipeline

# Fixes Applied

- Added `SingleJobController.abort_preflight()` in `src/phase_motion_app/core/job_state.py`.
- Updated `src/phase_motion_app/app/main_window.py` to:
  - keep the shell in `Pre-flight Check` until the worker actually reports rendering
  - restore shell state if worker bootstrap/start fails
  - launch through a lightweight bootstrap module instead of importing the heavy render module directly
  - carry probe rotation/sample-aspect-ratio facts into shell-side source classification
  - use normalized working source geometry for drift review, mask validation, pre-flight reporting, and resolution options
  - derive shell-side chunk planning from the same bounded scheduler logic used by the worker
- Added `src/phase_motion_app/worker/bootstrap.py` as the shell-safe spawn target and worker launch contract.
- Updated `src/phase_motion_app/core/ffprobe.py` so probe results preserve rotation and sample-aspect-ratio metadata.
- Added `src/phase_motion_app/core/source_normalization.py` so shell/media/worker all derive the same working cadence and square-pixel geometry.
- Updated `src/phase_motion_app/core/media_tools.py` so raw frame extraction and rawvideo decode disable `ffmpeg` autorotation, apply deterministic working-source normalization, and can stage a normalized scratch source.
- Updated `src/phase_motion_app/core/baseline_band.py` so baseline analysis uses the normalized working representation.
- Updated `src/phase_motion_app/core/models.py` and `src/phase_motion_app/core/sidecar.py` so sidecars preserve `output_details` and normalization-aware preflight metadata.
- Updated `src/phase_motion_app/core/preflight.py` to:
  - admit VFR and non-square-pixel inputs through deterministic normalization warnings instead of hard rejection
  - derive bounded scheduler inputs from measured RAM and resource policy
  - estimate RAM against the bounded batch pipeline instead of full-clip reconstruction
- Updated `src/phase_motion_app/core/storage.py` so staged MP4 CFR validation accepts either valid frame-count evidence or valid duration evidence.
- Updated `src/phase_motion_app/core/phase_engine.py` to add a `StreamingPhaseAmplifier` that preserves a global reference frame and temporal filter state across bounded chunks.
- Updated `src/phase_motion_app/worker/render.py` to:
  - use the shared bounded scheduler plan
  - run a bounded reference-scan decode pass
  - re-decode in bounded batches for phase processing
  - stream amplify/composite output directly into the encoder without holding the whole clip in RAM
  - record two-pass bounded-pipeline decisions in diagnostics
- Updated `systemDesign.md` to formalize the bounded-memory contract and allow sequential bounded decode passes when global reference statistics are required.

# Files Changed

- `codeReviewFindings.md`
- `systemDesign.md`
- `src/phase_motion_app/app/main_window.py`
- `src/phase_motion_app/core/baseline_band.py`
- `src/phase_motion_app/core/ffprobe.py`
- `src/phase_motion_app/core/job_state.py`
- `src/phase_motion_app/core/media_tools.py`
- `src/phase_motion_app/core/models.py`
- `src/phase_motion_app/core/phase_engine.py`
- `src/phase_motion_app/core/preflight.py`
- `src/phase_motion_app/core/sidecar.py`
- `src/phase_motion_app/core/source_normalization.py`
- `src/phase_motion_app/core/storage.py`
- `src/phase_motion_app/worker/bootstrap.py`
- `src/phase_motion_app/worker/render.py`
- `tests/test_ffprobe.py`
- `tests/test_drift_editor.py`
- `tests/test_job_state.py`
- `tests/test_main_window.py`
- `tests/test_media_tools.py`
- `tests/test_phase_engine.py`
- `tests/test_preflight.py`
- `tests/test_render_worker.py`
- `tests/test_sidecar.py`
- `tests/test_source_normalization.py`
- `tests/test_storage.py`
- `tests/test_worker_bootstrap.py`

# Remaining Risks

- The bounded scheduler is still heuristic.
  - It is materially better than the old full-clip admission model, but Python/NumPy memory behavior is still approximate rather than exact.

- The new streaming phase path is operationally safer than the old full-clip path, but it is not numerically identical to the previous whole-signal FFT-style surrogate.
  - That is an acceptable trade for bounded execution, but it remains an engineering tradeoff worth documenting.

- Automatic normalization still depends on scratch/storage headroom.
  - Large VFR or non-square-pixel sources may now pass RAM admission but still fail active-scratch admission if normalization staging cannot be reserved safely.

- Probe-to-policy coverage is still incomplete for some source-format blockers.
  - Display-transform, contradictory-color, and unsupported-decoder signals still need more authoritative probe integration.

# Open Issues

- Extend authoritative source probing so pre-flight can enforce the remaining blocked-input rules rather than relying on hardcoded defaults.
- Consider adding more integration tests around forced low-RAM scheduler choices in the spawned worker path, not just preflight math and diagnostics output.
- Consider recording observed peak batch memory in diagnostics when a portable cross-platform method is available.
- Consider tightening the streaming temporal filter characterization with more synthetic signal tests if output interpretability becomes a higher-stakes requirement.

# Final Assessment

Implementation quality is solid and materially better aligned with the architecture than it was at the start of the review. The shell/worker boundary is cleaner, sidecar fidelity is better, normalization is deterministic, staged-output validation matches the contract, and the render path no longer assumes the whole clip fits in RAM.

The largest prior architecture risk was the full-clip in-memory worker. That is now replaced with a bounded two-pass pipeline and matching RAM admission logic, which is the most important improvement from this review. The main remaining risks are heuristic rather than structural: scheduler estimation accuracy, scratch-heavy normalization on some inputs, and incomplete probe-to-policy enforcement for a few source-format blockers.
