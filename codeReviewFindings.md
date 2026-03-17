# Overview

This review treated `systemDesign.md` as the architectural source of truth and audited the repository as an offline engineering tool rather than a demo application.

The codebase already had strong coverage around IPC, terminal-outcome rules, staged MP4 validation, watchdog behavior, and most of the sidecar boundary. The main defects found in this pass were narrower but still important architecture issues:

- stale-source handling in the shell invalidated `Ready` but left stale probe/fingerprint context in place
- the watchdog could wait forever if a worker acknowledged the session and then went silent before its first heartbeat
- sidecar validation allowed normalized/VFR runs to omit required working-source metadata
- pre-flight retention warnings still used output-staging bytes as if they were retention evidence

All four defects were reproduced with new tests, fixed in small patches, and verified with the full suite.

# Review Scope

- `systemDesign.md`
- `src/phase_motion_app/app`
- `src/phase_motion_app/core`
- `src/phase_motion_app/worker`
- `tests`

Review emphasis:

- shell vs worker separation
- spawn-only assumptions
- IPC handshake, session binding, and terminal-message rules
- watchdog and silent-worker classification
- `Ready` vs `Fingerprint Pending`
- stale-source detection and recovery
- sidecar schema, compatibility, and reload boundaries
- pre-flight disk/RAM/retention separation
- storage/finalization and lone-MP4 handling
- CFR/VFR and normalized-working-source rules
- diagnostics-cap behavior

# Design Mismatches vs systemDesign.md

## Fixed in this review

- `main_window.py` only downgraded the shell from `Ready` to `Loaded` when the source changed on disk. It did not restart the authoritative probe/fingerprint/drift-review flow, which left stale source metadata visible after readiness was invalidated.
- `watchdog.py` only enforced heartbeat silence after at least one heartbeat had already arrived. A worker that completed the socket handshake and then hung could stay non-terminal indefinitely, which violated the watchdog policy in Sections 6 and 7.
- `sidecar.py` validated the general schema shape but did not enforce normalization metadata when a derived working source was used. That violated Sections 22 and 23.
- `preflight.py` used output-staging bytes when deciding whether to warn about the retention budget. That mixed the active-scratch bucket with the retention bucket in a way Section 12.2 / 25 explicitly forbids.

## Remaining partial mismatches

- Retention-budget warning logic is now separated from output staging, but it is still a coarse estimate. It mainly predicts future diagnostics pressure, not every possible retained failed-run artifact shape.
- The repository-level `docs/deviations.md` appears stale relative to the current worker implementation. The code now uses bounded sequential decode/process passes, so that document should be reconciled separately.

# Bugs and Weaknesses Found

- Fixed: stale-source invalidation in the shell left stale probe/fingerprint context in memory and did not restart authoritative source checks.
- Fixed: a post-handshake silent worker could evade both heartbeat timeout and stall classification forever.
- Fixed: sidecars for normalized/VFR runs could be accepted without `working_fps`, `working_source_resolution`, or `normalization_steps`.
- Fixed: retention warnings could fire purely because the output staging allowance was large, even when the retained-artifact budget itself was still acceptable.

Additional review observations:

- IPC/session binding, strict sequence handling, terminal success rules, cancellation handling, staged MP4 validation, lone-MP4 quarantine handling, diagnostics cap fallback order, and sidecar intent reload boundaries are already in good shape and are well covered by tests.
- The shell/worker split and spawn-only worker startup are implemented in a design-faithful way.

# Tests Run

- Baseline existing suite before changes: `pytest -q`
  - Result: `165 passed`
- Targeted regression pass after new tests and fixes: `pytest -q tests/test_preflight.py tests/test_sidecar.py tests/test_render_supervisor.py tests/test_main_window.py`
  - Result: `74 passed`
- Final full pass: `pytest -q`
  - Result: `169 passed in 7.30s`

# Tests Added

Four regression tests were added:

- `tests/test_preflight.py`
  - `test_preflight_does_not_treat_output_staging_as_retention_evidence`
- `tests/test_sidecar.py`
  - `test_sidecar_schema_validation_rejects_missing_normalization_metadata`
- `tests/test_render_supervisor.py`
  - `test_render_supervisor_classifies_post_handshake_silence_as_unresponsive`
- `tests/test_main_window.py`
  - `test_main_window_restarts_probe_and_fingerprint_when_source_goes_stale`

# Fixes Applied

- `src/phase_motion_app/app/main_window.py`
  - stale-source polling now clears stale authoritative source state and restarts probe, fingerprint, and frame extraction for the updated file instead of leaving the shell on stale context
- `src/phase_motion_app/core/watchdog.py`
  - added shell-local telemetry anchoring so a connected worker that never reaches its first heartbeat still times out
- `src/phase_motion_app/core/render_supervisor.py`
  - the supervisor now records handshake receipt as watchdog telemetry, which closes the post-handshake silent-worker gap
- `src/phase_motion_app/core/sidecar.py`
  - added semantic validation requiring normalization metadata when a derived working source was used
- `src/phase_motion_app/core/preflight.py`
  - retention warnings now use a retention-domain diagnostics allowance instead of output-staging bytes

# Files Changed

Files changed in this review pass:

- `codeReviewFindings.md`
- `src/phase_motion_app/app/main_window.py`
- `src/phase_motion_app/core/preflight.py`
- `src/phase_motion_app/core/render_supervisor.py`
- `src/phase_motion_app/core/sidecar.py`
- `src/phase_motion_app/core/watchdog.py`
- `tests/test_main_window.py`
- `tests/test_preflight.py`
- `tests/test_render_supervisor.py`
- `tests/test_sidecar.py`

Note:

- The worktree contained other pre-existing modified and untracked files before this pass. They were reviewed where relevant, but they were not altered by these fixes.

# Remaining Risks

- Retention forecasting is still approximate. The current warning logic is now bucket-correct, but it does not attempt to predict every possible retained failed-run partial.
- Some of the most important paths still rely more on unit/integration-style harnesses than on real-media end-to-end tests. That is acceptable for now, but the highest-risk operational paths remain the spawned worker plus real ffmpeg toolchain interactions.
- `docs/deviations.md` is likely no longer accurate and could mislead future maintenance unless it is updated.

# Open Issues

- Reconcile or replace `docs/deviations.md` so it matches the current implementation and does not contradict the actual worker pipeline.
- Add more real-media integration coverage around normalized working-source renders and worker failure recovery if test assets/practical runtime allow it.
- Consider a richer retention forecast model if operators rely heavily on retained failed-run evidence and large diagnostics bundles.

# Final Assessment

The repository is in generally solid shape and already aligned with most of `systemDesign.md` in the highest-risk areas. The shell/worker boundary, spawn-only process model, IPC contract, watchdog classifications, storage/finalization rules, and sidecar intent boundary are materially stronger than a typical desktop utility at this stage.

This review found four concrete defects that mattered to architectural correctness, and all four were fixed with direct regression coverage. After those fixes, the suite is green at `169 passed`, and the remaining concerns are mostly around documentation drift and additional end-to-end coverage rather than obvious logic or state-machine breaks.
