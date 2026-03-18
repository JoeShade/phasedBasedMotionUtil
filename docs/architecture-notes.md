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
- the worker runs a bounded two-pass render pipeline
- staged output is validated before paired finalization

## Coordinate and reproducibility rules

- mask zones and analysis ROI are stored in source-frame coordinates
- sidecars reload only reusable `intent`
- convenience app state is stored separately from reproducible export metadata

## Diagnostics

- diagnostics bundle writing happens in the worker
- retention cleanup is budget-based and oldest-first
- watchdog and IPC classification stay in the shell

## Developer references

- [docs/deviations.md](deviations.md) for temporary design mismatches
- [PARALLELIZATION_ANALYSIS.md](../PARALLELIZATION_ANALYSIS.md) for current worker/phase-engine concurrency notes
