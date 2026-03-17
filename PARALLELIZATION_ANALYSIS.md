# Render Pipeline Parallelization Analysis

**Analysis Date:** March 17, 2026  
**Scope:** Current worker/render pipeline and phase-engine concurrency model

---

## Current State

The worker no longer runs the processing pass as one monolithic serial loop.

Current implementation:
- pass 1: bounded reference decode scan
- pass 2: bounded decode -> compute -> encode stage pipeline inside the worker
- bounded internal queues between stage threads
- one compute coordinator thread that keeps chunk order and temporal filter state authoritative
- a reused bounded helper-thread pool inside the phase engine for:
  - row-band motion-grid estimation
  - frame-slice warp work
- optional bounded background quantitative-analysis collection

This is intentionally **bounded and deterministic**, not "parallelize everything".

---

## What Is Parallel Now

### 1. Stage-level worker pipeline

Inside the processing pass, decode, compute, and encode may overlap:

```text
decode-reader thread
  -> bounded decoded queue
compute coordinator thread
  -> bounded encoded queue
encode-writer thread
```

Properties:
- chunk order remains sequential
- queue depth is policy-driven and bounded
- the pipeline remains inside the render worker process

### 2. Warp-stage fan-out

After displacement grids are computed for a chunk, the warp stage splits work into fixed contiguous frame slices.

Properties:
- frames remain independent at this stage
- output ordering is deterministic because each helper owns one fixed slice
- helpers reuse a persistent pool across the render
- cached warp geometry avoids rebuilding interpolation coordinates every frame

### 3. Motion-grid estimation fan-out

Local phase-shift estimation can partition the motion grid by contiguous row bands.

Properties:
- temporal state is not touched here
- each helper writes only to its own row-band slice
- no work stealing is used
- partition order is deterministic

Current policy note:
- the code path exists, but the shipped resource-policy presets currently keep `motion_worker_count = 1`
- this leaves the motion-estimation path sequential in normal renders until more real-clip validation confirms that helper fan-out does not introduce visible instability

### 4. Quantitative-analysis handoff

Quantitative analysis can now receive chunks through a bounded background queue.

Properties:
- the main compute path only pays enqueue/copy cost
- chunk order into analysis remains deterministic
- finalize still waits for queue drain before artifact export
- helper failure is surfaced and routed through the existing warning path

---

## What Must Stay Sequential

The following boundaries remain authoritative and intentionally sequential:
- one active render at a time
- shell process and worker process remain separated
- reference-pass decode remains sequential
- the compute coordinator remains the sole owner of:
  - chunk order
  - streaming temporal filter state
  - final handoff order into the encoder queue
- encode input order remains sequential

These boundaries are required for reproducibility, diagnostics, and watchdog correctness.

---

## Why Threads, Not Nested Worker Processes

The current implementation uses a bounded helper-thread pool inside the worker instead of nested multiprocessing.

Reasons:
- NumPy-heavy FFT and resampling operations already release enough GIL to benefit from helper threads
- threads avoid large pickle traffic for chunk arrays
- cached geometry and reference state stay in-process
- teardown is simpler and more compatible with current watchdog expectations
- the design explicitly avoids nested render-worker fan-out

This keeps the architecture aligned with `systemDesign.md` while still using more cores.

---

## Resource-Policy Semantics

`Conservative`, `Balanced`, and `Aggressive` now drive real runtime parameters.

Each preset controls:
- chunk-size cap
- queue depth
- compute helper count
- warp helper count
- motion-estimation helper count
- quantitative-analysis mode and queue depth

Typical intent:
- `Conservative`: smallest queues and helper counts
- `Balanced`: moderate helper fan-out with bounded buffering
- `Aggressive`: deeper but still bounded buffering and larger helper counts when cores allow

---

## Remaining Bottlenecks

The main hotspots are still:
- local phase-shift estimation
- warp resampling

After the current changes, warp usually benefits the most from added helper concurrency. Motion-grid estimation also improves, but the temporal bandpass itself remains sequential by design.

Quantitative-analysis export after encode is still expensive because final artifact generation is large and intentionally ordered.

---

## Non-Goals

The implementation still does **not** do the following:
- unbounded prefetch
- dynamic work stealing
- nested multiprocessing trees
- cross-chunk temporal-state parallelism
- speculative out-of-order encode submission
- silent dropping of analysis work

---

## Recommended Local Validation

Use the developer perf-smoke tool:

```powershell
python tools/perf_smoke_phase_pipeline.py --frames 32 --width 192 --height 192 --repetitions 3
```

That prints:
- the scheduler plan actually chosen for each policy
- motion-estimation timing
- warp timing
- end-to-end `process_chunk()` timing
- analysis submit-time overhead

This document should be kept in sync with the real implementation. If the worker concurrency model changes again, update this file and `systemDesign.md` together.
