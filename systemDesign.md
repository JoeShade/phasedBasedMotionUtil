# System Design: Phase-Based Motion Amplification Desktop Application

## 1. Purpose and Context

This document captures the current agreed system design for a redesign of the existing video magnification stack into a **Python + PyQt6 desktop application**.

The application is intended for **offline engineering use**, with a focus on **phase-based motion amplification** for vibration visualization and corroboration in an NVH-style workflow. It is explicitly **not** intended to be a browser application, a live processing system, or a general-purpose media editor.

The design incorporates the following key directions:

- Desktop UI using **PyQt6**
- **Phase-based amplification only**
- **Offline video processing only**
- **Single active render** at a time
- **Instrument-style** UX
- **No live mode**
- **No full preview mode**
- Static mask zones with a **First vs Last frame drift check**
- **Downscale-only** output policy
- Automatic, resource-aware scheduling instead of manual core pinning
- Mandatory **pre-flight checks**
- Explicit **out-of-memory handling**
- Diagnostics and reproducibility features built in from the start
- Future-friendly design for a **portable program**

---

## 2. Product Goals

### 2.1 Primary Goals

- Provide a stable desktop workflow for running phase-based motion amplification on recorded videos.
- Allow engineers to adjust the key signal-processing and export parameters through a GUI.
- Support fixed-camera workflows with static mask zones so operators can either protect irrelevant regions or deliberately constrain amplification to an inclusion area.
- Export quantitative NVH-style analysis artifacts during render so later engineering review can compare ROI spectra, heatmaps, and run metadata outside the app.
- Preserve engineering integrity by preventing misleading output choices, especially upscaling and invalid temporal frequency selections.
- Provide useful operational visibility through progress, stage reporting, diagnostics, and reproducibility metadata.
- Fail safely and transparently under adverse conditions such as insufficient RAM, invalid frequency settings, corrupted sidecars, or worker-process failure.

### 2.2 Secondary Goals

- Support future packaging as a portable application.
- Keep the processing engine UI-agnostic so it can evolve independently of the desktop shell.
- Preserve enough structured metadata to support repeatability and future automation.

---

## 3. Explicit Non-Goals

The following are out of scope for the current design:

- Live webcam or live video-stream processing
- React, Flask, browser-based UI, or socket-driven web stack behavior
- Real-time preview of the full amplification effect
- Batch queues or multiple concurrent render jobs
- Arbitrary media editing workflows
- Automatically tracked or inferred ROI mode
- Tracked masks, object tracking, or frame-varying masks
- Upscaling exports
- Full preset-library UI in the first release
- Remote telemetry, cloud execution, or distributed scheduling

---

## 4. Core Product Decisions

### 4.1 Single Active Render

The application supports exactly **one active render job at a time**.

Rationale:
- Simplifies state management
- Simplifies cancellation and failure handling
- Keeps the UI instrument-like rather than queue-oriented
- Avoids unnecessary job orchestration complexity

### 4.2 Instrument-Style Desktop UX

The application should feel like an engineering instrument rather than a media editor.

Characteristics:
- Clean main screen
- Strong defaults
- Advanced settings hidden by default
- Deterministic workflows
- Explicit validation and diagnostics

### 4.3 Phase-Based Amplification Only

The application does not expose alternative amplification modes.

Rationale:
- Keeps scope narrow
- Aligns with the preferred method for the target use case
- Avoids carrying forward architectural complexity from the old stack

### 4.4 Downscale-Only Export Policy

The user may choose an output resolution equal to the processing resolution or smaller, but **never larger**.

Rationale:
- Prevents users from visually magnifying artifacts via upscaling
- Preserves engineering trustworthiness

UX requirement:
- The UI must explicitly explain why upscaling is disabled.
- Suggested wording: **"Upscaling is disabled to avoid inventing apparent motion detail or amplifying visual artefacts beyond the processed data."**

### 4.5 MP4 Only, Audio Stripped

- Output container: **MP4**
- Audio: **always stripped**

This is a container policy, not a complete codec policy. Codec, bit depth, and chroma requirements are defined separately in Section 17.

---

## 5. High-Level Architecture

The system is divided into two main runtime domains:

1. **PyQt6 Desktop Shell**
2. **Separate Worker Process** running the render engine

### 5.1 Desktop Shell Responsibilities

The PyQt6 shell is responsible for:
- Main window and UI state management
- Input file selection
- Settings editing
- Drift-check and exclusion-zone editor
- Triggering pre-flight and render jobs
- Displaying progress, stage, logs, and failures
- Loading settings from metadata sidecars
- Exposing diagnostics and cleanup controls

### 5.2 Worker Process Responsibilities

The worker process is responsible for:
- Pre-flight validation calculations
- Automatic resource scheduling
- Deterministic source normalization when cadence or pixel geometry must be regularized before processing
- Video decode
- Phase-based processing
- Quantitative analysis accumulation and artifact export
- Exclusion-zone compositing
- Video encode
- Diagnostics output
- Temporary-file management during execution
- Structured telemetry back to the shell

### 5.3 Mandatory Process Isolation

The render engine **must not** run in-process with the PyQt6 GUI.

Rationale:
- Prevents the GUI from freezing under heavy computation
- Avoids GIL-related responsiveness problems
- Improves crash isolation
- Makes worker death detection tractable

### 5.4 Process Start Model

V1 uses the **`spawn`** multiprocessing start method on all supported platforms.

Rationale:
- avoids Qt/fork interaction pathologies
- keeps worker bootstrap deterministic across Windows/macOS/Linux
- reduces hidden inherited state in probe/render workers

---

## 6. IPC Strategy

The shell and worker communicate through explicit, structured IPC. The design intentionally avoids using a plain high-volume `multiprocessing.Queue` as the sole telemetry channel because queue-backed pipes are vulnerable to blockage and awkward failure modes when a worker dies abruptly.

### 6.1 Required IPC Primitives

Primary IPC primitives:
- **Local loopback socket** for structured telemetry, handshakes, warnings, progress, and terminal status
- **`multiprocessing.Event`** for cooperative cancellation
- **OS-level process polling** using the parent-held `multiprocessing.Process` handle for liveness and exit-code inspection

Portability note:
- The transport abstraction should support loopback TCP on all supported platforms.
- Platform-specific transports such as Unix domain sockets may be added later, but the architecture must not depend on them.

### 6.2 Protocol Contract

The socket channel carries newline-delimited JSON messages. This is a protocol, not just a transport.

Required protocol rules:
- Every session begins with a shell-issued `hello` containing `protocol_version`, `session_token`, `job_id`, and an expected worker role (`probe` or `render`).
- The worker must respond with `hello_ack` containing the same `protocol_version`, the echoed `session_token`, its role, and its process ID.
- If protocol version or session token do not match, the shell must reject the session.
- Every subsequent message must include: `protocol_version`, `session_token`, `seq`, `message_type`, `job_id`, monotonic timestamp, and payload.
- `seq` must be strictly increasing per session.
- Worker-reported monotonic timestamps are diagnostic metadata only. Shell timeout and liveness decisions must use local receipt time, progress-token logic, authoritative child progress, and OS-level process state rather than trusting worker clocks across processes.
- Message size should remain bounded; routine telemetry messages should stay small and high-volume diagnostics must go to JSONL logs on disk rather than IPC.

### 6.3 Terminal Message Rule

Exactly one terminal IPC message is authoritative at the protocol level:
- `job_completed`
- `failure`
- `job_cancelled`

Rules:
- A successful job requires **both** a terminal `job_completed` message **and** worker exit code `0`.
- If exit code is `0` but no terminal success message arrives, classify as IPC/worker protocol failure.
- If a terminal success message arrives but exit code is non-zero, classify as worker termination after claimed success.
- Abrupt socket close without a terminal message is a failure unless cooperative cancellation is already in progress and confirmed.
- `artifact_paths` may appear before the terminal message but are not by themselves proof of successful completion.

### 6.4 Message Classes

Required message classes include:
- `job_started`
- `preflight_started`
- `preflight_report`
- `stage_started`
- `progress_update`
- `heartbeat`
- `warning`
- `artifact_paths`
- terminal messages defined above

High-volume diagnostic logs must be written by the worker directly to JSONL log files rather than streamed exhaustively over IPC.

### 6.5 Cancellation

The worker must periodically check the shared cancellation event and terminate cooperatively.

Cancellation must be safe during:
- pre-flight
- decode
- chunk processing
- compositing
- encode

### 6.6 IPC Reliability Policy

The shell must treat socket silence and process death as separate signals:
- missing heartbeats imply a liveness problem, not immediate death
- a non-alive worker process implies immediate failure classification
- the UI must not block waiting indefinitely on any IPC read
- disagreement between terminal IPC state and exit code must be classified explicitly

## 7. Worker Heartbeat and Watchdog

The Job Controller must include a heartbeat/watchdog mechanism to detect silent worker failures.

### 7.1 Worker Heartbeat

The worker process emits lightweight `heartbeat` messages at a fixed interval, even when progress has not changed.

### 7.2 Job Controller Watchdog

The shell tracks:
- last heartbeat timestamp
- last telemetry timestamp
- OS-level worker status via `process.is_alive()` and `process.exitcode`

It applies two thresholds:

- **Soft timeout**: logs a warning that the worker is unresponsive
- **Hard timeout**: transitions the job to a classified failed state

### 7.3 Silent Death Behavior

If the worker stops responding or dies unexpectedly:
- the job is marked failed
- the UI shows a specific failure reason
- partial output and temp files are retained per policy
- cleanup actions are presented to the user

### 7.4 Polling Guidance

The Job Controller should run a timer loop that:
- checks `process.is_alive()`
- inspects `process.exitcode` when applicable
- services socket reads without blocking the GUI event loop
- escalates cleanly when liveness and heartbeat signals disagree

---

## 8. UI State Model

The application follows this state model:

**Idle → Loaded → Fingerprint Pending / Drift Check / Mask Editing → Ready → Pre-flight Check → Rendering → Complete / Failed / Cancelled**

`Fingerprint Pending` is a first-class UI/runtime substate. It may overlap the user’s editing flow after fast probe results arrive, but `Ready` is not reachable until canonical fingerprinting has completed successfully and the source has not gone stale.

### 8.1 Idle
- No input loaded

### 8.2 Loaded
- Fast probe metadata available
- Settings editable
- Exclusion zones not yet necessarily defined
- Shell monitors obvious source-file changes (size / mtime) and must invalidate current readiness if the source appears stale before render

### 8.3 Fingerprint Pending / Drift Check / Mask Editing
- Canonical fingerprint may still be running
- First / Last frame inspection remains available
- Exclusion-zone creation and editing remains available
- Start Render stays disabled while canonical fingerprinting is pending

### 8.4 Ready
- All required settings complete
- Canonical fingerprint completed
- Source still matches the probed source state
- Ready to launch authoritative pre-flight and render

### 8.5 Pre-flight Check
- Resource and signal sanity validation

### 8.6 Rendering
- Active worker process
- Progress and telemetry visible

### 8.7 Complete / Failed / Cancelled
- Terminal run outcome screen
- Includes cleanup and reproducibility actions

---

## 9. Main UI Layout

The main setup screen should contain:

- Source video path
- Output folder and output name
- Source metadata
- Core phase settings
- Resource policy selector
- Processing resolution selector
- Output resolution selector
- Exclusion-zone editor launcher
- Analysis section with ROI and band controls
- Advanced settings drawer
- Diagnostics drawer
- Start Render action

### 9.1 Default Visible Controls

These should be visible in the default view:

- Magnification factor
- Low frequency cutoff
- High frequency cutoff
- Pyramid type
- Sigma
- Attenuate other frequencies
- Resource policy
- Processing resolution
- Output resolution
- Exclusion-zone editor button
- Quantitative-analysis enable toggle
- Quantitative-analysis ROI summary and editor button
- Quantitative-analysis band mode selector
- Load from Export Metadata button

### 9.2 Hidden Under Advanced

Advanced settings should include, as needed:
- Diagnostic level
- Temp retention budget mode and value
- Log options
- Dry-run validation trigger
- Internal precision, if exposed at all
- Optional feather width for exclusion-mask compositing
- Quantitative-analysis thresholds, auto-band count, manual bands, and internal-export toggle

---

## 10. Resource Policy and Scheduling

Manual core pinning has been removed.

### 10.1 User-Facing Resource Policy

The UI exposes:
- **Conservative**
- **Balanced**
- **Aggressive**

### 10.2 Internal Scheduler Responsibilities

The worker scheduler chooses execution parameters automatically based on:
- available RAM
- processing resolution
- frame count / duration
- active scratch-space headroom
- codec/output path requirements
- measured machine topology information available at startup

It determines:
- native-library thread limits
- chunk size
- memory safety margin
- staging plan for intermediate data

### 10.3 Aggressive Policy Guardrails

`Aggressive` must still preserve shell responsiveness.

 Rules:
 - Dynamically schedule up to `N-1` logical cores for aggressive policy, based on available CPU cores at runtime
 - Dynamically increase chunk size for aggressive policy if RAM allows, up to a safe maximum (e.g., 256 frames)
 - Reserve mandatory UI headroom
 - Reserve memory margin
 - Prefer completion stability over theoretical maximum throughput
 - Monitor actual CPU and RAM usage during render and adjust chunk size or thread limit if underutilized (future enhancement)

### 10.4 Authoritative Concurrency Model

V1 concurrency is intentionally conservative:
- one render worker process per attempt
- one main stage-execution thread inside the worker
- one dedicated control/heartbeat thread inside the worker
- no nested Python multiprocessing inside the render worker
- ffmpeg subprocesses for decode and encode as needed
- bounded native-library thread pools for BLAS/OpenMP/OpenCV/codec-related libraries

The scheduler may change thread-pool limits, but the set of authoritative participants in the resource model is fixed to the above categories.

### 10.5 Scheduler Bias

The scheduler should favor **successful completion and system stability** over peak throughput.

### 10.6 No Mid-Render Recovery Promise

The scheduler may choose conservative execution parameters at job start, but it must **not** promise graceful in-place recovery after a true out-of-memory event. Memory protection strategy is handled by hard-stop rules in Section 18.

## 11. Global vs Job Settings

The architecture distinguishes between **global application preferences** and **per-job settings**.

### 11.1 Global Preferences

Global preferences are not written into sidecar `intent` as authoritative re-run parameters unless they directly affect the produced output.

Examples:
- temp retention budget mode and value
- temp root location
- diagnostics storage root
- default diagnostics verbosity
- UI theme/state

Diagnostics verbosity is operational context. It may be persisted as convenience state or recorded in `observed_environment`, but it is not reproducible engineering intent and must not be restored as job intent by metadata reload.

### 11.2 Per-Job Settings

Per-job settings must be captured in sidecar `intent` when they can affect reproducibility or interpretation.

Examples:
- phase settings
- processing resolution
- output resolution
- mask zones
- resource policy
- output codec/profile selection

---

## 12. Pre-flight Check Phase

Pre-flight is mandatory before render.

It validates:
1. Active scratch-space requirement
2. Input FPS vs target frequency band
3. RAM admission feasibility
4. Constant-frame-rate suitability
5. Input/output codec compatibility requirements
6. Color / display compatibility requirements

### 12.1 Active Scratch Reservation

Pre-flight estimates the active scratch-space requirement based on:
- frame count
- processing resolution
- precision
- chunking plan
- intermediate storage format
- diagnostics level
- staged output requirements

It compares that estimate against:
- currently available disk space on the chosen scratch volume
- a hard minimum free-space floor after reservation
- the target output-volume requirement for staged finalization

This is a **hard execution gate**. If active scratch cannot be reserved safely, the run must not start.

### 12.2 Retained Artifact Budget Check

Retention budget is a separate storage policy bucket.

Pre-flight may warn if the run is likely to push retained failed-job evidence or diagnostics over the configured retention budget, but that policy must not be used as a proxy for active scratch feasibility.

### 12.3 Nyquist Guardrails

The pre-flight check must explicitly calculate the Nyquist limit from the cadence that will actually feed phase processing:

`f_limit = FPS_processing / 2`

Where `FPS_processing` is either:
- the native source FPS for a direct CFR path
- the normalized working FPS after deterministic CFR staging for a VFR source

The selected high-frequency cutoff must satisfy:

`f_target < FPS_processing / 2`

Behavior:
- **Warn** if the target band is near the Nyquist limit
- **Block** the render if the selected high-frequency cutoff is greater than or equal to the limit

Additional warnings:
- If clip duration is too short for the selected low-frequency bound to be meaningful
- If the chosen band is suspiciously narrow or otherwise poorly supported by the recording

### 12.4 Constant Frame Rate Check

The pre-flight stage must guarantee a **constant frame rate (CFR)** working stream before drift review or phase processing begins.

Guidance:
- Use `ffprobe` or equivalent metadata inspection to classify native cadence.
- If the input is already CFR, the app may process it directly.
- If the input is VFR but otherwise decodable, the app should create a deterministic CFR working transcode automatically before the phase-processing path begins.
- The original source file remains the provenance anchor. The normalized CFR working file is derived run evidence and must be recorded in diagnostics and sidecar results.
- If CFR normalization cannot be completed or validated, block rendering.

Rationale:
- Temporal frequency filtering assumes a stable sampling rate.
- VFR inputs invalidate the interpretation of target frequency bands unless cadence is regularized explicitly first.

### 12.5 RAM Admission Control

Pre-flight estimates working-set size using conservative assumptions and safety factors.

Inputs include:
- processing resolution
- chunk size
- precision
- intermediate buffers
- mask buffers
- expected native-library thread participation
- codec staging buffers

Rules:
- RAM admission must be based on the bounded execution plan that will actually run, not on a hypothetical full-clip in-memory reconstruction.
- Clip length alone must not trigger RAM rejection if a safe bounded chunk plan exists.
- The scheduler may shrink chunk size to fit the measured RAM budget before render starts, but a true runtime allocation failure is still terminal.

This estimate is admission control, not prediction. If no safe execution plan exists, pre-flight blocks the run.

### 12.6 Input Color / Display Compatibility Check

Pre-flight must validate that the input is either already compatible with the supported v1 display model or can be normalized into it deterministically.

Allowed normalization in v1:
- native square-pixel inputs may pass through unchanged
- supported non-square pixel aspect ratios may be resampled to square-pixel working frames before drift review and render

Blockers in v1 include:
- variable display transforms that the app does not normalize deterministically
- unsupported rotation/orientation metadata
- contradictory or clearly incompatible color metadata
- HDR / wide-gamut paths outside the documented SDR workflow

If display normalization occurs, the transform and resulting working geometry must be recorded in diagnostics and sidecar results.

### 12.7 Dry Run Validation

An optional dry-run mode may perform:
- first/last frame decode
- drift-check readiness
- mask validation
- pre-flight calculations
- render-plan generation

without processing the full clip.

## 13. Progress Model

ETA prediction has been removed.

The Rendering state should show:
- Overall progress bar
- Current stage
- Frames processed / total frames
- Rolling mean time per frame
- Elapsed time
- Event log

Rationale:
- More honest than speculative ETA
- Less likely to frustrate users with inaccurate predictions

---

## 14. Video Resolution Model

The system distinguishes three concepts conceptually, but constrains the exposed output policy.

### 14.1 Source Resolution
- Native input video resolution

### 14.2 Processing Resolution
- Resolution at which the phase engine operates

### 14.3 Output Resolution
- Encoded output resolution
- Must be equal to the processing resolution or smaller
- Upscaling is disabled

The UI must explain that upscaling is intentionally disabled to avoid amplifying or visually inventing artifacts.

---

## 15. Drift Check and Exclusion-Zone Editor

The masking tool is based on a fixed-camera workflow.

### 15.1 Purpose

Allow the user to verify camera stability and define regions that should remain unamplified.

### 15.2 Frame Inspection Modes

The editor supports:
- **First frame**
- **Last frame**
- **First vs Last overlay**
- Overlay opacity slider
- Optional blink toggle

This is used to detect camera drift before rendering.

### 15.3 Supported Shapes

The user may create and edit:
- Rectangles
- Circles

Visual style:
- Exclusion zones use a semi-transparent red fill with a red outline
- Inclusion zones use a semi-transparent green fill with a green outline

Editing operations:
- Select
- Move
- Resize
- Delete selected
- Clear all
- Zoom / Fit / 1:1
- Switch the selected zone, or the next zone to be created, between exclusion and inclusion mode

### 15.4 Coordinate Storage

Mask zones are stored in **source-frame coordinates** after source-display normalization.

Source-display normalization rules for v1:
- the render path operates on an upright, square-pixel working representation
- native square-pixel inputs may use the original decoded frames directly
- supported non-square pixel aspect ratios are normalized to square-pixel working frames before drift review, mask authoring, and processing
- unsupported rotation metadata or display transforms are blocked at pre-flight
- mask coordinates therefore map deterministically from normalized source frame to processing/output domains

### 15.5 Drift Policy

V1 drift handling is **warning-with-acknowledgement**, not a hard automatic classifier.

Rules:
- the editor may display a simple estimated global drift magnitude when it can be computed reliably
- visible drift or estimated drift above the advisory threshold must produce a warning state
- render remains allowed only after explicit user acknowledgement of that warning
- the acknowledgement must be recorded in the sidecar `results` and diagnostics as an operator attestation tied to that reviewed source state
- drift acknowledgement is **not** part of reusable rerun intent and must never be restored by `Load from Export Metadata`
- drift does not silently downgrade the fixed-camera assumption; it is surfaced as operator-attested risk

### 15.6 Compositing Semantics

Masked regions keep **unamplified source content resampled into the final output domain** wherever the final mask resolves to passthrough.

This is the authoritative meaning of “original pixels” for v1.

Rules:
1. The amplification branch operates at the chosen processing resolution.
2. The unamplified passthrough branch is derived from the decoded source and resampled into the final output resolution.
3. If output resolution is smaller than processing resolution, the amplified branch is downscaled into the final output domain before blending.
4. The final mask used for blending is rasterized in the **final output domain** from the stored source-coordinate shapes.
5. If one or more inclusion zones are present, areas outside those inclusion zones default to unamplified passthrough.
6. Exclusion zones always override inclusion zones where they overlap.
7. Final compositing occurs **only once**, in the output domain, using the amplified branch, the unamplified passthrough branch, and the feathered output-domain mask.

### 15.7 Edge Treatment

Mask feathering is mandatory, not optional polish.

Rules:
- a default feather width must be applied
- final blending must use a feathered mask, not a hard binary edge
- blending occurs in a linear-light float32 working domain before final encode conversion

### 15.8 Performance Guidance

The mask should be rasterized once per required domain and reused. Compositing should be implemented as a vectorized masked blend, not by redrawing shapes on each frame.

### 15.9 UI Performance Guidance

The drift-check canvas is a high-risk UI area. The implementation should avoid repeated full-resolution `QImage` conversion churn in the main thread. A GPU-backed or otherwise accelerated presentation path should be preferred if standard Qt image rendering proves too sluggish for large frames.

## 16. Rendering Pipeline

A typical render pipeline is:

1. Input scan / metadata read
2. First/last frame decode for drift-check context
3. Authoritative pre-flight validation
4. Build global processing-reference statistics using a bounded decode pass when required by the numeric engine
5. Launch the bounded processing decode pass for the render attempt
6. Transfer bounded decoded frame chunks into the numeric pipeline
7. Phase decomposition and temporal filtering at processing resolution
8. Reconstruction of the amplified branch
9. Resample amplified branch into output domain if output is smaller than processing resolution
10. Build the unamplified passthrough branch in output domain from decoded source frames
11. Apply output-domain feathered exclusion-mask compositing
12. Encode staged MP4 output
13. Validate staged MP4
14. Write sidecar metadata and diagnostics artifacts
15. Commit final MP4 + sidecar pair

### 16.1 Decode-to-Processor Data Transfer Contract

V1 uses a **bounded streamed chunk contract** between each long-lived decode pass and the worker's numeric pipeline.

Rules:
- the decode child streams raw decoded frame chunks for the active window only
- the worker owns backpressure and must never allow unbounded decode buffering
- only a small bounded number of chunks may be in flight at once
- chunk transfer is authoritative for the worker memory budget and must be accounted for in admission control
- generic stdout/stderr traffic from decode is not part of the data plane and must not be used as proof of forward progress

V1 does **not** use arbitrary per-frame temp-image staging as the primary data plane.

### 16.2 Decode Subprocess Lifetime

V1 uses **one long-lived decode subprocess per sequential decode pass**.

Rules:
- after authoritative pre-flight succeeds, the render worker may launch one bounded reference-scan decode pass and one bounded processing decode pass for the same working source
- each decode child serves successive bounded chunks under worker-controlled backpressure for its pass
- V1 does not relaunch a fresh decode child for every chunk during normal operation
- unexpected decode-child exit classifies the attempt as a decode failure unless the user starts a new render attempt
- if a reference-scan pass is used, its purpose and pass count must be recorded in diagnostics

### 16.3 Child-Process Supervision

The render worker owns its ffmpeg subprocess tree.

Required supervision rules:
- the worker must continuously drain authoritative ffmpeg progress output and non-authoritative stdout/stderr
- cancellation must first stop further numeric scheduling, then stop decode/input production, then close encoder input when safe, then allow bounded encoder drain, then escalate to termination/kill if needed
- child processes must not be orphaned if the worker is cancelled or crashes
- teardown must be bounded by explicit timeouts and classified if escalation is required

## 17. Codec, Color, Bit Depth, and Intermediate Storage Policy

The system is MP4-only at the container level, but this is not sufficient to guarantee useful engineering output. Codec profile, bit depth, chroma, color tags, and intermediate storage must be explicit.

### 17.1 Media Stack Commitment

V1 uses packaged **`ffprobe` / `ffmpeg` subprocesses** for probe, decode, validation, and encode.

This is a deliberate architectural choice, not an implementation footnote.

### 17.2 Output Codec Policy

Preferred output policy:
- **MP4 + HEVC / H.265 Main10** when encoder support is available

Fallback output policy:
- **MP4 + H.264 High** only when Main10 output is not available
- When falling back to 8-bit H.264, the app must warn that subtle amplified motion and tonal gradients may be less trustworthy due to quantization and chroma loss

### 17.3 Input Validation for Bit Depth

The pre-flight check should inspect input bit depth and surface it clearly.

Guidance:
- 10-bit or better sources are preferred for high magnification factors
- 8-bit inputs are allowed only through the supported SDR path and must warn that amplified output may show banding or quantization artefacts
- The sidecar must record the input bit depth and the output codec/profile actually used

### 17.4 Chroma Policy

The design must record the effective chroma handling used by the output encoder. If the output path falls back to 4:2:0 delivery, that fact must be recorded in the sidecar and diagnostics bundle.

### 17.5 Input Color Gate

V1 supports only a documented **SDR Rec.709-class** workflow.

Accepted inputs must satisfy one of:
- explicit compatible Rec.709-class metadata, or
- an explicitly documented heuristic SDR acceptance path

The heuristic path is still a heuristic and must be described as such.

Allow-with-warning heuristic SDR path requires all of the following:
- no HDR / wide-gamut markers
- no contradictory primaries / transfer / matrix tags
- decoded format compatible with the supported SDR path
- the normalization assumption recorded in sidecar results and diagnostics

If these conditions are not met, pre-flight must block.

### 17.6 Output Color Metadata Policy

V1 output must be tagged explicitly for the supported SDR path.

Required output metadata policy:
- write explicit color metadata on output
- use the documented SDR Rec.709-class primaries / transfer / matrix / range policy for accepted inputs
- record the exact tags written in sidecar `results`
- if the input color interpretation is too ambiguous to map into the supported SDR path with confidence, block rather than silently guessing

### 17.7 Intermediate Storage Policy

Intermediates must **not** use visually lossy compressed image/video formats.

Preferred policy:
- chunk intermediates stored as **Zstandard-compressed raw array blocks** or equivalent lossless numeric blocks

Rationale:
- avoids visually lossy artefacts entering a phase-sensitive pipeline
- reduces disk volume compared with fully uncompressed arrays
- preserves deterministic numeric reconstruction for diagnostics and resume-style analysis

The sidecar or diagnostics bundle should record the intermediate storage policy used for the run.

## 18. Out-of-Memory Handling

OOM handling is a first-class requirement.

### 18.1 Pre-Emptive Protection

Before render:
- estimate memory conservatively
- apply safety multipliers
- choose safe chunk sizes and concurrency
- reuse and release bounded batch buffers instead of retaining full-clip arrays

### 18.2 Hard-Stop Policy

If memory pressure becomes unsafe or a real allocation failure occurs:
- stop the render
- classify the job as failed
- retain partial output and temp files per policy
- write a detailed failure record
- return the UI to a clean terminal state

The system must **not** rely on in-place mid-render recovery after a true `MemoryError`, allocator failure, or OS-level memory kill event.

### 18.3 User Guidance on OOM

The failure screen should recommend actions such as:
- lower processing resolution
- switch resource policy to Conservative
- reduce diagnostics level if it materially impacts storage/memory
- verify temp volume headroom

### 18.4 OOM Forensics

If possible, capture:
- pre-flight memory estimate
- last planned chunk size
- resource policy in use
- worker exit code or signal
- last successful stage / chunk

---

## 19. Failure Classification

Failures should be classified, not shown as generic errors.

Examples:
- Active scratch reservation failed
- Output volume staging space insufficient
- Invalid frequency band
- Frequency above Nyquist limit
- Automatic CFR normalization failed
- Unsupported source-display normalization
- Unsupported input decode
- Out of memory
- Temp write failure
- Worker unresponsive
- Worker stalled
- Worker terminated unexpectedly
- Mask geometry error
- Encode failure
- Internal processing exception

Retention-budget overflow is not, by itself, an execution-admission failure. It is a cleanup-policy condition that should trigger purge or warning behavior rather than masquerading as active-scratch infeasibility.

---

## 20. Diagnostics and Debugging

Diagnostics belong under **Advanced → Diagnostics**.

### 20.1 Diagnostic Levels

- **Off**
- **Basic**
- **Detailed**
- **Trace**

### 20.2 What Diagnostics Should Capture

At increasing levels, diagnostics may include:
- stage transitions
- pre-flight report
- resource policy decisions
- scheduler concurrency and chunking choices
- memory estimates and observed peaks
- chunk timings
- mask statistics
- failure details
- codec/profile/chroma details

### 20.3 Diagnostics Bundle

For failed or suspicious runs, the app should be able to write a diagnostics bundle containing:
- run log
- settings snapshot
- source metadata
- pre-flight report
- scheduler decisions
- memory estimate
- mask geometry
- stage timings
- failure reason
- paths to artifacts

### 20.4 Mask Debug Artifacts

Optional exports:
- First frame with zones burned in
- Last frame with zones burned in
- First/Last drift overlay image
- Rasterized processing mask image

### 20.5 Pre-flight Report Viewer

The UI should expose a readable pre-flight summary showing:
- Native source FPS
- Working FPS / normalization status
- Nyquist limit
- Selected band
- Source display normalization status
- Temp estimate
- RAM estimate
- Resource policy selected

### 20.6 Diagnostics Cap Fallback Rule

When a run approaches or exceeds the active diagnostics cap, artifacts must be dropped in a strict priority order.

Drop first:
1. optional visual debug exports (burned-in mask images, drift overlay exports, duplicate previews)
2. trace-level per-chunk detail artifacts
3. detailed timing breakdowns beyond stage-level summaries
4. verbose auxiliary logs that duplicate information already preserved elsewhere

Must never be dropped while the run record is being preserved:
- terminal failure classification
- core JSONL run log
- pre-flight summary/report
- sidecar `intent`, `observed_environment`, and `results`
- final or failed artifact manifest
- minimal watchdog/liveness evidence needed to explain termination

Any cap-triggered suppression or truncation must be recorded in the JSONL log and diagnostics summary.
- Warnings and blockers

---

## 21. Logging Standardization

Diagnostics logs should use a structured format.

### 21.1 Required Format

Use **JSONL** (JSON Lines) for machine-readable logs.

Each log event should contain fields such as:
- timestamp
- level
- event_type
- job_id
- stage
- message
- structured payload

### 21.2 Benefits

- Easier future analysis
- Better support tooling
- Better failure triage
- Stable diagnostics bundle contents

### 21.3 Relationship to IPC

The worker should write JSONL logs directly to disk. The IPC channel should transmit concise state and warning events, not the full diagnostic log stream.

---

## 22. Sidecar Metadata and Reproducibility

Every successful render must write a sidecar JSON file alongside the MP4.

Example:
- `output.mp4`
- `output.json`

### 22.1 Strict Sidecar Structure

The sidecar must be split into three top-level domains:
- `intent`
- `observed_environment`
- `results`

This separation is normative.

### 22.2 Intent

`intent` contains the reproducible job settings the user meant to run.

Examples:
- phase settings
- processing resolution
- output resolution
- exclusion-zone geometry
- resource policy selected by the user
- requested output path policy

Non-reproducible operational details such as diagnostics verbosity, actual scheduler clamps, warnings, fallbacks, or prior operator attestations do not belong in `intent`.

### 22.3 Observed Environment

`observed_environment` contains machine- and run-contingent facts that explain what happened but must not be treated as portable rerun intent.

Examples:
- app version
- engine version
- OS / platform details
- effective thread and library limits actually applied
- ffmpeg toolchain version
- actual scheduler clamps
- temp-root selection
- diagnostic cap values in force

### 22.4 Results

`results` contains facts about the completed run and its outputs.

Examples:
- render timestamp
- source fingerprint and source metadata, including any cadence/display normalization actually used
- pre-flight report and blockers/warnings actually encountered
- drift acknowledgement / operator attestation tied to that reviewed source state
- codec/profile/pixel-format selected in practice
- color tags written on output
- warnings and fallbacks
- artifact paths
- staged/final validation outcome
- diagnostics summary

### 22.5 Load from Export Metadata

The UI must support loading settings from a prior export sidecar.

Default behavior:
- restore **only `intent`**
- never silently apply machine-contingent `observed_environment` values as rerun settings
- never treat `results` as authoritative future intent
- never restore prior operator attestations (including drift acknowledgement) as future run intent
- never import diagnostics verbosity or other convenience-state controls as engineering intent

The UI may optionally show observed execution details for user review, but not import them by default.

### 22.6 No Full Preset Library in v1

A full preset browser/library UI is deferred.

Instead, v1 uses:
- Last Used Settings persistence
- Load from Export Metadata

## 23. JSON Sidecar Schema Validator

The app must validate sidecar metadata before loading it.

### 23.1 Validation Requirements

Check:
- `schema_version`
- required top-level domains (`intent`, `observed_environment`, `results`)
- numeric ranges
- enum values
- exclusion-zone structure
- coordinate validity
- version compatibility markers
- codec/profile field validity
- source cadence metadata presence
- normalization metadata presence when a derived working source was used

### 23.2 Validation Outcomes

- **Valid**: load fully
- **Valid with warnings**: load with explicit notice
- **Invalid**: block import and explain failures

### 23.3 Compatibility Policy

- Sidecars must be versioned
- A small number of older versions may be supported
- Migrations should be explicit and recorded
- Unsafe or ambiguous files should be rejected

This protects repeatability and stable re-runs.

---

## 24. Last Used Settings Persistence

The app should persist the last used settings for convenience, but this is not the primary reproducibility mechanism.

Persistable items:
- phase settings
- resolutions
- resource policy
- diagnostics level
- mask zones
- output preferences

On restart:
- restore last used settings if available
- do not silently resume failed jobs

---

## 25. Storage, Temp Files, and Cleanup

The storage model has three distinct buckets and they must not be conflated.

### 25.1 Active Scratch Space

Active scratch space is the guaranteed working space required for the current run.

Rules:
- it is a hard execution requirement checked at pre-flight
- it covers intermediate blocks, staged output, and run-scoped diagnostics required before completion
- if active scratch cannot be reserved safely, the run must not start

### 25.2 Retained Evidence

Retained evidence is failed-run material intentionally kept for diagnosis.

Examples:
- partial outputs
- retained temp blocks
- per-run diagnostics bundles

Retention of this evidence is governed by cleanup policy, not by active-run scratch admission.

### 25.3 Global Retention Budget

Default retention budget: **50 GB**.

This is a **global application preference**, not a per-job output parameter.

The implementation should support:
- absolute budget mode in GB
- optional percentage-of-available-disk mode

V1 default:
- absolute mode
- 50 GB

Successful-run diagnostics, logs, and optional retained bundles count toward the global retention budget only. They must not be folded into active scratch admission checks, and they must not be mislabeled as failed-run evidence.

### 25.4 Partial Output and Temp Files on Failure

On failure, the system retains partial outputs and temp files for diagnostics.

### 25.5 Required Cleanup Utility

Because retained artifacts can be large, the UI must provide:
- **Purge Temp Files for this Job**
- Purge failed-run temp files
- Open temp folder

The Failed screen must include immediate cleanup controls.

### 25.6 Retention Policy

The application should purge oldest retained evidence first when the configured retention budget is exceeded, subject to preserving the current failed job until the user acts.

### 25.7 Finalization and Output Pair Commit

The authoritative export is the pair:
- final MP4
- matching final sidecar JSON

Rules:
- the staged MP4 must pass validation before final commit proceeds
- output staging must occur on the output volume so final rename/move semantics remain predictable
- a lone visible MP4 without its matching final sidecar is not yet an authoritative completed export
- success requires that both files exist in their final locations
- if the MP4 reaches its final visible path but sidecar finalization fails, the run is classified as failed and the worker must immediately attempt to move that MP4 into a failed-evidence quarantine path on the same output volume
- if quarantine rename fails, the UI/logs must mark the visible MP4 as incomplete failed evidence and surface immediate cleanup actions; it must never be represented as a completed export

## 26. Complete / Failed Screen Requirements

### 26.1 Complete Screen

Should show:
- output path
- sidecar path
- key settings used
- open output folder
- open sidecar
- load settings from this export

### 26.2 Failed Screen

Should show:
- classified failure reason
- failure stage
- frames completed
- partial output path
- temp-file path
- latest log path
- suggested corrective action

Must include:
- **Purge Temp Files for this Job**
- Open temp folder
- Open diagnostics/log bundle
- Duplicate settings into a new run

---

## 27. Portability Considerations

The application is intended to become portable in the future.

This affects:
- where settings are stored
- where temp files are stored
- where logs are stored
- where retained evidence is stored
- how sidecar and diagnostics paths are resolved
- how external dependencies are packaged

### 27.1 Installed-Mode Filesystem Contract

In installed mode, defaults should be:
- per-user application data for settings
- configurable temp/scratch root for active run data
- configurable diagnostics/evidence root
- sidecars written adjacent to exported MP4s

### 27.2 Portable-Mode Filesystem Contract

Portable mode is a future packaging target, but the path contract must already be compatible with it.

Portable mode should use relocatable paths rooted under a portable data directory rather than fixed OS-specific application-data assumptions.

At minimum the design must permit separate relocatable roots for:
- settings
- temp/scratch
- diagnostics/evidence

### 27.3 Portable-Friendly Design Guidance

- Keep the engine self-contained
- Avoid hard-coded system service dependencies
- Avoid assumptions that require installation-level privileges
- Keep settings and logs separable from executables
- Keep IPC transport abstracted from platform-specific socket assumptions
- Keep packaged ffmpeg/ffprobe discoverable through explicit path configuration rather than hidden system dependencies

## 28. Highest-Risk Implementation Areas

### 28.1 Memory Estimation Accuracy

Estimating working-set size in Python-heavy numeric workloads is difficult and should be treated as safety-biased approximation, not truth. The design mitigates this by using conservative estimates, safety multipliers, and hard-stop behavior rather than trusting a perfect estimate.

### 28.2 Drift Check UX

A performant, flicker-free First/Last overlay in PyQt6 for large frames is a meaningful UI implementation challenge.

### 28.3 Compositing Latency

Exclusion-zone compositing adds per-frame work. This is mitigated by pre-rasterizing masks and using vectorized masked compositing.

### 28.4 Shell / Engine Boundary

The telemetry contract must remain narrow and stable so the worker stays UI-agnostic while the shell still gets enough information to display progress and diagnose failures.

### 28.5 Worker Liveness Detection

The watchdog must combine heartbeat monitoring with OS-level process polling. Heartbeats alone are not sufficient when a worker segfaults inside a C-extension.

### 28.6 Codec / Bit-Depth Integrity

If the output path falls back to lower-bit-depth or more lossy codec settings, the resulting render may be visually smoother to share but less trustworthy to interpret. The architecture must surface that downgrade clearly.

---

## 29. Implementation Guidance Summary

### Keep
- Single-job instrument paradigm
- Mandatory pre-flight checks
- Drift-check workflow
- Downscale-only output policy
- Separate worker process
- Diagnostics bundle
- Sidecar-based reproducibility
- Worker watchdog

### Enforce
- No in-process phase engine execution
- No upscaling options
- Nyquist hard block at or above `FPS_processing / 2`
- CFR validation and deterministic normalization of VFR sources before phase processing
- Deterministic square-pixel normalization for supported non-square-pixel sources before drift review or phase processing
- Structured JSONL logging
- Sidecar schema validation before metadata import
- Hard-stop policy on true OOM

### Simplify
- No speculative ETA
- No preset library UI in v1
- No manual core pinning
- No reliance on queue-only IPC for instrument-critical telemetry

### Add
- Load from Export Metadata
- Purge Temp Files for this Job
- 50 GB default temp budget with configurable mode
- Explicit codec/profile/bit-depth reporting
- Lossless intermediate storage policy
- OS-level worker polling in the watchdog

---

## 30. Decision Rationale Summary

This section summarizes why the major architectural decisions were made. It is intentionally brief and decision-oriented rather than explanatory prose.

### 30.1 Why a desktop instrument instead of a web stack

- local file processing is simpler, more deterministic, and easier to validate
- the target workflow is offline engineering review, not collaborative browser editing
- removing React/Flask/socket plumbing reduces operational failure modes and packaging complexity

### 30.2 Why single active render only

- avoids queue orchestration, resource arbitration, and multi-job UI complexity
- keeps cancellation, failure handling, and diagnostics attributable to one run at a time
- matches the instrument-style product goal

### 30.3 Why the render engine is out-of-process

- native numeric/media workloads can freeze or crash the interpreter
- Qt responsiveness and watchdog correctness depend on strict process isolation
- a separate worker gives cleaner failure classification and safer teardown

### 30.4 Why cadence normalization, Nyquist, and pre-flight checks are mandatory

- phase-based temporal filtering is only trustworthy when the working stream has a stable sampling rate
- deterministic normalization is acceptable; silent guessing is not
- blocking invalid or unnormalizable jobs early is preferable to producing visually convincing but mathematically unsound output
- an engineering tool should normalize explicitly or reject explicitly, never blur the distinction

### 30.5 Why there is no live mode and no full preview mode

- live behavior would force a very different latency, buffering, and UI architecture
- full preview would add media-editor complexity without improving the core fixed-camera batch workflow
- the design instead uses drift-check plus full render as the deliberate operating model

### 30.6 Why masks remain static operator-authored zones

- the fixed-camera assumption makes static masking a strong simplification
- rectangles/circles are sufficient for the intended v1 use case
- exclusion remains the safer default for avoiding accidental omission of relevant structure
- explicit inclusion zones are still allowed when the operator deliberately wants to constrain amplification to a reviewed region

### 30.7 Why output is downscale-only

- upscaling can visually invent detail and make artifacts look like signal
- forbidding upscale protects interpretability and keeps the product epistemically conservative

### 30.8 Why sidecars are split into intent / observed_environment / results

- reproducible engineering settings should be separated from machine-contingent runtime facts
- future reruns should restore what the user meant to do, not everything that happened on one machine on one day
- operator attestations belong to results, not reusable intent

### 30.9 Why full-file SHA-256 remains mandatory

- source provenance must be unambiguous across probe, edit, and render phases
- a weaker identity rule would reduce load latency but increase reproducibility ambiguity
- the design accepts the UX cost explicitly rather than pretending it is free

### 30.10 Why ffmpeg/ffprobe subprocesses are explicit dependencies

- media capability probing, cadence/display normalization, encode validation, and child progress reporting are clearer with an explicit toolchain
- subprocess isolation keeps codec failures out of the PyQt process and makes supervision more tractable

### 30.11 Why diagnostics are structured and capped

- compute-heavy offline tools fail in ways that must be diagnosable after the fact
- JSONL logs and bounded bundles support post-mortem analysis without making diagnostics themselves an uncontrolled storage hazard

### 30.12 Why output completion is defined as MP4 + sidecar pair

- the MP4 alone is not enough to prove a trustworthy completed export
- the matching sidecar carries provenance, pre-flight outcomes, output tags, and runtime facts needed for later review
- treating the pair as authoritative is stricter, but it preserves reproducibility discipline

---

## 31. Quantitative Analysis Extension

This section is normative for the NVH-oriented quantitative-analysis extension now implemented in the desktop application.

### 31.1 Execution Boundary

- Quantitative analysis runs only alongside the authoritative render path
- The worker derives analysis traces from the internal motion-analysis backend, not from the encoded MP4
- Analysis is enabled by default but may be disabled per run
- Analysis uses the same configured render frequency range; there is no separate analysis-only band selection in v1

### 31.2 ROI Model

- Exactly one analysis ROI may be authored per run
- The ROI uses the same rectangle and circle authoring tools as the drift / mask editor
- If no manual ROI is drawn, the worker records and uses a visible **Whole-frame ROI**
- Whole-frame ROI means the full frame, constrained by any inclusion zones and then reduced by exclusion zones
- The ROI label written to exported artifacts is `Whole-frame ROI` for the fallback path

### 31.3 Internal Analysis Hierarchy

The implemented analysis hierarchy is:

1. dense motion-grid traces over the processing-resolution working frames
2. adaptive per-cell combination of X/Y/projected motion components
3. ROI-cell traces and spectra from weighted groups of dense grid cells
4. robust ROI spectrum from median aggregation of valid ROI-cell spectra
5. supported-peak detection using valid-cell support fraction
6. auto-band or manual-band resolution
7. band-energy heatmaps on the dense analysis grid
8. fixed-filename artifact export plus sidecar / diagnostics registration

### 31.4 Quality and Reporting Rules

- ROI-cell hard failures include masked-out cells, texture-poor cells, and cells with no usable motion signal
- Rejected cells are excluded from ROI-spectrum aggregation but still penalize the overall ROI quality score
- The exported ROI quality score records these sub-scores:
  - texture adequacy
  - valid-cell fraction
  - inter-cell agreement
  - peak consistency
  - mask impact
  - drift impact
- Peak reporting is suppressed when overall ROI quality stays below the configured ROI quality cutoff
- Heatmaps and traces still export when peak reporting is suppressed

### 31.5 Band Selection

- `Auto-bands` uses the aggregated ROI spectrum only
- `Manual single band` uses one explicit low/high band
- `Manual multiple bands` accepts up to five explicit low/high bands
- Auto-band count is capped at five and is operator-configurable in Advanced settings
- Nearby auto-bands are merged when they overlap or are insufficiently separated

### 31.6 Heatmaps

- Heatmaps use a denser grid than the ROI-cell grid
- Each heatmap cell stores normalized band energy: band energy divided by total motion energy
- Low-confidence cells remain visible in monochrome and are excluded from shared scale fitting
- All heatmaps in one render share a robust percentile display scale

### 31.7 Artifact Contract

The worker writes these fixed filenames into the render output directory:

- `roi_metrics.csv`
- `roi_spectrum.json`
- `roi_trace.csv`
- `cell_traces.csv`
- `cell_spectra.csv`
- `analysis_metadata.json`
- `heatmap_<band_id>.csv`
- `heatmap_<band_id>.png`

The sidecar `results.analysis` payload records:

- analysis enablement and status
- ROI mode, label, geometry, and quality score
- reported peaks
- generated heatmap bands
- artifact paths
- rejection statistics
- auto-band merge steps
- suppressed-peak reasons
- shared heatmap-scale metadata

### 31.8 Failure Boundary

- If the MP4 render succeeds but quantitative-analysis export fails, the run still completes as a render success
- The worker records the analysis issue as a warning, writes a minimal analysis metadata record when possible, and still finalizes the MP4 + sidecar pair
- Diagnostics bundles include any analysis artifact paths that were successfully written

## 32. Quantitative Analysis Implementation Notes

The current implementation uses a dedicated dense motion-analysis grid inside the worker. That grid reuses the same local motion-estimation backend as the render engine, but it is kept separate from the amplification warp grid so the quantitative-analysis export contract can evolve without changing the encoded video path.

UI contract:

- A dedicated **Analysis** section exposes the enable toggle, ROI summary, ROI editor, and band mode selector
- Advanced settings carry the curated analysis thresholds, auto-band count, manual bands, and the internal-export toggle
- Completed-render UI remains export-oriented; quantitative outputs are written to artifacts rather than rendered inline in the completion screen

---

## 33. Binding Clarifications from Latest Architecture Review

This section is normative. If any earlier wording is looser or conflicts with this section, this section wins.

### 33.1 Full-file SHA-256 provenance cost

V1 keeps **full-file SHA-256** as the canonical provenance rule even though it is expensive on large clips.

This is a deliberate correctness trade-off, not a lightweight convenience feature.

Operational policy:
- source load may enter `Fingerprint pending` after fast probe results are available
- non-render editing may continue while fingerprinting runs
- the UI must show that the source is not yet authoritative
- the render path must still recompute canonical SHA-256 before accepting probe-derived context

The architecture explicitly accepts this UX tax in exchange for a single unambiguous provenance rule.

### 33.2 Long fingerprint wait UX

If fingerprinting remains in progress beyond a short UX threshold, the shell must:

- continue to permit non-render editing
- keep **Start Render** disabled
- show an explicit long-running fingerprint status indicator
- prefer visible hash progress when available
- make clear that the source is not yet ready for authoritative render

### 33.2A Meaning of `Ready`

`Ready` is reserved for the state in which canonical fingerprinting has completed successfully and the source has not been invalidated by shell-side stale-source checks.

A source that is still editable but awaiting canonical fingerprint completion is in `Fingerprint Pending`, not `Ready`.

### 33.3 CPU headroom policy honesty

The `N-1 logical cores` rule is a **coarse v1 heuristic**.

It is not a guarantee of shell responsiveness and must not be described as one.

Required wording/policy:
- the scheduler may clamp below `N-1` on low-core systems or where topology/SMT makes `N-1` misleading
- applied caps must be logged
- diagnostics must record the effective thread/process limits actually used

### 33.4 Stall-evidence reset rule

A prolonged-stall timer may be reset only by **qualifying evidence of forward motion**.

Qualifying evidence is limited to:
- a worker-issued `progress_update` with an advanced progress token
- a worker-issued stage-transition message
- authoritative ffmpeg progress-channel counters that move forward for the active decode or encode child

The following do **not** reset a prolonged-stall timer:
- repeated heartbeats with unchanged progress token
- generic stdout/stderr chatter
- warning messages without a completed work boundary
- trace-log traffic without a corresponding progress-token advance

### 33.5 Authoritative child-activity source

For watchdog decisions, **dedicated ffmpeg progress output** is the authoritative child-activity signal when available.

Rules:
- ffmpeg progress output must be machine-readable
- stdout/stderr must still be drained to avoid deadlock
- stdout/stderr chatter is diagnostic only and is not authoritative evidence of forward progress

### 33.5A Worker timestamp authority

Worker-reported monotonic timestamps are diagnostic breadcrumbs only.

Rules:
- shell-side timeout and stall classification must use local receipt time, local watchdog timers, progress-token advancement, authoritative child-progress counters, and OS-level liveness
- worker clocks must not be treated as authoritative for cross-process timeout truth
- timestamp skew or missing timestamp fields are protocol-quality problems, not by themselves evidence of forward motion or stall clearance

### 33.6 Prolonged `alive_no_progress` escalation

`alive_no_progress` is not an indefinite resting state.

Rules:
- if the worker remains alive but without qualifying forward-motion evidence beyond the stage-specific prolonged-stall ceiling, the shell must re-check process liveness and authoritative child progress
- if the worker is still alive but no qualifying forward motion is observed, classify the state as `worker_stalled`
- `worker_stalled` is a terminal failed state unless the worker resumes before the hard ceiling expires and emits qualifying evidence of progress

### 33.7 Decode subprocess lifetime and data plane

V1 uses **one long-lived decode subprocess per sequential pass** and a bounded streamed chunk contract.

Rules:
- the worker may use one bounded reference-scan pass and one bounded processing pass against the same working source when global reference statistics are required
- within a pass, the decode child remains alive across successive chunk windows
- the decode child streams only the active bounded chunk window
- the worker owns backpressure and must prevent unbounded buffering
- raw decoded chunk transfer, not generic child chatter, is the data plane

### 33.7A Source normalization contract

Drift review and phase processing operate on a normalized working source representation.

Rules:
- native CFR, upright, square-pixel inputs may use a direct path without derived media generation
- VFR inputs must be deterministically transcoded to CFR before drift review or phase processing
- supported non-square pixel aspect ratio inputs must be deterministically normalized to square-pixel working frames before drift review or phase processing
- unsupported rotation metadata or ambiguous display transforms remain blockers
- the original source remains the provenance anchor and must retain canonical fingerprint ownership
- any derived working media is runtime evidence, not replacement source identity
- normalization plan, outcome, and resulting working cadence/geometry must be recorded in diagnostics and sidecar results
- if normalization fails or cannot be validated, the render must not start

### 33.7B Bounded memory contract

The authoritative RAM contract is the bounded batch plan, not full clip length.

Rules:
- the worker must not require the full clip to be resident in RAM at once
- global reference statistics may be accumulated across a bounded scan pass, then reused during the bounded processing pass
- chunk-size decisions may be reduced at job start to fit the measured RAM budget and selected resource policy
- per-batch arrays must be released after each bounded work window so long clips can exceed available RAM as long as the batch plan remains admissible
- a true allocation failure after admission remains terminal and must still be classified as out-of-memory

### 33.8 Child shutdown rationale

Default cancellation/teardown order is:
1. stop scheduling further numeric work
2. stop decode/input production first
3. finalize/close the worker-to-encoder input path when safe
4. allow bounded encoder drain
5. then signal/kill the encoder if it still does not exit
6. finally force-kill remaining children if needed

Rationale:
- upstream stop prevents creation of additional unfinished work
- bounded encoder drain improves the chance of retaining a coherent partial evidence file
- bounded shutdown is still preferred over indefinite drain waiting

### 33.9 Staged MP4 validation contract

Before final rename/commit, the staged MP4 must pass all of the following:
- encoder exit code `0`
- file exists, is closed, and has non-zero size
- `ffprobe` parses the container successfully
- exactly one video stream and no audio stream
- expected output width/height
- codec/profile/pixel-format consistent with the selected output path
- CFR-consistent duration or equivalent frame-count evidence within no more than one frame period tolerance

If any validation item fails, finalization must not proceed.

### 33.10 Paired-output completion semantics

The authoritative export is the pair:
- final MP4
- matching final sidecar JSON

This means a final-path MP4 may be briefly visible before the export is authoritative.

Rules:
- external tools and operators must treat a lone MP4 without its matching sidecar as incomplete
- UI copy should make this explicit
- success requires that both files exist in their final locations
- if sidecar finalization fails after the MP4 becomes visible at its final path, the worker must try to quarantine that MP4 as failed evidence on the same volume before reporting terminal failure

### 33.10A Lone-MP4 failure handling

If a final-path MP4 becomes visible but the matching sidecar cannot be finalized:

- the export state is `failed`, not `completed`
- the worker must first attempt same-volume quarantine of that MP4 into the failed-evidence area for the job
- if quarantine succeeds, the quarantine path is recorded in logs and shown in the failed-run UI
- if quarantine cannot be completed, the MP4 may remain visible in the output folder, but it must be labeled in UI/logs as incomplete failed evidence and accompanied by a purge action

This rule exists to prevent a visible MP4 from being mistaken for an authoritative completed export.

### 33.11 Strict sidecar reload boundary

`Load from Export Metadata` restores **only `intent` by default**.

Rules:
- `observed_environment` must never be silently imported as future run intent
- `results` must never be treated as default future run settings
- the UI may display `observed_environment` and `results` for review, but not apply them automatically

### 33.12 Output-domain compositing rule

When output resolution is smaller than processing resolution:
- the amplified branch is downscaled into the output domain first
- the unamplified passthrough branch is independently resampled from source into the same output domain
- the final feathered mask is rasterized in the output domain
- compositing occurs once, in output domain, in linear-light float32

This is the authoritative meaning of “original pixels” for downscaled output.

### 33.13 Output color metadata policy

For accepted SDR-path renders, the encoder must write explicit output color metadata and record the exact tags in sidecar `results`.

If the input cannot be mapped into the supported SDR policy with sufficient confidence, pre-flight must block rather than guessing.

### 33.14 Effective concurrency observability

Every run must record the concurrency settings actually applied, including:
- ffmpeg thread arguments
- OpenCV thread limit if used
- BLAS/OpenMP-related environment limits
- any stricter runtime clamp chosen by the scheduler

These values belong in:
- JSONL log
- diagnostics bundle
- sidecar `observed_environment`
