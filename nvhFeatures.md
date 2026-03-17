# NVH Quantitative Analysis Design
## Phase-based Motion Amplification Desktop Utility

## Purpose

This document defines the **quantitative-analysis extension** for the Phase-based Motion Amplification Desktop Utility.

The goal is to add **NVH-relevant quantitative outputs** derived from the same internal phase-based motion-analysis pipeline used for render-time amplification, while keeping the product within its current scope:

- offline only
- single active render
- no live mode
- no general preview mode
- fixed-camera workflow
- phase-based analysis only

This analysis layer is intended to support **relative motion interpretation**, not absolute metrology.

It should help answer questions like:

- Which region is most active at a problematic tonal frequency?
- What are the dominant motion peaks inside a chosen region?
- Are peaks spatially consistent across the selected region?
- Which frequency bands should be visualized as heatmaps?

It should **not** claim:

- absolute displacement in mm
- damping
- force path identification
- true operational deflection shapes
- calibrated sensor-equivalent motion magnitude

---

## High-level goals

Add **render-time** quantitative outputs for:

1. **ROI spectrum**
2. **ROI quality score**
3. **Band-energy heatmaps**
4. **Exported analysis artifacts** that preserve enough information for later run-to-run comparison outside the tool

The analysis must be:

- generated during the authoritative render path
- derived from the internal phase-analysis representation, not from the output MP4
- optional, but enabled by default
- tied to the same frequency range as the visual amplification render
- exportable in machine-readable formats

---

## Scope decisions

### In scope
- one analysis ROI per run
- ROI spectrum
- ROI quality score and sub-scores
- auto-band detection from ROI spectrum
- user-selectable manual band modes
- render-time export of analysis artifacts
- heatmaps derived from internal phase-based analysis
- enough exported data for later comparison outside the app

### Out of scope
- absolute displacement claims
- sensor fusion
- microphone / accelerometer import
- modal reconstruction
- automatic diagnosis
- in-app run-to-run comparison
- post-render interactive analysis workflows
- analysis-only mode

---

## Key product constraints

- Quantitative analysis runs **only alongside video render**
- If the render succeeds but quantitative analysis export fails, treat as:
  - **render success with analysis failure warning**
- Analysis artifacts are written **after full render completes**
- Quantitative analysis is **optional but enabled by default**
- No analysis-only mode in v1

---

## Analysis ROI model

### ROI creation
- Uses the **same shape tools** as exclusion zones:
  - rectangle
  - circle

### ROI count
- Exactly **one analysis ROI** per run

### If no ROI is drawn
The tool falls back to a **whole-frame ROI**:

- region = full frame minus exclusion zones
- shown explicitly in the UI
- treated exactly like a user-defined ROI for:
  - ROI spectrum
  - ROI quality score
  - heatmap generation

### Output labeling
If fallback is used, outputs should label the ROI as:

- **Whole-frame ROI**

---

## Core architectural principle

Quantitative analysis must branch from the **internal phase-analysis path**, not from:

- the amplified output MP4
- displayed frames
- encoded frames
- post-render image measurements

This keeps the quantitative outputs tied to the same internal representation that drives the render.

---

## Analysis dataflow

### Primary path
1. Input video decoded
2. Phase-based processing runs
3. Internal phase-derived responses become available
4. Quantitative analysis derives cell-level traces from those internal phase responses
5. Spectral and spatial analysis products are generated
6. Visual render completes
7. Analysis artifacts are exported

### Important rule
Quantitative analysis uses the **same frequency range configured for render amplification**.

There is no separate analysis frequency range in v1.

---

## ROI analysis architecture

### ROI subdivision
The ROI is split into internal sub-regions.

This is required for robustness.

### Why
A single ROI-wide scalar is too fragile because it can be dominated by:

- local reflections
- low-texture patches
- one unstable area
- masking contamination
- small numerical artifacts

### Internal grid design
Use a **sub-region grid** inside the ROI.

Suggested implementation defaults:
- rectangles: regular internal grid
- circles: inscribed internal grid or equivalent clipped cell layout

This internal grid is not the same as the heatmap grid.

---

## Cell-level analysis pipeline

For each ROI cell:

1. extract phase-derived responses by **scale** and **orientation**
2. compute multiple intermediate traces
3. score each component **per cell**
4. apply **per-cell adaptive weighting**
5. combine weighted components into one **cell motion trace**
6. compute one **cell spectrum**
7. compute cell quality metrics
8. either keep or reject the cell for spectral aggregation

### Important design choice
The cell spectrum is the **primary analysis object**.

The aggregated ROI time trace is supportive only.

---

## Component combination per cell

### Chosen model
Use a **weighted sum** of intermediate scale/orientation components.

### Weighting basis
Weights are **per-cell adaptive**, not global.

They should reflect:
- signal strength
- stability
- band relevance
- local usefulness

### Reason
This is more robust than early averaging and more flexible than median-combining all components blindly.

---

## Cell rejection model

### Rejection policy
Use a **composite quality score** with a few hard fails.

### Hard fail examples
A cell should hard-fail if it has issues such as:
- too much excluded area
- effectively no usable texture
- invalid or unstable trace generation
- too little valid signal duration

### Otherwise
The cell receives a composite quality score and is rejected only if that score falls below threshold.

### Cell scoring inputs
Examples:
- texture adequacy
- trace stability
- component agreement
- band relevance
- mask penalty
- drift penalty

### Quality accounting rule
Rejected cells:
- are **excluded from spectral aggregation**
- still **penalize the overall ROI quality score**

---

## ROI spectrum

### Primary output basis
The final ROI spectrum should come from:

- **robust aggregation of cell spectra**
- not from averaging traces first

### Chosen default
Use **median / robust aggregation of cell spectra**.

### Reason
This is more robust to:
- one bad cell
- reflections
- weak texture
- local numerical instability

### Aggregated ROI trace
Still export an aggregated ROI trace, but treat it as:
- secondary
- supportive
- diagnostic

The time trace should be sampled at the **native video frame rate**.

---

## Peak detection and reporting

### Peak support rule
A peak is only reportable if it is:
1. present in the robust aggregated ROI spectrum
2. present in at least a minimum fraction of valid cells

### Cell-support rule
Use a support-fraction threshold.

This threshold should be user-adjustable in Advanced settings.

### Peak export policy
- Export **all supported peaks above threshold**
- Highlight **top peaks** in summary-oriented outputs only

### Ranking
Rank peaks using a **fixed composite formula** based on:
- spectral amplitude
- cell-support fraction
- ROI quality

This formula is fixed in v1.

---

## ROI quality score

### Required outputs
User-facing:
- overall ROI quality score
- confidence label

Export/internal:
- overall score
- sub-scores

### Required sub-scores
At minimum:
- texture adequacy
- valid-cell fraction
- inter-cell agreement
- peak consistency
- mask impact
- drift impact

### Accounting rule
Rejected cells must still reduce the overall ROI quality score.

### Low-quality ROI behavior
If the ROI quality score falls below the reporting threshold:
- suppress peak reporting
- still export trace/heatmap outputs
- record a quality warning

The quality thresholds should be a **small curated set** of Advanced controls.

---

## Heatmap architecture

### Heatmap purpose
Heatmaps should show **where band activity is concentrated spatially**.

### Main display value
Each heatmap cell should display:

- **normalized band energy**
- not raw band energy

Formula conceptually:
- band energy / total motion energy

### Reason
This highlights band-specific activity rather than generic high-motion areas.

### Low-confidence cells
Low-confidence cells should:
- not be hidden
- not be colored normally
- be shown as **monochrome black/white**
- be excluded from scale fitting

Also export:
- low-confidence mask
- low-confidence count
- reason summary

---

## Heatmap grid

### Separate grid
Use a **denser heatmap grid** than the ROI-analysis grid.

### Rule
The heatmap grid:
- is independent from the ROI analysis grid
- uses the same underlying phase-analysis backend
- does not redefine the ROI spectrum

### Purpose
- ROI analysis grid = robust quantitative estimation
- heatmap grid = denser spatial visualization/analysis

### Density
Heatmap density should be **adaptive to ROI size/resolution**.

---

## Heatmap scaling

### Display scale
Use **one fixed scale across all heatmaps in the same render**.

### Scale calculation
Use a **robust percentile range** over all valid heatmap cells from all generated heatmaps in the run.

### Why
This keeps multiple heatmaps visually comparable.

### Export metadata
Per render, store:
- normalization method
- lower percentile
- upper percentile
- actual numeric display bounds
- clipped-cell counts

---

## Heatmap bands

### Auto-band source
Auto-band detection is based on the **ROI aggregated spectrum only**.

### Reason
This keeps analysis focused on the selected component/region.

### Auto-band width
Use **adaptive band width**, not fixed width.

### Width behavior
- estimate local spread around each supported peak
- apply minimum width
- apply maximum width
- merge overlapping or near-overlapping bands

### Peak merge rule
If nearby peaks are close, **merge them into one band** unless the valley between them is clearly deep.

### Default count
Auto-band mode should be **user-adjustable in Advanced settings** with default:
- **5 auto-bands**

### Band modes
Menu should support:
1. auto-bands
2. one user-selected manual band
3. multiple manual bands

### Manual-band mode
- define bands using explicit **low/high frequency bounds**
- cap manual multiple-band mode to **maximum 5 bands**
- when auto-band mode is selected, generate **auto-bands only**

---

## Whole-frame fallback behavior

If no ROI is manually drawn:

- the tool creates a visible **Whole-frame ROI**
- region = full frame minus exclusion zones
- spectrum and quality score are computed exactly as for a manual ROI
- the whole-frame ROI is shown in the UI
- outputs label it clearly as **Whole-frame ROI**

---

## Artifact generation policy

### When artifacts are written
Analysis artifacts are written **after the full render completes**.

### Why
This avoids partial artifact handling complexity during render.

### Failure behavior
If render succeeds but analysis export fails:
- final status = render success with analysis failure warning

### Analysis artifact inclusion in diagnostics
Quantitative analysis artifacts should be included in the **diagnostics bundle automatically**.

---

## Export formats

### Required formats
- CSV for tables / traces / grids
- JSON for structured spectra / metadata
- PNG for human-readable heatmaps

### Output layout
Artifacts are written in the **main output folder**, not a subfolder.

### Filenames
Use **fixed predictable names**, not parameter-heavy names.

---

## Required core analysis exports

Always export:

- `roi_metrics.csv`
- `roi_spectrum.json`
- `roi_trace.csv`
- `heatmap_<band_id>.png`
- `heatmap_<band_id>.csv`

### Per-cell exports
Always export:
- one combined per-cell trace file
- one combined per-cell spectrum file

Do **not** create separate files per cell.

---

## Export content details

### `roi_metrics.csv`
Should include at least:
- ROI label
- ROI mode (manual / whole-frame)
- overall ROI quality score
- confidence label
- sub-scores
- valid-cell count
- rejected-cell count
- rejection penalty contribution
- reported peak count
- top peak frequencies
- support fractions
- analysis mode

### `roi_spectrum.json`
Should include:
- final aggregated ROI spectrum
- reported peaks only

Do **not** include all per-cell support and rejection detail in this file.

### `roi_trace.csv`
Should include:
- aggregated ROI trace
- sampled at native frame rate

### Combined per-cell traces file
Should include:
- cell identifier
- time axis or frame index
- per-cell trace values
- cell quality flags

### Combined per-cell spectra file
Should include:
- cell identifier
- frequency axis or bin index
- per-cell spectral values
- per-cell quality flags

### `heatmap_<band_id>.csv`
Should include:
- final normalized band-energy grid only

Not included in this file:
- raw band energy
- confidence mask
- extra layers

Those may be stored elsewhere if needed, but not in the main heatmap CSV.

### `heatmap_<band_id>.png`
Should show:
- heat values using the shared render-wide scale
- low-confidence cells in monochrome black/white

---

## Sidecar / results recording

The render sidecar/results should include quantitative-analysis metadata.

It must record:
- whether analysis was enabled
- ROI mode
- ROI geometry
- ROI quality score
- confidence label
- all final reported peaks
- auto-band selection results
- all generated heatmap band definitions
- artifact paths

It should also record intermediate analysis decisions, including:
- cell rejection statistics
- auto-band merge steps
- suppressed low-confidence peak reasons

### Important boundary
This data is for:
- render record
- diagnostics
- later offline comparison

It is not intended to make the app itself perform in-app run-to-run comparison in v1.

---

## Advanced settings

Expose a **small curated set** of quantitative-analysis settings in Advanced UI.

Examples:
- minimum cell-support fraction
- ROI quality cutoff
- low-confidence suppression threshold
- auto-band count (default 5)
- manual-band definitions
- export advanced/internal analysis files

Do **not** expose every scoring coefficient or every internal weighting parameter in v1.

---

## UI placement

Quantitative analysis controls should live inside an **Analysis** section.

Behavior:
- basic controls visible
- advanced options collapsed

### Completed-render screen
Do **not** display quantitative analysis outputs in the completed-render screen.

The outputs are exported artifacts only.

---

## Confidence and reporting rules

### If ROI quality is high enough
Export:
- ROI spectrum
- reported peaks
- heatmaps
- traces
- per-cell exports

### If ROI quality is below threshold
Export:
- trace
- heatmaps
- quality warnings
- supporting files

Suppress:
- peak reporting

---

## Architecture modules to add

### 1. ROI Analysis Engine
Owns:
- ROI subdivision
- cell-level trace creation
- cell-level spectrum creation
- robust ROI spectrum aggregation
- peak support logic
- ROI quality scoring

### 2. Heatmap Engine
Owns:
- denser heatmap grid generation
- band-energy calculation
- normalization
- confidence masking
- heatmap PNG/CSV generation

### 3. Analysis Artifact Writer
Owns:
- CSV/JSON/PNG writing
- fixed file naming
- sidecar/results registration
- diagnostics-bundle inclusion

### 4. Analysis Configuration Model
Owns:
- mode selection
- auto/manual band selection
- threshold settings
- ROI mode
- export controls

---

## Internal analysis hierarchy

The quantitative analysis pipeline should be structured as:

1. phase components by scale/orientation
2. per-cell weighted trace
3. per-cell spectrum
4. robust ROI spectrum
5. supported peak set
6. adaptive merged bands
7. denser heatmap band-energy grids
8. artifact export

This hierarchy should be shared where possible between ROI analysis and heatmap generation.

---

## Decision rationale

### Why ROI-centric, not whole-frame-global by default
NVH interpretation usually cares about a component or suspect region. Whole-frame-only metrics are too contaminated by irrelevant motion, lighting changes, and background structure.

### Why robust aggregation of cell spectra
Spectral aggregation is more stable than averaging raw time traces and better reflects the frequency-domain focus of this tool.

### Why one ROI in v1
It keeps UI, data structures, artifact layout, and quality accounting simpler while still adding strong quantitative value.

### Why no absolute displacement claims
The tool is video-only. Without calibrated sensor fusion or camera-motion calibration, absolute motion claims would be misleading.

### Why normalized heatmaps
Normalized band-energy maps show band-specific activity better than raw energy maps and are more comparable across regions.

### Why quality gating matters
A strong-looking spectrum from a weak or inconsistent ROI can be misleading. Quality must be part of the output, not an afterthought.

### Why analysis artifacts are exported instead of shown in-app
This keeps the application within its instrument-style render-focused scope while still preserving enough data for external engineering review and later comparison workflows.

---

## Summary of locked decisions

- video-only quantitative analysis
- analysis runs only alongside render
- one ROI per run
- no ROI falls back to visible Whole-frame ROI
- ROI uses same shape tools as exclusion zones
- robust ROI spectrum from per-cell spectra
- per-cell traces derived from phase pipeline directly
- per-cell adaptive weighting of scale/orientation components
- composite cell quality score with hard fails
- rejected cells excluded from aggregation but still penalize ROI quality
- ROI quality score includes sub-scores
- suppress peak reporting when ROI quality too low
- peak reporting requires cell-support rule
- export all supported peaks, highlight top ones separately if needed
- peak ranking uses fixed composite formula
- auto-bands come from ROI spectrum only
- auto-band widths are adaptive
- nearby peaks merge unless clearly separated
- auto-band count is user-adjustable, default 5
- manual-band mode uses low/high bounds, maximum 5 bands
- heatmap uses independent denser grid
- heatmap intensity = normalized band energy
- low-confidence heatmap cells shown as monochrome black/white
- one shared robust percentile display scale across heatmaps in a render
- artifacts written after render completes
- render success can coexist with analysis export failure warning
- artifacts in main output folder
- fixed filenames
- analysis artifacts automatically included in diagnostics bundle
- completed-render screen does not display quantitative outputs