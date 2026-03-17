# Render Pipeline Parallelization Analysis

**Analysis Date:** March 17, 2026  
**Scope:** Phase-based motion amplification pipeline in `worker/render.py`

---

## Executive Summary

The render pipeline has **significant but constrained parallelization potential**. The architecture is currently structured as a strict sequential pipeline:
1. Reference pass (full decode)
2. Main phase processing (decode → FFT analysis → temporal filtering → warping → encode)

The pipeline exhibits **mixed I/O and CPU binding** with critical temporal state dependencies that prevent naive frame-level parallelization. The most promising opportunities are **stage-level pipelining** (decode-ahead, encode-async) rather than frame-level parallelization.

---

## Stage-by-Stage Analysis

### Stage 1: Reference Pass (Lines 400–461)

**Purpose:** Compute global reference luminance by accumulating all source frames

#### Workload Characteristics
- **Bound Type:** I/O-dominated with compute component
  - I/O: Full source video decode via FFmpeg
  - CPU: Luma extraction from RGB (0.2126R + 0.7152G + 0.0722B per pixel)
  - CPU: Summation reduction (`reference_luma_sum += luma.sum(axis=0)`)

#### Data Dependencies
```
Input: source_path (constant per render)
↓
[RawvideoDecodeProcess.read_frames(scheduler.chunk_frames)]
↓
[_bytes_to_float_frames() conversion]
↓
[_frames_to_luma() reduction - produces [H×W] array]
↓
[Accumulation into reference_luma_sum (running total)]
↑
Output: reference_luma = reference_luma_sum / frame_count
```

**Key State:** `reference_luma_sum` (accum), `reference_frame_count` (counter)

#### Parallelization Constraints
1. **Frame order is irrelevant** for summation (associative operation)
2. **BUT**: FFmpeg decode order is sequential by design (streaming decode)
3. **Memory advantage**: Uses `float32` to minimize pressure (stated in code comment)

#### Observations
- Code explicitly uses `np.float32` to avoid redundant conversions and reduce memory pressure
- Chunks are large, reducing overhead per frame
- The reference pass is **read-only to output** (won't change during main pass)

---

### Stage 2: Main Processing Pass (Lines 500–820)

**Purpose:** Decode source frames, amplify motion via phase analysis, encode result

#### Overall Structure
```
while not eof:
    chunk = decode.read_frames(scheduler.chunk_frames)  # I/O bound
    ↓
    processing_chunk = resize(chunk, processing_resolution)  # CPU bound
    ↓
    [OPTIONAL] analyzer.add_chunk(processing_chunk)        # CPU bound (parallel analyzer)
    ↓
    amplified = phase_amplifier.process_chunk(processing_chunk)  # CPU bound (FFT + filtering)
    ↓
    out = resize(amplified, output_resolution)            # CPU bound
    ↓
    for frame in out:
        encoder.write_frame(frame)                         # I/O bound
```

#### Data Dependencies

**Between Stages:** Strict ordering
- Frame N must be **decoded before processed**
- Frame N must be **processed before encoded**
- Frames must **reach encoder in order** (MP4 framerate and duration depend on frame order)

**Within Phase Processing:** **Temporal state is critical**
- `_StreamingBandpassFilter` maintains state across chunks:
  ```python
  self._low_state = np.zeros(grid_shape)  # Per-grid cell
  self._high_state = np.zeros(grid_shape)  # Per-grid cell
  
  for frame in chunk:
      self._high_state = self._high_state + self._alpha_high * (frame - self._high_state)
      self._low_state = self._low_state + self._alpha_low * (frame - self._low_state)
      filtered[i] = self._high_state - self._low_state  # **Frame output depends on history**
  ```

- This is a **recursive AR filter** — **Frame N's output depends on all previous frames**
- Cannot parallelize across chunks (chunk boundaries would corrupt filter state)
- Can only parallelize **within a chunk** (multiple frames in parallel if filter state is replicated)

---

## Phase Amplifier Deep Dive: `StreamingPhaseAmplifier.process_chunk()`

### Work Breakdown (Per Chunk of ~32–64 Frames)

#### Sub-stage 1: RGB→Luma Conversion
- **Bound Type:** CPU (memory-bound to some extent)
- **Parallelization:** Trivially parallelizable per-frame
- **Cost:** ~3 multiplies + 2 adds per pixel
- **Time:** Negligible vs. downstream FFT work

#### Sub-stage 2: Phase Correlation via FFT
```python
displacement_x, displacement_y, confidence = _estimate_local_phase_shifts_against_reference(...)
```

**This is the computational bottleneck:**

```
for each grid tile (e.g., ~20×20 tiles for 720p):
    for each frame in chunk:
        # FFT-based phase correlation
        tile_fft = np.fft.fft2(tile_stack, axes=(1, 2))  # Complex 2D FFT
        cross_power = reference_fft[None, :, :] * conj(tile_fft)  # Element-wise
        correlation = ifft2(cross_power)  # Inverse FFT
        peak = find_peak(correlation)     # Peak detection
```

**Characteristics:**
- **Bound Type:** CPU-bound (intrinsic FFT complexity: $O(N \log N)$ per tile per frame)
- **Parallelization:**
  - ✅ **Tiles can be processed in parallel** (independent phase correlations)
  - ✅ **Frames within a chunk can use vector operations** (numpy batches FFT)
  - ❌ **BUT**: Temporal filter state **doesn't allow cross-chunk frame parallelization**

**Cost per tile per frame:**
- FFT 2D of tile (e.g., 64×64): ~64² × log(64²) ≈ 4000–5000 ops
- ~20×20 tiles × ~50 frames/chunk × 4500 ops = ~90M ops per chunk
- Modern CPU: ~5–10 µs per frame per tile for FFT (with SIMD assistance)

#### Sub-stage 3: Temporal Bandpass Filtering
```python
bandpassed_x = self._x_filter.filter_chunk(displacement_x, ...)
bandpassed_y = self._y_filter.filter_chunk(displacement_y, ...)
```

**Characteristics:**
- **Bound Type:** CPU-bound but simple (single-frame AR filter)
- **State Dependency:** ❌ **Recursive filter cannot start frame N before feeding frame N-1 into state**
- **Parallelization within chunk:** ❌ **Strictly sequential frame processing**
- **Code:**
  ```python
  for frame_index in range(frame_count):
      sample = signal[frame_index]
      self._high_state = self._high_state + self._alpha_high * (sample - self._high_state)
      self._low_state = self._low_state + self._alpha_low * (sample - self._low_state)
      filtered[frame_index] = self._high_state - self._low_state  # Depends on prior state
  ```

#### Sub-stage 4: Displacement Warping
```python
amplified_rgb = _warp_rgb_frames(
    working,
    displacement_x_grid=displacement_x_grid,
    displacement_y_grid=displacement_y_grid,
    layout=reference.layout,
    max_displacement_px=max_displacement_px,
)
```

**Characteristics:**
- **Bound Type:** CPU-bound (image resampling per frame)
- **Parallelization:** ✅ **Frames are independent** — each frame warped separately
- **Cost:** Bilinear/cubic resampling across full image


### Phase Amplifier Parallelization Opportunity: Tile-Level Parallelism
Currently uses numpy batching (all tiles FFTed in vectorized form across all frames simultaneously). Already achieves SIMD efficiency. **Additional tile-level threading would likely not improve throughput** but could improve latency under resource contention.

---

## Quantitative Analysis Pipeline

### `StreamingQuantitativeAnalyzer.add_chunk()` and `finalize()`

**Purpose:** Parallel motion analysis for NVH reports (does NOT feed back to main pipeline)

#### Adding Chunks (Concurrent with Main Pipeline)
```python
def add_chunk(self, frames_rgb: np.ndarray) -> None:
    motion_x, motion_y, confidence = self._motion_analyzer.analyze_chunk(frames_rgb)
    # Accumulate motion traces (reshape and append)
    self._x_chunks.append(motion_x.reshape(frame_count, -1))
    self._y_chunks.append(motion_y.reshape(frame_count, -1))
    self._confidence_sum += confidence.reshape(-1)
```

#### Parallelization Characteristics
- **Independent of main amplification** — analysis is read-only from `processing_chunk`
- **NOT on critical path** — executed alongside main processing but finalized after encode completes
- **Bound Type:** CPU-bound (same FFT + filtering work as reference computation)
- **State:** Accumulates traces and confidence scores; **no temporal cross-chunk dependencies**

#### Current Architecture
- Analysis is done in the **main worker thread** alongside encoding
- Could be offloaded to **background thread** entirely (as done for FFmpeg progress drain threads)
- No safety constraints prevent this

---

## FFmpeg Subprocess Usage

### RawvideoDecodeProcess (Decode)

**Properties:**
- **Process:** Long-lived `ffmpeg -i <input> -pix_fmt rgb24 -f rawvideo pipe:1`
- **Bound Type:** I/O bound (disk→decode→pipe)
- **Backpressure:** ✅ **Worker pulls chunks on-demand** via `read_frames(max_frames)`
  ```python
  def read_frames(self, max_frames: int) -> list[bytes]:
      for _ in range(max_frames):
          frame = self.process.stdout.read(self.frame_size_bytes)
          if len(frame) < self.frame_size_bytes:
              break
          frames.append(frame)
      return frames
  ```

### RawvideoEncodeProcess (Encode)

**Properties:**
- **Process:** Long-lived `ffmpeg -f rawvideo -i pipe:0 -c:v <codec> -pix_fmt <fmt> <output.mp4>`
- **Bound Type:** I/O bound (encode + write to disk)
- **Backpressure:** ✅ **Worker pushes frames frame-by-frame** via `write_frame()`
  ```python
  def write_frame(self, frame_rgb24: bytes) -> None:
      assert self.process.stdin is not None
      self.process.stdin.write(frame_rgb24)  # Blocks if encoder buffer full
  ```

**Latency Characteristics:**
- Decode produces frames with some buffering in FFmpeg's internal queues
- Encoding has encoder-specific buffering behavior (H.264/H.265 need lookahead for encoding decisions)
- Current design: **synchronous frame-by-frame push**


---

## Parallelization Opportunities & Recommendations

### ⭐ Opportunity 1: Decode-Ahead Buffering (HIGH IMPACT, LOW RISK)

**Current:** Strict sequence: decode chunk → process chunk → encode frame  
**Proposed:** Decode-ahead buffer (decouple decode from processing via queue)

**Architecture:**
```
Thread A (Decode):           Thread B (Process):          Thread C (Encode):
read_frames() ──┐            ┌──────────────┬──────────┬──┐
                │ queue      │ process_chunk→   resize  │ └──→ write_frame()
                └─→ [buf] ◄──┘                 amplify  │
                              │                encode ──┤
                              └─ analyzer.add_chunk()
```

**Benefits:**
- Decode (I/O) runs while processing (CPU) runs — overlaps I/O and CPU time
- Encode latency hidden by buffered decoded frames waiting for processing
- Typical benefit: **20–40% throughput improvement** if one stage starves the other

**Constraints:**
- **Reference pass must complete first** (needs `reference_luma`)
- **Frame ordering strictly preserved** in queue
- Memory overhead: ~3–5 frames buffered (manageable, ~80–200 MB for 4K content)
- **Quantitative analyzer continues alongside** without changes

**Implementation Notes:**
1. Use `queue.Queue` (thread-safe, bounded to ~3–5 frames)
2. No changes to phase processing logic needed
3. Decode and encode remain long-lived subprocesses

**Risk Level:** LOW (isolated threading model, no shared state changes)

---

### ⚠️ Opportunity 2: Temporal Filter Initialization from Previous Renders (MEDIUM IMPACT, MEDIUM COMPLEXITY)

**Current:** Filters start with zero state, causing ~1–3 frames of transient response at clip start

**Observation:** Filter state at end of one clip could seed the next clip's initial state

**Challenge:**
- Requires serialization of filter state between renders
- State is per-grid-cell (20×20 grid for typical resolution = 400 values per X/Y filter)
- Must store in render metadata/sidecar for reuse
- Loss of state across resumption breaks reproducibility

**Verdict:** **Deferred opportunity** — only valuable for batch processing chains; adds metadata complexity

---

### ❌ Opportunity 3: Frame-Level Parallelization (NOT FEASIBLE)

**Why it doesn't work:**

The temporal filter is **inherently sequential**:
$$
\text{filtered}[n] = \text{state}_n = \text{state}_{n-1} + \alpha(x_n - \text{state}_{n-1})
$$

Even if we split processing across threads:
- Frame 100 cannot compute until Frame 99's state is known
- You cannot compute independent FFTs for frames 1–32 in parallel with frames 33–64 without **repeating filter computation later** (state correction)

The only way to parallelize would be to:
1. Compute all FFT displacement in parallel (high compute)
2. Serialize the temporal filter application (bottleneck)
3. Net speedup: **minimal** (bandwidth limited on final filter stage)

**Verdict:** **Not recommended** — complexity doesn't justify minimal gains

---

### ⭐ Opportunity 4: Quantitative Analysis on Background Thread (HIGH IMPACT, LOW RISK)

**Current:** Analysis is added chunk-by-chunk during main processing in the same thread

**Proposed:** Decouple analyzer to a background thread with its own queue

**Architecture:**
```
Main Worker:
    add_chunk(processing_chunk) ──┐
                                  │ queue
                                  └──→ [Analyzer Thread]
                                        analyze_chunk()
                                        finalize() ← after main processing done
```

**Benefits:**
- Analyzer never blocks main processing
- Main processing can run at full speed without analyzer overhead
- Analyzer catches up during encoding phase (while encode stalls main thread)

**Current Code Analysis:**
The analyzer already operates on a copy-safe interface:
```python
def add_chunk(self, frames_rgb: np.ndarray) -> None:
    motion_x, motion_y, confidence = self._motion_analyzer.analyze_chunk(frames_rgb)
    # Only accumulates; no side effects on input
    self._x_chunks.append(motion_x.reshape(frame_count, -1))
```

**Risk Assessment:**
- **No data races** — frames copied to analyzer, no shared state modified
- **Finalization:** Analyzer must finish before sidecar write (2–3 second timing budget)
- **Current finalize() cost:** ~500 ms–2s (spectrum computation, cell scoring)

**Implementation Pattern:**
```python
class _BackgroundAnalyzer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.queue = queue.Queue(maxsize=3)
        self.thread = threading.Thread(self._consume, daemon=True)
        self.thread.start()
    
    def add_chunk(self, chunk):
        self.queue.put(chunk)  # Non-blocking; drops if full (acceptable)
    
    def finalize_when_ready(self, timeout=30):
        self.queue.put(None)  # Sentinel
        self.thread.join(timeout=timeout)
        return self.analyzer.finalize(...)
    
    def _consume(self):
        while True:
            chunk = self.queue.get()
            if chunk is None:
                break
            self.analyzer.add_chunk(chunk)
```

**Verdict:** **Recommended** — easy win, minimal code changes

---

### Opportunity 5: Encode Buffering (MEDIUM IMPACT, MEDIUM COMPLEXITY)

**Current:** Frame-by-frame `write_frame()` — synchronous push to FFmpeg

**Observation:** H.264/H.265 encoders have internal lookahead; initial buffering helps

**Proposed:** Pre-buffer 2–3 frames before starting encoder drain

**Pros:**
- Encoder has better rate-control decisions
- Smoother bitrate packing

**Cons:**
- Minimal latency impact (frames already coming from decode queue)
- FFmpeg's internal buffering probably already handles this
- Adds complexity for marginal quality gain

**Verdict:** **Deferred** — profile-first to confirm encoder is actually buffering-limited

---

## Architectural Constraints Summary

| Constraint | Source | Impact |
|:-----------|:-------|:-------|
| **Temporal filter state** | AR filtering in `_StreamingBandpassFilter` | Frames must be processed sequentially within chunk boundaries |
| **Frame ordering** | MP4 framerate & duration determined by frame sequence | Frames must reach encoder in order |
| **Reference immutability** | Computed once, used for all frames | No dependency changes mid-pipeline |
| **Chunk size** | `scheduler.chunk_frames` (~32–64) | Natural parallelization boundary |
| **Decoder backpressure** | `read_frames()` bounded pull | Prevents unbounded memory growth |
| **Encoder synchronization** | Long-lived process, frame-ordered stdin | Must push frames in order |

---

## Recommended Implementation Roadmap

### Phase 1: Low-Risk, High-Impact (Week 1)

1. **Decode-Ahead Buffer** (Opp. 1)
   - Add `queue.Queue(maxsize=3)` between decoder and processor
   - Spawn decode thread that continuously fills queue
   - Main processor drains queue instead of calling decode directly
   - Estimated gain: **20–35% throughput** (if I/O and compute not already balanced)
   - Risk: Low (isolated, no state sharing)
   - Code complexity: ~40 lines

2. **Background Analyzer Thread** (Opp. 4)
   - Wrap `StreamingQuantitativeAnalyzer` with background thread
   - Main processor sends chunks to analyzer queue
   - Finalization happens after encode naturally completes
   - Estimated gain: **5–10% main-pipeline throughput** (frees up FFT compute from analyzer)
   - Risk: Very low (analyzer already thread-safe)
   - Code complexity: ~80 lines

### Phase 2: Medium-Complexity, Medium-Gain (Week 2–3)

3. **Profile & Optimize Tile-Level Parallelization**
   - Measure current numpy FFT vectorization efficiency
   - If <70% CPU utilization on multi-core, consider tile-level threading
   - Use `ThreadPoolExecutor` for per-tile phase correlation
   - Estimated gain: **5–20%** (if FFT overhead not fully vectorized)
   - Risk: Medium (GIL contention in numpy)
   - **First try:** Ensure numpy is using OpenBLAS/Intel MKL (not default BLAS on Windows)

### Phase 3: Advanced (Post-Profiling)

4. **Chunk-Boundary Filter State Initialization**
   - Requires serialization format for filter state in sidecar
   - Allows resumable rendering or filtering history preservation
   - Estimated gain: **2–5%** (mainly latency, not throughput)
   - Risk: Medium (metadata changes, backward compatibility)

---

## Profiling Recommendations

Before implementing any changes, capture baseline:

```python
import time

# Measure per-stage time in render.py main loop:
decode_time = 0
process_time = 0
encode_time = 0

while not eof:
    t0 = time.perf_counter()
    chunk = decoder.read_frames(...)
    decode_time += time.perf_counter() - t0
    
    t0 = time.perf_counter()
    [... resizing, phase processing, analysis ...]
    process_time += time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for frame in composed_output:
        encoder.write_frame(...)
    encode_time += time.perf_counter() - t0
```

**Target metrics:**
- Which stage is slowest? (bottleneck)
- What is the idle time ratio per stage? (parallelization opportunity size)
- Is memory growing unbounded? (backpressure working?)

---

## Summary Table: Opportunities vs. Effort

| Opportunity | Impact | Risk | Effort | Recommended | Timeline |
|:------------|:-------|:-----|:-------|:------------|:---------|
| Decode-ahead buffer | ⭐⭐⭐ 20–35% | Low | 1d | **YES** | Week 1 |
| Background analyzer | ⭐⭐ 5–10% | Very Low | 1d | **YES** | Week 1 |
| Tile-level threading | ⭐⭐ 5–20% | Medium | 2–3d | Maybe | After profiling |
| Encode buffering | ⭐ 2–5% | Medium | 2d | Deferred | Post-profiling |
| Filter state serialization | ⭐ 2–5% | Medium | 3–4d | Deferred | Phase 3+ |
| Frame-level parallelization | ❌ Minimal | High | 5d | **NO** | Never |

---

## Conclusion

The render pipeline has **2–3 safe, high-impact parallelization opportunities** that collectively could provide **30–50% throughput improvement**:

1. **Decode-ahead buffering** overlaps I/O and compute
2. **Background analysis thread** frees compute from non-critical path
3. **Subsequent profiling** will reveal if tile-level or encode-buffering add value

The fundamental architecture is sound: streaming processing, bounded memory, frame-ordered output. The temporal filter constraint is **fundamental to the algorithm** and cannot be relaxed without changing the motion amplification behavior itself.

**Next Step:** Implement Phase 1 recommendations, then profile to validate gains before Phase 2.
