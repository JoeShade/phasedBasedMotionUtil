# phase-based-motion-utility

[![version](https://img.shields.io/badge/version-0.1.0-1f6feb.svg?style=flat-square)](./pyproject.toml)
[![python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?style=flat-square)](./pyproject.toml)
[![desktop](https://img.shields.io/badge/interface-PyQt6-41CD52.svg?style=flat-square)](./README.md)
[![license](https://img.shields.io/badge/license-AGPL--3.0-blue.svg?style=flat-square)](./LICENSE)

`phase-based-motion-utility` is an offline PyQt6 desktop utility for phase-based motion amplification on recorded video. It provides a desktop shell, a testable core domain layer, and a supervised worker process for the render path so operators can review a clip, configure a narrow amplification workflow, and produce an amplified MP4 with matching metadata.

For background on the underlying method, see [Wadhwa et al. (2013)](https://doi.org/10.1145/2461912.2461966): N. Wadhwa, M. Rubinstein, F. Durand, and W. T. Freeman, "Phase-Based Video Motion Processing," *ACM Transactions on Graphics*, 32(4), Article 80, pp. 1-10.

## Disclaimer

This project is an engineering and review tool, not a general-purpose video editor. Outputs and analysis artifacts should be reviewed before you rely on them, and the current scope is intentionally limited to a narrow offline workflow.

## Install

### Python requirement

Python 3.10 or newer is required.

### Clone the repository

Clone the repository from GitHub using the HTTPS or SSH URL shown on the project page, then open a shell in the checkout root.

### Default install

Install from the repository root with an editable install:

```powershell
python -m pip install -e .
```

This is the main install path for local use. It installs the application entrypoint and the runtime dependencies declared in `pyproject.toml`.

### Runtime dependencies

The default install brings in these runtime packages through pip:

- `PyQt6`
- `jsonschema`
- `numpy`
- `psutil`
- `static-ffmpeg`

### `ffmpeg` and `ffprobe`

The application resolves `ffmpeg` and `ffprobe` through `static-ffmpeg` by default, so the standard install path does not require a separate manual toolchain setup.

If you need to override the detected tools, set both environment variables before launch:

```powershell
$env:PHASE_MOTION_FFMPEG="C:\path\to\ffmpeg.exe"
$env:PHASE_MOTION_FFPROBE="C:\path\to\ffprobe.exe"
```

Partial overrides are rejected. Setting only one of these variables is treated as a configuration error.

### Optional GPU acceleration

GPU acceleration is optional. The project can use CuPy for selected compute-heavy paths, but the CPU path remains supported and authoritative when GPU acceleration is unavailable or disabled.

One example install path for a CUDA 12 environment is:

```powershell
python -m pip install cupy-cuda12x nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
```

If you do not need GPU acceleration, skip this step.

### Development dependencies

If you plan to run tests or contribute changes, install the development extras:

```powershell
python -m pip install -e .[dev]
```

### Launch the app

After installation, start the desktop app with:

```powershell
phase-motion-app
```

On Windows, a source checkout also includes a convenience launcher:

```powershell
.\run.bat
```

`run.bat` is intended for Windows source checkouts and prefers `pyw` or `pythonw` when available so the shell starts as a GUI process instead of inheriting console-style Python launcher behavior. It falls back to `py` or `python` if needed.
When launched from a source checkout, the shell also loads repo-level branding assets such as `assets/programIcon.ico`.

## Features / Current Capabilities

- Offline processing of recorded video sources
- Phase-based motion amplification with a supervised worker process for the heavy render path
- Source probing, SHA-256 fingerprinting, and first/last-frame extraction before render
- Drift review before rendering so operators can confirm the source state
- Static include and exclude mask zones with feathering
- One optional quantitative-analysis ROI, plus render-time analysis artifact export
- Shell-side dry-run pre-flight followed by worker-side authoritative pre-flight
- One active render at a time with watchdog supervision and explicit terminal outcomes
- Paired output finalization with an MP4 render, matching JSON sidecar metadata, and diagnostics artifacts when produced
- Optional CuPy-backed acceleration for selected kernels, with explicit CPU fallback

## Non-goals / Limitations

- This is not a general-purpose video editor or batch transcoding tool.
- The workflow is limited to recorded video. There is no live capture or live preview mode.
- The project supports phase-based amplification only.
- Only one render can run at a time.
- Output is currently MP4 only, and audio is stripped from the rendered result.
- Mask geometry is static in source-space. Moving masks and tracked ROIs are out of scope.
- Only one optional quantitative-analysis ROI is supported.
- Quantitative analysis runs alongside a render. There is no analysis-only mode.
- Processing and final output share one effective render resolution. There is no separate operator-controlled output resize.

## How to Run

1. Launch the app with `phase-motion-app`, or use `.\run.bat` from a Windows source checkout.
2. Choose a recorded video source.
3. Let the app probe the source, fingerprint it, and prepare the first/last-frame review.
4. Review drift, define any static mask zones, and optionally add one quantitative-analysis ROI.
5. Run pre-flight checks, then start the render.
6. Review the resulting MP4, matching sidecar metadata, and any diagnostics artifacts produced for the run.

When you run the app from a source checkout, repo-local directories such as `input/`, `output/`, `temp/`, and `diagnostics/` may be used for runtime work. Those runtime folders and similar temporary artifacts are not the main tracked source content of the repository.

## Development / Tests

Install the development extras with:

```powershell
python -m pip install -e .[dev]
```

Run the full test suite with:

```powershell
python -m pytest
```

When behavior or architecture changes, keep the implementation, tests, and documentation aligned in the same change. `systemDesign.md` is the source of truth for repository design and supported behavior.

## Repository Layout

- `src/phase_motion_app/app`: PyQt6 shell, dialogs, shell-side validation, and worker supervision
- `src/phase_motion_app/core`: domain models, sidecars, diagnostics, toolchain helpers, storage rules, pre-flight logic, and numeric processing
- `src/phase_motion_app/worker`: spawned worker bootstrap and the render worker
- `tests`: regression suite
- `tools`: developer utilities

## Supporting Docs

- [systemDesign.md](systemDesign.md): authoritative design and behavior reference for the repository
- [docs/architecture-notes.md](docs/architecture-notes.md): short implementation-focused notes that supplement the design document
- [docs/deviations.md](docs/deviations.md): active temporary mismatches between code and design, if any
- [AGENTS.md](AGENTS.md): repository workflow guide for contributors and automated agents

## Contributing

Contributions are welcome when they fit the repository's current scope. Start with [contributing.md](contributing.md) for setup, testing, and pull request guidance, and use [AGENTS.md](AGENTS.md) as the workflow guide for repository-specific expectations.

## License

This project uses the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for the full license text.
