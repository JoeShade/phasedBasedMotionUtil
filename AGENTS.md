# AGENTS.md

## Repository at a glance

This repository is the Phase-based Motion Amplification Desktop Utility: a PyQt6 shell, a testable core domain layer, and a separate spawned worker for heavy render execution.

- `src/phase_motion_app/app`: UI, shell-side validation, and worker supervision
- `src/phase_motion_app/core`: domain models, pre-flight, sidecars, diagnostics, storage, media helpers, numeric processing
- `src/phase_motion_app/worker`: worker bootstrap plus render worker
- `tests`: regression suite

`systemDesign.md` is the design source of truth.

## Expected working method

- Make small, explicit changes.
- Prefer test-first or test-in-lockstep development.
- Keep UI concerns in `app`, reusable logic in `core`, and heavy render/tool execution in `worker`.
- Do not leave code, tests, docs, and design half-updated. Move them together.

## Comments and explainers

- Add comments only where logic is non-obvious.
- Keep module docstrings and function/class explainers for state machines, IPC rules, scheduler math, sidecar boundaries, and numeric heuristics.
- Avoid low-value narration comments.

## Testing expectations

- Install dev dependencies with `python -m pip install -e .[dev]`.
- Add or update tests with every substantive behavior change.
- Prefer regression tests for bug fixes.
- Run targeted tests while iterating.
- Run `python -m pytest` before finalizing.

## Using `systemDesign.md`

- Treat `systemDesign.md` as authoritative for supported behavior and boundaries.
- Update it whenever architecture, supported workflow, outputs, or repository rules change.
- If code and design disagree, either fix the code or rewrite the design in the same change. Do not leave drift for later.

## Documentation expectations

- Update `README.md` when install/run/test instructions, capabilities, or operator-visible behavior change.
- Keep `docs/architecture-notes.md` short and implementation-focused.
- Use `docs/deviations.md` only for live temporary mismatches.
- Do not keep stale review notes, superseded design drafts, or one-off planning debris in the tracked tree.

## Validation and review

- After implementation, verify the touched behavior directly and run the relevant tests.
- Before finalizing, do a self-review for correctness, architecture fit, documentation drift, and test coverage gaps.
- Report assumptions, residual risks, and follow-up items explicitly.

## Repository hygiene

- Avoid committing generated runtime data, scratch output, ad-hoc logs, or disposable review artifacts.
- Remove obsolete references when deleting files or features.
- Keep developer utilities in `tools/`, not under `tests/`.

## Source footer

- Applicable maintained source files in this repository include Python modules, Python tests, Python developer scripts, and `run.bat`.
- Preserve the standard footer block already present at the bottom of those files.
- New applicable source files should include the same footer block. The canonical content was migrated from the removed `source-code-footer.txt`; copy an existing source file and adapt only the comment syntax if needed.
- Do not apply the footer to assets, binaries, generated files, Markdown docs, lockfiles, or machine-readable files where extra footer text would be invalid or harmful.
