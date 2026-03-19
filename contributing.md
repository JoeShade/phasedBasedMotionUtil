# Contributing

`phase-based-motion-utility` accepts focused improvements that fit the repository's current scope.

## Development Setup

- Use Python 3.10 or newer.
- Clone the repository and open a shell in the checkout root.
- Install the package and development dependencies with:

```powershell
python -m pip install -e .[dev]
```

This installs the project in editable mode and includes the tools needed to run the test suite.

Optional CuPy acceleration is not required for general development. The CPU path remains supported and should continue to work even when GPU-related code changes are made.

## Running Tests

Run the full suite before opening a pull request:

```powershell
python -m pytest
```

Targeted test runs are fine while iterating, but the repository expectation is that the full suite passes before final review.

## Design and Documentation Expectations

- Treat `systemDesign.md` as the repository source of truth for supported behavior and boundaries.
- Keep code, tests, and docs aligned in the same change.
- Update `README.md` when install steps, operator-visible behavior, or capabilities change.
- Update `systemDesign.md` when architecture, supported workflow, or repository rules change.
- Keep `docs/architecture-notes.md` short and implementation-focused.
- Use `docs/deviations.md` only for live temporary mismatches between code and design.
- Do not introduce features that conflict with the documented non-goals without discussion first.

`AGENTS.md` is the repository workflow guide for contributors and automated agents. Follow it for repository-specific expectations around testing, documentation, and change hygiene.

## Pull Requests

- Keep changes small, explicit, and reviewable.
- Add or update tests with substantive behavior changes.
- Re-check the public docs and design docs before submitting.
- Do not commit generated runtime output, scratch files, diagnostics bundles, or ad-hoc logs.
- If a change would expand the product beyond its current narrow scope, open an issue and get agreement before investing in a large implementation.

## Issues

Open an issue for bugs, regressions, usability problems, or scoped feature proposals.

- For bugs, include the environment you used, the steps to reproduce, and the expected versus actual result.
- Include relevant logs, diagnostics, or sidecar details when they help explain the problem.
- For feature requests, describe the operator problem first and explain how the proposal fits the current project scope.
