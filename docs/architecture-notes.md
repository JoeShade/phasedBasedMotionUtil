# Architecture Notes

The codebase is split into three layers so the heavy video engine never needs to run in the PyQt process:

- `phase_motion_app.core`: pure logic, schema validation, state handling, pre-flight checks, storage policy, IPC contracts, and watchdog logic
- `phase_motion_app.worker`: the separate render worker process and its process-facing helpers
- `phase_motion_app.app`: the PyQt6 shell and thin UI adapters around the tested core

The first implementation slices focused on pure logic with tests because those rules are the most tightly specified in `systemDesign.md` and they had to stay stable as the GUI grew.

The current masking contract stores exclusion zones in source-frame pixel coordinates and rasterizes the final feathered mask in output space. That keeps the editor, sidecar intent, and later compositing path aligned with Section 15 of the design document.

The shell now uses a dedicated `RenderSupervisor` around the spawn worker, loopback socket, and watchdog policy. That keeps process polling, terminal-message checks, and UI status updates outside the PyQt event handlers.

Convenience persistence lives in `settings_store.py` and intentionally remains separate from sidecar loading. The shell restores last-used settings and global diagnostics/temp preferences without treating them as reproducible engineering intent.

Diagnostics bundle writing happens in the worker so success and failure paths both leave a JSONL log plus a structured bundle manifest under the run diagnostics directory.
