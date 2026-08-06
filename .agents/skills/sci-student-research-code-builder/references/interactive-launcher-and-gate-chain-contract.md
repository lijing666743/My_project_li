# Interactive Launcher And Gate Chain Contract

Use this reference for repeated manual failures around student-facing launchers, `method_id=all`, optional artifact readers, and serial audit/gate chains.

## Launcher Contract

- Test the real no-argument `python main.py` path with mock input.
- Cover top-level menu routing, default option, abort path, invalid retry, output path creation, and dashboard generation.
- When the default route changes, update menu text and mock-input tests in the same wave.
- Expert-only CLI helpers may exist, but they do not prove the human entry path works.

## All-Mode Contract

For `method_id=all` or serial multi-method execution, validate both layers:

- per-method logs, checkpoints, dashboards, and summaries
- aggregate/all-method comparison CSV, summary, and dashboard

If per-method artifacts already exist but aggregate outputs are missing, prefer a no-training repair/backfill entrypoint.

## Serial Gate Chain Contract

- Draw artifact dependencies before execution.
- Run downstream stages after the upstream CSV/JSON artifact exists.
- Downstream stages should fail blocked-not-crash when inputs are absent.
- A blocked downstream result should be diagnosed as dependency ordering vs real code/evidence failure.

## Shared Runtime Options

- Normalize shared runtime settings such as `device` once at the orchestrator layer.
- Print requested/resolved diagnostics.
- Map the resolved value into each branch's real config field instead of assuming all branches read the same name.

## Optional Artifact Readers

- Treat empty paths, directories, and nonexistent paths as absent inputs.
- Guard optional CSV/JSON reads with `exists() and is_file()`.
- Do not let `Path("")` resolve to `.` and become a directory-read attempt.
