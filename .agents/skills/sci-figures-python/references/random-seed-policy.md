# Random Seed Policy

Use this policy for research experiments, training runs, simulations, randomized baselines, and random figure generation.

## Defaults

- If robustness testing is not required, use `seed=42`.
- If robustness testing is required and no seed count is specified, use `42, 23, 16, 15, 8, 4`.
- If robustness testing requires `n<=6` seeds, use the first `n` seeds from `42, 23, 16, 15, 8, 4`.
- If robustness testing requires `n>6` seeds, use `42, 23, 16, 15, 8, 4` first, then append date-derived seeds `YYYYMMDD00`, `YYYYMMDD01`, and so on until the requested count is reached.
- Project-specific accepted seed policies override this global default.

## Artifact Contract

- Record `seed` for a single-seed run.
- Record the full `seed_set` for multi-seed runs.
- Keep compared methods on the same seed set unless the project explicitly documents a justified exception.
- Write seed policy into config, manifest, logs, reports, and dashboard summaries when those artifacts exist.

## Claim Boundary

- Single-seed results can support smoke checks, diagnostics, implementation sanity, or restricted evidence only.
- Robustness, stability, generalization, convergence, or superiority claims require multi-seed or explicit statistical evidence.
- Do not hide the seed count when summarizing evidence.
