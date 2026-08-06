# Diagnostic Dashboard Boundary

Use this reference for Python-generated dashboards, tuning panels, and training-process visuals.

Student-run research programs may automatically save dashboards under `plots/*dashboard*.png` while their backing metrics live under `logs/*.csv`. Treat the CSV as the data source and the dashboard as a diagnostic view unless the project explicitly admits it as manuscript evidence.

## Classification

- Diagnostic dashboard: used for manual inspection, tuning, smoke validation, or failure analysis.
- Evidence figure: used in a manuscript or response letter to support an admitted claim.
- Illustrative figure: explains workflow or architecture but does not prove a result.

## Required checks before manuscript use

- Locate the source data and run ID.
- Confirm manifest/config and summary CSV/JSON exist.
- Confirm the backing `logs/*.csv` exists when the dashboard comes from a student-friendly research program.
- Confirm units, denominator, horizon, seed/sample policy, and transformation.
- Confirm the repo admits the figure as manuscript evidence rather than diagnostic-only output.
- Confirm caption wording matches the admissible claim and does not upgrade proxy metrics.

## Guardrails

- Do not convert a tuning dashboard into a paper result figure by visual cleanup alone.
- Do not add uncertainty, error bars, or statistical markers unless computed from real data.
- Do not hide mixed or negative diagnostics by selecting only favorable panels.
- State what the figure does not prove.
