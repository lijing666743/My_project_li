# Figure Surface Lifecycle

Use this to separate manuscript figures from diagnostic plots and stale artifacts.

## Surfaces

- Final manuscript include targets: files referenced by the manuscript or intended to be referenced.
- Source assets: scripts, CSV/TSV/JSON data, schemas, and transformation notes needed to regenerate the figure.
- QA/diagnostic outputs: dashboard PNGs, tuning panels, quick visual checks, gray checks, and intermediate renderings.
- Legacy/provenance outputs: old or superseded figures retained only to explain history or enable rollback.

## Required Checks

- The final output path is explicit and distinct from diagnostic-only plots.
- The script and source data can regenerate the final figure.
- Source denominator, units, seeds, horizon, and transformations match the caption.
- Dashboard or diagnostic plots are not promoted into manuscript evidence without claim/evidence gate approval.
- Stale root-level files do not shadow the current final output.

## Output Requirement

For any manuscript-facing Python figure, report:

- final include target
- source script
- source data
- QA/diagnostic output
- legacy/provenance artifacts
- unsupported claims
