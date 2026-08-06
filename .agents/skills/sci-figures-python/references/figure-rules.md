# Figure Rules

## Required Contract

Every manuscript figure should have:

- Source data path or source artifact
- Unit, denominator, and time horizon
- Transformation steps
- Exact output path
- Caption claim
- Statement of what the figure does not prove

## Visual Standards

- Use readable axis labels with units.
- Prefer colorblind-safe palettes.
- Keep chart type matched to data type and claim.
- Export a vector format when the target venue supports it.
- Keep scripts deterministic unless randomness is part of the analysis and seed is fixed.
- Record `seed` or `seed_set` for seed-dependent figures.

## Review Checks

- Does the caption overclaim the data?
- Are error bars or confidence intervals actually computed?
- Are baseline names and metric definitions consistent with the manuscript?
- Can another run regenerate the same figure from the recorded data?
- Does the caption avoid robustness or superiority wording when the figure is single-seed only?
