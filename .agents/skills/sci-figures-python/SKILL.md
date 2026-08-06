---
name: sci-figures-python
description: "Use for SCI manuscript figures made with Python: reproducible data visualizations, publication figure scripts, chart audits, figure-caption alignment, visual evidence checks, and PNG/SVG/PDF export planning. Trigger when the user asks to create, revise, audit, or standardize paper figures from data. Do not use to invent data or create decorative figures without evidence."
---

# SCI Python Figures

## Core Rules

- A figure is evidence. Do not fabricate data, labels, comparisons, error bars, or statistical annotations.
- Before editing or generating a figure, identify the source data, units, denominator, horizon, sample size, and transformation steps.
- For seed-dependent experiment figures, identify the `seed` or full `seed_set` and whether the figure is single-seed diagnostic or multi-seed evidence.
- If repo governance files exist, read `docs/current_state.md`, `docs/next_task.md`, and `docs/claim_gate.md` before figure work.
- Prefer reproducible scripts over manual image edits.
- When axes, legends, annotations, or captions contain mathematical symbols, read the shared mathematical-notation-governance reference and align labels with the manuscript notation record. Ordinary numeric labels do not trigger the full notation workflow.
- Save figure scripts and outputs together so the figure can be regenerated.
- For training dashboards, tuning panels, diagnostic plots, or `plots/*dashboard*.png` outputs from student-run research programs, read `references/diagnostic-dashboard-boundary.md` before treating the visual as manuscript evidence.
- For manuscript-facing figures, read `references/figure-surface-lifecycle.md` and distinguish final include targets, source CSV/scripts, QA plots, and legacy/provenance artifacts.
- For repos under `Manuscript/`, read `C:\Users\didgm\.codex\skills\.shared\repo-governance\research-repo-surface-contract.md`: `figs/` stores manuscript figure sources/final assets and may contain only `legacy/`; `plots/` stores dashboards, diagnostics, and non-final plot outputs.
- For Nature-family, top-journal, or backend-selection figure tasks, read `references/nature-figure-backend-and-evidence-gate.md` before selecting Python, TikZ/PGFPlots, R, or another plotting route.
- For manuscript-facing data plots, read `C:\Users\didgm\.codex\skills\.shared\sci-figure-style-bank\source-boundary.md`, `C:\Users\didgm\.codex\skills\.shared\sci-figure-style-bank\multi-curve-data-figure-preference.md`, and `C:\Users\didgm\.codex\skills\.shared\sci-figure-style-bank\palette-line-marker-patterns.md` before selecting the plot type and palette. If the data has an ordered x-axis and multiple supported series, prefer a multi-curve line plot or multi-panel multi-curve layout using Kevin's data-figure palette unless a project-approved palette or journal rule overrides it.
- Read `C:\Users\didgm\.codex\skills\.shared\sci-figure-style-bank\data-to-viz-inspiration-policy.md` when plot-type selection is uncertain, a non-curve chart may be better, too many series risk a spaghetti plot, or the user asks for Data-to-Viz inspiration.

## Workflow

1. Confirm the figure purpose: main result, ablation, dataset description, workflow, error analysis, or appendix.
2. Confirm data source and admissible claim, including whether the output is diagnostic-only or manuscript-admissible evidence.
3. Check figure surface lifecycle: script/CSV source, final output path, QA output path, root manuscript include target, and legacy/provenance status.
4. Choose plot type based on data and claim, not aesthetics. Apply the multi-curve preference first for ordered x-axis comparisons; use Data-to-Viz as a chart-selection and caveat reference when curves are uncertain or unsuitable; use bars, heatmaps, scatter, distribution/range, or spatial plots only when the data shape makes curves unsuitable. Choose backend based on evidence, reproducibility, local toolchain, journal output, and maintainability; do not switch to R/ggplot2/ComplexHeatmap merely because an external skill suggests it.
5. Generate or edit a Python script using stable inputs and explicit output paths.
6. Export publication formats such as PNG and SVG/PDF when appropriate.
7. Verify the output exists and that axes, units, legends, uncertainty, and captions match the data.
8. State what the figure does not prove.

## Safe Template

Use `assets/figure_template.py` as a starting point when the repo lacks a local figure style. The template intentionally contains no synthetic demo data.

## Output Shape

```md
## Figure Contract
- Purpose:
- Data source:
- Unit/denominator/horizon:
- Claim supported:
- Claim not supported:
- Multi-curve preference decision:
- Palette source:
- Notation source/status, when mathematical labels are present:
- Data-to-Viz consult:
- Surface lifecycle:

## Files
- Script:
- Output:

## Verification
- Regeneration command:
- Checks performed:
- Remaining risks:
```

## Safety Boundaries

- Do not generate random demo data for manuscript figures.
- Do not add error bars or statistical markers unless calculated from real data.
- Do not smooth, normalize, filter, or drop data without documenting the transformation.
- Do not replace result figures with illustrative visuals when the paper needs evidence.

## References

- Mathematical notation governance: C:\Users\didgm\.codex\skills\.shared\mathematical-notation-governance.md

- Figure rules: `references/figure-rules.md`
- Upstream attribution: `references/upstream-attribution.md`
- Diagnostic dashboard boundary: `references/diagnostic-dashboard-boundary.md`
- Figure surface lifecycle: `references/figure-surface-lifecycle.md`
- Random seed policy: `references/random-seed-policy.md`
- Nature/top-journal backend and evidence gate: `references/nature-figure-backend-and-evidence-gate.md`
- Multi-curve data-figure preference: `C:\Users\didgm\.codex\skills\.shared\sci-figure-style-bank\multi-curve-data-figure-preference.md`
- Data-to-Viz inspiration policy: `C:\Users\didgm\.codex\skills\.shared\sci-figure-style-bank\data-to-viz-inspiration-policy.md`
- Palette, line, and marker patterns: `C:\Users\didgm\.codex\skills\.shared\sci-figure-style-bank\palette-line-marker-patterns.md`
- Manuscript repo surface contract: `C:\Users\didgm\.codex\skills\.shared\repo-governance\research-repo-surface-contract.md`