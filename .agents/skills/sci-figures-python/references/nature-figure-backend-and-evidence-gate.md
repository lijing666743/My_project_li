# Nature Figure Backend And Evidence Gate

This reference adapts abstract lessons from an external Nature-oriented figure skill without importing its code or assuming R as the default backend.

## Backend Gate

- Use Python when the project already has Python data pipelines, scripts, CSV/NumPy/Pandas artifacts, or reproducibility constraints.
- Use `sci-tikz-publication-figures` when the output is a LaTeX/TikZ/PGFPlots figure, needs Times-family integration, or depends on TeX-side panel geometry.
- Consider a future R-specific skill only when the project already uses R, ggplot2, ComplexHeatmap, Seurat, Bioconductor, or journal templates that are materially easier in R.
- Do not create decorative or schematic figures here; route Image2/TikZ/scenario/diagram tasks appropriately.

## Evidence Gate

- Lock data, units, transformations, uncertainty, and admissible claim before figure design.
- Treat every plotted panel as evidence with a claim boundary.
- Do not let top-journal aesthetics hide missing baselines, sample sizes, statistics, seeds, or source data.
- Do not copy external Nature figures or captions; use only abstract layout lessons.
