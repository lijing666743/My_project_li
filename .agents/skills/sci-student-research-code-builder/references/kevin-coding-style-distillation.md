# Kevin Coding Style Distillation

Use this reference when building or refactoring Kevin's student-facing research code, especially under a SUT Coding-stage topic repo.

## Evidence Scope

This distillation comes from read-only inspection of these student projects:

- `M22_LianXiaoHui`
- `M22_LiuXue`
- `M22_WangWeiRan`
- `U19_LiPengZhao`
- `U20_ChenSiXue`
- `U20_LiJiaNi`

Treat `U20_LiJiaNi/P3` as the closest current-quality example because it includes a short README entry, phase results, summaries, manifests, dashboards, tests, profiles, and claim-boundary language. Treat older P1/P2 projects as evidence of durable preferences, not as target architecture to copy.

## Adopted Defaults

- Bias toward small, compartmentalized implementation specs with one validator per wave.
- Keep a student-facing launcher. The default path should be simple to run with no memorized long command, even when expert CLI options also exist.
- Centralize defaults and accepted run profiles in one place, such as `Parameters.py`, `profiles.py`, or a project-specific registry. Do not scatter scientific defaults or magic constants across modules.
- Split implementation by research concepts: topology, state, action choice, reward, model, normalization, logging, plotting, validation, and experiment phases should be readable as separate responsibilities when the project size justifies it.
- Make execution observable. Training or simulation loops should report concise epoch, episode, loss, reward, and KPI progress without requiring the user to inspect internals.
- Save durable artifacts. Numeric outputs belong in `logs/` as CSV, JSON, or Markdown reports; diagnostic figures and dashboards belong in `plots/`; phase results should include summary, manifest, raw outputs, and dashboard when the project has staged experiments.
- Record run identity, config, data source, seed or seed set, validation status, and claim boundary in the run artifacts.
- Add focused tests for core semantics and boundary conditions when a project moves beyond a toy script.
- Use short entry documents for mature repos: README for human orientation, current-state or next-task surfaces for handoff, and an indexed result surface when phase outputs accumulate.
- Keep Coding evidence separate from manuscript claims. A run artifact can support a bounded claim only after the claim gate admits it.

## Legacy Patterns Not To Preserve

- Do not preserve single-file mega-scripts as the target shape when a project has multiple concepts or phases.
- Do not require bare `input()` prompts without defaults, ranges, or a mockable no-argument path.
- Do not rely on `plt.show()` as the only visual output; save figures needed for review.
- Do not leave random behavior unseeded when outputs are compared or reused.
- Do not treat debug output, one-off console text, single-seed diagnostics, or partial-horizon runs as evidence for robustness, superiority, stability, generalization, or manuscript readiness.
- Do not keep duplicate version directories as the preferred experiment-management pattern; use profiles, phase registries, or result manifests for new work.

## Quality Criteria

A finished Kevin-style Coding implementation should satisfy:

- The accepted public entrypoint runs through the student-facing path and documents any expert-only CLI path.
- Defaults are discoverable, centralized, and recorded in artifacts.
- Core modules map to the research problem, not only to incidental programming mechanics.
- Logs and dashboards can be inspected after the run without rerunning code.
- Tests or smoke validators cover the highest-risk semantics.
- README, handoff, or phase-index surfaces tell the next Codex instance what to read first.
- Claim boundaries are explicit, and no diagnostic artifact is promoted to manuscript evidence without the appropriate verification gate.
