# DL/RL Cross-Paradigm Rescue Contract

Use this reference before implementing a DL->RL or RL->DL rescue path in student-friendly research code.

## Required Lock Before Implementation

Record these items in config, docs, or run manifest:

- Original baseline: DL-only or RL-only code path that already runs.
- Bottleneck: representation learning, dynamic decision/control, scheduling, allocation, sample selection, loss weighting, reward approximation, or another explicit blocker.
- Minimal complementary component: the smallest DL or RL addition that tests the bottleneck.
- Control/learning target:
  - If adding RL to DL: state, action, reward, transition, update target, reset rule, and what the agent controls.
  - If adding DL to RL: input representation, encoder/predictor target, supervised or self-supervised loss, gradient path, and how the representation enters state/value/policy/reward/transition.
- Gradient boundary: which modules receive DL gradients, RL losses, both, or neither.
- Baseline/ablation plan: baseline, hybrid, and component-removed variants.
- Evidence stage: concept demo, diagnostic experiment, validation experiment, or manuscript evidence.
- Reproducibility: seed or seed set, logs, metrics CSV, dashboard, config, and source revision.

## Safe Implementation Defaults

- Add one complementary component at a time.
- Keep the original single-paradigm baseline runnable.
- Keep student-facing parameters explicit and range-checked.
- Log both baseline and hybrid runs in comparable CSV fields.
- Generate dashboards that compare only admitted metric surfaces.

## Do Not Implement Hybrid Rescue When

- The baseline does not run.
- The blocker is an implementation bug, missing data preprocessing, unclear metric, missing logs, or reward/KPI mismatch.
- There is no feasible baseline or ablation.
- The user wants a manuscript claim but the code only supports a toy or smoke-test result.

## Evidence Boundary

A hybrid rescue supports stronger claims only after baseline, ablation, seed policy, metric denominator, and artifact contracts are satisfied. Until then, describe it as a candidate design, diagnostic experiment, or concept demo.
