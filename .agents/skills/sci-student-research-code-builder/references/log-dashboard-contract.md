# Log And Dashboard Contract

Use this contract for student-run training programs and research experiments.

## Required Outputs

At the end of a run:

- write metrics CSV to `logs/<run_id>_metrics.csv`
- write dashboard PNG to `plots/<run_id>_dashboard.png`

The CSV is the source of truth. The dashboard is a diagnostic view unless admitted through the project's evidence and claim gate.

## Required CSV Columns

Use these columns when applicable:

- `run_id`
- `epoch`
- `episodes_per_epoch`
- `seed`
- `seed_set` when a row summarizes multiple seeds
- `alpha`
- `epsilon`
- `loss`
- `loss_type` when a model loss is present, for example `MSE` or `Huber`
- `reward`
- `reward_positive` when positive reward terms are separable
- `penalty_total` when penalty terms are separable
- `signal_gate_status` when the run evaluates RL reward/loss readiness
- one or more KPI columns, for example `kpi`

Add task-specific KPI columns as needed, but keep names stable once reports depend on them.

## Console Output

Print one concise line per epoch:

```text
epoch=<n> lr=<alpha> eps=<epsilon> loss=<loss> reward=<reward> kpi=<kpi>
```

Avoid printing per-step or per-episode spam by default.

## Dashboard Panels

Default panels:

- loss vs epoch, with raw and smoothed curves when the run evaluates convergence trend
- reward vs epoch, with raw and smoothed curves when the run evaluates convergence trend
- KPI vs epoch
- hyperparameter trace or compact run summary
- seed or seed set summary

For RL/DRL/DL+RL runs, dashboards should make it possible to judge whether reward trends slightly above the declared neutral baseline and whether loss trends downward or stabilizes after warmup. Oscillation is acceptable only when the smoothed mean stabilizes or improves and the widening envelope is explained.

If matplotlib is unavailable, report the missing dependency rather than silently omitting the dashboard.
