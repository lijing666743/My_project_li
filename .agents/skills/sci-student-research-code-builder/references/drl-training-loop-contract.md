# DRL Training Loop Contract

Use this contract when building or refactoring DRL code.

## Loop Semantics

- `epoch` is the outer loop.
- `episode` is the inner loop.
- At the start of every epoch:
  - reset the scenario/environment
  - set episode counting back to `0`
  - reset any epoch-local accumulators
- Inside each epoch, run `episodes_per_epoch` episodes.

Pseudo-flow:

```text
for epoch in range(epochs):
    scenario.reset()
    for episode in range(episodes_per_epoch):
        run one episode
    update epoch metrics
    print one concise epoch line
```

## Default Hyperparameters

Use these defaults unless project docs or accepted local tuning override them:

- exploration rate: `epsilon=0.1`
- learning rate: `alpha=0.001`
- random seed: `seed=42` for non-robustness runs

These are implementation defaults, not evidence that the parameters are optimal.

Use `references/random-seed-policy.md` when robustness testing requires multiple seeds.

## Reward/Loss Signal Gate

Before accepting a DRL loop as basically usable, define and record:

- reward formula, neutral baseline, scale, denominator or normalization, and positive reward versus penalty balance
- loss type, normally `MSE` or `Huber`, and the update target the loss trains
- smoothing/window settings for judging raw and smoothed reward/loss curves
- `signal_gate_status`: `signal-pass`, `signal-watch`, `signal-fail`, `insufficient-horizon`, or `scale-undefined`

Reward should trend slightly above the declared neutral baseline in the final window, with positive rewards and penalties balanced but biased positive. Loss may oscillate early, but later epochs should show a downward or stabilizing smoothed trend. Passing this gate supports only diagnostic training-signal readiness unless stronger evidence gates pass.

## Console Line

Print one line per epoch. Include:

- epoch index
- learning rate
- exploration rate
- loss
- reward
- key KPI values

Example:

```text
epoch=3 lr=0.001 eps=0.100 loss=0.234 reward=18.700 kpi=0.812
```

Keep the line compact. Avoid multi-line progress spam.
