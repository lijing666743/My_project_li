# RL Reward/Loss KPI Gate

Use this gate when designing, refactoring, or evaluating RL, DRL, or DL+RL training signals.

## Required Contract

Lock these items before treating a run as more than diagnostic:

- Reward formula, neutral baseline, scale, denominator or normalization, clipping or scaling, and delayed-reward boundary.
- Positive reward terms and negative reward or penalty terms. When terms are separable, log `reward_positive` and `penalty_total`.
- Loss type: `MSE`, `Huber`, or a project-defined alternative with rationale.
- Loss target: TD error, Q target, policy objective, actor-critic loss, model-based objective, supervised auxiliary loss, or other update target.
- Smoothing/window settings: warmup fraction, final-window fraction, smoothing window, and minimum epoch count.
- `signal_gate_status`: `signal-pass`, `signal-watch`, `signal-fail`, `insufficient-horizon`, or `scale-undefined`.

## Reward Gate

- Net reward should trend slightly above the declared neutral baseline, normally `0`, in the final window.
- Positive reward and penalty terms should be balanced but biased positive in the final window. Penalties may dominate during warmup, but late penalty dominance requires an explicit project reason.
- Oscillation is acceptable when the smoothed mean stabilizes or improves and the envelope is not widening without explanation.
- If reward scale, neutral baseline, or margin is not declared, classify as `scale-undefined` or `signal-watch`, not `signal-pass`.

## Loss Gate

- `MSE` and `Huber` are both valid baseline choices. Prefer Huber when outliers or large TD-error spikes are expected.
- Early epochs may show large oscillations.
- Later epochs should show a downward or stabilizing smoothed trend, finite values, and no sustained divergence.
- A loss trend is diagnostic evidence only unless seed policy, baselines, and claim gates support stronger statements.

## Default Diagnostic Windows

- Warmup: first 20% of epochs unless a project profile overrides it.
- Final window: last 20% of epochs, with at least 5 epochs where available.
- Fewer than 10 total epochs, missing final-window data, or non-finite required values should produce `insufficient-horizon` or `signal-fail`.

## Claim Boundary

- `signal-pass` means the training signal is usable or diagnostically converging under the checked run.
- It does not prove robustness, superiority, generalization, stability, theoretical convergence, or manuscript readiness.
- Single-seed diagnostics cannot support those stronger claims without the matching multi-seed, statistical, baseline, and evidence gates.

## Optional Checker

Use `C:\Users\didgm\.codex\skills\.shared\scripts\check_rl_signal_gate.py` when a metrics CSV exists. Treat the checker as a diagnostic assistant; project-specific reward scale and margin still need human or profile-level definition.
