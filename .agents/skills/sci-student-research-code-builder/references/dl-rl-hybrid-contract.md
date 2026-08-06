# DL+RL Hybrid Contract

Use this reference when building, refactoring, or evaluating a research program that combines deep learning with reinforcement learning.

## Required Lock Before Implementation

Define these items before writing or approving code:

- RL control target: what the agent controls, such as loss weights, sample selection, hyperparameters, architecture choices, scheduling, routing, or inference policy.
- State `s_t`: tensors, metrics, environment observations, or model outputs visible to the RL agent.
- Action `a_t`: exact action space, constraints, discretization, and whether actions are student-visible parameters.
- Reward `r_t`: formula, sign, neutral baseline, scale, denominator, positive reward terms, penalty terms, delayed effects, and whether it uses validation metrics or training-only proxies.
- Transition: what changes after an action, what counts as next state `s_{t+1}`, and when scenario/environment reset occurs.
- Update target: Q-learning target, policy objective, actor-critic loss, model-based objective, or other learning rule.
- Loss owner and loss type: which module owns the loss, whether it is `MSE`, `Huber`, or another justified objective, and whether the loss is supervised, RL-derived, or auxiliary.
- Reward/loss signal gate: smoothing/window settings and `signal_gate_status` for early diagnostic training readiness.
- Gradient boundary: which tensors remain differentiable, which values are intentionally detached, and which modules receive gradients from supervised DL loss versus RL loss.
- Evidence stage: concept demo, smoke run, diagnostic experiment, validation experiment, or manuscript evidence.

## Red Flags

- RL action changes a DL loss but no DL optimizer step updates the DL modules.
- Tensor values are converted to lists, NumPy arrays, or Python scalars in a path that is expected to remain differentiable.
- Reward compares against a previous value but `next_state` is not defined separately from current state.
- Reward scale, positive reward, penalties, or loss type are not recorded, but the run is described as convergent or evidence-grade.
- A DQN-style method lacks a clear replay buffer, target network, or explanation for why a simplified update is acceptable.
- Import-time code prints parameters, loads data, or mutates global state before `main.py` starts.
- A toy image, synthetic vector, or single seed is used as if it supports robustness, superiority, or generalization.

## Student-Friendly Output Requirements

For runnable research code, keep the global student contract:

- root `main.py`
- implementation under `src/`
- interactive parameter prompts with ranges, defaults, and default-source labels
- recorded `seed` or `seed_set`
- one concise console line per epoch
- `logs/*metrics.csv`
- `plots/*dashboard*.png`

## Evidence Boundary

A DL+RL hybrid prototype is not manuscript evidence unless it has a real data contract, reproducible seeds, baseline or ablation comparison, logged metrics, a reward/loss signal gate, and a claim gate admitting the metric surface.

Single-run toy prototypes may support only:

- architecture exploration
- smoke testing
- teaching or explanation
- implementation sanity checks

They do not support:

- robustness
- generalization
- superiority
- deployment readiness
- statistical or journal-facing performance claims
