# DL+RL Formalization

Use this reference when a manuscript method combines deep learning representation, prediction, or fusion modules with reinforcement-learning control or decision modules.

## Candidate Formal Blocks

Check whether the manuscript evidence supports:

- state definition $s_t$
- action set $\mathcal{A}$
- reward function $r_t$
- transition relation or environment reset rule
- Q-value, policy, actor-critic, or model-based update equation
- supervised DL loss $\mathcal{L}_{\mathrm{DL}}$
- RL loss or objective $\mathcal{L}_{\mathrm{RL}}$
- joint objective or scheduling rule when both learning signals interact
- representation-to-state mapping when DL features enter RL state
- policy-controlled loss weighting, sample selection, curriculum, routing, scheduling, allocation, or inference-time decision rule when RL is added to a DL-centered method
- supervised warm-start, learned encoder, reward predictor, transition predictor, value/Q/policy function approximator, or environment model when DL is added to an RL-centered method
- baseline, hybrid, and ablation algorithm block when the manuscript claims the added DL or RL component matters
- gradient-boundary statement, especially detached features or non-differentiable decisions
- algorithm block with epoch, episode, reset, update, logging, and validation steps
- complexity statement if module sizes and iteration counts are known

## Evidence Gate

Only add a formal block when the source method or code actually defines it. If the prototype does not define `next_state`, replay, target network, baseline, or statistical validation, do not create equations that imply those features exist.

## Preferred Placement

- System model: state, action, environment, and constraints.
- Method: losses, reward, update target, and algorithm.
- Experiment setup: seeds, metrics, baseline/ablation, logging, and evidence boundary.
- Limitations: concept-demo or diagnostic-only scope when applicable.

## Rejection Examples

Reject or downgrade:

- theorem-style convergence claims without proof assumptions
- robustness equations from a single seed
- superiority statements without a baseline
- end-to-end differentiability claims when tensors are detached or converted to lists
- temporal RL claims when `state` and `next_state` are not distinguishable
- hybrid superiority claims when the original DL-only or RL-only baseline and component ablation are missing
- convergence or optimality theorem-style blocks for rescue hybrids without proof assumptions and evidence
