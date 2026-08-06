# Model Recommendation Policy

Use this policy before implementing DL, RL, or DRL programs.

## Required Behavior

- Recommend model families appropriate to the task, data shape, action/state space, and validation budget.
- Separate stable baselines from current frontier candidates.
- For claims about the latest model families, browse or inspect current primary sources, official docs, papers, or user-supplied PDFs during the task.
- If network access or source access is unavailable, state that the recommendation is not live-verified and default to stable baselines.

## Do Not

- Hard-code a permanent "latest model" list into the skill.
- Add a complex model just because it is new.
- Recommend models that the repo cannot validate.
- Upgrade model choice into a manuscript novelty claim without evidence and literature review.

## Practical Defaults

When no live verification is available:

- Use simple supervised baselines for tabular or regression tasks.
- Use standard MLP/CNN/RNN/Transformer-family choices only when the data modality supports them.
- Use stable RL baselines such as DQN-family, actor-critic, PPO-family, or SAC-family only when the state/action space and validation path fit.

Mark these as baseline candidates, not current frontier recommendations.
