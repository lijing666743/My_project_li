---
name: sci-student-research-code-builder
description: Create or refactor student-friendly SCI research code repositories with a root `main.py` entrypoint, implementation under `src/`, interactive parameter input with ranges and defaults, seed policy, DL/RL/DRL training loops, reward/loss KPI signal gates, per-epoch console summaries, `logs/*.csv` metrics, and `plots/*dashboard*.png` outputs. Use when building Python research programs, training pipelines, reinforcement learning code, reward/loss curves, MSE or Huber loss choices, deep learning experiments, student-run code, or when a repo should be runnable without command-line subcommands and many parameters. Do not use for manuscript-only work, figure-only work, or expert-only automation unless the user explicitly asks to keep a non-interactive CLI.
---

# SCI Student Research Code Builder

## Purpose

Build research code that students can run and debug without memorizing command-line subcommands or long parameter lists.

This skill defines a code-architecture contract, not a scientific claim. Keep model choices, defaults, logs, and dashboard outputs evidence-gated.

## Core Contract

Use this default structure unless the user or repo guidance explicitly says otherwise:

```text
repo/
  main.py
  requirements.txt
  src/
  logs/
  plots/
```

- `main.py` is the student-facing entrypoint.
- `src/` contains implementation modules.
- Runtime parameters are collected through an interactive interface with allowed ranges, defaults, and default-source labels.
- Randomized runs expose `seed` in the student-facing parameter set and use `seed=42` unless robustness testing or project docs require otherwise.
- `logs/` stores run metrics as CSV.
- `plots/` stores an automatic dashboard PNG after execution.

For Kevin-style student research code, read `references/kevin-coding-style-distillation.md` before designing or changing entrypoints, defaults, profiles, logs, plots, phase artifacts, tests, or handoff surfaces.

For Coding/Testing repos, read C:\Users\didgm\.codex\skills\.shared\repo-governance\repository-environment-contract.md and record conda base, requirements.txt, direct dependency constraints, and any explicit override.

For project repos under `SUTResearchHub/`, also read `C:\Users\didgm\.codex\skills\.shared\repo-governance\research-repo-surface-contract.md`. Treat `SUTResearchHub/Common/` as a shared-library exception. Coding-stage student project roots should follow the current surface contract: `AGENTS.md`, `main.py`, `README.md`, `knowledge/`, `logs/`, `plots/`, `sections/`, and `src/`; project-local `.gitignore`, `.vscode/`, and `.idea/` belong at the parent repo or global level unless a topic-level `AGENTS.md` records an accepted exception.

Read `references/student-entrypoint-contract.md` before creating or refactoring an entrypoint.

Read `references/interactive-launcher-and-gate-chain-contract.md` when the task changes `main.py`, menu defaults, `method_id=all`, serial multi-method execution, no-training audit chains, optional CSV/JSON artifact readers, or downstream stages that depend on upstream artifacts.

## DL/RL/DRL Behavior

- Recommend suitable DL and/or RL model families before implementation.
- For "latest" model recommendations, verify with current literature, official docs, or user-supplied sources during the task. Do not hard-code a stale "latest" list into code.
- For DRL, default to an outer `epoch` loop and inner `episode` loop. At the start of every epoch, reset the scenario/environment and set episode counting back to `0`.
- Default fixed hyperparameters are `epsilon=0.1` and `alpha=0.001` unless accepted project evidence overrides them.
- For RL, DRL, or DL+RL framework work, read `references/rl-reward-loss-kpi-gate.md` before designing rewards, losses, dashboards, profiles, or evidence claims. Lock reward scale, positive reward versus penalty balance, loss type, smoothing/window settings, and signal gate status before treating a run as more than diagnostic.

For DL+RL hybrid systems, read `references/dl-rl-hybrid-contract.md` before implementation or evaluation. Lock what the RL agent controls, the state/action/reward/transition/update contract, gradient boundaries between DL and RL modules, and whether the result is only a concept demo or evidence-grade experiment.

For DL/RL cross-paradigm rescue, read `references/dl-rl-cross-paradigm-rescue-contract.md` before implementation. Confirm the original DL-only or RL-only baseline, the bottleneck being addressed, the minimal complementary component, required ablation, seed/log/dashboard contract, and the maximum claim level allowed before validation.

Read `references/drl-training-loop-contract.md` and `references/model-recommendation-policy.md` for details.

Read `references/random-seed-policy.md` before adding randomness, robustness testing, seed ranges, or multi-seed execution.

## Logging And Dashboard

Per epoch, print one concise console line containing learning rate, exploration rate, loss, reward, and key KPI values.

At completion:

- write `logs/<run_id>_metrics.csv`
- write `plots/<run_id>_dashboard.png`

The CSV is the data source. The dashboard is diagnostic unless a project claim gate admits it as manuscript evidence.

Read `references/log-dashboard-contract.md` before changing outputs.

For training, simulation, or validator runs expected to exceed 10 minutes, use monitored execution or an equivalent polling pattern. Treat epoch, episode, loss, reward, KPI summaries, `logs/` metrics, checkpoints, and `plots/` dashboards as default progress signals. Brief at least every 10 minutes, and mark `suspected-stall` after two quiet checks with no stdout/stderr, log, checkpoint, metric, or dashboard update unless a quiet-run exception was declared before launch. For runs expected to exceed 60 minutes or that actually pass 60 minutes, define a run phase plan such as setup, baseline, training, evaluation, dashboard, and validator; brief phase-level coverage about every hour with completed phases, current phase, pending phases, missing artifacts, and any continue/pause/restart decision.

When documenting training parameters in Markdown, distinguish code parameters such as `epsilon`, `alpha`, `loss`, and `reward` from mathematical symbols such as $\epsilon$, $\alpha$, $\mathcal{L}$, and $r_t$.

## Template Asset

Use `assets/research-program-template/` when bootstrapping a new small Python research program or when the repo lacks a stronger local pattern. Copy the template into the target repo, then replace toy dynamics with the real method.

Do not present the template's placeholder training logic as a scientific result.

## Coordination

- Use `project-intake-scope-lock` first if active source-of-truth, allowed surface, or validation path is unclear.
- Use `junshi` for medium or large implementation waves.
- Use `bounded-execution-validator-first` when the edit boundary is clear and the validator is known.
- Use `sci-figures-python` only when converting logged data into paper-grade figures or auditing figure evidence.
- Use `sci-experiment-campaign-planner` before implementing long manuscript-evidence campaigns, final all-method comparisons, resume-safe multi-seed runs, or "one wave produces all paper-needed data" tasks.

## Validation Checklist

For any implementation using this contract, verify:

- `python main.py` can run through the student-facing path.
- Prompts show parameter names, ranges, defaults, and default-source labels.
- Randomized code records `seed` or `seed_set` in config/log artifacts.
- DRL loop order matches `epoch -> reset -> episode loop` when applicable.
- RL/DRL/DL+RL code records the reward/loss signal gate contract, including `loss_type`, reward scale, positive reward/penalty balance when applicable, and `signal_gate_status`.
- DL+RL hybrid code documents `state`, `action`, `reward`, `transition`, update target, gradient boundary, and training signal owner for each module.
- DL/RL cross-paradigm rescue code records the single-paradigm baseline, minimal complementary component, ablation plan, and evidence stage before making hybrid claims.
- Console output is one line per epoch.
- `logs/*metrics.csv` exists and contains required metric columns.
- `plots/*dashboard*.png` exists.
- For launcher or all-mode changes, the real no-argument `python main.py` path is tested with mock input, and aggregate comparison artifacts are checked in addition to per-method artifacts.
- No dashboard is claimed as manuscript evidence without claim-gate admission.
- Markdown docs distinguish code identifiers from mathematical notation.
- For `SUTResearchHub/` project repos, root-surface and requirements compliance is checked with `C:\Users\didgm\.codex\skills\.shared\repo-governance\check_research_repo_surface.py --mode sut <repo>`, and any cleanup remains a separate bounded task.
