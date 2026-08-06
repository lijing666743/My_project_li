# Student Entrypoint Contract

Use this contract for research code that students will run locally.

## Required Shape

- Keep one root `main.py` as the default user-facing entrypoint.
- Put implementation modules under `src/`.
- Keep generated run metrics under `logs/`.
- Keep generated dashboards and diagnostic plots under `plots/`.
- Avoid making students run command-line subcommands or long option lists.

Expert-only CLI flags may exist for automation or tests, but they must not be the only normal path.

## Interactive Parameters

Every interactive parameter should show:

- parameter name
- short meaning
- allowed range or allowed choices
- default value
- default-source label

Default-source labels:

- `accepted-local-best`: user and J.A.R.V.I.S. have accepted this as the current local feasible or best value.
- `accepted-feasible`: accepted as currently runnable and safe, but not necessarily best.
- `conservative-default`: no accepted tuning record exists.
- `project-doc`: value is pinned by repo documentation.
- `global-default-single-seed`: default `seed=42` from the global random seed policy.

Do not call a default "optimal" unless accepted evidence or project docs say so.

## Prompt Behavior

- Empty input uses the default.
- Invalid input prints the allowed range and asks again.
- Values outside range are rejected.
- Use small smoke defaults in templates, but replace them with accepted project defaults before formal runs.
- Expose `seed` as an ordinary interactive parameter when randomness affects results; default to `42` unless a project-specific seed policy overrides it.

## Main Entrypoint Skeleton

The entrypoint should read like:

```python
from src.config import interactive_config
from src.runner import run_training
from src.logging_utils import write_metrics_csv
from src.dashboard import plot_dashboard


def main() -> None:
    config = interactive_config()
    metrics = run_training(config)
    csv_path = write_metrics_csv(metrics, config)
    png_path = plot_dashboard(metrics, config)
    print(f"Saved metrics: {csv_path}")
    print(f"Saved dashboard: {png_path}")


if __name__ == "__main__":
    main()
```
