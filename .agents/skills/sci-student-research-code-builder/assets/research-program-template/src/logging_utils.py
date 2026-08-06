from __future__ import annotations

import csv
from pathlib import Path

from .config import RunConfig


def write_metrics_csv(rows: list[dict[str, object]], config: RunConfig) -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{config.run_id}_metrics.csv"
    if not rows:
        raise ValueError("No metrics rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
