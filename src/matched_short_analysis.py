"""Pure stop gates, matched-triplet analysis, and diagnostic plot pipeline."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STOP_STATUSES = (
    "PASS",
    "CONTINUE",
    "STOP_NUMERICAL",
    "STOP_PPO_INSTABILITY",
    "STOP_GRADIENT_CLIPPING",
    "STOP_REMOTE_STARVATION",
    "STOP_REMOTE_OVERUSE",
    "STOP_KPI_REGRESSION",
    "INCONCLUSIVE",
)
_EPSILON = 1.0e-12


class MatchedAnalysisError(RuntimeError):
    """Raised when matched diagnostic facts are missing or inconsistent."""


@dataclass(frozen=True)
class StopEvaluation:
    status: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STOP_STATUSES:
            raise MatchedAnalysisError(f"unknown stop status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons)}


def _reject_json_constant(value: str) -> None:
    raise MatchedAnalysisError(f"non-finite JSON constant {value!r}")


def load_live_jsonl(path: str | Path) -> tuple[tuple[dict[str, Any], ...], bool]:
    """Read canonical JSONL, tolerating only one interrupted final fragment."""

    source = Path(path)
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    truncated_tail_ignored = False
    for index, line in enumerate(lines):
        is_last = index == len(lines) - 1
        complete = line.endswith((b"\n", b"\r"))
        try:
            value = json.loads(
                line.decode("utf-8"), parse_constant=_reject_json_constant
            )
        except (UnicodeDecodeError, json.JSONDecodeError, MatchedAnalysisError) as exc:
            if is_last and not complete:
                truncated_tail_ignored = True
                break
            raise MatchedAnalysisError(
                f"invalid live JSONL record at line {index + 1}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise MatchedAnalysisError("live JSONL records must be objects")
        records.append(value)
    return tuple(records), truncated_tail_ignored


def _records(records: Sequence[Mapping[str, Any]], record_type: str) -> list[dict[str, Any]]:
    return [dict(item) for item in records if item.get("record_type") == record_type]


def _sorted_unique(
    records: Sequence[Mapping[str, Any]], record_type: str, key: str
) -> list[dict[str, Any]]:
    selected = sorted(_records(records, record_type), key=lambda item: int(item[key]))
    keys = [int(item[key]) for item in selected]
    if len(keys) != len(set(keys)):
        raise MatchedAnalysisError(f"duplicate {record_type} {key}")
    return selected


def trailing_mean(values: Sequence[float | None], window: int = 5) -> list[float | None]:
    if window <= 0:
        raise ValueError("window must be positive")
    result: list[float | None] = []
    for index in range(len(values)):
        present = [
            float(value)
            for value in values[max(0, index + 1 - window) : index + 1]
            if value is not None
        ]
        result.append(None if not present else math.fsum(present) / len(present))
    return result


def late_window(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    count = math.ceil(len(records) * 0.20)
    return [dict(item) for item in records[-count:]]


def warmup_window(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    count = math.ceil(len(records) * 0.20)
    return [dict(item) for item in records[:count]]


def warmup_count(records: Sequence[Mapping[str, Any]]) -> int:
    return math.ceil(len(records) * 0.20)


def _consecutive(
    records: Sequence[Mapping[str, Any]],
    length: int,
    predicate: Any,
) -> bool:
    streak = 0
    for record in records:
        if predicate(record):
            streak += 1
            if streak >= length:
                return True
        else:
            streak = 0
    return False


def _is_finite_record(record: Mapping[str, Any]) -> bool:
    for value in record.values():
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def _merged_rollouts(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rollouts = _sorted_unique(records, "rollout", "update_index")
    updates = {
        int(item["update_index"]): item
        for item in _sorted_unique(records, "update", "update_index")
    }
    merged = []
    for rollout in rollouts:
        row = dict(rollout)
        row.update(
            {
                key: value
                for key, value in updates.get(int(rollout["update_index"]), {}).items()
                if value is not None
            }
        )
        merged.append(row)
    return merged


def _three_update_threshold(
    updates: Sequence[Mapping[str, Any]], field: str, threshold: float
) -> bool:
    return _consecutive(
        updates,
        3,
        lambda row: row.get(field) is not None and float(row[field]) > threshold,
    )


def symmetric_degradation(
    baseline: float | None,
    candidate: float | None,
    *,
    higher_is_better: bool,
) -> float | None:
    """Return a bounded symmetric degradation without a near-zero explosion."""

    if baseline is None or candidate is None:
        return None
    baseline = float(baseline)
    candidate = float(candidate)
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        return None
    worse = baseline - candidate if higher_is_better else candidate - baseline
    if worse <= 0.0:
        return 0.0
    denominator = abs(baseline) + abs(candidate)
    if denominator <= _EPSILON:
        return 0.0
    return 2.0 * worse / denominator


def _episode_windows(records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, float | None]]:
    episodes = _sorted_unique(records, "episode", "episode_index")
    fields = (
        "episode_return",
        "completion_rate",
        "expiration_rate",
        "energy_j_per_env_step",
    )
    smoothed = {
        field: trailing_mean(
            [None if row.get(field) is None else float(row[field]) for row in episodes], 5
        )
        for field in fields
    }
    return {
        int(row["environment_step"]): {
            field: smoothed[field][index] for field in fields
        }
        for index, row in enumerate(episodes)
    }


def kpi_regression_gate(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.10,
) -> StopEvaluation:
    """Compare matched trailing-five episode windows at identical steps."""

    baseline = _episode_windows(baseline_records)
    candidate = _episode_windows(candidate_records)
    common_steps = sorted(set(baseline) & set(candidate))
    if len(common_steps) < 3:
        return StopEvaluation(
            "INCONCLUSIVE", ("fewer than 3 matched completed-episode windows",)
        )
    directions = {
        "episode_return": True,
        "completion_rate": True,
        "expiration_rate": False,
        "energy_j_per_env_step": False,
    }
    streaks = {field: 0 for field in directions}
    for step in common_steps:
        for field, higher_is_better in directions.items():
            degradation = symmetric_degradation(
                baseline[step][field],
                candidate[step][field],
                higher_is_better=higher_is_better,
            )
            if degradation is not None and degradation > threshold:
                streaks[field] += 1
                if streaks[field] >= 3:
                    return StopEvaluation(
                        "STOP_KPI_REGRESSION",
                        (f"{field} degraded by >{threshold:.0%} for 3 matched windows",),
                    )
            else:
                streaks[field] = 0
    return StopEvaluation("CONTINUE")


def evaluate_stop(
    records: Sequence[Mapping[str, Any]],
    *,
    group: str,
    baseline_records: Sequence[Mapping[str, Any]] | None = None,
) -> StopEvaluation:
    """Pure stop decision; never mutates a model, file, or input record."""

    if any(not _is_finite_record(item) for item in records):
        return StopEvaluation("STOP_NUMERICAL", ("NaN/Inf diagnostic",))
    updates = _sorted_unique(records, "update", "update_index")
    if _three_update_threshold(updates, "approx_kl_max", 0.02):
        return StopEvaluation("STOP_PPO_INSTABILITY", ("KL > 0.02 for 3 updates",))
    if _three_update_threshold(updates, "clip_fraction_max", 0.30):
        return StopEvaluation(
            "STOP_PPO_INSTABILITY", ("clip fraction > 0.30 for 3 updates",)
        )
    # Gradient clipping frequency is diagnostic-only. Established runs can
    # remain finite and optimize safely under persistent clipping; nonfinite
    # gradients are still rejected by the updater and numerical gate.
    if group == "baseline":
        return StopEvaluation("CONTINUE")
    merged = _merged_rollouts(records)
    if _consecutive(
        merged,
        5,
        lambda row: int(row.get("route_active_count") or 0) > 0
        and int(row.get("remote_count") or 0) == 0
        and int(row.get("stability_valid_count") or 0) > 0
        and row.get("scaled_stability_loss") is not None
        and float(row["scaled_stability_loss"]) > 0.0
        and row.get("conditional_remote_mass") is not None
        and float(row["conditional_remote_mass"]) < 0.05,
    ):
        return StopEvaluation(
            "STOP_REMOTE_STARVATION", ("Remote starvation persisted for 5 rollouts",)
        )
    if _consecutive(
        merged,
        5,
        lambda row: int(row.get("route_active_count") or 0) > 0
        and row.get("remote_share") is not None
        and float(row["remote_share"]) > 0.95,
    ):
        return StopEvaluation(
            "STOP_REMOTE_OVERUSE", ("Remote share > 0.95 for 5 rollouts",)
        )
    if baseline_records is not None:
        kpi = kpi_regression_gate(baseline_records, records)
        if kpi.status == "STOP_KPI_REGRESSION":
            return kpi
    return StopEvaluation("CONTINUE")


def detect_baseline_clear_collapse(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen (OR) AND local-share AND gradient composite rule."""

    rollouts = late_window(_merged_rollouts(records))
    route_window = _consecutive(
        rollouts,
        5,
        lambda row: int(row.get("route_active_count") or 0) > 0
        and (
            int(row.get("remote_count") or 0) == 0
            or (
                row.get("conditional_remote_mass") is not None
                and float(row["conditional_remote_mass"]) < 0.05
            )
        )
        and row.get("local_share") is not None
        and float(row["local_share"]) > 0.95,
    )
    ratios = [
        float(row["remote_local_grad_norm_ratio"])
        for row in rollouts
        if row.get("gradient_ratio_valid") is True
        and row.get("remote_local_grad_norm_ratio") is not None
    ]
    gradient_median = None if not ratios else float(statistics.median(ratios))
    gradient_collapsed = gradient_median is not None and gradient_median < 0.05
    return {
        "clear_local_collapse": route_window and gradient_collapsed,
        "route_and_local_five_rollout_window": route_window,
        "late_gradient_ratio_median": gradient_median,
        "late_gradient_ratio_below_0_05": gradient_collapsed,
        "late_rollout_count": len(rollouts),
    }


def evaluate_candidate_final(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    group: str,
) -> dict[str, Any]:
    baseline_collapse = detect_baseline_clear_collapse(baseline_records)
    if not baseline_collapse["clear_local_collapse"]:
        return {"status": "INCONCLUSIVE", "baseline_collapse": baseline_collapse}
    rollouts = late_window(_merged_rollouts(candidate_records))
    ratios = [
        float(row["remote_local_grad_norm_ratio"])
        for row in rollouts
        if row.get("gradient_ratio_valid") is True
        and row.get("remote_local_grad_norm_ratio") is not None
    ]
    if len(ratios) < 5:
        return {
            "status": "INCONCLUSIVE",
            "reason": "fewer than 5 valid late gradient rollouts",
            "baseline_collapse": baseline_collapse,
        }
    gates = {
        "no_remote_zero_streak": not _consecutive(
            rollouts,
            5,
            lambda row: int(row.get("route_active_count") or 0) > 0
            and int(row.get("remote_count") or 0) == 0,
        ),
        "no_low_conditional_mass_streak": not _consecutive(
            rollouts,
            5,
            lambda row: row.get("conditional_remote_mass") is not None
            and float(row["conditional_remote_mass"]) < 0.05,
        ),
        "no_remote_overuse_streak": not _consecutive(
            rollouts,
            5,
            lambda row: int(row.get("route_active_count") or 0) > 0
            and row.get("remote_share") is not None
            and float(row["remote_share"]) > 0.95,
        ),
        "late_gradient_median": statistics.median(ratios) >= 0.05,
        "final_five_gradient_mean": math.fsum(ratios[-5:]) / 5 >= 0.05,
        "no_near_zero_gradient_streak": not _consecutive(
            rollouts,
            5,
            lambda row: row.get("gradient_ratio_valid") is True
            and row.get("remote_local_grad_norm_ratio") is not None
            and float(row["remote_local_grad_norm_ratio"]) < 0.01,
        ),
    }
    stop = evaluate_stop(candidate_records, group=group)
    kpi = kpi_regression_gate(baseline_records, candidate_records)
    gates["ppo"] = stop.status == "CONTINUE"
    gates["kpi"] = kpi.status == "CONTINUE"
    if all(gates.values()):
        status = "PASS"
    elif kpi.status == "INCONCLUSIVE" and all(
        value for name, value in gates.items() if name != "kpi"
    ):
        status = "INCONCLUSIVE"
    else:
        status = "FAIL"
    return {
        "status": status,
        "gates": gates,
        "stop_evaluation": stop.to_dict(),
        "kpi_evaluation": kpi.to_dict(),
        "baseline_collapse": baseline_collapse,
        "late_gradient_ratio_median": float(statistics.median(ratios)),
        "final_five_gradient_ratio_mean": math.fsum(ratios[-5:]) / 5,
    }


def select_candidate(
    baseline_records: Sequence[Mapping[str, Any]],
    a1_records: Sequence[Mapping[str, Any]],
    a2_records: Sequence[Mapping[str, Any]],
    *,
    hard_cap_reached: bool = False,
) -> dict[str, Any]:
    baseline = detect_baseline_clear_collapse(baseline_records)
    if not baseline["clear_local_collapse"]:
        return {
            "selection": (
                "INCONCLUSIVE_AT_HARD_CAP" if hard_cap_reached else "INCONCLUSIVE"
            ),
            "baseline_collapse": baseline,
        }
    a1 = evaluate_candidate_final(baseline_records, a1_records, group="a1")
    a2 = evaluate_candidate_final(baseline_records, a2_records, group="a2")
    if a1["status"] == "PASS":
        selection = "SELECT_A1"
    elif a2["status"] == "PASS":
        selection = "SELECT_A2"
    elif "INCONCLUSIVE" in {a1["status"], a2["status"]}:
        selection = "INCONCLUSIVE"
    else:
        selection = "REJECT"
    return {"selection": selection, "baseline_collapse": baseline, "a1": a1, "a2": a2}


_ROLLOUT_TIMESERIES_FIELDS = (
    "route_active_count",
    "local_count",
    "remote_count",
    "defer_count",
    "idle_count",
    "remote_share",
    "local_share",
    "conditional_remote_mass",
    "remote_local_binary_entropy",
    "floor_violation_fraction",
    "stability_valid_count",
    "unscaled_stability_loss",
    "scaled_stability_loss",
    "remote_scorer_grad_norm",
    "local_route_row_grad_norm",
    "remote_local_grad_norm_ratio",
    "approx_kl_mean",
    "approx_kl_max",
    "clip_fraction_mean",
    "clip_fraction_max",
    "policy_loss",
    "critic_loss",
    "route_entropy",
    "global_grad_norm_before_clip",
    "gradient_clip_fraction",
)
_EPISODE_TIMESERIES_FIELDS = (
    "episode_return",
    "completed_tasks",
    "expired_tasks",
    "truncated_tasks",
    "generated_tasks",
    "actual_energy_joules",
    "completion_rate",
    "expiration_rate",
    "truncation_rate",
    "energy_j_per_env_step",
    "remote_completed",
    "remote_expired",
    "remote_truncated",
    "local_completed",
    "local_expired",
    "local_truncated",
)


def _timeseries_rows(
    records: Sequence[Mapping[str, Any]], group: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rollouts = _merged_rollouts(records)
    for field in _ROLLOUT_TIMESERIES_FIELDS:
        smooth = trailing_mean(
            [None if row.get(field) is None else float(row[field]) for row in rollouts], 5
        )
        for index, row in enumerate(rollouts):
            rows.append(
                {
                    "group": group,
                    "record_type": "rollout",
                    "environment_step": row["environment_step"],
                    "series_index": row["update_index"],
                    "metric": field,
                    "raw_value": row.get(field),
                    "trailing_5_mean": smooth[index],
                }
            )
    episodes = _sorted_unique(records, "episode", "episode_index")
    for field in _EPISODE_TIMESERIES_FIELDS:
        smooth = trailing_mean(
            [None if row.get(field) is None else float(row[field]) for row in episodes], 5
        )
        for index, row in enumerate(episodes):
            rows.append(
                {
                    "group": group,
                    "record_type": "episode",
                    "environment_step": row["environment_step"],
                    "series_index": row["episode_index"],
                    "metric": field,
                    "raw_value": row.get(field),
                    "trailing_5_mean": smooth[index],
                }
            )
    return rows


def analyze_matched_triplet(
    baseline_path: str | Path,
    a1_path: str | Path,
    a2_path: str | Path,
    *,
    output_directory: str | Path,
    hard_cap_reached: bool = False,
) -> dict[str, Any]:
    """Analyze three fact-source JSONLs and publish summary plus plot data."""

    loaded = {}
    tail_status = {}
    for group, path in (("baseline", baseline_path), ("a1", a1_path), ("a2", a2_path)):
        records, ignored = load_live_jsonl(path)
        if any(item.get("group") != group for item in records):
            raise MatchedAnalysisError(f"{group} JSONL contains another group identity")
        loaded[group] = records
        tail_status[group] = ignored
    selection = select_candidate(
        loaded["baseline"], loaded["a1"], loaded["a2"], hard_cap_reached=hard_cap_reached
    )
    windows = {}
    stop_evaluations = {}
    for group in ("baseline", "a1", "a2"):
        rollouts = _merged_rollouts(loaded[group])
        episodes = _sorted_unique(loaded[group], "episode", "episode_index")
        warmup_rollouts = warmup_window(rollouts)
        late_rollouts = late_window(rollouts)
        warmup_episodes = warmup_window(episodes)
        late_episodes = late_window(episodes)
        windows[group] = {
            "rollout_count": len(rollouts),
            "rollout_warmup_count": len(warmup_rollouts),
            "rollout_late_count": len(late_rollouts),
            "rollout_warmup_indices": [
                row["update_index"] for row in warmup_rollouts
            ],
            "rollout_late_indices": [
                row["update_index"] for row in late_rollouts
            ],
            "episode_count": len(episodes),
            "episode_warmup_count": len(warmup_episodes),
            "episode_late_count": len(late_episodes),
            "episode_warmup_indices": [
                row["episode_index"] for row in warmup_episodes
            ],
            "episode_late_indices": [
                row["episode_index"] for row in late_episodes
            ],
        }
        stop_evaluations[group] = evaluate_stop(
            loaded[group],
            group=group,
            baseline_records=(None if group == "baseline" else loaded["baseline"]),
        ).to_dict()
    summary = {
        "schema_version": 1,
        "source": "live_training_diagnostics.jsonl",
        "single_seed_diagnostic_only": True,
        "warmup_fraction": 0.20,
        "late_fraction": 0.20,
        "trailing_rollout_window": 5,
        "trailing_episode_window": 5,
        "truncated_tail_ignored": tail_status,
        "analysis_windows": windows,
        "stop_evaluations": stop_evaluations,
        "selection": selection,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "matched_summary.json"
    csv_path = output / "matched_timeseries.csv"
    for path in (summary_path, csv_path):
        if path.exists():
            raise MatchedAnalysisError(f"matched analyzer refuses overwrite: {path}")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = [
        row
        for group in ("baseline", "a1", "a2")
        for row in _timeseries_rows(loaded[group], group)
    ]
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "group",
                "record_type",
                "environment_step",
                "series_index",
                "metric",
                "raw_value",
                "trailing_5_mean",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return {**summary, "summary_path": str(summary_path), "timeseries_path": str(csv_path)}


def plot_matched_timeseries(
    timeseries_csv: str | Path,
    *,
    output_directory: str | Path,
) -> tuple[str, ...]:
    """Generate ten diagnostic plots only from a real matched timeseries CSV."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as exc:
        raise MatchedAnalysisError("matplotlib is required for matched plots") from exc
    with Path(timeseries_csv).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metrics = (
        "remote_share",
        "local_share",
        "conditional_remote_mass",
        "remote_local_binary_entropy",
        "floor_violation_fraction",
        "remote_local_grad_norm_ratio",
        "episode_return",
        "completion_rate",
        "expiration_rate",
        "route_entropy",
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {"baseline": "#555555", "a1": "#1f77b4", "a2": "#d62728"}
    for metric in metrics:
        path = output / f"matched_{metric}.png"
        if path.exists():
            raise MatchedAnalysisError(f"matched plot refuses overwrite: {path}")
        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        found = False
        for group in ("baseline", "a1", "a2"):
            selected = [row for row in rows if row["group"] == group and row["metric"] == metric]
            if not selected:
                continue
            x = [int(row["environment_step"]) for row in selected]
            raw = [math.nan if row["raw_value"] == "" else float(row["raw_value"]) for row in selected]
            smooth = [
                math.nan if row["trailing_5_mean"] == "" else float(row["trailing_5_mean"])
                for row in selected
            ]
            axis.plot(x, raw, color=colors[group], alpha=0.20, linewidth=0.8)
            axis.plot(x, smooth, color=colors[group], label=group, linewidth=1.8)
            found = True
        if not found:
            plt.close(fig)
            raise MatchedAnalysisError(f"timeseries CSV lacks metric {metric}")
        axis.set_xlabel("Environment steps")
        axis.set_ylabel(metric)
        axis.set_title(f"{metric} (single-seed diagnostic)")
        axis.grid(True, alpha=0.25, linestyle="--")
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return tuple(paths)


__all__ = [
    "MatchedAnalysisError",
    "STOP_STATUSES",
    "StopEvaluation",
    "analyze_matched_triplet",
    "detect_baseline_clear_collapse",
    "evaluate_candidate_final",
    "evaluate_stop",
    "kpi_regression_gate",
    "late_window",
    "load_live_jsonl",
    "plot_matched_timeseries",
    "select_candidate",
    "symmetric_degradation",
    "trailing_mean",
    "warmup_count",
    "warmup_window",
]
