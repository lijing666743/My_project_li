"""Interactive and direct command-line launchers for the experiment runner."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any, Sequence

from .config import (
    ConfigError,
    DEFAULT_SEED,
    SUPPORTED_SCENARIOS,
    load_run_config,
)
from .runner import Runner


MENU = """U2U-MEC Experiment Launcher

[1] Environment sanity check
[2] Gate 0 tests
[3] Random policy rollout
[4] Heuristic policy rollout
[5] Baseline experiment
[6] RL training
[7] Evaluation
[8] Ablation
[9] Plot results
[0] Exit"""

MENU_ROUTES: dict[str, tuple[str, str]] = {
    "1": ("environment_sanity", "environment"),
    "2": ("gate0", "environment"),
    "3": ("random", "random"),
    "4": ("heuristic", "heuristic"),
    "5": ("baseline", "factorized_action_gat_qmix"),
    "6": ("rl", "ca_gat_mappo"),
    "7": ("evaluation", "ca_gat_mappo"),
    "8": ("ablation", "ca_gat_mappo"),
    "9": ("plot", "environment"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="U2U-MEC experiment launcher")
    parser.add_argument("--mode", help="execution mode, e.g. random")
    parser.add_argument("--method-id", help="registered method identifier")
    parser.add_argument("--scenario-id", choices=SUPPORTED_SCENARIOS, help="small, medium, or large")
    parser.add_argument("--config", help="JSON or YAML configuration file")
    parser.add_argument("--seed", type=int, help=f"master seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set a typed config leaf, may be repeated",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="print the canonical RunConfig and stop before dispatch",
    )
    return parser


def build_run_config_from_args(args: argparse.Namespace):
    cli_overrides: dict[str, Any] = {}
    for name in ("mode", "method_id", "scenario_id", "seed"):
        value = getattr(args, name, None)
        if value is not None:
            cli_overrides[name] = value
    for raw_override in getattr(args, "overrides", []) or []:
        key, separator, raw_value = raw_override.partition("=")
        if not separator or not key.strip():
            raise ConfigError(f"invalid --set override {raw_override!r}; expected KEY=VALUE")
        cli_overrides[key.strip()] = _parse_cli_value(raw_value.strip())
    return load_run_config(args.config, cli_overrides=cli_overrides)


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run the same config -> registry -> runner chain for every launch path."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return interactive_main(input_fn=input_fn, output_fn=output_fn)

    parser = build_arg_parser()
    args = parser.parse_args(arguments)
    try:
        config = build_run_config_from_args(args)
    except ConfigError as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    if args.show_config:
        output_fn(json.dumps(config.snapshot_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return _dispatch(config, output_fn=output_fn)


def interactive_main(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    output_fn(MENU)
    choice = _prompt_choice(input_fn, output_fn, "Select an option [1-9, 0]: ", tuple(MENU_ROUTES) + ("0",), "1")
    if choice == "0":
        output_fn("Exiting.")
        return 0

    mode, method_id = MENU_ROUTES[choice]
    scenario_id = _prompt_choice(
        input_fn,
        output_fn,
        "Scenario [small/medium/large] (default: small): ",
        SUPPORTED_SCENARIOS,
        "small",
    )
    config_path = _prompt_optional(input_fn, output_fn, "Config path (optional, press Enter for defaults): ")
    try:
        base_config = load_run_config(config_path or None)
        seed = _prompt_int(
            input_fn,
            output_fn,
            f"Master seed (default: {base_config.seed}, source: config/default): ",
            base_config.seed,
            minimum=0,
        )
        interactive_overrides = {
            "mode": mode,
            "method_id": method_id,
            "scenario_id": scenario_id,
            "seed": seed,
        }
        config = load_run_config(config_path or None, interactive_overrides=interactive_overrides)
    except ConfigError as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    return _dispatch(config, output_fn=output_fn)


def _dispatch(config, *, output_fn: Callable[[str], None]) -> int:
    output_fn(
        f"Resolved RunConfig: mode={config.mode}, method_id={config.method_id}, "
        f"scenario_id={config.scenario_id}, seed={config.seed}, run_id={config.run_id}"
    )
    result = Runner().run(config)
    output_fn(f"status={result.status}: {result.message}")
    if result.artifacts:
        output_fn("artifacts=" + ", ".join(result.artifacts))
    return 0 if result.status in {"completed", "unavailable"} else 1


def _prompt_choice(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    prompt: str,
    choices: Sequence[str],
    default: str,
) -> str:
    choice_set = set(choices)
    while True:
        value = input_fn(prompt).strip() or default
        if value in choice_set:
            return value
        output_fn(f"Invalid choice {value!r}; allowed choices: {', '.join(choices)}")


def _prompt_int(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    prompt: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    while True:
        raw = input_fn(prompt).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            output_fn("Invalid integer; please try again.")
            continue
        if minimum is not None and value < minimum:
            output_fn(f"Value must be >= {minimum}; please try again.")
            continue
        return value


def _prompt_optional(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    prompt: str,
) -> str:
    del output_fn  # kept in the signature for consistent test injection
    return input_fn(prompt).strip()


def _parse_cli_value(raw: str) -> Any:
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


__all__ = [
    "MENU",
    "build_arg_parser",
    "build_run_config_from_args",
    "interactive_main",
    "main",
]
