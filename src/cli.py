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
from .execution import ExecutionContext, validate_execution_context
from .runner import Runner


MENU = """U2U-MEC Experiment Launcher

[1] Environment sanity check
[2] Gate 0 tests
[3] Random policy rollout
[4] Heuristic policy rollout
[5] Local-only baseline experiment
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
    "5": ("baseline", "local_only"),
    "6": ("rl", "ca_gat_mappo"),
    "7": ("evaluation", "ca_gat_mappo"),
    "8": ("ablation", "ca_gat_mappo"),
    "9": ("plot", "environment"),
}

RL_PROFILES: dict[str, dict[str, Any]] = {
    "rl-smoke": {
        "mode": "rl",
        "method_id": "ca_gat_mappo",
        "training.formal_rl_enabled": True,
        "training.mappo.training_device": "cpu",
        "environment.episode_horizon": 32,
        "training.mappo.max_training_episodes": 8,
        "training.mappo.max_training_environment_steps": 256,
        "training.mappo.evaluation_interval_steps": 256,
        "training.mappo.checkpoint_interval_steps": 128,
    },
    "rl-long-smoke": {
        "mode": "rl",
        "method_id": "ca_gat_mappo",
        "scenario_id": "small",
        "seed": DEFAULT_SEED,
        "training.formal_rl_enabled": True,
        "training.mappo.training_device": "cuda",
        "environment.episode_horizon": 500,
        "training.mappo.max_training_episodes": 100,
        "training.mappo.max_training_environment_steps": 50000,
        "training.mappo.evaluation_interval_steps": 50000,
        "training.mappo.checkpoint_interval_steps": 2500,
    },
    "rl-formal": {
        "mode": "rl",
        "method_id": "ca_gat_mappo",
        "training.formal_rl_enabled": True,
        "training.mappo.training_device": "cuda",
    },
}

RL_PROFILE_ALIASES = {
    "smoke": "rl-smoke",
    "long-smoke": "rl-long-smoke",
    "formal": "rl-formal",
    "rl-smoke": "rl-smoke",
    "rl-long-smoke": "rl-long-smoke",
    "rl-formal": "rl-formal",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="U2U-MEC experiment launcher")
    parser.add_argument("--mode", help="execution mode, e.g. random")
    parser.add_argument("--method-id", help="registered method identifier")
    parser.add_argument("--scenario-id", choices=SUPPORTED_SCENARIOS, help="small, medium, or large")
    parser.add_argument("--config", help="JSON or YAML configuration file")
    parser.add_argument(
        "--profile",
        choices=tuple(RL_PROFILE_ALIASES),
        help=(
            "RL launch profile: rl-smoke (CPU diagnostic), or "
            "rl-long-smoke and rl-formal (CUDA)"
        ),
    )
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
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        metavar="CHECKPOINT_PATH",
        help="resume one explicit periodic checkpoint for an rl-formal run",
    )
    parser.add_argument(
        "--evaluate-from",
        dest="evaluate_from",
        metavar="CHECKPOINT_PATH",
        help="evaluate one Validation Gate V1 allowlisted checkpoint",
    )
    parser.add_argument(
        "--evaluation-device",
        choices=("cpu", "cuda"),
        help="explicit actor-only evaluation device; no automatic fallback",
    )
    return parser


def build_run_config_from_args(args: argparse.Namespace):
    profile = getattr(args, "profile", None)
    cli_overrides = _rl_profile_overrides(profile) if profile else {}
    for name in ("mode", "method_id", "scenario_id", "seed"):
        value = getattr(args, name, None)
        if value is not None:
            cli_overrides[name] = value
    for raw_override in getattr(args, "overrides", []) or []:
        key, separator, raw_value = raw_override.partition("=")
        if not separator or not key.strip():
            raise ConfigError(f"invalid --set override {raw_override!r}; expected KEY=VALUE")
        cli_overrides[key.strip()] = _parse_cli_value(raw_value.strip())
    config = load_run_config(args.config, cli_overrides=cli_overrides)
    _validate_formal_evaluation_cli_args(args, config.mode)
    return config


def _validate_formal_evaluation_cli_args(
    args: argparse.Namespace,
    resolved_mode: str,
) -> None:
    if resolved_mode != "evaluation":
        return
    forbidden: list[str] = []
    for name in ("config", "profile", "scenario_id", "seed"):
        if getattr(args, name, None) is not None:
            forbidden.append(f"--{name.replace('_', '-')}")
    if getattr(args, "overrides", None):
        forbidden.append("--set")
    if forbidden:
        raise ConfigError(
            "Validation Gate V1 does not accept CLI training/environment config: "
            + ", ".join(forbidden)
        )


def build_execution_context_from_args(args: argparse.Namespace) -> ExecutionContext:
    """Build launch instructions without adding them to ``RunConfig``."""

    if (
        getattr(args, "evaluate_from", None) is not None
        or getattr(args, "evaluation_device", None) is not None
    ):
        return ExecutionContext.from_paths(
            resume_from=getattr(args, "resume_from", None),
            evaluate_from=getattr(args, "evaluate_from", None),
            evaluation_device=getattr(args, "evaluation_device", None),
        )
    return ExecutionContext.from_resume_path(getattr(args, "resume_from", None))


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
        execution_context = build_execution_context_from_args(args)
        validate_execution_context(config, execution_context)
    except ConfigError as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    if args.show_config:
        output_fn(json.dumps(config.snapshot_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return _dispatch(
        config,
        execution_context=execution_context,
        output_fn=output_fn,
    )


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
    profile_overrides: dict[str, Any] = {}
    if mode == "rl":
        profile = _prompt_choice(
            input_fn,
            output_fn,
            "RL profile [smoke/long-smoke/formal] "
            "(default: smoke, source: conservative-default): ",
            ("smoke", "long-smoke", "formal"),
            "smoke",
        )
        profile_overrides = _rl_profile_overrides(profile)
    if mode == "evaluation":
        try:
            config = load_run_config(
                interactive_overrides={
                    "mode": "evaluation",
                    "method_id": "ca_gat_mappo",
                }
            )
        except ConfigError as exc:
            output_fn(f"Configuration error: {exc}")
            return 2
        checkpoint_path = _prompt_optional(
            input_fn,
            output_fn,
            "Validation Gate V1 checkpoint path (required): ",
        )
        evaluation_device = _prompt_choice(
            input_fn,
            output_fn,
            "Evaluation device [cpu/cuda] "
            "(default: cpu, source: conservative-default): ",
            ("cpu", "cuda"),
            "cpu",
        )
        execution_context = ExecutionContext.from_evaluation_path(
            checkpoint_path or None,
            evaluation_device,
        )
        try:
            validate_execution_context(config, execution_context)
        except ConfigError as exc:
            output_fn(f"Configuration error: {exc}")
            return 2
        return _dispatch(
            config,
            execution_context=execution_context,
            output_fn=output_fn,
        )
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
            **profile_overrides,
            "mode": mode,
            "method_id": method_id,
            "scenario_id": scenario_id,
            "seed": seed,
        }
        config = load_run_config(config_path or None, interactive_overrides=interactive_overrides)
    except ConfigError as exc:
        output_fn(f"Configuration error: {exc}")
        return 2
    return _dispatch(
        config,
        execution_context=None,
        output_fn=output_fn,
    )


def _dispatch(
    config,
    *,
    execution_context: ExecutionContext | None = None,
    output_fn: Callable[[str], None],
) -> int:
    output_fn(
        f"Resolved RunConfig: mode={config.mode}, method_id={config.method_id}, "
        f"scenario_id={config.scenario_id}, seed={config.seed}, run_id={config.run_id}"
    )
    if config.mode == "rl":
        requested_device = config.training.mappo.training_device
        cuda_available = _cuda_available()
        resolved_device = (
            requested_device
            if requested_device == "cpu" or cuda_available
            else "unavailable"
        )
        output_fn(
            "Runtime device provenance: "
            f"requested={requested_device}, resolved={resolved_device}, "
            f"cuda_available={str(cuda_available).lower()}, "
            f"cuda_runtime_version={config.reproducibility.cuda_version}"
        )
    if config.mode == "evaluation" and execution_context is not None:
        requested_device = execution_context.evaluation_device
        cuda_available = _cuda_available()
        resolved_device = (
            requested_device
            if requested_device == "cpu" or cuda_available
            else "unavailable"
        )
        output_fn(
            "Evaluation device provenance: "
            f"requested={requested_device}, resolved={resolved_device}, "
            f"cuda_available={str(cuda_available).lower()}"
        )
    result = Runner().run(config, execution_context=execution_context)
    output_fn(f"status={result.status}: {result.message}")
    if result.artifacts:
        output_fn("artifacts=" + ", ".join(result.artifacts))
    return 0 if result.status == "completed" else 1


def _rl_profile_overrides(profile: str) -> dict[str, Any]:
    try:
        canonical = RL_PROFILE_ALIASES[profile]
    except KeyError as exc:
        raise ConfigError(f"unknown RL profile: {profile!r}") from exc
    return {
        **RL_PROFILES[canonical],
        "launch_profile": canonical,
    }


def _cuda_available() -> bool:
    try:
        import torch
    except (ImportError, OSError):
        return False
    return bool(torch.cuda.is_available())


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
    "MENU_ROUTES",
    "RL_PROFILES",
    "build_arg_parser",
    "build_execution_context_from_args",
    "build_run_config_from_args",
    "interactive_main",
    "main",
]
