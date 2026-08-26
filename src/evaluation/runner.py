"""Independent deterministic Validation Gate V1 Evaluation Runner."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ..artifacts import ArtifactConflictError, atomic_write_text
from ..config import RunConfig
from ..env.actions import ActionProposal
from ..env.environment import U2UMECEnvironment
from ..models.ca_gat_mappo import ActorObservationTensorizer
from ..models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)
from ..policies.heuristic_policy import HeuristicPolicy
from ..policies.local_only_policy import LocalOnlyPolicy
from ..policies.random_policy import RandomPolicy
from .actor_loader import (
    LoadedEvaluationActor,
    checkpoint_sha256,
    load_actor_for_evaluation,
)
from .metrics import (
    FORMAL_EVALUATION_SCHEMA_VERSION,
    FORMAL_METRIC_FIELDS,
    EvaluationEpisodeResult,
    aggregate_episode_metrics,
    build_episode_metrics,
)
from .protocol import EvaluationProtocol


FORMAL_METHOD_SUITE = (
    "ca_gat_mappo",
    "heuristic",
    "local_only",
    "random",
)
EVALUATION_ARTIFACT_FILENAMES = (
    "validation_protocol_snapshot.json",
    "episode_metrics.jsonl",
    "aggregate_metrics.json",
    "aggregate_metrics.csv",
    "evaluation_manifest.json",
)
FAILED_RUN_MARKER_FILENAME = "RUN_FAILED.json"
_CONSUMED_ENVIRONMENT_STREAMS = (
    "reset_mobility",
    "task_arrival",
    "task_workload",
    "channel_fading",
    "csi_error",
)


class FormalEvaluationError(RuntimeError):
    """Raised when Validation Gate V1 cannot complete honestly."""


def _normalize_evaluation_device_identity(value: str) -> str:
    """Return a stable Windows-safe identity for one resolved torch device."""

    if not isinstance(value, str):
        raise FormalEvaluationError("resolved evaluation device identity is invalid")
    match = re.fullmatch(r"(cpu|cuda)(?::([0-9]+))?", value.strip().lower())
    if match is None:
        raise FormalEvaluationError("resolved evaluation device identity is invalid")
    device_type, index = match.groups()
    return device_type if index is None else f"{device_type}-{index}"


def _normalize_evaluation_dtype_identity(value: str) -> str:
    """Return a stable Windows-safe identity for one resolved torch dtype."""

    if not isinstance(value, str):
        raise FormalEvaluationError("resolved evaluation dtype identity is invalid")
    normalized = value.strip().lower()
    if normalized.startswith("torch."):
        normalized = normalized.removeprefix("torch.")
    if re.fullmatch(r"[a-z][a-z0-9]*", normalized) is None:
        raise FormalEvaluationError("resolved evaluation dtype identity is invalid")
    return normalized


@dataclass(frozen=True)
class EvaluationWorkspaceLayout:
    """Machine-local source/input and evaluator/output roots."""

    evaluator_root: Path
    source_training_root: Path

    def __post_init__(self) -> None:
        evaluator = self.evaluator_root.resolve()
        source = self.source_training_root.resolve()
        if evaluator == source:
            raise FormalEvaluationError(
                "evaluator and source training worktrees must be different"
            )
        object.__setattr__(self, "evaluator_root", evaluator)
        object.__setattr__(self, "source_training_root", source)

    @classmethod
    def discover_formal_v1(cls) -> "EvaluationWorkspaceLayout":
        evaluator = Path(__file__).resolve().parents[2]
        required = Path(r"D:\My_project_li_validation_gate_v1")
        try:
            required = required.resolve(strict=True)
        except OSError as exc:
            raise FormalEvaluationError(
                f"formal Evaluation worktree is unavailable: {exc}"
            ) from exc
        if evaluator != required:
            raise FormalEvaluationError(
                "Validation Gate V1 must run from "
                r"D:\My_project_li_validation_gate_v1"
            )
        source = evaluator.parent / "My_project_li"
        if not source.is_dir():
            raise FormalEvaluationError(
                f"source training worktree is unavailable: {source}"
            )
        return cls(evaluator_root=evaluator, source_training_root=source)

    def checkpoint_directory(self, source_run_id: str) -> Path:
        return (
            self.source_training_root
            / "logs"
            / source_run_id
            / "checkpoints"
        )


@dataclass(frozen=True)
class FormalEvaluationResult:
    """Complete in-memory evaluation plus optional immutable artifacts."""

    evaluation_run_id: str
    manifest: Mapping[str, Any]
    episodes: tuple[EvaluationEpisodeResult, ...]
    aggregate_metrics: Mapping[str, Any]
    artifacts: tuple[str, ...]


class FormalEvaluationRunner:
    """Evaluate a frozen Actor and three baselines on aligned realizations."""

    def __init__(
        self,
        evaluator_config: RunConfig,
        checkpoint_path: str | Path,
        *,
        evaluation_device: str,
        protocol: EvaluationProtocol,
        workspace_layout: EvaluationWorkspaceLayout | None = None,
        environment_factory: Callable[[RunConfig], U2UMECEnvironment] | None = None,
    ) -> None:
        if not isinstance(evaluator_config, RunConfig):
            raise TypeError("evaluator_config must be a RunConfig")
        if not isinstance(protocol, EvaluationProtocol):
            raise TypeError("protocol must be an EvaluationProtocol")
        evaluator_config.validate()
        self._validate_evaluator_config(evaluator_config, protocol)
        if protocol.method_ids != FORMAL_METHOD_SUITE:
            raise FormalEvaluationError("protocol method suite differs from runner")
        self.evaluator_config = evaluator_config
        self.protocol = protocol
        self.workspace_layout = (
            workspace_layout or EvaluationWorkspaceLayout.discover_formal_v1()
        )
        self.environment_factory = environment_factory or U2UMECEnvironment
        target, expected_checkpoint = self._resolve_checkpoint(checkpoint_path)
        self.loaded_actor = load_actor_for_evaluation(
            target,
            protocol=protocol,
            expected_checkpoint=expected_checkpoint,
            evaluation_device=evaluation_device,
        )
        self.config = self.loaded_actor.source_run_config
        self.actor_tensorizer = ActorObservationTensorizer(self.config)
        self.action_distribution = CAGATMAPPOActionDistribution(
            self.loaded_actor.actor,
            self.config,
        )
        self.evaluation_config_identity = self._evaluation_config_identity()
        self.evaluation_config_sha256 = _canonical_sha256(
            self.evaluation_config_identity
        )
        self.evaluation_run_id = self._evaluation_run_id()

    def run(self, *, write_artifacts: bool = True) -> FormalEvaluationResult:
        """Run all protocol seeds and delay publication until every gate passes."""

        run_directory = self.run_directory()
        if write_artifacts and run_directory.exists():
            raise ArtifactConflictError(
                "formal evaluation run directory already exists; refusing overwrite: "
                f"{run_directory}"
            )

        actor = self.loaded_actor.actor
        actor.eval()
        before_state = _state_dict_snapshot(actor)
        before_requires_grad = tuple(
            (name, parameter.requires_grad)
            for name, parameter in actor.named_parameters()
        )

        episodes: list[EvaluationEpisodeResult] = []
        for evaluation_seed in self.protocol.validation_seeds:
            for method_id in FORMAL_METHOD_SUITE:
                episodes.append(self._run_episode(method_id, evaluation_seed))

        aggregate = aggregate_episode_metrics(episodes, FORMAL_METHOD_SUITE)
        aggregate["evaluation_run_id"] = self.evaluation_run_id
        checkpoint_hash_after_evaluation = checkpoint_sha256(
            self.loaded_actor.source_checkpoint_path
        )
        self._validate_checkpoint_hashes(checkpoint_hash_after_evaluation)
        shared_traces = self._validate_external_traces(tuple(episodes))
        self._validate_actor_unchanged(
            before_state=before_state,
            before_requires_grad=before_requires_grad,
        )
        manifest = self._manifest(
            tuple(episodes),
            before_requires_grad,
            checkpoint_hash_after_evaluation,
            shared_traces,
        )
        artifacts = (
            self._write_artifacts(manifest, tuple(episodes), aggregate)
            if write_artifacts
            else ()
        )
        return FormalEvaluationResult(
            evaluation_run_id=self.evaluation_run_id,
            manifest=manifest,
            episodes=tuple(episodes),
            aggregate_metrics=aggregate,
            artifacts=artifacts,
        )

    def run_directory(self) -> Path:
        return (
            self.workspace_layout.evaluator_root
            / "logs"
            / "evaluations"
            / self.protocol.evaluation_phase
            / self.evaluation_run_id
        )

    def artifact_paths(self) -> Mapping[str, Path]:
        """Return all five formal paths without creating their directory."""

        root = self.run_directory()
        names = (
            "protocol_snapshot",
            "episode_metrics",
            "aggregate_metrics",
            "aggregate_metrics_csv",
            "manifest",
        )
        return {
            name: root / filename
            for name, filename in zip(names, EVALUATION_ARTIFACT_FILENAMES)
        }

    def _resolve_checkpoint(self, checkpoint_path: str | Path):
        candidate = Path(checkpoint_path)
        try:
            selected = self.protocol.checkpoint_for_filename(candidate.name)
        except ValueError as exc:
            raise FormalEvaluationError(str(exc)) from exc
        expected_path = (
            self.workspace_layout.checkpoint_directory(self.protocol.source_run_id)
            / selected.filename
        )
        try:
            target = candidate.resolve(strict=True)
            expected = expected_path.resolve(strict=True)
        except OSError as exc:
            raise FormalEvaluationError(
                f"formal checkpoint path cannot be resolved: {exc}"
            ) from exc
        if target != expected:
            raise FormalEvaluationError(
                "checkpoint must be the allowlisted file in the fixed source run"
            )
        try:
            target.relative_to(self.workspace_layout.evaluator_root)
        except ValueError:
            pass
        else:
            raise FormalEvaluationError(
                "checkpoint input must not be inside the Evaluation worktree"
            )
        return target, selected

    def _run_episode(
        self,
        method_id: str,
        evaluation_seed: int,
    ) -> EvaluationEpisodeResult:
        episode_config = replace(
            self.config,
            mode="evaluation",
            method_id=method_id,
            launch_profile=None,
            seed=evaluation_seed,
        )
        episode_config.validate()
        environment = self.environment_factory(episode_config)
        reset = environment.reset()
        observations = reset.observations
        external_trace: list[Mapping[str, Any]] = [
            self._external_trace_record(environment, phase="reset", arrival=None)
        ]
        slot_infos: list[Mapping[str, Any]] = []
        action_trace: list[tuple[Mapping[str, Any], ...]] = []
        hidden_trace_sha256: list[str] = []

        policy: RandomPolicy | HeuristicPolicy | LocalOnlyPolicy | None = None
        hidden: torch.Tensor | None = None
        if method_id == "ca_gat_mappo":
            actor = self.loaded_actor.actor
            actor.eval()
            hidden = actor.initial_hidden(
                1,
                device=self.loaded_actor.evaluation_device,
                dtype=torch.float32,
            )
            if torch.count_nonzero(hidden).item() != 0:
                raise FormalEvaluationError("actor initial hidden state is not zero")
            hidden_trace_sha256.append(_tensor_sha256(hidden))
        elif method_id == "local_only":
            policy = LocalOnlyPolicy(episode_config, policy_seed=evaluation_seed)
        elif method_id == "heuristic":
            policy = HeuristicPolicy(episode_config, policy_seed=evaluation_seed)
        elif method_id == "random":
            policy = RandomPolicy(evaluation_seed)
        else:
            raise FormalEvaluationError(f"unknown formal method {method_id!r}")

        while True:
            if method_id == "ca_gat_mappo":
                assert hidden is not None
                proposals, hidden = self._actor_actions(observations, hidden)
                hidden_trace_sha256.append(_tensor_sha256(hidden))
            else:
                assert policy is not None
                proposals = tuple(policy.act(item) for item in observations)
            action_trace.append(tuple(_proposal_record(item) for item in proposals))
            step = environment.step(proposals)
            slot_infos.append(step.info)
            external_trace.append(
                self._external_trace_record(
                    environment,
                    phase=f"slot_{int(step.info['slot'])}_complete",
                    arrival=step.info["arrival"],
                )
            )
            if step.terminated or step.truncated:
                if step.observations is not None or step.centralized_state is not None:
                    raise FormalEvaluationError(
                        "terminal evaluation step exposed a fake next state"
                    )
                break
            if step.observations is None:
                raise FormalEvaluationError(
                    "nonterminal evaluation step omitted actor observations"
                )
            observations = step.observations

        if len(slot_infos) != self.protocol.episode_horizon_slots:
            raise FormalEvaluationError(
                "evaluation episode did not execute the protocol horizon"
            )
        environment.assert_invariants()
        action_trace_tuple = tuple(action_trace)
        external_trace_tuple = tuple(external_trace)
        return build_episode_metrics(
            method_id=method_id,
            evaluation_seed=evaluation_seed,
            environment=environment,
            slot_infos=slot_infos,
            action_trace=action_trace_tuple,
            action_trace_sha256=_canonical_sha256(action_trace_tuple),
            external_trace=external_trace_tuple,
            external_trace_sha256=_canonical_sha256(external_trace_tuple),
            hidden_trace_sha256=tuple(hidden_trace_sha256),
        )

    def _actor_actions(
        self,
        observations: Sequence[Any],
        hidden: torch.Tensor,
    ) -> tuple[tuple[ActionProposal, ...], torch.Tensor]:
        actor = self.loaded_actor.actor
        if actor.training:
            raise FormalEvaluationError("actor must remain in eval mode")
        with torch.no_grad():
            batch = self.actor_tensorizer.encode_step(
                observations,
                device=self.loaded_actor.evaluation_device,
                dtype=torch.float32,
                episode_start=int(observations[0].slot) == 0,
            )
            masks = SequentialActionMaskBatch.from_observations(observations)
            output = self.action_distribution.deterministic_actions(
                batch,
                masks,
                hidden,
            )
        if output.mode != "deterministic":
            raise FormalEvaluationError("actor did not use deterministic action mode")
        return tuple(output.proposals[0][0]), output.hidden_out.detach()

    def _external_trace_record(
        self,
        environment: U2UMECEnvironment,
        *,
        phase: str,
        arrival: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if (
            environment.mobility_model is None
            or environment.traffic is None
            or environment.channel_model is None
            or environment.channel_history is None
            or environment.mobility is None
            or environment.channel is None
        ):
            raise FormalEvaluationError("environment trace components are unavailable")
        rngs = {
            "reset_mobility": environment.mobility_model.rng,
            "task_arrival": environment.traffic.arrival_rng,
            "task_workload": environment.traffic.workload_rng,
            "channel_fading": environment.channel_model.rng,
            "csi_error": environment.channel_history.csi_error_rng,
        }
        return {
            "phase": phase,
            "environment_seed": environment.config.seed,
            "rng_state_sha256": {
                name: _canonical_sha256(generator.bit_generator.state)
                for name, generator in rngs.items()
            },
            "mobility": {
                "slot": environment.mobility.slot,
                "positions_sha256": _array_sha256(environment.mobility.positions_m),
                "velocities_sha256": _array_sha256(
                    environment.mobility.velocities_mps
                ),
            },
            "channel": {
                "slot": environment.channel.slot,
                "true_channel_sha256": _array_sha256(environment.channel.channel),
                "shadowing_sha256": _array_sha256(
                    environment.channel.shadowing_db
                ),
            },
            "csi": (
                {
                    "slot": environment.channel_features.slot,
                    "stale_csi_sha256": _array_sha256(
                        environment.channel_features.stale_csi
                    ),
                    "valid_mask_sha256": _array_sha256(
                        environment.channel_features.csi_valid_mask
                    ),
                }
                if environment.channel_features is not None
                else {"terminal_state": "not_exposed"}
            ),
            "arrival": arrival,
            "interference_measurement_rng": {
                "stream_id": int(
                    environment.config.reproducibility.stream_ids[
                        "interference_measurement"
                    ]
                ),
                "status": "reserved_not_consumed_by_current_backend",
            },
        }

    def _validate_checkpoint_hashes(self, final_hash: str) -> None:
        loaded = self.loaded_actor
        if not (
            loaded.checkpoint_expected_sha256
            == loaded.checkpoint_sha256_before
            == loaded.checkpoint_sha256_after_actor_load
            == final_hash
        ):
            raise FormalEvaluationError(
                "checkpoint SHA-256 changed during complete Evaluation"
            )

    def _validate_external_traces(
        self,
        episodes: tuple[EvaluationEpisodeResult, ...],
    ) -> Mapping[str, str]:
        shared: dict[str, str] = {}
        for seed in self.protocol.validation_seeds:
            selected = tuple(item for item in episodes if item.evaluation_seed == seed)
            if tuple(item.method_id for item in selected) != FORMAL_METHOD_SUITE:
                raise FormalEvaluationError(
                    f"episode method suite is incomplete for seed {seed}"
                )
            traces = {item.method_id: item.external_trace_sha256 for item in selected}
            if len(set(traces.values())) != 1:
                raise FormalEvaluationError(
                    "external environment realization diverged across methods for "
                    f"evaluation seed {seed}: {traces}"
                )
            shared[str(seed)] = selected[0].external_trace_sha256
        return shared

    def _validate_actor_unchanged(
        self,
        *,
        before_state: Mapping[str, torch.Tensor],
        before_requires_grad: tuple[tuple[str, bool], ...],
    ) -> None:
        actor = self.loaded_actor.actor
        if actor.training:
            raise FormalEvaluationError("frozen actor left evaluation mode")
        after_state = _state_dict_snapshot(actor)
        if tuple(before_state) != tuple(after_state) or any(
            not torch.equal(before_state[name], after_state[name])
            for name in before_state
        ):
            raise FormalEvaluationError("frozen actor state changed during evaluation")
        after_requires_grad = tuple(
            (name, parameter.requires_grad)
            for name, parameter in actor.named_parameters()
        )
        if before_requires_grad != after_requires_grad:
            raise FormalEvaluationError(
                "actor parameter requires_grad state changed during evaluation"
            )

    def _evaluation_config_identity(self) -> Mapping[str, Any]:
        return {
            "source_training_config_hash": self.loaded_actor.source_training_config_hash,
            "scenario_id": self.config.scenario_id,
            "environment": _jsonable(asdict(self.config.environment)),
            "action": _jsonable(asdict(self.config.action)),
            "evaluation_protocol_sha256": self.protocol.sha256,
            "evaluation_seeds": list(self.protocol.validation_seeds),
            "episode_horizon_slots": self.protocol.episode_horizon_slots,
            "method_suite": list(FORMAL_METHOD_SUITE),
            "evaluation_device": self.loaded_actor.evaluation_device,
            "dtype": self.loaded_actor.dtype,
        }

    def _evaluation_run_id(self) -> str:
        loaded = self.loaded_actor
        source_digest = _canonical_sha256(self.protocol.source_run_id)[:8]
        device_identity = _normalize_evaluation_device_identity(
            loaded.evaluation_device
        )
        dtype_identity = _normalize_evaluation_dtype_identity(loaded.dtype)
        kind = {
            "PERIODIC_RESUME": "pr",
            "FINAL_COMPLETED": "fc",
        }[loaded.source_checkpoint_kind]
        return (
            f"val-v1__src-{source_digest}__s-{loaded.source_training_seed}__"
            f"k-{kind}__n-{loaded.source_checkpoint_step}__"
            f"c-{loaded.checkpoint_expected_sha256[:8]}__"
            f"p-{self.protocol.sha256[:8]}__"
            f"dev-{device_identity}__dtype-{dtype_identity}__"
            f"e-{self.evaluator_config.git_commit[:8]}"
        )

    def _manifest(
        self,
        episodes: tuple[EvaluationEpisodeResult, ...],
        requires_grad: tuple[tuple[str, bool], ...],
        checkpoint_hash_after_evaluation: str,
        shared_traces: Mapping[str, str],
    ) -> Mapping[str, Any]:
        traces_by_seed: dict[str, Any] = {}
        for seed in self.protocol.validation_seeds:
            selected = tuple(item for item in episodes if item.evaluation_seed == seed)
            traces_by_seed[str(seed)] = {
                "shared_external_trace_sha256": shared_traces[str(seed)],
                "method_trace_sha256": {
                    item.method_id: item.external_trace_sha256 for item in selected
                },
            }
        loaded = self.loaded_actor
        stream_ids = self.config.reproducibility.stream_ids
        return {
            "run_status": "completed",
            "evaluation_schema_version": FORMAL_EVALUATION_SCHEMA_VERSION,
            "evaluation_run_id": self.evaluation_run_id,
            "evaluation_phase": self.protocol.evaluation_phase,
            "evaluation_protocol_version": self.protocol.evaluation_protocol_version,
            "evaluation_protocol_sha256": self.protocol.sha256,
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "source_run_id": loaded.source_run_id,
            "source_training_seed": loaded.source_training_seed,
            "source_training_git_commit": loaded.source_training_git_commit,
            "evaluator_git_commit": self.evaluator_config.git_commit,
            "evaluator_git_branch": self.evaluator_config.git_branch,
            "evaluator_git_dirty": self.evaluator_config.git_dirty,
            "source_training_config_hash": loaded.source_training_config_hash,
            "recomputed_source_training_config_hash": (
                loaded.recomputed_source_training_config_hash
            ),
            "scenario_id": self.protocol.scenario_id,
            "episode_horizon_slots": self.protocol.episode_horizon_slots,
            "evaluation_seeds": list(self.protocol.validation_seeds),
            "method_suite": list(FORMAL_METHOD_SUITE),
            "source_checkpoint_path": str(loaded.source_checkpoint_path),
            "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
            "source_checkpoint_kind": loaded.source_checkpoint_kind,
            "checkpoint_kind": loaded.source_checkpoint_kind,
            "checkpoint_step": loaded.source_checkpoint_step,
            "checkpoint_expected_sha256": loaded.checkpoint_expected_sha256,
            "checkpoint_sha256_before": loaded.checkpoint_sha256_before,
            "checkpoint_sha256_after_actor_load": (
                loaded.checkpoint_sha256_after_actor_load
            ),
            "checkpoint_sha256_after_evaluation": checkpoint_hash_after_evaluation,
            "checkpoint_sha256_after": checkpoint_hash_after_evaluation,
            "source_method_id": loaded.source_method_id,
            "actor_architecture_identity": dict(loaded.actor_architecture_identity),
            "action_domain_identity": dict(loaded.action_domain_identity),
            "training_device": loaded.source_training_device,
            "evaluation_device": loaded.evaluation_device,
            "dtype": loaded.dtype,
            "checkpoint_map_location_contract": (
                "structured payload validated on CPU; actor state only copied to "
                "the explicit evaluation device"
            ),
            "deterministic_action_mode": "deterministic",
            "masked_argmax": True,
            "actor_eval_mode": True,
            "torch_no_grad": True,
            "actor_requires_grad_state": {
                name: value for name, value in requires_grad
            },
            "cpu_bitwise_repeatability_required": loaded.evaluation_device == "cpu",
            "cuda_bitwise_repeatability_claimed": False,
            "environment_rng_stream_contract": {
                name: {
                    "stream_id": int(stream_ids[name]),
                    "status": "consumed_and_traced",
                }
                for name in _CONSUMED_ENVIRONMENT_STREAMS
            }
            | {
                "interference_measurement": {
                    "stream_id": int(stream_ids["interference_measurement"]),
                    "status": "reserved_not_consumed_by_current_backend",
                    "action_dependent_measurement_not_forced_equal": True,
                }
            },
            "external_trace_sha256_by_seed": dict(shared_traces),
            "shared_external_trace_by_seed": traces_by_seed,
            "artifact_filenames": list(EVALUATION_ARTIFACT_FILENAMES),
        }

    def _write_artifacts(
        self,
        manifest: Mapping[str, Any],
        episodes: tuple[EvaluationEpisodeResult, ...],
        aggregate: Mapping[str, Any],
    ) -> tuple[str, ...]:
        paths = self.artifact_paths()
        run_directory = self.run_directory()
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactConflictError(
                "formal evaluation run directory already exists; refusing overwrite: "
                f"{run_directory}"
            ) from exc
        contents = (
            ("protocol_snapshot", self.protocol.snapshot_text),
            (
                "episode_metrics",
                "".join(
                    _compact_json(item.artifact_record(self.evaluation_run_id)) + "\n"
                    for item in episodes
                ),
            ),
            ("aggregate_metrics", _pretty_json(aggregate)),
            ("aggregate_metrics_csv", self._evaluation_csv(episodes)),
            ("manifest", _pretty_json(manifest)),
        )
        stage = "directory_created"
        try:
            for stage, content in contents:
                atomic_write_text(paths[stage], content)
        except Exception as exc:
            marker = run_directory / FAILED_RUN_MARKER_FILENAME
            failure = {
                "run_status": "failed",
                "evaluation_run_id": self.evaluation_run_id,
                "failed_stage": stage,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            try:
                atomic_write_text(marker, _pretty_json(failure))
            except Exception:
                pass
            raise FormalEvaluationError(
                f"artifact publication failed at {stage}: {exc}"
            ) from exc
        return tuple(str(paths[name]) for name, _content in contents)

    def _evaluation_csv(
        self,
        episodes: Sequence[EvaluationEpisodeResult],
    ) -> str:
        fields = (
            "evaluation_run_id",
            "method_id",
            "evaluation_seed",
            "episode_horizon",
            *FORMAL_METRIC_FIELDS,
            "completed_task_e2e_delay_valid_sample_count",
            "outage_rate_valid_sample_count",
            "action_trace_sha256",
            "external_trace_sha256",
        )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in episodes:
            row: dict[str, Any] = {
                "evaluation_run_id": self.evaluation_run_id,
                "method_id": item.method_id,
                "evaluation_seed": item.evaluation_seed,
                "episode_horizon": item.episode_horizon,
                **dict(item.metrics),
                "completed_task_e2e_delay_valid_sample_count": len(
                    item.completed_delay_samples_s
                ),
                "outage_rate_valid_sample_count": int(
                    item.metrics["outage_denominator"]
                ),
                "action_trace_sha256": item.action_trace_sha256,
                "external_trace_sha256": item.external_trace_sha256,
            }
            writer.writerow(
                {key: "NA" if value is None else value for key, value in row.items()}
            )
        return stream.getvalue()

    @staticmethod
    def _validate_evaluator_config(
        config: RunConfig,
        protocol: EvaluationProtocol,
    ) -> None:
        if config.mode != "evaluation" or config.method_id != "ca_gat_mappo":
            raise FormalEvaluationError(
                "Validation Gate V1 requires evaluation/ca_gat_mappo dispatch"
            )
        if not config.git_commit or config.git_commit == "unknown":
            raise FormalEvaluationError("evaluator git provenance is unavailable")
        if config.git_dirty:
            raise FormalEvaluationError(
                "formal Validation requires a clean evaluator worktree"
            )
        if config.git_commit == protocol.source_training_git_commit:
            raise FormalEvaluationError(
                "evaluator commit must be independent from the training source commit"
            )
        inference = protocol.inference_contract
        if (
            not inference.actor_eval
            or not inference.torch_no_grad
            or inference.action_selection != "deterministic_masked_argmax"
            or inference.gru_hidden_reset_scope != "episode"
            or inference.network_updates
        ):
            raise FormalEvaluationError("EvaluationProtocol inference contract is unsafe")


def _proposal_record(proposal: ActionProposal) -> Mapping[str, Any]:
    return {
        "uav_id": proposal.uav_id,
        "route": proposal.route,
        "tx_select": proposal.tx_select,
        "resource_group": proposal.resource_group,
        "resource_width": proposal.resource_width,
        "power_level": proposal.power_level,
        "cpu_queue": proposal.cpu_queue,
        "cpu_frequency": proposal.cpu_frequency,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_compact_json(value).encode("utf-8")).hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "EVALUATION_ARTIFACT_FILENAMES",
    "FAILED_RUN_MARKER_FILENAME",
    "FORMAL_METHOD_SUITE",
    "EvaluationWorkspaceLayout",
    "FormalEvaluationError",
    "FormalEvaluationResult",
    "FormalEvaluationRunner",
]
