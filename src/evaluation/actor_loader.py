"""Strict actor-only loading from immutable Validation Gate V1 checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..config import (
    CHECKPOINT_KIND_FINAL_COMPLETED,
    ConfigError,
    RunConfig,
    _construct_run_config,
)
from ..models.ca_gat_mappo import CAGATMAPPOActor, MAPPOTensorSpec
from ..models.ca_gat_mappo_checkpoint import CheckpointError, load_checkpoint_payload
from .protocol import EvaluationCheckpoint, EvaluationProtocol


EVALUATION_DTYPE = torch.float32
_ARCHITECTURE_FIELDS = (
    "encoder_hidden_dimension",
    "gat_layer_count",
    "attention_head_count",
    "gru_hidden_dimension",
)


class EvaluationCheckpointError(RuntimeError):
    """Raised when a checkpoint is not safe for formal actor-only evaluation."""


@dataclass(frozen=True)
class LoadedEvaluationActor:
    """Frozen actor plus every validated source identity used by evaluation."""

    actor: CAGATMAPPOActor
    source_run_config: RunConfig
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    checkpoint_expected_sha256: str
    checkpoint_sha256_before: str
    checkpoint_sha256_after_actor_load: str
    source_checkpoint_kind: str
    source_checkpoint_step: int
    source_run_id: str
    source_method_id: str
    source_training_seed: int
    source_training_config_hash: str
    recomputed_source_training_config_hash: str
    source_training_git_commit: str
    source_training_device: str
    evaluation_device: str
    dtype: str
    actor_architecture_identity: Mapping[str, Any]
    action_domain_identity: Mapping[str, Any]


def checkpoint_sha256(path: str | Path) -> str:
    """Hash one explicit checkpoint path without globbing or fallback."""

    target = Path(path)
    if not target.is_file():
        raise EvaluationCheckpointError(
            f"evaluation checkpoint is not a regular file: {target}"
        )
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvaluationCheckpointError(
            f"evaluation checkpoint cannot be hashed: {exc}"
        ) from exc
    return digest.hexdigest()


def load_actor_for_evaluation(
    checkpoint_path: str | Path,
    *,
    protocol: EvaluationProtocol,
    expected_checkpoint: EvaluationCheckpoint,
    evaluation_device: str,
) -> LoadedEvaluationActor:
    """Load Actor state only after the complete V1 identity preflight."""

    if not isinstance(protocol, EvaluationProtocol):
        raise TypeError("protocol must be an EvaluationProtocol")
    if not isinstance(expected_checkpoint, EvaluationCheckpoint):
        raise TypeError("expected_checkpoint must be an EvaluationCheckpoint")
    device = _validate_evaluation_device(evaluation_device)
    target = _resolve_checkpoint(checkpoint_path)
    if target.name != expected_checkpoint.filename:
        raise EvaluationCheckpointError(
            "checkpoint filename differs from the EvaluationProtocol allowlist"
        )
    source_hash_before = checkpoint_sha256(target)
    if source_hash_before != expected_checkpoint.expected_sha256:
        raise EvaluationCheckpointError(
            "checkpoint SHA-256 differs from the EvaluationProtocol allowlist"
        )

    payload = _load_payload(target)
    _validate_checkpoint_payload_identity(payload, expected_checkpoint)
    source_config, source, metadata, recomputed_hash = _source_config_from_payload(
        payload,
        protocol=protocol,
    )
    runtime = _validate_runtime(payload)
    actor = _construct_actor(source_config, payload, device)
    source_hash_after_actor_load = checkpoint_sha256(target)
    if not (
        expected_checkpoint.expected_sha256
        == source_hash_before
        == source_hash_after_actor_load
    ):
        raise EvaluationCheckpointError(
            "evaluation checkpoint changed while Actor was being loaded"
        )
    return _loaded_actor_record(
        actor=actor,
        actor_config=source_config,
        source_config=source_config,
        target=target,
        payload=payload,
        source=source,
        metadata=metadata,
        recomputed_hash=recomputed_hash,
        runtime=runtime,
        expected_sha256=expected_checkpoint.expected_sha256,
        source_hash_before=source_hash_before,
        source_hash_after_actor_load=source_hash_after_actor_load,
        device=device,
    )


def load_final_actor_for_evaluation(
    config: RunConfig,
    checkpoint_path: str | Path,
    *,
    evaluation_device: str,
) -> LoadedEvaluationActor:
    """Compatibility wrapper for the pre-V1 FINAL_COMPLETED API."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    if config.mode != "evaluation" or config.method_id != "ca_gat_mappo":
        raise EvaluationCheckpointError(
            "formal actor evaluation requires evaluation/ca_gat_mappo"
        )
    device = _validate_evaluation_device(evaluation_device)
    target = _resolve_checkpoint(checkpoint_path)
    source_hash_before = checkpoint_sha256(target)
    payload = _load_payload(target)
    if payload["checkpoint_kind"] != CHECKPOINT_KIND_FINAL_COMPLETED:
        raise EvaluationCheckpointError(
            "formal evaluation requires checkpoint kind FINAL_COMPLETED; "
            f"got {payload['checkpoint_kind']!r}"
        )
    source_config, source, metadata, recomputed_hash = _source_config_from_payload(
        payload,
        protocol=None,
    )
    _validate_source_compatibility(config, source)
    runtime = _validate_runtime(payload)
    if not config.git_commit or config.git_commit == "unknown":
        raise EvaluationCheckpointError("evaluator git provenance is unavailable")
    actor = _construct_actor(config, payload, device)
    source_hash_after_actor_load = checkpoint_sha256(target)
    if source_hash_before != source_hash_after_actor_load:
        raise EvaluationCheckpointError(
            "evaluation checkpoint changed while Actor was being loaded"
        )
    return _loaded_actor_record(
        actor=actor,
        actor_config=config,
        source_config=source_config,
        target=target,
        payload=payload,
        source=source,
        metadata=metadata,
        recomputed_hash=recomputed_hash,
        runtime=runtime,
        expected_sha256=source_hash_before,
        source_hash_before=source_hash_before,
        source_hash_after_actor_load=source_hash_after_actor_load,
        device=device,
    )


def _resolve_checkpoint(path: str | Path) -> Path:
    try:
        return Path(path).resolve(strict=True)
    except OSError as exc:
        raise EvaluationCheckpointError(
            f"evaluation checkpoint cannot be resolved: {exc}"
        ) from exc


def _load_payload(target: Path) -> Mapping[str, Any]:
    try:
        return load_checkpoint_payload(target)
    except CheckpointError as exc:
        raise EvaluationCheckpointError(str(exc)) from exc


def _validate_checkpoint_payload_identity(
    payload: Mapping[str, Any],
    expected: EvaluationCheckpoint,
) -> None:
    if payload["checkpoint_kind"] != expected.checkpoint_kind:
        raise EvaluationCheckpointError("checkpoint kind differs from protocol")
    if payload["method_id"] != "ca_gat_mappo":
        raise EvaluationCheckpointError("checkpoint method_id mismatch")
    step = payload["training_state"]["collected_environment_transitions"]
    if isinstance(step, bool) or not isinstance(step, int):
        raise EvaluationCheckpointError("checkpoint transition step is invalid")
    if step != expected.checkpoint_step:
        raise EvaluationCheckpointError("checkpoint step differs from protocol")


def _source_config_from_payload(
    payload: Mapping[str, Any],
    *,
    protocol: EvaluationProtocol | None,
) -> tuple[RunConfig, Mapping[str, Any], Mapping[str, Any], str]:
    snapshot = payload["config_snapshot"]
    if not isinstance(snapshot, Mapping):
        raise EvaluationCheckpointError("checkpoint config snapshot is invalid")
    metadata = snapshot.get("_metadata")
    if not isinstance(metadata, Mapping):
        raise EvaluationCheckpointError("checkpoint config metadata is invalid")
    resolved = {key: value for key, value in snapshot.items() if key != "_metadata"}
    canonical = json.dumps(
        resolved,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if recomputed != payload["config_hash"]:
        raise EvaluationCheckpointError(
            "checkpoint training config hash does not match its canonical snapshot"
        )
    if metadata.get("config_hash") != recomputed:
        raise EvaluationCheckpointError(
            "checkpoint training config hash metadata is inconsistent"
        )
    if metadata.get("git_commit") != payload["git_commit"]:
        raise EvaluationCheckpointError(
            "checkpoint git provenance differs from its config snapshot"
        )
    if resolved.get("mode") != "rl" or resolved.get("method_id") != "ca_gat_mappo":
        raise EvaluationCheckpointError(
            "checkpoint source config is not an rl/ca_gat_mappo run"
        )
    if protocol is not None:
        if resolved.get("scenario_id") != protocol.scenario_id:
            raise EvaluationCheckpointError(
                "checkpoint scenario_id differs from EvaluationProtocol"
            )
        environment = resolved.get("environment")
        if not isinstance(environment, Mapping):
            raise EvaluationCheckpointError("checkpoint environment config is invalid")
        if environment.get("episode_horizon") != protocol.episode_horizon_slots:
            raise EvaluationCheckpointError(
                "checkpoint episode horizon differs from EvaluationProtocol"
            )
    try:
        source_config = _construct_run_config(resolved)
        source_config.validate()
    except (ConfigError, TypeError, ValueError) as exc:
        raise EvaluationCheckpointError(
            f"checkpoint RunConfig snapshot cannot be reconstructed: {exc}"
        ) from exc
    if source_config.config_hash != recomputed:
        raise EvaluationCheckpointError(
            "reconstructed RunConfig hash differs from checkpoint snapshot"
        )
    object.__setattr__(
        source_config, "git_branch", str(metadata.get("git_branch", "unknown"))
    )
    object.__setattr__(source_config, "git_commit", str(payload["git_commit"]))
    object.__setattr__(
        source_config, "git_dirty", bool(metadata.get("git_dirty", False))
    )

    if protocol is not None:
        if recomputed != protocol.source_training_config_hash:
            raise EvaluationCheckpointError(
                "checkpoint training config hash differs from EvaluationProtocol"
            )
        if payload["git_commit"] != protocol.source_training_git_commit:
            raise EvaluationCheckpointError(
                "checkpoint training git commit differs from EvaluationProtocol"
            )
        if resolved.get("seed") != protocol.source_training_seed:
            raise EvaluationCheckpointError(
                "checkpoint source training seed differs from EvaluationProtocol"
            )
        if metadata.get("run_id") != protocol.source_run_id:
            raise EvaluationCheckpointError(
                "checkpoint source run id differs from EvaluationProtocol"
            )
        if source_config.run_id != protocol.source_run_id:
            raise EvaluationCheckpointError(
                "reconstructed source RunConfig run id differs from EvaluationProtocol"
            )
    return source_config, resolved, metadata, recomputed


def _validate_runtime(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = payload["runtime_provenance"]
    source_device = str(runtime["device_type"])
    if source_device not in {"cpu", "cuda"}:
        raise EvaluationCheckpointError("checkpoint training device is invalid")
    if runtime["dtype"] != str(EVALUATION_DTYPE):
        raise EvaluationCheckpointError("checkpoint actor dtype is not torch.float32")
    source_git = str(payload["git_commit"])
    if not source_git or source_git == "unknown":
        raise EvaluationCheckpointError(
            "checkpoint training git provenance is unavailable"
        )
    return runtime


def _construct_actor(
    config: RunConfig,
    payload: Mapping[str, Any],
    device: torch.device,
) -> CAGATMAPPOActor:
    actor = CAGATMAPPOActor(config).to(device=device, dtype=EVALUATION_DTYPE)
    try:
        actor.load_state_dict(payload["model_state"]["actor"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise EvaluationCheckpointError(
            f"actor architecture/state is incompatible: {exc}"
        ) from exc
    actor.eval()
    if actor.training:
        raise EvaluationCheckpointError("actor did not enter evaluation mode")
    return actor


def _loaded_actor_record(
    *,
    actor: CAGATMAPPOActor,
    actor_config: RunConfig,
    source_config: RunConfig,
    target: Path,
    payload: Mapping[str, Any],
    source: Mapping[str, Any],
    metadata: Mapping[str, Any],
    recomputed_hash: str,
    runtime: Mapping[str, Any],
    expected_sha256: str,
    source_hash_before: str,
    source_hash_after_actor_load: str,
    device: torch.device,
) -> LoadedEvaluationActor:
    tensor_spec = MAPPOTensorSpec.from_config(actor_config)
    architecture_identity = {
        "tensor_spec": _jsonable(asdict(tensor_spec)),
        "network_config": {
            name: getattr(actor_config.training.mappo, name)
            for name in _ARCHITECTURE_FIELDS
        },
    }
    step = int(payload["training_state"]["collected_environment_transitions"])
    return LoadedEvaluationActor(
        actor=actor,
        source_run_config=source_config,
        source_checkpoint_path=target,
        source_checkpoint_sha256=source_hash_before,
        checkpoint_expected_sha256=expected_sha256,
        checkpoint_sha256_before=source_hash_before,
        checkpoint_sha256_after_actor_load=source_hash_after_actor_load,
        source_checkpoint_kind=str(payload["checkpoint_kind"]),
        source_checkpoint_step=step,
        source_run_id=str(metadata.get("run_id", source_config.run_id)),
        source_method_id=str(payload["method_id"]),
        source_training_seed=int(source["seed"]),
        source_training_config_hash=str(payload["config_hash"]),
        recomputed_source_training_config_hash=recomputed_hash,
        source_training_git_commit=str(payload["git_commit"]),
        source_training_device=str(runtime["device_type"]),
        evaluation_device=device.type,
        dtype=str(EVALUATION_DTYPE),
        actor_architecture_identity=architecture_identity,
        action_domain_identity=_action_domain_identity(actor_config),
    )


def _validate_evaluation_device(value: str) -> torch.device:
    if value not in {"cpu", "cuda"}:
        raise EvaluationCheckpointError(
            "evaluation device must be explicitly specified as 'cpu' or 'cuda'"
        )
    if value == "cuda" and not torch.cuda.is_available():
        raise EvaluationCheckpointError(
            "evaluation device 'cuda' was requested but CUDA is unavailable; "
            "silent CPU fallback is forbidden"
        )
    return torch.device(value)


def _validate_source_compatibility(
    config: RunConfig,
    source: Mapping[str, Any],
) -> None:
    if source.get("scenario_id") != config.scenario_id:
        raise EvaluationCheckpointError("checkpoint scenario_id mismatch")
    if source.get("environment") != _jsonable(asdict(config.environment)):
        raise EvaluationCheckpointError(
            "checkpoint environment/observation domain is incompatible"
        )
    if source.get("action") != _jsonable(asdict(config.action)):
        raise EvaluationCheckpointError("checkpoint action domain is incompatible")
    source_training = source.get("training")
    if not isinstance(source_training, Mapping):
        raise EvaluationCheckpointError("checkpoint training config is invalid")
    source_mappo = source_training.get("mappo")
    if not isinstance(source_mappo, Mapping):
        raise EvaluationCheckpointError("checkpoint MAPPO config is invalid")
    for name in _ARCHITECTURE_FIELDS:
        if source_mappo.get(name) != getattr(config.training.mappo, name):
            raise EvaluationCheckpointError(
                f"checkpoint actor architecture mismatch: {name}"
            )


def _action_domain_identity(config: RunConfig) -> Mapping[str, Any]:
    return {
        "uav_count": config.environment.uav_count,
        "resource_group_count": config.environment.resource_group_count,
        "sampling_order": list(config.action.sampling_order),
        "route_fixed_actions": list(config.action.route_fixed_actions),
        "tx_select_idle_action": config.action.tx_select_idle_action,
        "resource_width_options": list(config.action.resource_width_options),
        "power_levels": list(config.action.power_levels),
        "cpu_queue_idle_action": config.action.cpu_queue_idle_action,
        "cpu_frequency_levels": list(config.action.cpu_frequency_levels),
        "canonical_inactive_values": _jsonable(
            dict(config.action.canonical_inactive_values)
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "EVALUATION_DTYPE",
    "EvaluationCheckpointError",
    "LoadedEvaluationActor",
    "checkpoint_sha256",
    "load_actor_for_evaluation",
    "load_final_actor_for_evaluation",
]
