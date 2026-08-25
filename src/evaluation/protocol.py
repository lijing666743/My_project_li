"""Machine-independent protocol for Validation Gate V1."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FORMAL_PHASE = "validation"
_FORMAL_VERSION = "validation_gate_v1"
_FORMAL_SCENARIO = "small"
_FORMAL_HORIZON = 500
_FORMAL_SOURCE_RUN_ID = (
    "ca_gat_mappo__small__seed-42__cfg-e819fe0b2d47__git-f630ec5d0c2b"
)
_FORMAL_SOURCE_SEED = 42
_FORMAL_SOURCE_COMMIT = "f630ec5d0c2b7e410c347a9c6ce17cd69510ecc1"
_FORMAL_SOURCE_CONFIG_HASH = "e819fe0b2d4789318512bd0c0b33e0137a0a5a9d6928989ba632c6c3ba7b6584"
_FORMAL_SEEDS = (1042, 1043, 1044, 1045, 1046)
_FORMAL_METHODS = (
    ("ca_gat_mappo", "CA-GAT-MAPPO"),
    ("heuristic", "Greedy Heuristic"),
    ("local_only", "Local-only"),
    ("random", "Random"),
)
_FORMAL_CHECKPOINTS = (
    (
        "step_100000.pt",
        "PERIODIC_RESUME",
        100000,
        "476061dba7689452495974f8bef68438428bb0ff3142c356bafe221f1733f5a0",
    ),
    (
        "step_200000.pt",
        "PERIODIC_RESUME",
        200000,
        "b53ebbcd935a7b62a4cac3bd75c736dcd8ed923894ab9fb7a7da21d9d4c883ab",
    ),
    (
        "final.pt",
        "FINAL_COMPLETED",
        500000,
        "df8582648aa3b3dce16711fee231d082084ed428e1c1e43129acdc6f5a582f5f",
    ),
)
_PROTOCOL_FIELDS = frozenset(
    {
        "evaluation_phase",
        "evaluation_protocol_version",
        "source_run_id",
        "source_training_seed",
        "source_training_git_commit",
        "source_training_config_hash",
        "scenario_id",
        "episode_horizon_slots",
        "validation_seeds",
        "methods",
        "checkpoints",
        "inference_contract",
    }
)
_METHOD_FIELDS = frozenset({"id", "display_name"})
_CHECKPOINT_FIELDS = frozenset(
    {"filename", "checkpoint_kind", "checkpoint_step", "expected_sha256"}
)
_INFERENCE_FIELDS = frozenset(
    {
        "actor_eval",
        "torch_no_grad",
        "action_selection",
        "gru_hidden_reset_scope",
        "network_updates",
    }
)


class EvaluationProtocolError(ValueError):
    """Raised when the independent Validation protocol is not exact."""


@dataclass(frozen=True)
class EvaluationMethod:
    method_id: str
    display_name: str

    def canonical_dict(self) -> dict[str, Any]:
        return {"id": self.method_id, "display_name": self.display_name}


@dataclass(frozen=True)
class EvaluationCheckpoint:
    filename: str
    checkpoint_kind: str
    checkpoint_step: int
    expected_sha256: str

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "checkpoint_kind": self.checkpoint_kind,
            "checkpoint_step": self.checkpoint_step,
            "expected_sha256": self.expected_sha256,
        }


@dataclass(frozen=True)
class InferenceContract:
    actor_eval: bool
    torch_no_grad: bool
    action_selection: str
    gru_hidden_reset_scope: str
    network_updates: bool

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "actor_eval": self.actor_eval,
            "torch_no_grad": self.torch_no_grad,
            "action_selection": self.action_selection,
            "gru_hidden_reset_scope": self.gru_hidden_reset_scope,
            "network_updates": self.network_updates,
        }


@dataclass(frozen=True)
class EvaluationProtocol:
    evaluation_phase: str
    evaluation_protocol_version: str
    source_run_id: str
    source_training_seed: int
    source_training_git_commit: str
    source_training_config_hash: str
    scenario_id: str
    episode_horizon_slots: int
    validation_seeds: tuple[int, ...]
    methods: tuple[EvaluationMethod, ...]
    checkpoints: tuple[EvaluationCheckpoint, ...]
    inference_contract: InferenceContract

    def __post_init__(self) -> None:
        if not self.source_run_id or any(
            separator in self.source_run_id for separator in ("/", "\\")
        ):
            raise EvaluationProtocolError(
                "source_run_id must be one machine-independent path component"
            )
        if (
            isinstance(self.source_training_seed, bool)
            or not isinstance(self.source_training_seed, int)
            or self.source_training_seed < 0
        ):
            raise EvaluationProtocolError("source_training_seed is invalid")
        if not _GIT_COMMIT_PATTERN.fullmatch(self.source_training_git_commit):
            raise EvaluationProtocolError("source_training_git_commit is invalid")
        if not _SHA256_PATTERN.fullmatch(self.source_training_config_hash):
            raise EvaluationProtocolError("source_training_config_hash is invalid")
        if not self.validation_seeds or len(set(self.validation_seeds)) != len(
            self.validation_seeds
        ):
            raise EvaluationProtocolError("validation_seeds must be unique and non-empty")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.validation_seeds
        ):
            raise EvaluationProtocolError("validation_seeds are invalid")
        method_ids = tuple(item.method_id for item in self.methods)
        if not method_ids or len(set(method_ids)) != len(method_ids):
            raise EvaluationProtocolError("method ids must be unique and non-empty")
        checkpoint_names = tuple(item.filename for item in self.checkpoints)
        if not checkpoint_names or len(set(checkpoint_names)) != len(checkpoint_names):
            raise EvaluationProtocolError(
                "checkpoint filenames must be unique and non-empty"
            )
        expected_hashes = tuple(item.expected_sha256 for item in self.checkpoints)
        if len(set(expected_hashes)) != len(expected_hashes) or any(
            not _SHA256_PATTERN.fullmatch(value) for value in expected_hashes
        ):
            raise EvaluationProtocolError(
                "checkpoint expected_sha256 values must be distinct lowercase SHA-256"
            )

    @classmethod
    def from_path(cls, path: str | Path) -> "EvaluationProtocol":
        target = Path(path)
        if not target.is_file():
            raise EvaluationProtocolError(
                f"evaluation protocol is not a regular file: {target}"
            )
        try:
            mapping = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationProtocolError(
                f"evaluation protocol cannot be read: {exc}"
            ) from exc
        protocol = cls.from_mapping(mapping)
        protocol.validate_formal_v1()
        return protocol

    @classmethod
    def from_mapping(cls, value: Any) -> "EvaluationProtocol":
        mapping = _strict_mapping(value, _PROTOCOL_FIELDS, "protocol")
        methods_raw = mapping["methods"]
        checkpoints_raw = mapping["checkpoints"]
        if not isinstance(methods_raw, list):
            raise EvaluationProtocolError("protocol.methods must be a list")
        if not isinstance(checkpoints_raw, list):
            raise EvaluationProtocolError("protocol.checkpoints must be a list")
        methods = tuple(
            EvaluationMethod(
                method_id=_strict_str(
                    item_mapping["id"], f"protocol.methods[{index}].id"
                ),
                display_name=_strict_str(
                    item_mapping["display_name"],
                    f"protocol.methods[{index}].display_name",
                ),
            )
            for index, item_mapping in (
                (index, _strict_mapping(item, _METHOD_FIELDS, f"protocol.methods[{index}]"))
                for index, item in enumerate(methods_raw)
            )
        )
        checkpoints = tuple(
            EvaluationCheckpoint(
                filename=_strict_str(
                    item_mapping["filename"],
                    f"protocol.checkpoints[{index}].filename",
                ),
                checkpoint_kind=_strict_str(
                    item_mapping["checkpoint_kind"],
                    f"protocol.checkpoints[{index}].checkpoint_kind",
                ),
                checkpoint_step=_strict_int(
                    item_mapping["checkpoint_step"],
                    f"protocol.checkpoints[{index}].checkpoint_step",
                ),
                expected_sha256=_strict_str(
                    item_mapping["expected_sha256"],
                    f"protocol.checkpoints[{index}].expected_sha256",
                ),
            )
            for index, item_mapping in (
                (
                    index,
                    _strict_mapping(
                        item,
                        _CHECKPOINT_FIELDS,
                        f"protocol.checkpoints[{index}]",
                    ),
                )
                for index, item in enumerate(checkpoints_raw)
            )
        )
        inference = _strict_mapping(
            mapping["inference_contract"],
            _INFERENCE_FIELDS,
            "protocol.inference_contract",
        )
        seeds_raw = mapping["validation_seeds"]
        if not isinstance(seeds_raw, list):
            raise EvaluationProtocolError("protocol.validation_seeds must be a list")
        return cls(
            evaluation_phase=_strict_str(
                mapping["evaluation_phase"], "protocol.evaluation_phase"
            ),
            evaluation_protocol_version=_strict_str(
                mapping["evaluation_protocol_version"],
                "protocol.evaluation_protocol_version",
            ),
            source_run_id=_strict_str(mapping["source_run_id"], "protocol.source_run_id"),
            source_training_seed=_strict_int(
                mapping["source_training_seed"], "protocol.source_training_seed"
            ),
            source_training_git_commit=_strict_str(
                mapping["source_training_git_commit"],
                "protocol.source_training_git_commit",
            ),
            source_training_config_hash=_strict_str(
                mapping["source_training_config_hash"],
                "protocol.source_training_config_hash",
            ),
            scenario_id=_strict_str(mapping["scenario_id"], "protocol.scenario_id"),
            episode_horizon_slots=_strict_int(
                mapping["episode_horizon_slots"], "protocol.episode_horizon_slots"
            ),
            validation_seeds=tuple(
                _strict_int(seed, f"protocol.validation_seeds[{index}]")
                for index, seed in enumerate(seeds_raw)
            ),
            methods=methods,
            checkpoints=checkpoints,
            inference_contract=InferenceContract(
                actor_eval=_strict_bool(
                    inference["actor_eval"], "protocol.inference_contract.actor_eval"
                ),
                torch_no_grad=_strict_bool(
                    inference["torch_no_grad"],
                    "protocol.inference_contract.torch_no_grad",
                ),
                action_selection=_strict_str(
                    inference["action_selection"],
                    "protocol.inference_contract.action_selection",
                ),
                gru_hidden_reset_scope=_strict_str(
                    inference["gru_hidden_reset_scope"],
                    "protocol.inference_contract.gru_hidden_reset_scope",
                ),
                network_updates=_strict_bool(
                    inference["network_updates"],
                    "protocol.inference_contract.network_updates",
                ),
            ),
        )

    def validate_formal_v1(self) -> None:
        if self.evaluation_phase != _FORMAL_PHASE:
            raise EvaluationProtocolError("formal evaluation_phase must be validation")
        if self.evaluation_protocol_version != _FORMAL_VERSION:
            raise EvaluationProtocolError(
                "formal evaluation_protocol_version must be validation_gate_v1"
            )
        if (
            self.source_run_id != _FORMAL_SOURCE_RUN_ID
            or self.source_training_seed != _FORMAL_SOURCE_SEED
            or self.source_training_git_commit != _FORMAL_SOURCE_COMMIT
            or self.source_training_config_hash != _FORMAL_SOURCE_CONFIG_HASH
        ):
            raise EvaluationProtocolError(
                "formal source training identity differs from V1"
            )
        if self.scenario_id != _FORMAL_SCENARIO:
            raise EvaluationProtocolError("formal scenario_id must be small")
        if self.episode_horizon_slots != _FORMAL_HORIZON:
            raise EvaluationProtocolError(
                "formal episode_horizon_slots must be 500"
            )
        if self.validation_seeds != _FORMAL_SEEDS:
            raise EvaluationProtocolError("formal validation_seeds differ from V1")
        if tuple(
            (item.method_id, item.display_name) for item in self.methods
        ) != _FORMAL_METHODS:
            raise EvaluationProtocolError("formal method suite differs from V1")
        if tuple(
            (
                item.filename,
                item.checkpoint_kind,
                item.checkpoint_step,
                item.expected_sha256,
            )
            for item in self.checkpoints
        ) != _FORMAL_CHECKPOINTS:
            raise EvaluationProtocolError("formal checkpoint suite differs from V1")
        expected_inference = InferenceContract(
            actor_eval=True,
            torch_no_grad=True,
            action_selection="deterministic_masked_argmax",
            gru_hidden_reset_scope="episode",
            network_updates=False,
        )
        if self.inference_contract != expected_inference:
            raise EvaluationProtocolError("formal inference contract differs from V1")

    def checkpoint_for_filename(self, filename: str) -> EvaluationCheckpoint:
        selected = tuple(item for item in self.checkpoints if item.filename == filename)
        if len(selected) != 1:
            raise EvaluationProtocolError(
                f"checkpoint filename is not uniquely allowlisted: {filename!r}"
            )
        return selected[0]

    @property
    def method_ids(self) -> tuple[str, ...]:
        return tuple(item.method_id for item in self.methods)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "evaluation_phase": self.evaluation_phase,
            "evaluation_protocol_version": self.evaluation_protocol_version,
            "source_run_id": self.source_run_id,
            "source_training_seed": self.source_training_seed,
            "source_training_git_commit": self.source_training_git_commit,
            "source_training_config_hash": self.source_training_config_hash,
            "scenario_id": self.scenario_id,
            "episode_horizon_slots": self.episode_horizon_slots,
            "validation_seeds": list(self.validation_seeds),
            "methods": [item.canonical_dict() for item in self.methods],
            "checkpoints": [item.canonical_dict() for item in self.checkpoints],
            "inference_contract": self.inference_contract.canonical_dict(),
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def snapshot_text(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationProtocolError(f"{path} must be an object")
    keys = set(value)
    if keys != fields:
        missing = sorted(fields - keys)
        unknown = sorted(keys - fields)
        raise EvaluationProtocolError(
            f"{path} fields differ; missing={missing}, unknown={unknown}"
        )
    return value


def _strict_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationProtocolError(f"{path} must be a non-negative integer")
    return value


def _strict_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationProtocolError(f"{path} must be a boolean")
    return value


def _strict_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationProtocolError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "EvaluationCheckpoint",
    "EvaluationMethod",
    "EvaluationProtocol",
    "EvaluationProtocolError",
    "InferenceContract",
]
