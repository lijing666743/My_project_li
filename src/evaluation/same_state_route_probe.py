"""Same-state, zero-hidden route-preference probe for two frozen actors.

The probe is deliberately outside the Oracle-1 implementation.  Oracle records
are immutable replay targets and historical annotations only; no Oracle shadow
transition or Oracle classification is performed here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..artifacts import atomic_write_text, require_artifact_targets_absent
from ..config import RunConfig, load_run_config
from ..env.environment import ResetResult, StepResult, U2UMECEnvironment
from ..env.observation import ActorObservation
from ..env.randomness import derive_training_episode_seed
from ..models.ca_gat_mappo import ActorObservationTensorizer, ActorTensorBatch
from ..models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)
from ..models.ca_gat_mappo_runtime import CAGATMAPPOFactualActorRuntime
from .actor_loader import (
    LoadedEvaluationActor,
    load_final_actor_for_oracle_diagnostic,
)
from .actor_only_route_oracle import _tracked_git_identity
from .route_oracle import actor_state_digest, tensor_digest


SAME_STATE_ROUTE_PROBE_SCHEMA_VERSION = 1
SAME_STATE_ROUTE_PROBE_FILENAME = "same_state_route_probe_v1.jsonl"
EXPECTED_DECISIONS_PER_PILOT = 16
EXPECTED_TOTAL_DECISIONS = 32
_LABELS = ("4K", "60K")
_ORACLE_ANNOTATION_FIELDS = (
    "ORACLE1_BEST_ROUTE",
    "ORACLE1_BEST_ROUTE_SET",
    "ORACLE1_BEST_ROUTE_SCOPE",
    "ORACLE1_ROUTE_POLICY_MISS",
    "ORACLE_REMOTE_BETTER",
    "ORACLE_LOCAL_BETTER",
    "ORACLE_TIE",
    "ORACLE_UNRESOLVED",
)


class SameStateRouteProbeError(RuntimeError):
    """Raised when a same-state probe reliability gate fails."""


@dataclass(frozen=True)
class ProbeActorSpec:
    """One explicit FINAL actor and its immutable expected identity."""

    label: str
    config: RunConfig
    checkpoint_path: Path
    expected_checkpoint_sha256: str
    expected_actor_digest: str

    def __post_init__(self) -> None:
        if self.label not in _LABELS:
            raise SameStateRouteProbeError(f"unknown actor label {self.label!r}")
        if not isinstance(self.config, RunConfig):
            raise TypeError("actor config must be a RunConfig")
        self.config.validate()
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        _validate_sha256(self.expected_checkpoint_sha256, "checkpoint SHA-256")
        _validate_sha256(self.expected_actor_digest, "actor digest")


@dataclass(frozen=True)
class ReplayPilotSpec:
    """Pinned Oracle pilot used solely as a strict replay target manifest."""

    label: str
    config: RunConfig
    sidecar_path: Path
    expected_sidecar_sha256: str
    expected_run_id: str
    expected_config_hash: str
    expected_decision_keys: tuple[str, ...]
    max_factual_environment_transitions: int = 200

    def __post_init__(self) -> None:
        if self.label not in _LABELS:
            raise SameStateRouteProbeError(f"unknown pilot label {self.label!r}")
        if not isinstance(self.config, RunConfig):
            raise TypeError("pilot config must be a RunConfig")
        self.config.validate()
        object.__setattr__(self, "sidecar_path", Path(self.sidecar_path))
        object.__setattr__(self, "expected_decision_keys", tuple(self.expected_decision_keys))
        _validate_sha256(self.expected_sidecar_sha256, "pilot sidecar SHA-256")
        _validate_sha256(self.expected_config_hash, "pilot config hash")
        if self.config.config_hash != self.expected_config_hash:
            raise SameStateRouteProbeError(
                f"{self.label} replay config hash does not match the pinned pilot"
            )
        if not self.expected_run_id:
            raise SameStateRouteProbeError("pilot run ID cannot be empty")
        if len(self.expected_decision_keys) != EXPECTED_DECISIONS_PER_PILOT:
            raise SameStateRouteProbeError(
                f"{self.label} must pin exactly {EXPECTED_DECISIONS_PER_PILOT} decisions"
            )
        if len(set(self.expected_decision_keys)) != len(self.expected_decision_keys):
            raise SameStateRouteProbeError(f"{self.label} expected decision keys repeat")
        for key in self.expected_decision_keys:
            _validate_sha256(key, "decision key")
        cap = self.max_factual_environment_transitions
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise SameStateRouteProbeError("factual replay cap must be a positive integer")


@dataclass(frozen=True)
class SameStateRouteProbeResult:
    """Identity and accounting for one completely published probe."""

    probe_id: str
    run_directory: str
    sidecar_path: str
    state_count: int
    decision_count: int
    factual_transitions_4k: int
    factual_transitions_60k: int
    checkpoint_sha256_4k: str
    checkpoint_sha256_60k: str
    actor_digest_4k: str
    actor_digest_60k: str
    status: str = "complete"


@dataclass(frozen=True)
class _PilotArtifact:
    header: Mapping[str, Any]
    decisions: tuple[Mapping[str, Any], ...]
    sidecar_sha256: str


@dataclass(frozen=True)
class _ReplayOutput:
    state_records: tuple[Mapping[str, Any], ...]
    route_pairs: tuple[Mapping[str, Any], ...]
    factual_transitions: int


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise SameStateRouteProbeError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SameStateRouteProbeError(f"{name} must be a SHA-256 hex digest") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise SameStateRouteProbeError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_loads(line: str) -> Any:
    def reject_constant(value: str) -> None:
        raise SameStateRouteProbeError(f"JSON contains non-finite constant {value}")

    try:
        return json.loads(line, parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SameStateRouteProbeError(f"invalid JSON record: {exc}") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SameStateRouteProbeError(f"{name} must be a mapping")
    return value


def _typed_route_key(value: Any) -> str:
    return _canonical_json({"type": type(value).__name__, "value": value})


def _same_policy_identity(record: Mapping[str, Any], actor: ProbeActorSpec) -> None:
    identity = _mapping(record.get("policy_identity"), "Oracle policy identity")
    if identity.get("checkpoint_sha256") != actor.expected_checkpoint_sha256:
        raise SameStateRouteProbeError(
            f"{actor.label} pilot is bound to a different checkpoint"
        )
    if identity.get("actor_state_digest") != actor.expected_actor_digest:
        raise SameStateRouteProbeError(
            f"{actor.label} pilot is bound to a different actor digest"
        )


def load_replay_pilot(spec: ReplayPilotSpec, actor: ProbeActorSpec) -> _PilotArtifact:
    """Load and validate one immutable source sidecar without changing it."""

    if spec.label != actor.label:
        raise SameStateRouteProbeError("pilot/actor labels are cross-bound")
    before = _file_sha256(spec.sidecar_path)
    if before != spec.expected_sidecar_sha256:
        raise SameStateRouteProbeError(f"{spec.label} pilot sidecar SHA-256 mismatch")
    try:
        lines = spec.sidecar_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SameStateRouteProbeError(f"cannot read pilot sidecar: {exc}") from exc
    after = _file_sha256(spec.sidecar_path)
    if after != before:
        raise SameStateRouteProbeError("pilot sidecar changed while it was read")
    if not lines or any(not line.strip() for line in lines):
        raise SameStateRouteProbeError("pilot sidecar is empty or contains blank records")
    records = tuple(_strict_json_loads(line) for line in lines)
    if any(not isinstance(item, Mapping) for item in records):
        raise SameStateRouteProbeError("pilot sidecar records must be mappings")
    header = _mapping(records[0], "pilot schema")
    if header.get("record_type") != "schema" or header.get("schema_version") != 1:
        raise SameStateRouteProbeError("pilot must start with the Oracle-1 V1 schema")
    if header.get("run_id") != spec.expected_run_id:
        raise SameStateRouteProbeError(f"{spec.label} pilot run ID mismatch")
    if header.get("config_hash") != spec.expected_config_hash:
        raise SameStateRouteProbeError(f"{spec.label} pilot config hash mismatch")
    _same_policy_identity(header, actor)

    decisions: list[Mapping[str, Any]] = []
    branches: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    decision_keys: set[str] = set()
    branch_keys: set[tuple[str, str]] = set()
    for record in records[1:]:
        kind = record.get("record_type")
        if kind == "oracle_decision":
            key = record.get("decision_key")
            if not isinstance(key, str) or key in decision_keys:
                raise SameStateRouteProbeError("pilot contains a duplicate or invalid decision key")
            _same_policy_identity(record, actor)
            decision_keys.add(key)
            decisions.append(record)
        elif kind == "oracle_branch":
            key = record.get("decision_key")
            if not isinstance(key, str):
                raise SameStateRouteProbeError("Oracle branch has no decision key")
            branch_key = (key, _typed_route_key(record.get("branch_route")))
            if branch_key in branch_keys:
                raise SameStateRouteProbeError("pilot contains a duplicate Oracle branch")
            branch_keys.add(branch_key)
            branches[key].append(record)
        else:
            raise SameStateRouteProbeError(f"unexpected pilot record type {kind!r}")

    expected = set(spec.expected_decision_keys)
    if len(decisions) != EXPECTED_DECISIONS_PER_PILOT or decision_keys != expected:
        missing = sorted(expected - decision_keys)
        extra = sorted(decision_keys - expected)
        raise SameStateRouteProbeError(
            f"{spec.label} target set mismatch: missing={missing}, extra={extra}"
        )
    if set(branches) != decision_keys:
        raise SameStateRouteProbeError("Oracle branch/decision join is incomplete")
    for decision in decisions:
        key = str(decision["decision_key"])
        count = decision.get("branch_count")
        if isinstance(count, bool) or not isinstance(count, int) or count != len(branches[key]):
            raise SameStateRouteProbeError("Oracle branch_count does not match joined branches")
        required = (
            "episode_id",
            "global_environment_step",
            "route_slot",
            "source_uav",
            "task_id",
            "FACTUAL_ROUTE",
            "initial_state_fingerprint",
            "initial_rng_fingerprint",
            "initial_exogenous_fingerprint",
            "initial_hidden_digest",
        )
        if any(name not in decision for name in required):
            raise SameStateRouteProbeError("Oracle decision lacks replay coordinates")
    ordered = tuple(sorted(decisions, key=lambda item: spec.expected_decision_keys.index(str(item["decision_key"]))))
    return _PilotArtifact(header=header, decisions=ordered, sidecar_sha256=before)


def _pilot_execution_device(label: str, artifact: _PilotArtifact) -> str:
    identity = _mapping(
        artifact.header.get("oracle_diagnostic_identity"),
        f"{label} pilot oracle_diagnostic_identity",
    )
    device = identity.get("execution_device")
    if device not in {"cpu", "cuda"}:
        raise SameStateRouteProbeError(
            f"{label} pilot execution_device must be explicitly cpu or cuda"
        )
    return str(device)


def _torch_rng_snapshot() -> tuple[Tensor, tuple[Tensor, ...]]:
    cpu = torch.get_rng_state().clone()
    cuda = tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return cpu, cuda


def _torch_rng_equal(
    left: tuple[Tensor, tuple[Tensor, ...]],
    right: tuple[Tensor, tuple[Tensor, ...]],
) -> bool:
    return torch.equal(left[0], right[0]) and len(left[1]) == len(right[1]) and all(
        torch.equal(a, b) for a, b in zip(left[1], right[1])
    )


def _restore_torch_rng(snapshot: tuple[Tensor, tuple[Tensor, ...]]) -> None:
    torch.set_rng_state(snapshot[0])
    if snapshot[1]:
        torch.cuda.set_rng_state_all(list(snapshot[1]))


def _load_actor_rng_isolated(spec: ProbeActorSpec, device: str) -> LoadedEvaluationActor:
    before = _torch_rng_snapshot()
    try:
        loaded = load_final_actor_for_oracle_diagnostic(
            spec.config,
            spec.checkpoint_path,
            execution_device=device,
            expected_checkpoint_sha256=spec.expected_checkpoint_sha256,
            expected_actor_digest=spec.expected_actor_digest,
        )
    except Exception:
        after = _torch_rng_snapshot()
        if not _torch_rng_equal(before, after):
            _restore_torch_rng(before)
        raise
    after = _torch_rng_snapshot()
    if not _torch_rng_equal(before, after):
        _restore_torch_rng(before)
        raise SameStateRouteProbeError("actor loading changed global Torch RNG state")
    return loaded


def _tensor_snapshot(value: Tensor) -> Mapping[str, Any]:
    tensor = value.detach().cpu().contiguous()
    digest = tensor_digest(tensor)
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "nbytes": tensor.numpy().nbytes,
        "sha256": digest,
        "values": tensor.tolist(),
    }


def actor_batch_snapshot(batch: ActorTensorBatch) -> Mapping[str, Any]:
    """Return the complete JSON-safe actor input used by both policies."""

    return {
        "self_features": _tensor_snapshot(batch.self_features),
        "neighbor_public_features": _tensor_snapshot(batch.neighbor_public_features),
        "edge_features": _tensor_snapshot(batch.edge_features),
        "neighbor_mask": _tensor_snapshot(batch.neighbor_mask),
        "action_masks": {name: _tensor_snapshot(value) for name, value in batch.action_masks.items()},
        "action_indices": {name: _tensor_snapshot(value) for name, value in batch.action_indices.items()},
        "episode_starts": _tensor_snapshot(batch.episode_starts),
    }


def _actor_batch_digest(batch: ActorTensorBatch) -> str:
    snapshot = actor_batch_snapshot(batch)
    compact = {
        name: (
            {key: value["sha256"] for key, value in item.items()}
            if name in {"action_masks", "action_indices"}
            else item["sha256"]
        )
        for name, item in snapshot.items()
    }
    return _sha256_json(compact)


def _route_policy_record(
    label: str,
    loaded: LoadedEvaluationActor,
    output: SequentialActionDistributionOutput,
    observation: ActorObservation,
    source_uav: int,
) -> Mapping[str, Any]:
    logits_tensor = output.raw_logits["route"][0, 0, source_uav].detach().cpu()
    probabilities_tensor = output.probabilities["route"][0, 0, source_uav].detach().cpu()
    mask_tensor = output.action_masks["route"][0, 0, source_uav].detach().cpu()
    if not torch.isfinite(logits_tensor).all() or not torch.isfinite(probabilities_tensor).all():
        raise SameStateRouteProbeError("route logits/probabilities contain NaN or Inf")
    domain = tuple(observation.action_masks.route_domain)
    if len(domain) != logits_tensor.numel() or mask_tensor.numel() != len(domain):
        raise SameStateRouteProbeError("route domain and actor output dimensions differ")
    logits = [float(value) for value in logits_tensor.tolist()]
    probabilities = [float(value) for value in probabilities_tensor.tolist()]
    legal = [bool(value) for value in mask_tensor.tolist()]
    if any((not is_legal) and probability != 0.0 for is_legal, probability in zip(legal, probabilities)):
        raise SameStateRouteProbeError("illegal route received non-zero probability")
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-7):
        raise SameStateRouteProbeError("route probabilities do not sum to one")
    selected_index = int(output.action_indices["route"][0, 0, source_uav].item())
    selected_route = domain[selected_index]
    selected_logit = logits[selected_index]
    ties = [domain[index] for index, (value, allowed) in enumerate(zip(logits, legal)) if allowed and value == selected_logit]
    if selected_index != min(index for index, (value, allowed) in enumerate(zip(logits, legal)) if allowed and value == selected_logit):
        raise SameStateRouteProbeError("masked argmax did not use the lowest-index exact tie")
    remote_indices = [
        index
        for index, (route, allowed) in enumerate(zip(domain, legal))
        if allowed and isinstance(route, int) and not isinstance(route, bool)
    ]
    local_index = domain.index("local") if "local" in domain else None
    defer_index = domain.index("defer") if "defer" in domain else None
    route_active = bool(observation.action_masks.route_branch_active)
    remote_by_destination = [
        {"destination_uav": int(domain[index]), "probability": probabilities[index]}
        for index in remote_indices
    ]
    best_remote = max(remote_indices, key=lambda index: (logits[index], -index)) if remote_indices else None
    remote_logits = torch.tensor([logits[index] for index in remote_indices], dtype=torch.float64)
    remote_margin = None
    remote_lse_margin = None
    if route_active and remote_indices and local_index is not None and legal[local_index]:
        remote_margin = max(logits[index] for index in remote_indices) - logits[local_index]
        remote_lse_margin = float(torch.logsumexp(remote_logits, dim=0).item()) - logits[local_index]
    entropy = -math.fsum(value * math.log(value) for value in probabilities if value > 0.0)
    return {
        "actor_label": label,
        "checkpoint_sha256": loaded.source_checkpoint_sha256,
        "actor_digest": loaded.actor_state_digest,
        "route_domain": list(domain),
        "route_mask": legal,
        "raw_logits": logits,
        "masked_probabilities": probabilities,
        "masked_argmax_index": selected_index,
        "masked_argmax_route": selected_route,
        "exact_argmax_ties": ties,
        "route_branch_active": route_active,
        "p_local": None if local_index is None else probabilities[local_index],
        "p_defer": None if defer_index is None else probabilities[defer_index],
        "p_remote_total": math.fsum(item["probability"] for item in remote_by_destination),
        "p_remote_by_destination": remote_by_destination,
        "best_legal_remote_id": None if best_remote is None else int(domain[best_remote]),
        "route_entropy": entropy,
        "max_legal_remote_logit_minus_local": remote_margin,
        "logsumexp_legal_remote_logits_minus_local": remote_lse_margin,
    }


def _paired_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> Mapping[str, Any]:
    if left["route_domain"] != right["route_domain"] or left["route_mask"] != right["route_mask"]:
        raise SameStateRouteProbeError("4K and 60K route domains/masks differ")

    def delta(name: str) -> float | None:
        a, b = left[name], right[name]
        return None if a is None or b is None else float(b) - float(a)

    left_remote = {item["destination_uav"]: item["probability"] for item in left["p_remote_by_destination"]}
    right_remote = {item["destination_uav"]: item["probability"] for item in right["p_remote_by_destination"]}
    if set(left_remote) != set(right_remote):
        raise SameStateRouteProbeError("4K and 60K legal Remote sets differ")
    return {
        "direction": "60K_minus_4K",
        "p_local": delta("p_local"),
        "p_defer": delta("p_defer"),
        "p_remote_total": delta("p_remote_total"),
        "route_entropy": delta("route_entropy"),
        "max_legal_remote_logit_minus_local": delta("max_legal_remote_logit_minus_local"),
        "logsumexp_legal_remote_logits_minus_local": delta("logsumexp_legal_remote_logits_minus_local"),
        "p_remote_by_destination": [
            {"destination_uav": destination, "delta": right_remote[destination] - left_remote[destination]}
            for destination in sorted(left_remote)
        ],
        "argmax_transition": [left["masked_argmax_route"], right["masked_argmax_route"]],
    }


def compare_zero_hidden_route_preferences(
    *,
    loaded_4k: LoadedEvaluationActor,
    loaded_60k: LoadedEvaluationActor,
    config: RunConfig,
    actor_batch: ActorTensorBatch,
    action_masks: SequentialActionMaskBatch,
    observations: Sequence[ActorObservation],
    source_uavs: Sequence[int],
    factual_policy_generator: torch.Generator | None = None,
    environment: U2UMECEnvironment | None = None,
    factual_hidden: Tensor | None = None,
) -> Mapping[int, Mapping[str, Any]]:
    """Run two deterministic actors on one byte-identical input with zero hidden."""

    if loaded_4k.actor is loaded_60k.actor:
        raise SameStateRouteProbeError("4K and 60K resolve to the same actor object")
    if loaded_4k.source_checkpoint_path.resolve() == loaded_60k.source_checkpoint_path.resolve():
        raise SameStateRouteProbeError("4K and 60K resolve to the same checkpoint path")
    if loaded_4k.source_checkpoint_sha256 == loaded_60k.source_checkpoint_sha256:
        raise SameStateRouteProbeError("4K and 60K checkpoint SHA-256 values are identical")
    if loaded_4k.actor_state_digest == loaded_60k.actor_state_digest:
        raise SameStateRouteProbeError("4K and 60K actor digests are identical")
    if loaded_4k.actor_architecture_identity != loaded_60k.actor_architecture_identity:
        raise SameStateRouteProbeError("4K and 60K actor architectures differ")
    if loaded_4k.action_domain_identity != loaded_60k.action_domain_identity:
        raise SameStateRouteProbeError("4K and 60K action domains differ")
    copied_sources = tuple(int(item) for item in source_uavs)
    if not copied_sources or len(set(copied_sources)) != len(copied_sources):
        raise SameStateRouteProbeError("source UAVs must be non-empty and unique")
    copied_observations = tuple(observations)
    if any(source < 0 or source >= len(copied_observations) for source in copied_sources):
        raise SameStateRouteProbeError("source UAV lies outside the observation batch")

    before_torch = _torch_rng_snapshot()
    before_policy = None if factual_policy_generator is None else factual_policy_generator.get_state().clone()
    before_environment = None if environment is None else (environment.state_fingerprint(), environment.rng_fingerprint())
    before_hidden = None if factual_hidden is None else tensor_digest(factual_hidden)
    before_input = _actor_batch_digest(actor_batch)
    before_actor_4k = actor_state_digest(loaded_4k.actor)
    before_actor_60k = actor_state_digest(loaded_60k.actor)
    batch_shape = actor_batch.validate(loaded_4k.actor.spec)
    if batch_shape[0:2] != (1, 1):
        raise SameStateRouteProbeError("probe requires one [batch,time] state")

    zero_4k = loaded_4k.actor.initial_hidden(1, device=actor_batch.self_features.device, dtype=actor_batch.self_features.dtype)
    zero_60k = loaded_60k.actor.initial_hidden(1, device=actor_batch.self_features.device, dtype=actor_batch.self_features.dtype)
    if torch.count_nonzero(zero_4k).item() or torch.count_nonzero(zero_60k).item():
        raise SameStateRouteProbeError("probe hidden input is not all-zero")
    distribution_4k = CAGATMAPPOActionDistribution(loaded_4k.actor, config)
    distribution_60k = CAGATMAPPOActionDistribution(loaded_60k.actor, config)
    try:
        with torch.inference_mode():
            output_4k = distribution_4k.deterministic_actions(actor_batch, action_masks, zero_4k)
            output_60k = distribution_60k.deterministic_actions(actor_batch, action_masks, zero_60k)
    except Exception as exc:
        raise SameStateRouteProbeError(f"paired actor inference failed: {exc}") from exc

    if not _torch_rng_equal(before_torch, _torch_rng_snapshot()):
        raise SameStateRouteProbeError("paired inference changed global Torch CPU/CUDA RNG")
    if before_policy is not None and not torch.equal(before_policy, factual_policy_generator.get_state()):
        raise SameStateRouteProbeError("paired inference changed factual policy generator RNG")
    if before_environment is not None and before_environment != (environment.state_fingerprint(), environment.rng_fingerprint()):
        raise SameStateRouteProbeError("paired inference changed environment state or RNG")
    if before_hidden is not None and before_hidden != tensor_digest(factual_hidden):
        raise SameStateRouteProbeError("paired inference changed factual hidden state")
    if before_input != _actor_batch_digest(actor_batch):
        raise SameStateRouteProbeError("paired inference changed actor input tensors")
    if before_actor_4k != actor_state_digest(loaded_4k.actor) or before_actor_60k != actor_state_digest(loaded_60k.actor):
        raise SameStateRouteProbeError("paired inference changed actor weights")

    result: dict[int, Mapping[str, Any]] = {}
    for source in copied_sources:
        left = _route_policy_record("4K", loaded_4k, output_4k, copied_observations[source], source)
        right = _route_policy_record("60K", loaded_60k, output_60k, copied_observations[source], source)
        result[source] = {
            "hidden_mode": "zero_each_state",
            "outputs": {"4K": left, "60K": right},
            "delta": _paired_delta(left, right),
        }
    return result


def _focal_task_id(observation: ActorObservation) -> int | None:
    queue = observation.private_queues.unbound
    if not bool(queue.head_valid_mask):
        return None
    if queue.head_task_id is None:
        raise SameStateRouteProbeError("valid unbound queue head has no task ID")
    return int(queue.head_task_id)


def _validate_reset(reset: ResetResult) -> tuple[ActorObservation, ...]:
    if not isinstance(reset, ResetResult) or not reset.observations:
        raise SameStateRouteProbeError("environment reset did not return observations")
    observations = tuple(reset.observations)
    if any(int(item.slot) != 0 for item in observations):
        raise SameStateRouteProbeError("replay episode did not reset at slot zero")
    return observations


def _validate_step(slot: int, step: StepResult) -> bool:
    if not isinstance(step, StepResult) or step.info.get("slot") != slot:
        raise SameStateRouteProbeError("environment step contract mismatch")
    boundary = bool(step.terminated or step.truncated)
    if boundary != (step.observations is None):
        raise SameStateRouteProbeError("environment boundary observations are inconsistent")
    return boundary


class SameStateRouteProbeRunner:
    """Strict two-pilot replay runner with no training or Oracle lifecycle."""

    def __init__(
        self,
        actor_4k: ProbeActorSpec,
        actor_60k: ProbeActorSpec,
        pilot_4k: ReplayPilotSpec,
        pilot_60k: ReplayPilotSpec,
        *,
        execution_device: str,
        output_root: str | Path,
        environment_factory: Callable[[RunConfig], U2UMECEnvironment] | None = None,
    ) -> None:
        if (actor_4k.label, actor_60k.label, pilot_4k.label, pilot_60k.label) != ("4K", "60K", "4K", "60K"):
            raise SameStateRouteProbeError("runner actor/pilot labels are not 4K/60K bound")
        if execution_device not in {"cpu", "cuda"}:
            raise SameStateRouteProbeError("execution_device must be explicitly cpu or cuda")
        self.actor_specs = {"4K": actor_4k, "60K": actor_60k}
        self.pilot_specs = {"4K": pilot_4k, "60K": pilot_60k}
        self.execution_device = execution_device
        self.output_root = Path(output_root)
        self.environment_factory = environment_factory or U2UMECEnvironment
        self.pilots = {
            "4K": load_replay_pilot(pilot_4k, actor_4k),
            "60K": load_replay_pilot(pilot_60k, actor_60k),
        }
        pilot_devices = {
            label: _pilot_execution_device(label, self.pilots[label])
            for label in _LABELS
        }
        if not (
            pilot_devices["4K"]
            == pilot_devices["60K"]
            == self.execution_device
        ):
            raise SameStateRouteProbeError(
                "pilot/replay execution_device mismatch: "
                f"4K={pilot_devices['4K']}, "
                f"60K={pilot_devices['60K']}, "
                f"runner={self.execution_device}"
            )
        if execution_device == "cuda" and not torch.cuda.is_available():
            raise SameStateRouteProbeError("CUDA execution requested but unavailable")
        self.loaded_actors = {
            "4K": _load_actor_rng_isolated(actor_4k, execution_device),
            "60K": _load_actor_rng_isolated(actor_60k, execution_device),
        }
        self._validate_actor_pair()
        git_identity = _tracked_git_identity()
        self.identity = {
            "schema_version": SAME_STATE_ROUTE_PROBE_SCHEMA_VERSION,
            "current_git_commit": git_identity.commit,
            "tracked_worktree_diff_sha256": git_identity.tracked_diff_sha256,
            "execution_device": execution_device,
            "actors": {
                label: {
                    "checkpoint_sha256": self.loaded_actors[label].source_checkpoint_sha256,
                    "actor_digest": self.loaded_actors[label].actor_state_digest,
                }
                for label in _LABELS
            },
            "pilots": {
                label: {
                    "run_id": self.pilot_specs[label].expected_run_id,
                    "sidecar_sha256": self.pilots[label].sidecar_sha256,
                    "decision_keys": list(self.pilot_specs[label].expected_decision_keys),
                }
                for label in _LABELS
            },
            "hidden_mode": "zero_each_state",
        }
        self.probe_id = f"same-state-zero-hidden-v1__{_sha256_json(self.identity)[:24]}"
        self.run_directory = self.output_root / self.probe_id
        self.sidecar_path = self.run_directory / SAME_STATE_ROUTE_PROBE_FILENAME
        self._has_run = False

    def _validate_actor_pair(self) -> None:
        left, right = self.loaded_actors["4K"], self.loaded_actors["60K"]
        if left.actor is right.actor:
            raise SameStateRouteProbeError("4K and 60K loaded the same actor object")
        if left.source_checkpoint_path.resolve() == right.source_checkpoint_path.resolve():
            raise SameStateRouteProbeError("4K and 60K loaded the same checkpoint path")
        if left.source_checkpoint_sha256 == right.source_checkpoint_sha256:
            raise SameStateRouteProbeError("4K and 60K checkpoint SHA-256 values are identical")
        if left.actor_state_digest == right.actor_state_digest:
            raise SameStateRouteProbeError("4K and 60K actor digests are identical")
        if left.actor_architecture_identity != right.actor_architecture_identity:
            raise SameStateRouteProbeError("4K and 60K architectures are incompatible")
        if left.action_domain_identity != right.action_domain_identity:
            raise SameStateRouteProbeError("4K and 60K action domains are incompatible")

    def run(self) -> SameStateRouteProbeResult:
        if self._has_run:
            raise SameStateRouteProbeError("same-state runner instances are single-use")
        self._has_run = True
        require_artifact_targets_absent((self.run_directory,), group_name="same-state route probe")
        state_records: list[Mapping[str, Any]] = []
        route_pairs: list[Mapping[str, Any]] = []
        transition_counts: dict[str, int] = {}
        for label in _LABELS:
            replay = self._replay_one(label)
            state_records.extend(replay.state_records)
            route_pairs.extend(replay.route_pairs)
            transition_counts[label] = replay.factual_transitions
        if len(route_pairs) != EXPECTED_TOTAL_DECISIONS:
            raise SameStateRouteProbeError("partial decision set cannot be published")
        decision_keys = [str(item["decision_key"]) for item in route_pairs]
        if len(set(decision_keys)) != EXPECTED_TOTAL_DECISIONS:
            raise SameStateRouteProbeError("paired decision keys are not globally unique")
        summary = _summary_record(state_records, route_pairs, transition_counts)
        records: list[Mapping[str, Any]] = [self._schema_record(), *state_records, *route_pairs, summary]
        _validate_probe_records(records)
        content = "\n".join(_canonical_json(item) for item in records) + "\n"
        atomic_write_text(self.sidecar_path, content)
        return SameStateRouteProbeResult(
            probe_id=self.probe_id,
            run_directory=str(self.run_directory),
            sidecar_path=str(self.sidecar_path),
            state_count=len(state_records),
            decision_count=len(route_pairs),
            factual_transitions_4k=transition_counts["4K"],
            factual_transitions_60k=transition_counts["60K"],
            checkpoint_sha256_4k=self.loaded_actors["4K"].source_checkpoint_sha256,
            checkpoint_sha256_60k=self.loaded_actors["60K"].source_checkpoint_sha256,
            actor_digest_4k=self.loaded_actors["4K"].actor_state_digest,
            actor_digest_60k=self.loaded_actors["60K"].actor_state_digest,
        )

    def _schema_record(self) -> Mapping[str, Any]:
        return {
            "record_type": "schema",
            "schema_version": SAME_STATE_ROUTE_PROBE_SCHEMA_VERSION,
            "probe_id": self.probe_id,
            "sidecar": "same_state_route_probe_v1",
            "semantics": "SAME_OBSERVATION_ZERO_INPUT_HIDDEN_ROUTE_PREFERENCE",
            "identity": self.identity,
            "actors": {
                label: {
                    "checkpoint_path": str(self.loaded_actors[label].source_checkpoint_path),
                    "checkpoint_sha256": self.loaded_actors[label].source_checkpoint_sha256,
                    "actor_digest": self.loaded_actors[label].actor_state_digest,
                    "source_training_config_hash": self.loaded_actors[label].source_training_config_hash,
                    "architecture": self.loaded_actors[label].actor_architecture_identity,
                    "action_domain": self.loaded_actors[label].action_domain_identity,
                }
                for label in _LABELS
            },
            "pilots": {
                label: {
                    "source_sidecar_path": str(self.pilot_specs[label].sidecar_path.resolve()),
                    "source_sidecar_sha256": self.pilots[label].sidecar_sha256,
                    "source_run_id": self.pilot_specs[label].expected_run_id,
                    "source_config_hash": self.pilot_specs[label].expected_config_hash,
                    "decision_count": EXPECTED_DECISIONS_PER_PILOT,
                    "decision_keys": list(self.pilot_specs[label].expected_decision_keys),
                    "max_factual_environment_transitions": self.pilot_specs[label].max_factual_environment_transitions,
                }
                for label in _LABELS
            },
            "zero_hidden_contract": {
                "hidden_in": "fresh_all_zero_per_unique_state_and_actor",
                "episode_start": "real_slot_boundary",
                "hidden_out": "computed_then_discarded",
                "observation_history": "preserved",
                "interpretation": "full_actor_preference_under_zero_input_hidden_only",
            },
            "oracle_annotation_contract": {
                "source": "immutable_original_pilot_record",
                "scope": "source_policy_only",
                "reclassification": "forbidden",
                "shadow_transitions": 0,
            },
            "rng_isolation": [
                "factual_policy_generator",
                "torch_cpu",
                "torch_cuda_all_devices",
                "environment_rng",
            ],
        }

    def _replay_one(self, label: str) -> _ReplayOutput:
        pilot_spec = self.pilot_specs[label]
        artifact = self.pilots[label]
        source_actor = self.loaded_actors[label]
        targets_by_position: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
        for decision in artifact.decisions:
            coordinate = (int(decision["episode_id"]), int(decision["global_environment_step"]))
            targets_by_position[coordinate].append(decision)
        seen: set[str] = set()
        state_records: list[Mapping[str, Any]] = []
        route_pairs: list[Mapping[str, Any]] = []
        transitions = 0
        episode_index = 0
        runtime = CAGATMAPPOFactualActorRuntime(
            source_actor.actor,
            pilot_spec.config,
            device=self.execution_device,
            dtype=torch.float32,
        )
        while len(seen) < EXPECTED_DECISIONS_PER_PILOT:
            episode_config = replace(
                pilot_spec.config,
                seed=derive_training_episode_seed(pilot_spec.config.seed, episode_index),
            )
            episode_config.validate()
            environment = self.environment_factory(episode_config)
            if not isinstance(environment, U2UMECEnvironment):
                raise SameStateRouteProbeError("environment_factory returned an incompatible environment")
            observations = _validate_reset(environment.reset())
            hidden = source_actor.actor.initial_hidden(
                1, device=self.execution_device, dtype=torch.float32
            )
            while True:
                slot = int(observations[0].slot)
                if any(int(item.slot) != slot for item in observations):
                    raise SameStateRouteProbeError("replay observations are not slot-aligned")
                hidden_in = hidden.detach()
                factual_step = runtime.sample_step(observations, hidden_in, episode_start=slot == 0)
                proposals = tuple(factual_step.action_output.proposals[0][0])
                hidden_out = factual_step.action_output.hidden_out.detach()
                targets = targets_by_position.get((episode_index, transitions), ())
                if targets:
                    state_record, pairs = self._probe_target_state(
                        label=label,
                        environment=environment,
                        observations=observations,
                        factual_proposals=proposals,
                        factual_hidden=hidden_out,
                        factual_runtime=runtime,
                        actor_batch=factual_step.actor_batch,
                        action_masks=factual_step.action_masks,
                        targets=targets,
                    )
                    state_records.append(state_record)
                    route_pairs.extend(pairs)
                    seen.update(str(item["decision_key"]) for item in targets)
                    if len(seen) == EXPECTED_DECISIONS_PER_PILOT:
                        expected = set(pilot_spec.expected_decision_keys)
                        if seen != expected:
                            raise SameStateRouteProbeError("replay completed the wrong target set")
                        return _ReplayOutput(tuple(state_records), tuple(route_pairs), transitions)
                if transitions >= pilot_spec.max_factual_environment_transitions:
                    raise SameStateRouteProbeError(
                        f"{label} replay budget exhausted before every target was joined"
                    )
                step = environment.step(proposals)
                boundary = _validate_step(slot, step)
                transitions += 1
                hidden = hidden_out
                if boundary:
                    episode_index += 1
                    break
                assert step.observations is not None
                observations = tuple(step.observations)
        raise SameStateRouteProbeError("replay exited without a complete target set")

    def _probe_target_state(
        self,
        *,
        label: str,
        environment: U2UMECEnvironment,
        observations: tuple[ActorObservation, ...],
        factual_proposals: Sequence[Any],
        factual_hidden: Tensor,
        factual_runtime: CAGATMAPPOFactualActorRuntime,
        actor_batch: ActorTensorBatch,
        action_masks: SequentialActionMaskBatch,
        targets: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
        state = environment.state_fingerprint()
        rng = environment.rng_fingerprint()
        exogenous = environment.exogenous_fingerprint()
        hidden = tensor_digest(factual_hidden)
        expected_common = {
            "initial_state_fingerprint": state,
            "initial_rng_fingerprint": rng,
            "initial_exogenous_fingerprint": exogenous,
            "initial_hidden_digest": hidden,
        }
        source_uavs: list[int] = []
        for target in targets:
            for name, actual in expected_common.items():
                if target.get(name) != actual:
                    raise SameStateRouteProbeError(
                        f"{label} replay fingerprint mismatch for {name}"
                    )
            if int(target["route_slot"]) != int(environment.slot):
                raise SameStateRouteProbeError("replay route slot mismatch")
            source = int(target["source_uav"])
            if source in source_uavs:
                raise SameStateRouteProbeError("partial target state repeats a source UAV")
            source_uavs.append(source)
            observation = observations[source]
            if not observation.action_masks.route_branch_active:
                raise SameStateRouteProbeError("target replay route branch is inactive")
            if _focal_task_id(observation) != int(target["task_id"]):
                raise SameStateRouteProbeError("target replay focal task mismatch")
            if factual_proposals[source].route != target["FACTUAL_ROUTE"]:
                raise SameStateRouteProbeError("target replay factual route mismatch")

        paired = compare_zero_hidden_route_preferences(
            loaded_4k=self.loaded_actors["4K"],
            loaded_60k=self.loaded_actors["60K"],
            config=self.actor_specs["4K"].config,
            actor_batch=actor_batch,
            action_masks=action_masks,
            observations=observations,
            source_uavs=source_uavs,
            factual_policy_generator=factual_runtime.policy_generator,
            environment=environment,
            factual_hidden=factual_hidden,
        )
        state_id = _sha256_json(
            {
                "pilot": label,
                "episode_id": targets[0]["episode_id"],
                "global_environment_step": targets[0]["global_environment_step"],
                "state_fingerprint": state,
            }
        )
        snapshot = actor_batch_snapshot(actor_batch)
        state_record = {
            "record_type": "state",
            "schema_version": SAME_STATE_ROUTE_PROBE_SCHEMA_VERSION,
            "state_id": state_id,
            "source_pilot": label,
            "episode_id": int(targets[0]["episode_id"]),
            "global_environment_step": int(targets[0]["global_environment_step"]),
            "route_slot": int(environment.slot),
            "decision_keys": [str(item["decision_key"]) for item in targets],
            "actor_input": snapshot,
            "actor_input_digest": _actor_batch_digest(actor_batch),
            "candidate_neighbor_masks": [item.candidate_neighbor_mask.tolist() for item in observations],
            "route_domains": [list(item.action_masks.route_domain) for item in observations],
            "route_masks": [item.action_masks.route_mask.tolist() for item in observations],
            "initial_state_fingerprint": state,
            "initial_rng_fingerprint": rng,
            "initial_exogenous_fingerprint": exogenous,
            "factual_hidden_out_digest": hidden,
            "isolation_checks": {
                "paired_input_identical": True,
                "factual_policy_rng_unchanged": True,
                "torch_cpu_rng_unchanged": True,
                "torch_cuda_rng_unchanged": True,
                "environment_state_rng_unchanged": True,
                "factual_hidden_unchanged": True,
                "actor_weights_unchanged": True,
            },
        }
        route_pairs = []
        for target in targets:
            source = int(target["source_uav"])
            annotation = {name: target.get(name) for name in _ORACLE_ANNOTATION_FIELDS}
            annotation["source_policy_identity"] = target.get("policy_identity")
            route_pairs.append(
                {
                    "record_type": "route_pair",
                    "schema_version": SAME_STATE_ROUTE_PROBE_SCHEMA_VERSION,
                    "state_id": state_id,
                    "decision_key": str(target["decision_key"]),
                    "source_pilot": label,
                    "episode_id": int(target["episode_id"]),
                    "global_environment_step": int(target["global_environment_step"]),
                    "source_uav": source,
                    "task_id": int(target["task_id"]),
                    "factual_route": target["FACTUAL_ROUTE"],
                    "historical_oracle_annotation": annotation,
                    **paired[source],
                }
            )
        return state_record, tuple(route_pairs)


def _summary_record(
    states: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    transitions: Mapping[str, int],
) -> Mapping[str, Any]:
    by_source = Counter(str(item["source_pilot"]) for item in pairs)
    if by_source != Counter({"4K": 16, "60K": 16}):
        raise SameStateRouteProbeError("summary source counts are incomplete")

    def mean_delta(name: str, source: str | None = None) -> float | None:
        values = [
            float(item["delta"][name])
            for item in pairs
            if (source is None or item["source_pilot"] == source)
            and item["delta"].get(name) is not None
        ]
        return None if not values else math.fsum(values) / len(values)

    transitions_table = Counter(
        _canonical_json(item["delta"]["argmax_transition"]) for item in pairs
    )
    return {
        "record_type": "summary",
        "schema_version": SAME_STATE_ROUTE_PROBE_SCHEMA_VERSION,
        "status": "complete",
        "state_count": len(states),
        "decision_count": len(pairs),
        "decision_count_by_source_pilot": dict(sorted(by_source.items())),
        "unique_task_count": len({(item["source_pilot"], item["task_id"]) for item in pairs}),
        "factual_environment_transitions": dict(transitions),
        "oracle_shadow_transitions": 0,
        "mean_delta_60k_minus_4k": {
            source: {
                "p_remote_total": mean_delta("p_remote_total", None if source == "equal_weight_all" else source),
                "max_legal_remote_logit_minus_local": mean_delta(
                    "max_legal_remote_logit_minus_local",
                    None if source == "equal_weight_all" else source,
                ),
                "logsumexp_legal_remote_logits_minus_local": mean_delta(
                    "logsumexp_legal_remote_logits_minus_local",
                    None if source == "equal_weight_all" else source,
                ),
            }
            for source in ("4K", "60K", "equal_weight_all")
        },
        "argmax_transition_counts": dict(sorted(transitions_table.items())),
        "failure_closed": True,
        "interpretation_boundary": (
            "paired full-actor route preference under fresh zero hidden input; "
            "not normal recurrent-policy performance or module/root-cause attribution"
        ),
    }


def _validate_probe_records(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(records) < 3 or records[0].get("record_type") != "schema":
        raise SameStateRouteProbeError("probe sidecar lacks a leading schema")
    if records[-1].get("record_type") != "summary" or records[-1].get("status") != "complete":
        raise SameStateRouteProbeError("probe sidecar lacks a complete final summary")
    schema = records[0]
    schema_actors = _mapping(schema.get("actors"), "schema actors")
    schema_pilots = _mapping(schema.get("pilots"), "schema pilots")
    if tuple(schema_actors) != _LABELS or tuple(schema_pilots) != _LABELS:
        raise SameStateRouteProbeError("schema identities are not ordered 4K/60K bindings")
    actor_identities = {
        label: _mapping(schema_actors[label], f"{label} schema actor")
        for label in _LABELS
    }
    if (
        actor_identities["4K"].get("checkpoint_sha256")
        == actor_identities["60K"].get("checkpoint_sha256")
        or actor_identities["4K"].get("actor_digest")
        == actor_identities["60K"].get("actor_digest")
    ):
        raise SameStateRouteProbeError("schema binds 4K and 60K to the same actor identity")
    expected_decisions: set[str] = set()
    expected_by_pilot: dict[str, set[str]] = {}
    for label in _LABELS:
        actor_identity = actor_identities[label]
        _validate_sha256(actor_identity.get("checkpoint_sha256"), "schema checkpoint SHA-256")
        _validate_sha256(actor_identity.get("actor_digest"), "schema actor digest")
        pilot_identity = _mapping(schema_pilots[label], f"{label} schema pilot")
        keys = pilot_identity.get("decision_keys")
        if not isinstance(keys, list) or len(keys) != EXPECTED_DECISIONS_PER_PILOT:
            raise SameStateRouteProbeError("schema pilot target count is not exactly 16")
        if len(set(keys)) != len(keys):
            raise SameStateRouteProbeError("schema pilot decision keys repeat")
        expected_by_pilot[label] = {str(key) for key in keys}
        expected_decisions.update(str(key) for key in keys)
    if len(expected_decisions) != EXPECTED_TOTAL_DECISIONS:
        raise SameStateRouteProbeError("schema pilot target sets overlap")

    states = [item for item in records[1:-1] if item.get("record_type") == "state"]
    pairs = [item for item in records[1:-1] if item.get("record_type") == "route_pair"]
    if len(states) + len(pairs) != len(records) - 2:
        raise SameStateRouteProbeError("probe sidecar contains an unexpected record type")
    state_ids = [str(item.get("state_id")) for item in states]
    decision_keys = [str(item.get("decision_key")) for item in pairs]
    if len(state_ids) != len(set(state_ids)):
        raise SameStateRouteProbeError("probe state IDs repeat")
    if (
        len(pairs) != EXPECTED_TOTAL_DECISIONS
        or len(decision_keys) != len(set(decision_keys))
        or set(decision_keys) != expected_decisions
    ):
        raise SameStateRouteProbeError("probe decision join is incomplete or duplicated")
    state_set = set(state_ids)
    if any(str(item.get("state_id")) not in state_set for item in pairs):
        raise SameStateRouteProbeError("route_pair references an absent state")
    actual_by_pilot: dict[str, set[str]] = {label: set() for label in _LABELS}
    referenced: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        source_pilot = pair.get("source_pilot")
        if source_pilot not in _LABELS:
            raise SameStateRouteProbeError("route_pair has an invalid source pilot")
        actual_by_pilot[str(source_pilot)].add(str(pair["decision_key"]))
        referenced[str(pair["state_id"])].add(str(pair["decision_key"]))
        outputs = _mapping(pair.get("outputs"), "paired outputs")
        if tuple(outputs) != _LABELS:
            raise SameStateRouteProbeError("route_pair output labels are not 4K/60K bound")
        for label in _LABELS:
            output = _mapping(outputs[label], f"{label} route output")
            actor_identity = _mapping(schema_actors[label], f"{label} schema actor")
            if output.get("checkpoint_sha256") != actor_identity["checkpoint_sha256"]:
                raise SameStateRouteProbeError("route output checkpoint binding differs from schema")
            if output.get("actor_digest") != actor_identity["actor_digest"]:
                raise SameStateRouteProbeError("route output actor binding differs from schema")
    if actual_by_pilot != expected_by_pilot:
        raise SameStateRouteProbeError("probe decision keys are cross-bound between pilots")
    for state in states:
        if referenced[str(state["state_id"])] != set(str(item) for item in state["decision_keys"]):
            raise SameStateRouteProbeError("state/route_pair decision join failed")
    summary = records[-1]
    if summary.get("decision_count") != len(pairs) or summary.get("state_count") != len(states):
        raise SameStateRouteProbeError("summary accounting cannot be reconstructed")
    if summary.get("decision_count_by_source_pilot") != {"4K": 16, "60K": 16}:
        raise SameStateRouteProbeError("summary pilot counts are not exactly 16/16")
    _canonical_json(records)
    return summary


def inspect_same_state_route_probe(path: str | Path) -> Mapping[str, Any]:
    """Strictly inspect a completely published V1 sidecar."""

    target = Path(path)
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SameStateRouteProbeError(f"cannot read probe sidecar: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise SameStateRouteProbeError("probe sidecar is empty or contains blank records")
    records = tuple(_strict_json_loads(line) for line in lines)
    if any(not isinstance(item, Mapping) for item in records):
        raise SameStateRouteProbeError("probe sidecar records must be mappings")
    return _validate_probe_records(records)


_DECISION_KEYS_4K = (
    "eec421d202b8ee222bd778883eab583c5e06d728bfd33cfb390900fa7d6b74ec",
    "2b708a7bf82fff77b57bc79af97ad82561fd4ae89ff2b89e0d7e2c22cc2e0044",
    "5f78b2b85c39b62157d4881bc658e76427c5fdb0d1979c3d6afe64815e2bafd4",
    "d8a4769e4d67e3cc760dd2251e598d42b7160307469944251e15f9e71ca27b53",
    "df4358f8a83b4a15343f0b1e7eaa1d4548d1489367815d8138fcf318dae6a7eb",
    "06c58320272f97f31f92d2aab6ac0ccaac5f2693d171e35d2fd7096c22d3e295",
    "bd6b26a2ba5740a4e30f7e3b255933257be23fddc892022dc24c99f1161286c8",
    "8020a3453153857815cb6b026c9dc6c10cfd8fac943fd4689a091f3a40c0840b",
    "53c5519d8939957d64f4bbb1face103f5b7112dd82f87f6855495d5090e6663b",
    "4d026ee57321c245bbbc2cb6ce2680da5033a181e4399d1f83034b46e13c6c54",
    "64e94121a8815511af17df8fd71957df69a251c498c28e711656dd56052790e3",
    "77fd9ab44cdd61d85bafe2214d855ec5b2db40293851c3b69a11d81cd51605a1",
    "342d78e8aaf0085c12cd2322733c2d812ea873d9eb240e811acb61020c60364e",
    "3101954f63986d2df24018cc7eb420b59866072373d14737b5cd46b9e8788fd6",
    "047eec70f58347f13d32f7686bb28caa9b0baa426dc4c0c83ee393fb0252f90c",
    "2b4786bbfb9ca62e2bc94c33378735d24a75029c91b7cb2e07ea754dd3f2380e",
)

_DECISION_KEYS_60K = (
    "c4065c4c13953f1196f98794554af88a20b2877b349a8b7d535649fc1a26ef20",
    "1776dea939370495a88b1e5115e36de59b951507dcfa01091e219d57514cf3f2",
    "28a24a0d133f0144f7d33bc152a63d350b6bc7fb80e4a88eb2691c6801031c59",
    "81c7daa89836ed5cd1144d8b67aa29f3e834c66fd1e5e3f0b901a35c717cc4d8",
    "9eb651556ca31cd4510dc9c580fb7c38a340e41b4dd3b8a25536a66b3b948d34",
    "040e78cd0eed26beca32b92c1432f2c55c788e8a5b1a7ecad86173e8c0724a0e",
    "dc9e4610d520378b3785230655cdba80ec7a31b53f0728415ff48750d81ddea9",
    "8a64c88824ed6cd6f6862eebe714821741c19323f70fb66e20779e6721d30076",
    "c5184ec30f93b82c4c786a6414cc78764701e72ce60807fc5fe064e6445b3155",
    "6ab1dbebc6c6fc4a7a82b78b5fb4ed9d02ccef1c6f385d9f89b323f84a36fd4c",
    "7e516b879c39da7ca978529653cf4382ae47fc8cf9fd36db81be2c5ed6a5ee41",
    "453250a5542f9916d4e424dab1331c920100dc76307553a179dc5fd22ab956c4",
    "1ec9566b71938ddbe1c3e4a0e7a89efbdd4e147d8000a4fb8f3fd8b364c3074f",
    "331a1b5c3e16941e50196c6b949c909631d08cac56125475bfde66d956bf0972",
    "9941009e5ff675359e242432acd5eb94e05d9f1217d50971624350cfa5b32bcc",
    "651bb04799458a5e07a0284d8a04255d1bb70a3b3f3ba46231b112b8dabf1ccd",
)


def default_same_state_probe_specs(
    repo_root: str | Path,
) -> tuple[ProbeActorSpec, ProbeActorSpec, ReplayPilotSpec, ReplayPilotSpec]:
    """Build the two pinned real-pilot specs without running either replay."""

    root = Path(repo_root).resolve()
    source_4k = root / "logs/ca_gat_mappo__small__seed-42__cfg-8226914aeecf__git-d6a1de72ae68"
    source_60k = root / "logs/ca_gat_mappo__small__seed-42__cfg-538ecd14d252__git-d6a1de72ae68"

    def diagnostic_config(source: Path) -> RunConfig:
        base = load_run_config(source / "config_snapshot.yaml")
        result = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(
                    base.training.mappo,
                    route_oracle_counterfactual_enabled=True,
                    route_oracle_selection_rate_ppm=1_000_000,
                ),
            ),
        )
        object.__setattr__(result, "git_commit", base.git_commit)
        object.__setattr__(result, "git_branch", base.git_branch)
        object.__setattr__(result, "git_dirty", base.git_dirty)
        result.validate()
        return result

    config_4k = diagnostic_config(source_4k)
    config_60k = diagnostic_config(source_60k)
    actor_4k = ProbeActorSpec(
        "4K",
        config_4k,
        source_4k / "checkpoints/final.pt",
        "a73f28c8fb8635a8868b6f27ed8aeb40c8e27d5025fc12703e905851f601fb99",
        "455af1a7dc686b5a54e64331b110a3d19b111c4b6b98892cbf047afaab1c147e",
    )
    actor_60k = ProbeActorSpec(
        "60K",
        config_60k,
        source_60k / "checkpoints/final.pt",
        "0db1a7b1a9a83a03fdea6a02fcddfbf17e4e84c5be6f746f809d7559b9d5497b",
        "b40cb0984c5f235c4b644bf5dd3216e2ed53f3527e1efd7a76d3803c5c11097b",
    )
    pilot_4k = ReplayPilotSpec(
        "4K",
        config_4k,
        root / "logs/oracle_precollapse_4k_pilot/actor-only-oracle-v1__948f6c0a6faf257489d22689/route_oracle_counterfactual_v1.jsonl",
        "5f53ef4d6ba83528ede486f6f123ae887a4edf8a719533ee891c88052dc95178",
        "actor-only-oracle-v1__948f6c0a6faf257489d22689",
        "feb93191dab13c7dac7a7bf9ca443eb926ba893b60f548805d2cfaaad8e3680d",
        _DECISION_KEYS_4K,
    )
    pilot_60k = ReplayPilotSpec(
        "60K",
        config_60k,
        root / "logs/oracle_postcollapse_60k_pilot/actor-only-oracle-v1__68dbcecb73c0ed9b208f22b2/route_oracle_counterfactual_v1.jsonl",
        "4a497021a64f12d6e772df1b666086606bd4d5867a04e3f559fd6abc0083f711",
        "actor-only-oracle-v1__68dbcecb73c0ed9b208f22b2",
        "89bc77a5c2b0907f881667a26c4edbf875c10697d3ad0bdb1a6055ec2714fc62",
        _DECISION_KEYS_60K,
    )
    return actor_4k, actor_60k, pilot_4k, pilot_60k


__all__ = [
    "EXPECTED_DECISIONS_PER_PILOT",
    "EXPECTED_TOTAL_DECISIONS",
    "ProbeActorSpec",
    "ReplayPilotSpec",
    "SAME_STATE_ROUTE_PROBE_FILENAME",
    "SAME_STATE_ROUTE_PROBE_SCHEMA_VERSION",
    "SameStateRouteProbeError",
    "SameStateRouteProbeResult",
    "SameStateRouteProbeRunner",
    "actor_batch_snapshot",
    "compare_zero_hidden_route_preferences",
    "default_same_state_probe_specs",
    "inspect_same_state_route_probe",
    "load_replay_pilot",
]
