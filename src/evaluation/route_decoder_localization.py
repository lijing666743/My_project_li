"""Read-only Route Decoder Localization V1 for two frozen FINAL actors.

The diagnostic consumes the already-published same-state sidecar.  It never
reconstructs states through an environment, trains a model, or changes actor
parameters.  Forward hooks are temporary observation points only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..artifacts import atomic_write_text_group, require_artifact_targets_absent
from ..config import load_run_config
from ..models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorNetworkOutput,
    ActorTensorBatch,
    CAGATMAPPOActor,
    CandidateAwareRouteDecoderV1,
)
from .actor_loader import (
    LoadedEvaluationActor,
    checkpoint_sha256,
    load_final_actor_for_oracle_diagnostic,
)
from .route_oracle import actor_state_digest, tensor_digest
from .same_state_route_probe import (
    EXPECTED_TOTAL_DECISIONS,
    ProbeActorSpec,
    inspect_same_state_route_probe,
)


LOCALIZATION_SCHEMA_VERSION = 1
PARAMETER_DIFF_FILENAME = "parameter_diff.csv"
ACTIVATION_DIFF_FILENAME = "activation_diff.csv"
ROUTE_DECISION_FILENAME = "route_decision.csv"
SUMMARY_FILENAME = "summary.json"
_LABELS = ("4K", "60K")
_EXPECTED_BASELINE = {
    "p_remote_mean_4k": 0.849920,
    "p_local_mean_4k": 0.081005,
    "p_remote_mean_60k": 0.009310,
    "p_local_mean_60k": 0.771500,
    "p_remote_mean_delta": -0.840610,
    "entropy_mean_4k": 1.106612,
    "entropy_mean_60k": 0.574332,
}
_BASELINE_TOLERANCE = 1.0e-5
_FLOAT_EPSILON = 1.0e-12


class RouteDecoderLocalizationError(RuntimeError):
    """Raised when a read-only localization reliability gate fails."""


@dataclass(frozen=True)
class LocalizationState:
    """One immutable actor input shared by one or more route decisions."""

    state_id: str
    source_pilot: str
    decision_keys: tuple[str, ...]
    actor_batch: ActorTensorBatch
    route_domains: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class LocalizationDecision:
    """One focal source-UAV decision from the existing paired sidecar."""

    state_id: str
    decision_key: str
    source_pilot: str
    source_uav: int
    task_id: int
    expected_outputs: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class LocalizationFixture:
    """Strictly validated complete-input fixture for localization."""

    source_path: Path
    source_sha256: str
    schema: Mapping[str, Any]
    states: tuple[LocalizationState, ...]
    decisions: tuple[LocalizationDecision, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise RouteDecoderLocalizationError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    before = _file_sha256(path)

    def reject_constant(value: str) -> None:
        raise RouteDecoderLocalizationError(
            f"fixture contains non-finite JSON constant {value}"
        )

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = tuple(
            json.loads(line, parse_constant=reject_constant) for line in lines
        )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise RouteDecoderLocalizationError(f"cannot read fixture: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise RouteDecoderLocalizationError("fixture is empty or has blank records")
    if any(not isinstance(record, Mapping) for record in records):
        raise RouteDecoderLocalizationError("fixture records must be mappings")
    if _file_sha256(path) != before:
        raise RouteDecoderLocalizationError("fixture changed while it was read")
    return records


_DTYPES = {
    "torch.bool": torch.bool,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int64": torch.int64,
}


def _tensor_from_snapshot(snapshot: Any, name: str) -> Tensor:
    if not isinstance(snapshot, Mapping):
        raise RouteDecoderLocalizationError(f"{name} snapshot must be a mapping")
    required = {"dtype", "shape", "nbytes", "sha256", "values"}
    if not required.issubset(snapshot):
        missing = sorted(required - set(snapshot))
        raise RouteDecoderLocalizationError(
            f"{name} snapshot lacks complete tensor fields: {missing}"
        )
    dtype_name = snapshot["dtype"]
    if dtype_name not in _DTYPES:
        raise RouteDecoderLocalizationError(
            f"{name} snapshot dtype is unsupported: {dtype_name!r}"
        )
    shape_value = snapshot["shape"]
    if not isinstance(shape_value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in shape_value
    ):
        raise RouteDecoderLocalizationError(f"{name} snapshot shape is invalid")
    shape = tuple(shape_value)
    try:
        tensor = torch.tensor(snapshot["values"], dtype=_DTYPES[dtype_name])
        tensor = tensor.reshape(shape).contiguous()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RouteDecoderLocalizationError(
            f"{name} snapshot values do not match shape/dtype: {exc}"
        ) from exc
    if tensor_digest(tensor) != snapshot["sha256"]:
        raise RouteDecoderLocalizationError(f"{name} snapshot SHA-256 mismatch")
    actual_nbytes = tensor.numpy().nbytes
    if isinstance(snapshot["nbytes"], bool) or snapshot["nbytes"] != actual_nbytes:
        raise RouteDecoderLocalizationError(f"{name} snapshot nbytes mismatch")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise RouteDecoderLocalizationError(f"{name} snapshot contains NaN or Inf")
    return tensor


def _actor_batch_from_snapshot(snapshot: Any) -> ActorTensorBatch:
    if not isinstance(snapshot, Mapping):
        raise RouteDecoderLocalizationError("state actor_input must be a mapping")
    required = {
        "self_features",
        "neighbor_public_features",
        "edge_features",
        "neighbor_mask",
        "action_masks",
        "action_indices",
        "episode_starts",
    }
    if set(snapshot) != required:
        raise RouteDecoderLocalizationError(
            "state actor_input does not contain the exact ActorTensorBatch fields"
        )
    action_masks = snapshot["action_masks"]
    action_indices = snapshot["action_indices"]
    if not isinstance(action_masks, Mapping) or not isinstance(action_indices, Mapping):
        raise RouteDecoderLocalizationError(
            "state action_masks/action_indices must be mappings"
        )
    if set(action_masks) != set(ACTION_BRANCH_ORDER) or set(action_indices) != set(
        ACTION_BRANCH_ORDER
    ):
        raise RouteDecoderLocalizationError(
            "state action mappings do not contain exactly seven branches"
        )
    batch = ActorTensorBatch(
        self_features=_tensor_from_snapshot(
            snapshot["self_features"], "self_features"
        ),
        neighbor_public_features=_tensor_from_snapshot(
            snapshot["neighbor_public_features"], "neighbor_public_features"
        ),
        edge_features=_tensor_from_snapshot(
            snapshot["edge_features"], "edge_features"
        ),
        neighbor_mask=_tensor_from_snapshot(
            snapshot["neighbor_mask"], "neighbor_mask"
        ),
        action_masks={
            branch: _tensor_from_snapshot(
                action_masks[branch], f"action_masks.{branch}"
            )
            for branch in ACTION_BRANCH_ORDER
        },
        action_indices={
            branch: _tensor_from_snapshot(
                action_indices[branch], f"action_indices.{branch}"
            )
            for branch in ACTION_BRANCH_ORDER
        },
        episode_starts=_tensor_from_snapshot(
            snapshot["episode_starts"], "episode_starts"
        ),
    )
    return batch


def _actor_batch_digest(batch: ActorTensorBatch) -> str:
    compact = {
        "self_features": tensor_digest(batch.self_features),
        "neighbor_public_features": tensor_digest(batch.neighbor_public_features),
        "edge_features": tensor_digest(batch.edge_features),
        "neighbor_mask": tensor_digest(batch.neighbor_mask),
        "action_masks": {
            branch: tensor_digest(batch.action_masks[branch])
            for branch in ACTION_BRANCH_ORDER
        },
        "action_indices": {
            branch: tensor_digest(batch.action_indices[branch])
            for branch in ACTION_BRANCH_ORDER
        },
        "episode_starts": tensor_digest(batch.episode_starts),
    }
    return hashlib.sha256(_canonical_json(compact).encode("utf-8")).hexdigest()


def load_localization_fixture(path: str | Path) -> LocalizationFixture:
    """Load the existing 32-decision sidecar and require complete input tensors."""

    target = Path(path).resolve()
    inspect_same_state_route_probe(target)
    source_sha256 = _file_sha256(target)
    records = _strict_json_records(target)
    schema = records[0]
    if schema.get("schema_version") != 1:
        raise RouteDecoderLocalizationError("fixture schema version is not V1")
    state_records = [
        record for record in records if record.get("record_type") == "state"
    ]
    pair_records = [
        record for record in records if record.get("record_type") == "route_pair"
    ]
    if len(pair_records) != EXPECTED_TOTAL_DECISIONS:
        raise RouteDecoderLocalizationError("fixture does not contain 32 decisions")

    states: list[LocalizationState] = []
    state_ids: set[str] = set()
    for record in state_records:
        state_id = str(record.get("state_id"))
        if state_id in state_ids:
            raise RouteDecoderLocalizationError("fixture state IDs repeat")
        batch = _actor_batch_from_snapshot(record.get("actor_input"))
        expected_digest = record.get("actor_input_digest")
        if _actor_batch_digest(batch) != expected_digest:
            raise RouteDecoderLocalizationError(
                f"state {state_id} actor_input_digest mismatch"
            )
        if tuple(batch.self_features.shape[:2]) != (1, 1):
            raise RouteDecoderLocalizationError(
                f"state {state_id} is not one [batch,time] actor input"
            )
        decision_keys = record.get("decision_keys")
        route_domains = record.get("route_domains")
        if not isinstance(decision_keys, list) or not decision_keys:
            raise RouteDecoderLocalizationError(
                f"state {state_id} has no decision join keys"
            )
        if not isinstance(route_domains, list) or len(route_domains) != batch.self_features.shape[2]:
            raise RouteDecoderLocalizationError(
                f"state {state_id} route domains are incomplete"
            )
        states.append(
            LocalizationState(
                state_id=state_id,
                source_pilot=str(record.get("source_pilot")),
                decision_keys=tuple(str(item) for item in decision_keys),
                actor_batch=batch,
                route_domains=tuple(tuple(domain) for domain in route_domains),
            )
        )
        state_ids.add(state_id)

    decisions: list[LocalizationDecision] = []
    joined: dict[str, set[str]] = {state.state_id: set() for state in states}
    for record in pair_records:
        state_id = str(record.get("state_id"))
        if state_id not in state_ids:
            raise RouteDecoderLocalizationError(
                "fixture decision references an absent state"
            )
        outputs = record.get("outputs")
        if not isinstance(outputs, Mapping) or tuple(outputs) != _LABELS:
            raise RouteDecoderLocalizationError(
                "fixture decision outputs are not ordered 4K/60K"
            )
        source_uav = record.get("source_uav")
        task_id = record.get("task_id")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (source_uav, task_id)
        ):
            raise RouteDecoderLocalizationError(
                "fixture decision source_uav/task_id is invalid"
            )
        decision_key = str(record.get("decision_key"))
        joined[state_id].add(decision_key)
        decisions.append(
            LocalizationDecision(
                state_id=state_id,
                decision_key=decision_key,
                source_pilot=str(record.get("source_pilot")),
                source_uav=source_uav,
                task_id=task_id,
                expected_outputs=outputs,
            )
        )
    for state in states:
        if joined[state.state_id] != set(state.decision_keys):
            raise RouteDecoderLocalizationError(
                f"state {state.state_id} decision join is incomplete"
            )
    if _file_sha256(target) != source_sha256:
        raise RouteDecoderLocalizationError("fixture changed during validation")
    return LocalizationFixture(
        source_path=target,
        source_sha256=source_sha256,
        schema=schema,
        states=tuple(states),
        decisions=tuple(decisions),
    )


def _stage_for_state_name(name: str) -> str:
    if name.startswith(("self_encoder", "public_encoder", "graph_encoder", "gru")):
        return "A"
    if name.startswith("action_heads.route"):
        if ".fixed_head" in name or ".remote_scorer" in name:
            return "C"
        if name.endswith("remote_candidate_ids"):
            return "C"
        return "B"
    return "OUT_OF_ROUTE_PATH"


def _numeric_metrics(left: Tensor, right: Tensor) -> dict[str, float | None]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RouteDecoderLocalizationError("paired tensors differ in shape or dtype")
    if left.dtype == torch.bool:
        left64 = left.to(torch.float64)
        right64 = right.to(torch.float64)
    elif left.is_floating_point() or left.dtype in (torch.int32, torch.int64):
        left64 = left.detach().to(device="cpu", dtype=torch.float64)
        right64 = right.detach().to(device="cpu", dtype=torch.float64)
    else:
        raise RouteDecoderLocalizationError(f"unsupported audit dtype {left.dtype}")
    left64 = left64.reshape(-1)
    right64 = right64.reshape(-1)
    if not torch.isfinite(left64).all() or not torch.isfinite(right64).all():
        raise RouteDecoderLocalizationError("paired tensors contain NaN or Inf")
    delta = right64 - left64
    left_norm = float(torch.linalg.vector_norm(left64).item())
    right_norm = float(torch.linalg.vector_norm(right64).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    denominator = 0.5 * (left_norm + right_norm)
    cosine = None
    if left_norm > _FLOAT_EPSILON and right_norm > _FLOAT_EPSILON:
        cosine = float(torch.dot(left64, right64).item() / (left_norm * right_norm))
    left_centered = left64 - left64.mean() if left64.numel() else left64
    right_centered = right64 - right64.mean() if right64.numel() else right64
    left_centered_norm = float(torch.linalg.vector_norm(left_centered).item())
    right_centered_norm = float(torch.linalg.vector_norm(right_centered).item())
    centered_cosine = None
    if (
        left_centered_norm > _FLOAT_EPSILON
        and right_centered_norm > _FLOAT_EPSILON
    ):
        centered_cosine = float(
            torch.dot(left_centered, right_centered).item()
            / (left_centered_norm * right_centered_norm)
        )
    count = left64.numel()
    return {
        "left_mean": float(left64.mean().item()) if count else 0.0,
        "right_mean": float(right64.mean().item()) if count else 0.0,
        "left_std": float(left64.std(unbiased=False).item()) if count else 0.0,
        "right_std": float(right64.std(unbiased=False).item()) if count else 0.0,
        "left_rms": float(torch.sqrt(torch.mean(left64.square())).item()) if count else 0.0,
        "right_rms": float(torch.sqrt(torch.mean(right64.square())).item()) if count else 0.0,
        "left_l2": left_norm,
        "right_l2": right_norm,
        "delta_l2": delta_norm,
        "relative_l2": delta_norm / max(denominator, _FLOAT_EPSILON),
        "mae": float(delta.abs().mean().item()) if count else 0.0,
        "max_abs": float(delta.abs().max().item()) if count else 0.0,
        "cosine": cosine,
        "centered_cosine": centered_cosine,
    }


def audit_actor_parameters(
    actor_4k: CAGATMAPPOActor,
    actor_60k: CAGATMAPPOActor,
) -> list[dict[str, Any]]:
    """Audit every actor parameter/buffer without changing either module."""

    groups = (
        ("parameter", dict(actor_4k.named_parameters()), dict(actor_60k.named_parameters())),
        ("buffer", dict(actor_4k.named_buffers()), dict(actor_60k.named_buffers())),
    )
    rows: list[dict[str, Any]] = []
    for kind, left_items, right_items in groups:
        if tuple(left_items) != tuple(right_items):
            raise RouteDecoderLocalizationError(
                f"4K/60K actor {kind} key ordering differs"
            )
        for name in left_items:
            left = left_items[name].detach().cpu()
            right = right_items[name].detach().cpu()
            metrics = _numeric_metrics(left, right)
            rows.append(
                {
                    "stage": _stage_for_state_name(name),
                    "kind": kind,
                    "name": name,
                    "shape": _canonical_json(list(left.shape)),
                    "dtype": str(left.dtype),
                    "numel": left.numel(),
                    "exact_equal": torch.equal(left, right),
                    **metrics,
                }
            )
    return rows


def _flatten_tensors(value: Any, prefix: str) -> dict[str, Tensor]:
    if isinstance(value, Tensor):
        return {prefix: value.detach().cpu().contiguous().clone()}
    if isinstance(value, Mapping):
        result: dict[str, Tensor] = {}
        for key, item in value.items():
            result.update(_flatten_tensors(item, f"{prefix}.{key}"))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten_tensors(item, f"{prefix}.{index}"))
        return result
    return {}


class ActivationHookRecorder:
    """Temporary pre/post hook recorder for the frozen actor route path."""

    def __init__(self, actor: CAGATMAPPOActor) -> None:
        decoder = actor.route_decoder
        if not isinstance(decoder, CandidateAwareRouteDecoderV1):
            raise RouteDecoderLocalizationError(
                "Route Decoder Localization V1 requires candidate_aware_v1"
            )
        self._modules: tuple[tuple[str, str, nn.Module], ...] = (
            ("A", "self_encoder", actor.self_encoder),
            ("A", "public_encoder", actor.public_encoder),
            ("A", "graph_encoder", actor.graph_encoder),
            ("A", "gru", actor.gru),
            ("B", "edge_projection", decoder.edge_projection),
            ("B", "candidate_normalization", decoder.candidate_normalization),
            ("B", "query_projection", decoder.query_projection),
            ("B", "candidate_projection", decoder.candidate_projection),
            ("C", "fixed_head", decoder.fixed_head),
            ("C", "remote_scorer", decoder.remote_scorer),
            ("C", "route_logits", decoder),
        )
        if len({id(module) for _stage, _name, module in self._modules}) != len(
            self._modules
        ):
            raise RouteDecoderLocalizationError("hook manifest contains duplicate modules")
        self._handles: list[Any] = []
        self.captures: dict[str, Tensor] = {}

    @property
    def manifest(self) -> tuple[str, ...]:
        return tuple(f"{stage}.{name}" for stage, name, _module in self._modules)

    def clear(self) -> None:
        self.captures.clear()

    def _record(self, base: str, value: Any) -> None:
        items = _flatten_tensors(value, base)
        for key, tensor in items.items():
            if key in self.captures:
                raise RouteDecoderLocalizationError(
                    f"hook point was invoked more than once for one state: {key}"
                )
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise RouteDecoderLocalizationError(
                    f"activation contains NaN or Inf: {key}"
                )
            self.captures[key] = tensor

    def __enter__(self) -> "ActivationHookRecorder":
        if self._handles:
            raise RouteDecoderLocalizationError("hook recorder is already active")
        for stage, name, module in self._modules:
            base = f"{stage}.{name}"

            def pre_hook(
                _module: nn.Module,
                inputs: tuple[Any, ...],
                *,
                key: str = base,
            ) -> None:
                self._record(f"{key}.pre", inputs)

            def post_hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                key: str = base,
            ) -> None:
                self._record(f"{key}.post", output)

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()


def compare_activation_captures(
    state_id: str,
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
) -> list[dict[str, Any]]:
    if tuple(left) != tuple(right):
        raise RouteDecoderLocalizationError(
            f"state {state_id} 4K/60K hook manifests differ"
        )
    rows: list[dict[str, Any]] = []
    for key in left:
        parts = key.split(".", 2)
        rows.append(
            {
                "state_id": state_id,
                "stage": parts[0],
                "hook": parts[1],
                "tensor_path": parts[2],
                "shape": _canonical_json(list(left[key].shape)),
                "dtype": str(left[key].dtype),
                "numel": left[key].numel(),
                **_numeric_metrics(left[key], right[key]),
            }
        )
    return rows


def _linear_cka(left: Sequence[Tensor], right: Sequence[Tensor]) -> float | None:
    if len(left) != len(right) or not left:
        raise RouteDecoderLocalizationError("CKA activation collections are incomplete")
    left_matrix = torch.stack(
        [tensor.reshape(-1).to(torch.float64) for tensor in left]
    )
    right_matrix = torch.stack(
        [tensor.reshape(-1).to(torch.float64) for tensor in right]
    )
    if left_matrix.shape != right_matrix.shape:
        raise RouteDecoderLocalizationError("CKA activation matrices differ in shape")
    left_gram = left_matrix @ left_matrix.T
    right_gram = right_matrix @ right_matrix.T
    left_centered = (
        left_gram
        - left_gram.mean(dim=0, keepdim=True)
        - left_gram.mean(dim=1, keepdim=True)
        + left_gram.mean()
    )
    right_centered = (
        right_gram
        - right_gram.mean(dim=0, keepdim=True)
        - right_gram.mean(dim=1, keepdim=True)
        + right_gram.mean()
    )
    numerator = float(torch.sum(left_centered * right_centered).item())
    denominator = math.sqrt(
        float(torch.sum(left_centered.square()).item())
        * float(torch.sum(right_centered.square()).item())
    )
    return None if denominator <= _FLOAT_EPSILON else numerator / denominator


def _route_row(
    *,
    decision: LocalizationDecision,
    label: str,
    route_domain: Sequence[Any],
    output: ActorNetworkOutput,
) -> dict[str, Any]:
    source = decision.source_uav
    logits = output.raw_logits["route"][0, 0, source].detach().cpu()
    mask = output.action_masks["route"][0, 0, source].detach().cpu()
    if len(route_domain) != logits.numel() or mask.numel() != logits.numel():
        raise RouteDecoderLocalizationError("route domain/logit/mask dimensions differ")
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked_logits, dim=-1)
    expected = decision.expected_outputs[label]
    expected_logits = torch.tensor(expected["raw_logits"], dtype=logits.dtype)
    expected_probabilities = torch.tensor(
        expected["masked_probabilities"], dtype=probabilities.dtype
    )
    if not torch.allclose(logits, expected_logits, rtol=1e-6, atol=1e-7):
        raise RouteDecoderLocalizationError(
            f"{label} raw logits do not reproduce fixture decision {decision.decision_key}"
        )
    if not torch.allclose(
        probabilities, expected_probabilities, rtol=1e-6, atol=1e-7
    ):
        raise RouteDecoderLocalizationError(
            f"{label} probabilities do not reproduce fixture decision {decision.decision_key}"
        )
    remote_indices = [
        index
        for index, route in enumerate(route_domain)
        if isinstance(route, int) and not isinstance(route, bool) and bool(mask[index])
    ]
    local_index = route_domain.index("local") if "local" in route_domain else None
    if local_index is None or not bool(mask[local_index]) or not remote_indices:
        raise RouteDecoderLocalizationError(
            "localization decision lacks legal Local or Remote routes"
        )
    p_remote = math.fsum(float(probabilities[index]) for index in remote_indices)
    p_local = float(probabilities[local_index])
    entropy = -math.fsum(
        float(value) * math.log(float(value))
        for value in probabilities.tolist()
        if value > 0.0
    )
    best_remote_logit = max(float(logits[index]) for index in remote_indices)
    remote_logits = logits[remote_indices].to(torch.float64)
    local_logit = float(logits[local_index])
    argmax_index = int(torch.argmax(masked_logits).item())
    return {
        "state_id": decision.state_id,
        "decision_key": decision.decision_key,
        "source_pilot": decision.source_pilot,
        "source_uav": source,
        "task_id": decision.task_id,
        "actor_label": label,
        "raw_logits": _canonical_json([float(value) for value in logits.tolist()]),
        "route_mask": _canonical_json([bool(value) for value in mask.tolist()]),
        "p_remote_total": p_remote,
        "p_local": p_local,
        "route_entropy": entropy,
        "max_legal_remote_logit_minus_local": best_remote_logit - local_logit,
        "logsumexp_legal_remote_logits_minus_local": float(
            torch.logsumexp(remote_logits, dim=0).item()
        )
        - local_logit,
        "argmax_index": argmax_index,
        "argmax_route": _canonical_json(route_domain[argmax_index]),
    }


def _baseline_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_label = {
        label: [row for row in rows if row["actor_label"] == label]
        for label in _LABELS
    }
    if any(len(items) != EXPECTED_TOTAL_DECISIONS for items in by_label.values()):
        raise RouteDecoderLocalizationError("route decision accounting is incomplete")

    def mean(label: str, key: str) -> float:
        return math.fsum(float(row[key]) for row in by_label[label]) / len(
            by_label[label]
        )

    result = {
        "p_remote_mean_4k": mean("4K", "p_remote_total"),
        "p_local_mean_4k": mean("4K", "p_local"),
        "p_remote_mean_60k": mean("60K", "p_remote_total"),
        "p_local_mean_60k": mean("60K", "p_local"),
        "entropy_mean_4k": mean("4K", "route_entropy"),
        "entropy_mean_60k": mean("60K", "route_entropy"),
    }
    result["p_remote_mean_delta"] = (
        result["p_remote_mean_60k"] - result["p_remote_mean_4k"]
    )
    pairs = {
        label: {str(row["decision_key"]): row for row in by_label[label]}
        for label in _LABELS
    }
    transitions = 0
    for key in pairs["4K"]:
        route_4k = json.loads(pairs["4K"][key]["argmax_route"])
        route_60k = json.loads(pairs["60K"][key]["argmax_route"])
        is_remote_4k = isinstance(route_4k, int) and not isinstance(route_4k, bool)
        transitions += int(is_remote_4k and route_60k == "local")
    result["argmax_transition_count_4k_remote_60k_local"] = transitions
    for key, expected in _EXPECTED_BASELINE.items():
        if not math.isclose(
            float(result[key]),
            expected,
            rel_tol=0.0,
            abs_tol=_BASELINE_TOLERANCE,
        ):
            raise RouteDecoderLocalizationError(
                f"baseline gate failed for {key}: {result[key]} != {expected}"
            )
    if result["argmax_transition_count_4k_remote_60k_local"] != 32:
        raise RouteDecoderLocalizationError(
            "baseline argmax transition gate is not 32/32 Remote-to-Local"
        )
    return result


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise RouteDecoderLocalizationError("refusing to publish an empty CSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _stage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in ("A", "B", "C", "OUT_OF_ROUTE_PATH"):
        selected = [row for row in rows if row["stage"] == stage]
        if not selected:
            continue
        result[stage] = {
            "row_count": len(selected),
            "exact_equal_count": sum(bool(row.get("exact_equal")) for row in selected),
            "median_relative_l2": median(float(row["relative_l2"]) for row in selected),
            "median_cosine": median(
                float(row["cosine"])
                for row in selected
                if row.get("cosine") is not None
            )
            if any(row.get("cosine") is not None for row in selected)
            else None,
        }
    return result


def _activation_summary(
    rows: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Mapping[str, Sequence[Tensor]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, values in vectors.items():
        selected = [
            row
            for row in rows
            if f"{row['stage']}.{row['hook']}.{row['tensor_path']}" == key
        ]
        if not selected:
            raise RouteDecoderLocalizationError(f"activation summary lacks rows for {key}")
        result[key] = {
            "state_count": len(selected),
            "median_relative_l2": median(float(row["relative_l2"]) for row in selected),
            "median_cosine": median(
                float(row["cosine"])
                for row in selected
                if row.get("cosine") is not None
            )
            if any(row.get("cosine") is not None for row in selected)
            else None,
            "linear_cka": _linear_cka(values["4K"], values["60K"]),
        }
    return result


def _torch_rng_snapshot() -> tuple[Tensor, tuple[Tensor, ...]]:
    return (
        torch.get_rng_state().clone(),
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else (),
    )


def _torch_rng_equal(
    left: tuple[Tensor, tuple[Tensor, ...]],
    right: tuple[Tensor, tuple[Tensor, ...]],
) -> bool:
    return torch.equal(left[0], right[0]) and len(left[1]) == len(right[1]) and all(
        torch.equal(a, b) for a, b in zip(left[1], right[1])
    )


def _load_bound_actor(
    label: str,
    fixture: LocalizationFixture,
    checkpoint_path: Path,
) -> LoadedEvaluationActor:
    actors = fixture.schema.get("actors")
    if not isinstance(actors, Mapping) or label not in actors:
        raise RouteDecoderLocalizationError(f"fixture lacks {label} actor identity")
    identity = actors[label]
    if not isinstance(identity, Mapping):
        raise RouteDecoderLocalizationError(f"fixture {label} actor identity is invalid")
    recorded_path = Path(str(identity.get("checkpoint_path"))).resolve()
    if checkpoint_path.resolve() != recorded_path:
        raise RouteDecoderLocalizationError(
            f"{label} checkpoint path is not the fixture-bound FINAL"
        )
    if checkpoint_sha256(checkpoint_path) != identity.get("checkpoint_sha256"):
        raise RouteDecoderLocalizationError(f"{label} checkpoint SHA-256 mismatch")
    config_path = checkpoint_path.resolve().parent.parent / "config_snapshot.yaml"
    config = load_run_config(config_path)
    spec = ProbeActorSpec(
        label=label,
        config=config,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=str(identity["checkpoint_sha256"]),
        expected_actor_digest=str(identity["actor_digest"]),
    )
    before = _torch_rng_snapshot()
    actor = load_final_actor_for_oracle_diagnostic(
        spec.config,
        spec.checkpoint_path,
        execution_device="cpu",
        expected_checkpoint_sha256=spec.expected_checkpoint_sha256,
        expected_actor_digest=spec.expected_actor_digest,
    )
    if not _torch_rng_equal(before, _torch_rng_snapshot()):
        raise RouteDecoderLocalizationError(f"loading {label} actor changed Torch RNG")
    return actor


def run_route_decoder_localization(
    *,
    checkpoint_4k: str | Path,
    checkpoint_60k: str | Path,
    paired_fixture: str | Path,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Run one read-only localization pass over the existing complete fixture."""

    fixture = load_localization_fixture(paired_fixture)
    target_dir = Path(output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    logs_root = (repo_root / "logs").resolve()
    if logs_root not in target_dir.parents:
        raise RouteDecoderLocalizationError("output_dir must be a new directory under logs/")
    require_artifact_targets_absent(
        (target_dir,), group_name="route decoder localization output directory"
    )

    rng_before = _torch_rng_snapshot()
    loaded = {
        "4K": _load_bound_actor("4K", fixture, Path(checkpoint_4k)),
        "60K": _load_bound_actor("60K", fixture, Path(checkpoint_60k)),
    }
    if loaded["4K"].actor_architecture_identity != loaded["60K"].actor_architecture_identity:
        raise RouteDecoderLocalizationError("4K/60K actor architectures differ")
    if loaded["4K"].action_domain_identity != loaded["60K"].action_domain_identity:
        raise RouteDecoderLocalizationError("4K/60K action domains differ")
    if any(item.actor.training for item in loaded.values()):
        raise RouteDecoderLocalizationError("localization actor is not in eval mode")
    if any(
        parameter.requires_grad
        for item in loaded.values()
        for parameter in item.actor.parameters()
    ):
        raise RouteDecoderLocalizationError("localization actor is not frozen")

    digests_before = {
        label: actor_state_digest(item.actor) for label, item in loaded.items()
    }
    parameter_rows = audit_actor_parameters(loaded["4K"].actor, loaded["60K"].actor)
    decisions_by_state: dict[str, list[LocalizationDecision]] = {}
    for decision in fixture.decisions:
        decisions_by_state.setdefault(decision.state_id, []).append(decision)

    activation_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    activation_vectors: dict[str, dict[str, list[Tensor]]] = {}
    recorders = {
        label: ActivationHookRecorder(item.actor) for label, item in loaded.items()
    }
    if recorders["4K"].manifest != recorders["60K"].manifest:
        raise RouteDecoderLocalizationError("4K/60K hook manifests differ")
    with recorders["4K"], recorders["60K"], torch.inference_mode():
        for state in fixture.states:
            outputs: dict[str, ActorNetworkOutput] = {}
            captures: dict[str, dict[str, Tensor]] = {}
            input_digest = _actor_batch_digest(state.actor_batch)
            for label in _LABELS:
                recorder = recorders[label]
                recorder.clear()
                actor = loaded[label].actor
                state.actor_batch.validate(actor.spec)
                hidden = actor.initial_hidden(
                    1, device="cpu", dtype=state.actor_batch.self_features.dtype
                )
                if torch.count_nonzero(hidden).item() != 0:
                    raise RouteDecoderLocalizationError("hidden input is not all-zero")
                outputs[label] = actor(state.actor_batch, hidden)
                captures[label] = dict(recorder.captures)
                if _actor_batch_digest(state.actor_batch) != input_digest:
                    raise RouteDecoderLocalizationError("actor input changed during forward")
            compared = compare_activation_captures(
                state.state_id, captures["4K"], captures["60K"]
            )
            activation_rows.extend(compared)
            for key in captures["4K"]:
                entry = activation_vectors.setdefault(key, {"4K": [], "60K": []})
                entry["4K"].append(captures["4K"][key])
                entry["60K"].append(captures["60K"][key])
            for decision in decisions_by_state[state.state_id]:
                if decision.source_uav < 0 or decision.source_uav >= len(state.route_domains):
                    raise RouteDecoderLocalizationError("source UAV lies outside state")
                domain = state.route_domains[decision.source_uav]
                for label in _LABELS:
                    route_rows.append(
                        _route_row(
                            decision=decision,
                            label=label,
                            route_domain=domain,
                            output=outputs[label],
                        )
                    )

    digests_after = {
        label: actor_state_digest(item.actor) for label, item in loaded.items()
    }
    if digests_after != digests_before:
        raise RouteDecoderLocalizationError("actor state changed during localization")
    if not _torch_rng_equal(rng_before, _torch_rng_snapshot()):
        raise RouteDecoderLocalizationError("localization changed Torch RNG state")

    baseline = _baseline_summary(route_rows)
    activation_summary = _activation_summary(activation_rows, activation_vectors)
    stage_activation: dict[str, Any] = {}
    for stage in ("A", "B", "C"):
        values = [
            item
            for key, item in activation_summary.items()
            if key.startswith(f"{stage}.") and ".post" in key
        ]
        stage_activation[stage] = {
            "hook_tensor_count": len(values),
            "median_relative_l2": median(
                float(item["median_relative_l2"]) for item in values
            ),
            "median_linear_cka": median(
                float(item["linear_cka"])
                for item in values
                if item["linear_cka"] is not None
            )
            if any(item["linear_cka"] is not None for item in values)
            else None,
        }
    stage_rank = sorted(
        ("A", "B", "C"),
        key=lambda stage: stage_activation[stage]["median_relative_l2"],
        reverse=True,
    )
    summary = {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "status": "complete",
        "fixture": {
            "path": str(fixture.source_path),
            "sha256": fixture.source_sha256,
            "complete_input_tensor_fixture": True,
            "unique_state_count": len(fixture.states),
            "decision_count": len(fixture.decisions),
        },
        "actors": {
            label: {
                "checkpoint_path": str(item.source_checkpoint_path),
                "checkpoint_sha256": item.source_checkpoint_sha256,
                "actor_digest_before": digests_before[label],
                "actor_digest_after": digests_after[label],
            }
            for label, item in loaded.items()
        },
        "execution_contract": {
            "device": "cpu",
            "dtype": "torch.float32",
            "eval": True,
            "inference_mode": True,
            "zero_hidden_each_state": True,
            "environment_reset_or_step": False,
            "training_or_backward": False,
            "model_state_changed": False,
        },
        "hook_manifest": list(recorders["4K"].manifest),
        "parameter_stage_summary": _stage_summary(parameter_rows),
        "activation_by_hook": activation_summary,
        "activation_stage_summary": stage_activation,
        "activation_relative_l2_rank": stage_rank,
        "baseline_regression": baseline,
        "localization_boundary": (
            "A=shared encoder/GAT/GRU; B=candidate/edge/query representation; "
            "C=fixed-head, remote scorer, and final route logits. The current "
            "candidate-aware actor has no separate post-decoder route head."
        ),
        "interpretation_boundary": (
            "read-only same-state activation/parameter localization evidence; "
            "not a training-causal proof or performance claim"
        ),
    }
    artifacts = (
        (target_dir / PARAMETER_DIFF_FILENAME, _csv_text(parameter_rows)),
        (target_dir / ACTIVATION_DIFF_FILENAME, _csv_text(activation_rows)),
        (target_dir / ROUTE_DECISION_FILENAME, _csv_text(route_rows)),
        (
            target_dir / SUMMARY_FILENAME,
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        ),
    )
    atomic_write_text_group(artifacts, group_name="route decoder localization V1")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Route Decoder Localization V1"
    )
    parser.add_argument("--checkpoint-4k", required=True)
    parser.add_argument("--checkpoint-60k", required=True)
    parser.add_argument("--paired-fixture", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_route_decoder_localization(
        checkpoint_4k=args.checkpoint_4k,
        checkpoint_60k=args.checkpoint_60k,
        paired_fixture=args.paired_fixture,
        output_dir=args.output_dir,
    )
    print(
        "Route Decoder Localization V1 complete: "
        f"states={summary['fixture']['unique_state_count']} "
        f"decisions={summary['fixture']['decision_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVATION_DIFF_FILENAME",
    "ActivationHookRecorder",
    "LocalizationDecision",
    "LocalizationFixture",
    "LocalizationState",
    "PARAMETER_DIFF_FILENAME",
    "ROUTE_DECISION_FILENAME",
    "RouteDecoderLocalizationError",
    "SUMMARY_FILENAME",
    "audit_actor_parameters",
    "compare_activation_captures",
    "load_localization_fixture",
    "run_route_decoder_localization",
]
