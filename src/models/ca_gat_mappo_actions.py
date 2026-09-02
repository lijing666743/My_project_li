"""Sequential masked categorical actions for the CA-GAT-MAPPO actor.

This module implements only the proposal distribution boundary: stochastic
sampling, deterministic masked selection, selected-action log-probability,
active-branch entropy, and re-evaluation of an existing proposal. It does not
implement rollout storage, GAE, PPO updates, optimizers, trainers, or training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from ..config import RunConfig, STREAM_IDS
from ..env.actions import ActionProposal
from ..env.observation import ActionMasks, ActorObservation
from .ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorNetworkOutput,
    ActorTensorBatch,
    CAGATMAPPOActor,
    MAPPOTensorSpec,
)


class ActionDistributionError(ValueError):
    """Raised when sequential policy distribution inputs violate the contract."""


ProposalGrid = tuple[tuple[tuple[ActionProposal, ...], ...], ...]


def _derived_policy_seed(master_seed: int, stream_id: int) -> int:
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ActionDistributionError("master_seed must be a non-negative integer")
    if isinstance(stream_id, bool) or not isinstance(stream_id, int) or stream_id < 0:
        raise ActionDistributionError("policy stream ID must be a non-negative integer")
    state = np.random.SeedSequence([master_seed, stream_id]).generate_state(
        1, dtype=np.uint64
    )
    return int(state[0] % np.uint64(np.iinfo(np.int64).max))


@dataclass(frozen=True)
class SequentialActionMaskBatch:
    """Real environment ``ActionMasks`` arranged as [batch][time][agent]."""

    contracts: tuple[tuple[tuple[ActionMasks, ...], ...], ...]

    def __post_init__(self) -> None:
        normalized = tuple(
            tuple(tuple(agent_contracts) for agent_contracts in time_steps)
            for time_steps in self.contracts
        )
        object.__setattr__(self, "contracts", normalized)
        if not normalized or not normalized[0] or not normalized[0][0]:
            raise ActionDistributionError("action-mask batch cannot be empty")
        time_count = len(normalized[0])
        agent_count = len(normalized[0][0])
        for batch_index, time_steps in enumerate(normalized):
            if len(time_steps) != time_count:
                raise ActionDistributionError("action-mask batch has ragged time axes")
            for time_index, agent_contracts in enumerate(time_steps):
                if len(agent_contracts) != agent_count:
                    raise ActionDistributionError("action-mask batch has ragged agent axes")
                for agent_index, contract in enumerate(agent_contracts):
                    if not isinstance(contract, ActionMasks):
                        raise TypeError("each action-mask contract must be an ActionMasks")
                    if contract.uav_id != agent_index:
                        raise ActionDistributionError(
                            "action-mask contracts must use stable zero-based agent order"
                        )
                    if tuple(contract.sampling_order) != ACTION_BRANCH_ORDER:
                        raise ActionDistributionError(
                            "action-mask sampling order differs from the frozen order"
                        )

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[ActorObservation],
    ) -> "SequentialActionMaskBatch":
        """Build a [1,1,A] contract batch from one real environment step."""

        return cls.from_time_steps((observations,))

    @classmethod
    def from_time_steps(
        cls,
        steps: Sequence[Sequence[ActorObservation]],
    ) -> "SequentialActionMaskBatch":
        """Build a [1,T,A] contract batch from ordered real observations."""

        copied_steps = tuple(tuple(step) for step in steps)
        if not copied_steps:
            raise ActionDistributionError("at least one observation step is required")
        contracts: list[tuple[ActionMasks, ...]] = []
        for step in copied_steps:
            if not step or any(not isinstance(item, ActorObservation) for item in step):
                raise TypeError("each step must contain ActorObservation instances")
            contracts.append(tuple(item.action_masks for item in step))
        return cls((tuple(contracts),))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            len(self.contracts),
            len(self.contracts[0]),
            len(self.contracts[0][0]),
        )

    def validate(self, expected: tuple[int, int, int]) -> None:
        if self.shape != expected:
            raise ActionDistributionError(
                f"action-mask batch shape {self.shape} does not match actor shape {expected}"
            )

    def repeat_batch(self, repeats: int) -> "SequentialActionMaskBatch":
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise ActionDistributionError("repeats must be a positive integer")
        return SequentialActionMaskBatch(self.contracts * repeats)

    def flattened(self) -> tuple[ActionMasks, ...]:
        return tuple(
            contract
            for time_steps in self.contracts
            for agent_contracts in time_steps
            for contract in agent_contracts
        )


@dataclass(frozen=True)
class SequentialActionDistributionOutput:
    """Proposal probabilities and active contributions with [B,T,A] axes."""

    proposals: ProposalGrid
    raw_logits: Mapping[str, Tensor]
    action_masks: Mapping[str, Tensor]
    probabilities: Mapping[str, Tensor]
    action_indices: Mapping[str, Tensor]
    active_branches: Mapping[str, Tensor]
    branch_log_probs: Mapping[str, Tensor]
    joint_log_prob: Tensor
    branch_entropies: Mapping[str, Tensor]
    entropy: Tensor
    recurrent_features: Tensor
    hidden_out: Tensor
    mode: str

    def validate(self, spec: MAPPOTensorSpec) -> tuple[int, int, int]:
        if self.recurrent_features.ndim != 4:
            raise ActionDistributionError("recurrent features must be [B,T,A,H]")
        batch, time, agents, hidden = self.recurrent_features.shape
        if agents != spec.uav_count or hidden != spec.gru_hidden_dimension:
            raise ActionDistributionError("recurrent features differ from the tensor spec")
        shape = (batch, time, agents)
        if tuple(self.hidden_out.shape) != (
            batch,
            agents,
            spec.gru_hidden_dimension,
        ):
            raise ActionDistributionError("hidden_out must be [B,A,H]")
        if self.mode not in {"stochastic", "deterministic", "evaluation"}:
            raise ActionDistributionError("unknown action distribution mode")
        if _proposal_grid_shape(self.proposals) != shape:
            raise ActionDistributionError("proposal grid differs from the tensor axes")

        mappings = (
            self.raw_logits,
            self.action_masks,
            self.probabilities,
            self.action_indices,
            self.active_branches,
            self.branch_log_probs,
            self.branch_entropies,
        )
        if any(tuple(mapping) != ACTION_BRANCH_ORDER for mapping in mappings):
            raise ActionDistributionError("distribution mappings must use seven ordered branches")
        for branch in ACTION_BRANCH_ORDER:
            dimension = spec.action_dimensions[branch]
            logits = self.raw_logits[branch]
            mask = self.action_masks[branch]
            probabilities = self.probabilities[branch]
            indices = self.action_indices[branch]
            active = self.active_branches[branch]
            log_prob = self.branch_log_probs[branch]
            entropy = self.branch_entropies[branch]
            if tuple(logits.shape) != (*shape, dimension):
                raise ActionDistributionError(f"{branch} logits have an invalid shape")
            if tuple(mask.shape) != (*shape, dimension) or mask.dtype != torch.bool:
                raise ActionDistributionError(f"{branch} mask has an invalid shape or dtype")
            if not torch.all(torch.any(mask, dim=-1)):
                raise ActionDistributionError(f"{branch} mask is all-invalid")
            if tuple(probabilities.shape) != (*shape, dimension):
                raise ActionDistributionError(f"{branch} probabilities have an invalid shape")
            if tuple(indices.shape) != shape or indices.dtype != torch.long:
                raise ActionDistributionError(f"{branch} indices must be long [B,T,A]")
            if tuple(active.shape) != shape or active.dtype != torch.bool:
                raise ActionDistributionError(f"{branch} activity must be bool [B,T,A]")
            if tuple(log_prob.shape) != shape or tuple(entropy.shape) != shape:
                raise ActionDistributionError(f"{branch} statistics must be [B,T,A]")
            if not torch.isfinite(logits).all():
                raise ActionDistributionError(f"{branch} logits contain NaN or Inf")
            if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0.0):
                raise ActionDistributionError(f"{branch} probabilities are invalid")
            if torch.any(probabilities.masked_select(~mask) != 0.0):
                raise ActionDistributionError(f"{branch} assigns probability to illegal actions")
            probability_sum = probabilities.sum(dim=-1)
            if not torch.allclose(probability_sum, torch.ones_like(probability_sum)):
                raise ActionDistributionError(f"{branch} probabilities do not sum to one")
            if not torch.isfinite(log_prob).all() or not torch.isfinite(entropy).all():
                raise ActionDistributionError(f"{branch} statistics contain NaN or Inf")
            chosen_legal = torch.gather(mask, -1, indices.unsqueeze(-1)).squeeze(-1)
            if not torch.all(chosen_legal):
                raise ActionDistributionError(f"{branch} selected an illegal action")
            if torch.any(log_prob.masked_select(~active) != 0.0):
                raise ActionDistributionError(f"{branch} inactive log-prob is not zero")
            if torch.any(entropy.masked_select(~active) != 0.0):
                raise ActionDistributionError(f"{branch} inactive entropy is not zero")

        expected_joint = sum(self.branch_log_probs.values())
        expected_entropy = sum(self.branch_entropies.values())
        if tuple(self.joint_log_prob.shape) != shape or not torch.allclose(
            self.joint_log_prob, expected_joint
        ):
            raise ActionDistributionError("joint log-prob is not the active-branch sum")
        if tuple(self.entropy.shape) != shape or not torch.allclose(
            self.entropy, expected_entropy
        ):
            raise ActionDistributionError("entropy is not the active-branch sum")
        if not torch.isfinite(self.joint_log_prob).all() or not torch.isfinite(
            self.entropy
        ).all():
            raise ActionDistributionError("aggregate statistics contain NaN or Inf")
        return shape

    def proposal_at(self, batch: int, time: int, agent: int) -> ActionProposal:
        return self.proposals[batch][time][agent]


class CAGATMAPPOActionDistribution:
    """Shared sequential masked distribution for sampling and PPO re-evaluation."""

    def __init__(self, actor: CAGATMAPPOActor, config: RunConfig) -> None:
        if not isinstance(actor, CAGATMAPPOActor):
            raise TypeError("actor must be a CAGATMAPPOActor")
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        expected_spec = MAPPOTensorSpec.from_config(config)
        if actor.spec != expected_spec:
            raise ActionDistributionError("actor and config tensor specs differ")
        self.actor = actor
        self.spec = expected_spec
        self.master_seed = config.seed
        self.policy_stream_id = int(config.reproducibility.stream_ids["torch_policy_sampling"])
        if self.policy_stream_id != STREAM_IDS["torch_policy_sampling"]:
            raise ActionDistributionError("policy stream ID differs from the frozen stream 110")
        self.policy_seed = _derived_policy_seed(self.master_seed, self.policy_stream_id)
        self._generators: dict[tuple[str, int | None], torch.Generator] = {}

    @property
    def seed_material(self) -> dict[str, int]:
        return {
            "master_seed": self.master_seed,
            "policy_stream_id": self.policy_stream_id,
            "derived_torch_seed": self.policy_seed,
        }

    def reset_sampling_generators(self) -> None:
        """Restart owned CPU/CUDA policy streams from the derived seed."""

        self._generators.clear()

    def sample_actions(
        self,
        batch: ActorTensorBatch,
        action_masks: SequentialActionMaskBatch,
        hidden_in: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> SequentialActionDistributionOutput:
        """Sample legal proposals from the seven conditional categoricals."""

        return self._run(
            batch,
            action_masks,
            hidden_in,
            mode="stochastic",
            proposals=None,
            generator=generator,
        )

    def deterministic_actions(
        self,
        batch: ActorTensorBatch,
        action_masks: SequentialActionMaskBatch,
        hidden_in: Tensor | None = None,
    ) -> SequentialActionDistributionOutput:
        """Select the lowest-index legal argmax at every conditional branch."""

        return self._run(
            batch,
            action_masks,
            hidden_in,
            mode="deterministic",
            proposals=None,
            generator=None,
        )

    def evaluate_actions(
        self,
        batch: ActorTensorBatch,
        action_masks: SequentialActionMaskBatch,
        proposals: Sequence[Sequence[Sequence[ActionProposal]]],
        hidden_in: Tensor | None = None,
    ) -> SequentialActionDistributionOutput:
        """Re-evaluate existing proposals with the same logits, masks, and order."""

        return self._run(
            batch,
            action_masks,
            hidden_in,
            mode="evaluation",
            proposals=proposals,
            generator=None,
        )

    def _run(
        self,
        batch: ActorTensorBatch,
        action_masks: SequentialActionMaskBatch,
        hidden_in: Tensor | None,
        *,
        mode: str,
        proposals: Sequence[Sequence[Sequence[ActionProposal]]] | None,
        generator: torch.Generator | None,
    ) -> SequentialActionDistributionOutput:
        if not isinstance(action_masks, SequentialActionMaskBatch):
            raise TypeError("action_masks must be a SequentialActionMaskBatch")
        network_output = self.actor(batch, hidden_in)
        recurrent = network_output.recurrent_features
        shape = tuple(recurrent.shape[:3])
        action_masks.validate(shape)
        contracts = action_masks.flattened()
        external = (
            None
            if proposals is None
            else _flatten_proposals(proposals, shape, contracts)
        )
        contexts: list[dict[str, str | int | float]] = [
            {} for _ in range(len(contracts))
        ]
        selected_indices: dict[str, Tensor] = {}
        raw_logits: dict[str, Tensor] = {}
        dynamic_masks: dict[str, Tensor] = {}
        probabilities: dict[str, Tensor] = {}
        selected_log_probs: dict[str, Tensor] = {}
        branch_entropies: dict[str, Tensor] = {}
        resolved_generator = (
            self._resolve_generator(recurrent.device, generator)
            if mode == "stochastic"
            else None
        )

        for branch in ACTION_BRANCH_ORDER:
            logits = (
                network_output.raw_logits["route"]
                if branch == "route"
                else self.actor.branch_logits(branch, recurrent, selected_indices)
            )
            dimension = self.spec.action_dimensions[branch]
            mask_rows: list[np.ndarray] = []
            for contract, context in zip(contracts, contexts):
                mask = np.asarray(contract.mask_for(branch, context), dtype=np.bool_)
                if mask.shape != (dimension,):
                    raise ActionDistributionError(f"{branch} mask has an invalid shape")
                if not np.any(mask):
                    raise ActionDistributionError(f"{branch} mask is all-invalid")
                mask_rows.append(mask)
            mask_tensor = torch.as_tensor(
                np.stack(mask_rows).reshape(*shape, dimension),
                dtype=torch.bool,
                device=recurrent.device,
            )
            masked_logits = logits.masked_fill(~mask_tensor, -torch.inf)
            distribution = torch.distributions.Categorical(
                logits=masked_logits,
                validate_args=False,
            )
            branch_probabilities = distribution.probs
            if not torch.isfinite(branch_probabilities).all():
                raise ActionDistributionError(f"{branch} probabilities contain NaN or Inf")

            if external is not None:
                flat_indices = []
                flat_mask = mask_tensor.reshape(-1, dimension)
                for item_index, (contract, proposal) in enumerate(
                    zip(contracts, external)
                ):
                    index = _proposal_branch_index(contract, branch, proposal)
                    if not bool(flat_mask[item_index, index].item()):
                        raise ActionDistributionError(
                            f"external proposal uses an illegal {branch} action"
                        )
                    flat_indices.append(index)
                indices = torch.as_tensor(
                    flat_indices,
                    dtype=torch.long,
                    device=recurrent.device,
                ).reshape(shape)
            elif mode == "deterministic":
                indices = torch.argmax(masked_logits, dim=-1)
            elif mode == "stochastic":
                indices = torch.multinomial(
                    branch_probabilities.reshape(-1, dimension),
                    num_samples=1,
                    replacement=True,
                    generator=resolved_generator,
                ).reshape(shape)
            else:
                raise ActionDistributionError(f"unknown selection mode {mode!r}")

            flat_selected = indices.detach().cpu().reshape(-1).tolist()
            for contract, context, index in zip(contracts, contexts, flat_selected):
                context[branch] = contract.domain_for(branch)[int(index)]
            raw_logits[branch] = logits
            dynamic_masks[branch] = mask_tensor
            probabilities[branch] = branch_probabilities
            selected_indices[branch] = indices
            selected_log_probs[branch] = distribution.log_prob(indices)
            branch_entropies[branch] = distribution.entropy()

        resolved_proposals = (
            _proposals_from_contexts(contexts, shape)
            if external is None
            else _proposal_grid_from_flat(external, shape)
        )
        flat_resolved = tuple(
            proposal
            for time_steps in resolved_proposals
            for agent_proposals in time_steps
            for proposal in agent_proposals
        )
        active_rows: list[tuple[bool, ...]] = []
        for contract, proposal in zip(contracts, flat_resolved):
            if not contract.is_legal(proposal):
                raise ActionDistributionError("resolved proposal violates its real action masks")
            active_rows.append(contract.active_indicators(proposal))
        active_tensor = torch.as_tensor(
            np.asarray(active_rows, dtype=np.bool_).reshape(*shape, len(ACTION_BRANCH_ORDER)),
            dtype=torch.bool,
            device=recurrent.device,
        )

        active_branches: dict[str, Tensor] = {}
        contributed_log_probs: dict[str, Tensor] = {}
        contributed_entropies: dict[str, Tensor] = {}
        for branch_index, branch in enumerate(ACTION_BRANCH_ORDER):
            active = active_tensor[..., branch_index]
            active_branches[branch] = active
            contributed_log_probs[branch] = torch.where(
                active,
                selected_log_probs[branch],
                torch.zeros_like(selected_log_probs[branch]),
            )
            contributed_entropies[branch] = torch.where(
                active,
                branch_entropies[branch],
                torch.zeros_like(branch_entropies[branch]),
            )
        joint_log_prob = sum(contributed_log_probs.values())
        entropy = sum(contributed_entropies.values())
        result = SequentialActionDistributionOutput(
            proposals=resolved_proposals,
            raw_logits=raw_logits,
            action_masks=dynamic_masks,
            probabilities=probabilities,
            action_indices=selected_indices,
            active_branches=active_branches,
            branch_log_probs=contributed_log_probs,
            joint_log_prob=joint_log_prob,
            branch_entropies=contributed_entropies,
            entropy=entropy,
            recurrent_features=recurrent,
            hidden_out=network_output.hidden_out,
            mode=mode,
        )
        result.validate(self.spec)
        return result

    def _resolve_generator(
        self,
        device: torch.device,
        supplied: torch.Generator | None,
    ) -> torch.Generator:
        if supplied is not None:
            supplied_device = torch.device(supplied.device)
            if supplied_device.type != device.type:
                raise ActionDistributionError(
                    "sampling generator and actor tensors must share a device type"
                )
            if (
                device.type == "cuda"
                and supplied_device.index is not None
                and device.index is not None
                and supplied_device.index != device.index
            ):
                raise ActionDistributionError(
                    "sampling generator and actor tensors must use the same CUDA device"
                )
            return supplied
        key = (device.type, device.index)
        if key not in self._generators:
            owned = torch.Generator(device=device)
            owned.manual_seed(self.policy_seed)
            self._generators[key] = owned
        return self._generators[key]


def _proposal_grid_shape(proposals: ProposalGrid) -> tuple[int, int, int]:
    if not proposals or not proposals[0] or not proposals[0][0]:
        raise ActionDistributionError("proposal grid cannot be empty")
    return (len(proposals), len(proposals[0]), len(proposals[0][0]))


def _flatten_proposals(
    proposals: Sequence[Sequence[Sequence[ActionProposal]]],
    shape: tuple[int, int, int],
    contracts: tuple[ActionMasks, ...],
) -> tuple[ActionProposal, ...]:
    normalized: ProposalGrid = tuple(
        tuple(tuple(agent_proposals) for agent_proposals in time_steps)
        for time_steps in proposals
    )
    if _proposal_grid_shape(normalized) != shape:
        raise ActionDistributionError("external proposal grid differs from actor axes")
    batch, time, agents = shape
    if any(len(normalized[index]) != time for index in range(batch)):
        raise ActionDistributionError("external proposal grid has a ragged time axis")
    if any(
        len(normalized[batch_index][time_index]) != agents
        for batch_index in range(batch)
        for time_index in range(time)
    ):
        raise ActionDistributionError("external proposal grid has a ragged agent axis")
    flattened = tuple(
        proposal
        for time_steps in normalized
        for agent_proposals in time_steps
        for proposal in agent_proposals
    )
    if any(not isinstance(proposal, ActionProposal) for proposal in flattened):
        raise TypeError("external proposals must be ActionProposal instances")
    for contract, proposal in zip(contracts, flattened):
        if proposal.uav_id != contract.uav_id:
            raise ActionDistributionError("external proposal UAV IDs are misaligned")
        if not contract.is_legal(proposal):
            raise ActionDistributionError("external proposal violates sequential action masks")
    return flattened


def _proposal_branch_index(
    contract: ActionMasks,
    branch: str,
    proposal: ActionProposal,
) -> int:
    value = getattr(proposal, branch)
    if branch == "cpu_queue" and value == "local":
        value = contract.uav_id
    if branch in {"power_level", "cpu_frequency"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ActionDistributionError(f"{branch} must be numeric")
        value = float(value)
    if branch == "resource_width" and (
        isinstance(value, bool) or not isinstance(value, int)
    ):
        raise ActionDistributionError("resource_width must be an integer")
    for index, candidate in enumerate(contract.domain_for(branch)):
        if (
            isinstance(candidate, float)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and candidate == float(value)
        ):
            return index
        if type(candidate) is type(value) and candidate == value:
            return index
    raise ActionDistributionError(f"external {branch} action is outside its domain")


def _proposals_from_contexts(
    contexts: Sequence[Mapping[str, str | int | float]],
    shape: tuple[int, int, int],
) -> ProposalGrid:
    proposals = tuple(
        ActionProposal(
            uav_id=index % shape[2],
            route=context["route"],
            tx_select=context["tx_select"],
            resource_group=context["resource_group"],
            resource_width=int(context["resource_width"]),
            power_level=float(context["power_level"]),
            cpu_queue=context["cpu_queue"],
            cpu_frequency=float(context["cpu_frequency"]),
        )
        for index, context in enumerate(contexts)
    )
    return _proposal_grid_from_flat(proposals, shape)


def _proposal_grid_from_flat(
    proposals: Sequence[ActionProposal],
    shape: tuple[int, int, int],
) -> ProposalGrid:
    batch, time, agents = shape
    copied = tuple(proposals)
    return tuple(
        tuple(
            tuple(
                copied[(batch_index * time + time_index) * agents + agent_index]
                for agent_index in range(agents)
            )
            for time_index in range(time)
        )
        for batch_index in range(batch)
    )


__all__ = [
    "ActionDistributionError",
    "CAGATMAPPOActionDistribution",
    "SequentialActionDistributionOutput",
    "SequentialActionMaskBatch",
]
