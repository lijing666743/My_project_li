from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.env.environment import U2UMECEnvironment
from src.env.randomness import derive_training_episode_seed
from src.evaluation.same_state_route_probe import (
    ProbeActorSpec,
    ReplayPilotSpec,
    SameStateRouteProbeError,
    SameStateRouteProbeRunner,
    inspect_same_state_route_probe,
)
from src.models.ca_gat_mappo import CAGATMAPPOActor
from src.models.ca_gat_mappo_runtime import CAGATMAPPOFactualActorRuntime
from tests.test_route_oracle_diagnostic import make_oracle_config
from tests.test_same_state_route_probe import loaded_actor


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_synthetic_pilot(path: Path, label: str, config, loaded) -> tuple[str, ...]:
    episode_config = replace(
        config,
        seed=derive_training_episode_seed(config.seed, 0),
    )
    environment = U2UMECEnvironment(episode_config)
    reset = environment.reset()
    observations = tuple(reset.observations)
    runtime = CAGATMAPPOFactualActorRuntime(
        loaded.actor,
        config,
        device="cpu",
        dtype=torch.float32,
    )
    hidden = loaded.actor.initial_hidden(1, device="cpu", dtype=torch.float32)
    transitions = 0
    records = []
    decision_keys = []
    policy_identity = {
        "checkpoint_sha256": loaded.source_checkpoint_sha256,
        "actor_state_digest": loaded.actor_state_digest,
    }
    header = {
        "record_type": "schema",
        "schema_version": 1,
        "run_id": f"synthetic-{label.lower()}",
        "config_hash": config.config_hash,
        "policy_identity": policy_identity,
        "oracle_diagnostic_identity": {"execution_device": "cpu"},
    }
    while len(decision_keys) < 16:
        slot = observations[0].slot
        factual = runtime.sample_step(
            observations,
            hidden.detach(),
            episode_start=slot == 0,
        )
        proposals = tuple(factual.action_output.proposals[0][0])
        hidden_out = factual.action_output.hidden_out.detach()
        for source, observation in enumerate(observations):
            queue = observation.private_queues.unbound
            if (
                len(decision_keys) >= 16
                or not observation.action_masks.route_branch_active
                or not queue.head_valid_mask
            ):
                continue
            key = hashlib.sha256(
                f"{label}:{transitions}:{source}:{queue.head_task_id}".encode()
            ).hexdigest()
            decision_keys.append(key)
            decision = {
                "record_type": "oracle_decision",
                "schema_version": 1,
                "decision_key": key,
                "branch_count": 1,
                "episode_id": 0,
                "global_environment_step": transitions,
                "route_slot": slot,
                "source_uav": source,
                "task_id": int(queue.head_task_id),
                "FACTUAL_ROUTE": proposals[source].route,
                "initial_state_fingerprint": environment.state_fingerprint(),
                "initial_rng_fingerprint": environment.rng_fingerprint(),
                "initial_exogenous_fingerprint": environment.exogenous_fingerprint(),
                "initial_hidden_digest": __import__(
                    "src.evaluation.route_oracle",
                    fromlist=["tensor_digest"],
                ).tensor_digest(hidden_out),
                "policy_identity": policy_identity,
            }
            records.extend(
                (
                    decision,
                    {
                        "record_type": "oracle_branch",
                        "schema_version": 1,
                        "decision_key": key,
                        "branch_route": proposals[source].route,
                    },
                )
            )
        if len(decision_keys) == 16:
            break
        step = environment.step(proposals)
        if step.observations is None:
            raise AssertionError("synthetic pilot ended before 16 decisions")
        observations = tuple(step.observations)
        hidden = hidden_out
        transitions += 1
    path.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in (header, *records)
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(decision_keys)


class SameStateRouteProbeRunnerTests(unittest.TestCase):
    def test_cuda_pilots_with_cpu_runner_fail_before_actor_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_oracle_config(horizon=20)
            decision_keys = tuple(
                hashlib.sha256(f"decision-{index}".encode()).hexdigest()
                for index in range(16)
            )
            actor_4k = ProbeActorSpec(
                "4K", config, root / "4k.pt", "1" * 64, "a" * 64
            )
            actor_60k = ProbeActorSpec(
                "60K", config, root / "60k.pt", "2" * 64, "b" * 64
            )
            pilot_4k = ReplayPilotSpec(
                "4K",
                config,
                root / "pilot-4k.jsonl",
                "3" * 64,
                "synthetic-4k",
                config.config_hash,
                decision_keys,
            )
            pilot_60k = ReplayPilotSpec(
                "60K",
                config,
                root / "pilot-60k.jsonl",
                "4" * 64,
                "synthetic-60k",
                config.config_hash,
                decision_keys,
            )
            cuda_pilot = SimpleNamespace(
                header={
                    "oracle_diagnostic_identity": {
                        "execution_device": "cuda"
                    }
                }
            )
            with patch(
                "src.evaluation.same_state_route_probe.load_replay_pilot",
                side_effect=(cuda_pilot, cuda_pilot),
            ) as load_pilot, patch(
                "src.evaluation.same_state_route_probe._load_actor_rng_isolated"
            ) as load_actor:
                with self.assertRaisesRegex(
                    SameStateRouteProbeError,
                    r"4K=cuda, 60K=cuda, runner=cpu",
                ):
                    SameStateRouteProbeRunner(
                        actor_4k,
                        actor_60k,
                        pilot_4k,
                        pilot_60k,
                        execution_device="cpu",
                        output_root=root / "output",
                    )
            self.assertEqual(load_pilot.call_count, 2)
            load_actor.assert_not_called()

    def test_two_synthetic_16_decision_replays_publish_one_complete_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = make_oracle_config(horizon=20)
            config = replace(
                base,
                environment=replace(
                    base.environment,
                    arrival_probabilities=(1.0, 1.0),
                ),
            )
            config.validate()
            actor_4k = CAGATMAPPOActor(config)
            actor_60k = copy.deepcopy(actor_4k)
            with torch.no_grad():
                actor_60k.action_heads["route"].bias.add_(0.25)
                actor_60k.action_heads["route"].bias[-1].add_(0.75)
            loaded_4k = loaded_actor(actor_4k, config, root / "4k.pt", "1")
            loaded_60k = loaded_actor(actor_60k, config, root / "60k.pt", "2")
            actor_spec_4k = ProbeActorSpec(
                "4K",
                config,
                root / "4k.pt",
                loaded_4k.source_checkpoint_sha256,
                loaded_4k.actor_state_digest,
            )
            actor_spec_60k = ProbeActorSpec(
                "60K",
                config,
                root / "60k.pt",
                loaded_60k.source_checkpoint_sha256,
                loaded_60k.actor_state_digest,
            )
            sidecar_4k = root / "pilot-4k.jsonl"
            sidecar_60k = root / "pilot-60k.jsonl"
            keys_4k = write_synthetic_pilot(
                sidecar_4k, "4K", config, loaded_4k
            )
            keys_60k = write_synthetic_pilot(
                sidecar_60k, "60K", config, loaded_60k
            )
            pilot_4k = ReplayPilotSpec(
                "4K",
                config,
                sidecar_4k,
                file_sha256(sidecar_4k),
                "synthetic-4k",
                config.config_hash,
                keys_4k,
                max_factual_environment_transitions=20,
            )
            pilot_60k = ReplayPilotSpec(
                "60K",
                config,
                sidecar_60k,
                file_sha256(sidecar_60k),
                "synthetic-60k",
                config.config_hash,
                keys_60k,
                max_factual_environment_transitions=20,
            )
            with patch(
                "src.evaluation.same_state_route_probe._load_actor_rng_isolated",
                side_effect=(loaded_4k, loaded_60k),
            ), patch(
                "src.evaluation.same_state_route_probe._tracked_git_identity",
                return_value=SimpleNamespace(
                    commit="a" * 40,
                    tracked_diff_sha256="b" * 64,
                ),
            ):
                runner = SameStateRouteProbeRunner(
                    actor_spec_4k,
                    actor_spec_60k,
                    pilot_4k,
                    pilot_60k,
                    execution_device="cpu",
                    output_root=root / "output",
                )
                result = runner.run()
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.decision_count, 32)
            self.assertEqual(result.state_count, 16)
            self.assertEqual(result.factual_transitions_4k, 8)
            self.assertEqual(result.factual_transitions_60k, 8)
            summary = inspect_same_state_route_probe(result.sidecar_path)
            self.assertEqual(
                summary["decision_count_by_source_pilot"],
                {"4K": 16, "60K": 16},
            )
            lines = Path(result.sidecar_path).read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual(records[0]["oracle_annotation_contract"]["shadow_transitions"], 0)
            self.assertEqual(records[-1]["status"], "complete")
            self.assertTrue(
                all(
                    item["hidden_mode"] == "zero_each_state"
                    for item in records
                    if item.get("record_type") == "route_pair"
                )
            )


if __name__ == "__main__":
    unittest.main()
