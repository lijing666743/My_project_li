from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from torch import nn

from src.evaluation.actor_loader import LoadedEvaluationActor, checkpoint_sha256
from src.evaluation.route_oracle import actor_state_digest
import src.evaluation.same_state_route_probe as probe
from src.evaluation.same_state_route_probe import (
    ProbeActorSpec,
    SameStateRouteProbeError,
    actor_batch_snapshot,
    compare_zero_hidden_route_preferences,
    default_same_state_probe_specs,
    inspect_same_state_route_probe,
    load_replay_pilot,
)
from src.models.ca_gat_mappo import ActorObservationTensorizer, CAGATMAPPOActor
from src.models.ca_gat_mappo_actions import SequentialActionMaskBatch
from tests.test_actor_only_route_oracle import make_actor_only_fixture
from tests.test_route_oracle_diagnostic import active_fixture


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loaded_actor(actor: CAGATMAPPOActor, config, path: Path, checkpoint_digit: str):
    actor.eval()
    actor.requires_grad_(False)
    return LoadedEvaluationActor(
        actor=actor,
        source_checkpoint_path=path,
        source_checkpoint_sha256=checkpoint_digit * 64,
        actor_state_digest=actor_state_digest(actor),
        source_checkpoint_kind="FINAL_COMPLETED",
        source_method_id="ca_gat_mappo",
        source_training_config_hash=config.config_hash,
        source_training_actor_ratio_mode="joint",
        source_training_agent_credit_mode="team",
        source_training_git_commit="a" * 40,
        source_training_device="cpu",
        evaluation_device=next(actor.parameters()).device.type,
        dtype=str(next(actor.parameters()).dtype),
        actor_architecture_identity={"fixture": "same"},
        action_domain_identity={"fixture": "same"},
    )


def pair_fixture(root: Path, *, device: str = "cpu"):
    config, environment, observations, _proposals, _remote = active_fixture(
        factual_route="local"
    )
    actor_4k = CAGATMAPPOActor(config)
    actor_60k = copy.deepcopy(actor_4k)
    route_4k = actor_4k.action_heads["route"]
    route_60k = actor_60k.action_heads["route"]
    if not isinstance(route_4k, nn.Linear) or not isinstance(route_60k, nn.Linear):
        raise AssertionError("test fixture expects the legacy linear route head")
    with torch.no_grad():
        route_4k.weight.zero_()
        route_4k.bias.zero_()
        route_4k.bias[observations[0].action_masks.route_domain.index("local")] = 3.0
        route_60k.weight.zero_()
        route_60k.bias.zero_()
        remote_index = next(
            index
            for index, value in enumerate(observations[0].action_masks.route_domain)
            if isinstance(value, int) and not isinstance(value, bool)
        )
        route_60k.bias[remote_index] = 3.0
    actor_4k.to(device)
    actor_60k.to(device)
    batch = ActorObservationTensorizer(config).encode_step(
        observations,
        device=device,
        dtype=torch.float32,
        episode_start=False,
    )
    masks = SequentialActionMaskBatch.from_observations(observations)
    return (
        config,
        environment,
        tuple(observations),
        batch,
        masks,
        loaded_actor(actor_4k, config, root / "4k.pt", "1"),
        loaded_actor(actor_60k, config, root / "60k.pt", "2"),
    )


class ActorLoadingRNGIsolationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_initialized_actor_constructor_preserves_global_torch_rng(self):
        config, _environment, _observations, _proposals, _remote = active_fixture()
        torch.cuda.get_rng_state_all()
        cpu_before = torch.get_rng_state().clone()
        cuda_before = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

        actor = CAGATMAPPOActor(config)

        cpu_after = torch.get_rng_state()
        cuda_after = tuple(torch.cuda.get_rng_state_all())
        self.assertEqual(next(actor.parameters()).device.type, "cpu")
        self.assertTrue(torch.equal(cpu_before, cpu_after))
        self.assertEqual(len(cuda_before), len(cuda_after))
        self.assertTrue(
            all(torch.equal(before, after) for before, after in zip(cuda_before, cuda_after))
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_real_cuda_actor_loader_preserves_global_torch_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, diagnostic, checkpoint, source_actor = make_actor_only_fixture(root)
            expected_checkpoint_sha256 = checkpoint_sha256(checkpoint)
            expected_actor_digest = actor_state_digest(source_actor)
            spec = ProbeActorSpec(
                "4K",
                diagnostic,
                checkpoint,
                expected_checkpoint_sha256,
                expected_actor_digest,
            )
            torch.cuda.get_rng_state_all()
            cpu_before = torch.get_rng_state().clone()
            cuda_before = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

            loaded = probe._load_actor_rng_isolated(spec, "cuda")

            cpu_after = torch.get_rng_state()
            cuda_after = tuple(torch.cuda.get_rng_state_all())
            self.assertEqual(loaded.evaluation_device, "cuda")
            self.assertEqual(next(loaded.actor.parameters()).device.type, "cuda")
            self.assertFalse(loaded.actor.training)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in loaded.actor.parameters())
            )
            self.assertEqual(
                loaded.source_checkpoint_sha256,
                expected_checkpoint_sha256,
            )
            self.assertEqual(loaded.actor_state_digest, expected_actor_digest)
            self.assertTrue(torch.equal(cpu_before, cpu_after))
            self.assertEqual(len(cuda_before), len(cuda_after))
            self.assertTrue(
                all(torch.equal(before, after) for before, after in zip(cuda_before, cuda_after))
            )


class ZeroHiddenPairTests(unittest.TestCase):
    def test_distinct_actor_outputs_are_identity_bound_and_rng_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, environment, observations, batch, masks, actor_4k, actor_60k = (
                pair_fixture(root)
            )
            generator = torch.Generator(device="cpu").manual_seed(9876)
            factual_hidden = torch.ones(
                (1, config.environment.uav_count, config.training.mappo.gru_hidden_dimension)
            )
            before_policy = generator.get_state().clone()
            before_torch = torch.get_rng_state().clone()
            before_environment = (
                environment.state_fingerprint(),
                environment.rng_fingerprint(),
            )
            before_hidden = factual_hidden.clone()
            before_batch = actor_batch_snapshot(batch)

            result = compare_zero_hidden_route_preferences(
                loaded_4k=actor_4k,
                loaded_60k=actor_60k,
                config=config,
                actor_batch=batch,
                action_masks=masks,
                observations=observations,
                source_uavs=(0,),
                factual_policy_generator=generator,
                environment=environment,
                factual_hidden=factual_hidden,
            )[0]

            self.assertEqual(
                result["outputs"]["4K"]["checkpoint_sha256"],
                actor_4k.source_checkpoint_sha256,
            )
            self.assertEqual(
                result["outputs"]["60K"]["actor_digest"],
                actor_60k.actor_state_digest,
            )
            self.assertNotEqual(
                result["outputs"]["4K"]["masked_probabilities"],
                result["outputs"]["60K"]["masked_probabilities"],
            )
            self.assertLess(result["delta"]["p_local"], 0.0)
            self.assertGreater(result["delta"]["p_remote_total"], 0.0)
            self.assertTrue(torch.equal(before_policy, generator.get_state()))
            self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
            self.assertEqual(
                before_environment,
                (environment.state_fingerprint(), environment.rng_fingerprint()),
            )
            self.assertTrue(torch.equal(before_hidden, factual_hidden))
            self.assertEqual(before_batch, actor_batch_snapshot(batch))

    def test_same_actor_is_rejected_and_swapping_negates_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = pair_fixture(Path(directory))
            config, _environment, observations, batch, masks, actor_4k, actor_60k = fixture
            with self.assertRaisesRegex(SameStateRouteProbeError, "same actor object"):
                compare_zero_hidden_route_preferences(
                    loaded_4k=actor_4k,
                    loaded_60k=actor_4k,
                    config=config,
                    actor_batch=batch,
                    action_masks=masks,
                    observations=observations,
                    source_uavs=(0,),
                )
            forward = compare_zero_hidden_route_preferences(
                loaded_4k=actor_4k,
                loaded_60k=actor_60k,
                config=config,
                actor_batch=batch,
                action_masks=masks,
                observations=observations,
                source_uavs=(0,),
            )[0]["delta"]
            reverse = compare_zero_hidden_route_preferences(
                loaded_4k=actor_60k,
                loaded_60k=actor_4k,
                config=config,
                actor_batch=batch,
                action_masks=masks,
                observations=observations,
                source_uavs=(0,),
            )[0]["delta"]
            for name in (
                "p_local",
                "p_defer",
                "p_remote_total",
                "route_entropy",
                "max_legal_remote_logit_minus_local",
                "logsumexp_legal_remote_logits_minus_local",
            ):
                self.assertAlmostEqual(forward[name], -reverse[name], places=7)

    def test_factual_hidden_value_and_sample_order_do_not_affect_zero_hidden_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = pair_fixture(Path(directory))
            config, _environment, observations, batch, masks, actor_4k, actor_60k = fixture
            zeros = torch.zeros(
                (1, config.environment.uav_count, config.training.mappo.gru_hidden_dimension)
            )
            arbitrary = torch.full_like(zeros, 17.0)
            first = compare_zero_hidden_route_preferences(
                loaded_4k=actor_4k,
                loaded_60k=actor_60k,
                config=config,
                actor_batch=batch,
                action_masks=masks,
                observations=observations,
                source_uavs=(0,),
                factual_hidden=zeros,
            )
            second = compare_zero_hidden_route_preferences(
                loaded_4k=actor_4k,
                loaded_60k=actor_60k,
                config=config,
                actor_batch=batch,
                action_masks=masks,
                observations=observations,
                source_uavs=(0,),
                factual_hidden=arbitrary,
            )
            self.assertEqual(first, second)

    def test_non_finite_actor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = pair_fixture(Path(directory))
            config, _environment, observations, batch, masks, actor_4k, actor_60k = fixture
            with torch.no_grad():
                actor_60k.actor.action_heads["route"].bias[0] = float("nan")
            with self.assertRaises(Exception):
                compare_zero_hidden_route_preferences(
                    loaded_4k=actor_4k,
                    loaded_60k=actor_60k,
                    config=config,
                    actor_batch=batch,
                    action_masks=masks,
                    observations=observations,
                    source_uavs=(0,),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_rng_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = pair_fixture(Path(directory), device="cuda")
            config, _environment, observations, batch, masks, actor_4k, actor_60k = fixture
            before = tuple(item.clone() for item in torch.cuda.get_rng_state_all())
            compare_zero_hidden_route_preferences(
                loaded_4k=actor_4k,
                loaded_60k=actor_60k,
                config=config,
                actor_batch=batch,
                action_masks=masks,
                observations=observations,
                source_uavs=(0,),
            )
            after = tuple(torch.cuda.get_rng_state_all())
            self.assertEqual(len(before), len(after))
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(before, after)))


class FixedPilotManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.actor_4k, cls.actor_60k, cls.pilot_4k, cls.pilot_60k = (
            default_same_state_probe_specs(cls.repo_root)
        )

    def test_real_pilot_manifests_are_exactly_16_plus_16_and_identity_bound(self):
        four = load_replay_pilot(self.pilot_4k, self.actor_4k)
        sixty = load_replay_pilot(self.pilot_60k, self.actor_60k)
        self.assertEqual(len(four.decisions), 16)
        self.assertEqual(len(sixty.decisions), 16)
        self.assertEqual(
            {item["decision_key"] for item in four.decisions},
            set(self.pilot_4k.expected_decision_keys),
        )
        self.assertEqual(
            {item["decision_key"] for item in sixty.decisions},
            set(self.pilot_60k.expected_decision_keys),
        )
        self.assertNotEqual(
            self.actor_4k.expected_checkpoint_sha256,
            self.actor_60k.expected_checkpoint_sha256,
        )
        self.assertNotEqual(
            self.actor_4k.expected_actor_digest,
            self.actor_60k.expected_actor_digest,
        )

    def test_real_pilot_configs_preserve_current_git_provenance(self):
        current_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertRegex(current_head, r"^[0-9a-f]{40}$")
        for actor_spec in (self.actor_4k, self.actor_60k):
            with self.subTest(label=actor_spec.label):
                self.assertNotEqual(actor_spec.config.git_commit, "unknown")
                self.assertNotEqual(actor_spec.config.git_branch, "unknown")
                self.assertEqual(actor_spec.config.git_commit, current_head)

    def _mutated_spec(self, root: Path, mutate):
        source = self.pilot_4k.sidecar_path
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        mutate(rows)
        target = root / "mutated.jsonl"
        target.write_text(
            "\n".join(json.dumps(item, separators=(",", ":")) for item in rows) + "\n",
            encoding="utf-8",
        )
        return replace(
            self.pilot_4k,
            sidecar_path=target,
            expected_sidecar_sha256=file_sha256(target),
        )

    def test_missing_duplicate_extra_and_join_failure_all_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def missing(rows):
                key = self.pilot_4k.expected_decision_keys[0]
                rows[:] = [item for item in rows if item.get("decision_key") != key]

            def duplicate(rows):
                rows.append(copy.deepcopy(next(item for item in rows if item.get("record_type") == "oracle_decision")))

            def extra(rows):
                original = self.pilot_4k.expected_decision_keys[0]
                replacement = "f" * 64
                for item in rows:
                    if item.get("decision_key") == original:
                        item["decision_key"] = replacement

            def broken_join(rows):
                decision = next(item for item in rows if item.get("record_type") == "oracle_decision")
                key = decision["decision_key"]
                index = next(
                    index
                    for index, item in enumerate(rows)
                    if item.get("record_type") == "oracle_branch"
                    and item.get("decision_key") == key
                )
                rows.pop(index)

            for name, mutation in (
                ("missing", missing),
                ("duplicate", duplicate),
                ("extra", extra),
                ("join", broken_join),
            ):
                with self.subTest(name=name):
                    spec = self._mutated_spec(root, mutation)
                    with self.assertRaises(SameStateRouteProbeError):
                        load_replay_pilot(spec, self.actor_4k)

    def test_cross_bound_actor_and_sidecar_sha_fail(self):
        with self.assertRaisesRegex(SameStateRouteProbeError, "labels are cross-bound"):
            load_replay_pilot(self.pilot_4k, self.actor_60k)
        with self.assertRaisesRegex(SameStateRouteProbeError, "sidecar SHA-256"):
            load_replay_pilot(
                replace(self.pilot_4k, expected_sidecar_sha256="0" * 64),
                self.actor_4k,
            )


def complete_probe_records():
    keys_4k = [hashlib.sha256(f"4k-{index}".encode()).hexdigest() for index in range(16)]
    keys_60k = [hashlib.sha256(f"60k-{index}".encode()).hexdigest() for index in range(16)]
    actor_4k = {"checkpoint_sha256": "1" * 64, "actor_digest": "3" * 64}
    actor_60k = {"checkpoint_sha256": "2" * 64, "actor_digest": "4" * 64}
    schema = {
        "record_type": "schema",
        "actors": {"4K": actor_4k, "60K": actor_60k},
        "pilots": {
            "4K": {"decision_keys": keys_4k},
            "60K": {"decision_keys": keys_60k},
        },
    }
    states = [
        {"record_type": "state", "state_id": "state-4k", "decision_keys": keys_4k},
        {"record_type": "state", "state_id": "state-60k", "decision_keys": keys_60k},
    ]
    pairs = []
    for label, keys, state_id in (
        ("4K", keys_4k, "state-4k"),
        ("60K", keys_60k, "state-60k"),
    ):
        for index, key in enumerate(keys):
            pairs.append(
                {
                    "record_type": "route_pair",
                    "state_id": state_id,
                    "decision_key": key,
                    "source_pilot": label,
                    "task_id": index,
                    "outputs": {
                        "4K": actor_4k,
                        "60K": actor_60k,
                    },
                    "delta": {
                        "p_remote_total": 0.0,
                        "max_legal_remote_logit_minus_local": 0.0,
                        "logsumexp_legal_remote_logits_minus_local": 0.0,
                        "argmax_transition": ["local", "local"],
                    },
                }
            )
    summary = {
        "record_type": "summary",
        "status": "complete",
        "state_count": 2,
        "decision_count": 32,
        "decision_count_by_source_pilot": {"4K": 16, "60K": 16},
    }
    return [schema, *states, *pairs, summary]


class FailureClosedArtifactTests(unittest.TestCase):
    def _write_records(self, path: Path, records) -> None:
        path.write_text(
            "\n".join(json.dumps(item, allow_nan=True) for item in records) + "\n",
            encoding="utf-8",
        )

    def test_complete_sidecar_is_accepted_and_partial_join_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            records = complete_probe_records()
            self._write_records(path, records)
            summary = inspect_same_state_route_probe(path)
            self.assertEqual(summary["decision_count"], 32)

            records.pop(3)
            self._write_records(path, records)
            with self.assertRaises(SameStateRouteProbeError):
                inspect_same_state_route_probe(path)

    def test_same_actor_schema_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            records = complete_probe_records()
            records[0]["actors"]["60K"] = copy.deepcopy(records[0]["actors"]["4K"])
            for pair in records[3:-1]:
                pair["outputs"]["60K"] = copy.deepcopy(pair["outputs"]["4K"])
            self._write_records(path, records)
            with self.assertRaisesRegex(SameStateRouteProbeError, "same actor identity"):
                inspect_same_state_route_probe(path)

    def test_decision_keys_cross_bound_between_pilots_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            records = complete_probe_records()
            pair_4k = next(
                item for item in records if item.get("source_pilot") == "4K"
            )
            pair_60k = next(
                item for item in records if item.get("source_pilot") == "60K"
            )
            pair_4k["source_pilot"] = "60K"
            pair_60k["source_pilot"] = "4K"
            self._write_records(path, records)
            with self.assertRaisesRegex(SameStateRouteProbeError, "cross-bound"):
                inspect_same_state_route_probe(path)

    def test_non_finite_and_non_complete_summary_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            records = complete_probe_records()
            records[3]["delta"]["p_remote_total"] = float("nan")
            self._write_records(path, records)
            with self.assertRaises(SameStateRouteProbeError):
                inspect_same_state_route_probe(path)

            records = complete_probe_records()
            records[-1]["status"] = "failed"
            self._write_records(path, records)
            with self.assertRaises(SameStateRouteProbeError):
                inspect_same_state_route_probe(path)

    def test_runner_failure_never_publishes_a_success_sidecar(self):
        failure_messages = (
            "replay fingerprint mismatch",
            "replay budget exhausted",
            "route logits/probabilities contain NaN or Inf",
            "partial decision",
            "sidecar join failure",
        )
        for message in failure_messages:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                runner = probe.SameStateRouteProbeRunner.__new__(
                    probe.SameStateRouteProbeRunner
                )
                runner._has_run = False
                runner.run_directory = Path(directory) / "run"
                runner.sidecar_path = runner.run_directory / probe.SAME_STATE_ROUTE_PROBE_FILENAME

                def fail(_label, text=message):
                    raise SameStateRouteProbeError(text)

                runner._replay_one = fail
                with self.assertRaisesRegex(SameStateRouteProbeError, message):
                    runner.run()
                self.assertFalse(runner.sidecar_path.exists())
                self.assertFalse(runner.run_directory.exists())


if __name__ == "__main__":
    unittest.main()
