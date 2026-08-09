"""Implementation 06 tests for masked random policy and unified rollout."""

from __future__ import annotations

import csv
import inspect
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from src.cli import main as cli_main
from src.config import STREAM_IDS, load_run_config
from src.env.actions import ActionProposal
from src.env.environment import U2UMECEnvironment
from src.policies.heuristic_policy import (
    HeuristicPolicy,
    HeuristicSpecificationBlocker,
    heuristic_blocker_message,
)
from src.policies.random_policy import RandomPolicy
from src.registry import build_default_registry
from src.runner import Runner
from src.rollout import DASHBOARD_COLUMNS, RolloutRunner


BRANCH_ORDER = (
    "route",
    "tx_select",
    "resource_group",
    "resource_width",
    "power_level",
    "cpu_queue",
    "cpu_frequency",
)


def make_random_config(
    *,
    horizon: int = 8,
    arrivals: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    output_root: Path | None = None,
):
    overrides: dict[str, object] = {
        "mode": "random",
        "method_id": "random",
        "scenario_id": "small",
        "seed": 42,
        "environment.episode_horizon": horizon,
        "environment.arrival_probabilities": list(arrivals),
    }
    if output_root is not None:
        overrides.update({
            "output.logs_dir": str(output_root / "logs"),
            "output.dashboard_logs_dir": str(output_root / "dashboard_logs"),
            "output.plots_dir": str(output_root / "plots"),
        })
    return load_run_config(cli_overrides=overrides)


def observations_with_active_queues():
    config = make_random_config(horizon=6)
    environment = U2UMECEnvironment(config)
    reset = environment.reset()
    environment.step(environment.canonical_proposals())
    assert environment.current_observations is not None

    proposals: list[ActionProposal] = []
    remote_used = False
    canonical = config.action.canonical_inactive_values
    for observation in environment.current_observations:
        masks = observation.action_masks
        legal_routes = [
            value
            for value, allowed in zip(masks.route_domain, masks.route_mask)
            if bool(allowed)
        ]
        remote = next((value for value in legal_routes if isinstance(value, int)), None)
        route = remote if remote is not None and not remote_used else "local"
        remote_used = remote_used or remote is not None
        proposals.append(ActionProposal(
            uav_id=observation.uav_id,
            route=route,
            tx_select=canonical["tx_select"],
            resource_group=canonical["resource_group"],
            resource_width=int(canonical["resource_width"]),
            power_level=float(canonical["power_level"]),
            cpu_queue=canonical["cpu_queue"],
            cpu_frequency=float(canonical["cpu_frequency"]),
        ))
    environment.step(proposals)
    assert environment.current_observations is not None
    return config, environment, environment.current_observations


def assert_finite_or_none(test: unittest.TestCase, value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        test.assertTrue(math.isfinite(value))
        return
    if isinstance(value, dict):
        for nested in value.values():
            assert_finite_or_none(test, nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            assert_finite_or_none(test, nested)


class TestRandomPolicy(unittest.TestCase):
    def test_sampling_uses_exact_seven_branch_order(self) -> None:
        config = make_random_config(horizon=2, arrivals=(0.0, 0.0, 0.0, 0.0))
        observation = U2UMECEnvironment(config).reset().observations[0]
        original = observation.action_masks
        calls: list[str] = []

        class RecordingMasks:
            sampling_order = original.sampling_order

            @staticmethod
            def domain_for(branch: str):
                return original.domain_for(branch)

            @staticmethod
            def mask_for(branch: str, previous=None):
                calls.append(branch)
                return original.mask_for(branch, previous)

            @staticmethod
            def is_legal(proposal: ActionProposal) -> bool:
                return original.is_legal(proposal)

        actor_only = SimpleNamespace(
            uav_id=observation.uav_id,
            action_masks=RecordingMasks(),
        )
        proposal = RandomPolicy(42).act(actor_only)
        self.assertEqual(calls, list(BRANCH_ORDER))
        self.assertEqual(len(proposal.branches), 7)
        self.assertEqual(original.sampling_order, BRANCH_ORDER)

    def test_only_valid_masked_actions_are_sampled(self) -> None:
        _, _, observations = observations_with_active_queues()
        policy = RandomPolicy(42)
        for _ in range(64):
            for observation in observations:
                proposal = policy.act(observation)
                self.assertTrue(observation.action_masks.is_legal(proposal))

    def test_inactive_branches_use_frozen_canonical_values(self) -> None:
        config = make_random_config(horizon=1, arrivals=(0.0, 0.0, 0.0, 0.0))
        reset = U2UMECEnvironment(config).reset()
        canonical = config.action.canonical_inactive_values
        for observation in reset.observations:
            proposal = RandomPolicy(42).act(observation)
            self.assertEqual(proposal.route, canonical["route"])
            self.assertEqual(proposal.tx_select, canonical["tx_select"])
            self.assertEqual(proposal.resource_group, canonical["resource_group"])
            self.assertEqual(proposal.resource_width, canonical["resource_width"])
            self.assertEqual(proposal.power_level, canonical["power_level"])
            self.assertEqual(proposal.cpu_queue, canonical["cpu_queue"])
            self.assertEqual(proposal.cpu_frequency, canonical["cpu_frequency"])

    def test_same_seed_reproduces_and_different_seed_changes_trajectory(self) -> None:
        _, _, observations = observations_with_active_queues()
        first = RandomPolicy(42)
        second = RandomPolicy(42)
        different = RandomPolicy(43)
        first_trajectory = [
            first.act(observations[index % len(observations)]).branches
            for index in range(64)
        ]
        second_trajectory = [
            second.act(observations[index % len(observations)]).branches
            for index in range(64)
        ]
        different_trajectory = [
            different.act(observations[index % len(observations)]).branches
            for index in range(64)
        ]
        self.assertEqual(first_trajectory, second_trajectory)
        self.assertNotEqual(first_trajectory, different_trajectory)
        self.assertEqual(first.policy_stream_id, STREAM_IDS["torch_policy_sampling"])

    def test_policy_rng_consumption_does_not_change_environment_rng(self) -> None:
        config = make_random_config(horizon=3)
        first = U2UMECEnvironment(config)
        second = U2UMECEnvironment(config)
        first_reset = first.reset()
        second.reset()
        policy = RandomPolicy(42)
        for _ in range(100):
            for observation in first_reset.observations:
                policy.act(observation)
        first.step(first.canonical_proposals())
        second.step(second.canonical_proposals())
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_policy_receives_no_centralized_or_post_action_argument(self) -> None:
        parameters = tuple(inspect.signature(RandomPolicy.act).parameters)
        self.assertEqual(parameters, ("self", "observation"))
        _, _, observations = observations_with_active_queues()
        observation = observations[0]
        actor_only = SimpleNamespace(
            uav_id=observation.uav_id,
            action_masks=observation.action_masks,
        )
        proposal = RandomPolicy(42).act(actor_only)
        self.assertTrue(observation.action_masks.is_legal(proposal))


class TestRolloutRunner(unittest.TestCase):
    def test_complete_episode_uses_real_backend_and_no_fake_terminal_state(self) -> None:
        config = make_random_config(horizon=12)
        outcome = RolloutRunner(config, RandomPolicy(42)).run(write_artifacts=False)
        self.assertEqual(len(outcome.raw_records), 12)
        self.assertEqual(outcome.artifacts, ())
        self.assertTrue(outcome.raw_records[-1]["truncated"])
        self.assertIsNone(outcome.raw_records[-1]["next_decision_slot"])
        self.assertFalse(outcome.raw_records[-1]["bootstrap_allowed"])
        self.assertFalse(outcome.summary["episode"]["fake_terminal_state_created"])
        self.assertTrue(outcome.summary["conservation"]["is_conserved"])
        self.assertGreater(outcome.summary["tasks"]["generated"], 0)
        assert_finite_or_none(self, outcome.summary)

    def test_na_semantics_are_preserved_without_samples(self) -> None:
        config = make_random_config(
            horizon=1,
            arrivals=(0.0, 0.0, 0.0, 0.0),
        )
        outcome = RolloutRunner(config, RandomPolicy(42)).run(write_artifacts=False)
        self.assertIsNone(outcome.summary["tasks"]["completion_rate"])
        self.assertIsNone(outcome.summary["tasks"]["expiration_rate"])
        self.assertIsNone(outcome.summary["e2e_latency_s"]["mean"])
        self.assertEqual(outcome.summary["e2e_latency_s"]["valid_sample_count"], 0)
        self.assertIsNone(outcome.summary["transmission"]["outage_rate"])
        self.assertEqual(outcome.summary["transmission"]["outage_valid_sample_count"], 0)

    def test_real_artifacts_have_fixed_schema_and_no_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_random_config(horizon=5, output_root=root)
            outcome = RolloutRunner(config, RandomPolicy(42)).run()
            self.assertEqual(len(outcome.artifacts), 4)
            for artifact in outcome.artifacts:
                self.assertTrue(Path(artifact).is_file())
            raw_path = Path(config.artifact_paths()["raw_metrics"])
            records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 5)
            self.assertEqual(records[0]["run_id"], config.run_id)
            self.assertEqual(records[0]["policy_seed"], 42)
            self.assertEqual(records[0]["policy_stream_id"], 110)
            self.assertIn("proposal_action", records[0])
            self.assertIn("executed_action_summary", records[0])
            self.assertIn("rejection_or_downgrade_summary", records[0])

            aggregate = json.loads(
                Path(config.artifact_paths()["aggregate_metrics"]).read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate, outcome.summary)
            with Path(config.artifact_paths()["dashboard_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0]), DASHBOARD_COLUMNS)
            self.assertEqual(len(rows), 5)
            self.assertFalse(Path(config.artifact_paths()["dashboard_png"]).exists())
            self.assertFalse(Path(config.artifact_paths()["figure_input"]).exists())

    def test_direct_and_interactive_cli_share_random_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "smoke.json"
            config_path.write_text(json.dumps({
                "scenario_id": "small",
                "environment": {
                    "episode_horizon": 3,
                    "arrival_probabilities": [1.0, 1.0, 1.0, 1.0],
                },
                "output": {
                    "logs_dir": str(root / "logs"),
                    "dashboard_logs_dir": str(root / "dashboard_logs"),
                    "plots_dir": str(root / "plots"),
                },
            }), encoding="utf-8")
            direct_output: list[str] = []
            direct_status = cli_main(
                ["--mode", "random", "--config", str(config_path), "--seed", "42"],
                output_fn=direct_output.append,
            )
            answers = iter(("3", "small", str(config_path), "42"))
            interactive_output: list[str] = []
            interactive_status = cli_main(
                [],
                input_fn=lambda _prompt: next(answers),
                output_fn=interactive_output.append,
            )
            self.assertEqual(direct_status, 0)
            self.assertEqual(interactive_status, 0)
            self.assertTrue(any("status=completed" in line for line in direct_output))
            self.assertTrue(any("status=completed" in line for line in interactive_output))
            registry = build_default_registry()
            self.assertEqual(
                registry.resolve("random", "random").__name__,
                "random_rollout_handler",
            )


class TestHeuristicSpecificationBlocker(unittest.TestCase):
    def test_heuristic_is_unavailable_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_run_config(cli_overrides={
                "mode": "heuristic",
                "method_id": "heuristic",
                "scenario_id": "small",
                "seed": 42,
                "output.logs_dir": str(root / "logs"),
                "output.dashboard_logs_dir": str(root / "dashboard_logs"),
                "output.plots_dir": str(root / "plots"),
            })
            result = Runner().run(config)
            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result.artifacts, ())
            self.assertIn("HEURISTIC SPECIFICATION BLOCKER", result.message)
            self.assertEqual(result.message, heuristic_blocker_message())
            self.assertFalse((root / "logs").exists())
            with self.assertRaises(HeuristicSpecificationBlocker):
                HeuristicPolicy()


if __name__ == "__main__":
    unittest.main()
