from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.config import CHECKPOINT_KIND_FINAL_COMPLETED
from src.evaluation.actor_loader import (
    EvaluationCheckpointError,
    checkpoint_sha256,
    load_final_actor_for_oracle_diagnostic,
)
from src.evaluation.actor_only_route_oracle import (
    ActorOnlyRouteOracleRunner,
    _TrackedGitIdentity,
    _tracked_git_identity,
)
from src.evaluation.route_oracle import (
    OracleBudgetExhausted,
    OracleCollectionBudget,
    actor_state_digest,
    inspect_route_oracle_artifact,
    tensor_digest,
)
from src.models.ca_gat_mappo_checkpoint import atomic_save_checkpoint
from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from tests.test_ca_gat_mappo_checkpoint import make_payload
from tests.test_route_oracle_diagnostic import (
    RecordingFactory,
    make_oracle_config,
)


def make_actor_only_fixture(
    root: Path,
    *,
    horizon: int = 4,
    training_budget: int | None = None,
    arrivals: tuple[float, ...] = (1.0, 0.0),
):
    training_budget = training_budget or 2 * horizon
    base = make_oracle_config(horizon=horizon, output_dir=str(root / "logs"))
    environment = replace(
        base.environment,
        arrival_probabilities=arrivals,
    )
    source_mappo = replace(
        base.training.mappo,
        max_training_episodes=max(2, training_budget // horizon),
        max_training_environment_steps=training_budget,
        checkpoint_interval_steps=horizon,
        route_oracle_counterfactual_enabled=False,
        route_oracle_selection_rate_ppm=0,
    )
    source = replace(
        base,
        mode="rl",
        method_id="ca_gat_mappo",
        environment=environment,
        training=replace(
            base.training,
            formal_rl_enabled=True,
            mappo=source_mappo,
        ),
    )
    object.__setattr__(source, "git_branch", "checkpoint-source")
    object.__setattr__(source, "git_commit", "a" * 40)
    object.__setattr__(source, "git_dirty", False)
    source.validate()
    payload, actor, _critic, _updater = make_payload(
        source, CHECKPOINT_KIND_FINAL_COMPLETED
    )
    checkpoint = root / "final.pt"
    atomic_save_checkpoint(payload, checkpoint)

    diagnostic = replace(
        source,
        training=replace(
            source.training,
            mappo=replace(
                source.training.mappo,
                route_oracle_counterfactual_enabled=True,
                route_oracle_selection_rate_ppm=1_000_000,
            ),
        ),
    )
    object.__setattr__(diagnostic, "git_branch", "diagnostic")
    object.__setattr__(diagnostic, "git_commit", "b" * 40)
    object.__setattr__(diagnostic, "git_dirty", True)
    diagnostic.validate()
    return source, diagnostic, checkpoint, actor


def make_runner(
    root: Path,
    config,
    checkpoint: Path,
    actor,
    *,
    factual_cap: int,
    selected_cap: int | None = None,
    shadow_cap: int | None = 6_000,
    environment_factory=None,
) -> ActorOnlyRouteOracleRunner:
    return ActorOnlyRouteOracleRunner(
        config,
        checkpoint,
        expected_checkpoint_sha256=checkpoint_sha256(checkpoint),
        expected_actor_digest=actor_state_digest(actor),
        oracle_budget=OracleCollectionBudget(
            target_selected_decisions=selected_cap,
            max_factual_environment_transitions=factual_cap,
            max_shadow_transitions=shadow_cap,
        ),
        max_factual_environment_transitions=factual_cap,
        execution_device="cpu",
        output_root=root / "oracle_diagnostics",
        environment_factory=environment_factory,
    )


class ActorOnlyLoaderTests(unittest.TestCase):
    def test_semantic_compatibility_allows_different_config_hash_and_freezes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, diagnostic, checkpoint, actor = make_actor_only_fixture(root)
            self.assertNotEqual(source.config_hash, diagnostic.config_hash)
            loaded = load_final_actor_for_oracle_diagnostic(
                diagnostic,
                checkpoint,
                execution_device="cpu",
                expected_checkpoint_sha256=checkpoint_sha256(checkpoint),
                expected_actor_digest=actor_state_digest(actor),
            )
            self.assertFalse(loaded.actor.training)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in loaded.actor.parameters())
            )
            self.assertEqual(loaded.actor_state_digest, actor_state_digest(actor))

    def test_expected_hash_digest_and_state_shape_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, diagnostic, checkpoint, actor = make_actor_only_fixture(root)
            with self.assertRaisesRegex(
                EvaluationCheckpointError, "checkpoint SHA256"
            ):
                load_final_actor_for_oracle_diagnostic(
                    diagnostic,
                    checkpoint,
                    execution_device="cpu",
                    expected_checkpoint_sha256="0" * 64,
                    expected_actor_digest=actor_state_digest(actor),
                )
            with self.assertRaisesRegex(
                EvaluationCheckpointError, "actor digest"
            ):
                load_final_actor_for_oracle_diagnostic(
                    diagnostic,
                    checkpoint,
                    execution_device="cpu",
                    expected_checkpoint_sha256=checkpoint_sha256(checkpoint),
                    expected_actor_digest="0" * 64,
                )

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            first_name = next(iter(payload["model_state"]["actor"]))
            tensor = payload["model_state"]["actor"][first_name]
            payload["model_state"]["actor"][first_name] = tensor.reshape(-1)[:-1]
            incompatible = root / "incompatible.pt"
            torch.save(payload, incompatible)
            with self.assertRaisesRegex(
                EvaluationCheckpointError, "architecture/state"
            ):
                load_final_actor_for_oracle_diagnostic(
                    diagnostic,
                    incompatible,
                    execution_device="cpu",
                    expected_checkpoint_sha256=checkpoint_sha256(incompatible),
                    expected_actor_digest=actor_state_digest(actor),
                )


class ActorOnlyRunnerTests(unittest.TestCase):
    def test_zero_training_side_effects_and_no_checkpoint_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, config, checkpoint, actor = make_actor_only_fixture(
                root, arrivals=(0.0, 0.0)
            )
            runner = make_runner(
                root, config, checkpoint, actor, factual_cap=1
            )
            before = actor_state_digest(runner.loaded_actor.actor)
            with patch.object(
                torch.optim.Optimizer,
                "step",
                side_effect=AssertionError("optimizer.step is forbidden"),
            ), patch.object(
                torch.Tensor,
                "backward",
                side_effect=AssertionError("backward is forbidden"),
            ), patch(
                "src.models.ca_gat_mappo_checkpoint.restore_model_optimizer_and_rng",
                side_effect=AssertionError("full restore is forbidden"),
            ), patch(
                "src.models.ca_gat_mappo_checkpoint.restore_policy_rng",
                side_effect=AssertionError("policy RNG restore is forbidden"),
            ), patch(
                "src.models.ca_gat_mappo_checkpoint.restore_active_rollout",
                side_effect=AssertionError("rollout restore is forbidden"),
            ):
                result = runner.run()
            self.assertEqual(result.ppo_update_count, 0)
            self.assertTrue(result.actor_frozen)
            self.assertTrue(result.actor_weights_unchanged)
            self.assertEqual(result.actor_digest, before)
            contents = tuple(Path(result.run_directory).iterdir())
            self.assertEqual(contents, (Path(result.sidecar_path),))
            self.assertFalse((Path(result.run_directory) / "checkpoints").exists())

    def test_horizon_500_stops_exactly_at_factual_cap_200(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, config, checkpoint, actor = make_actor_only_fixture(
                root,
                horizon=500,
                training_budget=1_000,
                arrivals=(0.0, 0.0),
            )
            result = make_runner(
                root,
                config,
                checkpoint,
                actor,
                factual_cap=200,
                selected_cap=16,
            ).run()
            self.assertEqual(result.total_environment_transitions, 200)
            self.assertEqual(
                result.stop_reason, "MAX_FACTUAL_ENVIRONMENT_TRANSITIONS"
            )
            self.assertEqual(result.started_episode_count, 1)
            self.assertEqual(result.completed_episode_count, 0)

    def test_selected_cap_starts_no_extra_decision_and_sidecar_join_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, config, checkpoint, actor = make_actor_only_fixture(root)
            runner = make_runner(
                root,
                config,
                checkpoint,
                actor,
                factual_cap=20,
                selected_cap=1,
            )
            result = runner.run()
            self.assertEqual(result.selected_oracle_decisions, 1)
            self.assertEqual(result.valid_oracle_decisions, 1)
            self.assertEqual(result.stop_reason, "TARGET_SELECTED_DECISIONS")
            summary = inspect_route_oracle_artifact(
                config, path=result.sidecar_path
            )
            self.assertEqual(summary["decision_count"], 1)
            self.assertEqual(summary["valid_decision_count"], 1)
            self.assertEqual(summary["invalid_branch_count"], 0)
            self.assertGreaterEqual(summary["branch_count"], 2)

    def test_shadow_exhaustion_does_not_count_or_write_partial_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, config, checkpoint, actor = make_actor_only_fixture(root)
            runner = make_runner(
                root,
                config,
                checkpoint,
                actor,
                factual_cap=20,
                selected_cap=1,
                shadow_cap=1,
            )
            with self.assertRaises(OracleBudgetExhausted):
                runner.run()
            self.assertEqual(runner.oracle_budget.selected_decisions, 0)
            summary = inspect_route_oracle_artifact(
                config, path=runner.sidecar_path
            )
            self.assertEqual(summary["decision_count"], 0)
            self.assertEqual(summary["branch_count"], 0)

    def test_episode_boundary_resets_hidden_and_continues_policy_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, config, checkpoint, actor = make_actor_only_fixture(
                root,
                horizon=2,
                training_budget=4,
                arrivals=(0.0, 0.0),
            )
            runner = make_runner(
                root, config, checkpoint, actor, factual_cap=3
            )
            hidden_inputs = []
            original = runner.factual_actor_runtime.sample_step

            def capture(observations, hidden, *, episode_start):
                hidden_inputs.append(
                    (episode_start, hidden.detach().cpu().clone())
                )
                return original(
                    observations, hidden, episode_start=episode_start
                )

            runner.factual_actor_runtime.sample_step = capture
            result = runner.run()
            self.assertEqual(result.started_episode_count, 2)
            self.assertEqual(result.completed_episode_count, 1)
            self.assertEqual([item[0] for item in hidden_inputs], [True, False, True])
            self.assertEqual(torch.count_nonzero(hidden_inputs[0][1]).item(), 0)
            self.assertEqual(torch.count_nonzero(hidden_inputs[2][1]).item(), 0)

    def test_factual_actions_environment_and_hidden_match_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, config, checkpoint, actor = make_actor_only_fixture(
                root,
                horizon=2,
                training_budget=4,
                arrivals=(0.0, 0.0),
            )
            runner_factory = RecordingFactory()
            trainer_factory = RecordingFactory()
            runner = make_runner(
                root,
                config,
                checkpoint,
                actor,
                factual_cap=4,
                environment_factory=runner_factory,
            )
            trainer = CAGATMAPPOTrainer(
                config,
                actor=copy.deepcopy(actor),
                environment_factory=trainer_factory,
                progress_logger=lambda _message: None,
                oracle_budget=OracleCollectionBudget(
                    max_factual_environment_transitions=4,
                    max_shadow_transitions=6_000,
                ),
            )
            runner_hidden = []
            trainer_hidden = []
            runner_sample = runner.factual_actor_runtime.sample_step
            trainer_sample = trainer.factual_actor_runtime.sample_step

            def capture_runner(observations, hidden, *, episode_start):
                result = runner_sample(
                    observations, hidden, episode_start=episode_start
                )
                runner_hidden.append(tensor_digest(result.action_output.hidden_out))
                return result

            def capture_trainer(observations, hidden, *, episode_start):
                result = trainer_sample(
                    observations, hidden, episode_start=episode_start
                )
                trainer_hidden.append(tensor_digest(result.action_output.hidden_out))
                return result

            runner.factual_actor_runtime.sample_step = capture_runner
            trainer.factual_actor_runtime.sample_step = capture_trainer
            diagnostic_result = runner.run()
            training_result = trainer.train()
            self.assertEqual(diagnostic_result.ppo_update_count, 0)
            self.assertEqual(training_result.ppo_update_count, 0)
            self.assertEqual(runner_hidden, trainer_hidden)
            self.assertEqual(
                [item for env in runner_factory.environments for item in env.factual_actions],
                [item for env in trainer_factory.environments for item in env.factual_actions],
            )
            self.assertEqual(
                [
                    item
                    for env in runner_factory.environments
                    for item in env.factual_state_fingerprints
                ],
                [
                    item
                    for env in trainer_factory.environments
                    for item in env.factual_state_fingerprints
                ],
            )
            self.assertEqual(
                [
                    item
                    for env in runner_factory.environments
                    for item in env.factual_rng_fingerprints
                ],
                [
                    item
                    for env in trainer_factory.environments
                    for item in env.factual_rng_fingerprints
                ],
            )


class DiagnosticIdentityTests(unittest.TestCase):
    def test_identity_uses_only_commit_and_tracked_diff(self):
        with patch(
            "src.evaluation.actor_only_route_oracle._git_output",
            side_effect=(
                b"D:/My_project_li\n",
                b"3be72455f1b9372fd4021cd4312b88d6cebbb010\n",
                b"tracked diff only",
            ),
        ) as git_output:
            identity = _tracked_git_identity("D:/My_project_li")
        self.assertEqual(
            identity,
            _TrackedGitIdentity(
                commit="3be72455f1b9372fd4021cd4312b88d6cebbb010",
                tracked_diff_sha256=__import__("hashlib").sha256(
                    b"tracked diff only"
                ).hexdigest(),
            ),
        )
        diff_arguments = git_output.call_args_list[2].args[0]
        self.assertNotIn("status", diff_arguments)
        self.assertIn(":(exclude)knowledge/papers/1.pdf", diff_arguments)
        self.assertIn(":(exclude)knowledge/papers/7121.pdf", diff_arguments)

    def test_tracked_diff_digest_changes_diagnostic_id(self):
        base = {
            "runner_schema_version": 1,
            "current_git_commit": "3be72455f1b9372fd4021cd4312b88d6cebbb010",
            "tracked_worktree_diff_sha256": "0" * 64,
        }
        changed = dict(base)
        changed["tracked_worktree_diff_sha256"] = "1" * 64
        self.assertNotEqual(
            ActorOnlyRouteOracleRunner._diagnostic_id(base),
            ActorOnlyRouteOracleRunner._diagnostic_id(changed),
        )


if __name__ == "__main__":
    unittest.main()
