from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.artifacts import ArtifactConflictError, atomic_write_text
from src.cli import (
    build_arg_parser,
    build_execution_context_from_args,
    build_run_config_from_args,
    main as cli_main,
)
from src.config import (
    CHECKPOINT_KIND_FINAL_COMPLETED,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    ConfigError,
    RunConfig,
)
from src.evaluation.actor_loader import (
    EvaluationCheckpointError,
    checkpoint_sha256,
    load_actor_for_evaluation,
    load_final_actor_for_evaluation,
)
from src.evaluation.metrics import FORMAL_METRIC_FIELDS
from src.evaluation.protocol import (
    EvaluationCheckpoint,
    EvaluationMethod,
    EvaluationProtocol,
    EvaluationProtocolError,
    InferenceContract,
)
from src.evaluation.runner import (
    FAILED_RUN_MARKER_FILENAME,
    FORMAL_METHOD_SUITE,
    EvaluationWorkspaceLayout,
    FormalEvaluationError,
    FormalEvaluationRunner,
)
from src.execution import ExecutionContext, validate_execution_context
from src.models.ca_gat_mappo_checkpoint import (
    atomic_save_checkpoint,
    load_checkpoint_payload,
)
from src.registry import RunResult, build_default_registry
from tests.test_ca_gat_mappo_checkpoint import make_checkpoint_config, make_payload


FORMAL_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "formal_validation.json"
)


def _test_protocol(
    source: RunConfig,
    *,
    selected_filename: str,
    selected_kind: str,
    selected_step: int,
    selected_sha256: str,
    evaluation_seeds: tuple[int, ...],
) -> EvaluationProtocol:
    hashes = {
        "step_100000.pt": "1" * 64,
        "step_200000.pt": "2" * 64,
        "final.pt": "3" * 64,
    }
    hashes[selected_filename] = selected_sha256
    if len(set(hashes.values())) != 3:
        hashes["step_100000.pt"] = "4" * 64
    kinds = {
        "step_100000.pt": CHECKPOINT_KIND_PERIODIC_RESUME,
        "step_200000.pt": CHECKPOINT_KIND_PERIODIC_RESUME,
        "final.pt": CHECKPOINT_KIND_FINAL_COMPLETED,
    }
    steps = {"step_100000.pt": 4, "step_200000.pt": 4, "final.pt": 8}
    kinds[selected_filename] = selected_kind
    steps[selected_filename] = selected_step
    return EvaluationProtocol(
        evaluation_phase="validation",
        evaluation_protocol_version="validation_gate_v1",
        source_run_id=source.run_id,
        source_training_seed=source.seed,
        source_training_git_commit=source.git_commit,
        source_training_config_hash=source.config_hash,
        scenario_id=source.scenario_id,
        episode_horizon_slots=source.environment.episode_horizon,
        validation_seeds=evaluation_seeds,
        methods=tuple(
            EvaluationMethod(method_id=method_id, display_name=display_name)
            for method_id, display_name in (
                ("ca_gat_mappo", "CA-GAT-MAPPO"),
                ("heuristic", "Greedy Heuristic"),
                ("local_only", "Local-only"),
                ("random", "Random"),
            )
        ),
        checkpoints=tuple(
            EvaluationCheckpoint(
                filename=filename,
                checkpoint_kind=kinds[filename],
                checkpoint_step=steps[filename],
                expected_sha256=hashes[filename],
            )
            for filename in ("step_100000.pt", "step_200000.pt", "final.pt")
        ),
        inference_contract=InferenceContract(
            actor_eval=True,
            torch_no_grad=True,
            action_selection="deterministic_masked_argmax",
            gru_hidden_reset_scope="episode",
            network_updates=False,
        ),
    )


def make_evaluation_fixture(
    root: Path,
    *,
    filename: str = "final.pt",
    kind: str = CHECKPOINT_KIND_FINAL_COMPLETED,
    evaluation_seeds: tuple[int, ...] = (1042,),
    arrivals: tuple[float, ...] | None = None,
    actor_seed: int = 7,
):
    source_root = root / "My_project_li"
    evaluator_root = root / "My_project_li_validation_gate_v1"
    source_root.mkdir(parents=True)
    evaluator_root.mkdir(parents=True)
    source = make_checkpoint_config(source_root, horizon=4, interval=4, budget=8)
    if arrivals is not None:
        source = replace(
            source,
            environment=replace(source.environment, arrival_probabilities=arrivals),
        )
    object.__setattr__(source, "git_branch", "test-training")
    object.__setattr__(source, "git_commit", "a" * 40)
    object.__setattr__(source, "git_dirty", False)
    torch.manual_seed(actor_seed)
    payload, source_actor, _critic, _updater = make_payload(source, kind)
    step = int(payload["training_state"]["collected_environment_transitions"])
    checkpoint_dir = source_root / "logs" / source.run_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / filename
    atomic_save_checkpoint(payload, checkpoint)
    expected_hash = checkpoint_sha256(checkpoint)
    evaluator = replace(
        source,
        mode="evaluation",
        method_id="ca_gat_mappo",
        launch_profile=None,
    )
    object.__setattr__(evaluator, "git_branch", "codex/validation-gate-v1-test")
    object.__setattr__(evaluator, "git_commit", "b" * 40)
    object.__setattr__(evaluator, "git_dirty", False)
    evaluator.validate()
    protocol = _test_protocol(
        source,
        selected_filename=filename,
        selected_kind=kind,
        selected_step=step,
        selected_sha256=expected_hash,
        evaluation_seeds=evaluation_seeds,
    )
    return SimpleNamespace(
        source=source,
        evaluator=evaluator,
        checkpoint=checkpoint,
        source_actor=source_actor,
        protocol=protocol,
        layout=EvaluationWorkspaceLayout(evaluator_root, source_root),
        payload=payload,
    )


def protocol_with_checkpoint_hash(
    protocol: EvaluationProtocol,
    filename: str,
    digest: str,
) -> EvaluationProtocol:
    return replace(
        protocol,
        checkpoints=tuple(
            replace(item, expected_sha256=digest)
            if item.filename == filename
            else item
            for item in protocol.checkpoints
        ),
    )


def rewrite_payload_snapshot(fixture, mutator) -> None:
    payload = load_checkpoint_payload(fixture.checkpoint)
    mutator(payload["config_snapshot"])
    resolved = {
        key: value
        for key, value in payload["config_snapshot"].items()
        if key != "_metadata"
    }
    canonical = json.dumps(
        resolved,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["config_hash"] = digest
    payload["config_snapshot"]["_metadata"]["config_hash"] = digest
    torch.save(payload, fixture.checkpoint)
    fixture.protocol = protocol_with_checkpoint_hash(
        fixture.protocol,
        fixture.checkpoint.name,
        checkpoint_sha256(fixture.checkpoint),
    )


def make_runner(fixture, *, environment_factory=None) -> FormalEvaluationRunner:
    return FormalEvaluationRunner(
        fixture.evaluator,
        fixture.checkpoint,
        evaluation_device="cpu",
        protocol=fixture.protocol,
        workspace_layout=fixture.layout,
        environment_factory=environment_factory,
    )


class TestEvaluationProtocol(unittest.TestCase):
    def test_formal_protocol_is_exact_and_machine_independent(self) -> None:
        protocol = EvaluationProtocol.from_path(FORMAL_PROTOCOL_PATH)
        self.assertEqual(protocol.validation_seeds, (1042, 1043, 1044, 1045, 1046))
        self.assertEqual(protocol.method_ids, FORMAL_METHOD_SUITE)
        self.assertEqual(protocol.scenario_id, "small")
        self.assertEqual(protocol.episode_horizon_slots, 500)
        self.assertEqual(
            protocol.source_run_id,
            "ca_gat_mappo__small__seed-42__cfg-e819fe0b2d47__git-f630ec5d0c2b",
        )
        self.assertNotIn("D:\\", protocol.canonical_json)
        expected = {
            "step_100000.pt": "476061dba7689452495974f8bef68438428bb0ff3142c356bafe221f1733f5a0",
            "step_200000.pt": "b53ebbcd935a7b62a4cac3bd75c736dcd8ed923894ab9fb7a7da21d9d4c883ab",
            "final.pt": "df8582648aa3b3dce16711fee231d082084ed428e1c1e43129acdc6f5a582f5f",
        }
        self.assertEqual(
            {item.filename: item.expected_sha256 for item in protocol.checkpoints},
            expected,
        )
        self.assertEqual(len(set(expected.values())), 3)
        self.assertTrue(all(len(value) == 64 for value in expected.values()))
        self.assertEqual(len(protocol.sha256), 64)

    def test_protocol_does_not_change_training_run_config_hash(self) -> None:
        config = RunConfig()
        before = config.config_hash
        EvaluationProtocol.from_path(FORMAL_PROTOCOL_PATH)
        self.assertEqual(config.config_hash, before)
        self.assertNotIn("evaluation_protocol_version", config.snapshot_dict())

    def test_protocol_rejects_unknown_fields_and_duplicate_hashes(self) -> None:
        mapping = json.loads(FORMAL_PROTOCOL_PATH.read_text(encoding="utf-8"))
        mapping["unexpected"] = True
        with self.assertRaises(EvaluationProtocolError):
            EvaluationProtocol.from_mapping(mapping)
        mapping.pop("unexpected")
        mapping["checkpoints"][1]["expected_sha256"] = mapping["checkpoints"][0][
            "expected_sha256"
        ]
        with self.assertRaisesRegex(EvaluationProtocolError, "distinct"):
            EvaluationProtocol.from_mapping(mapping)


class TestEvaluationExecutionContext(unittest.TestCase):
    def test_direct_cli_keeps_checkpoint_path_outside_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final.pt"
            checkpoint.write_bytes(b"synthetic")
            args = build_arg_parser().parse_args(
                [
                    "--mode",
                    "evaluation",
                    "--method-id",
                    "ca_gat_mappo",
                    "--evaluate-from",
                    str(checkpoint),
                    "--evaluation-device",
                    "cpu",
                ]
            )
            config = build_run_config_from_args(args)
            context = build_execution_context_from_args(args)
            self.assertEqual(context.evaluate_from, checkpoint)
            self.assertNotIn("evaluate_from", config.snapshot_dict())

    def test_cli_rejects_validation_scenario_config_seed_and_set(self) -> None:
        parser = build_arg_parser()
        for extra in (
            ("--scenario-id", "medium"),
            ("--seed", "43"),
            ("--set", "environment.episode_horizon=4"),
            ("--config", "configs/default.json"),
        ):
            with self.subTest(extra=extra):
                args = parser.parse_args(
                    [
                        "--mode",
                        "evaluation",
                        "--method-id",
                        "ca_gat_mappo",
                        *extra,
                    ]
                )
                with self.assertRaises(ConfigError):
                    build_run_config_from_args(args)

    def test_resume_and_evaluate_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"checkpoint")
            config = build_run_config_from_args(
                build_arg_parser().parse_args(
                    ["--mode", "evaluation", "--method-id", "ca_gat_mappo"]
                )
            )
            with self.assertRaisesRegex(ConfigError, "mutually exclusive"):
                validate_execution_context(
                    config,
                    ExecutionContext.from_paths(
                        resume_from=path,
                        evaluate_from=path,
                        evaluation_device="cpu",
                    ),
                )

    def test_interactive_evaluation_skips_training_config_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final.pt"
            checkpoint.write_bytes(b"synthetic")
            answers = iter(("7", str(checkpoint), "cpu"))
            completed = RunResult(
                status="completed",
                run_id="evaluation-id",
                mode="evaluation",
                method_id="ca_gat_mappo",
                message="stubbed",
            )
            with patch("src.cli.Runner.run", return_value=completed) as run:
                status = cli_main(
                    [],
                    input_fn=lambda _prompt: next(answers),
                    output_fn=lambda _line: None,
                )
            self.assertEqual(status, 0)
            context = run.call_args.kwargs["execution_context"]
            self.assertEqual(context.evaluate_from, checkpoint)

    def test_registry_uses_formal_handler(self) -> None:
        self.assertEqual(
            build_default_registry().resolve(
                "evaluation", "ca_gat_mappo"
            ).__name__,
            "formal_evaluation_handler",
        )


class TestActorOnlyLoader(unittest.TestCase):
    def test_periodic_and_final_load_actor_only(self) -> None:
        for filename, kind in (
            ("step_100000.pt", CHECKPOINT_KIND_PERIODIC_RESUME),
            ("final.pt", CHECKPOINT_KIND_FINAL_COMPLETED),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                fixture = make_evaluation_fixture(
                    Path(directory), filename=filename, kind=kind
                )
                selected = fixture.protocol.checkpoint_for_filename(filename)
                with patch(
                    "src.models.ca_gat_mappo_checkpoint.restore_model_optimizer_and_rng"
                ) as restore_all, patch(
                    "src.models.ca_gat_mappo_checkpoint.restore_policy_rng"
                ) as restore_rng, patch(
                    "src.models.ca_gat_mappo_checkpoint.restore_active_rollout"
                ) as restore_rollout, patch.object(
                    torch.optim.Optimizer,
                    "step",
                    side_effect=AssertionError("optimizer.step forbidden"),
                ):
                    loaded = load_actor_for_evaluation(
                        fixture.checkpoint,
                        protocol=fixture.protocol,
                        expected_checkpoint=selected,
                        evaluation_device="cpu",
                    )
                restore_all.assert_not_called()
                restore_rng.assert_not_called()
                restore_rollout.assert_not_called()
                self.assertFalse(loaded.actor.training)
                self.assertEqual(loaded.source_checkpoint_kind, kind)
                self.assertEqual(
                    loaded.checkpoint_expected_sha256,
                    loaded.checkpoint_sha256_before,
                )
                self.assertEqual(
                    loaded.checkpoint_sha256_before,
                    loaded.checkpoint_sha256_after_actor_load,
                )
                for name, value in fixture.source_actor.state_dict().items():
                    self.assertTrue(
                        torch.equal(value.cpu(), loaded.actor.state_dict()[name].cpu())
                    )

    def test_legacy_final_wrapper_still_loads_and_rejects_periodic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory))
            loaded = load_final_actor_for_evaluation(
                fixture.evaluator,
                fixture.checkpoint,
                evaluation_device="cpu",
            )
            self.assertFalse(loaded.actor.training)
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(
                Path(directory),
                filename="step_100000.pt",
                kind=CHECKPOINT_KIND_PERIODIC_RESUME,
            )
            with self.assertRaisesRegex(EvaluationCheckpointError, "FINAL_COMPLETED"):
                load_final_actor_for_evaluation(
                    fixture.evaluator,
                    fixture.checkpoint,
                    evaluation_device="cpu",
                )

    def test_expected_sha_mismatch_fails_before_payload_and_actor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory))
            protocol = protocol_with_checkpoint_hash(
                fixture.protocol, fixture.checkpoint.name, "0" * 64
            )
            selected = protocol.checkpoint_for_filename(fixture.checkpoint.name)
            with patch(
                "src.evaluation.actor_loader.load_checkpoint_payload"
            ) as payload_load, patch(
                "src.evaluation.actor_loader.CAGATMAPPOActor"
            ) as actor_constructor:
                with self.assertRaisesRegex(EvaluationCheckpointError, "allowlist"):
                    load_actor_for_evaluation(
                        fixture.checkpoint,
                        protocol=protocol,
                        expected_checkpoint=selected,
                        evaluation_device="cpu",
                    )
            payload_load.assert_not_called()
            actor_constructor.assert_not_called()

    def test_wrong_source_path_is_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory))
            copied = fixture.layout.evaluator_root / "final.pt"
            copied.write_bytes(fixture.checkpoint.read_bytes())
            with self.assertRaisesRegex(FormalEvaluationError, "fixed source run"):
                FormalEvaluationRunner(
                    fixture.evaluator,
                    copied,
                    evaluation_device="cpu",
                    protocol=fixture.protocol,
                    workspace_layout=fixture.layout,
                )

    def test_scenario_horizon_and_config_hash_mismatch_fail(self) -> None:
        cases = (
            (
                lambda snapshot: snapshot.__setitem__("scenario_id", "medium"),
                "scenario_id",
            ),
            (
                lambda snapshot: snapshot["environment"].__setitem__(
                    "episode_horizon", 5
                ),
                "episode horizon",
            ),
        )
        for mutator, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                fixture = make_evaluation_fixture(Path(directory))
                rewrite_payload_snapshot(fixture, mutator)
                selected = fixture.protocol.checkpoint_for_filename("final.pt")
                with self.assertRaisesRegex(EvaluationCheckpointError, message):
                    load_actor_for_evaluation(
                        fixture.checkpoint,
                        protocol=fixture.protocol,
                        expected_checkpoint=selected,
                        evaluation_device="cpu",
                    )

        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory))
            payload = load_checkpoint_payload(fixture.checkpoint)
            payload["config_snapshot"]["seed"] = 43
            torch.save(payload, fixture.checkpoint)
            fixture.protocol = protocol_with_checkpoint_hash(
                fixture.protocol,
                "final.pt",
                checkpoint_sha256(fixture.checkpoint),
            )
            with self.assertRaisesRegex(EvaluationCheckpointError, "canonical snapshot"):
                load_actor_for_evaluation(
                    fixture.checkpoint,
                    protocol=fixture.protocol,
                    expected_checkpoint=fixture.protocol.checkpoint_for_filename(
                        "final.pt"
                    ),
                    evaluation_device="cpu",
                )


class TestFormalEvaluationRunner(unittest.TestCase):
    def test_repeatable_no_grad_masked_argmax_hidden_reset_and_fairness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(
                Path(directory), evaluation_seeds=(1042, 1043)
            )
            runner = make_runner(fixture)
            deterministic = runner.action_distribution.deterministic_actions

            def checked_deterministic(*args, **kwargs):
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(runner.loaded_actor.actor.training)
                return deterministic(*args, **kwargs)

            with patch.object(
                runner.action_distribution,
                "deterministic_actions",
                side_effect=checked_deterministic,
            ) as deterministic_call, patch.object(
                torch.Tensor,
                "backward",
                side_effect=AssertionError("backward forbidden"),
            ), patch.object(
                torch.optim.Optimizer,
                "step",
                side_effect=AssertionError("optimizer.step forbidden"),
            ):
                first = runner.run(write_artifacts=False)
            second = runner.run(write_artifacts=False)
            self.assertGreater(deterministic_call.call_count, 0)
            self.assertEqual(first.aggregate_metrics, second.aggregate_metrics)
            self.assertEqual(len(first.episodes), 2 * len(FORMAL_METHOD_SUITE))
            for seed in fixture.protocol.validation_seeds:
                selected = [item for item in first.episodes if item.evaluation_seed == seed]
                self.assertEqual(len({item.external_trace_sha256 for item in selected}), 1)
            actor_episodes = [
                item for item in first.episodes if item.method_id == "ca_gat_mappo"
            ]
            self.assertEqual(
                actor_episodes[0].hidden_trace_sha256[0],
                actor_episodes[1].hidden_trace_sha256[0],
            )
            self.assertFalse(runner.loaded_actor.actor.training)

    def test_complete_metric_schema_and_na_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory), arrivals=(0.0,))
            result = make_runner(fixture).run(write_artifacts=False)
            for episode in result.episodes:
                self.assertEqual(tuple(episode.metrics), FORMAL_METRIC_FIELDS)
                self.assertEqual(episode.metrics["generated"], 0)
                self.assertIsNone(episode.metrics["completion_ratio"])
                self.assertIsNone(episode.metrics["completed_task_e2e_delay_s"])
                self.assertIsNone(episode.metrics["energy_per_completed_task_j"])
            for method_id in FORMAL_METHOD_SUITE:
                stats = result.aggregate_metrics["methods"][method_id]["metrics"]
                self.assertEqual(
                    stats["completed_task_e2e_delay_s"]["valid_sample_count"], 0
                )

    def test_final_sha_and_trace_are_checked_before_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory))
            runner = make_runner(fixture)
            real_hash = checkpoint_sha256

            def checked_hash(path):
                self.assertFalse(runner.run_directory().exists())
                return real_hash(path)

            with patch("src.evaluation.runner.checkpoint_sha256", side_effect=checked_hash):
                result = runner.run(write_artifacts=True)
            self.assertEqual(len(result.artifacts), 5)
            self.assertTrue(runner.run_directory().is_dir())
            self.assertEqual(Path(result.artifacts[-1]).name, "evaluation_manifest.json")
            manifest = json.loads(Path(result.artifacts[-1]).read_text(encoding="utf-8"))
            hashes = {
                manifest["checkpoint_expected_sha256"],
                manifest["checkpoint_sha256_before"],
                manifest["checkpoint_sha256_after_actor_load"],
                manifest["checkpoint_sha256_after_evaluation"],
                manifest["checkpoint_sha256_after"],
            }
            self.assertEqual(len(hashes), 1)
            self.assertEqual(manifest["source_training_seed"], 42)
            self.assertEqual(manifest["evaluation_phase"], "validation")
            self.assertEqual(manifest["episode_horizon_slots"], 4)
            self.assertEqual(manifest["run_status"], "completed")
            snapshot = json.loads(Path(result.artifacts[0]).read_text(encoding="utf-8"))
            self.assertEqual(snapshot, fixture.protocol.canonical_dict())
            self.assertTrue(
                all(
                    Path(path).is_relative_to(fixture.layout.evaluator_root)
                    for path in result.artifacts
                )
            )

    def test_checkpoint_or_trace_failure_creates_no_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory))
            runner = make_runner(fixture)
            with patch("src.evaluation.runner.checkpoint_sha256", return_value="0" * 64):
                with self.assertRaisesRegex(FormalEvaluationError, "SHA-256"):
                    runner.run(write_artifacts=True)
            self.assertFalse(runner.run_directory().exists())
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory))
            runner = make_runner(fixture)
            with patch.object(
                runner,
                "_validate_external_traces",
                side_effect=FormalEvaluationError("trace mismatch"),
            ):
                with self.assertRaisesRegex(FormalEvaluationError, "trace mismatch"):
                    runner.run(write_artifacts=True)
            self.assertFalse(runner.run_directory().exists())

    def test_existing_or_failed_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory))
            calls = 0

            def environment_factory(run_config):
                nonlocal calls
                calls += 1
                from src.env.environment import U2UMECEnvironment

                return U2UMECEnvironment(run_config)

            runner = make_runner(fixture, environment_factory=environment_factory)
            runner.run_directory().mkdir(parents=True)
            with self.assertRaises(ArtifactConflictError):
                runner.run(write_artifacts=True)
            self.assertEqual(calls, 0)

        with tempfile.TemporaryDirectory() as directory:
            fixture = make_evaluation_fixture(Path(directory))
            runner = make_runner(fixture)

            def failing_write(path, content):
                if Path(path).name == "episode_metrics.jsonl":
                    raise OSError("synthetic publication failure")
                return atomic_write_text(path, content)

            with patch("src.evaluation.runner.atomic_write_text", side_effect=failing_write):
                with self.assertRaisesRegex(FormalEvaluationError, "publication failed"):
                    runner.run(write_artifacts=True)
            marker = runner.run_directory() / FAILED_RUN_MARKER_FILENAME
            self.assertTrue(marker.is_file())
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["run_status"],
                "failed",
            )
            with self.assertRaises(ArtifactConflictError):
                runner.run(write_artifacts=True)

    def test_checkpoint_outputs_are_isolated_and_cross_checkpoint_traces_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = (
                make_evaluation_fixture(
                    root / "first",
                    filename="step_100000.pt",
                    kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                    actor_seed=1,
                ),
                make_evaluation_fixture(
                    root / "second",
                    filename="step_200000.pt",
                    kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                    actor_seed=2,
                ),
                make_evaluation_fixture(root / "third", actor_seed=3),
            )
            results = tuple(make_runner(item).run(write_artifacts=False) for item in fixtures)
            self.assertEqual(len({item.evaluation_run_id for item in results}), 3)
            trace_maps = tuple(
                item.manifest["external_trace_sha256_by_seed"] for item in results
            )
            self.assertTrue(all(value == trace_maps[0] for value in trace_maps[1:]))


if __name__ == "__main__":
    unittest.main()
