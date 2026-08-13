"""Unit tests for the Config/CLI/Runner implementation wave."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cli import MENU_ROUTES, main
from src.config import ConfigError, load_run_config
from src.registry import build_default_registry
from src.runner import Runner


class ConfigTests(unittest.TestCase):
    def test_defaults_and_scenario_derivation(self) -> None:
        config = load_run_config()
        self.assertEqual(config.config_version, "section4.v1")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.environment.uav_count, 4)
        self.assertEqual(config.environment.ru_bandwidth_hz, 1_000_000.0)
        self.assertEqual(config.derived_stream_ids["task_arrival"], 20)
        runtime_versions = config.snapshot_dict()["_metadata"]["runtime_versions"]
        self.assertEqual(set(runtime_versions), {"python", "numpy", "torch", "cuda", "miniconda"})
        for value in runtime_versions.values():
            self.assertNotIn("TO VERIFY", value)

        medium = load_run_config(cli_overrides={"scenario_id": "medium"})
        self.assertEqual(medium.environment.uav_count, 6)
        self.assertEqual(len(medium.environment.profile_assignment), 6)

    def test_file_cli_and_interactive_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"mode": "random", "seed": 9, "scenario_id": "medium"}),
                encoding="utf-8",
            )
            config = load_run_config(
                path,
                cli_overrides={"seed": 11},
                interactive_overrides={"scenario_id": "large"},
            )
        self.assertEqual(config.mode, "random")
        self.assertEqual(config.method_id, "random")
        self.assertEqual(config.seed, 11)
        self.assertEqual(config.scenario_id, "large")
        self.assertEqual(config.environment.uav_count, 8)
        self.assertEqual(config.source_config_path, str(path))

    def test_unknown_field_is_rejected_before_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"not_a_field": 1}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_run_config(path)


class RunnerAndCliTests(unittest.TestCase):
    def test_local_only_baseline_config_registry_and_menu(self) -> None:
        default = load_run_config(cli_overrides={"mode": "baseline"})
        explicit = load_run_config(cli_overrides={
            "mode": "baseline",
            "method_id": "local_only",
        })
        qmix = load_run_config(cli_overrides={
            "mode": "baseline",
            "method_id": "factorized_action_gat_qmix",
        })
        registry = build_default_registry()
        self.assertEqual(default.method_id, "local_only")
        self.assertEqual(explicit.method_id, "local_only")
        self.assertEqual(MENU_ROUTES["5"], ("baseline", "local_only"))
        self.assertEqual(
            registry.resolve("baseline", "local_only").__name__,
            "local_only_rollout_handler",
        )
        self.assertEqual(Runner().run(qmix).status, "unavailable")
        self.assertEqual(
            registry.resolve("random", "random").__name__,
            "random_rollout_handler",
        )
        self.assertEqual(
            registry.resolve("heuristic", "heuristic").__name__,
            "heuristic_rollout_handler",
        )

    def test_direct_and_menu_local_only_paths_use_common_runner(self) -> None:
        direct_output: list[str] = []
        direct_status = main([
            "--mode", "baseline",
            "--method-id", "local_only",
            "--scenario-id", "small",
            "--seed", "42",
            "--show-config",
        ], output_fn=direct_output.append)
        self.assertEqual(direct_status, 0)
        direct_config = json.loads(direct_output[-1])
        self.assertEqual(direct_config["mode"], "baseline")
        self.assertEqual(direct_config["method_id"], "local_only")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "local_only.json"
            path.write_text(json.dumps({
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
            answers = iter(("5", "small", str(path), "42"))
            interactive_output: list[str] = []
            interactive_status = main(
                [],
                input_fn=lambda _prompt: next(answers),
                output_fn=interactive_output.append,
            )
        self.assertEqual(interactive_status, 0)
        self.assertTrue(any(
            "mode=baseline, method_id=local_only" in line
            for line in interactive_output
        ))
        self.assertTrue(any(
            "status=completed" in line for line in interactive_output
        ))

    def test_unavailable_runner_does_not_fabricate_artifacts(self) -> None:
        config = load_run_config(cli_overrides={
            "mode": "rl",
            "method_id": "ca_gat_mappo",
        })
        result = Runner().run(config)
        self.assertEqual(result.status, "unavailable")
        self.assertIn("not implemented", result.message)
        for artifact in config.artifact_paths().values():
            self.assertFalse(Path(artifact).exists())

    def test_direct_cli_uses_common_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "random.json"
            path.write_text(json.dumps({
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
            output: list[str] = []
            status = main(
                ["--mode", "random", "--config", str(path), "--seed", "7"],
                output_fn=output.append,
            )
            self.assertEqual(status, 0)
            self.assertTrue(any("mode=random" in line for line in output))
            self.assertTrue(any("status=completed" in line for line in output))
            self.assertTrue(any("raw_metrics.jsonl" in line for line in output))

    def test_real_no_argument_path_can_exit_from_menu(self) -> None:
        answers = iter(["0"])
        output: list[str] = []
        status = main([], input_fn=lambda _prompt: next(answers), output_fn=output.append)
        self.assertEqual(status, 0)
        self.assertTrue(any("U2U-MEC Experiment Launcher" in line for line in output))


if __name__ == "__main__":
    unittest.main()
