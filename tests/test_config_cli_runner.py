"""Unit tests for the Config/CLI/Runner implementation wave."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cli import main
from src.config import ConfigError, load_run_config
from src.runner import Runner


class ConfigTests(unittest.TestCase):
    def test_defaults_and_scenario_derivation(self) -> None:
        config = load_run_config()
        self.assertEqual(config.config_version, "section4.v1")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.environment.uav_count, 4)
        self.assertEqual(config.environment.ru_bandwidth_hz, 1_000_000.0)
        self.assertEqual(config.derived_stream_ids["task_arrival"], 20)

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
    def test_unavailable_runner_does_not_fabricate_artifacts(self) -> None:
        config = load_run_config(cli_overrides={"mode": "random"})
        result = Runner().run(config)
        self.assertEqual(result.status, "unavailable")
        for artifact in config.artifact_paths().values():
            self.assertFalse(Path(artifact).exists())

    def test_direct_cli_uses_common_runner(self) -> None:
        output: list[str] = []
        status = main(["--mode", "random", "--seed", "7"], output_fn=output.append)
        self.assertEqual(status, 0)
        self.assertTrue(any("mode=random" in line for line in output))
        self.assertTrue(any("status=unavailable" in line for line in output))

    def test_real_no_argument_path_can_exit_from_menu(self) -> None:
        answers = iter(["0"])
        output: list[str] = []
        status = main([], input_fn=lambda _prompt: next(answers), output_fn=output.append)
        self.assertEqual(status, 0)
        self.assertTrue(any("U2U-MEC Experiment Launcher" in line for line in output))


if __name__ == "__main__":
    unittest.main()
