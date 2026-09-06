"""Gates for Matched Short Training Infrastructure V1.

All metric rows in this test module are explicitly synthetic fixtures; they
are never written to repository experiment directories.
"""

from __future__ import annotations

import copy
import json
import math
import random
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.config import RouteDecoderMode, RunConfig
from src.matched_short_analysis import (
    analyze_matched_triplet,
    detect_baseline_clear_collapse,
    evaluate_candidate_final,
    evaluate_stop,
    kpi_regression_gate,
    late_window,
    load_live_jsonl,
    plot_matched_timeseries,
    select_candidate,
    symmetric_degradation,
    trailing_mean,
    warmup_window,
)
from src.matched_short_diagnostics import (
    LiveTrainingDiagnosticsWriter,
    build_update_records,
)
from src.matched_short_preflight import (
    DIAGNOSTIC_BUDGET,
    DIAGNOSTIC_CHECKPOINT_INTERVAL,
    EXTENSION_BUDGET,
    EXTENSION_CHECKPOINT_INTERVAL,
    FINAL_PARTIAL_ROLLOUT_BEHAVIOR,
    MATCHED_SHORT_BUDGET_SPECS,
    PERIODIC_PARTIAL_ROLLOUT_BEHAVIOR,
    SMOKE_BUDGET,
    SMOKE_CHECKPOINT_INTERVAL,
    MatchedPreflightError,
    build_group_definitions,
    capture_initial_identity,
    checkpoint_plan,
    inspect_output_collision,
    joint_lossless_boundary,
    matched_budget_plan,
    module_digest,
    next_joint_lossless_boundary,
    normalized_optimizer_digest,
    rng_digests,
    seed_global_rngs,
    synthetic_stability_gate,
    validate_exact_final_checkpoint_plan,
    validate_initial_identities,
    validate_matched_configs,
    validate_preflight_manifest,
)
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic
from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from src.models.ca_gat_mappo_update import (
    RecurrentPPOEpochDiagnostics,
    RecurrentPPOUpdateOutput,
    build_ca_gat_mappo_optimizers,
)


def matched_config(coefficient: float, *, budget: int = 4096) -> RunConfig:
    base = RunConfig()
    mappo = replace(
        base.training.mappo,
        encoder_hidden_dimension=8,
        attention_head_count=1,
        gru_hidden_dimension=8,
        training_device="cpu",
        route_decoder_mode=RouteDecoderMode.CANDIDATE_AWARE_V1,
        route_choice_stability_enabled=True,
        route_choice_entropy_floor_nats=0.20,
        route_choice_stability_coef=coefficient,
        max_training_environment_steps=budget,
        checkpoint_interval_steps=500,
    )
    config = replace(
        base,
        mode="rl",
        method_id="ca_gat_mappo",
        training=replace(base.training, formal_rl_enabled=True, mappo=mappo),
    )
    config.validate()
    return config


def triplet(*, budget: int = 4096) -> dict[str, RunConfig]:
    return {
        "baseline": matched_config(0.0, budget=budget),
        "a1": matched_config(0.02, budget=budget),
        "a2": matched_config(0.05, budget=budget),
    }


def identity_fixture() -> dict[str, dict[str, object]]:
    identity = {
        "actor_digest": "actor",
        "critic_digest": "critic",
        "optimizer_normalized_digest": "optimizer",
        "rng_digests": {
            "python": "python",
            "numpy": "numpy",
            "torch_cpu": "cpu",
            "torch_cuda": ["cuda"],
            "torch_cuda_device_count": 1,
        },
    }
    return {
        group: copy.deepcopy(identity)
        for group in ("baseline", "a1", "a2")
    }


def synthetic_epoch(index: int, *, valid_count: int, mass: float, ratio: float):
    route = SimpleNamespace(
        route_branch_active_count=4,
        route_selected_local_count=3,
        route_selected_remote_count=1,
        route_selected_defer_count=0,
        active_branch_matrix=(((True,) * 7,), ((True,) * 7,)),
        route_entropy_mean=0.4 + index * 0.01,
    )
    stability = SimpleNamespace(
        conditional_remote_mass=mass,
        remote_local_binary_entropy=0.2 + mass,
        floor_violation_fraction=0.5,
        unscaled_stability_loss=0.1 + mass,
        scaled_stability_loss=0.02 * (0.1 + mass),
        valid_sample_count=valid_count,
        remote_scorer_gradient_norm=ratio * 2.0,
        local_route_row_gradient_norm=2.0,
        gradient_norm_ratio=ratio,
    )
    return SimpleNamespace(
        epoch_index=index,
        actor_loss=1.0 + index,
        critic_loss=2.0 + index,
        approx_kl=0.01 + index * 0.001,
        clip_fraction=0.1 + index * 0.01,
        actor_grad_norm_before_clip=0.4 + index * 0.1,
        critic_grad_norm_before_clip=0.2,
        clip_max_norm=0.5,
        route_telemetry=route,
        route_choice_stability=stability,
    )


def update_fixture():
    counts = (1, 1, 3, 3)
    masses = (0.1, 0.2, 0.5, 0.7)
    ratios = (0.1, 0.3, 0.2, 0.4)
    epochs = tuple(
        synthetic_epoch(index, valid_count=counts[index], mass=masses[index], ratio=ratios[index])
        for index in range(4)
    )
    return SimpleNamespace(epoch_diagnostics=epochs)


def rollout_record(
    index: int,
    *,
    remote: int = 1,
    local_share: float = 0.8,
    remote_share: float = 0.2,
    mass: float = 0.2,
    ratio: float = 0.1,
    scaled: float = 0.01,
) -> list[dict[str, object]]:
    return [
        {
            "record_type": "rollout",
            "group": "synthetic",
            "update_index": index,
            "environment_step": (index + 1) * 256,
            "route_active_count": 10,
            "remote_count": remote,
            "local_count": 10 - remote,
            "local_share": local_share,
            "remote_share": remote_share,
        },
        {
            "record_type": "update",
            "group": "synthetic",
            "update_index": index,
            "environment_step": (index + 1) * 256,
            "stability_valid_count": 10,
            "scaled_stability_loss": scaled,
            "conditional_remote_mass": mass,
            "gradient_ratio_valid": True,
            "remote_local_grad_norm_ratio": ratio,
            "approx_kl_max": 0.01,
            "clip_fraction_max": 0.10,
            "gradient_clip_fraction": 0.0,
        },
    ]


def episode_record(index: int, *, group: str, return_value: float = 10.0):
    return {
        "record_type": "episode",
        "group": group,
        "episode_index": index,
        "environment_step": (index + 1) * 500,
        "episode_return": return_value,
        "completion_rate": 0.8,
        "expiration_rate": 0.1,
        "energy_j_per_env_step": 1.0,
    }


class MatchedIdentityGateTests(unittest.TestCase):
    def test_valid_triplet_and_initial_identities_pass(self) -> None:
        configs = triplet()
        gate = validate_matched_configs(configs)
        self.assertEqual(gate["status"], "PASS")
        identities = {
            group: capture_initial_identity(config)
            for group, config in configs.items()
        }
        identities["baseline"]["timestamp"] = "different-but-ignored"
        identities["a1"]["output_path"] = "also-ignored"
        before = rng_digests()
        self.assertEqual(validate_initial_identities(identities)["status"], "PASS")
        self.assertEqual(before, rng_digests())

    def test_extra_config_difference_fails(self) -> None:
        configs = triplet()
        a1 = configs["a1"]
        configs["a1"] = replace(a1, seed=43)
        with self.assertRaisesRegex(MatchedPreflightError, "STOP_CONFIG_DIFF"):
            validate_matched_configs(configs)

    def test_raw_actor_and_critic_digests_detect_tensor_changes(self) -> None:
        config = matched_config(0.0)
        actor = CAGATMAPPOActor(config)
        critic = MAPPOCentralizedCritic(config)
        actor_before = module_digest(actor)
        critic_before = module_digest(critic)
        with torch.no_grad():
            next(actor.parameters()).add_(1.0)
            next(critic.parameters()).add_(1.0)
        self.assertNotEqual(actor_before, module_digest(actor))
        self.assertNotEqual(critic_before, module_digest(critic))

    def test_actor_mismatch_fails_unified_identity_gate(self) -> None:
        identities = identity_fixture()
        identities["a1"]["actor_digest"] = "perturbed"
        with self.assertRaisesRegex(
            MatchedPreflightError, "STOP_INITIAL_IDENTITY: actor_digest"
        ):
            validate_initial_identities(identities)

    def test_critic_mismatch_fails_unified_identity_gate(self) -> None:
        identities = identity_fixture()
        identities["a1"]["critic_digest"] = "perturbed"
        with self.assertRaisesRegex(
            MatchedPreflightError, "STOP_INITIAL_IDENTITY: critic_digest"
        ):
            validate_initial_identities(identities)

    def test_optimizer_hyperparameter_perturbation_changes_digest(self) -> None:
        config = matched_config(0.0)
        actor = CAGATMAPPOActor(config)
        critic = MAPPOCentralizedCritic(config)
        bundle = build_ca_gat_mappo_optimizers(actor, critic, config)
        before, normalized = normalized_optimizer_digest(bundle, actor, critic)
        self.assertEqual(normalized["actor"]["state_semantics"], "canonical-empty-state")
        bundle.actor_optimizer.param_groups[0]["lr"] *= 2.0
        after, _ = normalized_optimizer_digest(bundle, actor, critic)
        self.assertNotEqual(before, after)
        identities = identity_fixture()
        identities["a2"]["optimizer_normalized_digest"] = after
        with self.assertRaisesRegex(
            MatchedPreflightError, "STOP_INITIAL_IDENTITY"
        ):
            validate_initial_identities(identities)

    def test_rng_stream_digests_change_when_consumed(self) -> None:
        seed_global_rngs(42)
        before = rng_digests()
        random.random()
        np.random.random()
        torch.rand(1)
        after = rng_digests()
        self.assertNotEqual(before["python"], after["python"])
        self.assertNotEqual(before["numpy"], after["numpy"])
        self.assertNotEqual(before["torch_cpu"], after["torch_cpu"])
        if torch.cuda.is_available():
            seed_global_rngs(42)
            cuda_before = rng_digests()["torch_cuda"]
            torch.rand(1, device="cuda")
            cuda_after = rng_digests()["torch_cuda"]
            self.assertNotEqual(cuda_before, cuda_after)

    def _assert_rng_mismatch_fails(self, field: str) -> None:
        identities = identity_fixture()
        identities["a1"]["rng_digests"][field] = ["changed"]
        with self.assertRaisesRegex(
            MatchedPreflightError,
            rf"STOP_INITIAL_IDENTITY: rng_digests\.{field}",
        ):
            validate_initial_identities(identities)

    def test_python_rng_mismatch_fails_unified_identity_gate(self) -> None:
        self._assert_rng_mismatch_fails("python")

    def test_numpy_rng_mismatch_fails_unified_identity_gate(self) -> None:
        self._assert_rng_mismatch_fails("numpy")

    def test_torch_cpu_rng_mismatch_fails_unified_identity_gate(self) -> None:
        self._assert_rng_mismatch_fails("torch_cpu")

    def test_torch_cuda_rng_mismatch_fails_unified_identity_gate(self) -> None:
        self._assert_rng_mismatch_fails("torch_cuda")

    def test_synthetic_stability_is_proportional_and_rng_neutral(self) -> None:
        result = synthetic_stability_gate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["scaled_loss"]["baseline"], 0.0)
        self.assertTrue(result["rng_unchanged"])


class LiveTelemetryTests(unittest.TestCase):
    def test_rollout_counts_are_emitted_once_and_epoch_aggregation_is_exact(self) -> None:
        records = build_update_records(
            matched_config(0.02),
            "a1",
            update_fixture(),
            update_index=7,
            environment_step=2048,
            uav_count=1,
        )
        self.assertEqual(sum(row["record_type"] == "rollout" for row in records), 1)
        self.assertEqual(sum(row["record_type"] == "ppo_epoch" for row in records), 4)
        aggregate = next(row for row in records if row["record_type"] == "update")
        expected_mass = (0.1 + 0.2 + 3 * 0.5 + 3 * 0.7) / 8
        self.assertAlmostEqual(aggregate["conditional_remote_mass"], expected_mass)
        self.assertAlmostEqual(aggregate["remote_local_grad_norm_ratio"], 0.25)
        self.assertAlmostEqual(aggregate["approx_kl_mean"], 0.0115)
        self.assertAlmostEqual(aggregate["approx_kl_max"], 0.013)
        self.assertAlmostEqual(aggregate["clip_fraction_mean"], 0.115)
        self.assertAlmostEqual(aggregate["clip_fraction_max"], 0.13)
        self.assertEqual(aggregate["gradient_clip_fraction"], 0.5)
        expected_global = sum(
            math.hypot(0.4 + index * 0.1, 0.2)
            for index in range(4)
        ) / 4
        self.assertAlmostEqual(
            aggregate["global_grad_norm_before_clip"], expected_global
        )

    def test_zero_valid_count_preserves_na(self) -> None:
        output = update_fixture()
        for epoch in output.epoch_diagnostics:
            epoch.route_choice_stability.valid_sample_count = 0
            epoch.route_choice_stability.conditional_remote_mass = None
            epoch.route_choice_stability.remote_local_binary_entropy = None
            epoch.route_choice_stability.floor_violation_fraction = None
        aggregate = build_update_records(
            matched_config(0.02), "a1", output, update_index=0, environment_step=256, uav_count=1
        )[-1]
        self.assertEqual(aggregate["stability_valid_count"], 0)
        self.assertIsNone(aggregate["conditional_remote_mass"])

    def test_partial_jsonl_survives_and_only_truncated_tail_is_ignored(self) -> None:
        config = matched_config(0.02)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.jsonl"
            writer = LiveTrainingDiagnosticsWriter(config, group="a1", path=path)
            record = build_update_records(
                config, "a1", update_fixture(), update_index=0, environment_step=256, uav_count=1
            )[0]
            writer.append(record)
            with path.open("ab") as stream:
                stream.write(b'{"incomplete":')
                stream.flush()
            loaded, ignored = load_live_jsonl(path)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(ignored)

    def test_trainer_calls_update_and_episode_observer_hooks(self) -> None:
        events = []

        class SpyWriter:
            def __init__(self) -> None:
                self.updates = []
                self.episodes = []

            def write_update(self, output, **metadata) -> None:
                events.append("write_update")
                self.updates.append((output, metadata))

            def write_episode(self, **record) -> None:
                events.append("write_episode")
                self.episodes.append(record)

        diagnostics = tuple(
            RecurrentPPOEpochDiagnostics(
                epoch_index=index,
                actor_loss=1.0,
                critic_loss=2.0,
                entropy_mean=0.5,
                total_loss=3.0,
                ratio_mean=1.0,
                actor_grad_norm_before_clip=0.4,
                critic_grad_norm_before_clip=0.5,
                clip_max_norm=0.5,
            )
            for index in range(4)
        )
        output = RecurrentPPOUpdateOutput(
            epoch_diagnostics=diagnostics,
            chunk_count=8,
            chunk_length=32,
            valid_transition_count=256,
            old_policy_snapshot_preserved=True,
        )

        class FixedUpdater:
            def update(self, _buffer, **_kwargs):
                events.append("update")
                return output

        update_spy = SpyWriter()
        update_trainer = CAGATMAPPOTrainer(
            matched_config(0.02),
            updater_factory=lambda *_: FixedUpdater(),
            matched_diagnostic_writer=update_spy,
        )
        update_trainer.rollout_buffer = SimpleNamespace(
            full=True,
            finalize=lambda: None,
            clear=lambda: None,
        )
        update_trainer._rollout_policy_version = update_trainer.policy_version
        update_trainer._perform_update([])
        self.assertEqual(events[:2], ["update", "write_update"])
        self.assertEqual(len(update_spy.updates), 1)
        self.assertEqual(update_spy.updates[0][1]["update_index"], 0)
        self.assertEqual(update_spy.updates[0][1]["environment_step"], 0)

        episode_spy = SpyWriter()
        base = matched_config(0.02, budget=501)
        episode_config = replace(
            base,
            environment=replace(base.environment, episode_horizon=1),
            training=replace(
                base.training,
                mappo=replace(
                    base.training.mappo,
                    max_training_episodes=1,
                ),
            ),
        )
        episode_config.validate()
        result = CAGATMAPPOTrainer(
            episode_config,
            matched_diagnostic_writer=episode_spy,
            progress_logger=lambda _message: None,
        ).train()
        self.assertEqual(result.completed_episode_count, 1)
        self.assertEqual(len(episode_spy.episodes), 1)
        self.assertEqual(episode_spy.episodes[0]["environment_step"], 1)
        self.assertIn("generated_task_count", episode_spy.episodes[0]["metrics"])

    def test_observer_none_is_legacy_and_mid_episode_has_no_episode_record(self) -> None:
        base = matched_config(0.02, budget=501)
        legacy_config = replace(
            base,
            environment=replace(base.environment, episode_horizon=1),
            training=replace(
                base.training,
                mappo=replace(base.training.mappo, max_training_episodes=1),
            ),
        )
        legacy_config.validate()
        result = CAGATMAPPOTrainer(
            legacy_config,
            matched_diagnostic_writer=None,
            progress_logger=lambda _message: None,
        ).train()
        self.assertEqual(result.completed_episode_count, 1)

        class SpyWriter:
            def __init__(self) -> None:
                self.episodes = []

            def write_update(self, _output, **_metadata) -> None:
                raise AssertionError("a partial rollout must not update")

            def write_episode(self, **record) -> None:
                self.episodes.append(record)

        writer = SpyWriter()
        partial_config = replace(
            base,
            environment=replace(base.environment, episode_horizon=3),
            training=replace(
                base.training,
                mappo=replace(
                    base.training.mappo,
                    max_training_environment_steps=4,
                    checkpoint_interval_steps=3,
                ),
            ),
        )
        partial_config.validate()
        partial = CAGATMAPPOTrainer(
            partial_config,
            matched_diagnostic_writer=writer,
            progress_logger=lambda _message: None,
        ).train()
        self.assertEqual(partial.completed_episode_count, 1)
        self.assertEqual(len(writer.episodes), 1)
        self.assertEqual(writer.episodes[0]["environment_step"], 3)

    def test_observer_failures_are_not_swallowed(self) -> None:
        diagnostics = tuple(
            RecurrentPPOEpochDiagnostics(
                epoch_index=index,
                actor_loss=1.0,
                critic_loss=2.0,
                entropy_mean=0.5,
                total_loss=3.0,
                ratio_mean=1.0,
                actor_grad_norm_before_clip=0.4,
                critic_grad_norm_before_clip=0.5,
                clip_max_norm=0.5,
            )
            for index in range(4)
        )
        output = RecurrentPPOUpdateOutput(
            epoch_diagnostics=diagnostics,
            chunk_count=8,
            chunk_length=32,
            valid_transition_count=256,
            old_policy_snapshot_preserved=True,
        )

        class FixedUpdater:
            def update(self, _buffer, **_kwargs):
                return output

        class RaisingWriter:
            def write_update(self, _output, **_metadata) -> None:
                raise RuntimeError("synthetic writer failure")

            def write_episode(self, **_record) -> None:
                pass

        class FailingUpdater:
            def update(self, _buffer, **_kwargs):
                raise RuntimeError("synthetic updater failure")

        class CountingWriter:
            def __init__(self) -> None:
                self.update_calls = 0

            def write_update(self, _output, **_metadata) -> None:
                self.update_calls += 1

            def write_episode(self, **_record) -> None:
                pass

        counting_writer = CountingWriter()
        failed_update_trainer = CAGATMAPPOTrainer(
            matched_config(0.02),
            updater_factory=lambda *_: FailingUpdater(),
            matched_diagnostic_writer=counting_writer,
        )
        failed_update_trainer.rollout_buffer = SimpleNamespace(
            full=True,
            finalize=lambda: None,
            clear=lambda: None,
        )
        failed_update_trainer._rollout_policy_version = (
            failed_update_trainer.policy_version
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic updater failure"):
            failed_update_trainer._perform_update([])
        self.assertEqual(counting_writer.update_calls, 0)

        trainer = CAGATMAPPOTrainer(
            matched_config(0.02),
            updater_factory=lambda *_: FixedUpdater(),
            matched_diagnostic_writer=RaisingWriter(),
        )
        trainer.rollout_buffer = SimpleNamespace(
            full=True,
            finalize=lambda: None,
            clear=lambda: None,
        )
        trainer._rollout_policy_version = trainer.policy_version
        with self.assertRaisesRegex(RuntimeError, "synthetic writer failure"):
            trainer._perform_update([])
        self.assertEqual(trainer.policy_version, 0)


class StopEvaluatorTests(unittest.TestCase):
    def _updates(self, *, kl: float = 0.01, clip: float = 0.1, clipping: float = 0.0):
        return [
            {
                "record_type": "update",
                "update_index": index,
                "approx_kl_max": kl,
                "clip_fraction_max": clip,
                "gradient_clip_fraction": clipping,
            }
            for index in range(3)
        ]

    def test_kl_and_clip_boundaries_are_inclusive(self) -> None:
        self.assertEqual(evaluate_stop(self._updates(kl=0.02), group="a1").status, "CONTINUE")
        self.assertEqual(evaluate_stop(self._updates(kl=0.020001), group="a1").status, "STOP_PPO_INSTABILITY")
        self.assertEqual(evaluate_stop(self._updates(clip=0.30), group="a1").status, "CONTINUE")
        self.assertEqual(evaluate_stop(self._updates(clip=0.300001), group="a1").status, "STOP_PPO_INSTABILITY")

    def test_persistent_gradient_clipping_stops(self) -> None:
        self.assertEqual(
            evaluate_stop(self._updates(clipping=1.0), group="a1").status,
            "STOP_GRADIENT_CLIPPING",
        )

    def test_five_rollout_starvation_stops_treatment_but_not_baseline(self) -> None:
        records = [row for index in range(5) for row in rollout_record(index, remote=0, mass=0.01)]
        self.assertEqual(evaluate_stop(records, group="a1").status, "STOP_REMOTE_STARVATION")
        self.assertEqual(evaluate_stop(records, group="baseline").status, "CONTINUE")

    def test_five_rollout_remote_overuse_stops(self) -> None:
        records = [
            row
            for index in range(5)
            for row in rollout_record(index, remote=10, local_share=0.0, remote_share=0.951)
        ]
        self.assertEqual(evaluate_stop(records, group="a2").status, "STOP_REMOTE_OVERUSE")

    def test_numerical_stop(self) -> None:
        records = self._updates()
        records[0]["approx_kl_max"] = math.nan
        self.assertEqual(evaluate_stop(records, group="baseline").status, "STOP_NUMERICAL")

    def test_symmetric_kpi_denominator_is_bounded_near_zero(self) -> None:
        self.assertEqual(symmetric_degradation(0.0, 0.0, higher_is_better=True), 0.0)
        self.assertEqual(symmetric_degradation(0.0, 0.01, higher_is_better=False), 2.0)
        self.assertAlmostEqual(symmetric_degradation(100.0, 89.0, higher_is_better=True), 22 / 189)

    def test_kpi_regression_requires_three_matched_windows(self) -> None:
        baseline = [episode_record(index, group="baseline", return_value=100.0) for index in range(7)]
        candidate = [episode_record(index, group="a1", return_value=80.0) for index in range(7)]
        self.assertEqual(
            evaluate_stop(candidate, group="a1", baseline_records=baseline).status,
            "STOP_KPI_REGRESSION",
        )
        short_baseline = baseline[:2]
        short_candidate = candidate[:2]
        self.assertEqual(
            kpi_regression_gate(short_baseline, short_candidate).status,
            "INCONCLUSIVE",
        )


class MatchedAnalyzerTests(unittest.TestCase):
    def _collapsed(self, count: int, *, group: str = "baseline"):
        rows = []
        for index in range(count):
            collapse = index >= count - math.ceil(count * 0.20)
            pair = rollout_record(
                index,
                remote=0 if collapse else 1,
                local_share=0.99 if collapse else 0.8,
                remote_share=0.01 if collapse else 0.2,
                mass=0.01 if collapse else 0.2,
                ratio=0.01 if collapse else 0.1,
            )
            for row in pair:
                row["group"] = group
            rows.extend(pair)
        rows.extend(episode_record(index, group=group) for index in range(12))
        return rows

    def _healthy(self, count: int, *, group: str):
        rows = []
        for index in range(count):
            pair = rollout_record(index, remote=2, local_share=0.8, remote_share=0.2, mass=0.2, ratio=0.1)
            for row in pair:
                row["group"] = group
            rows.extend(pair)
        rows.extend(episode_record(index, group=group) for index in range(12))
        return rows

    def test_exact_late_windows(self) -> None:
        records = [{"x": index} for index in range(128)]
        self.assertEqual(len(warmup_window(records)), 26)
        self.assertEqual(warmup_window(records)[-1]["x"], 25)
        self.assertEqual(len(late_window(records)), 26)
        self.assertEqual(late_window(records)[0]["x"], 102)
        self.assertEqual(len(late_window([{"x": index} for index in range(160)])), 32)

    def test_trailing_five_ignores_missing_without_zero_fill(self) -> None:
        values = trailing_mean([1.0, None, 3.0, None, 5.0, 7.0], 5)
        self.assertEqual(values[1], 1.0)
        self.assertEqual(values[4], 3.0)
        self.assertEqual(values[5], 5.0)

    def test_collapse_detector_requires_full_or_and_composite(self) -> None:
        collapsed = self._collapsed(30)
        self.assertTrue(detect_baseline_clear_collapse(collapsed)["clear_local_collapse"])
        no_gradient = self._collapsed(30)
        for row in no_gradient:
            if row["record_type"] == "update":
                row["remote_local_grad_norm_ratio"] = 0.1
        self.assertFalse(detect_baseline_clear_collapse(no_gradient)["clear_local_collapse"])

    def test_gradient_boundaries_and_selection_order(self) -> None:
        baseline = self._collapsed(30)
        a1 = self._healthy(30, group="a1")
        a2 = self._healthy(30, group="a2")
        result = select_candidate(baseline, a1, a2)
        self.assertEqual(result["selection"], "SELECT_A1")
        for row in a1:
            if row["record_type"] == "update":
                row["remote_local_grad_norm_ratio"] = 0.049999
        result = select_candidate(baseline, a1, a2)
        self.assertEqual(result["selection"], "SELECT_A2")

    def test_gradient_0_05_passes_and_five_below_0_01_fail(self) -> None:
        baseline = self._collapsed(30)
        boundary = self._healthy(30, group="a1")
        for row in boundary:
            if row["record_type"] == "update":
                row["remote_local_grad_norm_ratio"] = 0.05
        self.assertEqual(
            evaluate_candidate_final(baseline, boundary, group="a1")["status"],
            "PASS",
        )
        below = self._healthy(30, group="a1")
        for row in below:
            if row["record_type"] == "update" and int(row["update_index"]) >= 25:
                row["remote_local_grad_norm_ratio"] = 0.009
        self.assertEqual(
            evaluate_candidate_final(baseline, below, group="a1")["status"],
            "FAIL",
        )

    def test_insufficient_gradient_rows_are_inconclusive(self) -> None:
        baseline = self._collapsed(30)
        candidate = self._healthy(30, group="a1")
        for row in candidate:
            if row["record_type"] == "update" and int(row["update_index"]) >= 25:
                row["gradient_ratio_valid"] = False
                row["remote_local_grad_norm_ratio"] = None
        result = evaluate_candidate_final(baseline, candidate, group="a1")
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_insufficient_kpi_windows_propagate_inconclusive(self) -> None:
        baseline = self._collapsed(30)
        candidate = [
            row
            for row in self._healthy(30, group="a1")
            if row["record_type"] != "episode"
            or int(row["episode_index"]) < 2
        ]
        result = evaluate_candidate_final(baseline, candidate, group="a1")
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["kpi_evaluation"]["status"], "INCONCLUSIVE")

    def test_hard_cap_without_baseline_collapse_is_inconclusive(self) -> None:
        baseline = self._healthy(30, group="baseline")
        result = select_candidate(
            baseline,
            self._healthy(30, group="a1"),
            self._healthy(30, group="a2"),
            hard_cap_reached=True,
        )
        self.assertEqual(result["selection"], "INCONCLUSIVE_AT_HARD_CAP")

    def test_analyzer_writes_summary_and_timeseries_from_jsonl_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            groups = {
                "baseline": self._collapsed(30),
                "a1": self._healthy(30, group="a1"),
                "a2": self._healthy(30, group="a2"),
            }
            for group, records in groups.items():
                path = root / f"{group}.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
                )
                paths[group] = path
            result = analyze_matched_triplet(
                paths["baseline"], paths["a1"], paths["a2"], output_directory=root / "out"
            )
            self.assertEqual(result["selection"]["selection"], "SELECT_A1")
            self.assertTrue(Path(result["summary_path"]).is_file())
            self.assertTrue(Path(result["timeseries_path"]).is_file())
            plots = plot_matched_timeseries(
                result["timeseries_path"], output_directory=root / "plots"
            )
            self.assertEqual(len(plots), 10)
            self.assertTrue(all(Path(path).is_file() for path in plots))


class ManifestContractTests(unittest.TestCase):
    def _protocol_configs(self) -> dict[str, RunConfig]:
        configs = triplet(budget=32000)
        return {
            group: replace(
                config,
                environment=replace(
                    config.environment,
                    candidate_neighbor_radius_m=525.0,
                ),
            )
            for group, config in configs.items()
        }

    def _valid_manifest(self, output_root: Path) -> dict[str, object]:
        configs = self._protocol_configs()
        collision = inspect_output_collision(output_root)
        return {
            "status": "PASS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "group_definitions": build_group_definitions(
                configs, expected_seed=42
            ),
            "budget_plan": matched_budget_plan(configs["baseline"]),
            "output_collision_result": collision,
        }

    def test_timestamp_is_iso8601_timezone_aware_and_not_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._valid_manifest(Path(directory) / "new")
        parsed = datetime.fromisoformat(str(manifest["timestamp"]))
        self.assertIsNotNone(parsed.tzinfo)
        self.assertIsNotNone(parsed.utcoffset())
        identities = identity_fixture()
        identities["baseline"]["timestamp"] = str(manifest["timestamp"])
        identities["a1"]["timestamp"] = "different-metadata"
        self.assertEqual(validate_initial_identities(identities)["status"], "PASS")

    def test_group_definitions_are_explicit_and_do_not_replace_whitelist(self) -> None:
        configs = self._protocol_configs()
        definitions = build_group_definitions(configs, expected_seed=42)
        self.assertEqual(set(definitions), {"baseline", "a1", "a2"})
        self.assertEqual(
            {
                group: definition["route_choice_stability_coef"]
                for group, definition in definitions.items()
            },
            {"baseline": 0.0, "a1": 0.02, "a2": 0.05},
        )
        for field in (
            "route_choice_stability_enabled",
            "route_choice_entropy_floor_nats",
            "scenario",
            "seed",
            "cooperation_radius_m",
            "route_decoder",
        ):
            self.assertEqual(
                {definition[field] for definition in definitions.values()},
                {definitions["baseline"][field]},
            )
        self.assertEqual(validate_matched_configs(configs)["status"], "PASS")

    def test_budget_plan_contains_all_frozen_phases_and_legacy_rejections(self) -> None:
        plan = matched_budget_plan(self._protocol_configs()["baseline"])
        self.assertEqual(
            {
                phase: (
                    plan[phase]["budget"],
                    plan[phase]["checkpoint_interval"],
                    plan[phase]["episodes"],
                    plan[phase]["full_rollouts"],
                    plan[phase]["optimized_steps"],
                    plan[phase]["unused_final_tail"],
                )
                for phase in ("smoke", "diagnostic", "extension")
            },
            {
                "smoke": (1500, 500, 3, 5, 1280, 220),
                "diagnostic": (32000, 4000, 64, 125, 32000, 0),
                "extension": (42500, 4000, 85, 166, 42496, 4),
            },
        )
        self.assertTrue(plan["diagnostic"]["lossless"])
        self.assertTrue(plan["extension"]["fresh_campaign_only"])
        self.assertEqual(plan["episode_horizon"], 500)
        self.assertEqual(plan["rollout_capacity"], 256)
        self.assertEqual(plan["lcm_episode_rollout"], 32000)
        self.assertEqual(plan["next_lossless_boundary"], 64000)
        self.assertFalse(plan["legacy_invalid"]["32768"]["supported"])
        self.assertFalse(plan["legacy_invalid"]["40960"]["supported"])

    def test_output_collision_new_empty_and_nonempty_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_result = inspect_output_collision(root / "new")
            self.assertEqual(new_result["status"], "PASS")
            self.assertFalse(new_result["existed_before"])
            self.assertIsNone(new_result["was_empty_if_existing"])
            self.assertFalse(new_result["overwrite_performed"])

            empty = root / "empty"
            empty.mkdir()
            empty_result = inspect_output_collision(empty)
            self.assertEqual(empty_result["status"], "PASS")
            self.assertTrue(empty_result["existed_before"])
            self.assertTrue(empty_result["was_empty_if_existing"])
            self.assertFalse(empty_result["overwrite_performed"])

            nonempty = root / "nonempty"
            nonempty.mkdir()
            (nonempty / "history.json").write_text("fixture", encoding="utf-8")
            with self.assertRaisesRegex(
                MatchedPreflightError, "STOP_OUTPUT_COLLISION"
            ):
                inspect_output_collision(nonempty)

    def test_manifest_validator_fails_closed_for_each_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._valid_manifest(Path(directory) / "new")
        self.assertEqual(validate_preflight_manifest(manifest)["status"], "PASS")
        for field in (
            "timestamp",
            "group_definitions",
            "budget_plan",
            "output_collision_result",
        ):
            incomplete = dict(manifest)
            del incomplete[field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    MatchedPreflightError, "STOP_MANIFEST_CONTRACT"
                ):
                    validate_preflight_manifest(incomplete)
        naive = dict(manifest)
        naive["timestamp"] = "2026-09-06T00:00:00"
        with self.assertRaisesRegex(
            MatchedPreflightError, "timestamp must include timezone"
        ):
            validate_preflight_manifest(naive)


class CheckpointPlanTests(unittest.TestCase):
    def _plan_config(self, budget: int, *, interval: int = 4000) -> RunConfig:
        config = matched_config(0.0, budget=budget)
        return replace(
            config,
            training=replace(
                config.training,
                mappo=replace(
                    config.training.mappo,
                    checkpoint_interval_steps=interval,
                ),
            ),
        )

    def test_frozen_budget_specs_have_one_central_source(self) -> None:
        self.assertEqual(SMOKE_BUDGET, 1500)
        self.assertEqual(SMOKE_CHECKPOINT_INTERVAL, 500)
        self.assertEqual(DIAGNOSTIC_BUDGET, 32000)
        self.assertEqual(DIAGNOSTIC_CHECKPOINT_INTERVAL, 4000)
        self.assertEqual(EXTENSION_BUDGET, 42500)
        self.assertEqual(EXTENSION_CHECKPOINT_INTERVAL, 4000)
        self.assertEqual(
            {
                name: (spec.total_steps, spec.checkpoint_interval)
                for name, spec in MATCHED_SHORT_BUDGET_SPECS.items()
            },
            {
                "smoke": (1500, 500),
                "diagnostic": (32000, 4000),
                "extension": (42500, 4000),
            },
        )
        self.assertEqual(
            MATCHED_SHORT_BUDGET_SPECS["smoke"].purpose,
            "runtime_telemetry_and_safety_only",
        )
        self.assertTrue(
            MATCHED_SHORT_BUDGET_SPECS["extension"].fresh_campaign_required
        )

    def test_1500_smoke_plan_reports_partial_final_tail_honestly(self) -> None:
        plan = validate_exact_final_checkpoint_plan(
            self._plan_config(SMOKE_BUDGET, interval=SMOKE_CHECKPOINT_INTERVAL)
        )
        self.assertEqual(plan.episode_count, 3)
        self.assertEqual(plan.full_rollout_count, 5)
        self.assertEqual(plan.optimized_transition_count, 1280)
        self.assertEqual(plan.optimized_cutoff_step, 1280)
        self.assertEqual(plan.rollout_remainder, 220)
        self.assertEqual(plan.unused_final_tail_transitions, 220)
        self.assertEqual(plan.periodic_milestones, (500, 1000))
        self.assertEqual(
            dict(plan.periodic_active_rollout_lengths),
            {500: 244, 1000: 232},
        )
        self.assertEqual(plan.final_step, 1500)
        self.assertTrue(plan.episode_boundary_aligned)
        self.assertFalse(plan.rollout_boundary_aligned)
        self.assertTrue(plan.final_completed_supported)
        self.assertFalse(plan.final_requires_rollout_boundary)
        self.assertEqual(
            plan.final_tail_behavior,
            FINAL_PARTIAL_ROLLOUT_BEHAVIOR,
        )
        self.assertFalse(plan.final_partial_rollout_ppo_update)
        self.assertFalse(plan.final_tail_marks_tasks_truncated)
        self.assertEqual(plan.final_active_rollout_length, 0)

    def test_32000_diagnostic_plan_is_jointly_lossless(self) -> None:
        plan = validate_exact_final_checkpoint_plan(
            self._plan_config(
                DIAGNOSTIC_BUDGET,
                interval=DIAGNOSTIC_CHECKPOINT_INTERVAL,
            )
        )
        self.assertEqual(plan.episode_count, 64)
        self.assertEqual(plan.full_rollout_count, 125)
        self.assertEqual(plan.optimized_transition_count, 32000)
        self.assertEqual(plan.optimized_cutoff_step, 32000)
        self.assertEqual(plan.rollout_remainder, 0)
        self.assertEqual(plan.unused_final_tail_transitions, 0)
        self.assertEqual(
            plan.periodic_milestones,
            (4000, 8000, 12000, 16000, 20000, 24000, 28000),
        )
        self.assertEqual(
            dict(plan.periodic_active_rollout_lengths),
            {
                4000: 160,
                8000: 64,
                12000: 224,
                16000: 128,
                20000: 32,
                24000: 192,
                28000: 96,
            },
        )
        self.assertEqual(plan.final_step, 32000)
        self.assertTrue(plan.episode_boundary_aligned)
        self.assertTrue(plan.rollout_boundary_aligned)
        self.assertTrue(plan.final_completed_supported)
        self.assertEqual(plan.final_tail_behavior, "no_unused_final_tail")
        self.assertEqual(plan.final_active_rollout_length, 0)

    def test_42500_extension_plan_is_fresh_and_preserves_accounting(self) -> None:
        plan = validate_exact_final_checkpoint_plan(
            self._plan_config(
                EXTENSION_BUDGET,
                interval=EXTENSION_CHECKPOINT_INTERVAL,
            )
        )
        self.assertEqual(plan.episode_count, 85)
        self.assertEqual(plan.full_rollout_count, 166)
        self.assertEqual(plan.optimized_transition_count, 42496)
        self.assertEqual(plan.optimized_cutoff_step, 42496)
        self.assertEqual(plan.rollout_remainder, 4)
        self.assertEqual(plan.unused_final_tail_transitions, 4)
        self.assertEqual(
            plan.periodic_milestones,
            (
                4000,
                8000,
                12000,
                16000,
                20000,
                24000,
                28000,
                32000,
                36000,
                40000,
            ),
        )
        self.assertIn((32000, 0), plan.periodic_active_rollout_lengths)
        self.assertEqual(plan.final_step, 42500)
        self.assertTrue(plan.episode_boundary_aligned)
        self.assertFalse(plan.rollout_boundary_aligned)
        self.assertTrue(plan.final_completed_supported)
        self.assertEqual(
            plan.final_tail_behavior,
            FINAL_PARTIAL_ROLLOUT_BEHAVIOR,
        )
        self.assertEqual(
            plan.periodic_partial_rollout_behavior,
            PERIODIC_PARTIAL_ROLLOUT_BEHAVIOR,
        )
        self.assertFalse(plan.final_partial_rollout_ppo_update)
        self.assertFalse(plan.final_tail_marks_tasks_truncated)
        report = plan.to_dict()
        self.assertEqual(report["collected_environment_steps"], 42500)
        self.assertEqual(report["optimized_environment_steps"], 42496)
        self.assertEqual(report["unused_final_tail_transitions"], 4)
        self.assertTrue(
            MATCHED_SHORT_BUDGET_SPECS["extension"].fresh_campaign_required
        )

    def test_joint_lossless_boundaries(self) -> None:
        self.assertEqual(joint_lossless_boundary(500, 256), 32000)
        self.assertEqual(next_joint_lossless_boundary(32000, 500, 256), 64000)

    def test_32768_plan_has_requested_periodics_and_exposes_final_blocker(self) -> None:
        plan = checkpoint_plan(self._plan_config(32768))
        self.assertEqual(
            plan.periodic_steps,
            (4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000),
        )
        self.assertEqual(plan.final_step, 32768)
        self.assertEqual(plan.final_kind, "FINAL_COMPLETED")
        self.assertFalse(plan.final_completed_runtime_compatible)
        with self.assertRaisesRegex(
            MatchedPreflightError, "STOP_CHECKPOINT_PLAN"
        ):
            validate_exact_final_checkpoint_plan(self._plan_config(32768))

    def test_40960_extension_plan_has_requested_periodics_and_exposes_final_blocker(self) -> None:
        plan = checkpoint_plan(self._plan_config(40960))
        self.assertEqual(plan.periodic_steps[-2:], (36000, 40000))
        self.assertEqual(plan.final_step, 40960)
        self.assertFalse(plan.final_completed_runtime_compatible)
        with self.assertRaisesRegex(
            MatchedPreflightError, "STOP_CHECKPOINT_PLAN"
        ):
            validate_exact_final_checkpoint_plan(self._plan_config(40960))


if __name__ == "__main__":
    unittest.main()
