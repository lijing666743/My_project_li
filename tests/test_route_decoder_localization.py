"""Gates for read-only Route Decoder Localization V1."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.config import RouteDecoderMode
from src.evaluation.route_decoder_localization import (
    ActivationHookRecorder,
    RouteDecoderLocalizationError,
    _baseline_summary,
    _linear_cka,
    _numeric_metrics,
    _tensor_from_snapshot,
    audit_actor_parameters,
    compare_activation_captures,
    load_localization_fixture,
)
from src.evaluation.route_oracle import actor_state_digest, tensor_digest
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorTensorBatch,
    CAGATMAPPOActor,
)
from tests.test_candidate_aware_route_decoder_v1 import make_decoder_config


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_FIXTURE = (
    REPO_ROOT
    / "logs"
    / "same_state_route_probe"
    / "same-state-zero-hidden-v1__091296cba88ab05f376adb08"
    / "same_state_route_probe_v1.jsonl"
)


def actor_batch(actor: CAGATMAPPOActor) -> ActorTensorBatch:
    spec = actor.spec
    self_features = torch.linspace(
        -0.25,
        0.25,
        steps=spec.uav_count * spec.self_feature_dim,
        dtype=torch.float32,
    ).reshape(1, 1, spec.uav_count, spec.self_feature_dim)
    public_features = torch.linspace(
        -0.1,
        0.1,
        steps=spec.uav_count * spec.uav_count * spec.neighbor_public_feature_dim,
        dtype=torch.float32,
    ).reshape(
        1,
        1,
        spec.uav_count,
        spec.uav_count,
        spec.neighbor_public_feature_dim,
    )
    edge_features = torch.zeros(
        1,
        1,
        spec.uav_count,
        spec.uav_count,
        spec.edge_feature_dim,
        dtype=torch.float32,
    )
    masks = {
        branch: torch.ones(
            1,
            1,
            spec.uav_count,
            spec.action_dimensions[branch],
            dtype=torch.bool,
        )
        for branch in ACTION_BRANCH_ORDER
    }
    indices = {
        branch: torch.zeros(1, 1, spec.uav_count, dtype=torch.long)
        for branch in ACTION_BRANCH_ORDER
    }
    batch = ActorTensorBatch(
        self_features=self_features,
        neighbor_public_features=public_features,
        edge_features=edge_features,
        neighbor_mask=torch.ones(
            1, 1, spec.uav_count, spec.uav_count, dtype=torch.bool
        ),
        action_masks=masks,
        action_indices=indices,
        episode_starts=torch.zeros(1, 1, spec.uav_count, dtype=torch.bool),
    )
    batch.validate(spec)
    return batch


class CompleteFixtureGateTests(unittest.TestCase):
    def test_formal_fixture_has_complete_inputs_for_all_32_decisions(self) -> None:
        fixture = load_localization_fixture(FORMAL_FIXTURE)

        self.assertEqual(len(fixture.states), 29)
        self.assertEqual(len(fixture.decisions), 32)
        self.assertEqual(
            {decision.state_id for decision in fixture.decisions},
            {state.state_id for state in fixture.states},
        )
        self.assertTrue(
            all(state.actor_batch.self_features.numel() > 0 for state in fixture.states)
        )

    def test_formal_fixture_stored_outputs_pass_locked_baseline_gate(self) -> None:
        fixture = load_localization_fixture(FORMAL_FIXTURE)
        rows = []
        for decision in fixture.decisions:
            for label in ("4K", "60K"):
                output = decision.expected_outputs[label]
                rows.append(
                    {
                        "decision_key": decision.decision_key,
                        "actor_label": label,
                        "p_remote_total": output["p_remote_total"],
                        "p_local": output["p_local"],
                        "route_entropy": output["route_entropy"],
                        "argmax_route": json.dumps(
                            output["masked_argmax_route"], separators=(",", ":")
                        ),
                    }
                )

        summary = _baseline_summary(rows)

        self.assertEqual(
            summary["argmax_transition_count_4k_remote_60k_local"], 32
        )

    def test_missing_tensor_values_fails_without_reconstructing_state(self) -> None:
        records = [
            json.loads(line)
            for line in FORMAL_FIXTURE.read_text(encoding="utf-8").splitlines()
        ]
        state = next(record for record in records if record["record_type"] == "state")
        del state["actor_input"]["self_features"]["values"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    for record in records
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RouteDecoderLocalizationError, "complete tensor fields"
            ):
                load_localization_fixture(path)


class TensorMetricGateTests(unittest.TestCase):
    def test_tensor_snapshot_requires_matching_digest_shape_and_nbytes(self) -> None:
        tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        snapshot = {
            "dtype": "torch.float32",
            "shape": [1, 2],
            "nbytes": tensor.numpy().nbytes,
            "sha256": tensor_digest(tensor),
            "values": [[1.0, 2.0]],
        }
        self.assertTrue(torch.equal(_tensor_from_snapshot(snapshot, "x"), tensor))

        mutated = dict(snapshot, values=[[1.0, 3.0]])
        with self.assertRaisesRegex(RouteDecoderLocalizationError, "SHA-256"):
            _tensor_from_snapshot(mutated, "x")

    def test_numeric_metrics_and_cka_are_deterministic(self) -> None:
        left = torch.tensor([1.0, 2.0, 3.0])
        right = torch.tensor([1.0, 2.0, 4.0])
        first = _numeric_metrics(left, right)
        second = _numeric_metrics(left, right)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta_l2"], 1.0)
        self.assertAlmostEqual(
            _linear_cka([left, right], [left, right]), 1.0, places=12
        )


class HookAndParameterGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_decoder_config(RouteDecoderMode.CANDIDATE_AWARE_V1)
        self.actor = CAGATMAPPOActor(self.config).eval().requires_grad_(False)
        self.batch = actor_batch(self.actor)

    def test_hook_manifest_captures_a_b_c_and_is_removed(self) -> None:
        before = actor_state_digest(self.actor)
        recorder = ActivationHookRecorder(self.actor)

        with recorder, torch.inference_mode():
            hidden = self.actor.initial_hidden(1, device="cpu", dtype=torch.float32)
            self.actor(self.batch, hidden)
            captures = dict(recorder.captures)

        self.assertEqual(before, actor_state_digest(self.actor))
        self.assertTrue(any(key.startswith("A.gru.post") for key in captures))
        self.assertTrue(
            any(key.startswith("B.candidate_projection.post") for key in captures)
        )
        self.assertTrue(any(key.startswith("C.route_logits.post") for key in captures))
        self.assertEqual(recorder._handles, [])
        for _stage, _name, module in recorder._modules:
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

        rows = compare_activation_captures("state", captures, captures)
        self.assertTrue(rows)
        self.assertTrue(all(row["relative_l2"] == 0.0 for row in rows))

    def test_parameter_audit_localizes_terminal_scorer_change_to_c(self) -> None:
        other = copy.deepcopy(self.actor)
        with torch.no_grad():
            other.route_decoder.remote_scorer.bias.add_(0.5)

        rows = audit_actor_parameters(self.actor, other)
        changed = [row for row in rows if not row["exact_equal"]]

        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["stage"], "C")
        self.assertEqual(
            changed[0]["name"], "action_heads.route.remote_scorer.bias"
        )

    def test_legacy_decoder_is_rejected_before_hook_registration(self) -> None:
        legacy = CAGATMAPPOActor(
            make_decoder_config(RouteDecoderMode.LEGACY)
        )
        with self.assertRaisesRegex(
            RouteDecoderLocalizationError, "candidate_aware_v1"
        ):
            ActivationHookRecorder(legacy)


if __name__ == "__main__":
    unittest.main()
