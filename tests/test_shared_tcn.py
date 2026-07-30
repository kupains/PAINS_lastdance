from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lib.shared_tcn import (
    TCN_RAW_FEATURES,
    LeftCausalConv1d,
    SharedTCN,
    SharedTCNConfig,
    TCNPreprocessor,
    build_pitcher_index,
    build_tcn_sequences,
    pitcher_balanced_mean,
    torch_available,
)


def _outing_rows() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2023-09-01",
            "2023-09-10",
            "2024-03-20",
            "2024-04-01",
            "2024-04-15",
            "2024-04-20",
        ]
    )
    frame = pd.DataFrame(
        {
            "row_id": [f"row-{index}" for index in range(len(dates))],
            "pitcher": ["A"] * len(dates),
            "game_pk": np.arange(len(dates)),
            "game_date": dates,
            "history_eligible": [True, True, True, True, True, True],
            # A sub-50-pitch outing (row-2) is history, but not a target.
            "target_eligible": [False, False, False, False, True, False],
            "primary_fb_available": [True, True, True, False, True, True],
        }
    )
    for position, feature in enumerate(TCN_RAW_FEATURES):
        frame[feature] = np.arange(len(frame), dtype=float) + position / 100.0
    frame.loc[2, "pitch_count"] = 40.0
    frame.loc[3, "primary_fb_velocity"] = np.nan
    return frame


class SequenceConstructionTests(unittest.TestCase):
    def test_exact_raw_feature_contract(self) -> None:
        self.assertEqual(len(TCN_RAW_FEATURES), 15)
        self.assertNotIn("release_speed", TCN_RAW_FEATURES)
        self.assertFalse(any(name.endswith("_ma5") for name in TCN_RAW_FEATURES))
        self.assertFalse(any(name.endswith("_slope5") for name in TCN_RAW_FEATURES))

    def test_strictly_prior_cross_season_history_and_left_padding(self) -> None:
        frame = _outing_rows()
        target = frame.iloc[[4]]
        batch = build_tcn_sequences(
            frame,
            target,
            sequence_length=5,
            sequence_order="ordered",
        )
        self.assertEqual(batch.valid_timestep.tolist(), [[False, True, True, True, True]])
        self.assertEqual(
            batch.sequence_row_ids[0, batch.valid_timestep[0]].tolist(),
            ["row-0", "row-1", "row-2", "row-3"],
        )
        # The sub-50-pitch official start remains usable history.
        self.assertIn("row-2", batch.sequence_row_ids[0].tolist())
        batch.assert_strictly_prior()
        self.assertTrue(
            np.all(
                batch.sequence_game_dates[0, batch.valid_timestep[0]]
                < np.datetime64("2024-04-15")
            )
        )

    def test_same_day_and_future_rows_never_enter_history(self) -> None:
        frame = _outing_rows()
        duplicate = frame.iloc[[4]].copy()
        duplicate["row_id"] = "same-day"
        duplicate["game_pk"] = 999
        history = pd.concat([frame, duplicate], ignore_index=True)
        target = frame.iloc[[4]]
        batch = build_tcn_sequences(history, target, sequence_length=8)
        used = set(batch.sequence_row_ids[0, batch.valid_timestep[0]])
        self.assertNotIn("row-4", used)
        self.assertNotIn("same-day", used)
        self.assertNotIn("row-5", used)

    def test_shuffled_ablation_uses_same_rows_and_fixed_seed(self) -> None:
        frame = _outing_rows()
        target = frame.iloc[[5]]
        ordered = build_tcn_sequences(
            frame, target, sequence_length=5, sequence_order="ordered", seed=7
        )
        shuffled_a = build_tcn_sequences(
            frame, target, sequence_length=5, sequence_order="shuffled", seed=7
        )
        shuffled_b = build_tcn_sequences(
            frame, target, sequence_length=5, sequence_order="shuffled", seed=7
        )
        ordered_ids = ordered.sequence_row_ids[0, ordered.valid_timestep[0]]
        shuffled_ids = shuffled_a.sequence_row_ids[0, shuffled_a.valid_timestep[0]]
        self.assertEqual(set(ordered_ids), set(shuffled_ids))
        np.testing.assert_array_equal(
            shuffled_a.sequence_row_ids, shuffled_b.sequence_row_ids
        )

    def test_train_only_preprocessor_imputes_and_scales_finitely(self) -> None:
        frame = _outing_rows()
        train = frame.iloc[:4].copy()
        preprocessor = TCNPreprocessor().fit(train)
        batch = build_tcn_sequences(
            frame,
            frame.iloc[[4]],
            sequence_length=4,
            preprocessor=preprocessor,
        )
        self.assertTrue(np.isfinite(batch.values).all())
        self.assertFalse(batch.primary_fb_available[0, -1])

    def test_preprocessor_transforms_history_once_not_once_per_target(self) -> None:
        class CountingPreprocessor(TCNPreprocessor):
            def __init__(self) -> None:
                super().__init__()
                self.transform_calls = 0

            def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
                self.transform_calls += 1
                return super().transform(frame)

        frame = _outing_rows()
        preprocessor = CountingPreprocessor().fit(frame.iloc[:4])
        build_tcn_sequences(
            frame,
            frame.iloc[[4, 5]],
            sequence_length=4,
            preprocessor=preprocessor,
        )
        self.assertEqual(preprocessor.transform_calls, 1)

    def test_pitcher_index_is_stable_and_rejects_unknown(self) -> None:
        encoded, mapping = build_pitcher_index(["B", "A", "B"])
        self.assertEqual(mapping, {"A": 0, "B": 1})
        np.testing.assert_array_equal(encoded, [1, 0, 1])
        with self.assertRaises(KeyError):
            build_pitcher_index(["C"], mapping)

    def test_optional_dependency_fails_at_model_use_not_import(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        with self.assertRaisesRegex(ImportError, "requires PyTorch"):
            SharedTCN(num_pitchers=1)


@unittest.skipUnless(torch_available(), "PyTorch is an optional test dependency")
class TorchSharedTCNTests(unittest.TestCase):
    def test_left_causal_convolution_has_no_future_influence(self) -> None:
        import torch

        layer = LeftCausalConv1d(1, 1, kernel_size=2, dilation=1)
        with torch.no_grad():
            layer.conv.weight.fill_(1.0)
            layer.conv.bias.zero_()
        first = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        second = first.clone()
        second[:, :, -1] = 1000.0
        first_output = layer(first)
        second_output = layer(second)
        torch.testing.assert_close(first_output[:, :, :-1], second_output[:, :, :-1])

    def test_h1_and_h2_pitcher_corrections_start_at_zero(self) -> None:
        import torch

        values = torch.randn(4, 5, len(TCN_RAW_FEATURES))
        mask = torch.ones(4, 5, dtype=torch.bool)
        pitcher = torch.tensor([0, 1, 0, 1])
        for head_type in ["H1", "H2"]:
            model = SharedTCN(
                2,
                config=SharedTCNConfig(head_type=head_type, dropout=0.0),
            )
            output = model(values, mask, pitcher)
            torch.testing.assert_close(
                output["pitcher_correction"],
                torch.zeros_like(output["pitcher_correction"]),
            )

    def test_auxiliary_probabilities_are_ordered_and_normalized(self) -> None:
        import torch

        model = SharedTCN(
            2,
            config=SharedTCNConfig(
                head_type="H1", ordinal_weight=0.25, dropout=0.0
            ),
        )
        output = model(
            torch.randn(4, 5, len(TCN_RAW_FEATURES)),
            torch.ones(4, 5, dtype=torch.bool),
            torch.tensor([0, 1, 0, 1]),
            ewma_position=torch.zeros(4),
        )
        self.assertGreater(
            float(output["ordinal_threshold_2"].detach()),
            float(output["ordinal_threshold_1"].detach()),
        )
        self.assertTrue(torch.all(output["ordinal_probabilities"] >= 0.0))
        torch.testing.assert_close(
            output["ordinal_probabilities"].sum(dim=1), torch.ones(4)
        )

    def test_availability_masks_are_separate_from_raw_feature_dimension(self) -> None:
        import torch

        model = SharedTCN(
            2,
            config=SharedTCNConfig(head_type="H1", dropout=0.0),
        )
        values = torch.zeros(2, 5, len(TCN_RAW_FEATURES))
        valid = torch.ones(2, 5, dtype=torch.bool)
        output = model(
            values,
            valid,
            torch.tensor([0, 1]),
            primary_fb_available=torch.tensor(
                [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32
            ),
            stuff_plus_available=torch.ones(2, 5),
        )
        self.assertEqual(output["representation"].shape, (2, 4))
        self.assertEqual(model.encoder.input_dim, 15)

    def test_pitcher_balanced_loss_is_not_row_weighted(self) -> None:
        import torch

        losses = torch.tensor([1.0, 3.0, 9.0])
        pitchers = torch.tensor([0, 0, 1])
        self.assertAlmostEqual(
            float(pitcher_balanced_mean(losses, pitchers)), 5.5
        )


if __name__ == "__main__":
    unittest.main()
