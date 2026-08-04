from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lib.ordinal_linear import OrdinalLinearConfig, torch_available as ordinal_torch_available
from lib.shared_tcn import SharedTCNConfig, torch_available as tcn_torch_available
from lib.stuff_experiment import (
    COMPACT_FEATURES,
    ORDINAL_FEATURES,
    TCN_RAW_FEATURES,
    average_seed_predictions,
    ordinal_auxiliary_prediction,
    predict_ewma_fold,
    predict_ordinal_fold,
    predict_residual_fold,
    predict_tcn_fold,
    prepare_fold,
)


def _synthetic_outings() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2020-06-01",
            "2021-06-01",
            "2022-06-01",
            "2023-06-01",
            "2024-05-01",
            "2024-06-01",
        ]
    )
    n = len(dates)
    frame = pd.DataFrame(
        {
            "pitcher": [7] * n,
            "game_pk": np.arange(700, 700 + n),
            "game_date": dates,
            # Deliberately provide only the original target alias. prepare_fold
            # must expose stuff_plus/sp_stuff without changing these values.
            "target_y": [90.0, 100.0, 110.0, 120.0, 105.0, 115.0],
            "pitch_count": [80, 82, 84, 86, 88, 90],
            "history_eligible": [True] * n,
            "target_eligible": [True] * n,
            "ff_count": [30, 32, 34, 36, 1, 1],
            "si_count": [2, 3, 4, 5, 999, 999],
            "fa_count": [0] * n,
            "ff_velocity": [94.0, 94.5, 95.0, 95.5, 96.0, 96.5],
            "ff_spin": [2300, 2310, 2320, 2330, 2340, 2350],
            "ff_pfx_x": [-0.4] * n,
            "ff_pfx_z": [1.2] * n,
            "si_velocity": [91.0] * n,
            "si_spin": [2100] * n,
            "si_pfx_x": [-1.0] * n,
            "si_pfx_z": [0.5] * n,
            "fa_velocity": [np.nan] * n,
            "fa_spin": [np.nan] * n,
            "fa_pfx_x": [np.nan] * n,
            "fa_pfx_z": [np.nan] * n,
            "rest_days_log": np.log1p([0, 30, 30, 30, 30, 30]),
            "long_gap": [False, True, True, True, True, True],
            "season_start": [True] * n,
            "release_extension": [6.1, 6.2, 6.3, 6.4, 6.5, 6.6],
            "release_pos_x": [-1.5, -1.4, -1.3, -1.2, -1.1, -1.0],
            "release_pos_z": [5.8, 5.9, 6.0, 6.1, 6.2, 6.3],
            "arm_angle": [45, 46, 47, 48, 49, 50],
            "fastball_share": [0.6] * n,
            "breaking_share": [0.3] * n,
        }
    )
    for offset, feature in enumerate(COMPACT_FEATURES):
        if feature not in frame:
            frame[feature] = np.arange(n, dtype=float) + offset / 10.0
    # One evaluation row lacks a compact input. Ridge/XGBoost complete-case
    # coverage must drop this row without reducing EWMA or TCN coverage.
    frame.loc[n - 1, "release_speed_slope5"] = np.nan
    return frame


class PreparedFoldTests(unittest.TestCase):
    def test_public_feature_contracts_and_original_aliases(self) -> None:
        self.assertEqual(len(COMPACT_FEATURES), 16)
        self.assertEqual(len(ORDINAL_FEATURES), 17)
        self.assertEqual(len(TCN_RAW_FEATURES), 15)

        fold = prepare_fold(_synthetic_outings(), 2024)
        self.assertEqual(fold.spec.train_end, 2023)
        self.assertEqual(fold.thresholds.loc[0, "fit_year_max"], 2023)
        self.assertEqual(fold.primary_fastball_mapping.loc[0, "primary_fb_type"], "FF")
        self.assertEqual(
            set(fold.evaluation_targets["stuff_plus"]),
            set(fold.evaluation_targets["target_y"]),
        )
        self.assertTrue(fold.evaluation_targets["ewma4"].notna().all())
        # Validation SI usage is intentionally enormous, but cannot alter the
        # FF mapping fitted on 2020-2023.
        self.assertTrue(
            fold.evaluation_targets["primary_fb_type"].eq("FF").all()
        )

    def test_residual_complete_case_uses_only_actual_compact_features(self) -> None:
        fold = prepare_fold(_synthetic_outings(), 2024)
        ewma = predict_ewma_fold(fold)
        ridge, estimator = predict_residual_fold(
            fold,
            model="ridge",
            ridge_alpha=100.0,
            correction_scale=0.5,
        )
        self.assertEqual(len(estimator.models_), 1)
        self.assertEqual(int(ewma["deployable"].sum()), 2)
        self.assertEqual(int(ridge["feature_eligible"].sum()), 1)
        self.assertEqual(int(ridge["deployable"].sum()), 1)
        incomplete = ridge["release_speed_slope5"].isna()
        self.assertTrue(ridge.loc[incomplete, "predicted_class"].isna().all())

    def test_seed_average_uses_each_rows_fold_local_thresholds(self) -> None:
        rows = []
        for fold_year, q33, q67 in (
            (2022, 90.0, 100.0),
            (2023, 100.0, 110.0),
        ):
            for seed in (20260722, 20260723):
                rows.append(
                    {
                        "row_id": f"row-{fold_year}",
                        "pitcher": 7,
                        "fold_year": fold_year,
                        "model": "tcn_h1_ordered",
                        "candidate_id": "same-candidate",
                        "q33": q33,
                        "q67": q67,
                        "predicted_residual": 0.0,
                        "predicted_stuff_plus": 95.0,
                        "seed": seed,
                        "feature_eligible": True,
                    }
                )

        averaged = average_seed_predictions(pd.DataFrame(rows))
        classes = averaged.set_index("fold_year")["predicted_class"]
        self.assertEqual(classes.loc[2022], 1.0)
        self.assertEqual(classes.loc[2023], 0.0)
        self.assertTrue(averaged["seed_count"].eq(2).all())


@unittest.skipUnless(
    ordinal_torch_available() and tcn_torch_available(),
    "PyTorch is required for ordinal/TCN experiment smoke tests",
)
class NeuralAdapterTests(unittest.TestCase):
    def test_ordinal_and_shuffled_tcn_auxiliary_seed_mean(self) -> None:
        fold = prepare_fold(_synthetic_outings(), 2024)
        ordinal, _ = predict_ordinal_fold(
            fold,
            config=OrdinalLinearConfig(max_iter=20, seed=20260722),
        )
        self.assertEqual(int(ordinal["deployable"].sum()), 2)
        probabilities = ordinal[
            ["prob_low", "prob_middle", "prob_high"]
        ].to_numpy(float)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)

        tcn, _, history = predict_tcn_fold(
            fold,
            config=SharedTCNConfig(
                sequence_length=2,
                internal_channels=4,
                bottleneck_dim=4,
                head_type="H1",
                ordinal_weight=0.25,
                alpha=0.25,
                epochs=1,
                seed=20260722,
            ),
            sequence_order="shuffled",
            shuffle_seed=12345,
            device="cpu",
            candidate_id="smoke_tcn_aux",
        )
        self.assertEqual(tcn["sequence_order"].unique().tolist(), ["shuffled"])
        self.assertEqual(len(history), 1)
        auxiliary = ordinal_auxiliary_prediction(tcn)
        self.assertTrue(auxiliary["model"].eq("tcn_ordinal").all())
        self.assertEqual(int(auxiliary["deployable"].sum()), 2)

        second_seed = tcn.copy()
        second_seed["seed"] = 20260723
        averaged = average_seed_predictions(
            pd.concat([tcn, second_seed], ignore_index=True)
        )
        self.assertTrue(averaged["seed"].eq("mean").all())
        self.assertTrue(averaged["seed_count"].eq(2).all())
        self.assertTrue(
            averaged["predicted_residual_seed_std"].eq(0.0).all()
        )


if __name__ == "__main__":
    unittest.main()
