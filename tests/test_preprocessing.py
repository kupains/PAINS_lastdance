from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lib.preprocessing import PerPitcherStandardScaler, PerPitcherTrainImputerScaler


class PerPitcherStandardScalerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = pd.DataFrame(
            {
                "pitcher": ["A", "A", "B", "B"],
                "game_date": pd.to_datetime(
                    ["2020-01-01", "2021-01-01", "2020-01-02", "2021-01-02"]
                ),
                "feature": [1.0, 3.0, 10.0, 14.0],
            }
        )

    def test_scaler_metadata_and_pitcher_statistics_are_training_only(self) -> None:
        scaler = PerPitcherStandardScaler(["feature"]).fit(self.training)
        self.assertEqual(scaler.fit_years_, (2020, 2021))
        self.assertEqual(scaler.metadata_["fit_year_max"], 2021)
        validation = pd.DataFrame(
            {
                "pitcher": ["A", "A"],
                "game_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "feature": [3.0, 100000.0],
            }
        )
        transformed_once = scaler.transform(validation.iloc[[0]])["feature"].iloc[0]
        scaler.transform(validation)
        transformed_again = scaler.transform(validation.iloc[[0]])["feature"].iloc[0]
        self.assertEqual(transformed_once, transformed_again)
        self.assertAlmostEqual(transformed_once, 1.0)

    def test_unseen_pitcher_uses_pooled_training_stats(self) -> None:
        scaler = PerPitcherStandardScaler(["feature"]).fit(self.training)
        validation = pd.DataFrame(
            {
                "pitcher": ["UNSEEN"],
                "game_date": pd.to_datetime(["2024-01-01"]),
                "feature": [7.0],
            }
        )
        transformed = scaler.transform(validation)
        expected = (7.0 - self.training["feature"].mean()) / self.training["feature"].std(ddof=0)
        self.assertAlmostEqual(float(transformed.loc[0, "feature"]), float(expected))


class SequencePreprocessorTests(unittest.TestCase):
    def test_missing_values_use_training_pitcher_then_pooled_medians(self) -> None:
        training = pd.DataFrame(
            {
                "pitcher": ["A", "A", "B", "B"],
                "year": [2020, 2021, 2020, 2021],
                "velocity": [1.0, 3.0, 10.0, 14.0],
                "long_gap": [0.0, 1.0, 0.0, 0.0],
            }
        )
        preprocessor = PerPitcherTrainImputerScaler(
            ["velocity", "long_gap"],
            binary_columns=["long_gap"],
        ).fit(training)
        validation = pd.DataFrame(
            {
                "pitcher": ["A", "UNSEEN"],
                "year": [2024, 2024],
                "velocity": [np.nan, np.nan],
                "long_gap": [np.nan, np.nan],
            }
        )
        transformed = preprocessor.transform(validation)
        self.assertAlmostEqual(float(transformed.loc[0, "velocity"]), 0.0)
        pooled_median = float(training["velocity"].median())
        pooled_mean = float(training["velocity"].mean())
        pooled_scale = float(training["velocity"].std(ddof=0))
        self.assertAlmostEqual(
            float(transformed.loc[1, "velocity"]),
            (pooled_median - pooled_mean) / pooled_scale,
        )
        self.assertEqual(float(transformed.loc[0, "long_gap"]), 0.5)
        self.assertEqual(float(transformed.loc[1, "long_gap"]), 0.0)
        self.assertEqual(preprocessor.metadata_["fit_year_max"], 2021)

    def test_transform_before_fit_fails(self) -> None:
        preprocessor = PerPitcherTrainImputerScaler(["velocity"])
        with self.assertRaisesRegex(RuntimeError, "not been fitted"):
            preprocessor.transform(pd.DataFrame({"pitcher": ["A"], "velocity": [1.0]}))


if __name__ == "__main__":
    unittest.main()
