from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lib.ordinal_linear import (
    ORDINAL_PREDICTION_COLUMNS,
    OrdinalLinearConfig,
    PerPitcherOrdinalLinear,
    ProportionalOddsOrdinalLinear,
    fit_ordinal_linear,
    fit_standardizer,
    ordinal_probabilities_numpy,
    predict_ordinal_linear,
    torch_available,
)


class OrdinalProbabilityTests(unittest.TestCase):
    def test_numpy_probabilities_are_nonnegative_and_normalized(self) -> None:
        probabilities = ordinal_probabilities_numpy(
            latent_score=np.linspace(-20.0, 20.0, 101),
            threshold_1=-0.7,
            threshold_2=1.2,
        )
        self.assertEqual(probabilities.shape, (101, 3))
        self.assertTrue(np.all(probabilities >= 0.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)

    def test_numpy_probabilities_reject_unordered_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            ordinal_probabilities_numpy([0.0], threshold_1=1.0, threshold_2=1.0)

    def test_standardizer_uses_supplied_training_fallback(self) -> None:
        pooled = fit_standardizer(
            np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
        )
        pitcher = fit_standardizer(
            np.array([[2.0, np.nan], [4.0, np.nan]]), fallback=pooled
        )
        transformed = pitcher.transform(np.array([[np.nan, 1000.0]]))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(float(pitcher.median[1]), float(pooled.median[1]))

    def test_optional_dependency_fails_at_model_use_not_import(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        with self.assertRaisesRegex(ImportError, "requires PyTorch"):
            ProportionalOddsOrdinalLinear(2)


@unittest.skipUnless(torch_available(), "PyTorch is an optional test dependency")
class TorchOrdinalModelTests(unittest.TestCase):
    def test_ordered_threshold_parameterization_and_prediction_contract(self) -> None:
        model = ProportionalOddsOrdinalLinear(2)
        prediction = predict_ordinal_linear(
            model, np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        )
        self.assertEqual(list(prediction.columns), ORDINAL_PREDICTION_COLUMNS)
        self.assertTrue(
            np.all(prediction["threshold_2"] > prediction["threshold_1"])
        )
        np.testing.assert_allclose(
            prediction[["prob_low", "prob_middle", "prob_high"]].sum(axis=1),
            1.0,
            atol=1e-7,
        )

    def test_lbfgs_fit_is_deterministic(self) -> None:
        features = np.array(
            [[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]], dtype=float
        )
        labels = np.array([0, 0, 1, 1, 2, 2])
        config = OrdinalLinearConfig(
            l2_lambda=0.1,
            distance_weight=0.1,
            max_iter=60,
            seed=1234,
        )
        first, _ = fit_ordinal_linear(features, labels, config)
        second, _ = fit_ordinal_linear(features, labels, config)
        first_prediction = predict_ordinal_linear(first, features)
        second_prediction = predict_ordinal_linear(second, features)
        np.testing.assert_allclose(first_prediction, second_prediction, atol=1e-9)

    def test_per_pitcher_wrapper_scales_and_returns_required_columns(self) -> None:
        train = pd.DataFrame(
            {
                "pitcher": ["A"] * 6 + ["B"] * 6,
                "f1": [-2, -1, -0.5, 0.5, 1, 2] * 2,
                "f2": [0, 1, 2, 3, 4, 5] + [10, 11, 12, 13, 14, 15],
                "true_class": [0, 0, 1, 1, 2, 2] * 2,
            }
        )
        wrapper = PerPitcherOrdinalLinear(
            ["f1", "f2"],
            OrdinalLinearConfig(max_iter=40, l2_lambda=0.1, seed=99),
        ).fit(train)
        prediction = wrapper.predict(train.iloc[[0, 6]].copy())
        self.assertEqual(list(prediction.columns), ORDINAL_PREDICTION_COLUMNS)
        self.assertEqual(len(wrapper.models_), 2)
        np.testing.assert_allclose(
            prediction[["prob_low", "prob_middle", "prob_high"]].sum(axis=1),
            1.0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
