from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from lib.evaluation import (
    common_row_intersection,
    evaluate_common_and_deployable,
    locked_config_fingerprint,
    ordinal_classification_metrics,
    validate_locked_fingerprints,
    validate_locked_model_config,
)


def _valid_lock() -> dict:
    digest_a = "a" * 64
    digest_b = "b" * 64
    return {
        "schema_version": 1,
        "train_start": 2020,
        "validation_years": [2022, 2023, 2024],
        "final_train_end": 2024,
        "test_year": 2025,
        "qualified_pitchers": [1],
        "qualification_rule": "official starter",
        "target_definition": {},
        "pitch_type_mapping": {},
        "ewma": {},
        "ridge": {},
        "xgboost": {},
        "ordinal_linear": {},
        "shared_tcn": {},
        "feature_lists": {},
        "missing_value_policy": {},
        "random_seeds": [20260722],
        "selection_metric": "mean_balanced_accuracy_minus_half_std",
        "dataset_fingerprint": digest_a,
        "fold_manifest_fingerprint": digest_b,
        "code_version": "test",
    }


class OrdinalMetricTests(unittest.TestCase):
    def test_full_metrics_and_coverage(self) -> None:
        metrics = ordinal_classification_metrics(
            [0, 1, 2, 0],
            [0, 2, 0, np.nan],
        )
        self.assertEqual(metrics["n_total_eligible"], 4)
        self.assertEqual(metrics["n_predicted"], 3)
        self.assertAlmostEqual(metrics["coverage"], 0.75)
        self.assertAlmostEqual(metrics["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["ordinal_mae"], 1.0)
        self.assertEqual(metrics["high_to_low_error_count"], 1)
        self.assertEqual(metrics["extreme_error_count"], 1)
        self.assertEqual(metrics["predicted_low_count"], 2)
        self.assertEqual(metrics["predicted_middle_count"], 0)
        self.assertEqual(metrics["predicted_high_count"], 1)

    def test_macro_f1_keeps_all_three_ordinal_classes(self) -> None:
        metrics = ordinal_classification_metrics([0, 0], [0, 0])
        self.assertAlmostEqual(metrics["macro_f1"], 1.0 / 3.0)
        self.assertTrue(np.isnan(metrics["middle_recall"]))
        self.assertTrue(np.isnan(metrics["high_recall"]))

    def test_common_rows_are_the_prediction_intersection(self) -> None:
        frames = {
            "a": pd.DataFrame(
                {
                    "row_id": ["r1", "r2", "r3"],
                    "true_class": [0, 1, 2],
                    "predicted_class": [0, 1, 2],
                    "target_eligible": [True, True, True],
                }
            ),
            "b": pd.DataFrame(
                {
                    "row_id": ["r1", "r2", "r3"],
                    "true_class": [0, 1, 2],
                    "predicted_class": [2, np.nan, 2],
                    "target_eligible": [True, True, True],
                }
            ),
        }
        common = common_row_intersection(frames)
        self.assertEqual(list(common), ["r1", "r3"])
        common_metrics, deployable_metrics = evaluate_common_and_deployable(frames)
        self.assertEqual(set(common_metrics["n_total_eligible"]), {2})
        self.assertEqual(set(common_metrics["coverage"]), {1.0})
        b_deployable = deployable_metrics.loc[deployable_metrics["model"] == "b"].iloc[0]
        self.assertEqual(int(b_deployable["n_total_eligible"]), 3)
        self.assertAlmostEqual(float(b_deployable["coverage"]), 2.0 / 3.0)


class LockConfigTests(unittest.TestCase):
    def test_config_hash_is_key_order_invariant(self) -> None:
        config = _valid_lock()
        reversed_config = dict(reversed(list(config.items())))
        self.assertEqual(
            locked_config_fingerprint(config),
            locked_config_fingerprint(reversed_config),
        )

    def test_missing_lock_field_fails(self) -> None:
        config = _valid_lock()
        del config["ridge"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_locked_model_config(config)

    def test_fingerprint_mismatch_fails_explicitly(self) -> None:
        config = _valid_lock()
        with self.assertRaisesRegex(ValueError, "Dataset fingerprint mismatch"):
            validate_locked_fingerprints(
                config,
                dataset_fingerprint="c" * 64,
                fold_manifest_fingerprint="b" * 64,
            )
        changed = deepcopy(config)
        changed["dataset_fingerprint"] = "c" * 64
        validate_locked_fingerprints(
            changed,
            dataset_fingerprint="c" * 64,
            fold_manifest_fingerprint="b" * 64,
        )


if __name__ == "__main__":
    unittest.main()
