from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from lib.stuff_code_version import (
    ANALYSIS_SOURCE_FILES,
    analysis_code_version,
    analysis_source_sha256,
)
from lib.stuff_mlb_dataset import FEATURES
from lib.temporal_splits import (
    add_row_id,
    apply_primary_fastball_mapping,
    assign_true_classes,
    build_fold_manifest,
    dataframe_fingerprint,
    fit_primary_fastball_mapping,
    fold_manifest_fingerprint,
    rolling_fold_specs,
    stable_row_id,
)
from stuff_mlb_temporal_final import ModelConfig, _feature_names


class LegacyContractTests(unittest.TestCase):
    def test_compact_span4_feature_contract_is_exactly_16_columns(self) -> None:
        config = ModelConfig(
            family="xgboost",
            span=4,
            feature_set="compact",
            correction_scale=1.0,
            profile="shallow",
        )
        expected = [
            "prior_minus_ewma4",
            "mean5_minus_ewma4",
            "stuff_plus_slope_last5",
            "prev_start_pitch_count",
            "rest_days",
            "workload_density_3starts",
            "release_speed_slope5",
            "release_spin_rate_slope5",
            "release_extension_slope5",
            "release_pos_x_slope5",
            "release_pos_z_slope5",
            "arm_angle_slope5",
            "pfx_x_slope5",
            "pfx_z_slope5",
            "breaking_share_slope5",
            "offspeed_share_slope5",
        ]
        actual = _feature_names(config)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 16)
        self.assertEqual(len(FEATURES), 28)

    def test_q67_tie_remains_middle_like_legacy_classes(self) -> None:
        frame = pd.DataFrame(
            {
                "pitcher": [1, 1, 1, 1],
                "target_y": [90.0, 100.0, 110.0, 110.01],
            }
        )
        thresholds = pd.DataFrame({"pitcher": [1], "q33": [90.0], "q67": [110.0]})
        classified = assign_true_classes(frame, thresholds)
        self.assertEqual(classified["true_class"].tolist(), [0, 1, 1, 2])


class RowIdentityAndFingerprintTests(unittest.TestCase):
    def test_row_id_is_stable_across_integer_like_dtypes(self) -> None:
        integer_id = stable_row_id(123, 456, "2024-07-01")
        float_id = stable_row_id(123.0, 456.0, pd.Timestamp("2024-07-01 20:30"))
        self.assertEqual(integer_id, float_id)
        self.assertEqual(len(integer_id), 64)

    def test_add_row_id_rejects_duplicate_outings(self) -> None:
        frame = pd.DataFrame(
            {
                "pitcher": [1, 1],
                "game_pk": [10, 10],
                "game_date": ["2024-01-01", "2024-01-01"],
            }
        )
        with self.assertRaisesRegex(ValueError, "Duplicate outing row ids"):
            add_row_id(frame)

    def test_dataframe_fingerprint_is_order_invariant_and_value_sensitive(self) -> None:
        frame = add_row_id(
            pd.DataFrame(
                {
                    "pitcher": [2, 1],
                    "game_pk": [20, 10],
                    "game_date": ["2024-02-01", "2024-01-01"],
                    "target_y": [101.0, np.nan],
                    "diagnostic_only": ["a", "b"],
                }
            )
        )
        columns = ["row_id", "pitcher", "game_pk", "game_date", "target_y"]
        fingerprint = dataframe_fingerprint(frame, columns)
        reordered = frame.iloc[::-1][list(reversed(frame.columns))]
        self.assertEqual(fingerprint, dataframe_fingerprint(reordered, columns))

        changed_core = frame.copy()
        changed_core.loc[0, "target_y"] = 102.0
        self.assertNotEqual(
            fingerprint, dataframe_fingerprint(changed_core, columns)
        )

        changed_unselected = frame.copy()
        changed_unselected.loc[0, "diagnostic_only"] = "changed"
        self.assertEqual(
            fingerprint, dataframe_fingerprint(changed_unselected, columns)
        )

    def test_fingerprint_rejects_duplicate_or_missing_column_contracts(self) -> None:
        frame = pd.DataFrame({"a": [1], "b": [2]})
        with self.assertRaisesRegex(ValueError, "must be unique"):
            dataframe_fingerprint(frame, ["a", "a"])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            dataframe_fingerprint(frame, ["a", "missing"])


class FoldManifestTests(unittest.TestCase):
    def _outings(
        self,
        evaluation_2022: float = 999.0,
        evaluation_2025: float = 108.0,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pitcher": [1, 1, 1, 1, 1, 1],
                "game_pk": [100, 101, 102, 103, 104, 105],
                "game_date": pd.to_datetime(
                    [
                        "2020-06-01",
                        "2021-06-01",
                        "2022-06-01",
                        "2023-06-01",
                        "2024-06-01",
                        "2025-06-01",
                    ]
                ),
                "target_y": [
                    90.0,
                    110.0,
                    evaluation_2022,
                    100.0,
                    105.0,
                    evaluation_2025,
                ],
                "target_eligible": [True] * 6,
                "history_eligible": [True] * 6,
                # Qualification is intentionally upstream and may use 2025
                # availability without exposing 2025 outcomes to validation.
                "qualified_using_2021_2025_availability": [True] * 6,
            }
        )

    def test_canonical_fold_boundaries(self) -> None:
        folds = rolling_fold_specs()
        self.assertEqual([fold.fold_year for fold in folds], [2022, 2023, 2024, 2025])
        self.assertEqual(folds[0].train_years, (2020, 2021))
        self.assertEqual(folds[-1].evaluation_split, "test")
        self.assertEqual(folds[-1].train_end, 2024)

    def test_2022_thresholds_do_not_use_2022_value(self) -> None:
        first = build_fold_manifest(self._outings(evaluation_2022=999.0))
        second = build_fold_manifest(self._outings(evaluation_2022=-999.0))
        first_eval = first.loc[
            (first["fold_year"] == 2022) & (first["split"] == "validation")
        ]
        second_eval = second.loc[
            (second["fold_year"] == 2022) & (second["split"] == "validation")
        ]
        self.assertAlmostEqual(float(first_eval["q33"].iloc[0]), 96.6666666667)
        self.assertAlmostEqual(float(first_eval["q67"].iloc[0]), 103.3333333333)
        self.assertEqual(float(first_eval["q33"].iloc[0]), float(second_eval["q33"].iloc[0]))
        self.assertEqual(float(first_eval["q67"].iloc[0]), float(second_eval["q67"].iloc[0]))

    def test_2025_roster_rows_are_accepted_but_test_only(self) -> None:
        manifest = build_fold_manifest(self._outings())
        rows_2025 = manifest.loc[manifest["year"] == 2025]
        self.assertFalse(rows_2025.empty)
        self.assertEqual(set(rows_2025["fold_year"]), {2025})
        self.assertEqual(set(rows_2025["split"]), {"test"})
        self.assertTrue(
            rows_2025["qualified_using_2021_2025_availability"].all()
        )
        self.assertFalse(
            (
                manifest["year"].eq(2025)
                & manifest["split"].isin(["train", "validation"])
            ).any()
        )
        validation = manifest.loc[manifest["split"].eq("validation")]
        self.assertLessEqual(int(validation["year"].max()), 2024)

    def test_changing_2025_outcome_cannot_change_development_manifest(self) -> None:
        first = build_fold_manifest(self._outings(evaluation_2025=9999.0))
        second = build_fold_manifest(self._outings(evaluation_2025=-9999.0))
        first_development = first.loc[first["fold_year"].le(2024)]
        second_development = second.loc[second["fold_year"].le(2024)]
        self.assertEqual(
            fold_manifest_fingerprint(first_development),
            fold_manifest_fingerprint(second_development),
        )

    def test_manifest_fingerprint_is_order_invariant_and_split_sensitive(self) -> None:
        manifest = build_fold_manifest(self._outings())
        fingerprint = fold_manifest_fingerprint(manifest)
        self.assertEqual(
            fingerprint,
            fold_manifest_fingerprint(manifest.sample(frac=1.0, random_state=7)),
        )
        changed = manifest.copy()
        changed.loc[changed.index[0], "target_eligible"] = False
        self.assertNotEqual(fingerprint, fold_manifest_fingerprint(changed))


class PrimaryFastballTests(unittest.TestCase):
    def test_validation_usage_cannot_change_training_mapping(self) -> None:
        training = pd.DataFrame(
            {
                "pitcher": [1, 1],
                "game_date": pd.to_datetime(["2020-06-01", "2021-06-01"]),
                "ff_count": [40, 30],
                "si_count": [2, 3],
                "fa_count": [0, 0],
            }
        )
        mapping = fit_primary_fastball_mapping(training, fold_year=2022)
        self.assertEqual(mapping.loc[0, "primary_fb_type"], "FF")
        self.assertEqual(mapping.loc[0, "fit_year_max"], 2021)

        validation = pd.DataFrame(
            {
                "fold_year": [2022],
                "pitcher": [1],
                "ff_count": [1],
                "ff_velocity": [95.0],
                "ff_spin": [2400.0],
                "ff_pfx_x": [-0.4],
                "ff_pfx_z": [1.2],
                "si_count": [999],
                "si_velocity": [90.0],
                "si_spin": [2100.0],
                "si_pfx_x": [-1.1],
                "si_pfx_z": [0.5],
                "fa_count": [0],
            }
        )
        applied = apply_primary_fastball_mapping(validation, mapping)
        self.assertEqual(applied.loc[0, "primary_fb_type"], "FF")
        self.assertEqual(applied.loc[0, "primary_fb_velocity"], 95.0)
        self.assertTrue(bool(applied.loc[0, "primary_fb_available"]))

    def test_nullable_float_pitch_columns_apply_without_dtype_error(self) -> None:
        outings = pd.DataFrame(
            {
                "pitcher": [1, 1],
                "ff_count": pd.Series([10.0, pd.NA], dtype="Float64"),
                "ff_velocity": pd.Series([95.0, pd.NA], dtype="Float64"),
                "ff_spin": pd.Series([2300.0, pd.NA], dtype="Float64"),
                "ff_pfx_x": pd.Series([-5.0, pd.NA], dtype="Float64"),
                "ff_pfx_z": pd.Series([15.0, pd.NA], dtype="Float64"),
                "si_count": pd.Series([0.0, 0.0], dtype="Float64"),
                "fa_count": pd.Series([0.0, 0.0], dtype="Float64"),
            }
        )
        mapping = pd.DataFrame(
            {"pitcher": [1], "primary_fb_type": ["FF"]}
        )
        applied = apply_primary_fastball_mapping(outings, mapping)
        self.assertEqual(applied.loc[0, "primary_fb_velocity"], 95.0)
        self.assertTrue(np.isnan(applied.loc[1, "primary_fb_velocity"]))


class CodeVersionTests(unittest.TestCase):
    def test_current_analysis_source_set_exists_and_hashes_deterministically(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertTrue(all((root / relative).is_file() for relative in ANALYSIS_SOURCE_FILES))
        first = analysis_source_sha256(root)
        second = analysis_source_sha256(root)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        version = analysis_code_version(root)
        self.assertTrue(version.endswith(f"analysis-sha256:{first}"))


if __name__ == "__main__":
    unittest.main()
