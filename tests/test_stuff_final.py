from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from build_fold_manifest import build_manifest
from final_evaluate import make_parser, run_from_args
from lib.stuff_cli import load_stuff_data
from lib.stuff_code_version import analysis_code_version
from lib.stuff_experiment import (
    COMPACT_FEATURES,
    ORDINAL_FEATURES,
    TCN_RAW_FEATURES,
    prepare_fold,
)
from lib.stuff_final import (
    AGREEMENT_COLUMNS,
    ENSEMBLE_CORRECTION_POLICY,
    FINAL_ARTIFACTS,
    ORDINAL_PREDICTION_COLUMNS,
    ORDINAL_TRAIN_ONLY_POLICY,
    TABULAR_COMPLETE_CASE_POLICY,
    TCN_PREDICTION_COLUMNS,
    TCN_TRAIN_ONLY_POLICY,
    _model_agreement,
    run_locked_final_evaluation,
)
from lib.stuff_mlb_dataset import default_pitch_type_mapping


@pytest.fixture(scope="module")
def demo_bundle():
    return load_stuff_data(
        demo=True,
        demo_pitchers=2,
        demo_outings_per_season=6,
        qualification_mode="development",
    )


def _ewma_only_lock(bundle) -> dict:
    _, manifest_fingerprint = build_manifest(bundle)
    fold = prepare_fold(bundle.outings, 2025, evaluation_split="test")
    final_mapping = {
        str(row.pitcher): str(row.primary_fb_type)
        for row in fold.primary_fastball_mapping.itertuples(index=False)
        if pd.notna(row.primary_fb_type)
    }
    return {
        "schema_version": 1,
        "train_start": 2020,
        "validation_years": [2022, 2023, 2024],
        "final_train_end": 2024,
        "test_year": 2025,
        "qualified_pitchers": list(bundle.qualified_pitchers),
        "qualification_rule": bundle.qualification_rule,
        "target_definition": {
            "type": "pitcher_specific_training_tertile",
            "q_low": 0.3333333333,
            "q_high": 0.6666666667,
        },
        "pitch_type_mapping": {
            **default_pitch_type_mapping("fastball"),
            "cutter_category": "fastball",
            "primary_candidates": ["FF", "SI", "FA"],
            "final_primary_by_pitcher": final_mapping,
        },
        "ewma": {
            "status": "selected",
            "span": 4,
            "candidate_id": "ewma_span4",
        },
        "ridge": {
            "status": "not_requested",
            "selected_candidate_id": None,
            "selected": {},
        },
        "xgboost": {
            "status": "not_requested",
            "selected_candidate_id": None,
            "selected": {},
        },
        "ordinal_linear": {
            "status": "not_requested",
            "selected_candidate_id": None,
            "selected": {},
        },
        "shared_tcn": {
            "status": "not_requested",
            "selected_candidate_id": None,
            "selected": {},
            "ensemble": {
                "selected_candidate_id": None,
                "beta": 0.0,
                "ridge_candidate_id": None,
                "tcn_candidate_id": None,
                "correction_policy": ENSEMBLE_CORRECTION_POLICY,
            },
        },
        "feature_lists": {
            "tabular_compact_16": list(COMPACT_FEATURES),
            "ordinal": list(ORDINAL_FEATURES),
            "tcn_raw_15": list(TCN_RAW_FEATURES),
        },
        "missing_value_policy": {
            "tabular": TABULAR_COMPLETE_CASE_POLICY,
            "ordinal": ORDINAL_TRAIN_ONLY_POLICY,
            "tcn": TCN_TRAIN_ONLY_POLICY,
        },
        "random_seeds": [20260722],
        "selection_metric": (
            "mean balanced_accuracy over validation seed-year observations; "
            "selection_score = mean - 0.5 * seed-year std; "
            "0.003 simplicity margin"
        ),
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "dataset_fingerprint_columns": list(bundle.fingerprint_columns),
        "fold_manifest_fingerprint": manifest_fingerprint,
        "code_version": analysis_code_version(),
        "candidate_configs": {
            "ewma_span4": {"span": 4, "features": ["ewma4"]}
        },
        "smoke_test": True,
    }


def test_cli_requires_lock_and_missing_lock_file_fails(tmp_path):
    parser = make_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--demo", "--output-dir", str(tmp_path)])

    args = parser.parse_args(
        [
            "--demo",
            "--locked-config",
            str(tmp_path / "absent.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="Locked model config not found"):
        run_from_args(args)


def test_final_rejects_invalid_nested_schema(demo_bundle, tmp_path):
    lock = _ewma_only_lock(demo_bundle)
    del lock["ridge"]["status"]
    with pytest.raises(ValueError, match="ridge.*missing locked fields"):
        run_locked_final_evaluation(
            demo_bundle,
            lock,
            output_dir=tmp_path,
        )


def test_final_rejects_code_version_mismatch(demo_bundle, tmp_path):
    lock = _ewma_only_lock(demo_bundle)
    lock["code_version"] = "wrong+analysis-sha256:" + "0" * 64
    with pytest.raises(ValueError, match="Analysis code version mismatch"):
        run_locked_final_evaluation(
            demo_bundle,
            lock,
            output_dir=tmp_path,
        )
    assert not any(tmp_path.iterdir())


def test_final_rejects_dataset_fingerprint_mismatch(demo_bundle, tmp_path):
    lock = _ewma_only_lock(demo_bundle)
    lock["dataset_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
        run_locked_final_evaluation(
            demo_bundle,
            lock,
            output_dir=tmp_path,
        )
    assert not any(tmp_path.iterdir())


def test_final_rejects_fold_manifest_fingerprint_mismatch(
    demo_bundle, tmp_path
):
    lock = _ewma_only_lock(demo_bundle)
    lock["fold_manifest_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="Fold manifest fingerprint mismatch"):
        run_locked_final_evaluation(
            demo_bundle,
            lock,
            output_dir=tmp_path,
        )
    assert not any(tmp_path.iterdir())


def test_final_path_has_no_model_selection_import_or_call():
    root = Path(__file__).resolve().parents[1]
    for relative in ("lib/stuff_final.py", "final_evaluate.py"):
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        called_names.update(
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        )
        assert "lib.stuff_selection" not in imported_modules
        assert "run_model_selection" not in called_names


def test_model_agreement_handles_no_disagreements_and_missing_predictions():
    ridge = pd.DataFrame(
        {
            "row_id": ["r1", "r2"],
            "pitcher": ["p1", "p1"],
            "game_date": pd.to_datetime(["2025-04-01", "2025-04-08"]),
            "true_class": pd.Series([2, 1], dtype="Int64"),
            "predicted_class": pd.Series([2, pd.NA], dtype="Float64"),
            "predicted_residual": [1.0, float("nan")],
        }
    )
    tcn = pd.DataFrame(
        {
            "row_id": ["r1", "r2"],
            "predicted_class": pd.Series([2, pd.NA], dtype="Float64"),
            "predicted_residual": [0.5, float("nan")],
        }
    )
    agreement, diagnostics = _model_agreement(ridge, tcn)
    assert len(agreement) == 2
    assert diagnostics["n_common_deployable"] == 1
    assert diagnostics["disagreement_rows"] == 0
    assert diagnostics["category_agreement_rate"] == 1.0
    assert pd.isna(diagnostics["ridge_accuracy_on_disagreements"])
    assert pd.isna(diagnostics["tcn_accuracy_on_disagreements"])


def test_ewma_only_lock_writes_2025_artifact_contract(
    demo_bundle, tmp_path
):
    lock = _ewma_only_lock(demo_bundle)
    result = run_locked_final_evaluation(
        demo_bundle,
        lock,
        output_dir=tmp_path,
    )

    predictions = result["predictions"]
    assert set(predictions["year"].astype(int)) == {2025}
    assert set(predictions["fold_year"].astype(int)) == {2025}
    assert set(predictions["model"]) == {"ewma"}
    assert result["report"]["train_years"] == [2020, 2021, 2022, 2023, 2024]
    assert result["report"]["test_year"] == 2025
    assert result["report"]["selection_replayed"] is False
    assert result["report"]["hyperparameters_reselected"] is False
    assert result["report"]["models_evaluated"] == ["ewma"]

    for filename in FINAL_ARTIFACTS:
        assert (tmp_path / filename).is_file(), filename
    assert (tmp_path / "confusion_matrix_ewma.csv").is_file()
    assert (tmp_path / "extreme_errors_ewma.csv").is_file()

    saved_predictions = pd.read_parquet(
        tmp_path / "final_2025_predictions.parquet"
    )
    assert set(saved_predictions["year"].astype(int)) == {2025}
    common = pd.read_csv(
        tmp_path / "final_2025_common_sample_metrics.csv"
    )
    deployable = pd.read_csv(
        tmp_path / "final_2025_deployable_metrics.csv"
    )
    assert common["model"].tolist() == ["ewma"]
    assert common["sample_type"].tolist() == ["common"]
    assert deployable["model"].tolist() == ["ewma"]
    assert deployable["sample_type"].tolist() == ["deployable"]
    for table in (common, deployable):
        assert {
            "n_total_eligible",
            "n_predicted",
            "coverage",
        }.issubset(table.columns)

    ordinal = pd.read_parquet(
        tmp_path / "ordinal_probabilities_2025.parquet"
    )
    tcn = pd.read_parquet(tmp_path / "tcn_predictions_2025.parquet")
    agreement = pd.read_parquet(
        tmp_path / "model_agreement_2025.parquet"
    )
    assert ordinal.empty
    assert set(ORDINAL_PREDICTION_COLUMNS).issubset(ordinal.columns)
    assert tcn.empty
    assert set(TCN_PREDICTION_COLUMNS).issubset(tcn.columns)
    assert agreement.empty
    assert set(AGREEMENT_COLUMNS).issubset(agreement.columns)

    mapping = pd.read_csv(
        tmp_path / "primary_fastball_mapping_by_fold.csv"
    )
    assert set(mapping["fold_year"].astype(int)) == {2025}
    report = json.loads(
        (tmp_path / "final_report.json").read_text(encoding="utf-8")
    )
    assert report["models_evaluated"] == ["ewma"]


def test_inactive_requested_model_is_not_invented(demo_bundle, tmp_path):
    lock = deepcopy(_ewma_only_lock(demo_bundle))
    result = run_locked_final_evaluation(
        demo_bundle,
        lock,
        output_dir=tmp_path,
        models=("ewma", "ridge", "xgboost", "ordinal", "tcn"),
    )
    assert set(result["predictions"]["model"]) == {"ewma"}


def test_all_requested_models_inactive_fails(demo_bundle, tmp_path):
    lock = _ewma_only_lock(demo_bundle)
    with pytest.raises(ValueError, match="No requested model is active"):
        run_locked_final_evaluation(
            demo_bundle,
            lock,
            output_dir=tmp_path,
            models=("ridge",),
        )
