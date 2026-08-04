from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lib.stuff_selection as selection
from lib.evaluation import load_locked_model_config
from lib.stuff_cli import load_stuff_data
from lib.stuff_experiment import predict_ewma_fold as real_predict_ewma_fold
import select_models


@dataclass
class _DummyEstimator:
    backend_: str = "test-double"


def _fake_residual(
    fold,
    *,
    model,
    correction_scale,
    candidate_id,
    **kwargs,
):
    frame = real_predict_ewma_fold(fold)
    frame["model"] = model
    frame["candidate_id"] = candidate_id
    frame["predicted_residual"] = 0.0
    frame["correction_scale"] = float(correction_scale)
    frame["parameter_count"] = 16
    if model == "xgboost" and not frame.empty:
        first = frame.index[0]
        frame.loc[first, "predicted_class"] = np.nan
        frame.loc[first, "deployable"] = False
    return frame, _DummyEstimator()


def _fake_ordinal(fold, *, config, candidate_id, **kwargs):
    frame = real_predict_ewma_fold(fold)
    frame["model"] = (
        "ordinal_linear_distance"
        if float(config.distance_weight) > 0
        else "ordinal_linear"
    )
    frame["candidate_id"] = candidate_id
    classes = pd.to_numeric(frame["predicted_class"], errors="coerce")
    for value, column in enumerate(("prob_low", "prob_middle", "prob_high")):
        frame[column] = classes.eq(value).astype(float)
    frame["latent_score"] = classes
    frame["threshold_1"] = 0.5
    frame["threshold_2"] = 1.5
    frame["parameter_count"] = 48
    return frame, object()


def _fake_tcn(
    fold,
    *,
    config,
    sequence_order,
    candidate_id,
    **kwargs,
):
    frame = real_predict_ewma_fold(fold)
    model = f"tcn_{config.head_type.lower()}_{sequence_order}"
    if float(config.ordinal_weight) > 0:
        model = "tcn_h1_residual_ordinal_aux"
    frame["model"] = model
    frame["candidate_id"] = candidate_id
    frame["predicted_residual"] = 0.0
    frame["global_residual"] = 0.0
    frame["pitcher_correction"] = 0.0
    frame["seed"] = int(config.seed)
    frame["alpha"] = float(config.alpha)
    frame["head_type"] = str(config.head_type).upper()
    frame["sequence_order"] = sequence_order
    frame["sequence_length"] = int(config.sequence_length)
    frame["internal_channels"] = int(config.internal_channels)
    frame["bottleneck_dim"] = int(config.bottleneck_dim)
    classes = pd.to_numeric(frame["predicted_class"], errors="coerce")
    for value, column in enumerate(("prob_low", "prob_middle", "prob_high")):
        frame[column] = classes.eq(value).astype(float)
    frame["ordinal_score"] = classes
    frame["parameter_count"] = {
        "H0": 80,
        "H1": 90,
        "H2": 100,
    }[str(config.head_type).upper()]
    frame["preprocessing_fit_end"] = fold.spec.train_end
    return frame, object(), pd.DataFrame()


def _fake_xgboost_search(development, *, n_trials, random_seed):
    years = pd.to_numeric(development["year"], errors="raise")
    assert int(years.max()) <= 2024
    assert not years.eq(2025).any()
    result = {
        "protocol": "staged_tpe_32_to_10_to_3",
        "tpe_best_trial": 0,
        "candidate_id": "xgboost_test",
        "profile": {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.05,
        },
        "correction_scale": 0.5,
        "selection_score": 0.5,
        "mean_balanced_accuracy": 0.5,
        "std_balanced_accuracy": 0.0,
        "validation_years": [2022, 2023, 2024],
        "random_seed": int(random_seed),
        "n_trials": int(n_trials),
    }
    trials = pd.DataFrame(
        [{"stage": "test", "trial": 0, "selection_score": 0.5}]
    )
    return result, trials


def test_smoke_selection_writes_complete_development_only_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = load_stuff_data(
        demo=True,
        demo_pitchers=1,
        demo_outings_per_season=6,
    )
    monkeypatch.setattr(
        selection, "search_xgboost_tpe", _fake_xgboost_search
    )
    monkeypatch.setattr(
        selection, "predict_residual_fold", _fake_residual
    )
    monkeypatch.setattr(
        selection, "predict_ordinal_fold", _fake_ordinal
    )
    monkeypatch.setattr(selection, "predict_tcn_fold", _fake_tcn)

    result = selection.run_model_selection(
        bundle,
        output_dir=tmp_path,
        random_seeds=(selection.DEFAULT_RANDOM_SEEDS[0],),
        sequence_lengths=(5,),
        tcn_epochs=1,
        xgboost_trials=1,
        smoke_test=True,
    )

    expected_files = {
        "feature_coverage.csv",
        "fold_manifest.parquet",
        "locked_model_config.json",
        "primary_fastball_mapping_by_fold.csv",
        "selection_report.json",
        "validation_metrics.csv",
        "validation_metrics_by_seed.csv",
        "validation_metrics_by_year.csv",
        "validation_metrics_summary.csv",
        "validation_oof_predictions.parquet",
        "xgboost_tpe_trials.csv",
    }
    assert expected_files <= {path.name for path in tmp_path.iterdir()}

    oof = result["oof_predictions"]
    assert set(pd.to_numeric(oof["fold_year"]).unique()) == {
        2022,
        2023,
        2024,
    }
    assert int(pd.to_numeric(oof["year"]).max()) <= 2024
    metrics = result["metrics_by_year"]
    assert set(metrics["sample_type"]) == {"common", "deployable"}
    common = metrics.loc[metrics["sample_type"].eq("common")]
    assert common["coverage"].eq(1.0).all()
    xgboost_deployable = metrics.loc[
        metrics["model"].eq("xgboost")
        & metrics["sample_type"].eq("deployable")
    ]
    assert xgboost_deployable["coverage"].lt(1.0).all()
    coverage = result["feature_coverage"]
    expected_feature_counts = {
        "tabular_compact_16": len(selection.COMPACT_FEATURES),
        "tcn_raw_15": len(selection.TCN_RAW_FEATURES),
    }
    for representation, expected_count in expected_feature_counts.items():
        per_fold = coverage.loc[
            coverage["representation"].eq(representation)
        ].groupby("fold_year")["feature"].nunique()
        assert per_fold.eq(expected_count).all()

    lock = load_locked_model_config(tmp_path / "locked_model_config.json")
    assert lock["validation_years"] == [2022, 2023, 2024]
    assert lock["selection_data_max_year"] == 2024
    assert lock["selection_uses_2025_outcomes"] is False
    assert (
        lock["qualification"]["selection_uses_2025_outcomes"]
        is False
    )
    assert lock["smoke_test"] is True
    assert lock["ridge"]["status"] == "selected"
    assert lock["ordinal_linear"]["status"] == "selected"
    assert lock["shared_tcn"]["status"] == "selected"
    assert lock["ridge"]["candidate_grid"] == {
        "alpha": [100.0],
        "correction_scale": [0.5],
    }
    assert lock["ordinal_linear"]["candidate_grid"]["l2_lambda"] == [1.0]
    assert lock["shared_tcn"]["candidate_grid"]["sequence_length"] == [5]
    assert lock["shared_tcn"]["candidate_grid"]["alpha"] == [0.25]
    assert lock["shared_tcn"]["candidate_grid"]["epochs"] == 1
    assert (
        lock["shared_tcn"]["ensemble"]["selected_candidate_id"]
        is not None
    )

    candidate_ids = set(lock["candidate_configs"])
    for required in (
        "tcn_h0_global_only",
        "tcn_h1_pitcher_bias",
        "tcn_h2_pitcher_vector",
        "tcn_h1_shuffled",
        "tcn_h1_ordinal_aux",
    ):
        assert any(value.startswith(required) for value in candidate_ids)
    assert any(value.endswith("_ordinal_output") for value in candidate_ids)
    selected_tcn = lock["shared_tcn"]["selected"]
    assert "seed" not in selected_tcn
    assert selected_tcn["seeds"] == [selection.DEFAULT_RANDOM_SEEDS[0]]
    assert {
        "ensemble_beta0",
        "ensemble_beta0.1",
        "ensemble_beta0.2",
        "ensemble_beta0.3",
    } <= candidate_ids


def test_normal_tcn_selection_rejects_fewer_than_five_seeds(
    tmp_path: Path,
) -> None:
    bundle = load_stuff_data(
        demo=True,
        demo_pitchers=1,
        demo_outings_per_season=4,
    )
    with pytest.raises(ValueError, match="at least five"):
        selection.run_model_selection(
            bundle,
            output_dir=tmp_path / "not-created",
            models=("tcn",),
            random_seeds=selection.DEFAULT_RANDOM_SEEDS[:4],
            sequence_lengths=(5,),
            tcn_epochs=1,
            xgboost_trials=1,
            smoke_test=False,
        )
    assert not (tmp_path / "not-created").exists()


def test_beta_zero_ensemble_matches_ridge_across_fold_thresholds() -> None:
    ridge = pd.DataFrame(
        {
            "row_id": ["r-2022", "r-2023"],
            "fold_year": [2022, 2023],
            "pitcher": [1, 1],
            "model": ["ridge", "ridge"],
            "candidate_id": ["ridge-selected", "ridge-selected"],
            "predicted_residual": [8.0, 8.0],
            "predicted_stuff_plus": [99.0, 99.0],
            "predicted_class": [1.0, 0.0],
            "correction_scale": [0.5, 0.5],
            "ewma4": [95.0, 95.0],
            "q33": [90.0, 100.0],
            "q67": [100.0, 110.0],
            "true_class": [1, 0],
            "target_eligible": [True, True],
            "parameter_count": [10, 10],
        }
    )
    tcn = ridge.copy()
    tcn["model"] = "tcn_h1_ordered"
    tcn["candidate_id"] = "tcn-selected"
    tcn["seed"] = selection.DEFAULT_RANDOM_SEEDS[0]
    tcn["alpha"] = 0.25
    tcn["head_type"] = "H1"
    tcn["sequence_order"] = "ordered"
    tcn["sequence_length"] = 5
    tcn["internal_channels"] = 8
    tcn["bottleneck_dim"] = 4
    tcn["feature_eligible"] = True
    tcn["deployable"] = True
    tcn["predicted_residual"] = [-4.0, -4.0]

    ensembles = selection._ensemble_candidates(
        pd.concat([ridge, tcn], ignore_index=True),
        ridge_candidate_id="ridge-selected",
        tcn_candidate_id="tcn-selected",
    )
    beta_zero = next(
        frame
        for frame in ensembles
        if frame["candidate_id"].eq("ensemble_beta0").all()
    )
    assert beta_zero["predicted_stuff_plus"].tolist() == [99.0, 99.0]
    assert beta_zero["predicted_class"].tolist() == [1.0, 0.0]


def test_selection_never_falls_back_from_common_to_deployable_rows() -> None:
    rows = []
    for fold_year in (2022, 2023, 2024):
        for model, suffix in (("ewma", "a"), ("ridge", "b")):
            rows.append(
                {
                    "row_id": f"{fold_year}-{suffix}",
                    "fold_year": fold_year,
                    "model": model,
                    "candidate_id": f"{model}-candidate",
                    "seed": np.nan,
                    "target_eligible": True,
                    "true_class": 1,
                    "predicted_class": 1,
                    "parameter_count": 0,
                }
            )
    with pytest.raises(ValueError, match="No common evaluable row"):
        selection._metrics_for_predictions(pd.DataFrame(rows))


def test_tcn_selection_score_penalizes_seed_instability() -> None:
    rows = []
    true_classes = [0, 1, 2]
    for fold_year in (2022, 2023, 2024):
        for seed in (20260722, 20260723):
            for candidate_id, predicted_classes in (
                ("stable", [0, 1, 0]),
                (
                    "unstable",
                    [0, 1, 2] if seed == 20260722 else [0, 0, 0],
                ),
            ):
                for position, (true_class, predicted_class) in enumerate(
                    zip(
                        true_classes,
                        predicted_classes,
                        strict=True,
                    )
                ):
                    rows.append(
                        {
                            "row_id": f"{fold_year}-{position}",
                            "fold_year": fold_year,
                            "model": "tcn_h1_ordered",
                            "candidate_id": candidate_id,
                            "seed": seed,
                            "target_eligible": True,
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "parameter_count": 100,
                        }
                    )

    _, summary = selection._metrics_for_predictions(pd.DataFrame(rows))
    candidates = summary.set_index("candidate_id")
    assert candidates.loc["stable", "mean_balanced_accuracy"] == pytest.approx(
        2.0 / 3.0
    )
    assert candidates.loc[
        "unstable", "mean_balanced_accuracy"
    ] == pytest.approx(2.0 / 3.0)
    assert candidates.loc["stable", "std_balanced_accuracy"] == 0.0
    assert candidates.loc[
        "unstable", "std_balanced_accuracy"
    ] == pytest.approx(1.0 / 3.0)
    assert candidates.loc[
        "stable", "seed_standard_deviation"
    ] == 0.0
    assert candidates.loc[
        "unstable", "seed_standard_deviation"
    ] == pytest.approx(1.0 / 3.0)
    assert candidates.loc["stable", "year_standard_deviation"] == 0.0
    assert candidates.loc["unstable", "year_standard_deviation"] == 0.0
    assert candidates.loc[
        "stable", "n_seed_year_observations"
    ] == 6
    assert (
        candidates.loc["stable", "selection_score"]
        > candidates.loc["unstable", "selection_score"]
    )


def test_cli_profiles_keep_normal_and_smoke_runs_separate() -> None:
    parser = select_models.make_parser()
    smoke = parser.parse_args(["--demo", "--smoke-test"])
    smoke_profile = select_models._resolved_profile(smoke)
    assert smoke_profile == {
        "random_seeds": (selection.DEFAULT_RANDOM_SEEDS[0],),
        "sequence_lengths": (5,),
        "tcn_epochs": 2,
        "xgboost_trials": 1,
        "smoke_test": True,
    }

    normal = parser.parse_args(["--demo"])
    normal_profile = select_models._resolved_profile(normal)
    assert normal_profile["random_seeds"] == selection.DEFAULT_RANDOM_SEEDS
    assert normal_profile["sequence_lengths"] == (5, 8)
    assert normal_profile["tcn_epochs"] == 250
    assert normal_profile["xgboost_trials"] == 32
    assert normal_profile["smoke_test"] is False

    invalid = parser.parse_args(
        ["--demo", "--models", "tcn", "--num-seeds", "4"]
    )
    with pytest.raises(ValueError, match="Normal TCN selection"):
        select_models._resolved_profile(invalid)


def test_normal_tcn_plan_is_predeclared_and_not_full_cartesian() -> None:
    templates = selection._tcn_templates(smoke_test=False)
    plan = selection._tcn_search_plan(
        templates,
        (5, 8),
        smoke_test=False,
    )
    assert len(plan) == len(templates) + 1
    assert [
        template["name"]
        for length, template in plan
        if length == 5
    ] == ["tcn_h1_pitcher_bias"]
    assert {
        template["name"]
        for length, template in plan
        if length == 8
    } == {template["name"] for template in templates}

    smoke_templates = selection._tcn_templates(smoke_test=True)
    smoke_plan = selection._tcn_search_plan(
        smoke_templates,
        (5,),
        smoke_test=True,
    )
    assert len(smoke_plan) == len(smoke_templates)
