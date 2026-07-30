"""Replay an immutable Stuff+ model lock on the one-shot 2025 test fold.

This module deliberately contains no model-search code.  It validates the
recorded analysis contract, refits only the models marked active in that
contract on 2020-2024, and writes diagnostics for the 2025 rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from build_fold_manifest import build_manifest
from lib.evaluation import (
    confusion_matrix_frame,
    evaluate_common_and_deployable,
    evaluate_prediction_frame,
    validate_locked_fingerprints,
    validate_locked_model_config,
)
from lib.ordinal_linear import OrdinalLinearConfig
from lib.shared_tcn import SharedTCNConfig
from lib.stuff_cli import StuffDataBundle
from lib.stuff_code_version import analysis_code_version
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
from lib.stuff_tabular import classify_stuff_plus


FINAL_MODEL_FAMILIES = ("ewma", "ridge", "xgboost", "ordinal", "tcn")
LOCK_STATUSES = frozenset(
    {"selected", "not_requested", "no_deployable_candidate"}
)

TABULAR_COMPLETE_CASE_POLICY = (
    "complete-case on actual configured model features only; "
    "train-fitted Ridge scaling"
)
ORDINAL_TRAIN_ONLY_POLICY = (
    "per-pitcher training imputation/scaling with pooled training fallback"
)
TCN_TRAIN_ONLY_POLICY = (
    "pitcher training median, pooled training median fallback; "
    "pitcher training scaling; availability and valid-timestep masks"
)
ENSEMBLE_CORRECTION_POLICY = (
    "blend each model's scaled residual correction, then add EWMA4"
)

FINAL_ARTIFACTS = (
    "final_2025_predictions.parquet",
    "final_2025_common_sample_metrics.csv",
    "final_2025_deployable_metrics.csv",
    "final_2025_per_pitcher_metrics.csv",
    "ordinal_probabilities_2025.parquet",
    "tcn_predictions_2025.parquet",
    "model_agreement_2025.parquet",
    "feature_coverage.csv",
    "primary_fastball_mapping_by_fold.csv",
)

TCN_PREDICTION_COLUMNS = (
    "row_id",
    "pitcher",
    "game_date",
    "true_stuff_plus",
    "ewma4",
    "true_residual",
    "predicted_residual",
    "predicted_stuff_plus",
    "q33",
    "q67",
    "true_class",
    "predicted_class",
    "prob_low",
    "prob_middle",
    "prob_high",
    "seed",
    "head_type",
    "sequence_length",
    "internal_channels",
    "bottleneck_dim",
    "alpha",
)

ORDINAL_PREDICTION_COLUMNS = (
    "row_id",
    "pitcher",
    "game_date",
    "year",
    "model",
    "candidate_id",
    "true_class",
    "predicted_class",
    "prob_low",
    "prob_middle",
    "prob_high",
    "latent_score",
    "threshold_1",
    "threshold_2",
)

AGREEMENT_COLUMNS = (
    "row_id",
    "pitcher",
    "game_date",
    "true_class",
    "ridge_predicted_class",
    "tcn_predicted_class",
    "ridge_predicted_residual",
    "tcn_predicted_residual",
    "category_agreement",
    "ridge_correct",
    "tcn_correct",
    "tcn_fixed_ridge_error",
    "tcn_new_extreme_error",
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _require_keys(
    value: Mapping[str, Any],
    keys: Sequence[str],
    *,
    label: str,
) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{label} is missing locked fields: {missing}")


def _model_is_active(
    block: Mapping[str, Any],
    *,
    label: str,
    candidate_key: str = "selected_candidate_id",
) -> bool:
    status = block.get("status")
    if status not in LOCK_STATUSES:
        raise ValueError(
            f"{label}.status must be one of {sorted(LOCK_STATUSES)}"
        )
    candidate = block.get(candidate_key)
    if status == "selected":
        if not isinstance(candidate, str) or not candidate:
            raise ValueError(
                f"{label} is selected but {candidate_key} is missing"
            )
        return True
    if candidate not in (None, ""):
        raise ValueError(
            f"{label} is inactive but has {candidate_key}={candidate!r}"
        )
    return False


def _locked_candidate_config(
    lock: Mapping[str, Any],
    block: Mapping[str, Any],
    candidate_id: str,
    *,
    label: str,
) -> dict[str, Any]:
    selected = _mapping(block.get("selected"), label=f"{label}.selected")
    if not selected:
        raise ValueError(f"{label}.selected cannot be empty for an active model")
    catalog = lock.get("candidate_configs")
    if catalog is not None:
        catalog_map = _mapping(catalog, label="candidate_configs")
        if candidate_id not in catalog_map:
            raise ValueError(
                f"{label} candidate {candidate_id!r} is absent from "
                "candidate_configs"
            )
        catalog_value = _mapping(
            catalog_map[candidate_id],
            label=f"candidate_configs[{candidate_id!r}]",
        )
        if catalog_value != selected:
            raise ValueError(
                f"{label}.selected differs from candidate_configs"
            )
    return selected


def _validate_target_contract(target: Mapping[str, Any]) -> None:
    _require_keys(
        target,
        ("type", "q_low", "q_high"),
        label="target_definition",
    )
    if target["type"] != "pitcher_specific_training_tertile":
        raise ValueError("Unsupported locked target definition")
    if not np.isclose(float(target["q_low"]), 1.0 / 3.0, atol=1e-9):
        raise ValueError("Locked lower tertile probability is not 1/3")
    if not np.isclose(float(target["q_high"]), 2.0 / 3.0, atol=1e-9):
        raise ValueError("Locked upper tertile probability is not 2/3")


def _validate_feature_contract(lock: Mapping[str, Any]) -> None:
    feature_lists = _mapping(lock["feature_lists"], label="feature_lists")
    _require_keys(
        feature_lists,
        ("tabular_compact_16", "ordinal", "tcn_raw_15"),
        label="feature_lists",
    )
    expected = {
        "tabular_compact_16": tuple(COMPACT_FEATURES),
        "ordinal": tuple(ORDINAL_FEATURES),
        "tcn_raw_15": tuple(TCN_RAW_FEATURES),
    }
    for name, implemented in expected.items():
        locked = tuple(feature_lists[name])
        if locked != implemented:
            raise ValueError(
                f"Locked {name} feature contract differs from the "
                "implemented contract"
            )


def _tabular_missing_policy(lock: Mapping[str, Any]) -> str:
    policies = _mapping(
        lock["missing_value_policy"], label="missing_value_policy"
    )
    _require_keys(
        policies,
        ("tabular", "ordinal", "tcn"),
        label="missing_value_policy",
    )
    if policies["tabular"] != TABULAR_COMPLETE_CASE_POLICY:
        raise ValueError(
            "Unsupported locked tabular missing-value policy: "
            f"{policies['tabular']!r}"
        )
    if policies["ordinal"] != ORDINAL_TRAIN_ONLY_POLICY:
        raise ValueError(
            "Unsupported locked ordinal missing-value policy: "
            f"{policies['ordinal']!r}"
        )
    if policies["tcn"] != TCN_TRAIN_ONLY_POLICY:
        raise ValueError(
            "Unsupported locked TCN missing-value policy: "
            f"{policies['tcn']!r}"
        )
    return "complete_case"


def _validate_replay_contract(lock: Mapping[str, Any]) -> dict[str, bool]:
    if (
        int(lock["train_start"]) != 2020
        or tuple(int(year) for year in lock["validation_years"])
        != (2022, 2023, 2024)
        or int(lock["final_train_end"]) != 2024
        or int(lock["test_year"]) != 2025
    ):
        raise ValueError(
            "Final replay requires the locked 2020-2024 -> 2025 protocol "
            "with validation years 2022-2024"
        )
    _validate_target_contract(
        _mapping(lock["target_definition"], label="target_definition")
    )
    _validate_feature_contract(lock)
    _tabular_missing_policy(lock)

    ewma = _mapping(lock["ewma"], label="ewma")
    ridge = _mapping(lock["ridge"], label="ridge")
    xgboost = _mapping(lock["xgboost"], label="xgboost")
    ordinal = _mapping(lock["ordinal_linear"], label="ordinal_linear")
    shared_tcn = _mapping(lock["shared_tcn"], label="shared_tcn")
    for label, block in (
        ("ridge", ridge),
        ("xgboost", xgboost),
        ("ordinal_linear", ordinal),
        ("shared_tcn", shared_tcn),
    ):
        _require_keys(
            block,
            ("status", "selected_candidate_id", "selected"),
            label=label,
        )

    _require_keys(ewma, ("status", "span", "candidate_id"), label="ewma")
    active = {
        "ewma": _model_is_active(
            ewma, label="ewma", candidate_key="candidate_id"
        ),
        "ridge": _model_is_active(ridge, label="ridge"),
        "xgboost": _model_is_active(xgboost, label="xgboost"),
        "ordinal": _model_is_active(ordinal, label="ordinal_linear"),
        "tcn": _model_is_active(shared_tcn, label="shared_tcn"),
    }
    if active["ewma"]:
        if int(ewma["span"]) != 4 or ewma["candidate_id"] != "ewma_span4":
            raise ValueError(
                "The implemented EWMA replay supports only locked EWMA4"
            )
        catalog = lock.get("candidate_configs")
        if isinstance(catalog, Mapping) and ewma["candidate_id"] in catalog:
            if dict(catalog[ewma["candidate_id"]]) != {
                "span": 4,
                "features": ["ewma4"],
            }:
                raise ValueError(
                    "EWMA block differs from its locked candidate config"
                )

    for label, block in (
        ("ridge", ridge),
        ("xgboost", xgboost),
        ("ordinal_linear", ordinal),
        ("shared_tcn", shared_tcn),
    ):
        family = "ordinal" if label == "ordinal_linear" else (
            "tcn" if label == "shared_tcn" else label
        )
        if active[family]:
            _locked_candidate_config(
                lock,
                block,
                str(block["selected_candidate_id"]),
                label=label,
            )

    ensemble = _mapping(
        shared_tcn.get("ensemble", {}), label="shared_tcn.ensemble"
    )
    _require_keys(
        ensemble,
        (
            "selected_candidate_id",
            "beta",
            "ridge_candidate_id",
            "tcn_candidate_id",
            "correction_policy",
        ),
        label="shared_tcn.ensemble",
    )
    ensemble_active = bool(ensemble["selected_candidate_id"])
    if ensemble_active:
        if not active["ridge"] or not active["tcn"]:
            raise ValueError(
                "An active locked ensemble requires active Ridge and TCN blocks"
            )
        if ensemble["correction_policy"] != ENSEMBLE_CORRECTION_POLICY:
            raise ValueError("Unsupported locked ensemble correction policy")
        beta = float(ensemble["beta"])
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Locked ensemble beta must be in [0, 1]")
        if not ensemble["ridge_candidate_id"] or not ensemble["tcn_candidate_id"]:
            raise ValueError("Locked ensemble component candidate IDs are missing")
        if str(ensemble["ridge_candidate_id"]) != str(
            ridge["selected_candidate_id"]
        ):
            raise ValueError(
                "Locked ensemble Ridge component differs from selected Ridge"
            )
        catalog = _mapping(
            lock.get("candidate_configs"), label="candidate_configs"
        )
        ensemble_id = str(ensemble["selected_candidate_id"])
        if ensemble_id not in catalog:
            raise ValueError(
                "Selected ensemble is absent from candidate_configs"
            )
        expected_ensemble_config = {
            "beta": float(ensemble["beta"]),
            "ridge_candidate_id": ensemble["ridge_candidate_id"],
            "tcn_candidate_id": ensemble["tcn_candidate_id"],
            "correction_policy": ensemble["correction_policy"],
        }
        if dict(catalog[ensemble_id]) != expected_ensemble_config:
            raise ValueError(
                "Ensemble block differs from its locked candidate config"
            )
        if str(ensemble["tcn_candidate_id"]) not in catalog:
            raise ValueError(
                "Locked ensemble TCN component is absent from candidate_configs"
            )
    active["ensemble"] = ensemble_active
    return active


def _validate_final_primary_mapping(
    fold_mapping: pd.DataFrame,
    locked_mapping: Mapping[str, Any],
) -> None:
    expected_raw = locked_mapping.get("final_primary_by_pitcher")
    if not isinstance(expected_raw, Mapping):
        raise ValueError(
            "pitch_type_mapping.final_primary_by_pitcher must be a mapping"
        )
    expected = {str(key): str(value) for key, value in expected_raw.items()}
    current = {
        str(row.pitcher): str(row.primary_fb_type)
        for row in fold_mapping.itertuples(index=False)
        if pd.notna(row.primary_fb_type)
    }
    if current != expected:
        raise ValueError(
            "Final primary-fastball mapping mismatch: the 2020-2024 "
            "training mapping differs from the lock"
        )


def _shared_tcn_config(
    raw: Mapping[str, Any],
    *,
    seed: int,
) -> SharedTCNConfig:
    allowed = {item.name for item in fields(SharedTCNConfig)}
    # The candidate is shared across fits.  Its lock records ``seeds`` as a
    # list, while the dataclass receives one member of that list per fit.
    missing = (allowed - {"seed"}) - set(raw)
    if missing:
        raise ValueError(
            "Locked TCN config is incomplete; missing fields: "
            f"{sorted(missing)}"
        )
    values = {key: raw[key] for key in allowed if key in raw}
    values["seed"] = int(seed)
    return SharedTCNConfig.from_value(values)


def _fit_locked_tcn(
    fold,
    *,
    candidate_id: str,
    candidate_config: Mapping[str, Any],
    seeds: Sequence[int],
    device: str,
    force_residual_output: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_keys(
        candidate_config,
        ("sequence_order", "shuffle_seed", "seeds"),
        label=f"TCN candidate {candidate_id!r}",
    )
    sequence_order = str(candidate_config["sequence_order"])
    shuffle_seed = int(candidate_config["shuffle_seed"])
    output_kind = str(candidate_config.get("output", "residual"))
    if output_kind not in {"residual", "ordinal_probability_argmax"}:
        raise ValueError(
            f"Unsupported locked TCN output kind: {output_kind!r}"
        )
    candidate_seeds = tuple(
        int(value) for value in candidate_config["seeds"]
    )
    if candidate_seeds != tuple(int(value) for value in seeds):
        raise ValueError(
            f"TCN candidate {candidate_id!r} seed list differs from "
            "random_seeds"
        )

    seed_predictions: list[pd.DataFrame] = []
    for seed in seeds:
        config = _shared_tcn_config(candidate_config, seed=int(seed))
        prediction, _, _ = predict_tcn_fold(
            fold,
            config=config,
            sequence_order=sequence_order,
            shuffle_seed=shuffle_seed,
            device=device,
            candidate_id=candidate_id,
        )
        prediction["candidate_id"] = candidate_id
        if (
            output_kind == "ordinal_probability_argmax"
            and not force_residual_output
        ):
            prediction = ordinal_auxiliary_prediction(prediction)
            prediction["candidate_id"] = candidate_id
        seed_predictions.append(prediction)

    all_seeds = pd.concat(seed_predictions, ignore_index=True, sort=False)
    observed_seeds = {
        int(value)
        for value in pd.to_numeric(all_seeds["seed"], errors="raise").unique()
    }
    if observed_seeds != {int(seed) for seed in seeds}:
        raise AssertionError("TCN predictions do not contain the locked seeds")
    averaged = average_seed_predictions(all_seeds)
    if not averaged.empty:
        seed_counts = pd.to_numeric(
            averaged["seed_count"], errors="raise"
        ).astype(int)
        if not seed_counts.eq(len(seeds)).all():
            raise AssertionError(
                "TCN aggregate does not average every locked seed"
            )
    return all_seeds, averaged


def _as_residual_tcn_average(seed_rows: pd.DataFrame) -> pd.DataFrame:
    residual_rows = seed_rows.copy()
    residual_rows["model"] = residual_rows["model"].where(
        ~residual_rows["model"].eq("tcn_ordinal"),
        "tcn_h1_residual_ordinal_aux",
    )
    return average_seed_predictions(residual_rows)


def _locked_ensemble(
    ridge: pd.DataFrame,
    tcn: pd.DataFrame,
    *,
    candidate_id: str,
    beta: float,
    correction_policy: str,
) -> pd.DataFrame:
    if correction_policy != ENSEMBLE_CORRECTION_POLICY:
        raise ValueError("Unsupported locked ensemble correction policy")
    tcn_columns = [
        "row_id",
        "predicted_residual",
        "parameter_count",
        "feature_eligible",
    ]
    merged = ridge.merge(
        tcn[tcn_columns].rename(
            columns={
                "predicted_residual": "tcn_predicted_residual",
                "parameter_count": "tcn_parameter_count",
                "feature_eligible": "tcn_feature_eligible",
            }
        ),
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    merged["ridge_predicted_residual"] = pd.to_numeric(
        merged["predicted_residual"], errors="coerce"
    )
    ridge_scale = pd.to_numeric(
        merged["correction_scale"], errors="raise"
    )
    if ridge_scale.nunique(dropna=False) != 1:
        raise ValueError("Locked Ridge correction scale is not constant")
    if "alpha" not in tcn:
        raise ValueError("Locked TCN ensemble component has no alpha")
    tcn_alpha_by_row = tcn.loc[:, ["row_id", "alpha"]].rename(
        columns={"alpha": "tcn_alpha"}
    )
    merged = merged.merge(
        tcn_alpha_by_row,
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    tcn_alpha = pd.to_numeric(merged["tcn_alpha"], errors="raise")
    if tcn_alpha.nunique(dropna=False) != 1:
        raise ValueError("Locked TCN alpha is not constant")
    merged["ridge_residual_correction"] = (
        ridge_scale * merged["ridge_predicted_residual"]
    )
    merged["tcn_residual_correction"] = (
        tcn_alpha
        * pd.to_numeric(merged["tcn_predicted_residual"], errors="coerce")
    )
    merged["predicted_residual"] = (
        (1.0 - float(beta)) * merged["ridge_residual_correction"]
        + float(beta) * merged["tcn_residual_correction"]
    )
    merged["predicted_stuff_plus"] = (
        pd.to_numeric(merged["ewma4"], errors="coerce")
        + merged["predicted_residual"]
    )
    merged["predicted_class"] = classify_stuff_plus(
        merged["predicted_stuff_plus"],
        merged["pitcher"],
        merged.loc[:, ["pitcher", "q33", "q67"]].drop_duplicates(),
    )
    merged["model"] = "ridge_tcn_ensemble"
    merged["candidate_id"] = str(candidate_id)
    merged["beta"] = float(beta)
    merged["seed"] = "mean"
    merged["correction_policy"] = correction_policy
    merged["correction_scale"] = 1.0
    merged["feature_eligible"] = (
        np.isfinite(merged["ridge_predicted_residual"])
        & np.isfinite(merged["tcn_predicted_residual"])
    )
    merged["deployable"] = np.isfinite(merged["predicted_class"])
    ridge_parameters = pd.to_numeric(
        merged["parameter_count"], errors="coerce"
    ).fillna(0)
    tcn_parameters = pd.to_numeric(
        merged["tcn_parameter_count"], errors="coerce"
    ).fillna(0)
    merged["parameter_count"] = np.where(
        np.isclose(float(beta), 0.0),
        ridge_parameters,
        ridge_parameters + tcn_parameters,
    )
    return merged


def _model_agreement(
    ridge: pd.DataFrame,
    tcn: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = ridge[
        [
            "row_id",
            "pitcher",
            "game_date",
            "true_class",
            "predicted_class",
            "predicted_residual",
        ]
    ].rename(
        columns={
            "predicted_class": "ridge_predicted_class",
            "predicted_residual": "ridge_predicted_residual",
        }
    )
    right = tcn[
        ["row_id", "predicted_class", "predicted_residual"]
    ].rename(
        columns={
            "predicted_class": "tcn_predicted_class",
            "predicted_residual": "tcn_predicted_residual",
        }
    )
    frame = left.merge(right, on="row_id", how="inner", validate="one_to_one")
    true_class = pd.to_numeric(
        frame["true_class"], errors="coerce"
    ).astype(float)
    ridge_class = pd.to_numeric(
        frame["ridge_predicted_class"], errors="coerce"
    ).astype(float)
    tcn_class = pd.to_numeric(
        frame["tcn_predicted_class"], errors="coerce"
    ).astype(float)
    both_deployable = (
        true_class.isin([0.0, 1.0, 2.0])
        & ridge_class.isin([0.0, 1.0, 2.0])
        & tcn_class.isin([0.0, 1.0, 2.0])
    )
    frame["both_deployable"] = both_deployable
    for column in (
        "category_agreement",
        "ridge_correct",
        "tcn_correct",
        "tcn_fixed_ridge_error",
        "tcn_new_extreme_error",
    ):
        frame[column] = pd.Series(
            pd.NA, index=frame.index, dtype="boolean"
        )
    if both_deployable.any():
        valid_index = frame.index[both_deployable]
        category_agreement = ridge_class.loc[valid_index].eq(
            tcn_class.loc[valid_index]
        )
        ridge_correct = ridge_class.loc[valid_index].eq(
            true_class.loc[valid_index]
        )
        tcn_correct = tcn_class.loc[valid_index].eq(
            true_class.loc[valid_index]
        )
        ridge_extreme = (
            ridge_class.loc[valid_index] - true_class.loc[valid_index]
        ).abs().eq(2)
        tcn_extreme = (
            tcn_class.loc[valid_index] - true_class.loc[valid_index]
        ).abs().eq(2)
        frame.loc[valid_index, "category_agreement"] = (
            category_agreement.to_numpy(dtype=bool)
        )
        frame.loc[valid_index, "ridge_correct"] = (
            ridge_correct.to_numpy(dtype=bool)
        )
        frame.loc[valid_index, "tcn_correct"] = (
            tcn_correct.to_numpy(dtype=bool)
        )
        frame.loc[valid_index, "tcn_fixed_ridge_error"] = (
            (~ridge_correct & tcn_correct).to_numpy(dtype=bool)
        )
        frame.loc[valid_index, "tcn_new_extreme_error"] = (
            (tcn_extreme & ~ridge_extreme).to_numpy(dtype=bool)
        )
    residuals = frame[
        ["ridge_predicted_residual", "tcn_predicted_residual"]
    ].apply(pd.to_numeric, errors="coerce")
    correlation = residuals.corr().iloc[0, 1] if len(frame) else np.nan
    comparable = frame.loc[both_deployable]
    disagreement = comparable.loc[
        ~comparable["category_agreement"].astype(bool)
    ]
    diagnostics = {
        "n_rows": int(len(frame)),
        "n_common_deployable": int(len(comparable)),
        "category_agreement_rate": (
            float(comparable["category_agreement"].astype(bool).mean())
            if len(comparable)
            else np.nan
        ),
        "residual_prediction_correlation": (
            float(correlation) if np.isfinite(correlation) else np.nan
        ),
        "disagreement_rows": int(len(disagreement)),
        "ridge_accuracy_on_disagreements": (
            float(disagreement["ridge_correct"].astype(bool).mean())
            if len(disagreement)
            else np.nan
        ),
        "tcn_accuracy_on_disagreements": (
            float(disagreement["tcn_correct"].astype(bool).mean())
            if len(disagreement)
            else np.nan
        ),
        "ridge_errors_fixed_by_tcn": int(
            comparable["tcn_fixed_ridge_error"].astype(bool).sum()
        ),
        "new_tcn_extreme_errors": int(
            comparable["tcn_new_extreme_error"].astype(bool).sum()
        ),
    }
    return frame, diagnostics


def _per_pitcher_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, pitcher), frame in predictions.groupby(
        ["model", "pitcher"], sort=True
    ):
        rows.append(
            {
                "model": model,
                "pitcher": pitcher,
                **evaluate_prediction_frame(frame),
            }
        )
    return pd.DataFrame(rows)


def _write_diagnostics(
    predictions_by_model: Mapping[str, pd.DataFrame],
    output: Path,
) -> None:
    for model_name, frame in predictions_by_model.items():
        safe = _safe_name(model_name)
        confusion_matrix_frame(
            frame["true_class"], frame["predicted_class"]
        ).to_csv(output / f"confusion_matrix_{safe}.csv")
        true = pd.to_numeric(frame["true_class"], errors="coerce")
        predicted = pd.to_numeric(frame["predicted_class"], errors="coerce")
        frame.loc[(true - predicted).abs().eq(2)].to_csv(
            output / f"extreme_errors_{safe}.csv", index=False
        )


def _feature_coverage(
    bundle: StuffDataBundle,
    final_fold,
    feature_lists: Mapping[str, Any],
) -> pd.DataFrame:
    features = tuple(
        dict.fromkeys(
            [
                *feature_lists["tabular_compact_16"],
                *feature_lists["ordinal"],
                *feature_lists["tcn_raw_15"],
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    for feature in features:
        source = (
            final_fold.evaluation_targets[feature]
            if feature in final_fold.evaluation_targets
            else (
                bundle.outings[feature]
                if feature in bundle.outings
                else pd.Series(np.nan, index=bundle.outings.index)
            )
        )
        available = pd.to_numeric(source, errors="coerce").notna()
        rows.append(
            {
                "feature": feature,
                "n_rows": int(len(source)),
                "n_available": int(available.sum()),
                "coverage": float(available.mean()) if len(source) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _ensure_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
    leading = list(columns)
    trailing = [column for column in result if column not in leading]
    return result.loc[:, [*leading, *trailing]]


def _assert_final_year(frame: pd.DataFrame, *, label: str) -> None:
    if frame.empty:
        return
    year = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if not year.eq(2025).all():
        raise AssertionError(f"{label} contains a non-2025 prediction")
    if "fold_year" in frame:
        fold_year = pd.to_numeric(
            frame["fold_year"], errors="raise"
        ).astype(int)
        if not fold_year.eq(2025).all():
            raise AssertionError(f"{label} contains a non-test fold")


def run_locked_final_evaluation(
    bundle: StuffDataBundle,
    locked_config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    models: Sequence[str] = FINAL_MODEL_FAMILIES,
    device: str = "auto",
    requested_num_seeds: int | None = None,
) -> dict[str, Any]:
    """Refit locked models on 2020-2024 and evaluate 2025 exactly once."""

    lock = validate_locked_model_config(
        locked_config, require_fingerprints=True
    )
    active = _validate_replay_contract(lock)
    current_code_version = analysis_code_version()
    if str(lock["code_version"]) != current_code_version:
        raise ValueError(
            "Analysis code version mismatch: "
            f"locked={lock['code_version']}, current={current_code_version}"
        )

    requested = tuple(dict.fromkeys(str(name).lower() for name in models))
    unknown = sorted(set(requested) - set(FINAL_MODEL_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown final model families: {unknown}")
    if not requested:
        raise ValueError("At least one final model family must be requested")
    seeds = tuple(int(value) for value in lock["random_seeds"])
    if (
        requested_num_seeds is not None
        and int(requested_num_seeds) != len(seeds)
    ):
        raise ValueError(
            "--num-seeds cannot alter a locked analysis: "
            f"lock has {len(seeds)}, requested {requested_num_seeds}"
        )

    manifest, manifest_fingerprint = build_manifest(bundle)
    validate_locked_fingerprints(
        lock,
        dataset_fingerprint=bundle.dataset_fingerprint,
        fold_manifest_fingerprint=manifest_fingerprint,
    )
    locked_fingerprint_columns = tuple(
        lock.get("dataset_fingerprint_columns", bundle.fingerprint_columns)
    )
    if locked_fingerprint_columns != tuple(bundle.fingerprint_columns):
        raise ValueError("Dataset fingerprint column contract differs from the lock")
    if [str(item) for item in bundle.qualified_pitchers] != [
        str(item) for item in lock["qualified_pitchers"]
    ]:
        raise ValueError("Qualified pitcher list differs from the lock")
    if bundle.qualification_rule != str(lock["qualification_rule"]):
        raise ValueError("Pitcher qualification rule differs from the lock")
    locked_cutter = lock["pitch_type_mapping"].get("cutter_category")
    current_cutter = bundle.source_metadata.get("cutter_category")
    if (
        locked_cutter is not None
        and current_cutter is not None
        and str(locked_cutter) != str(current_cutter)
    ):
        raise ValueError("Cutter category differs from the locked analysis")

    fold = prepare_fold(
        bundle.outings,
        2025,
        train_start=int(lock["train_start"]),
        evaluation_split="test",
    )
    if fold.spec.train_end != 2024 or fold.spec.fold_year != 2025:
        raise AssertionError("Final fold is not the locked 2020-2024 -> 2025 split")
    _validate_final_primary_mapping(
        fold.primary_fastball_mapping,
        _mapping(lock["pitch_type_mapping"], label="pitch_type_mapping"),
    )

    tabular_features = tuple(
        lock["feature_lists"]["tabular_compact_16"]
    )
    ordinal_features = tuple(lock["feature_lists"]["ordinal"])
    tabular_missing_policy = _tabular_missing_policy(lock)

    predictions_by_model: dict[str, pd.DataFrame] = {}
    tcn_seed_frames: dict[str, pd.DataFrame] = {}
    tcn_averages: dict[str, pd.DataFrame] = {}
    ridge_prediction: pd.DataFrame | None = None
    agreement_tcn: pd.DataFrame | None = None

    need_ridge = (
        "ridge" in requested
        and active["ridge"]
    )
    need_tcn = "tcn" in requested and active["tcn"]
    need_ensemble = (
        "ridge" in requested
        and "tcn" in requested
        and active["ensemble"]
    )

    if "ewma" in requested and active["ewma"]:
        ewma_prediction = predict_ewma_fold(fold)
        ewma_prediction["candidate_id"] = lock["ewma"]["candidate_id"]
        predictions_by_model["ewma"] = ewma_prediction

    if need_ridge:
        ridge_block = lock["ridge"]
        ridge_id = str(ridge_block["selected_candidate_id"])
        ridge_config = _locked_candidate_config(
            lock, ridge_block, ridge_id, label="ridge"
        )
        _require_keys(
            ridge_config,
            ("alpha", "correction_scale", "features", "missing_policy"),
            label="ridge.selected",
        )
        if tuple(ridge_config["features"]) != tabular_features:
            raise ValueError("Locked Ridge feature list is internally inconsistent")
        if ridge_config["missing_policy"] != tabular_missing_policy:
            raise ValueError(
                "Locked Ridge missing-value policy is internally inconsistent"
            )
        ridge_prediction, _ = predict_residual_fold(
            fold,
            model="ridge",
            features=tabular_features,
            ridge_alpha=float(ridge_config["alpha"]),
            correction_scale=float(ridge_config["correction_scale"]),
            candidate_id=ridge_id,
            missing_policy=tabular_missing_policy,
        )
        predictions_by_model["ridge"] = ridge_prediction

    if "xgboost" in requested and active["xgboost"]:
        xgb_block = lock["xgboost"]
        xgb_id = str(xgb_block["selected_candidate_id"])
        xgb_config = _locked_candidate_config(
            lock, xgb_block, xgb_id, label="xgboost"
        )
        _require_keys(
            xgb_config,
            (
                "correction_scale",
                "profile",
                "random_seed",
                "backend",
                "features",
                "missing_policy",
            ),
            label="xgboost.selected",
        )
        if tuple(xgb_config["features"]) != tabular_features:
            raise ValueError(
                "Locked XGBoost feature list is internally inconsistent"
            )
        if xgb_config["missing_policy"] != tabular_missing_policy:
            raise ValueError(
                "Locked XGBoost missing-value policy is internally inconsistent"
            )
        xgb_prediction, estimator = predict_residual_fold(
            fold,
            model="xgboost",
            features=tabular_features,
            correction_scale=float(xgb_config["correction_scale"]),
            xgboost_params=_mapping(
                xgb_config["profile"], label="xgboost.selected.profile"
            ),
            random_state=int(xgb_config["random_seed"]),
            candidate_id=xgb_id,
            missing_policy=tabular_missing_policy,
        )
        if estimator.backend_ != xgb_config["backend"]:
            raise RuntimeError(
                "Locked XGBoost backend is unavailable or changed: "
                f"locked={xgb_config['backend']}, "
                f"current={estimator.backend_}"
            )
        predictions_by_model["xgboost"] = xgb_prediction

    if "ordinal" in requested and active["ordinal"]:
        ordinal_block = lock["ordinal_linear"]
        ordinal_id = str(ordinal_block["selected_candidate_id"])
        ordinal_raw = _locked_candidate_config(
            lock,
            ordinal_block,
            ordinal_id,
            label="ordinal_linear",
        )
        allowed = {item.name for item in fields(OrdinalLinearConfig)}
        if set(ordinal_raw) != {*allowed, "features"}:
            raise ValueError(
                "Locked ordinal config does not exactly match the "
                "implemented parameter contract"
            )
        if tuple(ordinal_raw["features"]) != ordinal_features:
            raise ValueError(
                "Locked ordinal feature list is internally inconsistent"
            )
        ordinal_config = OrdinalLinearConfig.from_value(
            {key: ordinal_raw[key] for key in allowed}
        )
        ordinal_prediction, _ = predict_ordinal_fold(
            fold,
            config=ordinal_config,
            features=ordinal_features,
            candidate_id=ordinal_id,
        )
        ordinal_name = str(ordinal_prediction["model"].iloc[0])
        predictions_by_model[ordinal_name] = ordinal_prediction

    tcn_block = lock["shared_tcn"]
    if need_tcn:
        tcn_id = str(tcn_block["selected_candidate_id"])
        tcn_config = _locked_candidate_config(
            lock, tcn_block, tcn_id, label="shared_tcn"
        )
        tcn_seed_frames[tcn_id], tcn_averages[tcn_id] = _fit_locked_tcn(
            fold,
            candidate_id=tcn_id,
            candidate_config=tcn_config,
            seeds=seeds,
            device=device,
        )
        tcn_average = tcn_averages[tcn_id]
        predictions_by_model[str(tcn_average["model"].iloc[0])] = tcn_average
        agreement_tcn = _as_residual_tcn_average(tcn_seed_frames[tcn_id])

    ensemble_block = _mapping(
        tcn_block["ensemble"], label="shared_tcn.ensemble"
    )
    if need_ensemble:
        if ridge_prediction is None:
            raise AssertionError("Active ensemble is missing its Ridge component")
        ensemble_tcn_id = str(ensemble_block["tcn_candidate_id"])
        if ensemble_tcn_id not in tcn_seed_frames:
            catalog = _mapping(
                lock.get("candidate_configs"), label="candidate_configs"
            )
            if ensemble_tcn_id not in catalog:
                raise ValueError(
                    "Locked ensemble TCN config is absent from candidate_configs"
                )
            ensemble_tcn_config = _mapping(
                catalog[ensemble_tcn_id],
                label=f"candidate_configs[{ensemble_tcn_id!r}]",
            )
            (
                tcn_seed_frames[ensemble_tcn_id],
                _,
            ) = _fit_locked_tcn(
                fold,
                candidate_id=ensemble_tcn_id,
                candidate_config=ensemble_tcn_config,
                seeds=seeds,
                device=device,
                force_residual_output=True,
            )
        ensemble_tcn = _as_residual_tcn_average(
            tcn_seed_frames[ensemble_tcn_id]
        )
        agreement_tcn = ensemble_tcn
        ensemble = _locked_ensemble(
            ridge_prediction,
            ensemble_tcn,
            candidate_id=str(ensemble_block["selected_candidate_id"]),
            beta=float(ensemble_block["beta"]),
            correction_policy=str(ensemble_block["correction_policy"]),
        )
        predictions_by_model["ridge_tcn_ensemble"] = ensemble

    if not predictions_by_model:
        inactive_requested = [
            family for family in requested if not active.get(family, False)
        ]
        raise ValueError(
            "No requested model is active in the lock"
            + (
                f"; inactive requested families: {inactive_requested}"
                if inactive_requested
                else ""
            )
        )

    for model_name, frame in predictions_by_model.items():
        _assert_final_year(frame, label=model_name)
        if frame["row_id"].duplicated().any():
            raise AssertionError(
                f"{model_name} produced duplicate final row IDs"
            )

    common_metrics, deployable_metrics = evaluate_common_and_deployable(
        predictions_by_model
    )
    common_metrics.insert(1, "sample_type", "common")
    deployable_metrics.insert(1, "sample_type", "deployable")
    parameter_counts = {
        name: (
            float(
                pd.to_numeric(
                    frame["parameter_count"], errors="coerce"
                ).max()
            )
            if "parameter_count" in frame
            else np.nan
        )
        for name, frame in predictions_by_model.items()
    }
    common_metrics["parameter_count"] = common_metrics["model"].map(
        parameter_counts
    )
    deployable_metrics["parameter_count"] = deployable_metrics["model"].map(
        parameter_counts
    )

    final_predictions = pd.concat(
        list(predictions_by_model.values()), ignore_index=True, sort=False
    )
    _assert_final_year(final_predictions, label="final predictions")
    probability_mask = (
        final_predictions["prob_low"].notna()
        if "prob_low" in final_predictions
        else pd.Series(False, index=final_predictions.index)
    )
    ordinal_rows = _ensure_columns(
        final_predictions.loc[probability_mask].copy(),
        ORDINAL_PREDICTION_COLUMNS,
    )

    if tcn_seed_frames:
        all_tcn_seeds = pd.concat(
            list(tcn_seed_frames.values()), ignore_index=True, sort=False
        )
        _assert_final_year(all_tcn_seeds, label="TCN seed predictions")
        all_tcn_seeds = _ensure_columns(
            all_tcn_seeds, TCN_PREDICTION_COLUMNS
        )
    else:
        all_tcn_seeds = _empty_frame(TCN_PREDICTION_COLUMNS)

    agreement = _empty_frame(AGREEMENT_COLUMNS)
    agreement_diagnostics: dict[str, Any] = {}
    if ridge_prediction is not None and agreement_tcn is not None:
        agreement, agreement_diagnostics = _model_agreement(
            ridge_prediction, agreement_tcn
        )
        agreement = _ensure_columns(agreement, AGREEMENT_COLUMNS)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output / "fold_manifest.parquet", index=False)
    fold.primary_fastball_mapping.assign(fold_year=2025).to_csv(
        output / "primary_fastball_mapping_by_fold.csv", index=False
    )
    final_predictions.to_parquet(
        output / "final_2025_predictions.parquet", index=False
    )
    common_metrics.to_csv(
        output / "final_2025_common_sample_metrics.csv", index=False
    )
    deployable_metrics.to_csv(
        output / "final_2025_deployable_metrics.csv", index=False
    )
    _per_pitcher_metrics(final_predictions).to_csv(
        output / "final_2025_per_pitcher_metrics.csv", index=False
    )
    ordinal_rows.to_parquet(
        output / "ordinal_probabilities_2025.parquet", index=False
    )
    all_tcn_seeds.to_parquet(
        output / "tcn_predictions_2025.parquet", index=False
    )
    agreement.to_parquet(
        output / "model_agreement_2025.parquet", index=False
    )
    _feature_coverage(
        bundle,
        fold,
        _mapping(lock["feature_lists"], label="feature_lists"),
    ).to_csv(output / "feature_coverage.csv", index=False)
    _write_diagnostics(predictions_by_model, output)

    report = {
        "selection_replayed": False,
        "hyperparameters_reselected": False,
        "test_year": 2025,
        "train_years": [2020, 2021, 2022, 2023, 2024],
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "fold_manifest_fingerprint": manifest_fingerprint,
        "code_version": current_code_version,
        "requested_model_families": list(requested),
        "active_locked_models": active,
        "models_evaluated": list(predictions_by_model),
        "locked_random_seeds": list(seeds),
        "agreement": agreement_diagnostics,
        "common_sample_metrics": common_metrics.to_dict(orient="records"),
        "deployable_sample_metrics": deployable_metrics.to_dict(
            orient="records"
        ),
    }
    (output / "final_report.json").write_text(
        json.dumps(
            _json_value(report),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "predictions": final_predictions,
        "common_metrics": common_metrics,
        "deployable_metrics": deployable_metrics,
        "tcn_seed_predictions": all_tcn_seeds,
        "agreement": agreement,
        "report": report,
        "output_dir": output,
    }
