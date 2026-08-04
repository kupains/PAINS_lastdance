"""Shared ordinal metrics, common-sample evaluation, and model-lock validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ORDINAL_CLASSES = (0, 1, 2)
CLASS_NAMES = {0: "low", 1: "middle", 2: "high"}

LOCK_REQUIRED_FIELDS = (
    "schema_version",
    "train_start",
    "validation_years",
    "final_train_end",
    "test_year",
    "qualified_pitchers",
    "qualification_rule",
    "target_definition",
    "pitch_type_mapping",
    "ewma",
    "ridge",
    "xgboost",
    "ordinal_linear",
    "shared_tcn",
    "feature_lists",
    "missing_value_policy",
    "random_seeds",
    "selection_metric",
    "dataset_fingerprint",
    "fold_manifest_fingerprint",
    "code_version",
)

LOCK_MAPPING_FIELDS = (
    "target_definition",
    "pitch_type_mapping",
    "ewma",
    "ridge",
    "xgboost",
    "ordinal_linear",
    "shared_tcn",
    "feature_lists",
    "missing_value_policy",
)


def _as_numeric_series(values: Any, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def ordinal_classification_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    eligible: Any | None = None,
    classes: Sequence[int] = ORDINAL_CLASSES,
) -> dict[str, Any]:
    """Compute fixed-class ordinal metrics and explicit prediction coverage."""

    true = _as_numeric_series(y_true).reset_index(drop=True)
    pred = _as_numeric_series(y_pred).reset_index(drop=True)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    if eligible is None:
        eligibility = pd.Series(True, index=true.index)
    else:
        eligibility = pd.Series(eligible).reset_index(drop=True)
        if len(eligibility) != len(true):
            raise ValueError("eligible must have the same length as y_true")
        eligibility = eligibility.fillna(False).astype(bool)
    allowed = {int(value) for value in classes}
    true_valid = true.notna() & true.isin(allowed)
    total_mask = eligibility & true_valid
    prediction_valid = pred.notna() & pred.isin(allowed)
    evaluated_mask = total_mask & prediction_valid
    n_total = int(total_mask.sum())
    n_predicted = int(evaluated_mask.sum())

    metrics: dict[str, Any] = {
        "n_total_eligible": n_total,
        "n_predicted": n_predicted,
        "coverage": float(n_predicted / n_total) if n_total else np.nan,
        "n_evaluated": n_predicted,
    }
    if n_predicted == 0:
        metrics.update(
            {
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "macro_f1": np.nan,
                "ordinal_mae": np.nan,
                "extreme_error_count": 0,
                "extreme_error_rate": np.nan,
                "low_to_high_error_count": 0,
                "high_to_low_error_count": 0,
                "low_to_high_error_rate": np.nan,
                "high_to_low_error_rate": np.nan,
            }
        )
        for class_value in classes:
            name = CLASS_NAMES.get(int(class_value), str(class_value))
            metrics[f"{name}_precision"] = np.nan
            metrics[f"{name}_recall"] = np.nan
            metrics[f"{name}_f1"] = np.nan
            metrics[f"predicted_{name}_count"] = 0
            metrics[f"predicted_{name}_rate"] = np.nan
            metrics[f"true_{name}_count"] = 0
        return metrics

    actual = true[evaluated_mask].astype(int).to_numpy()
    predicted = pred[evaluated_mask].astype(int).to_numpy()
    metrics["accuracy"] = float(np.mean(actual == predicted))
    metrics["ordinal_mae"] = float(np.mean(np.abs(actual - predicted)))

    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_value in classes:
        value = int(class_value)
        name = CLASS_NAMES.get(value, str(value))
        tp = int(np.sum((actual == value) & (predicted == value)))
        fp = int(np.sum((actual != value) & (predicted == value)))
        fn = int(np.sum((actual == value) & (predicted != value)))
        actual_count = int(np.sum(actual == value))
        predicted_count = int(np.sum(predicted == value))
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        # Macro-F1 uses the fixed Low/Middle/High label set.  A class with no
        # support or no predictions contributes zero instead of disappearing
        # from the macro average; the per-class precision/recall fields remain
        # NaN when their denominator is undefined.
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
            else 0.0
        )
        metrics[f"{name}_precision"] = float(precision) if np.isfinite(precision) else np.nan
        metrics[f"{name}_recall"] = float(recall) if np.isfinite(recall) else np.nan
        metrics[f"{name}_f1"] = float(f1) if np.isfinite(f1) else np.nan
        metrics[f"true_{name}_count"] = actual_count
        metrics[f"predicted_{name}_count"] = predicted_count
        metrics[f"predicted_{name}_rate"] = float(predicted_count / n_predicted)
        if np.isfinite(recall):
            recalls.append(float(recall))
        f1_scores.append(float(f1))

    metrics["balanced_accuracy"] = float(np.mean(recalls)) if recalls else np.nan
    metrics["macro_f1"] = float(np.mean(f1_scores)) if f1_scores else np.nan
    low_value = int(classes[0])
    high_value = int(classes[-1])
    low_to_high = int(np.sum((actual == low_value) & (predicted == high_value)))
    high_to_low = int(np.sum((actual == high_value) & (predicted == low_value)))
    low_count = int(np.sum(actual == low_value))
    high_count = int(np.sum(actual == high_value))
    metrics["low_to_high_error_count"] = low_to_high
    metrics["high_to_low_error_count"] = high_to_low
    metrics["low_to_high_error_rate"] = (
        float(low_to_high / low_count) if low_count else np.nan
    )
    metrics["high_to_low_error_rate"] = (
        float(high_to_low / high_count) if high_count else np.nan
    )
    metrics["extreme_error_count"] = low_to_high + high_to_low
    metrics["extreme_error_rate"] = float((low_to_high + high_to_low) / n_predicted)
    return metrics


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    true_col: str = "true_class",
    prediction_col: str = "predicted_class",
    eligible_col: str | None = "target_eligible",
) -> dict[str, Any]:
    required = [true_col, prediction_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot evaluate prediction frame; missing columns: {missing}")
    eligible = frame[eligible_col] if eligible_col is not None and eligible_col in frame else None
    return ordinal_classification_metrics(
        frame[true_col],
        frame[prediction_col],
        eligible=eligible,
    )


def common_row_intersection(
    predictions_by_model: Mapping[str, pd.DataFrame],
    *,
    row_id_col: str = "row_id",
    prediction_col: str = "predicted_class",
    true_col: str = "true_class",
    eligible_col: str | None = "target_eligible",
    eligible_row_ids: Iterable[str] | None = None,
) -> pd.Index:
    """Return sorted row ids evaluable for every supplied model."""

    if not predictions_by_model:
        return pd.Index([], dtype="string", name=row_id_col)
    common: set[str] | None = (
        {str(value) for value in eligible_row_ids} if eligible_row_ids is not None else None
    )
    for model_name, frame in predictions_by_model.items():
        required = [row_id_col, prediction_col]
        if true_col:
            required.append(true_col)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{model_name} predictions are missing columns: {missing}")
        if frame[row_id_col].duplicated().any():
            raise ValueError(f"{model_name} predictions contain duplicate row ids")
        mask = frame[row_id_col].notna()
        mask &= pd.to_numeric(frame[prediction_col], errors="coerce").isin(ORDINAL_CLASSES)
        if true_col:
            mask &= pd.to_numeric(frame[true_col], errors="coerce").isin(ORDINAL_CLASSES)
        if eligible_col is not None and eligible_col in frame.columns:
            mask &= frame[eligible_col].fillna(False).astype(bool)
        available = {str(value) for value in frame.loc[mask, row_id_col]}
        common = available if common is None else common & available
    return pd.Index(sorted(common or ()), dtype="string", name=row_id_col)


common_sample_row_ids = common_row_intersection


def filter_to_common_rows(
    predictions_by_model: Mapping[str, pd.DataFrame],
    *,
    row_id_col: str = "row_id",
    prediction_col: str = "predicted_class",
    true_col: str = "true_class",
    eligible_col: str | None = "target_eligible",
) -> dict[str, pd.DataFrame]:
    common = common_row_intersection(
        predictions_by_model,
        row_id_col=row_id_col,
        prediction_col=prediction_col,
        true_col=true_col,
        eligible_col=eligible_col,
    )
    common_set = set(common.astype(str))
    result: dict[str, pd.DataFrame] = {}
    for model_name, frame in predictions_by_model.items():
        filtered = frame.loc[frame[row_id_col].astype(str).isin(common_set)].copy()
        filtered = filtered.sort_values(row_id_col, kind="mergesort").reset_index(drop=True)
        result[model_name] = filtered
    return result


def evaluate_common_and_deployable(
    predictions_by_model: Mapping[str, pd.DataFrame],
    *,
    row_id_col: str = "row_id",
    prediction_col: str = "predicted_class",
    true_col: str = "true_class",
    eligible_col: str | None = "target_eligible",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return common-sample and per-model deployable-sample metric tables."""

    common_frames = filter_to_common_rows(
        predictions_by_model,
        row_id_col=row_id_col,
        prediction_col=prediction_col,
        true_col=true_col,
        eligible_col=eligible_col,
    )
    common_rows: list[dict[str, Any]] = []
    deployable_rows: list[dict[str, Any]] = []
    for model_name, frame in predictions_by_model.items():
        deployable_rows.append(
            {
                "model": model_name,
                **evaluate_prediction_frame(
                    frame,
                    true_col=true_col,
                    prediction_col=prediction_col,
                    eligible_col=eligible_col,
                ),
            }
        )
        common_frame = common_frames[model_name]
        common_rows.append(
            {
                "model": model_name,
                **ordinal_classification_metrics(
                    common_frame[true_col],
                    common_frame[prediction_col],
                    eligible=None,
                ),
            }
        )
    return pd.DataFrame(common_rows), pd.DataFrame(deployable_rows)


def confusion_matrix_frame(
    y_true: Any,
    y_pred: Any,
    *,
    classes: Sequence[int] = ORDINAL_CLASSES,
) -> pd.DataFrame:
    true = _as_numeric_series(y_true)
    pred = _as_numeric_series(y_pred)
    valid = true.isin(classes) & pred.isin(classes)
    matrix = pd.crosstab(
        true.loc[valid].astype(int),
        pred.loc[valid].astype(int),
        dropna=False,
    )
    return matrix.reindex(index=classes, columns=classes, fill_value=0).rename_axis(
        index="true_class",
        columns="predicted_class",
    )


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("Locked config cannot contain NaN or infinity")
        return numeric
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def locked_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash a lock config with sorted keys and compact canonical JSON."""

    canonical = _canonical_json_value(config)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


config_fingerprint = locked_config_fingerprint


def validate_locked_model_config(
    config: Mapping[str, Any],
    *,
    require_fingerprints: bool = True,
) -> dict[str, Any]:
    """Validate the complete analysis contract used by final evaluation."""

    if not isinstance(config, Mapping):
        raise TypeError("Locked model config must be a mapping")
    missing = [field for field in LOCK_REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Locked model config is missing fields: {missing}")
    if int(config["schema_version"]) != 1:
        raise ValueError(f"Unsupported locked config schema_version: {config['schema_version']}")
    integer_fields = ("train_start", "final_train_end", "test_year")
    values: dict[str, int] = {}
    for field in integer_fields:
        try:
            values[field] = int(config[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc
    if not values["train_start"] <= values["final_train_end"] < values["test_year"]:
        raise ValueError("Expected train_start <= final_train_end < test_year")

    validation_years = config["validation_years"]
    if not isinstance(validation_years, (list, tuple)) or not validation_years:
        raise ValueError("validation_years must be a non-empty list")
    years = tuple(int(year) for year in validation_years)
    if tuple(sorted(set(years))) != years:
        raise ValueError("validation_years must be unique and increasing")
    if max(years) > values["final_train_end"]:
        raise ValueError("Validation years cannot extend past final_train_end")
    if values["test_year"] in years:
        raise ValueError("test_year cannot be a validation year")

    if not isinstance(config["qualified_pitchers"], (list, tuple)):
        raise ValueError("qualified_pitchers must be a list")
    if not isinstance(config["random_seeds"], (list, tuple)):
        raise ValueError("random_seeds must be a list")
    if not config["random_seeds"]:
        raise ValueError("random_seeds cannot be empty")
    try:
        seeds = tuple(int(seed) for seed in config["random_seeds"])
    except (TypeError, ValueError) as exc:
        raise ValueError("random_seeds must contain integers") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError("random_seeds must be unique")
    shared_tcn = config.get("shared_tcn", {})
    if (
        isinstance(shared_tcn, Mapping)
        and shared_tcn.get("selected_candidate_id")
        and not bool(config.get("smoke_test", False))
        and len(seeds) < 5
    ):
        raise ValueError("Locked TCN analyses require at least 5 random seeds")
    if not isinstance(config["qualification_rule"], str):
        raise ValueError("qualification_rule must be a string")
    if not isinstance(config["selection_metric"], str):
        raise ValueError("selection_metric must be a string")
    if not isinstance(config["code_version"], str):
        raise ValueError("code_version must be a string")
    for field in LOCK_MAPPING_FIELDS:
        if not isinstance(config[field], Mapping):
            raise ValueError(f"{field} must be a mapping")
    if require_fingerprints:
        for field in ("dataset_fingerprint", "fold_manifest_fingerprint"):
            value = config[field]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{field} must be hexadecimal") from exc
    return dict(config)


validate_lock_config = validate_locked_model_config


def validate_locked_fingerprints(
    config: Mapping[str, Any],
    *,
    dataset_fingerprint: str,
    fold_manifest_fingerprint: str,
) -> None:
    """Fail explicitly if final-evaluation inputs differ from the selection lock."""

    validate_locked_model_config(config, require_fingerprints=True)
    expected_dataset = str(config["dataset_fingerprint"])
    expected_manifest = str(config["fold_manifest_fingerprint"])
    if expected_dataset != str(dataset_fingerprint):
        raise ValueError(
            "Dataset fingerprint mismatch: "
            f"locked={expected_dataset}, current={dataset_fingerprint}"
        )
    if expected_manifest != str(fold_manifest_fingerprint):
        raise ValueError(
            "Fold manifest fingerprint mismatch: "
            f"locked={expected_manifest}, current={fold_manifest_fingerprint}"
        )


def load_locked_model_config(
    path: str | Path,
    *,
    require_fingerprints: bool = True,
) -> dict[str, Any]:
    lock_path = Path(path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"Locked model config not found: {lock_path}")
    try:
        config = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Locked model config is not valid JSON: {lock_path}") from exc
    return validate_locked_model_config(config, require_fingerprints=require_fingerprints)


def save_locked_model_config(config: Mapping[str, Any], path: str | Path) -> Path:
    validated = validate_locked_model_config(config, require_fingerprints=True)
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            _canonical_json_value(validated),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return lock_path
