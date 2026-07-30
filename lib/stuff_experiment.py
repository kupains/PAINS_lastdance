"""Fold preparation and model prediction adapters for the unified Stuff+ run.

The public ``PreparedFold`` and ``predict_*`` functions are shared by model
selection and final evaluation.  This module owns fold-local assembly only; it
does not choose candidates or inspect post-fold outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from lib import stuff_mlb_dataset as dataset_contract
from lib.ordinal_linear import OrdinalLinearConfig, PerPitcherOrdinalLinear
from lib.shared_tcn import (
    SharedTCNConfig,
    TCNPreprocessor,
    TCN_RAW_FEATURES as MODEL_TCN_RAW_FEATURES,
    build_pitcher_index,
    build_tcn_sequences,
    fit_shared_tcn,
    model_parameter_count,
    predict_shared_tcn,
)
from lib.stuff_tabular import (
    PerPitcherResidualRegressor,
    classify_stuff_plus,
    classify_stuff_plus_rows,
    regression_prediction_frame,
)
from lib.temporal_splits import (
    FoldSpec,
    add_row_id,
    apply_primary_fastball_mapping,
    assign_true_classes,
    compute_pitcher_tertiles,
    fit_primary_fastball_mapping,
)


# This is the exact compact span-4 contract from the original repository.  The
# explicit copy makes accidental feature drift fail at import rather than
# silently changing the published 497-outing baseline.
LEGACY_COMPACT_FEATURES = (
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
)

_dataset_compact = tuple(
    getattr(dataset_contract, "COMPACT_FEATURES", LEGACY_COMPACT_FEATURES)
)
if _dataset_compact != LEGACY_COMPACT_FEATURES:
    raise RuntimeError(
        "Dataset COMPACT_FEATURES no longer matches the legacy span-4 compact16 contract."
    )
COMPACT_FEATURES = LEGACY_COMPACT_FEATURES

_dataset_tcn = tuple(
    getattr(dataset_contract, "TCN_RAW_FEATURES", MODEL_TCN_RAW_FEATURES)
)
if _dataset_tcn != tuple(MODEL_TCN_RAW_FEATURES):
    raise RuntimeError(
        "Dataset and shared-TCN raw feature contracts do not match."
    )
TCN_RAW_FEATURES = tuple(MODEL_TCN_RAW_FEATURES)
ORDINAL_FEATURES = (*COMPACT_FEATURES, "ewma_tertile_position")


@dataclass
class PreparedFold:
    """All fold-local data and metadata required by prediction adapters."""

    spec: FoldSpec
    history: pd.DataFrame
    train_history: pd.DataFrame
    train_targets: pd.DataFrame
    evaluation_targets: pd.DataFrame
    thresholds: pd.DataFrame
    primary_fastball_mapping: pd.DataFrame


def _coalesce_numeric_aliases(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    label: str,
) -> pd.Series | None:
    present = [column for column in aliases if column in frame.columns]
    if not present:
        return None
    resolved = pd.to_numeric(frame[present[0]], errors="coerce")
    for column in present[1:]:
        candidate = pd.to_numeric(frame[column], errors="coerce")
        common = resolved.notna() & candidate.notna()
        mismatch = common & ~np.isclose(
            resolved,
            candidate,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        if mismatch.any():
            examples = frame.index[mismatch].tolist()[:5]
            raise ValueError(
                f"Conflicting {label} aliases at rows {examples}: {present}"
            )
        resolved = resolved.combine_first(candidate)
    return resolved


def _strict_prior_ewma4(frame: pd.DataFrame) -> pd.Series:
    """Fallback EWMA4 for original ``target_y``-only outing tables."""

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    history_eligible = frame["history_eligible"].fillna(False).astype(bool)
    sort_columns = ["game_date"]
    if "game_pk" in frame:
        sort_columns.append("game_pk")
    for _, pitcher_rows in frame.groupby("pitcher", sort=False, dropna=False):
        history = pitcher_rows.loc[history_eligible.loc[pitcher_rows.index]].sort_values(
            sort_columns, kind="mergesort"
        )
        if history.empty:
            continue
        dates = history["game_date"].to_numpy(dtype="datetime64[ns]")
        values = pd.to_numeric(history["stuff_plus"], errors="coerce").to_numpy(float)
        for position, row_index in enumerate(history.index):
            prior_end = int(np.searchsorted(dates, dates[position], side="left"))
            prior = values[:prior_end]
            if not np.isfinite(prior).any():
                continue
            ewma = pd.Series(prior).ewm(
                span=4,
                adjust=False,
                min_periods=1,
                ignore_na=True,
            ).mean()
            result.at[row_index] = float(ewma.iloc[-1])
    return result


def _canonicalize_experiment_frame(outings: pd.DataFrame) -> pd.DataFrame:
    """Accept both the original and unified dataframe column contracts."""

    required = {"pitcher", "game_pk", "game_date"}
    missing = required - set(outings.columns)
    if missing:
        raise ValueError(
            f"Cannot prepare a Stuff+ fold; missing columns: {sorted(missing)}"
        )
    data = outings.copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    if data["game_date"].isna().any():
        raise ValueError("game_date contains invalid values")
    data["year"] = data["game_date"].dt.year.astype(int)
    data["history_eligible"] = data.get(
        "history_eligible", pd.Series(True, index=data.index)
    ).fillna(False).astype(bool)

    target = _coalesce_numeric_aliases(
        data,
        ("stuff_plus", "target_y", "sp_stuff"),
        label="Stuff+ target",
    )
    if target is None:
        raise ValueError(
            "Cannot prepare a Stuff+ fold without one of "
            "stuff_plus, target_y, or sp_stuff"
        )
    # Make the unified and original names exact aliases within this layer.
    data["stuff_plus"] = target
    data["target_y"] = target
    data["sp_stuff"] = target
    data["stuff_plus_available"] = target.notna()

    if "target_eligible" in data:
        data["target_eligible"] = (
            data["target_eligible"].fillna(False).astype(bool)
        )
    else:
        pitch_ok = (
            pd.to_numeric(data["pitch_count"], errors="coerce").ge(50)
            if "pitch_count" in data
            else pd.Series(True, index=data.index)
        )
        data["target_eligible"] = (
            data["history_eligible"] & pitch_ok & data["stuff_plus_available"]
        )

    ewma = _coalesce_numeric_aliases(
        data,
        ("ewma4", "stuff_plus_ewma4", "stuff_ewma4_prior"),
        label="EWMA4",
    )
    if ewma is None:
        ewma = _strict_prior_ewma4(data)
    data["ewma4"] = ewma
    data["stuff_plus_ewma4"] = ewma

    if "prior_minus_ewma4" not in data and "prior_stuff_plus" in data:
        data["prior_minus_ewma4"] = (
            pd.to_numeric(data["prior_stuff_plus"], errors="coerce") - ewma
        )
    if "mean5_minus_ewma4" not in data and "stuff_plus_mean_last5" in data:
        data["mean5_minus_ewma4"] = (
            pd.to_numeric(data["stuff_plus_mean_last5"], errors="coerce") - ewma
        )

    data = add_row_id(data)
    return data.sort_values(
        ["pitcher", "game_date", "game_pk"], kind="mergesort"
    ).reset_index(drop=True)


def _eligible_targets(frame: pd.DataFrame) -> pd.Series:
    eligible = frame.get(
        "target_eligible", pd.Series(True, index=frame.index)
    )
    return (
        eligible.fillna(False).astype(bool)
        & pd.to_numeric(frame["stuff_plus"], errors="coerce").notna()
        & pd.to_numeric(frame["ewma4"], errors="coerce").notna()
    )


def _actual_feature_complete_case(
    frame: pd.DataFrame,
    features: Iterable[str],
) -> pd.Series:
    names = tuple(features)
    missing = [column for column in names if column not in frame]
    if missing:
        raise ValueError(f"Missing configured model features: {missing}")
    numeric = (
        frame.loc[:, names]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return numeric.notna().all(axis=1)


def _add_tertile_position(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    width = (
        pd.to_numeric(result["q67"], errors="coerce")
        - pd.to_numeric(result["q33"], errors="coerce")
    )
    result["ewma_tertile_position"] = (
        pd.to_numeric(result["ewma4"], errors="coerce")
        - pd.to_numeric(result["q33"], errors="coerce")
    ) / (width + 1e-8)
    return result


def prepare_fold(
    outings: pd.DataFrame,
    fold_year: int,
    *,
    train_start: int = 2020,
    evaluation_split: str | None = None,
) -> PreparedFold:
    """Build all fold-local statistics strictly from pre-``fold_year`` rows."""

    split_name = evaluation_split or (
        "test" if int(fold_year) == 2025 else "validation"
    )
    spec = FoldSpec(
        fold_year=int(fold_year),
        train_start=int(train_start),
        evaluation_split=split_name,
    )
    data = _canonicalize_experiment_frame(outings)
    data = data.loc[
        data["year"].between(spec.train_start, spec.fold_year)
    ].copy()
    if data.empty:
        raise ValueError(f"No rows are available for fold {fold_year}")

    train_history = data.loc[
        data["year"].between(spec.train_start, spec.train_end)
        & data["history_eligible"].fillna(False).astype(bool)
    ].copy()
    mapping = fit_primary_fastball_mapping(
        train_history,
        fold_year=spec.fold_year,
    )
    mapped = apply_primary_fastball_mapping(data, mapping)
    mapped["fold_year"] = spec.fold_year

    raw_train_targets = mapped.loc[
        mapped["year"].between(spec.train_start, spec.train_end)
        & mapped["target_eligible"].fillna(False).astype(bool)
    ].copy()
    thresholds = compute_pitcher_tertiles(
        raw_train_targets,
        target_col="target_y",
    )
    train_targets = assign_true_classes(
        raw_train_targets,
        thresholds,
        target_col="target_y",
    )
    evaluation = mapped.loc[
        mapped["year"].eq(spec.fold_year)
        & mapped["target_eligible"].fillna(False).astype(bool)
    ].copy()
    evaluation = assign_true_classes(
        evaluation,
        thresholds,
        target_col="target_y",
    )

    train_targets = _add_tertile_position(train_targets)
    evaluation = _add_tertile_position(evaluation)
    train_targets = train_targets.loc[_eligible_targets(train_targets)].copy()
    evaluation = evaluation.loc[_eligible_targets(evaluation)].copy()
    if train_targets.empty:
        raise ValueError(f"Fold {fold_year} has no eligible training targets")
    if evaluation.empty:
        raise ValueError(f"Fold {fold_year} has no eligible evaluation targets")

    # Qualification is fixed outside this function.  This only removes a
    # pitcher when the fold-training window cannot define its tertiles.
    trained_pitchers = set(thresholds["pitcher"].astype(str))
    train_targets = train_targets.loc[
        train_targets["pitcher"].astype(str).isin(trained_pitchers)
    ].copy()
    evaluation = evaluation.loc[
        evaluation["pitcher"].astype(str).isin(trained_pitchers)
    ].copy()
    if evaluation.empty:
        raise ValueError(
            f"Fold {fold_year} has no evaluation targets with training thresholds"
        )

    return PreparedFold(
        spec=spec,
        history=mapped,
        train_history=mapped.loc[mapped["year"].le(spec.train_end)].copy(),
        train_targets=train_targets,
        evaluation_targets=evaluation,
        thresholds=thresholds,
        primary_fastball_mapping=mapping,
    )


def _prediction_base(fold: PreparedFold) -> pd.DataFrame:
    keep = [
        "row_id",
        "pitcher",
        "game_pk",
        "game_date",
        "year",
        "target_eligible",
        "history_eligible",
        "stuff_plus",
        "target_y",
        "ewma4",
        "q33",
        "q67",
        "true_class",
    ]
    missing = [column for column in keep if column not in fold.evaluation_targets]
    if missing:
        raise ValueError(f"Prepared evaluation rows are missing columns: {missing}")
    result = fold.evaluation_targets.loc[:, keep].copy()
    result["fold_year"] = fold.spec.fold_year
    result["true_stuff_plus"] = pd.to_numeric(
        result["stuff_plus"], errors="coerce"
    )
    result["true_residual"] = result["true_stuff_plus"] - pd.to_numeric(
        result["ewma4"], errors="coerce"
    )
    return result


def predict_ewma_fold(fold: PreparedFold) -> pd.DataFrame:
    base = _prediction_base(fold)
    base["model"] = "ewma"
    base["candidate_id"] = "ewma_span4"
    base["predicted_residual"] = 0.0
    base["predicted_stuff_plus"] = pd.to_numeric(
        base["ewma4"], errors="coerce"
    )
    base["predicted_class"] = classify_stuff_plus(
        base["predicted_stuff_plus"],
        base["pitcher"],
        fold.thresholds,
    )
    base["feature_eligible"] = base["predicted_stuff_plus"].notna()
    base["deployable"] = np.isfinite(base["predicted_class"])
    base["parameter_count"] = 0
    return base


def predict_residual_fold(
    fold: PreparedFold,
    *,
    model: str,
    features: Iterable[str] = COMPACT_FEATURES,
    ridge_alpha: float = 100.0,
    correction_scale: float = 0.25,
    xgboost_params: Mapping[str, Any] | None = None,
    random_state: int = 20260722,
    candidate_id: str | None = None,
    missing_policy: str = "complete_case",
) -> tuple[pd.DataFrame, PerPitcherResidualRegressor]:
    """Fit a legacy residual model using only its configured feature columns."""

    feature_names = tuple(features)
    estimator = PerPitcherResidualRegressor(
        model=model,
        features=feature_names,
        alpha=ridge_alpha,
        random_state=random_state,
        xgboost_params=dict(xgboost_params or {}),
        missing_policy=missing_policy,
    ).fit(
        fold.train_targets,
        target_col="stuff_plus",
        ewma_col="ewma4",
    )
    residual = estimator.predict(fold.evaluation_targets)
    predicted = regression_prediction_frame(
        fold.evaluation_targets,
        model_name=model,
        predicted_residual=residual,
        correction_scale=correction_scale,
        thresholds=fold.thresholds,
    )
    predicted["fold_year"] = fold.spec.fold_year
    predicted["candidate_id"] = candidate_id or (
        f"{model}_a{ridge_alpha:g}_scale{correction_scale:g}"
    )
    predicted["correction_scale"] = float(correction_scale)
    fitted = predicted["pitcher"].astype(str).isin(estimator.models_)
    if missing_policy == "complete_case":
        feature_eligible = _actual_feature_complete_case(
            fold.evaluation_targets, feature_names
        ).to_numpy()
    else:
        feature_eligible = np.ones(len(predicted), dtype=bool)
    predicted["feature_eligible"] = fitted.to_numpy() & feature_eligible
    predicted["deployable"] = np.isfinite(predicted["predicted_class"])
    predicted["parameter_count"] = estimator.parameter_count
    return predicted, estimator


def predict_ordinal_fold(
    fold: PreparedFold,
    *,
    config: OrdinalLinearConfig | Mapping[str, Any] | None = None,
    features: Iterable[str] = ORDINAL_FEATURES,
    candidate_id: str | None = None,
) -> tuple[pd.DataFrame, PerPitcherOrdinalLinear]:
    cfg = OrdinalLinearConfig.from_value(config)
    feature_names = tuple(features)
    model = PerPitcherOrdinalLinear(feature_names, cfg).fit(
        fold.train_targets,
        target_col="true_class",
    )
    base = _prediction_base(fold)
    fitted_pitchers = set(model.models_)
    predictable = fold.evaluation_targets["pitcher"].isin(fitted_pitchers)
    ordinal_columns = [
        "prob_low",
        "prob_middle",
        "prob_high",
        "predicted_class",
        "latent_score",
        "threshold_1",
        "threshold_2",
    ]
    for column in ordinal_columns:
        base[column] = np.nan
    if predictable.any():
        output = model.predict(fold.evaluation_targets.loc[predictable])
        for column in ordinal_columns:
            base.loc[predictable.to_numpy(), column] = output[column].to_numpy()
    base["model"] = (
        "ordinal_linear_distance"
        if cfg.distance_weight > 0
        else "ordinal_linear"
    )
    base["candidate_id"] = candidate_id or (
        f"ordinal_l2{cfg.l2_lambda:g}_distance{cfg.distance_weight:g}"
    )
    base["feature_eligible"] = predictable.to_numpy()
    base["deployable"] = np.isfinite(base["predicted_class"])
    base["parameter_count"] = int(
        sum(
            item.get("parameter_count", 0)
            for item in model.fit_diagnostics_.values()
        )
    )
    return base, model


def _tcn_targets_known_to_model(
    targets: pd.DataFrame,
    pitcher_mapping: Mapping[Any, int],
) -> pd.DataFrame:
    return targets.loc[
        targets["pitcher"].isin(pitcher_mapping)
        & _eligible_targets(targets)
        & pd.to_numeric(targets["true_class"], errors="coerce").isin([0, 1, 2])
    ].copy()


def predict_tcn_fold(
    fold: PreparedFold,
    *,
    config: SharedTCNConfig | Mapping[str, Any] | None = None,
    sequence_order: str = "ordered",
    shuffle_seed: int = 20260722,
    device: str = "auto",
    candidate_id: str | None = None,
) -> tuple[pd.DataFrame, Any, pd.DataFrame]:
    """Fit one ordered/shuffled shared TCN candidate on a prepared fold."""

    cfg = SharedTCNConfig.from_value(config)
    train_history = fold.train_history.loc[
        fold.train_history["history_eligible"].fillna(False).astype(bool)
    ].copy()
    preprocessor = TCNPreprocessor(TCN_RAW_FEATURES).fit(train_history)

    train_candidates = fold.train_targets.loc[
        _eligible_targets(fold.train_targets)
        & pd.to_numeric(
            fold.train_targets["true_class"], errors="coerce"
        ).isin([0, 1, 2])
    ].copy()
    if train_candidates.empty:
        raise ValueError("No TCN training targets are available")
    _, pitcher_mapping = build_pitcher_index(train_candidates["pitcher"])
    train_targets = _tcn_targets_known_to_model(
        train_candidates, pitcher_mapping
    )
    evaluation_targets = _tcn_targets_known_to_model(
        fold.evaluation_targets, pitcher_mapping
    )
    if evaluation_targets.empty:
        raise ValueError("No TCN evaluation targets have a trained pitcher")

    train_batch = build_tcn_sequences(
        fold.train_history,
        train_targets,
        feature_names=TCN_RAW_FEATURES,
        sequence_length=cfg.sequence_length,
        sequence_order=sequence_order,
        seed=int(shuffle_seed),
        preprocessor=preprocessor,
    )
    evaluation_batch = build_tcn_sequences(
        fold.history,
        evaluation_targets,
        feature_names=TCN_RAW_FEATURES,
        sequence_length=cfg.sequence_length,
        sequence_order=sequence_order,
        seed=int(shuffle_seed),
        preprocessor=preprocessor,
    )
    train_batch.assert_strictly_prior()
    evaluation_batch.assert_strictly_prior()
    train_pitcher_index, _ = build_pitcher_index(
        train_targets["pitcher"], pitcher_mapping
    )
    evaluation_pitcher_index, _ = build_pitcher_index(
        evaluation_targets["pitcher"], pitcher_mapping
    )
    train_true_residual = (
        pd.to_numeric(train_targets["stuff_plus"], errors="coerce")
        - pd.to_numeric(train_targets["ewma4"], errors="coerce")
    ).to_numpy(dtype=float)
    train_position = train_targets["ewma_tertile_position"].to_numpy(dtype=float)
    evaluation_position = evaluation_targets[
        "ewma_tertile_position"
    ].to_numpy(dtype=float)
    model, history = fit_shared_tcn(
        train_batch.values,
        train_batch.valid_timestep,
        train_pitcher_index,
        train_true_residual,
        num_pitchers=len(pitcher_mapping),
        config=cfg,
        true_class=train_targets["true_class"].astype(int).to_numpy(),
        ewma_position=train_position,
        primary_fb_available=train_batch.primary_fb_available,
        stuff_plus_available=train_batch.stuff_plus_available,
        device=device,
    )
    raw = predict_shared_tcn(
        model,
        evaluation_batch.values,
        evaluation_batch.valid_timestep,
        evaluation_pitcher_index,
        ewma_position=(
            evaluation_position if cfg.ordinal_weight > 0 else None
        ),
        primary_fb_available=evaluation_batch.primary_fb_available,
        stuff_plus_available=evaluation_batch.stuff_plus_available,
    )

    base = _prediction_base(fold)
    for column in [
        "global_residual",
        "pitcher_correction",
        "predicted_residual",
        "predicted_stuff_plus",
        "predicted_class",
        "prob_low",
        "prob_middle",
        "prob_high",
        "ordinal_score",
    ]:
        base[column] = np.nan
    aligned = evaluation_targets.set_index("row_id")
    output_by_row = raw.copy()
    output_by_row["row_id"] = evaluation_targets["row_id"].to_numpy()
    output_by_row = output_by_row.set_index("row_id")
    for position in base.index[base["row_id"].isin(output_by_row.index)]:
        row_id = base.at[position, "row_id"]
        item = output_by_row.loc[row_id]
        base.at[position, "global_residual"] = item["global_residual"]
        base.at[position, "pitcher_correction"] = item["pitcher_correction"]
        base.at[position, "predicted_residual"] = item["predicted_residual"]
        base.at[position, "predicted_stuff_plus"] = (
            float(aligned.at[row_id, "ewma4"])
            + cfg.alpha * float(item["predicted_residual"])
        )
        for probability in (
            "prob_low",
            "prob_middle",
            "prob_high",
            "ordinal_score",
        ):
            if probability in item:
                base.at[position, probability] = item[probability]

    base["predicted_class"] = classify_stuff_plus(
        base["predicted_stuff_plus"],
        base["pitcher"],
        fold.thresholds,
    )
    base["model"] = f"tcn_{cfg.head_type.lower()}_{sequence_order}"
    if cfg.ordinal_weight > 0:
        base["model"] = "tcn_h1_residual_ordinal_aux"
    base["candidate_id"] = candidate_id or (
        f"{base['model'].iloc[0]}_l{cfg.sequence_length}_"
        f"c{cfg.internal_channels}_b{cfg.bottleneck_dim}_"
        f"ord{cfg.ordinal_weight:g}_alpha{cfg.alpha:g}"
    )
    base["seed"] = cfg.seed
    base["head_type"] = cfg.head_type.upper()
    base["sequence_order"] = sequence_order
    base["sequence_length"] = cfg.sequence_length
    base["internal_channels"] = cfg.internal_channels
    base["bottleneck_dim"] = cfg.bottleneck_dim
    base["alpha"] = cfg.alpha
    base["feature_eligible"] = base["row_id"].isin(output_by_row.index)
    base["deployable"] = np.isfinite(base["predicted_class"])
    base["parameter_count"] = model_parameter_count(model)
    base["preprocessing_fit_end"] = fold.spec.train_end
    return base, model, history


def ordinal_auxiliary_prediction(prediction: pd.DataFrame) -> pd.DataFrame:
    """Expose the TCN auxiliary argmax as a separate, non-blended output."""

    required = ["prob_low", "prob_middle", "prob_high"]
    missing = [column for column in required if column not in prediction]
    if missing:
        raise ValueError(f"TCN auxiliary probabilities are missing: {missing}")
    result = prediction.copy()
    probabilities = result[required].to_numpy(dtype=float)
    finite = np.isfinite(probabilities).all(axis=1)
    predicted = np.full(len(result), np.nan)
    predicted[finite] = probabilities[finite].argmax(axis=1)
    result["predicted_class"] = predicted
    result["model"] = "tcn_ordinal"
    result["candidate_id"] = (
        result["candidate_id"].astype(str) + ":ordinal_output"
    )
    result["deployable"] = finite
    return result


def average_seed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average one locked TCN candidate across deterministic seed fits."""

    if predictions.empty:
        return predictions.copy()
    identity = [
        "row_id",
        "pitcher",
        "game_pk",
        "game_date",
        "year",
        "fold_year",
        "target_eligible",
        "history_eligible",
        "stuff_plus",
        "target_y",
        "true_stuff_plus",
        "ewma4",
        "true_residual",
        "q33",
        "q67",
        "true_class",
        "model",
        "candidate_id",
        "head_type",
        "sequence_order",
        "sequence_length",
        "internal_channels",
        "bottleneck_dim",
        "alpha",
        "feature_eligible",
    ]
    identity = [column for column in identity if column in predictions]
    numeric = [
        "global_residual",
        "pitcher_correction",
        "predicted_residual",
        "predicted_stuff_plus",
        "prob_low",
        "prob_middle",
        "prob_high",
        "ordinal_score",
        "parameter_count",
    ]
    numeric = [column for column in numeric if column in predictions]
    grouped_object = predictions.groupby(
        identity, dropna=False, sort=False
    )
    grouped = grouped_object[numeric].mean().reset_index()
    grouped["seed"] = "mean"
    grouped["seed_count"] = grouped_object.size().to_numpy(dtype=int)

    for column in ("predicted_residual", "predicted_stuff_plus"):
        if column in numeric:
            spread = (
                grouped_object[column]
                .std(ddof=0)
                .reset_index(name=f"{column}_seed_std")
            )
            grouped = grouped.merge(
                spread,
                on=identity,
                how="left",
                validate="one_to_one",
            )

    residual_models = ~grouped["model"].eq("tcn_ordinal")
    grouped["predicted_class"] = np.nan
    grouped.loc[residual_models, "predicted_class"] = classify_stuff_plus_rows(
        grouped.loc[residual_models, "predicted_stuff_plus"],
        grouped.loc[residual_models, "q33"],
        grouped.loc[residual_models, "q67"],
    )
    if (~residual_models).any():
        probabilities = grouped.loc[
            ~residual_models, ["prob_low", "prob_middle", "prob_high"]
        ].to_numpy(dtype=float)
        finite = np.isfinite(probabilities).all(axis=1)
        ordinal_prediction = np.full(len(probabilities), np.nan)
        ordinal_prediction[finite] = probabilities[finite].argmax(axis=1)
        grouped.loc[~residual_models, "predicted_class"] = ordinal_prediction
    if "feature_eligible" not in grouped:
        grouped["feature_eligible"] = np.isfinite(
            grouped.get("predicted_stuff_plus", np.nan)
        )
    grouped["deployable"] = np.isfinite(grouped["predicted_class"])
    return grouped


def tcn_config_dict(
    config: SharedTCNConfig | Mapping[str, Any],
) -> dict[str, Any]:
    return asdict(SharedTCNConfig.from_value(config))
