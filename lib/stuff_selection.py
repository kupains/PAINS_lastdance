"""Development-only model selection for the unified Stuff+ workflow.

All scored rows come from the 2022, 2023, and 2024 expanding-window folds.
The historical roster may use 2025 start availability, and the shared manifest
contains the reserved 2025 test fold, but no 2025 outcome is passed to a model
candidate or metric calculation in this module.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from build_fold_manifest import build_manifest
from lib.evaluation import (
    common_row_intersection,
    evaluate_prediction_frame,
    ordinal_classification_metrics,
    save_locked_model_config,
)
from lib.ordinal_linear import OrdinalLinearConfig
from lib.shared_tcn import SharedTCNConfig
from lib.stuff_code_version import analysis_code_version
from lib.stuff_cli import (
    RESEARCH_QUALIFICATION_YEARS,
    StuffDataBundle,
)
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
from lib.stuff_mlb_dataset import default_pitch_type_mapping
from lib.stuff_tabular import (
    classify_stuff_plus,
    classify_stuff_plus_rows,
)
from lib.temporal_splits import fit_primary_fastball_mapping
from lib.xgboost_tpe import search_xgboost_tpe


DEFAULT_RANDOM_SEEDS = (
    20260722,
    20260723,
    20260724,
    20260725,
    20260726,
)
VALIDATION_YEARS = (2022, 2023, 2024)
DEFAULT_MODELS = ("ewma", "ridge", "xgboost", "ordinal", "tcn")
SIMPLICITY_MARGIN = 0.003
RIDGE_ALPHAS = (1.0, 10.0, 30.0, 100.0)
RIDGE_CORRECTION_SCALES = (0.25, 0.5, 1.0)
ORDINAL_L2_VALUES = (0.1, 1.0, 10.0, 100.0)
ORDINAL_DISTANCE_WEIGHTS = (0.0, 0.1, 0.25)
TCN_ALPHA_VALUES = (0.1, 0.25, 0.5)
ENSEMBLE_BETA_VALUES = (0.0, 0.1, 0.2, 0.3)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _candidate_id(prefix: str, **values: Any) -> str:
    fields = "_".join(f"{key}{value}" for key, value in values.items())
    return f"{prefix}_{fields}" if fields else prefix


def _normalize_seed(value: Any) -> Any:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return int(value)
    return value


def _numeric_seed_mask(frame: pd.DataFrame) -> pd.Series:
    if "seed" not in frame:
        return pd.Series(False, index=frame.index)
    numeric = pd.to_numeric(frame["seed"], errors="coerce")
    return numeric.notna()


def _tcn_templates(smoke_test: bool) -> list[dict[str, Any]]:
    """Return the bounded required ablations and small architecture variants."""

    templates = [
        {
            "name": "tcn_h0_global_only",
            "head_type": "H0",
            "internal_channels": 8,
            "bottleneck_dim": 4,
            "sequence_order": "ordered",
            "ordinal_weight": 0.0,
        },
        {
            "name": "tcn_h1_pitcher_bias",
            "head_type": "H1",
            "internal_channels": 8,
            "bottleneck_dim": 4,
            "sequence_order": "ordered",
            "ordinal_weight": 0.0,
        },
        {
            "name": "tcn_h2_pitcher_vector",
            "head_type": "H2",
            "internal_channels": 8,
            "bottleneck_dim": 4,
            "sequence_order": "ordered",
            "ordinal_weight": 0.0,
        },
        {
            "name": "tcn_h1_shuffled",
            "head_type": "H1",
            "internal_channels": 8,
            "bottleneck_dim": 4,
            "sequence_order": "shuffled",
            "ordinal_weight": 0.0,
        },
        {
            "name": "tcn_h1_ordinal_aux",
            "head_type": "H1",
            "internal_channels": 8,
            "bottleneck_dim": 4,
            "sequence_order": "ordered",
            "ordinal_weight": 0.25,
        },
    ]
    if not smoke_test:
        templates.extend(
            [
                {
                    "name": "tcn_4",
                    "head_type": "H1",
                    "internal_channels": 4,
                    "bottleneck_dim": 4,
                    "sequence_order": "ordered",
                    "ordinal_weight": 0.0,
                },
                {
                    "name": "tcn_8",
                    "head_type": "H1",
                    "internal_channels": 8,
                    "bottleneck_dim": 8,
                    "sequence_order": "ordered",
                    "ordinal_weight": 0.0,
                },
                {
                    "name": "tcn_h1_ordinal_aux_w05",
                    "head_type": "H1",
                    "internal_channels": 8,
                    "bottleneck_dim": 4,
                    "sequence_order": "ordered",
                    "ordinal_weight": 0.5,
                },
            ]
        )
    return templates


def _tcn_search_plan(
    templates: Sequence[Mapping[str, Any]],
    sequence_lengths: Sequence[int],
    *,
    smoke_test: bool,
) -> list[tuple[int, dict[str, Any]]]:
    """Return the predeclared bounded architecture/length combinations.

    Normal runs compare every required architecture at the longest requested
    sequence (8 in the public profile), whose length matches the receptive
    field of the [1, 2, 4] dilation stack.  The ordered H1 pitcher-bias
    reference is additionally evaluated at every shorter requested length.
    Thus L={5, 8} is still selected on OOF data without multiplying unrelated
    architecture ablations across both lengths or adapting the plan after
    observing validation scores.

    A smoke run keeps the caller's tiny Cartesian plan so every required model
    path is exercised end to end.
    """

    lengths = tuple(dict.fromkeys(int(value) for value in sequence_lengths))
    if not lengths:
        raise ValueError("At least one TCN sequence length is required")
    copied = [dict(template) for template in templates]
    if smoke_test or len(lengths) == 1:
        return [
            (length, template)
            for length in lengths
            for template in copied
        ]

    reference_length = max(lengths)
    plan = [
        (reference_length, template)
        for template in copied
    ]
    baseline = next(
        (
            template
            for template in copied
            if template["name"] == "tcn_h1_pitcher_bias"
        ),
        None,
    )
    if baseline is None:
        raise ValueError(
            "TCN templates must include tcn_h1_pitcher_bias"
        )
    plan.extend(
        (length, dict(baseline))
        for length in lengths
        if length != reference_length
    )
    return plan


def _metrics_for_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute common/deployable metrics and seed-aware candidate summaries."""

    required = {
        "row_id",
        "fold_year",
        "model",
        "candidate_id",
        "true_class",
        "predicted_class",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {missing}")
    if predictions.empty:
        raise ValueError("Cannot score an empty OOF prediction table")
    scored_years = tuple(
        sorted(
            int(value)
            for value in pd.to_numeric(
                predictions["fold_year"], errors="raise"
            ).unique()
        )
    )
    if scored_years != VALIDATION_YEARS:
        raise AssertionError(
            f"Selection metrics require exactly {VALIDATION_YEARS}; got {scored_years}"
        )

    records: list[dict[str, Any]] = []
    group_columns = ["model", "candidate_id", "seed"]
    for fold_year in VALIDATION_YEARS:
        fold_predictions = predictions.loc[
            pd.to_numeric(
                predictions["fold_year"], errors="coerce"
            ).eq(fold_year)
        ]
        prediction_groups: list[
            tuple[tuple[Any, Any, Any], pd.DataFrame]
        ] = []
        common_inputs: dict[str, pd.DataFrame] = {}
        for group_number, (keys, frame) in enumerate(
            fold_predictions.groupby(
                group_columns,
                dropna=False,
                sort=False,
            )
        ):
            prediction_groups.append((keys, frame.copy()))
            common_inputs[str(group_number)] = frame.copy()
        if not prediction_groups:
            raise AssertionError(f"No prediction groups for fold {fold_year}")
        common_ids = common_row_intersection(common_inputs)
        if common_ids.empty:
            raise ValueError(
                f"No common evaluable row exists across candidates in "
                f"fold {fold_year}"
            )
        common_set = set(common_ids.astype(str))

        for (model, candidate_id, seed), frame in prediction_groups:
            metadata = {
                "fold_year": int(fold_year),
                "model": str(model),
                "candidate_id": str(candidate_id),
                "seed": _normalize_seed(seed),
                "parameter_count": (
                    float(
                        pd.to_numeric(
                            frame["parameter_count"], errors="coerce"
                        ).max()
                    )
                    if "parameter_count" in frame
                    and pd.to_numeric(
                        frame["parameter_count"], errors="coerce"
                    ).notna().any()
                    else np.nan
                ),
            }
            records.append(
                {
                    **metadata,
                    "sample_type": "deployable",
                    **evaluate_prediction_frame(frame),
                }
            )
            common = frame.loc[
                frame["row_id"].astype(str).isin(common_set)
            ]
            records.append(
                {
                    **metadata,
                    "sample_type": "common",
                    **ordinal_classification_metrics(
                        common["true_class"],
                        common["predicted_class"],
                        eligible=None,
                    ),
                }
            )

    by_year = pd.DataFrame(records)
    common_source = by_year.loc[
        by_year["sample_type"].eq("common")
        & pd.to_numeric(
            by_year["n_evaluated"], errors="coerce"
        ).gt(0)
    ].copy()
    if common_source.empty:
        raise ValueError(
            "No common evaluable row exists across all model candidates; "
            "selection cannot fall back to incomparable deployable samples"
        )
    selection_sample = "common"

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ordinal_mae",
        "low_recall",
        "middle_recall",
        "high_recall",
        "extreme_error_count",
        "coverage",
        "n_total_eligible",
        "n_predicted",
        "parameter_count",
    ]
    summary_rows: list[dict[str, Any]] = []
    for (model, candidate_id), group in common_source.groupby(
        ["model", "candidate_id"],
        sort=False,
    ):
        is_tcn = str(model).startswith("tcn_")
        seeded = group.loc[
            _numeric_seed_mask(group)
        ].copy() if is_tcn else pd.DataFrame()
        source = seeded if not seeded.empty else group
        if not seeded.empty:
            seed_sets = [
                tuple(
                    sorted(
                        pd.to_numeric(
                            fold_group["seed"], errors="raise"
                        ).astype(int).unique()
                    )
                )
                for _, fold_group in seeded.groupby(
                    "fold_year", sort=True
                )
            ]
            if (
                len(seed_sets) != len(VALIDATION_YEARS)
                or any(values != seed_sets[0] for values in seed_sets[1:])
            ):
                raise AssertionError(
                    f"TCN candidate {candidate_id!r} does not have the same "
                    "seed set in every validation fold"
                )

        # The normal TCN run contributes every deterministic seed/year
        # observation.  Its mean therefore cannot be driven by one lucky seed,
        # while the score's standard-deviation penalty explicitly incorporates
        # both temporal and seed instability.  Non-seeded candidates retain
        # the original three yearly observations.
        year_values = (
            source.groupby("fold_year", sort=True)[
                [name for name in metric_names if name in source]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        year_balanced = pd.to_numeric(
            year_values["balanced_accuracy"], errors="coerce"
        )
        balanced_observations = pd.to_numeric(
            source["balanced_accuracy"], errors="coerce"
        ).dropna()
        mean_balanced = float(balanced_observations.mean())
        std_balanced = float(balanced_observations.std(ddof=0))
        row: dict[str, Any] = {
            "model": str(model),
            "candidate_id": str(candidate_id),
            "selection_sample_type": selection_sample,
            "mean_balanced_accuracy": mean_balanced,
            "std_balanced_accuracy": std_balanced,
            "year_standard_deviation": float(
                year_balanced.std(ddof=0)
            ),
            "selection_score": mean_balanced - 0.5 * std_balanced,
            "n_validation_years": int(year_values["fold_year"].nunique()),
            "n_seed_year_observations": int(
                len(balanced_observations)
            ),
        }
        for metric in metric_names:
            if metric in source:
                row[f"mean_{metric}"] = float(
                    pd.to_numeric(
                        source[metric], errors="coerce"
                    ).mean()
                )
        if not seeded.empty:
            per_seed = (
                seeded.groupby("seed", sort=True)["balanced_accuracy"]
                .mean()
            )
            row["seed_standard_deviation"] = float(
                per_seed.std(ddof=0)
            )
            row["n_seeds"] = int(per_seed.size)
        else:
            row["seed_standard_deviation"] = 0.0
            row["n_seeds"] = 0
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise ValueError("No model candidate produced selection metrics")
    summary = summary.sort_values(
        [
            "selection_score",
            "mean_balanced_accuracy",
            "mean_ordinal_mae",
        ],
        ascending=[False, False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return by_year, summary


def _choose_with_simplicity_margin(
    summary: pd.DataFrame,
    *,
    model_names: Iterable[str],
) -> dict[str, Any] | None:
    candidates = summary.loc[
        summary["model"].isin(set(model_names))
    ].copy()
    candidates = candidates.loc[
        candidates["selection_score"].notna()
        & candidates["mean_balanced_accuracy"].notna()
    ]
    if candidates.empty:
        return None
    best_score = float(candidates["selection_score"].max())
    close = candidates.loc[
        candidates["selection_score"].ge(
            best_score - SIMPLICITY_MARGIN
        )
    ].copy()
    close["_params"] = pd.to_numeric(
        close.get("mean_parameter_count"), errors="coerce"
    ).fillna(np.inf)
    close["_ordinal"] = pd.to_numeric(
        close.get("mean_ordinal_mae"), errors="coerce"
    ).fillna(np.inf)
    chosen = close.sort_values(
        ["_params", "selection_score", "_ordinal"],
        ascending=[True, False, True],
        kind="mergesort",
    ).iloc[0]
    return {
        key: _json_value(value)
        for key, value in chosen.drop(
            labels=["_params", "_ordinal"]
        ).to_dict().items()
    }


def _rescale_tcn_prediction(
    prediction: pd.DataFrame,
    *,
    alpha: float,
    thresholds: pd.DataFrame,
    candidate_id: str,
) -> pd.DataFrame:
    result = prediction.copy()
    result["alpha"] = float(alpha)
    result["predicted_stuff_plus"] = (
        pd.to_numeric(result["ewma4"], errors="coerce")
        + float(alpha)
        * pd.to_numeric(
            result["predicted_residual"], errors="coerce"
        )
    )
    result["predicted_class"] = classify_stuff_plus(
        result["predicted_stuff_plus"],
        result["pitcher"],
        thresholds,
    )
    result["candidate_id"] = candidate_id
    result["deployable"] = np.isfinite(result["predicted_class"])
    return result


def _ensemble_candidates(
    predictions: pd.DataFrame,
    *,
    ridge_candidate_id: str | None,
    tcn_candidate_id: str | None,
    beta_values: Sequence[float] = ENSEMBLE_BETA_VALUES,
) -> list[pd.DataFrame]:
    """Create OOF ensembles only after standalone Ridge and TCN selection."""

    if not ridge_candidate_id or not tcn_candidate_id:
        return []
    ridge = predictions.loc[
        predictions["candidate_id"].eq(ridge_candidate_id)
        & predictions["model"].eq("ridge")
    ].copy()
    tcn_seed_rows = predictions.loc[
        predictions["candidate_id"].eq(tcn_candidate_id)
        & predictions["model"].astype(str).str.startswith("tcn_")
        & predictions["model"].ne("tcn_ordinal")
    ].copy()
    if ridge.empty or tcn_seed_rows.empty:
        return []
    tcn = average_seed_predictions(tcn_seed_rows)
    tcn_columns = [
        "row_id",
        "fold_year",
        "predicted_residual",
        "parameter_count",
        "alpha",
        "deployable",
    ]
    tcn_columns = [
        column for column in tcn_columns if column in tcn
    ]
    merged = ridge.merge(
        tcn[tcn_columns].rename(
            columns={
                "predicted_residual": "tcn_predicted_residual",
                "parameter_count": "tcn_parameter_count",
                "alpha": "tcn_alpha",
                "deployable": "tcn_deployable",
            }
        ),
        on=["row_id", "fold_year"],
        how="inner",
        validate="one_to_one",
    )
    output: list[pd.DataFrame] = []
    for beta in beta_values:
        frame = merged.copy()
        frame["ridge_predicted_residual"] = pd.to_numeric(
            frame["predicted_residual"], errors="coerce"
        )
        ridge_scale = pd.to_numeric(
            frame.get(
                "correction_scale",
                pd.Series(0.5, index=frame.index),
            ),
            errors="coerce",
        ).fillna(0.5)
        tcn_alpha = pd.to_numeric(
            frame.get(
                "tcn_alpha",
                pd.Series(0.25, index=frame.index),
            ),
            errors="coerce",
        ).fillna(0.25)
        frame["ridge_residual_correction"] = (
            ridge_scale * frame["ridge_predicted_residual"]
        )
        frame["tcn_residual_correction"] = (
            tcn_alpha
            * pd.to_numeric(
                frame["tcn_predicted_residual"], errors="coerce"
            )
        )
        frame["predicted_residual"] = (
            (1.0 - float(beta))
            * frame["ridge_residual_correction"]
            + float(beta) * frame["tcn_residual_correction"]
        )
        frame["predicted_stuff_plus"] = (
            pd.to_numeric(frame["ewma4"], errors="coerce")
            + frame["predicted_residual"]
        )
        frame["predicted_class"] = classify_stuff_plus_rows(
            frame["predicted_stuff_plus"],
            frame["q33"],
            frame["q67"],
        )
        frame["model"] = "ridge_tcn_ensemble"
        frame["candidate_id"] = f"ensemble_beta{float(beta):g}"
        frame["beta"] = float(beta)
        # The ensemble is one deterministic average across locked TCN seeds,
        # not another seed fit.  Keep this missing so the OOF parquet column
        # remains numeric and per-seed reporting includes only actual fits.
        frame["seed"] = np.nan
        frame["correction_scale"] = 1.0
        frame["feature_eligible"] = (
            np.isfinite(frame["ridge_predicted_residual"])
            & np.isfinite(frame["tcn_predicted_residual"])
        )
        frame["deployable"] = np.isfinite(
            frame["predicted_class"]
        )
        ridge_parameters = pd.to_numeric(
            frame["parameter_count"], errors="coerce"
        ).fillna(0)
        tcn_parameters = pd.to_numeric(
            frame.get(
                "tcn_parameter_count",
                pd.Series(0, index=frame.index),
            ),
            errors="coerce",
        ).fillna(0)
        frame["parameter_count"] = np.where(
            np.isclose(float(beta), 0.0),
            ridge_parameters,
            ridge_parameters + tcn_parameters,
        )
        output.append(frame)
    return output


def _feature_coverage(
    fold_histories: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    representations = {
        "tabular_compact_16": COMPACT_FEATURES,
        "tcn_raw_15": TCN_RAW_FEATURES,
    }
    rows: list[dict[str, Any]] = []
    for history in fold_histories:
        fold_year = int(history["fold_year"].iloc[0])
        for representation, features in representations.items():
            for feature in features:
                if feature not in history:
                    n_available = 0
                else:
                    n_available = int(
                        pd.to_numeric(
                            history[feature], errors="coerce"
                        ).notna().sum()
                    )
                rows.append(
                    {
                        "fold_year": fold_year,
                        "representation": representation,
                        "feature": feature,
                        "n_rows": int(len(history)),
                        "n_available": n_available,
                        "coverage": (
                            float(n_available / len(history))
                            if len(history)
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _seed_metric_table(by_year: pd.DataFrame) -> pd.DataFrame:
    seeded = by_year.loc[
        by_year["model"].astype(str).str.startswith("tcn_")
        & _numeric_seed_mask(by_year)
    ].copy()
    if seeded.empty:
        return pd.DataFrame(columns=by_year.columns)
    group_columns = [
        "fold_year",
        "model",
        "candidate_id",
        "sample_type",
    ]
    numeric_columns = [
        column
        for column in seeded.columns
        if column not in [*group_columns, "seed"]
        and pd.api.types.is_numeric_dtype(seeded[column])
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for keys, group in seeded.groupby(
        group_columns, sort=False
    ):
        metadata = dict(zip(group_columns, keys, strict=True))
        for statistic in ("mean", "std"):
            row: dict[str, Any] = {
                **metadata,
                "seed": statistic,
            }
            for column in numeric_columns:
                values = pd.to_numeric(
                    group[column], errors="coerce"
                )
                row[column] = float(
                    values.mean()
                    if statistic == "mean"
                    else values.std(ddof=0)
                )
            aggregate_rows.append(row)
    return pd.concat(
        [seeded, pd.DataFrame(aggregate_rows)],
        ignore_index=True,
        sort=False,
    )


def _model_lock_entry(
    *,
    requested: bool,
    chosen: dict[str, Any] | None,
    candidate_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_id = (
        str(chosen["candidate_id"]) if chosen else None
    )
    return {
        "status": (
            "selected"
            if chosen
            else (
                "no_deployable_candidate"
                if requested
                else "not_requested"
            )
        ),
        "selected_candidate_id": candidate_id,
        "selected": dict(
            candidate_configs.get(candidate_id, {})
        ) if candidate_id else {},
    }


def _validate_selection_inputs(
    *,
    models: Sequence[str],
    random_seeds: Sequence[int],
    sequence_lengths: Sequence[int],
    tcn_epochs: int,
    xgboost_trials: int,
    smoke_test: bool,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    selected_models = tuple(
        dict.fromkeys(str(name).lower() for name in models)
    )
    if not selected_models:
        raise ValueError("At least one model family is required")
    unknown = sorted(set(selected_models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown model families: {unknown}")
    seeds = tuple(int(value) for value in random_seeds)
    if not seeds:
        raise ValueError("At least one random seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Random seeds must be unique")
    if (
        "tcn" in selected_models
        and not smoke_test
        and len(seeds) < 5
    ):
        raise ValueError(
            "TCN selection requires at least five (5) seeds outside smoke mode"
        )
    lengths = tuple(
        dict.fromkeys(int(value) for value in sequence_lengths)
    )
    if not lengths or not set(lengths).issubset({5, 8}):
        raise ValueError("TCN sequence lengths must be selected from 5 and 8")
    if int(tcn_epochs) < 1:
        raise ValueError("tcn_epochs must be positive")
    if int(xgboost_trials) < 1:
        raise ValueError("xgboost_trials must be positive")
    return selected_models, seeds, lengths


def run_model_selection(
    bundle: StuffDataBundle,
    *,
    output_dir: str | Path,
    models: Sequence[str] = DEFAULT_MODELS,
    device: str = "auto",
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    sequence_lengths: Sequence[int] = (5, 8),
    tcn_epochs: int = 250,
    xgboost_trials: int = 32,
    smoke_test: bool = False,
    cutter_category: str = "fastball",
) -> dict[str, Any]:
    """Evaluate candidates on 2022-2024 and persist the complete model lock."""

    selected_models, seeds, lengths = _validate_selection_inputs(
        models=models,
        random_seeds=random_seeds,
        sequence_lengths=sequence_lengths,
        tcn_epochs=tcn_epochs,
        xgboost_trials=xgboost_trials,
        smoke_test=smoke_test,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest, manifest_fingerprint = build_manifest(bundle)
    manifest.to_parquet(
        output / "fold_manifest.parquet", index=False
    )
    all_years = pd.to_numeric(
        bundle.outings["year"], errors="raise"
    )
    development = bundle.outings.loc[all_years.le(2024)].copy()
    if development.empty:
        raise ValueError("Selection development data is empty")
    assert int(
        pd.to_numeric(development["year"], errors="raise").max()
    ) <= 2024
    if pd.to_numeric(
        development["year"], errors="raise"
    ).eq(2025).any():
        raise AssertionError("2025 rows entered the selection dataframe")

    xgboost_search: dict[str, Any] | None = None
    xgboost_trials_frame = pd.DataFrame(
        columns=[
            "stage",
            "trial",
            "candidate_id",
            "selection_score",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
        ]
    )
    if "xgboost" in selected_models:
        xgboost_search, xgboost_trials_frame = search_xgboost_tpe(
            development,
            n_trials=int(xgboost_trials),
            random_seed=seeds[0],
        )
    xgboost_trials_frame.to_csv(
        output / "xgboost_tpe_trials.csv", index=False
    )

    predictions: list[pd.DataFrame] = []
    mappings: list[pd.DataFrame] = []
    fold_histories: list[pd.DataFrame] = []
    candidate_configs: dict[str, dict[str, Any]] = {}
    validation_years_touched: set[int] = set()

    ridge_alphas = (
        (100.0,) if smoke_test else RIDGE_ALPHAS
    )
    ridge_scales = (
        (0.5,) if smoke_test else RIDGE_CORRECTION_SCALES
    )
    ordinal_l2 = (
        (1.0,) if smoke_test else ORDINAL_L2_VALUES
    )
    ordinal_distances = (
        (0.0, 0.1)
        if smoke_test
        else ORDINAL_DISTANCE_WEIGHTS
    )
    tcn_alphas = (
        (0.25,) if smoke_test else TCN_ALPHA_VALUES
    )
    tcn_templates = _tcn_templates(smoke_test)
    tcn_search_plan = _tcn_search_plan(
        tcn_templates,
        lengths,
        smoke_test=smoke_test,
    )
    tcn_search_keys = {
        (length, str(template["name"]))
        for length, template in tcn_search_plan
    }

    for fold_year in VALIDATION_YEARS:
        fold = prepare_fold(
            development,
            fold_year,
            evaluation_split="validation",
        )
        validation_years_touched.add(fold_year)
        mapping = fold.primary_fastball_mapping.copy()
        mapping["fold_year"] = fold_year
        mappings.append(mapping)
        fold_history = fold.history.copy()
        fold_history["fold_year"] = fold_year
        fold_histories.append(fold_history)

        if "ewma" in selected_models:
            frame = predict_ewma_fold(fold)
            frame["seed"] = np.nan
            predictions.append(frame)
            candidate_configs["ewma_span4"] = {
                "span": 4,
                "features": ["ewma4"],
            }

        if "ridge" in selected_models:
            for ridge_alpha in ridge_alphas:
                for scale in ridge_scales:
                    candidate = _candidate_id(
                        "ridge",
                        alpha=f"{ridge_alpha:g}",
                        scale=f"{scale:g}",
                    )
                    frame, _ = predict_residual_fold(
                        fold,
                        model="ridge",
                        ridge_alpha=ridge_alpha,
                        correction_scale=scale,
                        candidate_id=candidate,
                        missing_policy="complete_case",
                    )
                    frame["seed"] = np.nan
                    predictions.append(frame)
                    candidate_configs[candidate] = {
                        "alpha": ridge_alpha,
                        "correction_scale": scale,
                        "features": list(COMPACT_FEATURES),
                        "missing_policy": "complete_case",
                    }

        if "xgboost" in selected_models:
            if xgboost_search is None:
                raise AssertionError(
                    "XGBoost TPE search result is missing"
                )
            profile = dict(xgboost_search["profile"])
            scale = float(
                xgboost_search["correction_scale"]
            )
            candidate = _candidate_id(
                "xgboost",
                tpe=f"{int(xgboost_search['tpe_best_trial'])}",
                scale=f"{scale:g}",
            )
            frame, estimator = predict_residual_fold(
                fold,
                model="xgboost",
                correction_scale=scale,
                xgboost_params=profile,
                candidate_id=candidate,
                missing_policy="complete_case",
            )
            frame["seed"] = np.nan
            predictions.append(frame)
            candidate_configs[candidate] = {
                **xgboost_search,
                "backend": estimator.backend_,
                "features": list(COMPACT_FEATURES),
                "missing_policy": "complete_case",
            }

        if "ordinal" in selected_models:
            for l2_lambda in ordinal_l2:
                for distance_weight in ordinal_distances:
                    candidate = _candidate_id(
                        "ordinal",
                        l2=f"{l2_lambda:g}",
                        distance=f"{distance_weight:g}",
                    )
                    cfg = OrdinalLinearConfig(
                        l2_lambda=l2_lambda,
                        distance_weight=distance_weight,
                        seed=seeds[0],
                        max_iter=60 if smoke_test else 200,
                    )
                    frame, _ = predict_ordinal_fold(
                        fold,
                        config=cfg,
                        candidate_id=candidate,
                    )
                    frame["seed"] = np.nan
                    predictions.append(frame)
                    candidate_configs[candidate] = {
                        **asdict(cfg),
                        "features": list(ORDINAL_FEATURES),
                    }

        if "tcn" in selected_models:
            for sequence_length in lengths:
                for template in tcn_templates:
                    if (
                        sequence_length,
                        str(template["name"]),
                    ) not in tcn_search_keys:
                        continue
                    for seed in seeds:
                        base_id = _candidate_id(
                            template["name"],
                            length=sequence_length,
                            channels=template[
                                "internal_channels"
                            ],
                            bottleneck=template[
                                "bottleneck_dim"
                            ],
                            ord=f"{template['ordinal_weight']:g}",
                        )
                        cfg = SharedTCNConfig(
                            sequence_length=sequence_length,
                            internal_channels=int(
                                template["internal_channels"]
                            ),
                            bottleneck_dim=int(
                                template["bottleneck_dim"]
                            ),
                            head_type=str(template["head_type"]),
                            ordinal_weight=float(
                                template["ordinal_weight"]
                            ),
                            alpha=0.25,
                            epochs=int(tcn_epochs),
                            seed=int(seed),
                        )
                        frame, _, _ = predict_tcn_fold(
                            fold,
                            config=cfg,
                            sequence_order=str(
                                template["sequence_order"]
                            ),
                            shuffle_seed=DEFAULT_RANDOM_SEEDS[0],
                            device=device,
                            candidate_id=base_id,
                        )
                        locked_tcn_config = asdict(cfg)
                        locked_tcn_config.pop("seed")
                        for alpha in tcn_alphas:
                            candidate = (
                                f"{base_id}_alpha{alpha:g}"
                            )
                            scaled = _rescale_tcn_prediction(
                                frame,
                                alpha=alpha,
                                thresholds=fold.thresholds,
                                candidate_id=candidate,
                            )
                            predictions.append(scaled)
                            candidate_configs[candidate] = {
                                **locked_tcn_config,
                                "alpha": alpha,
                                "seeds": list(seeds),
                                "sequence_order": template[
                                    "sequence_order"
                                ],
                                "shuffle_seed": (
                                    DEFAULT_RANDOM_SEEDS[0]
                                ),
                                "template_name": template["name"],
                                "features": list(TCN_RAW_FEATURES),
                                "output": "residual",
                            }
                        if (
                            float(template["ordinal_weight"]) > 0
                        ):
                            ordinal_frame = (
                                ordinal_auxiliary_prediction(frame)
                            )
                            ordinal_candidate = (
                                f"{base_id}_ordinal_output"
                            )
                            ordinal_frame[
                                "candidate_id"
                            ] = ordinal_candidate
                            predictions.append(ordinal_frame)
                            candidate_configs[
                                ordinal_candidate
                            ] = {
                                **locked_tcn_config,
                                "seeds": list(seeds),
                                "output": (
                                    "ordinal_probability_argmax"
                                ),
                                "sequence_order": template[
                                    "sequence_order"
                                ],
                                "shuffle_seed": (
                                    DEFAULT_RANDOM_SEEDS[0]
                                ),
                                "features": list(TCN_RAW_FEATURES),
                            }

    if validation_years_touched != set(VALIDATION_YEARS):
        raise AssertionError(
            "Selection did not evaluate exactly 2022-2024"
        )
    if not predictions:
        raise ValueError("No selected model family produced predictions")
    oof = pd.concat(
        predictions, ignore_index=True, sort=False
    )
    if pd.to_numeric(
        oof["fold_year"], errors="raise"
    ).max() > 2024:
        raise AssertionError(
            "2025 predictions entered model selection"
        )
    if pd.to_numeric(
        oof["year"], errors="raise"
    ).max() > 2024:
        raise AssertionError(
            "A post-2024 target row entered OOF predictions"
        )

    by_year, summary = _metrics_for_predictions(oof)
    chosen_ridge = _choose_with_simplicity_margin(
        summary, model_names=["ridge"]
    )
    chosen_xgb = _choose_with_simplicity_margin(
        summary, model_names=["xgboost"]
    )
    chosen_ordinal = _choose_with_simplicity_margin(
        summary,
        model_names=[
            "ordinal_linear",
            "ordinal_linear_distance",
        ],
    )
    chosen_tcn = _choose_with_simplicity_margin(
        summary,
        model_names=[
            "tcn_h0_ordered",
            "tcn_h1_ordered",
            "tcn_h2_ordered",
            "tcn_h1_shuffled",
            "tcn_h1_residual_ordinal_aux",
            "tcn_ordinal",
        ],
    )
    chosen_tcn_residual = _choose_with_simplicity_margin(
        summary,
        model_names=[
            "tcn_h0_ordered",
            "tcn_h1_ordered",
            "tcn_h2_ordered",
            "tcn_h1_shuffled",
            "tcn_h1_residual_ordinal_aux",
        ],
    )

    # Ensemble candidates are created only after standalone TCN metrics and a
    # residual TCN candidate have been selected.
    ensemble_frames = _ensemble_candidates(
        oof,
        ridge_candidate_id=(
            str(chosen_ridge["candidate_id"])
            if chosen_ridge
            else None
        ),
        tcn_candidate_id=(
            str(chosen_tcn_residual["candidate_id"])
            if chosen_tcn_residual
            else None
        ),
    )
    chosen_ensemble = None
    if ensemble_frames:
        for beta in ENSEMBLE_BETA_VALUES:
            candidate_configs[f"ensemble_beta{beta:g}"] = {
                "beta": beta,
                "ridge_candidate_id": (
                    chosen_ridge["candidate_id"]
                    if chosen_ridge
                    else None
                ),
                "tcn_candidate_id": (
                    chosen_tcn_residual["candidate_id"]
                    if chosen_tcn_residual
                    else None
                ),
                "correction_policy": (
                    "blend each model's scaled residual correction, "
                    "then add EWMA4"
                ),
            }
        oof = pd.concat(
            [oof, *ensemble_frames],
            ignore_index=True,
            sort=False,
        )
        by_year, summary = _metrics_for_predictions(oof)
        chosen_ensemble = _choose_with_simplicity_margin(
            summary,
            model_names=["ridge_tcn_ensemble"],
        )

    ridge_score = (
        float(chosen_ridge["selection_score"])
        if chosen_ridge
        else np.nan
    )
    new_scores = [
        float(item["selection_score"])
        for item in (chosen_ordinal, chosen_tcn)
        if item is not None
        and item.get("selection_score") is not None
    ]
    replacement_gate = bool(
        np.isfinite(ridge_score)
        and new_scores
        and max(new_scores)
        > ridge_score + SIMPLICITY_MARGIN
    )

    final_training_history = bundle.outings.loc[
        pd.to_numeric(
            bundle.outings["year"], errors="coerce"
        ).between(2020, 2024)
        & bundle.outings[
            "history_eligible"
        ].fillna(False).astype(bool)
    ].copy()
    final_primary_mapping = fit_primary_fastball_mapping(
        final_training_history,
        fold_year=2025,
    )
    final_mapping = {
        str(row.pitcher): str(row.primary_fb_type)
        for row in final_primary_mapping.itertuples(index=False)
        if pd.notna(row.primary_fb_type)
    }
    mapping_table = pd.concat(
        [
            *mappings,
            final_primary_mapping.assign(fold_year=2025),
        ],
        ignore_index=True,
        sort=False,
    )
    mapping_table.to_csv(
        output / "primary_fastball_mapping_by_fold.csv",
        index=False,
    )

    best_ridge_id = (
        str(chosen_ridge["candidate_id"])
        if chosen_ridge
        else None
    )
    best_xgb_id = (
        str(chosen_xgb["candidate_id"])
        if chosen_xgb
        else None
    )
    best_ordinal_id = (
        str(chosen_ordinal["candidate_id"])
        if chosen_ordinal
        else None
    )
    best_tcn_id = (
        str(chosen_tcn["candidate_id"])
        if chosen_tcn
        else None
    )
    qualification_metadata = dict(
        bundle.qualification_metadata
    )
    ensemble_id = (
        str(chosen_ensemble["candidate_id"])
        if chosen_ensemble
        else None
    )
    ensemble_beta = (
        float(
            candidate_configs[ensemble_id]["beta"]
        )
        if ensemble_id
        else 0.0
    )
    existing_candidate_ids = {
        "ewma": (
            "ewma_span4"
            if "ewma" in selected_models
            else None
        ),
        "ridge": best_ridge_id,
        "xgboost": best_xgb_id,
    }
    lock: dict[str, Any] = {
        "schema_version": 1,
        "train_start": 2020,
        "validation_years": list(VALIDATION_YEARS),
        "final_train_end": 2024,
        "test_year": 2025,
        "qualified_pitchers": list(
            bundle.qualified_pitchers
        ),
        "qualification_rule": bundle.qualification_rule,
        "qualification": {
            "active_mode": qualification_metadata.get(
                "mode", bundle.source_mode
            ),
            "active_rule": bundle.qualification_rule,
            "active_fit_years": list(
                bundle.qualification_fit_years
            ),
            "research_rule": (
                "target_eligible official regular-season starts "
                ">=20 in every season 2021-2025"
            ),
            "research_years": list(
                RESEARCH_QUALIFICATION_YEARS
            ),
            "research_min_starts_per_year": 20,
            "uses_2025_roster_availability": bool(
                qualification_metadata.get(
                    "uses_2025_roster_availability",
                    2025 in bundle.qualification_fit_years,
                )
            ),
            "selection_validation_years": list(
                VALIDATION_YEARS
            ),
            "selection_uses_2025_outcomes": False,
            "counts_by_pitcher_year": (
                qualification_metadata.get(
                    "counts_by_pitcher_year", {}
                )
            ),
        },
        "target_definition": {
            "type": "pitcher_specific_training_tertile",
            "q_low": 0.3333333333,
            "q_high": 0.6666666667,
            "class_boundaries": {
                "low": "value <= q33",
                "middle": "q33 < value <= q67",
                "high": "value > q67",
            },
        },
        "pitch_type_mapping": {
            **default_pitch_type_mapping(
                cutter_category
            ),
            "cutter_category": cutter_category,
            "primary_candidates": ["FF", "SI", "FA"],
            "primary_mapping_fit_policy": (
                "per pitcher, fold-training usage only"
            ),
            "final_primary_fit_years": list(range(2020, 2025)),
            "final_primary_by_pitcher": final_mapping,
        },
        "ewma": {
            "status": (
                "selected"
                if "ewma" in selected_models
                else "not_requested"
            ),
            "span": 4,
            "candidate_id": (
                "ewma_span4"
                if "ewma" in selected_models
                else None
            ),
        },
        "ridge": {
            **_model_lock_entry(
                requested="ridge" in selected_models,
                chosen=chosen_ridge,
                candidate_configs=candidate_configs,
            ),
            "candidate_grid": {
                "alpha": list(ridge_alphas),
                "correction_scale": list(ridge_scales),
            },
        },
        "xgboost": {
            **_model_lock_entry(
                requested="xgboost" in selected_models,
                chosen=chosen_xgb,
                candidate_configs=candidate_configs,
            ),
            "search": {
                "protocol": (
                    xgboost_search.get("protocol")
                    if xgboost_search
                    else None
                ),
                "n_trials": (
                    int(xgboost_trials)
                    if "xgboost" in selected_models
                    else 0
                ),
            },
        },
        "ordinal_linear": {
            **_model_lock_entry(
                requested="ordinal" in selected_models,
                chosen=chosen_ordinal,
                candidate_configs=candidate_configs,
            ),
            "candidate_grid": {
                "losses": [
                    "ordinal_bce",
                    "ordinal_bce_plus_distance",
                ],
                "l2_lambda": list(ordinal_l2),
                "distance_weight": list(ordinal_distances),
            },
        },
        "shared_tcn": {
            **_model_lock_entry(
                requested="tcn" in selected_models,
                chosen=chosen_tcn,
                candidate_configs=candidate_configs,
            ),
            "candidate_grid": {
                "sequence_length": list(lengths),
                "ablation_reference_sequence_length": max(lengths),
                "sequence_length_design": (
                    "all architecture ablations at the longest requested "
                    "length; ordered H1 pitcher-bias reference at every "
                    "requested length"
                ),
                "n_training_configurations": len(tcn_search_plan),
                "alpha": list(tcn_alphas),
                "head_type": sorted(
                    {
                        str(template["head_type"])
                        for template in tcn_templates
                    }
                ),
                "sequence_order": sorted(
                    {
                        str(template["sequence_order"])
                        for template in tcn_templates
                    }
                ),
                "internal_channels": sorted(
                    {
                        int(template["internal_channels"])
                        for template in tcn_templates
                    }
                ),
                "bottleneck_dim": sorted(
                    {
                        int(template["bottleneck_dim"])
                        for template in tcn_templates
                    }
                ),
                "ordinal_weight": sorted(
                    {
                        float(template["ordinal_weight"])
                        for template in tcn_templates
                    }
                ),
                "epochs": int(tcn_epochs),
            },
            "ensemble": {
                "selected_candidate_id": ensemble_id,
                "beta": ensemble_beta,
                "ridge_candidate_id": best_ridge_id,
                "tcn_candidate_id": (
                    str(
                        chosen_tcn_residual[
                            "candidate_id"
                        ]
                    )
                    if chosen_tcn_residual
                    else None
                ),
                "correction_policy": (
                    "blend each model's scaled residual "
                    "correction, then add EWMA4"
                ),
            },
            "loss_weighting": (
                "mean within pitcher, then equal mean "
                "across pitchers"
            ),
            "shuffle_seed": DEFAULT_RANDOM_SEEDS[0],
        },
        "feature_lists": {
            "tabular_compact_16": list(
                COMPACT_FEATURES
            ),
            "ordinal": list(ORDINAL_FEATURES),
            "tcn_raw_15": list(TCN_RAW_FEATURES),
        },
        "missing_value_policy": {
            "tabular": (
                "complete-case on actual configured model "
                "features only; train-fitted Ridge scaling"
            ),
            "ordinal": (
                "per-pitcher training imputation/scaling "
                "with pooled training fallback"
            ),
            "tcn": (
                "pitcher training median, pooled training "
                "median fallback; pitcher training scaling; "
                "availability and valid-timestep masks"
            ),
        },
        "random_seeds": list(seeds),
        "random_seed_policy": {
            "model_seeds": list(seeds),
            "sequence_shuffle_seed": (
                DEFAULT_RANDOM_SEEDS[0]
            ),
            "minimum_tcn_seeds_normal_mode": 5,
        },
        "selection_metric": (
            "mean balanced_accuracy over validation seed-year observations; "
            "selection_score = mean - 0.5 * seed-year std; "
            "0.003 simplicity margin"
        ),
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "dataset_fingerprint_columns": list(
            bundle.fingerprint_columns
        ),
        "fold_manifest_fingerprint": (
            manifest_fingerprint
        ),
        "qualification_fit_years": list(
            bundle.qualification_fit_years
        ),
        "code_version": analysis_code_version(),
        "adopted_existing_models": [
            *(
                ["ewma"]
                if "ewma" in selected_models
                else []
            ),
            *(
                ["ridge"]
                if chosen_ridge is not None
                else []
            ),
            *(
                ["xgboost"]
                if chosen_xgb is not None
                else []
            ),
        ],
        "adopted_existing_candidates": (
            existing_candidate_ids
        ),
        "best_ordinal_candidate": chosen_ordinal,
        "best_tcn_candidate": chosen_tcn,
        "replacement_gate_passed": replacement_gate,
        "replacement_gate_definition": (
            "diagnostic only: best new-model selection_score "
            "must exceed selected Ridge by >0.003"
        ),
        "candidate_configs": candidate_configs,
        "requested_models": list(selected_models),
        "source_mode": bundle.source_mode,
        "source_metadata": dict(bundle.source_metadata),
        "selection_data_max_year": int(
            development["year"].max()
        ),
        "selection_uses_2025_outcomes": False,
        "smoke_test": bool(smoke_test),
    }
    save_locked_model_config(
        lock, output / "locked_model_config.json"
    )

    oof.to_parquet(
        output / "validation_oof_predictions.parquet",
        index=False,
    )
    by_year.to_csv(
        output / "validation_metrics_by_year.csv",
        index=False,
    )
    by_year.to_csv(
        output / "validation_metrics.csv",
        index=False,
    )
    summary.to_csv(
        output / "validation_metrics_summary.csv",
        index=False,
    )
    by_seed = _seed_metric_table(by_year)
    by_seed.to_csv(
        output / "validation_metrics_by_seed.csv",
        index=False,
    )
    feature_coverage = _feature_coverage(fold_histories)
    feature_coverage.to_csv(
        output / "feature_coverage.csv", index=False
    )
    report = {
        "development_max_year": int(
            development["year"].max()
        ),
        "validation_years_scored": list(
            VALIDATION_YEARS
        ),
        "test_year_predictions_used": False,
        "roster_may_use_2025_availability": bool(
            lock["qualification"][
                "uses_2025_roster_availability"
            ]
        ),
        "selection_uses_2025_outcomes": False,
        "dataset_fingerprint": (
            bundle.dataset_fingerprint
        ),
        "fold_manifest_fingerprint": (
            manifest_fingerprint
        ),
        "best_ridge": chosen_ridge,
        "best_xgboost": chosen_xgb,
        "best_ordinal": chosen_ordinal,
        "best_tcn": chosen_tcn,
        "best_ensemble": chosen_ensemble,
        "replacement_gate_passed": replacement_gate,
        "n_oof_rows": int(len(oof)),
        "tcn_search_plan": [
            {
                "sequence_length": length,
                "template_name": template["name"],
            }
            for length, template in tcn_search_plan
        ],
        "smoke_test": bool(smoke_test),
    }
    (
        output / "selection_report.json"
    ).write_text(
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
        "lock": lock,
        "manifest": manifest,
        "oof_predictions": oof,
        "metrics_by_year": by_year,
        "metrics_summary": summary,
        "metrics_by_seed": by_seed,
        "feature_coverage": feature_coverage,
        "output_dir": output,
    }
