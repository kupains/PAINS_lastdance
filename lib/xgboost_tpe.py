"""Bounded, development-only XGBoost search.

This preserves the repository's original 32 -> 10 -> 3 staged TPE protocol,
while exposing it as a reusable function for the unified selection entrypoint.
No 2025 fold is constructed or scored here.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from lib.evaluation import evaluate_prediction_frame


SEARCH_YEARS = (2022, 2023, 2024)
SIMPLICITY_MARGIN = 0.003

# Rounded values recorded by the existing final report.  This profile is kept
# as a named reference candidate; a normal 32-trial run still uses the exact
# original seeded TPE trajectory.
LEGACY_LOCKED_XGBOOST_PROFILE: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 2,
    "learning_rate": 0.04221,
    "min_child_weight": 15.043,
    "reg_lambda": 16.823,
    "reg_alpha": 0.706,
    "gamma": 0.0,
    "subsample": 0.9997,
    "colsample_bytree": 0.5856,
    "tree_method": "hist",
    "max_bin": 64,
}
LEGACY_LOCKED_CORRECTION_SCALE = 0.575


def _sample_params(trial: Any) -> dict[str, float | int]:
    use_alpha = trial.suggest_categorical("use_reg_alpha", [0, 1])
    use_gamma = trial.suggest_categorical("use_gamma", [0, 1])
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "n_estimators": trial.suggest_int(
            "n_estimators", 50, 300, step=25
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.08, log=True
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 5.0, 40.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 10.0, 500.0, log=True
        ),
        "reg_alpha": (
            trial.suggest_float(
                "reg_alpha_nonzero", 0.01, 10.0, log=True
            )
            if use_alpha
            else 0.0
        ),
        "gamma": (
            trial.suggest_float("gamma_nonzero", 0.01, 2.0, log=True)
            if use_gamma
            else 0.0
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),
        "correction_scale": trial.suggest_float(
            "correction_scale", 0.1, 0.75
        ),
    }


def _profile_from_params(
    params: dict[str, float | int],
) -> tuple[dict[str, Any], float]:
    profile = {
        key: value
        for key, value in params.items()
        if key != "correction_scale"
    }
    profile.update({"tree_method": "hist", "max_bin": 64})
    return profile, float(params["correction_scale"])


def _aggregate(
    row: dict[str, Any], years: Iterable[int]
) -> dict[str, Any]:
    years = tuple(int(value) for value in years)
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ordinal_mae",
        "high_recall",
        "extreme_error_count",
        "coverage",
    ):
        values = np.asarray(
            [float(row[f"{metric}_{year}"]) for year in years],
            dtype=float,
        )
        row[f"mean_{metric}"] = float(np.nanmean(values))
        row[f"std_{metric}"] = float(np.nanstd(values, ddof=0))
    row["selection_score"] = float(
        row["mean_balanced_accuracy"]
        - 0.5 * row["std_balanced_accuracy"]
    )
    return row


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        ["selection_score", "mean_ordinal_mae", "mean_macro_f1"],
        ascending=[False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _choose_simple_final(stage3: pd.DataFrame) -> pd.Series:
    if stage3.empty:
        raise ValueError("No XGBoost candidates reached the final stage")
    best = float(stage3["selection_score"].max())
    eligible = stage3.loc[
        stage3["selection_score"].ge(best - SIMPLICITY_MARGIN)
    ].copy()
    return eligible.sort_values(
        [
            "max_depth",
            "correction_scale",
            "reg_lambda",
            "min_child_weight",
            "n_estimators",
        ],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    ).iloc[0]


def _row_params(row: pd.Series) -> dict[str, float | int]:
    integer = {"max_depth", "n_estimators"}
    names = [
        "max_depth",
        "n_estimators",
        "learning_rate",
        "min_child_weight",
        "reg_lambda",
        "reg_alpha",
        "gamma",
        "subsample",
        "colsample_bytree",
        "correction_scale",
    ]
    return {
        name: int(row[name]) if name in integer else float(row[name])
        for name in names
    }


def _evaluate_params(
    params: dict[str, float | int],
    folds: dict[int, Any],
    years: Iterable[int],
    *,
    candidate_id: str,
) -> dict[str, Any]:
    # Local import avoids an import cycle while stuff_experiment imports the
    # generic tabular estimator.
    from lib.stuff_experiment import predict_residual_fold

    profile, correction_scale = _profile_from_params(params)
    row: dict[str, Any] = {**params, "candidate_id": candidate_id}
    for year in tuple(int(value) for value in years):
        prediction, _ = predict_residual_fold(
            folds[year],
            model="xgboost",
            correction_scale=correction_scale,
            xgboost_params=profile,
            candidate_id=candidate_id,
        )
        metrics = evaluate_prediction_frame(prediction)
        row[f"n_{year}"] = int(metrics["n_evaluated"])
        for name in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "ordinal_mae",
            "high_recall",
            "extreme_error_count",
            "coverage",
        ):
            row[f"{name}_{year}"] = float(metrics[name])
    return _aggregate(row, years)


def search_xgboost_tpe(
    development: pd.DataFrame,
    *,
    n_trials: int = 32,
    random_seed: int = 20260722,
    include_legacy_reference: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the original staged search using only 2022-2024 folds."""

    if int(n_trials) < 1:
        raise ValueError("n_trials must be positive")
    years = pd.to_numeric(development["year"], errors="raise")
    if int(years.max()) > 2024:
        raise ValueError("XGBoost TPE development data must end by 2024")
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost TPE selection requires Optuna; install requirements.txt"
        ) from exc

    from lib.stuff_experiment import prepare_fold

    folds = {
        year: prepare_fold(
            development, year, evaluation_split="validation"
        )
        for year in SEARCH_YEARS
    }
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=int(random_seed), n_startup_trials=min(12, int(n_trials))
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # A one-trial smoke run uses the recorded profile and still traverses the
    # same Optuna objective.  Full runs preserve the old un-enqueued trajectory.
    if int(n_trials) == 1:
        smoke_params = {
            **{
                key: value
                for key, value in LEGACY_LOCKED_XGBOOST_PROFILE.items()
                if key not in {"tree_method", "max_bin"}
            },
            "correction_scale": LEGACY_LOCKED_CORRECTION_SCALE,
            "use_reg_alpha": 1,
            "reg_alpha_nonzero": LEGACY_LOCKED_XGBOOST_PROFILE["reg_alpha"],
            "use_gamma": 0,
        }
        study.enqueue_trial(smoke_params)

    def objective(trial: Any) -> float:
        params = _sample_params(trial)
        result = _evaluate_params(
            params,
            folds,
            (2023,),
            candidate_id=f"xgboost_tpe_trial_{trial.number}",
        )
        trial.set_user_attr("resolved_params", params)
        trial.set_user_attr("stage1_metrics", result)
        trial.set_user_attr("validation_years", [2023])
        return float(result["balanced_accuracy_2023"])

    study.optimize(
        objective,
        n_trials=int(n_trials),
        n_jobs=1,
        show_progress_bar=False,
    )
    stage1_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        params = dict(trial.user_attrs["resolved_params"])
        row = dict(trial.user_attrs["stage1_metrics"])
        row.update(
            {
                "stage": "stage1_2023",
                "trial": int(trial.number),
                **params,
            }
        )
        stage1_rows.append(row)
    stage1 = _rank(pd.DataFrame(stage1_rows))

    stage2_rows: list[dict[str, Any]] = []
    for _, candidate in stage1.head(min(10, len(stage1))).iterrows():
        params = _row_params(candidate)
        row = _evaluate_params(
            params,
            folds,
            (2023, 2024),
            candidate_id=f"xgboost_tpe_trial_{int(candidate['trial'])}",
        )
        row.update(
            {"stage": "stage2_2023_2024", "trial": int(candidate["trial"])}
        )
        stage2_rows.append(row)
    stage2 = _rank(pd.DataFrame(stage2_rows))

    stage3_rows: list[dict[str, Any]] = []
    for _, candidate in stage2.head(min(3, len(stage2))).iterrows():
        params = _row_params(candidate)
        row = _evaluate_params(
            params,
            folds,
            SEARCH_YEARS,
            candidate_id=f"xgboost_tpe_trial_{int(candidate['trial'])}",
        )
        row.update(
            {"stage": "stage3_2022_2024", "trial": int(candidate["trial"])}
        )
        stage3_rows.append(row)

    if include_legacy_reference and int(n_trials) > 1:
        legacy_params: dict[str, float | int] = {
            key: value
            for key, value in LEGACY_LOCKED_XGBOOST_PROFILE.items()
            if key not in {"tree_method", "max_bin"}
        }
        legacy_params["correction_scale"] = (
            LEGACY_LOCKED_CORRECTION_SCALE
        )
        legacy = _evaluate_params(
            legacy_params,
            folds,
            SEARCH_YEARS,
            candidate_id="xgboost_legacy_locked_reference",
        )
        legacy.update({"stage": "stage3_2022_2024", "trial": -1})
        stage3_rows.append(legacy)

    stage3 = _rank(pd.DataFrame(stage3_rows))
    winner = _choose_simple_final(stage3)
    locked = _row_params(winner)
    profile, correction_scale = _profile_from_params(locked)
    result = {
        "protocol": "staged_tpe_32_to_10_to_3",
        "tpe_best_trial": int(winner["trial"]),
        "candidate_id": str(winner["candidate_id"]),
        "profile": profile,
        "correction_scale": correction_scale,
        "selection_score": float(winner["selection_score"]),
        "mean_balanced_accuracy": float(
            winner["mean_balanced_accuracy"]
        ),
        "std_balanced_accuracy": float(
            winner["std_balanced_accuracy"]
        ),
        "validation_years": list(SEARCH_YEARS),
        "random_seed": int(random_seed),
        "n_trials": int(n_trials),
    }
    trials = pd.concat(
        [stage1, stage2, stage3], ignore_index=True, sort=False
    )
    return result, trials


__all__ = [
    "LEGACY_LOCKED_CORRECTION_SCALE",
    "LEGACY_LOCKED_XGBOOST_PROFILE",
    "SEARCH_YEARS",
    "search_xgboost_tpe",
]
