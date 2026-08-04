from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from lib.stuff_mlb_dataset import FEATURES, make_dataset
from stuff_mlb_temporal_final import (
    ModelConfig,
    _classes,
    _feature_names,
    _metrics,
    _prepare_data,
)


SEED = 20260722
TRAIN_START = 2020
SEARCH_YEARS = (2022, 2023, 2024)
SPAN = 4
N_TRIALS = 32
PLAYER_WORKERS = 4
STAGE2_KEEP = 10
STAGE3_KEEP = 3
SIMPLICITY_MARGIN = 0.003


@dataclass(frozen=True)
class FoldPlayer:
    pitcher: int
    game_dates: np.ndarray
    target: np.ndarray
    x_train: np.ndarray
    residual_train: np.ndarray
    x_test: np.ndarray
    ewma_test: np.ndarray
    true_class: np.ndarray
    q33: float
    q67: float


def _compact_features() -> list[str]:
    return _feature_names(ModelConfig("xgboost", SPAN, "compact", None, 1.0, "search"))


def _build_fold(data: pd.DataFrame, year: int) -> list[FoldPlayer]:
    ewma = f"stuff_ewma{SPAN}_prior"
    features = _compact_features()
    required = [*FEATURES, "target_y", ewma]
    fold = []
    for pitcher, player in data.groupby("pitcher", sort=False):
        player = (
            player.sort_values("game_date")
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=required)
        )
        train = player.loc[player["year"].between(TRAIN_START, year - 1)]
        test = player.loc[player["year"].eq(year)]
        if len(train) < 20 or test.empty:
            continue
        q33, q67 = train["target_y"].quantile([1 / 3, 2 / 3]).to_numpy(float)
        fold.append(FoldPlayer(
            pitcher=int(pitcher),
            game_dates=test["game_date"].to_numpy(),
            target=test["target_y"].to_numpy(np.float32),
            x_train=np.ascontiguousarray(train[features].to_numpy(np.float32)),
            residual_train=np.ascontiguousarray(
                (train["target_y"] - train[ewma]).to_numpy(np.float32)
            ),
            x_test=np.ascontiguousarray(test[features].to_numpy(np.float32)),
            ewma_test=np.ascontiguousarray(test[ewma].to_numpy(np.float32)),
            true_class=_classes(test["target_y"], q33, q67),
            q33=float(q33),
            q67=float(q67),
        ))
    return fold


def _sample_params(trial: optuna.Trial) -> dict[str, float | int]:
    use_alpha = trial.suggest_categorical("use_reg_alpha", [0, 1])
    use_gamma = trial.suggest_categorical("use_gamma", [0, 1])
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 40.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 10.0, 500.0, log=True),
        "reg_alpha": (
            trial.suggest_float("reg_alpha_nonzero", 0.01, 10.0, log=True)
            if use_alpha else 0.0
        ),
        "gamma": (
            trial.suggest_float("gamma_nonzero", 0.01, 2.0, log=True)
            if use_gamma else 0.0
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "correction_scale": trial.suggest_float("correction_scale", 0.1, 0.75),
    }


def _model_params(params: dict[str, float | int]) -> dict[str, float | int | str]:
    return {
        key: value for key, value in params.items() if key != "correction_scale"
    } | {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_bin": 64,
        "random_state": SEED,
        "n_jobs": 1,
        "verbosity": 0,
    }


def _evaluate_fold(
    params: dict[str, float | int],
    fold: list[FoldPlayer],
    return_predictions: bool = False,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    actual_parts = []
    predicted_parts = []
    prediction_rows = []
    scale = float(params["correction_scale"])

    def fit_player(player: FoldPlayer) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
        model = XGBRegressor(**_model_params(params))
        model.fit(player.x_train, player.residual_train)
        score = player.ewma_test + scale * model.predict(player.x_test)
        prediction = _classes(score, player.q33, player.q67)
        frame = None
        if return_predictions:
            frame = pd.DataFrame({
                "pitcher": player.pitcher,
                "game_date": player.game_dates,
                "target_y": player.target,
                "q33": player.q33,
                "q67": player.q67,
                "true_class": player.true_class,
                "predicted_score": score,
                "predicted_class": prediction,
            })
        return player.true_class, prediction, frame

    with ThreadPoolExecutor(max_workers=PLAYER_WORKERS) as executor:
        results = list(executor.map(fit_player, fold))
    for actual, prediction, frame in results:
        actual_parts.append(actual)
        predicted_parts.append(prediction)
        if frame is not None:
            prediction_rows.append(frame)
    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(predicted_parts)
    frame = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else None
    return _metrics(actual, predicted), frame


def _baseline_ewma(fold: list[FoldPlayer]) -> dict[str, float]:
    actual = np.concatenate([player.true_class for player in fold])
    predicted = np.concatenate([
        _classes(player.ewma_test, player.q33, player.q67) for player in fold
    ])
    return _metrics(actual, predicted)


def _evaluate_ridge_fold(
    fold: list[FoldPlayer], return_predictions: bool = False
) -> tuple[dict[str, float], pd.DataFrame | None]:
    actual_parts = []
    predicted_parts = []
    prediction_rows = []
    for player in fold:
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        model.fit(player.x_train, player.residual_train)
        score = player.ewma_test + 0.5 * model.predict(player.x_test)
        prediction = _classes(score, player.q33, player.q67)
        actual_parts.append(player.true_class)
        predicted_parts.append(prediction)
        if return_predictions:
            prediction_rows.append(pd.DataFrame({
                "pitcher": player.pitcher,
                "game_date": player.game_dates,
                "target_y": player.target,
                "q33": player.q33,
                "q67": player.q67,
                "true_class": player.true_class,
                "predicted_score": score,
                "predicted_class": prediction,
            }))
    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(predicted_parts)
    frame = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else None
    return _metrics(actual, predicted), frame


def _ridge_validation(caches: dict[int, list[FoldPlayer]]) -> dict[str, object]:
    row: dict[str, object] = {}
    for year in SEARCH_YEARS:
        metrics, _ = _evaluate_ridge_fold(caches[year])
        for name, value in metrics.items():
            row[f"{name}_{year}"] = value
    return _aggregate(row, SEARCH_YEARS)


def _agreement_summary(xgboost: pd.DataFrame, ridge: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    ridge = ridge[["pitcher", "game_date", "predicted_score", "predicted_class"]].rename(
        columns={"predicted_score": "ridge_score", "predicted_class": "ridge_class"}
    )
    comparison = xgboost.rename(columns={
        "predicted_score": "tuned_xgboost_score",
        "predicted_class": "tuned_xgboost_class",
    }).merge(ridge, on=["pitcher", "game_date"], how="inner", validate="one_to_one")
    comparison["models_agree"] = comparison["tuned_xgboost_class"].eq(comparison["ridge_class"])
    agree = comparison["models_agree"]
    summary: dict[str, float | int] = {
        "n": len(comparison),
        "agreement_count": int(agree.sum()),
        "agreement_rate": float(agree.mean()),
        "accuracy_when_agree": float(
            comparison.loc[agree, "tuned_xgboost_class"].eq(
                comparison.loc[agree, "true_class"]
            ).mean()
        ),
        "xgboost_accuracy_when_disagree": float(
            comparison.loc[~agree, "tuned_xgboost_class"].eq(
                comparison.loc[~agree, "true_class"]
            ).mean()
        ),
        "ridge_accuracy_when_disagree": float(
            comparison.loc[~agree, "ridge_class"].eq(
                comparison.loc[~agree, "true_class"]
            ).mean()
        ),
    }
    return comparison, summary


def _aggregate(row: dict[str, object], years: tuple[int, ...]) -> dict[str, object]:
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "ordinal_mae"):
        values = np.array([float(row[f"{metric}_{year}"]) for year in years])
        row[f"mean_{metric}"] = float(values.mean())
        row[f"std_{metric}"] = float(values.std(ddof=0))
    row["selection_score"] = (
        float(row["mean_balanced_accuracy"])
        - 0.5 * float(row["std_balanced_accuracy"])
    )
    return row


def _evaluate_years(
    params: dict[str, float | int],
    caches: dict[int, list[FoldPlayer]],
    years: tuple[int, ...],
) -> dict[str, object]:
    row: dict[str, object] = dict(params)
    for year in years:
        metrics, _ = _evaluate_fold(params, caches[year])
        row[f"n_{year}"] = int(sum(len(player.true_class) for player in caches[year]))
        for name, value in metrics.items():
            row[f"{name}_{year}"] = value
    return _aggregate(row, years)


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["selection_score", "mean_ordinal_mae", "mean_macro_f1"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def _choose_simple_final(stage3: pd.DataFrame) -> pd.Series:
    best = float(stage3["selection_score"].max())
    eligible = stage3.loc[stage3["selection_score"].ge(best - SIMPLICITY_MARGIN)].copy()
    return eligible.sort_values(
        ["max_depth", "correction_scale", "reg_lambda", "min_child_weight", "n_estimators"],
        ascending=[True, True, False, False, True],
    ).iloc[0]


def _params_from_row(row: pd.Series) -> dict[str, float | int]:
    integer = {"max_depth", "n_estimators"}
    names = [
        "max_depth", "n_estimators", "learning_rate", "min_child_weight",
        "reg_lambda", "reg_alpha", "gamma", "subsample", "colsample_bytree",
        "correction_scale",
    ]
    return {
        name: int(row[name]) if name in integer else float(row[name])
        for name in names
    }


def _legacy_run_for_reproduction(
    statcast_dirs: list[Path], stuff_paths: list[Path], output_dir: Path
) -> dict[str, object]:
    """Historical mixed select-and-test implementation.

    Kept only so previously published results remain auditable.  The public
    ``run`` and CLI below route through the unified development-only selector.
    """
    started = time.perf_counter()
    data, qualified = make_dataset(statcast_dirs, stuff_paths)
    data = _prepare_data(data)
    development = data.loc[data["year"].le(2024)].copy()
    if development["year"].max() > 2024:
        raise ValueError("Search received post-2024 data.")
    caches = {year: _build_fold(development, year) for year in SEARCH_YEARS}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=12)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        metrics, _ = _evaluate_fold(params, caches[2023])
        trial.set_user_attr("params_resolved", params)
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return metrics["balanced_accuracy"]

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)
    stage1_rows = []
    for trial in study.trials:
        params = dict(trial.user_attrs["params_resolved"])
        row: dict[str, object] = {"trial": trial.number, **params}
        row["n_2023"] = int(sum(len(player.true_class) for player in caches[2023]))
        for name in ("accuracy", "balanced_accuracy", "macro_f1", "ordinal_mae"):
            row[f"{name}_2023"] = float(trial.user_attrs[name])
        stage1_rows.append(_aggregate(row, (2023,)))
    stage1 = _rank(pd.DataFrame(stage1_rows))

    stage2_rows = []
    for _, candidate in stage1.head(STAGE2_KEEP).iterrows():
        params = _params_from_row(candidate)
        row = _evaluate_years(params, caches, (2023, 2024))
        row["trial"] = int(candidate["trial"])
        stage2_rows.append(row)
    stage2 = _rank(pd.DataFrame(stage2_rows))

    stage3_rows = []
    for _, candidate in stage2.head(STAGE3_KEEP).iterrows():
        params = _params_from_row(candidate)
        row = _evaluate_years(params, caches, SEARCH_YEARS)
        row["trial"] = int(candidate["trial"])
        stage3_rows.append(row)
    stage3 = _rank(pd.DataFrame(stage3_rows))
    winner = _choose_simple_final(stage3)
    locked_params = _params_from_row(winner)

    test_cache = _build_fold(data, 2025)
    test_metrics, test_predictions = _evaluate_fold(
        locked_params, test_cache, return_predictions=True
    )
    ewma_metrics = _baseline_ewma(test_cache)
    ridge_validation = _ridge_validation(caches)
    ridge_test_metrics, ridge_predictions = _evaluate_ridge_fold(
        test_cache, return_predictions=True
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(output_dir / "stage1_32_candidates_2023.csv", index=False)
    stage2.to_csv(output_dir / "stage2_top10_2023_2024.csv", index=False)
    stage3.to_csv(output_dir / "stage3_top3_2022_2024.csv", index=False)
    assert test_predictions is not None
    assert ridge_predictions is not None
    test_predictions.to_parquet(output_dir / "locked_xgboost_2025_predictions.parquet", index=False)
    ridge_predictions.to_parquet(output_dir / "ridge_benchmark_2025_predictions.parquet", index=False)
    comparison, agreement = _agreement_summary(test_predictions, ridge_predictions)
    comparison.to_parquet(output_dir / "tuned_xgboost_vs_ridge_2025.parquet", index=False)

    result: dict[str, object] = {
        "search_guard": "all hyperparameter selection uses rows through 2024 only",
        "qualified_pitchers": qualified,
        "feature_set": "compact_16",
        "ewma_span": SPAN,
        "protocol": {
            "stage1": "32 TPE trials on 2023; keep 10",
            "stage2": "top 10 on 2023-2024; keep 3",
            "stage3": "top 3 on 2022-2024; choose within 0.3pp by simplicity",
            "selection_score": "mean balanced accuracy - 0.5 * yearly std",
        },
        "locked_params": locked_params,
        "validation": {
            key: float(winner[key])
            for key in winner.index
            if key.startswith(("mean_", "std_", "selection_score"))
        },
        "ridge_validation": ridge_validation,
        "xgboost_balanced_accuracy_advantage": (
            float(winner["mean_balanced_accuracy"])
            - float(ridge_validation["mean_balanced_accuracy"])
        ),
        "xgboost_passes_0_5pp_gate": (
            float(winner["mean_balanced_accuracy"])
            - float(ridge_validation["mean_balanced_accuracy"])
        ) >= 0.005,
        "test_2025": test_metrics,
        "ridge_test_2025": ridge_test_metrics,
        "ewma4_test_2025": ewma_metrics,
        "dual_model_agreement_2025": agreement,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "locked_xgboost_config.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def run(
    statcast_dirs: list[Path],
    stuff_paths: list[Path],
    output_dir: Path,
    **selection_options: object,
) -> dict[str, object]:
    """Compatibility API for the unified pre-2025 XGBoost selection path."""

    from lib.stuff_cli import load_stuff_data
    from lib.stuff_selection import run_model_selection

    bundle = load_stuff_data(
        statcast_paths=statcast_dirs,
        stuff_paths=stuff_paths,
        qualification_mode="research",
    )
    return run_model_selection(
        bundle,
        output_dir=output_dir,
        models=("ewma", "ridge", "xgboost"),
        **selection_options,
    )


def main(argv: list[str] | None = None) -> int:
    """Forward the legacy command name to ``select_models.py``.

    EWMA, Ridge, and XGBoost are the default families for this compatibility
    entrypoint.  It never evaluates the reserved 2025 outcomes.
    """

    import sys

    from select_models import main as unified_main

    forwarded = list(sys.argv[1:] if argv is None else argv)
    if "--models" not in forwarded:
        forwarded.extend(["--models", "ewma", "ridge", "xgboost"])
    return unified_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
