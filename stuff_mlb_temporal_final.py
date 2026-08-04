from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from lib.stuff_mlb_dataset import FEATURES, make_dataset


TRAIN_START = 2020
VALIDATION_YEARS = (2023, 2024)
FINAL_TRAIN_END = 2024
TEST_YEAR = 2025
EWMA_SPANS = (2, 3, 4, 5, 6, 8)
RIDGE_ALPHAS = (1.0, 10.0, 30.0, 100.0)
CORRECTION_SCALES = (0.25, 0.5, 1.0)
FEATURE_SETS = ("compact", "full")
XGB_PROFILES = {
    "shallow": {
        "n_estimators": 100,
        "max_depth": 1,
        "learning_rate": 0.03,
        "min_child_weight": 5,
    },
    "moderate": {
        "n_estimators": 150,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
    },
}


@dataclass(frozen=True)
class ModelConfig:
    family: str
    span: int
    feature_set: str | None = None
    alpha: float | None = None
    correction_scale: float = 0.0
    profile: str | None = None

    @property
    def model_id(self) -> str:
        if self.family == "ewma":
            return f"ewma_span{self.span}"
        if self.family == "ridge":
            return (
                f"ridge_span{self.span}_{self.feature_set}_alpha{self.alpha:g}"
                f"_scale{self.correction_scale:g}"
            )
        return (
            f"xgboost_span{self.span}_{self.feature_set}_{self.profile}"
            f"_scale{self.correction_scale:g}"
        )


def _classes(values: pd.Series | np.ndarray, q33: float, q67: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.select([values <= q33, values <= q67], [0, 1], default=2).astype(int)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "ordinal_mae": float(np.mean(np.abs(y - prediction))),
    }


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["pitcher", "game_date"]).copy()
    grouped = data.groupby("pitcher")["target_y"]
    for span in EWMA_SPANS:
        ewma = f"stuff_ewma{span}_prior"
        data[ewma] = grouped.transform(
            lambda values, value=span: values.shift(1).ewm(
                span=value, adjust=False, min_periods=1
            ).mean()
        )
        data[f"prior_minus_ewma{span}"] = data["prior_stuff_plus"] - data[ewma]
        data[f"mean5_minus_ewma{span}"] = data["stuff_plus_mean_last5"] - data[ewma]
    return data


def _feature_names(config: ModelConfig) -> list[str]:
    differences = [
        f"prior_minus_ewma{config.span}",
        f"mean5_minus_ewma{config.span}",
    ]
    if config.feature_set == "full":
        return list(dict.fromkeys([*FEATURES, *differences]))
    compact = [
        *differences,
        "stuff_plus_slope_last5",
        "prev_start_pitch_count",
        "rest_days",
        "workload_density_3starts",
        *[column for column in FEATURES if column.endswith("slope5")],
    ]
    return list(dict.fromkeys(compact))


def _candidate_configs() -> list[ModelConfig]:
    candidates = [ModelConfig("ewma", span) for span in EWMA_SPANS]
    for span in EWMA_SPANS:
        for feature_set in FEATURE_SETS:
            for scale in CORRECTION_SCALES:
                for alpha in RIDGE_ALPHAS:
                    candidates.append(ModelConfig(
                        "ridge", span, feature_set, alpha, scale
                    ))
                for profile in XGB_PROFILES:
                    candidates.append(ModelConfig(
                        "xgboost", span, feature_set, None, scale, profile
                    ))
    return candidates


def _predict_score(
    train: pd.DataFrame, test: pd.DataFrame, config: ModelConfig
) -> np.ndarray:
    ewma = f"stuff_ewma{config.span}_prior"
    base = test[ewma].to_numpy(float)
    if config.family == "ewma":
        return base

    feature_names = _feature_names(config)
    residual = train["target_y"] - train[ewma]
    if config.family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(config.alpha)))
    else:
        profile = XGB_PROFILES[str(config.profile)]
        model = XGBRegressor(
            **profile,
            objective="reg:squarederror",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    model.fit(train[feature_names], residual)
    return base + config.correction_scale * model.predict(test[feature_names])


def _fit_ridge_with_coefficients(
    train: pd.DataFrame, test: pd.DataFrame, config: ModelConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if config.family != "ridge":
        raise ValueError("Coefficient extraction requires a Ridge config.")
    ewma = f"stuff_ewma{config.span}_prior"
    feature_names = _feature_names(config)
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(config.alpha)))
    model.fit(train[feature_names], train["target_y"] - train[ewma])
    score = (
        test[ewma].to_numpy(float)
        + config.correction_scale * model.predict(test[feature_names])
    )
    coefficients = model.named_steps["ridge"].coef_ * config.correction_scale
    return score, np.asarray(coefficients, dtype=float), feature_names


def _complete_rows(player: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    required = [*FEATURES, "target_y", f"stuff_ewma{config.span}_prior"]
    return player.replace([np.inf, -np.inf], np.nan).dropna(subset=required)


def _validation_metrics(data: pd.DataFrame, config: ModelConfig) -> dict[str, float | str]:
    actual_by_year: dict[int, list[np.ndarray]] = {year: [] for year in VALIDATION_YEARS}
    predicted_by_year: dict[int, list[np.ndarray]] = {year: [] for year in VALIDATION_YEARS}

    for _, player in data.groupby("pitcher", sort=False):
        player = _complete_rows(player.sort_values("game_date"), config)
        for year in VALIDATION_YEARS:
            train = player.loc[player["year"].between(TRAIN_START, year - 1)]
            validation = player.loc[player["year"].eq(year)]
            if len(train) < 20 or validation.empty:
                continue
            q33, q67 = train["target_y"].quantile([1 / 3, 2 / 3]).to_numpy(float)
            actual = _classes(validation["target_y"], q33, q67)
            score = _predict_score(train, validation, config)
            actual_by_year[year].append(actual)
            predicted_by_year[year].append(_classes(score, q33, q67))

    row: dict[str, float | str] = {"model_id": config.model_id, **asdict(config)}
    yearly = []
    for year in VALIDATION_YEARS:
        actual = np.concatenate(actual_by_year[year])
        predicted = np.concatenate(predicted_by_year[year])
        values = _metrics(actual, predicted)
        yearly.append(values)
        row[f"n_{year}"] = len(actual)
        for name, value in values.items():
            row[f"{name}_{year}"] = value
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "ordinal_mae"):
        row[f"mean_{name}"] = float(np.mean([values[name] for values in yearly]))
    return row


def _config_from_row(row: pd.Series) -> ModelConfig:
    return ModelConfig(
        family=str(row["family"]),
        span=int(row["span"]),
        feature_set=None if pd.isna(row["feature_set"]) else str(row["feature_set"]),
        alpha=None if pd.isna(row["alpha"]) else float(row["alpha"]),
        correction_scale=float(row["correction_scale"]),
        profile=None if pd.isna(row["profile"]) else str(row["profile"]),
    )


def select_models(
    development_data: pd.DataFrame,
) -> tuple[ModelConfig, ModelConfig, pd.DataFrame]:
    if development_data["year"].max() > FINAL_TRAIN_END:
        raise ValueError("Model selection received post-2024 rows.")
    candidates = _candidate_configs()
    rows = [_validation_metrics(development_data, config) for config in candidates]
    results = pd.DataFrame(rows)
    complexity = results["family"].map({"ewma": 0, "ridge": 1, "xgboost": 2})
    results = (
        results.assign(complexity=complexity)
        .sort_values(
            ["mean_balanced_accuracy", "mean_ordinal_mae", "mean_macro_f1", "complexity"],
            ascending=[False, True, False, True],
        )
        .reset_index(drop=True)
    )
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    primary = _config_from_row(results.iloc[0])
    ridge = _config_from_row(results.loc[results["family"].eq("ridge")].iloc[0])
    return primary, ridge, results


def _final_test(
    data: pd.DataFrame,
    qualified: list[int],
    names: dict[int, str],
    primary: ModelConfig,
    ridge: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    coefficient_rows = []
    primary_name = f"primary:{primary.model_id}"
    ridge_name = f"interpretable:{ridge.model_id}"
    ewma_name = f"ewma_span{primary.span}"
    for pitcher in qualified:
        player = _complete_rows(
            data.loc[data["pitcher"].eq(pitcher)].sort_values("game_date"), primary
        )
        train = player.loc[player["year"].between(TRAIN_START, FINAL_TRAIN_END)]
        test = player.loc[player["year"].eq(TEST_YEAR)]
        if len(train) < 20 or test.empty:
            continue
        q33, q67 = train["target_y"].quantile([1 / 3, 2 / 3]).to_numpy(float)
        actual = _classes(test["target_y"], q33, q67)
        ridge_score, ridge_coefficients, ridge_features = _fit_ridge_with_coefficients(
            train, test, ridge
        )
        scores = {
            primary_name: _predict_score(train, test, primary),
            ridge_name: ridge_score,
            ewma_name: test[f"stuff_ewma{primary.span}_prior"].to_numpy(float),
            "prior_start": test["prior_stuff_plus"].to_numpy(float),
        }
        for feature, coefficient in zip(ridge_features, ridge_coefficients, strict=True):
            coefficient_rows.append({
                "pitcher": pitcher,
                "pitcher_name": names.get(pitcher, str(pitcher)),
                "feature": feature,
                "standardized_coefficient": coefficient,
            })
        train_classes = _classes(train["target_y"], q33, q67)
        majority = int(np.bincount(train_classes, minlength=3).argmax())
        for model_name, score in scores.items():
            frame = test[["pitcher", "game_date", "target_y"]].copy()
            frame["pitcher_name"] = names.get(pitcher, str(pitcher))
            frame["q33"] = q33
            frame["q67"] = q67
            frame["true_class"] = actual
            frame["model"] = model_name
            frame["predicted_score"] = score
            frame["predicted_class"] = _classes(score, q33, q67)
            frames.append(frame)
        majority_frame = test[["pitcher", "game_date", "target_y"]].copy()
        majority_frame["pitcher_name"] = names.get(pitcher, str(pitcher))
        majority_frame["q33"] = q33
        majority_frame["q67"] = q67
        majority_frame["true_class"] = actual
        majority_frame["model"] = "training_majority"
        majority_frame["predicted_score"] = np.nan
        majority_frame["predicted_class"] = majority
        frames.append(majority_frame)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coefficient_rows)


def _dual_model_comparison(
    predictions: pd.DataFrame, primary_name: str, ridge_name: str
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    keys = ["pitcher", "pitcher_name", "game_date", "target_y", "true_class"]
    adopted = predictions.loc[predictions["model"].isin([primary_name, ridge_name])]
    classes = adopted.pivot(index=keys, columns="model", values="predicted_class").reset_index()
    scores = adopted.pivot(index=keys, columns="model", values="predicted_score").reset_index()
    comparison = classes[keys].copy()
    comparison["xgboost_class"] = classes[primary_name].to_numpy(int)
    comparison["ridge_class"] = classes[ridge_name].to_numpy(int)
    comparison["xgboost_score"] = scores[primary_name].to_numpy(float)
    comparison["ridge_score"] = scores[ridge_name].to_numpy(float)
    comparison["models_agree"] = comparison["xgboost_class"].eq(comparison["ridge_class"])
    agree = comparison["models_agree"]
    summary: dict[str, float | int] = {
        "n": len(comparison),
        "agreement_count": int(agree.sum()),
        "agreement_rate": float(agree.mean()),
        "accuracy_when_agree": float(
            comparison.loc[agree, "xgboost_class"].eq(
                comparison.loc[agree, "true_class"]
            ).mean()
        ),
        "xgboost_accuracy_when_disagree": float(
            comparison.loc[~agree, "xgboost_class"].eq(
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


def _ridge_feature_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    summary = coefficients.groupby("feature")["standardized_coefficient"].agg(
        median_coefficient="median",
        mean_coefficient="mean",
        coefficient_q25=lambda values: values.quantile(0.25),
        coefficient_q75=lambda values: values.quantile(0.75),
        positive_share=lambda values: values.gt(0).mean(),
    ).reset_index()
    absolute = coefficients.assign(
        absolute_coefficient=coefficients["standardized_coefficient"].abs()
    ).groupby("feature")["absolute_coefficient"].median()
    summary["median_absolute_coefficient"] = summary["feature"].map(absolute)
    return summary.sort_values("median_absolute_coefficient", ascending=False)


def _summaries(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows = []
    pitcher_rows = []
    for model_name, group in predictions.groupby("model"):
        pooled_rows.append({
            "model": model_name,
            "n": len(group),
            **_metrics(group["true_class"].to_numpy(int), group["predicted_class"].to_numpy(int)),
        })
    for (pitcher, pitcher_name, model_name), group in predictions.groupby(
        ["pitcher", "pitcher_name", "model"]
    ):
        pitcher_rows.append({
            "pitcher": pitcher,
            "pitcher_name": pitcher_name,
            "model": model_name,
            "n_test": len(group),
            **_metrics(group["true_class"].to_numpy(int), group["predicted_class"].to_numpy(int)),
        })
    pooled = pd.DataFrame(pooled_rows).sort_values("balanced_accuracy", ascending=False)
    per_pitcher = pd.DataFrame(pitcher_rows).sort_values(["pitcher_name", "model"])
    return pooled, per_pitcher


def _legacy_run_for_reproduction(
    statcast_dirs: list[Path],
    stuff_paths: list[Path],
    official_stats_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Historical implementation retained only for result provenance."""
    data, qualified = make_dataset(statcast_dirs, stuff_paths)
    data = _prepare_data(data)
    development = data.loc[data["year"].le(FINAL_TRAIN_END)].copy()
    primary, ridge, validation = select_models(development)

    names = (
        pd.read_parquet(official_stats_path)[["player_id", "name"]]
        .drop_duplicates("player_id")
        .set_index("player_id")["name"]
        .to_dict()
    )
    predictions, ridge_coefficients = _final_test(
        data, qualified, names, primary, ridge
    )
    pooled, per_pitcher = _summaries(predictions)
    primary_name = f"primary:{primary.model_id}"
    ridge_name = f"interpretable:{ridge.model_id}"
    comparison, agreement = _dual_model_comparison(predictions, primary_name, ridge_name)
    ridge_summary = _ridge_feature_summary(ridge_coefficients)

    output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_dir / "validation_candidates.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    comparison.to_parquet(output_dir / "dual_model_comparison.parquet", index=False)
    ridge_coefficients.to_csv(output_dir / "ridge_coefficients.csv", index=False)
    ridge_summary.to_csv(output_dir / "ridge_feature_summary.csv", index=False)
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)
    per_pitcher.to_csv(output_dir / "per_pitcher_metrics.csv", index=False)
    for label, model_name in (("xgboost", primary_name), ("ridge", ridge_name)):
        chosen = predictions.loc[predictions["model"].eq(model_name)]
        matrix = confusion_matrix(
            chosen["true_class"], chosen["predicted_class"], labels=[0, 1, 2]
        )
        pd.DataFrame(
            matrix,
            index=["true_low", "true_middle", "true_high"],
            columns=["predicted_low", "predicted_middle", "predicted_high"],
        ).to_csv(output_dir / f"confusion_matrix_{label}.csv")

    report = {
        "selection_guard": "select_model receives rows through 2024 only; 2025 is evaluated after selection",
        "qualification": "20+ official starts with 50+ Statcast pitches in every 2021-2025 season",
        "qualification_years": [2021, 2022, 2023, 2024, 2025],
        "qualified_pitchers": qualified,
        "development_years": [TRAIN_START, FINAL_TRAIN_END],
        "validation_folds": [
            {"train": [TRAIN_START, year - 1], "validation": year}
            for year in VALIDATION_YEARS
        ],
        "selection_metric": "mean balanced accuracy across 2023 and 2024",
        "adopted_models": {
            "primary_prediction": asdict(primary),
            "interpretable_companion": asdict(ridge),
        },
        "primary_model_id": primary.model_id,
        "ridge_model_id": ridge.model_id,
        "primary_features": [] if primary.family == "ewma" else _feature_names(primary),
        "ridge_features": _feature_names(ridge),
        "dual_model_agreement": agreement,
        "final_train_years": [TRAIN_START, FINAL_TRAIN_END],
        "test_year": TEST_YEAR,
        "target": "player-specific absolute Stuff+ tertile using each fold's training-only thresholds",
        "missing_policy": "complete cases only using the common 28-feature candidate rows",
        "pooled_metrics": pooled.to_dict(orient="records"),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Primary selected without 2025: {primary.model_id}")
    print(f"Ridge selected without 2025: {ridge.model_id}")
    return pooled


def run(
    statcast_dirs: list[Path],
    stuff_paths: list[Path],
    official_stats_path: Path,
    output_dir: Path,
    *,
    locked_config_path: Path | None = None,
    device: str = "auto",
) -> dict[str, object]:
    """Compatibility API for exact locked final replay.

    The historical API silently selected a model and then scored 2025 in one
    call.  A lock is now mandatory so this entrypoint cannot reselect anything.
    """

    if locked_config_path is None:
        raise ValueError(
            "locked_config_path is required; run select_models.py before the "
            "locked 2025 final evaluation"
        )

    from lib.evaluation import load_locked_model_config
    from lib.stuff_cli import load_stuff_data
    from lib.stuff_final import run_locked_final_evaluation

    bundle = load_stuff_data(
        statcast_paths=statcast_dirs,
        stuff_paths=stuff_paths,
        official_stats_paths=[official_stats_path],
        qualification_mode="research",
    )
    return run_locked_final_evaluation(
        bundle,
        locked_config=load_locked_model_config(locked_config_path),
        output_dir=output_dir,
        device=device,
    )


def main(argv: list[str] | None = None) -> int:
    """Forward the legacy command name to ``final_evaluate.py``."""

    import sys

    from final_evaluate import main as unified_main

    return unified_main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
