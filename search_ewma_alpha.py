from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.stuff_mlb_dataset import FEATURES, make_dataset
from stuff_mlb_temporal_final import _classes, _metrics


TRAIN_START = 2020
VALIDATION_YEARS = (2022, 2023, 2024)
TEST_YEAR = 2025
DEFAULT_ALPHAS = np.round(np.arange(0.05, 0.951, 0.01), 2)
CURRENT_ALPHA = 0.40  # pandas ewm(span=4) -> alpha=2/(4+1)


def add_ewma(data: pd.DataFrame, alpha: float) -> pd.DataFrame:
    result = data.sort_values(["pitcher", "game_date"]).copy()
    result["ewma_prior"] = result.groupby("pitcher")["target_y"].transform(
        lambda values: values.shift(1).ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    )
    return result


def complete_rows(data: pd.DataFrame, alpha: float) -> pd.DataFrame:
    prepared = add_ewma(data, alpha)
    return prepared.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*FEATURES, "target_y", "ewma_prior"]
    )


def evaluate_year(data: pd.DataFrame, alpha: float, year: int) -> tuple[dict[str, float], pd.DataFrame]:
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    frames = []
    prepared = complete_rows(data, alpha)
    for pitcher, player in prepared.groupby("pitcher", sort=False):
        train = player.loc[player["year"].between(TRAIN_START, year - 1)]
        test = player.loc[player["year"].eq(year)]
        if len(train) < 20 or test.empty:
            continue
        q33, q67 = train["target_y"].quantile([1 / 3, 2 / 3]).to_numpy(float)
        actual = _classes(test["target_y"], q33, q67)
        score = test["ewma_prior"].to_numpy(float)
        predicted = _classes(score, q33, q67)
        actual_parts.append(actual)
        predicted_parts.append(predicted)
        frame = test[["pitcher", "game_date", "target_y"]].copy()
        frame["alpha"] = alpha
        frame["q33"] = q33
        frame["q67"] = q67
        frame["true_class"] = actual
        frame["predicted_score"] = score
        frame["predicted_class"] = predicted
        frames.append(frame)
    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(predicted_parts)
    predictions = pd.concat(frames, ignore_index=True)
    metrics = _metrics(actual, predicted)
    error = predictions["target_y"] - predictions["predicted_score"]
    metrics["continuous_mae"] = float(error.abs().mean())
    metrics["continuous_rmse"] = float(np.sqrt(np.mean(np.square(error))))
    metrics["n"] = len(predictions)
    return metrics, predictions


def validation_row(data: pd.DataFrame, alpha: float) -> dict[str, float]:
    row: dict[str, float] = {"alpha": alpha, "equivalent_span": 2 / alpha - 1}
    yearly = []
    for year in VALIDATION_YEARS:
        metrics, _ = evaluate_year(data, alpha, year)
        yearly.append(metrics)
        for name, value in metrics.items():
            row[f"{name}_{year}"] = value
    for metric in (
        "accuracy", "balanced_accuracy", "macro_f1", "ordinal_mae",
        "continuous_mae", "continuous_rmse",
    ):
        values = np.array([result[metric] for result in yearly], dtype=float)
        row[f"mean_{metric}"] = float(values.mean())
        row[f"std_{metric}"] = float(values.std(ddof=0))
    row["selection_score"] = (
        row["mean_balanced_accuracy"] - 0.5 * row["std_balanced_accuracy"]
    )
    return row


def search(data: pd.DataFrame, alphas: np.ndarray = DEFAULT_ALPHAS) -> pd.DataFrame:
    results = pd.DataFrame([validation_row(data, float(alpha)) for alpha in alphas])
    return results.sort_values(
        ["selection_score", "mean_ordinal_mae", "mean_continuous_mae"],
        ascending=[False, True, True],
    ).reset_index(drop=True).assign(rank=lambda frame: np.arange(1, len(frame) + 1))


def per_pitcher_metrics(predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for pitcher, group in predictions.groupby("pitcher", sort=False):
        rows.append({
            "pitcher": int(pitcher),
            "model": label,
            "n": len(group),
            **_metrics(group["true_class"].to_numpy(), group["predicted_class"].to_numpy()),
        })
    return pd.DataFrame(rows)


def run(statcast_dirs: list[Path], stuff_paths: list[Path], output_dir: Path) -> dict[str, object]:
    data, qualified = make_dataset(statcast_dirs, stuff_paths)
    results = search(data)
    winner = results.iloc[0]
    chosen_alpha = float(winner["alpha"])

    chosen_metrics, chosen_predictions = evaluate_year(data, chosen_alpha, TEST_YEAR)
    current_metrics, current_predictions = evaluate_year(data, CURRENT_ALPHA, TEST_YEAR)
    keys = ["pitcher", "game_date", "target_y", "true_class"]
    comparison = chosen_predictions[keys + ["predicted_score", "predicted_class"]].rename(
        columns={"predicted_score": "optimized_score", "predicted_class": "optimized_class"}
    ).merge(
        current_predictions[keys + ["predicted_score", "predicted_class"]].rename(
            columns={"predicted_score": "ewma4_score", "predicted_class": "ewma4_class"}
        ), on=keys, how="inner", validate="one_to_one"
    )
    comparison["classes_differ"] = comparison["optimized_class"].ne(comparison["ewma4_class"])

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "alpha_validation_grid.csv", index=False)
    chosen_predictions.to_parquet(output_dir / "optimized_alpha_2025_predictions.parquet", index=False)
    comparison.to_parquet(output_dir / "optimized_alpha_vs_ewma4_2025.parquet", index=False)
    pd.concat([
        per_pitcher_metrics(chosen_predictions, f"optimized_alpha_{chosen_alpha:.2f}"),
        per_pitcher_metrics(current_predictions, "current_alpha_0.40"),
    ], ignore_index=True).to_csv(output_dir / "per_pitcher_comparison.csv", index=False)

    top = results.head(10)[
        ["rank", "alpha", "equivalent_span", "selection_score", "mean_balanced_accuracy",
         "std_balanced_accuracy", "mean_macro_f1", "mean_ordinal_mae", "mean_continuous_mae"]
    ]
    result: dict[str, object] = {
        "selection_guard": "alpha selected with 2022-2024 forward validation only",
        "qualified_pitchers": qualified,
        "grid": {"minimum": 0.05, "maximum": 0.95, "step": 0.01, "candidate_count": len(results)},
        "selected_alpha": chosen_alpha,
        "selected_equivalent_span": float(winner["equivalent_span"]),
        "current_alpha": CURRENT_ALPHA,
        "top_validation_candidates": top.to_dict(orient="records"),
        "test_2025_after_selection": {
            "optimized": chosen_metrics,
            "current_ewma4": current_metrics,
            "prediction_class_changes": int(comparison["classes_differ"].sum()),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune one global EWMA alpha without using 2025.")
    parser.add_argument("--statcast-dir", required=True, nargs="+", type=Path)
    parser.add_argument("--stuff", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.statcast_dir, args.stuff, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
