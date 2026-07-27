# %%
"""Production entrypoint for the forward-rolling residual 3-class model.

Parallel to compare.py, but builds the denoised forward-rolling target and
applies the train-side window embargo (both split-aware, so they live here
rather than in the generic model.run). Reuses compare.py's artifact/logging
helpers and the same rolling-origin split as the single-outing baseline, so the
logged metrics are directly comparable.

    python run_rolling.py --k 5                 # shipped default (lift ~1.56)
    python run_rolling.py --k 10 --variant raw  # max lift (~1.82), talent-heavy

See docs/rolling_target_results.md for the k sweep and the identity/form
decomposition behind these defaults.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from compare import (
    _json_ready,
    filter_pitcher_sample,
    save_dataframe_artifacts,
    save_prediction_artifact,
)
from lib.data_prep import prepare_features
from lib.evaluate import log_experiment
from lib.labeling import create_labels
from lib.rolling_target import add_forward_rolling_target, rolling_columns, slice_train_test
from lib.split import rolling_origin_split
from models.model_classification_rolling_residual_tertile_xgboost import run as run_rolling_model


# %%
def load_labeled(input_path: Path, cache_path: Path | None, min_bf: int, min_ip: float) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        print(f"[cache] {cache_path}")
        return pd.read_parquet(cache_path)
    raw = pd.read_parquet(input_path)
    raw = filter_pitcher_sample(raw, min_bf=min_bf, min_ip=min_ip)
    labeled = create_labels(prepare_features(raw))
    labeled = labeled.loc[pd.to_numeric(labeled["target_y"], errors="coerce").notna()].copy()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        labeled.to_parquet(cache_path, index=False)
        print(f"[cache] wrote {cache_path}")
    return labeled


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/outings_mlb_bullpen_2021_2025.parquet"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache_labeled_full.parquet"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--k", type=int, default=5, help="Forward window size (outings).")
    parser.add_argument("--variant", choices=["centered", "raw"], default="centered",
                        help="centered = offset-removed within-pitcher; raw = includes pitcher offset.")
    parser.add_argument("--feature-set", default="decision_pregame_shrunk_xwoba")
    parser.add_argument("--min-pitcher-bf", type=int, default=100)
    parser.add_argument("--min-pitcher-ip", type=float, default=30)
    parser.add_argument("--embargo-days", type=int, default=3)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    k = int(args.k)
    cols = rolling_columns(k)
    target_col = cols["centered"] if args.variant == "centered" else cols["raw"]
    run_id = args.run_id or f"rolling_fwd{k}_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path("experiments") / "runs" / run_id

    cache = None if args.no_cache else args.cache
    labeled = load_labeled(args.input, cache, args.min_pitcher_bf, args.min_pitcher_ip)
    labeled = add_forward_rolling_target(labeled, k=k)

    split = rolling_origin_split(labeled)
    test_start = pd.Timestamp(split.test_start)
    embargo_start = test_start - pd.Timedelta(days=int(args.embargo_days))
    train_df, test_df = slice_train_test(labeled, target_col, cols["window_end"], test_start, embargo_start)

    # Leakage guard: no training label may borrow an outing on/after the embargo boundary.
    window_end = pd.to_datetime(train_df[cols["window_end"]], errors="coerce")
    assert (window_end < embargo_start).all(), "window embargo failed: train target crosses boundary"
    print(f"k={k} variant={args.variant} target={target_col}")
    print(f"train={len(train_df)} test={len(test_df)} test_start={test_start.date()} "
          f"max_train_window_end={window_end.max().date()}")

    result = run_rolling_model({"target_source": target_col, "feature_set": args.feature_set, "rolling_k": k,
                                "rolling_variant": args.variant}, train_df, test_df)
    log_experiment(result, run_id, len(train_df), len(test_df))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_artifact(test_df, result, output_dir)
    metrics = pd.DataFrame([{
        "model_name": result["model_name"], "k": k, "variant": args.variant,
        "feature_set": args.feature_set, **result["metrics"], "git_commit": result.get("git_commit", ""),
    }])
    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    report = {
        "run_id": run_id, "k": k, "variant": args.variant, "feature_set": args.feature_set,
        "target_col": target_col, "n_train": len(train_df), "n_test": len(test_df),
        "test_start": _json_ready(test_start), "embargo_start": _json_ready(embargo_start),
        "residual_class_thresholds": result["config"]["residual_class_thresholds"],
        "caveat": (
            "Denoised forward-rolling target. Lift is dominated by pitcher talent/form, "
            "not workload; see docs/rolling_target_results.md decomposition. "
            "Correlation, not causation."
        ),
        "metrics": result["metrics"],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Artifacts: {output_dir}")


# %%
if __name__ == "__main__":
    main()
