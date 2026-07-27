# %%
"""Sweep forward rolling window size k for the denoised within-pitcher target.

Same rolling-origin split and evaluation harness as the single-outing baseline,
so lift numbers are directly comparable to experiment_label_debias.py
(A single-outing raw residual = 1.207, D single-outing debiased = 1.109).

For each k in {1,3,5,10} we cross:
  target   : raw rolling mean   vs   debiased (offset-removed)
  features : personalized_workload_max   vs   + prior-skill

k=1 reproduces the single-outing references inside this identical harness.
The raw-vs-debiased gap attributes performance to the pitcher-identity offset;
the +prior-skill gap attributes it to explicit talent features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.evaluate import evaluate_classification
from lib.feature_sets import expand_feature_set
from lib.modeling import make_design_matrices
from lib.rolling_target import add_forward_rolling_target, rolling_columns, slice_train_test
from lib.split import rolling_origin_split
from models.model_classification_residual_tertile_xgboost import (
    _classify_residual,
    _make_sample_weight,
    _make_estimator,
    _align_proba,
    CONFIG as MODEL_CONFIG,
)


FULL_CACHE = Path("data/cache_labeled_full.parquet")
OUT_DIR = Path("experiments/runs/rolling_target_sweep")
PRIOR_SKILL = ["personal_prior_xwOBA", "normal_condition_count_prior"]
KS = [1, 3, 5, 10]
EMBARGO_DAYS = 3


# %%
def run_one(name, train_df, test_df, target_col, feature_set, extra_features):
    cfg = MODEL_CONFIG.copy()
    low_cut = float(train_df[target_col].quantile(cfg["lower_quantile"]))
    high_cut = float(train_df[target_col].quantile(cfg["upper_quantile"]))
    y_train = _classify_residual(train_df[target_col], low_cut, high_cut)
    y_test = _classify_residual(test_df[target_col], low_cut, high_cut)

    feature_columns = expand_feature_set(train_df, feature_set)
    if extra_features:
        feature_columns = list(dict.fromkeys(
            feature_columns + [c for c in extra_features if c in train_df.columns]))

    x_train, x_test, encoded = make_design_matrices(train_df, test_df, feature_columns)
    sample_weight, _ = _make_sample_weight(train_df, y_train, cfg)
    model = _make_estimator(cfg)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(x_train, y_train)
    proba = _align_proba(model.predict_proba(x_test), getattr(model, "classes_", None))
    predictions = np.argmax(proba, axis=1)

    eval_df = test_df.copy()
    eval_df["residual_class"] = y_test
    metrics = evaluate_classification(eval_df, predictions, predicted_proba=proba, target_col="residual_class")
    return {
        "config": name,
        "n_features": len(encoded),
        "n_train": len(train_df),
        "n_test": len(test_df),
        **{key: (round(float(val), 4) if isinstance(val, (int, float, np.floating)) and np.isfinite(val) else val)
           for key, val in metrics.items()},
    }


# %%
def identity_share(df, target_col):
    """Split-half (odd/even) correlation of per-pitcher mean target.

    High -> target still encodes stable pitcher identity; low -> debiased.
    """
    sub = df[["pitcher", target_col]].dropna().copy()
    sub["rank"] = sub.groupby("pitcher").cumcount()
    means = (
        sub.assign(half=sub["rank"] % 2)
        .groupby(["pitcher", "half"])[target_col].mean().unstack()
    )
    means = means.dropna()
    means = means[means.index.map(sub.groupby("pitcher").size()) >= 6]
    if len(means) < 20:
        return None
    return round(float(np.corrcoef(means.iloc[:, 0], means.iloc[:, 1])[0, 1]), 3)


# %%
def main():
    labeled = pd.read_parquet(FULL_CACHE)
    split = rolling_origin_split(labeled)
    test_start = pd.Timestamp(split.test_start)
    embargo_start = test_start - pd.Timedelta(days=EMBARGO_DAYS)
    print(f"test_start={test_start.date()} embargo_start={embargo_start.date()}")

    rows = []
    honesty = {}
    for k in KS:
        cols = rolling_columns(k)
        dk = add_forward_rolling_target(labeled, k=k)
        honesty[k] = {
            "raw_identity_r": identity_share(dk, cols["raw"]),
            "centered_identity_r": identity_share(dk, cols["centered"]),
        }
        for variant, target_col in (("raw", cols["raw"]), ("debiased", cols["centered"])):
            train_df, test_df = slice_train_test(
                dk, target_col, cols["window_end"], test_start, embargo_start)
            # leakage guard
            assert (pd.to_datetime(train_df[cols["window_end"]]) < embargo_start).all()
            for feat_tag, extra in (("base", None), ("+prior_skill", PRIOR_SKILL)):
                name = f"k{k}_{variant}_{feat_tag}"
                res = run_one(name, train_df, test_df, target_col,
                              "personalized_workload_max", extra)
                res.update({"k": k, "variant": variant, "features": feat_tag})
                rows.append(res)
                print(f"  {name:28s} lift={res.get('top20_risk_lift')}  "
                      f"bal_acc={res.get('balanced_accuracy')}  risk_prec={res.get('risk_precision')}")

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "sweep.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "honesty.json").write_text(json.dumps(honesty, indent=2), encoding="utf-8")

    key = ["config", "k", "variant", "features", "n_train", "n_test",
           "balanced_accuracy", "risk_precision", "risk_recall", "top20_risk_lift", "top20_risk_rate"]
    print("\n=== identity split-half r (per-pitcher mean target) ===")
    print(json.dumps(honesty, indent=2))
    print("\n=== rolling target sweep (same rolling-origin split) ===")
    print(table[[c for c in key if c in table.columns]].to_string(index=False))
    print(f"\nArtifacts: {OUT_DIR}")


# %%
if __name__ == "__main__":
    main()
