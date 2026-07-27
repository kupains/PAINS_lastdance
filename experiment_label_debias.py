# %%
"""Decompose "predictive performance" of the 3-class outing model.

The current label residual = baseline_skill - shrunk_xwOBA carries a large,
highly predictable per-pitcher offset (split-half r = 0.92). This script
quantifies how much headline performance is that offset (pitcher identity)
versus genuine within-pitcher "how did today's outing go" signal, and whether
debiasing the label recovers an honest, decision-relevant target.

All configs run on the same rolling-origin split for apples-to-apples metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.data_prep import prepare_features
from lib.labeling import create_labels
from lib.evaluate import evaluate_classification
from lib.feature_sets import expand_feature_set
from lib.modeling import make_design_matrices
from lib.split import rolling_origin_split
from compare import filter_pitcher_sample
from models.model_classification_residual_tertile_xgboost import (
    _classify_residual,
    _make_sample_weight,
    _make_estimator,
    _align_proba,
    CONFIG as MODEL_CONFIG,
)


FULL_CACHE = Path("data/cache_labeled_full.parquet")
INPUT = Path("data/outings_mlb_bullpen_2021_2025.parquet")
OUT_DIR = Path("experiments/runs/label_debias_decomposition")


# %%
def build_full_labeled() -> pd.DataFrame:
    if FULL_CACHE.exists():
        print(f"[cache] {FULL_CACHE}")
        return pd.read_parquet(FULL_CACHE)
    print("[build] full features + labels (this takes a few minutes)...")
    raw = pd.read_parquet(INPUT)
    raw = filter_pitcher_sample(raw, min_bf=100, min_ip=30)
    labeled = create_labels(prepare_features(raw))
    labeled = labeled.loc[pd.to_numeric(labeled["target_y"], errors="coerce").notna()].copy()
    FULL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(FULL_CACHE, index=False)
    print(f"[cache] wrote {FULL_CACHE}  rows={len(labeled)}")
    return labeled


# %%
def add_debiased_residual(df: pd.DataFrame, min_periods: int = 5) -> pd.DataFrame:
    """residual_centered = residual - prior-only expanding per-pitcher mean(residual).

    Leakage-safe: only outings strictly before the current one feed the offset
    estimate. Early outings (< min_periods of history) fall back to no
    adjustment (offset 0), so they keep the raw residual.
    """
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    offset = pd.Series(0.0, index=out.index)
    for _, group in out.groupby("pitcher", sort=False):
        r = pd.to_numeric(group["residual"], errors="coerce")
        prior_mean = r.shift(1).expanding(min_periods=min_periods).mean()
        offset.loc[group.index] = prior_mean.fillna(0.0).to_numpy()
    out["pitcher_offset_prior"] = offset
    out["residual_centered"] = pd.to_numeric(out["residual"], errors="coerce") - offset
    return out


# %%
def run_config(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_source: str,
    feature_set: str,
    extra_features: list[str] | None = None,
    allow_result_features: bool = False,
) -> dict:
    cfg = MODEL_CONFIG.copy()

    low_cut = float(train_df[target_source].quantile(cfg["lower_quantile"]))
    high_cut = float(train_df[target_source].quantile(cfg["upper_quantile"]))
    y_train = _classify_residual(train_df[target_source], low_cut, high_cut)
    y_test = _classify_residual(test_df[target_source], low_cut, high_cut)

    feature_columns = expand_feature_set(train_df, feature_set)
    if extra_features:
        feature_columns = feature_columns + [c for c in extra_features if c in train_df.columns]
        feature_columns = list(dict.fromkeys(feature_columns))
    # (result-feature guard intentionally skipped for labeled ceiling probes)

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
        "target": target_source,
        "feature_set": feature_set,
        "extra": extra_features or [],
        "n_features": len(encoded),
        "cuts": [round(low_cut, 5), round(high_cut, 5)],
        **{k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v)
           for k, v in metrics.items()},
    }


# %%
def offset_diagnostics(df, train_df, test_df) -> dict:
    """How predictable is the label from the pitcher offset alone?"""
    # rank test outings by the prior-only pitcher offset (pure identity signal)
    off = pd.to_numeric(test_df["pitcher_offset_prior"], errors="coerce").to_numpy()
    low_cut = float(train_df["residual"].quantile(1 / 3))
    y = _classify_residual(test_df["residual"], low_cut, float(train_df["residual"].quantile(2 / 3)))
    risk = (y == 0).astype(int)
    # low residual = risk, so low offset should rank risk high -> sort ascending
    order = np.argsort(off)
    k = max(1, int(np.ceil(len(y) * 0.20)))
    top_risk_rate = float(risk[order[:k]].mean())
    base = float(risk.mean())
    return {
        "identity_only_top20_risk_lift": round(top_risk_rate / base, 4) if base > 0 else None,
        "offset_share_of_residual_var_train": round(
            float(1 - train_df["residual_centered"].var() / train_df["residual"].var()), 4
        ),
    }


# %%
def main() -> None:
    labeled = build_full_labeled()
    labeled = add_debiased_residual(labeled)

    split = rolling_origin_split(labeled)
    train_df, test_df = split.train_df, split.test_df
    print(f"train={len(train_df)} test={len(test_df)}")

    configs = []
    # 1. Reproduce current baseline
    configs.append(run_config(
        "A_baseline_current_label", train_df, test_df,
        target_source="residual", feature_set="personalized_workload_max"))
    # 2. Offset ceiling via legitimate pregame prior-skill features
    configs.append(run_config(
        "B_add_prior_skill", train_df, test_df,
        target_source="residual", feature_set="personalized_workload_max",
        extra_features=["personal_prior_xwOBA", "normal_condition_count_prior"]))
    # 3. Absolute offset ceiling: hand the model the label's own prior-only anchor
    configs.append(run_config(
        "C_ceiling_baseline_skill", train_df, test_df,
        target_source="residual", feature_set="personalized_workload_max",
        extra_features=["baseline_skill", "personal_prior_xwOBA"],
        allow_result_features=True))
    # 4. Debiased label (honest within-pitcher target)
    configs.append(run_config(
        "D_debiased_label", train_df, test_df,
        target_source="residual_centered", feature_set="personalized_workload_max"))
    # 5. Does identity still help after debiasing? (should not, if debias worked)
    configs.append(run_config(
        "E_debiased_plus_prior_skill", train_df, test_df,
        target_source="residual_centered", feature_set="personalized_workload_max",
        extra_features=["personal_prior_xwOBA", "normal_condition_count_prior", "baseline_skill"],
        allow_result_features=True))

    diag = offset_diagnostics(labeled, train_df, test_df)

    table = pd.DataFrame(configs)
    key_cols = ["config", "target", "n_features", "balanced_accuracy", "macro_f1",
                "risk_precision", "risk_recall", "top20_risk_lift", "top20_risk_rate", "log_loss"]
    key_cols = [c for c in key_cols if c in table.columns]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print("\n=== offset diagnostics ===")
    print(json.dumps(diag, indent=2))
    print("\n=== config comparison (same rolling-origin split) ===")
    print(table[key_cols].to_string(index=False))
    print(f"\nArtifacts: {OUT_DIR}")


# %%
if __name__ == "__main__":
    main()
