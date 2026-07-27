# %%
"""3-class classifier on a denoised forward-rolling residual target.

Identical estimator to model_classification_residual_tertile_xgboost, but the
target is the mean residual over a pitcher's current + next (k-1) outings
instead of a single outing. The single-outing residual is ~99% sampling noise
at BF~4; averaging a forward window cancels most of that noise and recovers a
predictable "is this pitcher heading into a good/bad stretch vs. their own
baseline" signal.

The rolling target columns and the train-side window embargo are produced by
lib.rolling_target and applied by run_rolling.py before this model runs; this
module only tertiles the supplied target and fits the classifier, so it stays a
drop-in sibling of the single-outing baseline.

Default config = the shipped result: k=5, debiased (offset-removed) target,
decision_pregame_shrunk_xwoba features (workload + prior-skill). See
docs/rolling_target_results.md for the k sweep and the identity/form
decomposition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.evaluate import evaluate_classification
from lib.feature_sets import expand_feature_set
from lib.modeling import get_git_commit, make_design_matrices
from models.model_classification_residual_tertile_xgboost import (
    _align_proba,
    _classify_residual,
    _make_estimator,
    _make_sample_weight,
    CONFIG as SINGLE_OUTING_CONFIG,
)


CONFIG = {
    **SINGLE_OUTING_CONFIG,
    "feature_set": "decision_pregame_shrunk_xwoba",
    "target_source": "rolling_fwd5_residual_centered",
    "rolling_k": 5,
    "rolling_variant": "centered",
}


# %%
def run(config: dict, train_df, test_df) -> dict:
    cfg = CONFIG.copy()
    cfg.update(config or {})
    target = str(cfg["target_source"])
    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(
            f"classification_rolling_residual_tertile_xgboost requires target column '{target}'. "
            "Build it with lib.rolling_target.add_forward_rolling_target (see run_rolling.py)."
        )

    low_cut = float(train_df[target].quantile(float(cfg["lower_quantile"])))
    high_cut = float(train_df[target].quantile(float(cfg["upper_quantile"])))
    y_train = _classify_residual(train_df[target], low_cut, high_cut)
    y_test = _classify_residual(test_df[target], low_cut, high_cut)

    feature_columns = expand_feature_set(train_df, cfg["feature_set"])
    if not feature_columns:
        raise ValueError("classification_rolling_residual_tertile_xgboost has no available features.")

    x_train, x_test, encoded_features = make_design_matrices(train_df, test_df, feature_columns)
    sample_weight, class_weight_multipliers = _make_sample_weight(train_df, y_train, cfg)

    model = _make_estimator(cfg)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(x_train, y_train)

    proba = _align_proba(model.predict_proba(x_test), getattr(model, "classes_", None))
    predictions = np.argmax(proba, axis=1)

    eval_df = test_df.copy()
    eval_df["residual_class"] = y_test
    metrics = evaluate_classification(
        eval_df,
        predictions,
        predicted_proba=proba,
        target_col="residual_class",
    )
    cfg["feature_columns"] = feature_columns
    cfg["encoded_feature_count"] = len(encoded_features)
    cfg["class_labels"] = {
        "0": f"risk: {target} <= train lower quantile",
        "1": "normal: between train quantiles",
        "2": f"good: {target} >= train upper quantile",
    }
    cfg["residual_class_thresholds"] = {"low_cut": low_cut, "high_cut": high_cut}
    cfg["class_weight_multipliers"] = class_weight_multipliers
    cfg["train_class_counts"] = {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()}
    cfg["test_class_counts"] = {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()}

    return {
        "model_name": "classification_rolling_residual_tertile_xgboost",
        "task": "classification",
        "target_col": "residual_class",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "predicted_proba": proba,
        "model_object": model,
        "git_commit": get_git_commit(),
    }
