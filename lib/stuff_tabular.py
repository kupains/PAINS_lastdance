"""Pitcher-specific tabular residual models used by the unified workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _numeric_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing tabular feature columns: {missing}")
    return (
        frame.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64)
    )


@dataclass
class NumericTransform:
    """Training-only numeric transform.

    ``complete_case`` is the legacy Ridge/XGBoost policy.  ``train_median`` is
    available for explicit locked ablations, but is never fitted on validation
    or test values.
    """

    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    missing_policy: str = "complete_case"

    @classmethod
    def fit(
        cls, values: np.ndarray, *, missing_policy: str = "complete_case"
    ) -> "NumericTransform":
        if missing_policy not in {"complete_case", "train_median"}:
            raise ValueError(
                "missing_policy must be 'complete_case' or 'train_median'"
            )
        values = np.asarray(values, dtype=np.float64)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(values), values, medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales = np.where(
            np.isfinite(scales) & (scales > 1e-8), scales, 1.0
        )
        return cls(
            medians=medians,
            means=means,
            scales=scales,
            missing_policy=missing_policy,
        )

    def transform(
        self, values: np.ndarray, *, standardize: bool = True
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self.missing_policy == "complete_case" and not np.isfinite(
            values
        ).all():
            raise ValueError("complete-case transform received missing values")
        filled = np.where(np.isfinite(values), values, self.medians)
        if not standardize:
            return filled
        return (filled - self.means) / self.scales


@dataclass
class PitcherRegressor:
    estimator: Any
    transform: NumericTransform


class PerPitcherResidualRegressor:
    """Fit one residual model per pitcher with fold-training transforms only."""

    def __init__(
        self,
        *,
        model: str,
        features: Iterable[str],
        alpha: float = 100.0,
        random_state: int = 20260722,
        xgboost_params: dict[str, Any] | None = None,
        missing_policy: str = "complete_case",
    ) -> None:
        if model not in {"ridge", "xgboost"}:
            raise ValueError(f"Unsupported residual model: {model}")
        if missing_policy not in {"complete_case", "train_median"}:
            raise ValueError(f"Unsupported missing policy: {missing_policy}")
        self.model = model
        self.features = list(features)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.xgboost_params = dict(xgboost_params or {})
        self.missing_policy = missing_policy
        self.models_: dict[str, PitcherRegressor] = {}
        self.backend_: str | None = None

    def _new_estimator(self):
        if self.model == "ridge":
            self.backend_ = "sklearn.Ridge"
            return Ridge(alpha=self.alpha)
        try:
            from xgboost import XGBRegressor

            defaults: dict[str, Any] = {
                "n_estimators": 100,
                "max_depth": 2,
                "learning_rate": 0.04221,
                "min_child_weight": 15.043,
                "reg_lambda": 16.823,
                "reg_alpha": 0.706,
                "gamma": 0.0,
                "subsample": 0.9997,
                "colsample_bytree": 0.5856,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "max_bin": 64,
                "n_jobs": 1,
                "random_state": self.random_state,
                "verbosity": 0,
            }
            defaults.update(self.xgboost_params)
            self.backend_ = "xgboost.XGBRegressor"
            return XGBRegressor(**defaults)
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is required for the locked legacy model; install "
                "requirements.txt instead of silently changing the backend."
            ) from exc

    def fit(
        self,
        train: pd.DataFrame,
        *,
        pitcher_col: str = "pitcher",
        target_col: str = "stuff_plus",
        ewma_col: str = "ewma4",
    ) -> "PerPitcherResidualRegressor":
        required = {pitcher_col, target_col, ewma_col, *self.features}
        missing = required - set(train.columns)
        if missing:
            raise ValueError(
                f"Cannot fit {self.model}; missing columns: {sorted(missing)}"
            )
        eligible = train.get(
            "target_eligible", pd.Series(True, index=train.index)
        ).fillna(False).astype(bool)
        target = pd.to_numeric(train[target_col], errors="coerce")
        ewma = pd.to_numeric(train[ewma_col], errors="coerce")
        fit_rows = train.loc[eligible & target.notna() & ewma.notna()].copy()

        self.models_.clear()
        for pitcher, group in fit_rows.groupby(pitcher_col, sort=True):
            x_raw = _numeric_matrix(group, self.features)
            if self.missing_policy == "complete_case":
                valid = np.isfinite(x_raw).all(axis=1)
                group = group.iloc[np.flatnonzero(valid)]
                x_raw = x_raw[valid]
            if len(group) < 3:
                continue
            transform = NumericTransform.fit(
                x_raw, missing_policy=self.missing_policy
            )
            x = transform.transform(
                x_raw, standardize=self.model == "ridge"
            )
            y = (
                pd.to_numeric(group[target_col], errors="coerce")
                - pd.to_numeric(group[ewma_col], errors="coerce")
            ).to_numpy(dtype=np.float64)
            estimator = self._new_estimator()
            estimator.fit(x, y)
            self.models_[str(pitcher)] = PitcherRegressor(
                estimator, transform
            )
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        pitcher_col: str = "pitcher",
    ) -> np.ndarray:
        predictions = np.full(len(frame), np.nan, dtype=np.float64)
        for pitcher, positions in frame.groupby(
            pitcher_col, sort=False
        ).indices.items():
            bundle = self.models_.get(str(pitcher))
            if bundle is None:
                continue
            positional = np.asarray(positions, dtype=int)
            x_raw = _numeric_matrix(frame.iloc[positional], self.features)
            if self.missing_policy == "complete_case":
                valid = np.isfinite(x_raw).all(axis=1)
            else:
                valid = np.ones(len(x_raw), dtype=bool)
            if not valid.any():
                continue
            x = bundle.transform.transform(
                x_raw[valid], standardize=self.model == "ridge"
            )
            predictions[positional[valid]] = np.asarray(
                bundle.estimator.predict(x), dtype=np.float64
            )
        return predictions

    @property
    def parameter_count(self) -> int:
        total = 0
        for bundle in self.models_.values():
            estimator = bundle.estimator
            if hasattr(estimator, "coef_"):
                total += int(
                    np.asarray(estimator.coef_).size
                    + np.asarray(estimator.intercept_).size
                )
            elif hasattr(estimator, "get_booster"):
                booster = estimator.get_booster()
                try:
                    total += int(booster.num_boosted_rounds())
                except Exception:
                    pass
        return total


def classify_stuff_plus(
    predicted_stuff: Iterable[float],
    pitchers: Iterable[object],
    thresholds: pd.DataFrame | dict[object, tuple[float, float]],
) -> np.ndarray:
    """Apply the legacy threshold convention: Low <= q33, High > q67."""

    values = np.asarray(list(predicted_stuff), dtype=np.float64)
    pitcher_values = np.asarray(list(pitchers), dtype=object)
    result = np.full(len(values), np.nan)
    if isinstance(thresholds, pd.DataFrame):
        threshold_map = {
            str(row.pitcher): (float(row.q33), float(row.q67))
            for row in thresholds.loc[
                :, ["pitcher", "q33", "q67"]
            ].itertuples(index=False)
        }
    else:
        threshold_map = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in thresholds.items()
        }
    for index, (value, pitcher) in enumerate(
        zip(values, pitcher_values, strict=True)
    ):
        cuts = threshold_map.get(str(pitcher))
        if cuts is None or not np.isfinite(value):
            continue
        low, high = cuts
        result[index] = 0 if value <= low else (2 if value > high else 1)
    return result


def classify_stuff_plus_rows(
    predicted_stuff: Iterable[float],
    q33: Iterable[float],
    q67: Iterable[float],
) -> np.ndarray:
    """Classify with the threshold pair carried by each prediction row.

    This is the safe form for tables containing multiple temporal folds: the
    same pitcher can legitimately have different training-only tertiles in
    different folds.  The tie convention remains Low <= q33 and High > q67,
    so an observation exactly equal to q67 is Middle.
    """

    values = np.asarray(list(predicted_stuff), dtype=np.float64)
    low = np.asarray(list(q33), dtype=np.float64)
    high = np.asarray(list(q67), dtype=np.float64)
    if not (len(values) == len(low) == len(high)):
        raise ValueError(
            "predicted_stuff, q33, and q67 must have the same length"
        )

    valid = np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
    result = np.full(len(values), np.nan, dtype=np.float64)
    result[valid] = 1.0
    result[valid & (values <= low)] = 0.0
    result[valid & (values > high)] = 2.0
    return result


def regression_prediction_frame(
    validation: pd.DataFrame,
    *,
    model_name: str,
    predicted_residual: Iterable[float],
    correction_scale: float,
    thresholds: pd.DataFrame,
    ewma_col: str = "ewma4",
) -> pd.DataFrame:
    """Build the common long-form prediction contract for residual models."""

    result = validation.copy()
    residual = np.asarray(list(predicted_residual), dtype=np.float64)
    result["model"] = model_name
    result["predicted_residual"] = residual
    result["predicted_stuff_plus"] = (
        pd.to_numeric(result[ewma_col], errors="coerce").to_numpy(
            dtype=np.float64
        )
        + float(correction_scale) * residual
    )
    result["predicted_class"] = classify_stuff_plus(
        result["predicted_stuff_plus"],
        result["pitcher"],
        thresholds,
    )
    result["true_stuff_plus"] = pd.to_numeric(
        result["stuff_plus"], errors="coerce"
    )
    result["true_residual"] = result["true_stuff_plus"] - pd.to_numeric(
        result[ewma_col], errors="coerce"
    )
    return result
