"""Training-only, per-pitcher preprocessing for tabular and sequence models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _fit_metadata(
    frame: pd.DataFrame,
    *,
    year_col: str,
    game_date_col: str,
) -> dict[str, Any]:
    if year_col in frame.columns:
        years = pd.to_numeric(frame[year_col], errors="coerce").dropna().astype(int)
    elif game_date_col in frame.columns:
        years = pd.to_datetime(frame[game_date_col], errors="coerce").dt.year.dropna().astype(int)
    else:
        years = pd.Series(dtype=int)
    unique_years = tuple(sorted(int(year) for year in years.unique()))
    if game_date_col in frame.columns:
        dates = pd.to_datetime(frame[game_date_col], errors="coerce").dropna()
        date_min = dates.min().isoformat() if not dates.empty else None
        date_max = dates.max().isoformat() if not dates.empty else None
    else:
        date_min = None
        date_max = None
    return {
        "fit_years": unique_years,
        "fit_year_min": min(unique_years) if unique_years else None,
        "fit_year_max": max(unique_years) if unique_years else None,
        "fit_date_min": date_min,
        "fit_date_max": date_max,
        "fit_n_rows": int(len(frame)),
    }


def _resolve_features(
    configured: Sequence[str] | None,
    supplied: Iterable[str] | None,
) -> tuple[str, ...]:
    features = tuple(supplied) if supplied is not None else tuple(configured or ())
    if not features:
        raise ValueError("At least one feature column is required")
    if len(set(features)) != len(features):
        raise ValueError("Feature columns must be unique")
    return features


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}; missing columns: {missing}")


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")


class PerPitcherStandardScaler:
    """Standardize with pitcher training statistics and pooled fallbacks.

    ``fit`` only derives statistics from the dataframe passed to it. Transforming
    later validation/test frames does not update those statistics. Columns listed
    in ``passthrough_columns`` are returned unchanged.
    """

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        *,
        pitcher_col: str = "pitcher",
        passthrough_columns: Sequence[str] = (),
        year_col: str = "year",
        game_date_col: str = "game_date",
        epsilon: float = 1e-8,
    ) -> None:
        self.feature_columns = tuple(feature_columns) if feature_columns is not None else None
        self.pitcher_col = pitcher_col
        self.passthrough_columns = tuple(passthrough_columns)
        self.year_col = year_col
        self.game_date_col = game_date_col
        self.epsilon = float(epsilon)
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._is_fitted = False

    def fit(
        self,
        training: pd.DataFrame,
        feature_columns: Iterable[str] | None = None,
    ) -> "PerPitcherStandardScaler":
        features = _resolve_features(self.feature_columns, feature_columns)
        _require_columns(
            training,
            [self.pitcher_col, *features],
            "Cannot fit per-pitcher scaler",
        )
        passthrough = tuple(column for column in self.passthrough_columns if column in features)
        scaled = tuple(column for column in features if column not in passthrough)
        numeric = _numeric_frame(training, scaled) if scaled else pd.DataFrame(index=training.index)

        pooled_mean = numeric.mean(axis=0)
        pooled_scale = numeric.std(axis=0, ddof=0)
        pooled_mean = pooled_mean.fillna(0.0)
        pooled_scale = pooled_scale.where(pooled_scale.gt(self.epsilon), 1.0).fillna(1.0)

        if scaled:
            grouped = numeric.assign(_pitcher=training[self.pitcher_col].to_numpy()).groupby(
                "_pitcher",
                sort=True,
                dropna=False,
            )
            pitcher_mean = grouped[list(scaled)].mean()
            pitcher_scale = grouped[list(scaled)].std(ddof=0)
            pitcher_count = grouped[list(scaled)].count()
            pitcher_mean = pitcher_mean.fillna(pooled_mean)
            for column in scaled:
                invalid = pitcher_count[column].lt(2) | pitcher_scale[column].le(self.epsilon)
                pitcher_scale.loc[invalid, column] = pooled_scale[column]
            pitcher_scale = pitcher_scale.fillna(pooled_scale)
        else:
            unique_pitchers = pd.Index(training[self.pitcher_col].drop_duplicates())
            pitcher_mean = pd.DataFrame(index=unique_pitchers)
            pitcher_scale = pd.DataFrame(index=unique_pitchers)
            pitcher_count = pd.DataFrame(index=unique_pitchers)

        self.feature_columns_ = features
        self.scaled_columns_ = scaled
        self.passthrough_columns_ = passthrough
        self.pooled_mean_ = pooled_mean
        self.pooled_scale_ = pooled_scale
        self.pitcher_mean_ = pitcher_mean
        self.pitcher_scale_ = pitcher_scale
        self.pitcher_count_ = pitcher_count
        self.metadata_ = _fit_metadata(
            training,
            year_col=self.year_col,
            game_date_col=self.game_date_col,
        )
        self.fit_years_ = self.metadata_["fit_years"]
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        _require_columns(
            frame,
            [self.pitcher_col, *self.feature_columns_],
            "Cannot transform with per-pitcher scaler",
        )
        result = frame.copy()
        numeric = _numeric_frame(frame, self.scaled_columns_)
        pitcher_values = frame[self.pitcher_col]
        for column in self.scaled_columns_:
            mean_map = self.pitcher_mean_[column]
            scale_map = self.pitcher_scale_[column]
            means = pitcher_values.map(mean_map).fillna(self.pooled_mean_[column])
            scales = pitcher_values.map(scale_map).fillna(self.pooled_scale_[column])
            scales = scales.where(scales.gt(self.epsilon), self.pooled_scale_[column])
            result[column] = (numeric[column] - means) / scales
        return result

    def transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transform(frame)
        return transformed.loc[:, self.feature_columns_].copy()

    def fit_transform(
        self,
        training: pd.DataFrame,
        feature_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit(training, feature_columns).transform(training)

    def get_fit_metadata(self) -> dict[str, Any]:
        self._check_fitted()
        return dict(self.metadata_)


class PerPitcherTrainImputerScaler:
    """Impute and scale TCN features with training-only pitcher statistics.

    Missing values use the fitted pitcher's median, followed by the pooled
    training median. Numeric features are then standardized by fitted pitcher
    mean/std with pooled training fallback. Binary/mask features are imputed but
    deliberately not standardized.
    """

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        *,
        binary_columns: Sequence[str] = (),
        pitcher_col: str = "pitcher",
        year_col: str = "year",
        game_date_col: str = "game_date",
        epsilon: float = 1e-8,
    ) -> None:
        self.feature_columns = tuple(feature_columns) if feature_columns is not None else None
        self.binary_columns = tuple(binary_columns)
        self.pitcher_col = pitcher_col
        self.year_col = year_col
        self.game_date_col = game_date_col
        self.epsilon = float(epsilon)
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._is_fitted = False

    def fit(
        self,
        training: pd.DataFrame,
        feature_columns: Iterable[str] | None = None,
    ) -> "PerPitcherTrainImputerScaler":
        features = _resolve_features(self.feature_columns, feature_columns)
        _require_columns(
            training,
            [self.pitcher_col, *features],
            "Cannot fit sequence preprocessor",
        )
        unknown_binary = sorted(set(self.binary_columns) - set(features))
        if unknown_binary:
            raise ValueError(f"binary_columns are not configured features: {unknown_binary}")
        numeric_columns = tuple(column for column in features if column not in self.binary_columns)
        values = _numeric_frame(training, features)
        grouped = values.assign(_pitcher=training[self.pitcher_col].to_numpy()).groupby(
            "_pitcher",
            sort=True,
            dropna=False,
        )
        pooled_median = values.median(axis=0).fillna(0.0)
        pitcher_median = grouped[list(features)].median().fillna(pooled_median)

        imputed = values.copy()
        for column in features:
            medians = training[self.pitcher_col].map(pitcher_median[column]).fillna(
                pooled_median[column]
            )
            imputed[column] = imputed[column].fillna(medians)

        if numeric_columns:
            pooled_mean = imputed.loc[:, numeric_columns].mean(axis=0).fillna(0.0)
            pooled_scale = imputed.loc[:, numeric_columns].std(axis=0, ddof=0)
            pooled_scale = pooled_scale.where(pooled_scale.gt(self.epsilon), 1.0).fillna(1.0)
            imputed_grouped = imputed.loc[:, numeric_columns].assign(
                _pitcher=training[self.pitcher_col].to_numpy()
            ).groupby("_pitcher", sort=True, dropna=False)
            pitcher_mean = imputed_grouped[list(numeric_columns)].mean().fillna(pooled_mean)
            pitcher_scale = imputed_grouped[list(numeric_columns)].std(ddof=0)
            pitcher_count = imputed_grouped[list(numeric_columns)].count()
            for column in numeric_columns:
                invalid = pitcher_count[column].lt(2) | pitcher_scale[column].le(self.epsilon)
                pitcher_scale.loc[invalid, column] = pooled_scale[column]
            pitcher_scale = pitcher_scale.fillna(pooled_scale)
        else:
            pooled_mean = pd.Series(dtype=float)
            pooled_scale = pd.Series(dtype=float)
            pitcher_mean = pd.DataFrame(index=pitcher_median.index)
            pitcher_scale = pd.DataFrame(index=pitcher_median.index)
            pitcher_count = pd.DataFrame(index=pitcher_median.index)

        self.feature_columns_ = features
        self.numeric_columns_ = numeric_columns
        self.binary_columns_ = tuple(column for column in features if column in self.binary_columns)
        self.pooled_median_ = pooled_median
        self.pitcher_median_ = pitcher_median
        self.pooled_mean_ = pooled_mean
        self.pooled_scale_ = pooled_scale
        self.pitcher_mean_ = pitcher_mean
        self.pitcher_scale_ = pitcher_scale
        self.pitcher_count_ = pitcher_count
        self.metadata_ = _fit_metadata(
            training,
            year_col=self.year_col,
            game_date_col=self.game_date_col,
        )
        self.fit_years_ = self.metadata_["fit_years"]
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        _require_columns(
            frame,
            [self.pitcher_col, *self.feature_columns_],
            "Cannot transform with sequence preprocessor",
        )
        result = frame.copy()
        values = _numeric_frame(frame, self.feature_columns_)
        pitcher_values = frame[self.pitcher_col]
        for column in self.feature_columns_:
            medians = pitcher_values.map(self.pitcher_median_[column]).fillna(
                self.pooled_median_[column]
            )
            result[column] = values[column].fillna(medians)
        for column in self.numeric_columns_:
            means = pitcher_values.map(self.pitcher_mean_[column]).fillna(
                self.pooled_mean_[column]
            )
            scales = pitcher_values.map(self.pitcher_scale_[column]).fillna(
                self.pooled_scale_[column]
            )
            scales = scales.where(scales.gt(self.epsilon), self.pooled_scale_[column])
            result[column] = (pd.to_numeric(result[column], errors="coerce") - means) / scales
        return result

    def transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transform(frame)
        return transformed.loc[:, self.feature_columns_].copy()

    def fit_transform(
        self,
        training: pd.DataFrame,
        feature_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit(training, feature_columns).transform(training)

    def get_fit_metadata(self) -> dict[str, Any]:
        self._check_fitted()
        return dict(self.metadata_)


PerPitcherTCNPreprocessor = PerPitcherTrainImputerScaler
PitcherStandardScaler = PerPitcherStandardScaler
