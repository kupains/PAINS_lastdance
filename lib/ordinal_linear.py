from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # Keep dataset/report utilities importable without torch.
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


ORDINAL_PREDICTION_COLUMNS = [
    "prob_low",
    "prob_middle",
    "prob_high",
    "predicted_class",
    "latent_score",
    "threshold_1",
    "threshold_2",
]


def torch_available() -> bool:
    """Return whether the optional PyTorch backend can be imported."""
    return torch is not None


def require_torch() -> None:
    """Fail explicitly at model use time while allowing non-model imports."""
    if torch is None:
        raise ImportError(
            "Ordinal linear modeling requires PyTorch. "
            "Install the project requirements (torch>=2.2) before training."
        ) from _TORCH_IMPORT_ERROR


def set_deterministic_seed(seed: int) -> None:
    """Set NumPy/PyTorch seeds and deterministic-algorithm preferences."""
    np.random.seed(int(seed))
    require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), np.finfo(float).eps)
    # Stable inverse of log(1 + exp(x)).
    return float(value + np.log(-np.expm1(-value)))


def ordinal_probabilities_numpy(
    latent_score: np.ndarray | Iterable[float],
    threshold_1: float,
    threshold_2: float,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Convert ordered cumulative logits into normalized Low/Middle/High probabilities."""
    if float(threshold_2) <= float(threshold_1):
        raise ValueError("threshold_2 must be greater than threshold_1.")
    score = np.asarray(latent_score, dtype=float)
    q1 = 1.0 / (1.0 + np.exp(-np.clip(score - float(threshold_1), -35.0, 35.0)))
    q2 = 1.0 / (1.0 + np.exp(-np.clip(score - float(threshold_2), -35.0, 35.0)))
    probabilities = np.column_stack([1.0 - q1, q1 - q2, q2])
    probabilities = np.clip(probabilities, float(epsilon), None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class OrdinalLinearConfig:
    """Small, intentionally bounded configuration for one pitcher model."""

    l2_lambda: float = 1.0
    distance_weight: float = 0.0
    max_iter: int = 200
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 50
    epsilon: float = 1e-7
    seed: int = 20260722
    dtype: str = "float64"

    @classmethod
    def from_value(
        cls, value: "OrdinalLinearConfig | Mapping[str, Any] | None"
    ) -> "OrdinalLinearConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))


@dataclass(frozen=True)
class StandardizerState:
    """Train-only imputation and scaling statistics."""

    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.median):
            raise ValueError(
                f"Expected a 2-D matrix with {len(self.median)} columns; "
                f"got shape {matrix.shape}."
            )
        finite = np.where(np.isfinite(matrix), matrix, self.median)
        return (finite - self.mean) / self.scale


def fit_standardizer(
    values: np.ndarray,
    fallback: StandardizerState | None = None,
) -> StandardizerState:
    """Fit finite train-only column statistics, optionally borrowing pooled statistics."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {matrix.shape}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(np.where(np.isfinite(matrix), matrix, np.nan), axis=0)
    if fallback is not None:
        median = np.where(np.isfinite(median), median, fallback.median)
    median = np.where(np.isfinite(median), median, 0.0)

    filled = np.where(np.isfinite(matrix), matrix, median)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0, ddof=0)
    if fallback is not None:
        insufficient = np.sum(np.isfinite(matrix), axis=0) < 2
        mean = np.where(insufficient | ~np.isfinite(mean), fallback.mean, mean)
        scale = np.where(
            insufficient | ~np.isfinite(scale) | (scale <= 0.0),
            fallback.scale,
            scale,
        )
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    return StandardizerState(median=median, mean=mean, scale=scale)


if torch is not None:

    class ProportionalOddsOrdinalLinear(nn.Module):
        """Per-pitcher proportional-odds linear model with ordered thresholds."""

        def __init__(
            self,
            n_features: int,
            epsilon: float = 1e-7,
            dtype: torch.dtype = torch.float64,
        ) -> None:
            super().__init__()
            if int(n_features) < 1:
                raise ValueError("n_features must be positive.")
            self.n_features = int(n_features)
            self.epsilon = float(epsilon)
            self.beta = nn.Parameter(torch.zeros(self.n_features, dtype=dtype))
            self.threshold_1 = nn.Parameter(torch.tensor(-0.5, dtype=dtype))
            self.threshold_delta = nn.Parameter(
                torch.tensor(_inverse_softplus(1.0 - self.epsilon), dtype=dtype)
            )

        def ordered_thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
            threshold_2 = (
                self.threshold_1 + F.softplus(self.threshold_delta) + self.epsilon
            )
            return self.threshold_1, threshold_2

        def latent_score(self, features: torch.Tensor) -> torch.Tensor:
            return features @ self.beta

        def cumulative_probabilities(
            self, features: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            score = self.latent_score(features)
            threshold_1, threshold_2 = self.ordered_thresholds()
            q1 = torch.sigmoid(score - threshold_1)
            q2 = torch.sigmoid(score - threshold_2)
            return score, q1, q2

        def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            score, q1, q2 = self.cumulative_probabilities(features)
            probabilities = torch.stack((1.0 - q1, q1 - q2, q2), dim=-1)
            probabilities = probabilities.clamp_min(self.epsilon)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            threshold_1, threshold_2 = self.ordered_thresholds()
            return {
                "probabilities": probabilities,
                "latent_score": score,
                "q1": q1,
                "q2": q2,
                "threshold_1": threshold_1,
                "threshold_2": threshold_2,
            }

else:

    class ProportionalOddsOrdinalLinear:  # pragma: no cover - exercised without torch.
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


def ordinal_objective(
    model: ProportionalOddsOrdinalLinear,
    features,
    labels,
    l2_lambda: float = 1.0,
    distance_weight: float = 0.0,
):
    """Cumulative BCE + beta L2 + optional expected ordinal-distance loss."""
    require_torch()
    labels = labels.to(dtype=torch.long)
    if labels.ndim != 1 or torch.any((labels < 0) | (labels > 2)):
        raise ValueError("Ordinal labels must be a 1-D tensor containing only 0, 1, 2.")
    output = model(features)
    target_q1 = (labels >= 1).to(dtype=features.dtype)
    target_q2 = (labels >= 2).to(dtype=features.dtype)
    cumulative_bce = F.binary_cross_entropy_with_logits(
        output["latent_score"] - output["threshold_1"], target_q1
    )
    cumulative_bce = cumulative_bce + F.binary_cross_entropy_with_logits(
        output["latent_score"] - output["threshold_2"], target_q2
    )
    l2_penalty = float(l2_lambda) * torch.sum(model.beta.square())

    classes = torch.arange(3, dtype=features.dtype, device=features.device)
    distances = torch.abs(classes.unsqueeze(0) - labels.to(features.dtype).unsqueeze(1))
    expected_distance = torch.mean(
        torch.sum(output["probabilities"] * distances, dim=1)
    )
    total = (
        cumulative_bce
        + l2_penalty
        + float(distance_weight) * expected_distance
    )
    return {
        "loss": total,
        "ordinal_bce": cumulative_bce,
        "l2_penalty": l2_penalty,
        "expected_distance": expected_distance,
    }


def _torch_dtype(config: OrdinalLinearConfig):
    require_torch()
    if config.dtype == "float64":
        return torch.float64
    if config.dtype == "float32":
        return torch.float32
    raise ValueError("OrdinalLinearConfig.dtype must be 'float32' or 'float64'.")


def fit_ordinal_linear(
    features: np.ndarray,
    labels: np.ndarray,
    config: OrdinalLinearConfig | Mapping[str, Any] | None = None,
) -> tuple[ProportionalOddsOrdinalLinear, dict[str, float]]:
    """Fit one deterministic proportional-odds model with full-batch LBFGS."""
    require_torch()
    cfg = OrdinalLinearConfig.from_value(config)
    set_deterministic_seed(cfg.seed)

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError("features/labels must be non-empty and row-aligned.")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite after train-fitted preprocessing.")
    if not np.all(np.isin(y, [0, 1, 2])):
        raise ValueError("labels must contain only 0, 1, 2.")

    dtype = _torch_dtype(cfg)
    x_tensor = torch.as_tensor(x, dtype=dtype)
    y_tensor = torch.as_tensor(y.astype(np.int64), dtype=torch.long)
    model = ProportionalOddsOrdinalLinear(
        n_features=x.shape[1],
        epsilon=cfg.epsilon,
        dtype=dtype,
    )

    # Start thresholds at the empirical cumulative class frequencies.
    p_ge_1 = float(np.clip(np.mean(y >= 1), cfg.epsilon, 1.0 - cfg.epsilon))
    p_ge_2 = float(np.clip(np.mean(y >= 2), cfg.epsilon, 1.0 - cfg.epsilon))
    initial_t1 = -float(np.log(p_ge_1 / (1.0 - p_ge_1)))
    initial_t2 = -float(np.log(p_ge_2 / (1.0 - p_ge_2)))
    initial_t2 = max(initial_t2, initial_t1 + 10.0 * cfg.epsilon)
    with torch.no_grad():
        model.threshold_1.copy_(torch.tensor(initial_t1, dtype=dtype))
        gap = max(initial_t2 - initial_t1 - cfg.epsilon, cfg.epsilon)
        model.threshold_delta.copy_(
            torch.tensor(_inverse_softplus(gap), dtype=dtype)
        )

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=int(cfg.max_iter),
        tolerance_grad=float(cfg.tolerance_grad),
        tolerance_change=float(cfg.tolerance_change),
        history_size=int(cfg.history_size),
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        objective = ordinal_objective(
            model,
            x_tensor,
            y_tensor,
            l2_lambda=cfg.l2_lambda,
            distance_weight=cfg.distance_weight,
        )
        objective["loss"].backward()
        return objective["loss"]

    optimizer.step(closure)
    model.eval()
    with torch.no_grad():
        final = ordinal_objective(
            model,
            x_tensor,
            y_tensor,
            l2_lambda=cfg.l2_lambda,
            distance_weight=cfg.distance_weight,
        )
    diagnostics = {name: float(value.detach().cpu()) for name, value in final.items()}
    diagnostics["parameter_count"] = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    return model, diagnostics


def predict_ordinal_linear(
    model: ProportionalOddsOrdinalLinear,
    features: np.ndarray,
) -> pd.DataFrame:
    """Return the stable ordinal prediction-column contract."""
    require_torch()
    parameter = next(model.parameters())
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != model.n_features:
        raise ValueError(
            f"Expected shape (n, {model.n_features}), got {x.shape}."
        )
    if not np.isfinite(x).all():
        raise ValueError("features must be finite after train-fitted preprocessing.")
    with torch.no_grad():
        output = model(
            torch.as_tensor(x, dtype=parameter.dtype, device=parameter.device)
        )
    probabilities = output["probabilities"].detach().cpu().numpy()
    score = output["latent_score"].detach().cpu().numpy()
    threshold_1 = float(output["threshold_1"].detach().cpu())
    threshold_2 = float(output["threshold_2"].detach().cpu())
    return pd.DataFrame(
        {
            "prob_low": probabilities[:, 0],
            "prob_middle": probabilities[:, 1],
            "prob_high": probabilities[:, 2],
            "predicted_class": probabilities.argmax(axis=1).astype(int),
            "latent_score": score,
            "threshold_1": threshold_1,
            "threshold_2": threshold_2,
        }
    )


class PerPitcherOrdinalLinear:
    """Fit one train-scaled proportional-odds model per pitcher."""

    def __init__(
        self,
        feature_names: Iterable[str],
        config: OrdinalLinearConfig | Mapping[str, Any] | None = None,
        pitcher_col: str = "pitcher",
    ) -> None:
        self.feature_names = list(feature_names)
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty.")
        self.config = OrdinalLinearConfig.from_value(config)
        self.pitcher_col = pitcher_col
        self.models_: dict[Any, ProportionalOddsOrdinalLinear] = {}
        self.scalers_: dict[Any, StandardizerState] = {}
        self.fit_diagnostics_: dict[Any, dict[str, float]] = {}
        self.pooled_scaler_: StandardizerState | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "true_class",
    ) -> "PerPitcherOrdinalLinear":
        require_torch()
        required = [self.pitcher_col, target_col, *self.feature_names]
        missing = [column for column in required if column not in train_df.columns]
        if missing:
            raise ValueError(f"Missing ordinal training columns: {missing}")
        labels = pd.to_numeric(train_df[target_col], errors="coerce")
        valid = labels.isin([0, 1, 2])
        data = train_df.loc[valid, required].copy()
        if data.empty:
            raise ValueError("No valid ordinal training rows.")

        pooled_values = data[self.feature_names].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        self.pooled_scaler_ = fit_standardizer(pooled_values)
        self.models_.clear()
        self.scalers_.clear()
        self.fit_diagnostics_.clear()

        for offset, (pitcher, group) in enumerate(
            data.groupby(self.pitcher_col, sort=True, dropna=False)
        ):
            raw = group[self.feature_names].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            scaler = fit_standardizer(raw, fallback=self.pooled_scaler_)
            x = scaler.transform(raw)
            y = pd.to_numeric(group[target_col], errors="raise").to_numpy(dtype=int)
            pitcher_cfg = OrdinalLinearConfig(
                **{
                    **asdict(self.config),
                    "seed": int(self.config.seed) + int(offset),
                }
            )
            model, diagnostics = fit_ordinal_linear(x, y, pitcher_cfg)
            self.models_[pitcher] = model
            self.scalers_[pitcher] = scaler
            self.fit_diagnostics_[pitcher] = diagnostics
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        require_torch()
        required = [self.pitcher_col, *self.feature_names]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing ordinal prediction columns: {missing}")
        unknown = sorted(
            set(frame[self.pitcher_col].drop_duplicates()) - set(self.models_),
            key=str,
        )
        if unknown:
            raise KeyError(f"No fitted ordinal model for pitchers: {unknown}")

        output = pd.DataFrame(index=frame.index, columns=ORDINAL_PREDICTION_COLUMNS)
        for pitcher, group in frame.groupby(
            self.pitcher_col, sort=False, dropna=False
        ):
            raw = group[self.feature_names].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            x = self.scalers_[pitcher].transform(raw)
            predicted = predict_ordinal_linear(self.models_[pitcher], x)
            predicted.index = group.index
            output.loc[group.index, ORDINAL_PREDICTION_COLUMNS] = predicted

        for column in ["prob_low", "prob_middle", "prob_high", "latent_score",
                       "threshold_1", "threshold_2"]:
            output[column] = pd.to_numeric(output[column], errors="coerce")
        output["predicted_class"] = pd.to_numeric(
            output["predicted_class"], errors="raise"
        ).astype(int)
        return output
