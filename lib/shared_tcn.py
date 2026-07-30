from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from lib.preprocessing import PerPitcherTrainImputerScaler

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # Dataset construction remains usable without torch.
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


# Raw prior-outing values only. In particular, this contract intentionally excludes
# release_speed (the all-pitch mean), every moving average/slope, and EWMA-difference
# summary features.
TCN_RAW_FEATURES = [
    "stuff_plus",
    "pitch_count",
    "rest_days_log",
    "long_gap",
    "season_start",
    "primary_fb_velocity",
    "primary_fb_spin",
    "primary_fb_pfx_x",
    "primary_fb_pfx_z",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "arm_angle",
    "fastball_share",
    "breaking_share",
]
RAW_SEQUENCE_FEATURES = TCN_RAW_FEATURES
TCN_BINARY_FEATURES = {"long_gap", "season_start"}


def torch_available() -> bool:
    return torch is not None


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "Shared TCN modeling requires PyTorch. "
            "Install the project requirements (torch>=2.2) before training."
        ) from _TORCH_IMPORT_ERROR


def set_deterministic_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch seeds and deterministic algorithm preferences."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def ewma_tertile_position(
    ewma: np.ndarray | Iterable[float],
    q33: np.ndarray | Iterable[float],
    q67: np.ndarray | Iterable[float],
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Fold-train-defined normalized EWMA position used by ordinal auxiliary heads."""
    ewma_values = np.asarray(ewma, dtype=float)
    low = np.asarray(q33, dtype=float)
    high = np.asarray(q67, dtype=float)
    return (ewma_values - low) / (high - low + float(epsilon))


def residual_to_stuff_plus(
    ewma4: np.ndarray | Iterable[float],
    predicted_residual: np.ndarray | Iterable[float],
    alpha: float,
) -> np.ndarray:
    return np.asarray(ewma4, dtype=float) + float(alpha) * np.asarray(
        predicted_residual, dtype=float
    )


@dataclass
class TCNSequenceBatch:
    """Leakage-auditable raw sequence tensors plus their masks and provenance."""

    values: np.ndarray
    valid_timestep: np.ndarray
    primary_fb_available: np.ndarray
    stuff_plus_available: np.ndarray
    pitchers: np.ndarray
    target_index: np.ndarray
    target_game_dates: np.ndarray
    sequence_game_dates: np.ndarray
    sequence_row_ids: np.ndarray
    feature_names: tuple[str, ...]
    sequence_order: str

    def __post_init__(self) -> None:
        n, length, n_features = self.values.shape
        if n_features != len(self.feature_names):
            raise ValueError("values feature dimension does not match feature_names.")
        for name in [
            "valid_timestep",
            "primary_fb_available",
            "stuff_plus_available",
            "sequence_game_dates",
            "sequence_row_ids",
        ]:
            if getattr(self, name).shape != (n, length):
                raise ValueError(f"{name} must have shape {(n, length)}.")
        if len(self.pitchers) != n or len(self.target_index) != n:
            raise ValueError("Target metadata must align with sequence rows.")

    @property
    def sequence_length(self) -> int:
        return int(self.values.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    def assert_strictly_prior(self) -> None:
        for position in range(self.n_samples):
            dates = self.sequence_game_dates[position][
                self.valid_timestep[position]
            ]
            if len(dates) and not np.all(dates < self.target_game_dates[position]):
                raise AssertionError(
                    "TCN sequence leakage: a history date is not strictly before "
                    f"target index {self.target_index[position]!r}."
                )


def _coerce_bool(series: pd.Series, default: bool = True) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(default).to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(int(default)).astype(bool).to_numpy()


def _stable_history_frame(
    history_df: pd.DataFrame,
    pitcher_col: str,
    date_col: str,
) -> pd.DataFrame:
    required = [pitcher_col, date_col]
    missing = [column for column in required if column not in history_df.columns]
    if missing:
        raise ValueError(f"Missing TCN history columns: {missing}")
    history = history_df.copy()
    history["_tcn_original_index"] = history.index.astype(object)
    history["_tcn_original_order"] = np.arange(len(history), dtype=int)
    history[date_col] = pd.to_datetime(history[date_col], errors="coerce")
    if history[date_col].isna().any():
        raise ValueError("TCN history contains an invalid game_date.")
    sort_columns = [pitcher_col, date_col]
    if "game_pk" in history.columns:
        sort_columns.append("game_pk")
    sort_columns.append("_tcn_original_order")
    return history.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def build_tcn_sequences(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame | None = None,
    *,
    feature_names: Sequence[str] = TCN_RAW_FEATURES,
    sequence_length: int = 5,
    sequence_order: str = "ordered",
    seed: int = 20260722,
    pitcher_col: str = "pitcher",
    date_col: str = "game_date",
    row_id_col: str = "row_id",
    history_eligible_col: str = "history_eligible",
    target_eligible_col: str = "target_eligible",
    preprocessor: "TCNPreprocessor | None" = None,
) -> TCNSequenceBatch:
    """Build strictly-prior, cross-season sequences with deterministic left padding.

    `sequence_order="shuffled"` uses the identical most-recent row set as the
    ordered ablation and permutes only its temporal order with the fixed seed.
    """
    features = tuple(feature_names)
    if int(sequence_length) < 1:
        raise ValueError("sequence_length must be positive.")
    if sequence_order not in {"ordered", "shuffled"}:
        raise ValueError("sequence_order must be 'ordered' or 'shuffled'.")
    missing_features = [column for column in features if column not in history_df]
    if missing_features:
        raise ValueError(f"Missing TCN raw features: {missing_features}")
    if "release_speed" in features or any(
        name.endswith(("_ma5", "_slope5")) for name in features
    ):
        raise ValueError(
            "TCN raw sequences cannot include all-pitch release_speed or rolling/slope features."
        )

    history = _stable_history_frame(history_df, pitcher_col, date_col)
    if history_eligible_col in history:
        history = history.loc[_coerce_bool(history[history_eligible_col])].copy()

    if target_df is None:
        targets = history_df.copy()
        if target_eligible_col in targets:
            targets = targets.loc[
                _coerce_bool(targets[target_eligible_col], default=False)
            ].copy()
    else:
        targets = target_df.copy()
    required_targets = [pitcher_col, date_col]
    missing_targets = [column for column in required_targets if column not in targets]
    if missing_targets:
        raise ValueError(f"Missing TCN target columns: {missing_targets}")
    targets[date_col] = pd.to_datetime(targets[date_col], errors="coerce")
    if targets[date_col].isna().any():
        raise ValueError("TCN targets contain an invalid game_date.")

    raw_matrix = history.loc[:, features].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    transformed_matrix: np.ndarray | None = None
    if preprocessor is not None and hasattr(
        preprocessor, "transform_features"
    ):
        # The fitted transform is a property of each history row, not of the
        # target sequence that later selects it. Transform the history once.
        # Calling a pandas transform separately for every target made a single
        # real fold spend minutes repeating identical map/fill operations.
        transformed = preprocessor.transform_features(history.copy())
        transformed_matrix = transformed.loc[:, features].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
    history_groups = {
        pitcher: np.asarray(positions, dtype=int)
        for pitcher, positions in history.groupby(
            pitcher_col, sort=False, dropna=False
        ).indices.items()
    }
    history_dates = history[date_col].to_numpy(dtype="datetime64[ns]")
    row_ids = (
        history[row_id_col].astype(object).to_numpy()
        if row_id_col in history
        else history["_tcn_original_index"].astype(object).to_numpy()
    )
    rng = np.random.default_rng(int(seed))

    n = len(targets)
    length = int(sequence_length)
    values = np.zeros((n, length, len(features)), dtype=np.float32)
    valid = np.zeros((n, length), dtype=bool)
    primary_available = np.zeros((n, length), dtype=bool)
    stuff_available = np.zeros((n, length), dtype=bool)
    sequence_dates = np.full((n, length), np.datetime64("NaT"), dtype="datetime64[ns]")
    sequence_ids = np.full((n, length), None, dtype=object)

    for target_position, (_, target) in enumerate(targets.iterrows()):
        pitcher = target[pitcher_col]
        possible = history_groups.get(pitcher, np.empty(0, dtype=int))
        target_date = np.datetime64(target[date_col].to_datetime64(), "ns")
        possible = possible[history_dates[possible] < target_date]
        selected = possible[-length:]
        if sequence_order == "shuffled" and len(selected) > 1:
            selected = selected[rng.permutation(len(selected))]
        start = length - len(selected)
        if not len(selected):
            continue

        selected_values = (
            transformed_matrix[selected]
            if transformed_matrix is not None
            else raw_matrix[selected]
        )
        if preprocessor is not None and transformed_matrix is None:
            if hasattr(preprocessor, "transform_array"):
                selected_values = preprocessor.transform_array(
                    selected_values, pitcher
                )
            elif hasattr(preprocessor, "transform"):
                transformed = preprocessor.transform(history.iloc[selected].copy())
                selected_values = transformed.loc[:, features].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=float)
            else:
                raise TypeError(
                    "preprocessor must provide transform_array(values, pitcher) "
                    "or transform(frame)."
                )
        values[target_position, start:] = selected_values.astype(np.float32)
        valid[target_position, start:] = True
        sequence_dates[target_position, start:] = history_dates[selected]
        sequence_ids[target_position, start:] = row_ids[selected]

        stuff_idx = features.index("stuff_plus") if "stuff_plus" in features else None
        if "stuff_plus_available" in history:
            stuff_available[target_position, start:] = _coerce_bool(
                history.iloc[selected]["stuff_plus_available"], default=False
            )
        elif stuff_idx is not None:
            stuff_available[target_position, start:] = np.isfinite(
                raw_matrix[selected, stuff_idx]
            )
        else:
            stuff_available[target_position, start:] = True

        if "primary_fb_available" in history:
            primary_available[target_position, start:] = _coerce_bool(
                history.iloc[selected]["primary_fb_available"], default=False
            )
        elif "primary_fb_velocity" in features:
            velocity_idx = features.index("primary_fb_velocity")
            primary_available[target_position, start:] = np.isfinite(
                raw_matrix[selected, velocity_idx]
            )
        else:
            primary_available[target_position, start:] = True

    target_indices = targets.index.astype(object).to_numpy()
    target_dates = targets[date_col].to_numpy(dtype="datetime64[ns]")
    batch = TCNSequenceBatch(
        values=values,
        valid_timestep=valid,
        primary_fb_available=primary_available,
        stuff_plus_available=stuff_available,
        pitchers=targets[pitcher_col].astype(object).to_numpy(),
        target_index=target_indices,
        target_game_dates=target_dates,
        sequence_game_dates=sequence_dates,
        sequence_row_ids=sequence_ids,
        feature_names=features,
        sequence_order=sequence_order,
    )
    batch.assert_strictly_prior()
    return batch


class TCNPreprocessor(PerPitcherTrainImputerScaler):
    """Convenience specialization of the shared fold-train preprocessing utility."""

    def __init__(
        self,
        feature_names: Sequence[str] = TCN_RAW_FEATURES,
        binary_features: Iterable[str] = TCN_BINARY_FEATURES,
        pitcher_col: str = "pitcher",
    ) -> None:
        self.feature_names = tuple(feature_names)
        super().__init__(
            feature_columns=self.feature_names,
            binary_columns=tuple(binary_features),
            pitcher_col=pitcher_col,
        )


@dataclass(frozen=True)
class SharedTCNConfig:
    sequence_length: int = 5
    internal_channels: int = 8
    bottleneck_dim: int = 4
    kernel_size: int = 2
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    head_type: str = "H1"
    huber_delta: float = 1.0
    pitcher_weight_l2: float = 0.01
    pitcher_bias_l2: float = 0.01
    encoder_l2: float = 0.0
    ordinal_weight: float = 0.0
    alpha: float = 0.25
    learning_rate: float = 1e-3
    epochs: int = 250
    gradient_clip: float = 5.0
    seed: int = 20260722
    dtype: str = "float32"

    @classmethod
    def from_value(
        cls, value: "SharedTCNConfig | Mapping[str, Any] | None"
    ) -> "SharedTCNConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        incoming = dict(value)
        if "dilations" in incoming:
            incoming["dilations"] = tuple(incoming["dilations"])
        return cls(**incoming)


def build_pitcher_index(
    pitchers: Iterable[Any],
    mapping: Mapping[Any, int] | None = None,
) -> tuple[np.ndarray, dict[Any, int]]:
    values = np.asarray(list(pitchers), dtype=object)
    if pd.isna(values).any():
        raise ValueError("Pitcher identifiers cannot be missing.")
    if mapping is None:
        resolved = {
            pitcher: position
            for position, pitcher in enumerate(sorted(pd.unique(values), key=str))
        }
    else:
        resolved = dict(mapping)
    unknown = sorted(set(values) - set(resolved), key=str)
    if unknown:
        raise KeyError(f"Unknown pitchers for shared TCN: {unknown}")
    return np.asarray([resolved[pitcher] for pitcher in values], dtype=np.int64), resolved


if torch is not None:

    class LeftCausalConv1d(nn.Module):
        """Conv1d with left-only padding and unchanged output length."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
        ) -> None:
            super().__init__()
            self.left_padding = (int(kernel_size) - 1) * int(dilation)
            self.conv = nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                dilation=int(dilation),
                padding=0,
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            padded = F.pad(values, (self.left_padding, 0))
            return self.conv(padded)


    class CausalResidualBlock(nn.Module):
        def __init__(
            self,
            channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.conv = LeftCausalConv1d(
                channels, channels, kernel_size=kernel_size, dilation=dilation
            )
            self.normalization = nn.LayerNorm(channels)
            self.dropout = nn.Dropout(float(dropout))

        def forward(
            self, values: torch.Tensor, valid_timestep: torch.Tensor
        ) -> torch.Tensor:
            residual = values
            output = self.conv(values).transpose(1, 2)
            output = self.normalization(output)
            output = self.dropout(F.gelu(output)).transpose(1, 2)
            output = (residual + output) * valid_timestep.unsqueeze(1)
            return output


    class SharedTCNEncoder(nn.Module):
        def __init__(
            self,
            input_dim: int = len(TCN_RAW_FEATURES),
            internal_channels: int = 8,
            bottleneck_dim: int = 4,
            kernel_size: int = 2,
            dilations: Sequence[int] = (1, 2, 4),
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.input_projection = nn.Linear(self.input_dim, int(internal_channels))
            # Missingness masks are separate from the exact 15 raw values, so the
            # documented raw feature dimension remains stable.
            self.availability_projection = nn.Linear(
                2, int(internal_channels), bias=False
            )
            self.blocks = nn.ModuleList(
                [
                    CausalResidualBlock(
                        int(internal_channels),
                        kernel_size=int(kernel_size),
                        dilation=int(dilation),
                        dropout=float(dropout),
                    )
                    for dilation in dilations
                ]
            )
            self.bottleneck = nn.Linear(
                int(internal_channels), int(bottleneck_dim)
            )

        def forward(
            self,
            values: torch.Tensor,
            valid_timestep: torch.Tensor,
            availability: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if values.ndim != 3 or values.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected values shape (batch, length, {self.input_dim}); "
                    f"got {tuple(values.shape)}."
                )
            if valid_timestep.shape != values.shape[:2]:
                raise ValueError("valid_timestep must match batch and sequence dimensions.")
            valid = valid_timestep.to(dtype=values.dtype)
            output = self.input_projection(values)
            if availability is not None:
                if availability.shape != (*values.shape[:2], 2):
                    raise ValueError(
                        "availability must have shape (batch, length, 2)."
                    )
                output = output + self.availability_projection(
                    availability.to(dtype=values.dtype)
                )
            output = output * valid.unsqueeze(-1)
            output = output.transpose(1, 2)
            for block in self.blocks:
                output = block(output, valid)
            output = output.transpose(1, 2)

            positions = torch.arange(
                values.shape[1], device=values.device, dtype=torch.long
            ).unsqueeze(0)
            last_position = positions.expand(values.shape[0], -1).masked_fill(
                ~valid_timestep.bool(), -1
            ).max(dim=1).values
            has_history = last_position >= 0
            gather_position = last_position.clamp_min(0)
            last = output[
                torch.arange(values.shape[0], device=values.device),
                gather_position,
            ]
            last = last * has_history.to(last.dtype).unsqueeze(1)
            bottleneck = self.bottleneck(last)
            return bottleneck * has_history.to(bottleneck.dtype).unsqueeze(1)


    class SharedTCN(nn.Module):
        """Shared encoder/global residual plus H0/H1/H2 pitcher correction."""

        def __init__(
            self,
            num_pitchers: int,
            input_dim: int = len(TCN_RAW_FEATURES),
            config: SharedTCNConfig | Mapping[str, Any] | None = None,
        ) -> None:
            super().__init__()
            self.config = SharedTCNConfig.from_value(config)
            self.head_type = self.config.head_type.upper()
            if self.head_type not in {"H0", "H1", "H2"}:
                raise ValueError("head_type must be H0, H1, or H2.")
            if int(num_pitchers) < 1:
                raise ValueError("num_pitchers must be positive.")
            self.num_pitchers = int(num_pitchers)
            self.encoder = SharedTCNEncoder(
                input_dim=input_dim,
                internal_channels=self.config.internal_channels,
                bottleneck_dim=self.config.bottleneck_dim,
                kernel_size=self.config.kernel_size,
                dilations=self.config.dilations,
                dropout=self.config.dropout,
            )
            self.global_head = nn.Linear(self.config.bottleneck_dim, 1)
            self.pitcher_bias = None
            self.pitcher_vector = None
            if self.head_type in {"H1", "H2"}:
                self.pitcher_bias = nn.Embedding(self.num_pitchers, 1)
                nn.init.zeros_(self.pitcher_bias.weight)
            if self.head_type == "H2":
                self.pitcher_vector = nn.Embedding(
                    self.num_pitchers, self.config.bottleneck_dim
                )
                nn.init.zeros_(self.pitcher_vector.weight)

            self.ordinal_score = None
            if self.config.ordinal_weight > 0.0:
                self.ordinal_score = nn.Linear(
                    self.config.bottleneck_dim + 1, 1
                )
                self.ordinal_threshold_1 = nn.Parameter(torch.tensor(-0.5))
                self.ordinal_threshold_delta = nn.Parameter(torch.tensor(0.5))

        def pitcher_penalty(self) -> torch.Tensor:
            reference = self.global_head.weight
            penalty = torch.zeros((), dtype=reference.dtype, device=reference.device)
            if self.pitcher_vector is not None:
                penalty = penalty + float(
                    self.config.pitcher_weight_l2
                ) * self.pitcher_vector.weight.square().sum()
            if self.pitcher_bias is not None:
                penalty = penalty + float(
                    self.config.pitcher_bias_l2
                ) * self.pitcher_bias.weight.square().sum()
            return penalty

        def encoder_penalty(self) -> torch.Tensor:
            reference = self.global_head.weight
            if self.config.encoder_l2 <= 0.0:
                return torch.zeros(
                    (), dtype=reference.dtype, device=reference.device
                )
            return float(self.config.encoder_l2) * sum(
                parameter.square().sum() for parameter in self.encoder.parameters()
            )

        def forward(
            self,
            values: torch.Tensor,
            valid_timestep: torch.Tensor,
            pitcher_index: torch.Tensor,
            ewma_position: torch.Tensor | None = None,
            primary_fb_available: torch.Tensor | None = None,
            stuff_plus_available: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            if pitcher_index.ndim != 1 or len(pitcher_index) != len(values):
                raise ValueError("pitcher_index must be one integer per sequence.")
            availability = None
            if primary_fb_available is not None or stuff_plus_available is not None:
                if primary_fb_available is None or stuff_plus_available is None:
                    raise ValueError(
                        "Both primary_fb_available and stuff_plus_available are "
                        "required when availability masks are supplied."
                    )
                availability = torch.stack(
                    [primary_fb_available, stuff_plus_available], dim=-1
                )
            representation = self.encoder(
                values, valid_timestep, availability=availability
            )
            global_residual = self.global_head(representation).squeeze(1)
            correction = torch.zeros_like(global_residual)
            if self.pitcher_bias is not None:
                correction = correction + self.pitcher_bias(pitcher_index).squeeze(1)
            if self.pitcher_vector is not None:
                correction = correction + torch.sum(
                    self.pitcher_vector(pitcher_index) * representation, dim=1
                )
            output = {
                "representation": representation,
                "global_residual": global_residual,
                "pitcher_correction": correction,
                "predicted_residual": global_residual + correction,
            }

            if self.ordinal_score is not None:
                if ewma_position is None:
                    raise ValueError(
                        "ewma_position is required when ordinal_weight is positive."
                    )
                ordinal_input = torch.cat(
                    [representation, ewma_position.reshape(-1, 1)], dim=1
                )
                score = self.ordinal_score(ordinal_input).squeeze(1)
                threshold_1 = self.ordinal_threshold_1
                threshold_2 = (
                    threshold_1
                    + F.softplus(self.ordinal_threshold_delta)
                    + 1e-7
                )
                q1 = torch.sigmoid(score - threshold_1)
                q2 = torch.sigmoid(score - threshold_2)
                probabilities = torch.stack((1.0 - q1, q1 - q2, q2), dim=1)
                probabilities = probabilities.clamp_min(1e-7)
                probabilities = probabilities / probabilities.sum(
                    dim=1, keepdim=True
                )
                output.update(
                    {
                        "ordinal_score": score,
                        "ordinal_q1": q1,
                        "ordinal_q2": q2,
                        "ordinal_probabilities": probabilities,
                        "ordinal_threshold_1": threshold_1,
                        "ordinal_threshold_2": threshold_2,
                    }
                )
            return output

else:

    class LeftCausalConv1d:  # pragma: no cover - exercised without torch.
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


    class CausalResidualBlock:
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


    class SharedTCNEncoder:
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


    class SharedTCN:
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


def pitcher_balanced_mean(losses, pitcher_index):
    """Average within pitcher first, then give each represented pitcher equal weight."""
    require_torch()
    if losses.ndim != 1 or pitcher_index.shape != losses.shape:
        raise ValueError("losses and pitcher_index must be aligned 1-D tensors.")
    means = [
        losses[pitcher_index == pitcher].mean()
        for pitcher in torch.unique(pitcher_index, sorted=True)
    ]
    if not means:
        raise ValueError("Cannot aggregate an empty loss tensor.")
    return torch.stack(means).mean()


def shared_tcn_objective(
    model: SharedTCN,
    output: Mapping[str, Any],
    true_residual,
    pitcher_index,
    true_class=None,
) -> dict[str, Any]:
    """Pitcher-balanced Huber plus shrinkage and optional cumulative ordinal BCE."""
    require_torch()
    residual_losses = F.huber_loss(
        output["predicted_residual"],
        true_residual,
        reduction="none",
        delta=float(model.config.huber_delta),
    )
    residual_loss = pitcher_balanced_mean(residual_losses, pitcher_index)
    pitcher_penalty = model.pitcher_penalty()
    encoder_penalty = model.encoder_penalty()
    ordinal_loss = torch.zeros_like(residual_loss)

    if model.config.ordinal_weight > 0.0:
        if true_class is None or "ordinal_q1" not in output:
            raise ValueError(
                "true_class and ordinal outputs are required for auxiliary training."
            )
        if torch.any((true_class < 0) | (true_class > 2)):
            raise ValueError("true_class must contain only 0, 1, 2.")
        target_q1 = (true_class >= 1).to(dtype=true_residual.dtype)
        target_q2 = (true_class >= 2).to(dtype=true_residual.dtype)
        row_ordinal = F.binary_cross_entropy_with_logits(
            output["ordinal_score"] - output["ordinal_threshold_1"],
            target_q1,
            reduction="none",
        )
        row_ordinal = row_ordinal + F.binary_cross_entropy_with_logits(
            output["ordinal_score"] - output["ordinal_threshold_2"],
            target_q2,
            reduction="none",
        )
        ordinal_loss = pitcher_balanced_mean(row_ordinal, pitcher_index)

    total = (
        residual_loss
        + pitcher_penalty
        + encoder_penalty
        + float(model.config.ordinal_weight) * ordinal_loss
    )
    return {
        "loss": total,
        "residual_loss": residual_loss,
        "pitcher_penalty": pitcher_penalty,
        "encoder_penalty": encoder_penalty,
        "ordinal_loss": ordinal_loss,
    }


def _resolve_device(device: str):
    require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def _resolve_dtype(config: SharedTCNConfig):
    require_torch()
    if config.dtype == "float32":
        return torch.float32
    if config.dtype == "float64":
        return torch.float64
    raise ValueError("SharedTCNConfig.dtype must be 'float32' or 'float64'.")


def fit_shared_tcn(
    sequences: np.ndarray,
    valid_timestep: np.ndarray,
    pitcher_index: np.ndarray,
    true_residual: np.ndarray,
    *,
    num_pitchers: int,
    config: SharedTCNConfig | Mapping[str, Any] | None = None,
    true_class: np.ndarray | None = None,
    ewma_position: np.ndarray | None = None,
    primary_fb_available: np.ndarray | None = None,
    stuff_plus_available: np.ndarray | None = None,
    device: str = "auto",
) -> tuple[SharedTCN, pd.DataFrame]:
    """Fit a deterministic shared TCN with full-dataset pitcher-balanced loss."""
    require_torch()
    cfg = SharedTCNConfig.from_value(config)
    set_deterministic_seed(cfg.seed)
    # Copy caller-owned arrays before torch.as_tensor. Pandas/NumPy can expose
    # read-only views, which PyTorch otherwise warns may have undefined writes.
    x = np.asarray(sequences, dtype=float).copy()
    mask = np.asarray(valid_timestep, dtype=bool).copy()
    pitcher = np.asarray(pitcher_index, dtype=np.int64).copy()
    residual = np.asarray(true_residual, dtype=float).copy()
    if x.ndim != 3 or x.shape[2] != len(TCN_RAW_FEATURES):
        raise ValueError(
            f"sequences must have shape (n, length, {len(TCN_RAW_FEATURES)})."
        )
    if mask.shape != x.shape[:2] or len(pitcher) != len(x) or len(residual) != len(x):
        raise ValueError("TCN training arrays are not row-aligned.")
    if not np.isfinite(x).all() or not np.isfinite(residual).all():
        raise ValueError("TCN training values must be finite after preprocessing.")
    if np.any((pitcher < 0) | (pitcher >= int(num_pitchers))):
        raise ValueError("pitcher_index contains an out-of-range value.")
    if cfg.ordinal_weight > 0.0 and (true_class is None or ewma_position is None):
        raise ValueError(
            "true_class and ewma_position are required when ordinal_weight is positive."
        )
    if (primary_fb_available is None) != (stuff_plus_available is None):
        raise ValueError(
            "Both primary_fb_available and stuff_plus_available must be supplied together."
        )
    for name, availability in [
        ("primary_fb_available", primary_fb_available),
        ("stuff_plus_available", stuff_plus_available),
    ]:
        if availability is not None and np.asarray(availability).shape != mask.shape:
            raise ValueError(f"{name} must have shape {mask.shape}.")

    resolved_device = _resolve_device(device)
    dtype = _resolve_dtype(cfg)
    model = SharedTCN(
        num_pitchers=num_pitchers,
        input_dim=x.shape[2],
        config=cfg,
    ).to(device=resolved_device, dtype=dtype)
    x_tensor = torch.as_tensor(x, dtype=dtype, device=resolved_device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=resolved_device)
    pitcher_tensor = torch.as_tensor(
        pitcher, dtype=torch.long, device=resolved_device
    )
    residual_tensor = torch.as_tensor(
        residual, dtype=dtype, device=resolved_device
    )
    class_tensor = (
        torch.as_tensor(
            np.asarray(true_class, dtype=np.int64).copy(),
            dtype=torch.long,
            device=resolved_device,
        )
        if true_class is not None
        else None
    )
    ewma_tensor = (
        torch.as_tensor(
            np.asarray(ewma_position, dtype=float).copy(),
            dtype=dtype,
            device=resolved_device,
        )
        if ewma_position is not None
        else None
    )
    primary_available_tensor = (
        torch.as_tensor(
            np.asarray(primary_fb_available, dtype=bool).copy(),
            dtype=dtype,
            device=resolved_device,
        )
        if primary_fb_available is not None
        else None
    )
    stuff_available_tensor = (
        torch.as_tensor(
            np.asarray(stuff_plus_available, dtype=bool).copy(),
            dtype=dtype,
            device=resolved_device,
        )
        if stuff_plus_available is not None
        else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    history: list[dict[str, float]] = []
    for epoch in range(int(cfg.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            x_tensor,
            mask_tensor,
            pitcher_tensor,
            ewma_position=ewma_tensor,
            primary_fb_available=primary_available_tensor,
            stuff_plus_available=stuff_available_tensor,
        )
        objective = shared_tcn_objective(
            model,
            output,
            residual_tensor,
            pitcher_tensor,
            true_class=class_tensor,
        )
        objective["loss"].backward()
        if cfg.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.gradient_clip)
            )
        optimizer.step()
        history.append(
            {
                "epoch": epoch + 1,
                **{
                    name: float(value.detach().cpu())
                    for name, value in objective.items()
                },
            }
        )
    model.eval()
    return model, pd.DataFrame(history)


def predict_shared_tcn(
    model: SharedTCN,
    sequences: np.ndarray,
    valid_timestep: np.ndarray,
    pitcher_index: np.ndarray,
    *,
    ewma_position: np.ndarray | None = None,
    primary_fb_available: np.ndarray | None = None,
    stuff_plus_available: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return residual predictions and, when fitted, separate auxiliary probabilities."""
    require_torch()
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    x = np.asarray(sequences, dtype=float).copy()
    mask = np.asarray(valid_timestep, dtype=bool).copy()
    pitcher = np.asarray(pitcher_index, dtype=np.int64).copy()
    if not np.isfinite(x).all():
        raise ValueError("TCN prediction values must be finite after preprocessing.")
    if (primary_fb_available is None) != (stuff_plus_available is None):
        raise ValueError(
            "Both primary_fb_available and stuff_plus_available must be supplied together."
        )
    with torch.no_grad():
        output = model(
            torch.as_tensor(x, dtype=dtype, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device),
            torch.as_tensor(pitcher, dtype=torch.long, device=device),
            ewma_position=(
                torch.as_tensor(
                    np.asarray(ewma_position, dtype=float).copy(),
                    dtype=dtype,
                    device=device,
                )
                if ewma_position is not None
                else None
            ),
            primary_fb_available=(
                torch.as_tensor(
                    np.asarray(primary_fb_available, dtype=bool).copy(),
                    dtype=dtype,
                    device=device,
                )
                if primary_fb_available is not None
                else None
            ),
            stuff_plus_available=(
                torch.as_tensor(
                    np.asarray(stuff_plus_available, dtype=bool).copy(),
                    dtype=dtype,
                    device=device,
                )
                if stuff_plus_available is not None
                else None
            ),
        )
    result = pd.DataFrame(
        {
            "global_residual": output["global_residual"].detach().cpu().numpy(),
            "pitcher_correction": output["pitcher_correction"]
            .detach()
            .cpu()
            .numpy(),
            "predicted_residual": output["predicted_residual"]
            .detach()
            .cpu()
            .numpy(),
        }
    )
    if "ordinal_probabilities" in output:
        probabilities = output["ordinal_probabilities"].detach().cpu().numpy()
        result["prob_low"] = probabilities[:, 0]
        result["prob_middle"] = probabilities[:, 1]
        result["prob_high"] = probabilities[:, 2]
        result["predicted_class"] = probabilities.argmax(axis=1).astype(int)
        result["ordinal_score"] = output["ordinal_score"].detach().cpu().numpy()
    return result


def model_parameter_count(model: SharedTCN) -> int:
    require_torch()
    return int(sum(parameter.numel() for parameter in model.parameters()))


def config_as_dict(
    config: SharedTCNConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return asdict(SharedTCNConfig.from_value(config))
