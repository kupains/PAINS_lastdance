"""Common/individual hybrid Ridge, XGBoost, and TCN experiments.

The common branch uses the user-selected 13 prior-outing signals and learns
one population model.  Every other usable, prior-known signal is routed to a
small pitcher-specific correction branch.  Target-outing physical values are
never used; raw physical inputs are assembled only from strictly prior games.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from lib import stuff_mlb_dataset as dataset_contract
from lib.preprocessing import PerPitcherTrainImputerScaler
from lib.shared_tcn import (
    SharedTCNEncoder,
    build_pitcher_index,
    build_tcn_sequences,
    pitcher_balanced_mean,
    require_torch,
    set_deterministic_seed,
    torch,
)
from lib.stuff_experiment import PreparedFold
from lib.stuff_tabular import classify_stuff_plus_rows


COMMON_SEQUENCE_FEATURES = (
    "stuff_plus",
    "pitch_count",
    "rest_days_log",
    "long_gap",
    "season_start",
    "primary_fb_velocity",
    "primary_fb_spin",
    "primary_fb_pfx_x",
    "primary_fb_pfx_z",
    "arm_angle",
    "fastball_share",
    "breaking_share",
    "offspeed_share",
)

# Raw values are read only from games strictly before the prediction target.
# release_pos_y is intentionally included although the locked models omit it.
INDIVIDUAL_RAW_CANDIDATES = (
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "spin_axis_sin",
    "spin_axis_cos",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "outing_xwOBA",
    "known_pitch_share",
    "ff_count",
    "ff_velocity",
    "ff_spin",
    "ff_pfx_x",
    "ff_pfx_z",
    "si_count",
    "si_velocity",
    "si_spin",
    "si_pfx_x",
    "si_pfx_z",
    "fa_count",
    "fa_velocity",
    "fa_spin",
    "fa_pfx_x",
    "fa_pfx_z",
)

DERIVED_EXTRA_CANDIDATES = (
    "rest_days",
    "rest_days_capped",
    "prev_start_pitch_count",
    "workload_density_3starts",
    "pitcher_prior_outing_count",
    "prior_stuff_plus",
    "stuff_plus_lag1",
    "stuff_plus_mean_last5",
    "stuff_plus_prior_mean5",
    "stuff_plus_prior_std5",
    "stuff_plus_slope_last5",
    "stuff_plus_prior_slope5",
    "pitch_count_lag1",
    "pitch_count_prior_mean5",
    "prior_minus_ewma4",
    "mean5_minus_ewma4",
    "fastball_share_prior_mean5",
    "breaking_share_prior_mean5",
    "release_extension_prior_mean5",
    "release_pos_x_prior_mean5",
    "release_pos_y_prior_mean5",
    "release_pos_z_prior_mean5",
    "arm_angle_prior_mean5",
)

HYBRID_BINARY_FEATURES = ("long_gap", "season_start")


@dataclass(frozen=True)
class HybridTabularConfig:
    model: str = "ridge"
    sequence_length: int = 8
    global_ridge_alpha: float = 100.0
    individual_ridge_alpha: float = 100.0
    common_n_estimators: int = 100
    common_max_depth: int = 2
    common_learning_rate: float = 0.05
    individual_n_estimators: int = 25
    individual_max_depth: int = 1
    individual_learning_rate: float = 0.05
    min_child_weight: float = 8.0
    reg_lambda: float = 20.0
    reg_alpha: float = 0.5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    prediction_alpha: float = 0.25
    seed: int = 20260722


@dataclass(frozen=True)
class HybridTCNConfig:
    sequence_length: int = 8
    internal_channels: int = 4
    bottleneck_dim: int = 4
    kernel_size: int = 2
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    individual_l2: float = 0.1
    pitcher_bias_l2: float = 0.01
    huber_delta: float = 1.0
    learning_rate: float = 1e-3
    epochs: int = 200
    gradient_clip: float = 5.0
    prediction_alpha: float = 0.1
    seed: int = 20260722


@dataclass
class HybridFoldData:
    train_targets: pd.DataFrame
    evaluation_targets: pd.DataFrame
    train_common_sequence: np.ndarray
    evaluation_common_sequence: np.ndarray
    train_valid: np.ndarray
    evaluation_valid: np.ndarray
    train_primary_available: np.ndarray
    evaluation_primary_available: np.ndarray
    train_stuff_available: np.ndarray
    evaluation_stuff_available: np.ndarray
    train_common_tabular: np.ndarray
    evaluation_common_tabular: np.ndarray
    train_individual_tabular: np.ndarray
    evaluation_individual_tabular: np.ndarray
    train_pitcher_index: np.ndarray
    evaluation_pitcher_index: np.ndarray
    pitcher_mapping: dict[Any, int]
    common_feature_names: tuple[str, ...]
    individual_raw_feature_names: tuple[str, ...]
    individual_derived_feature_names: tuple[str, ...]
    individual_tabular_feature_names: tuple[str, ...]


def _eligible_targets(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame["target_eligible"].fillna(False).astype(bool)
    true_class = pd.to_numeric(frame["true_class"], errors="coerce")
    stuff = pd.to_numeric(frame["stuff_plus"], errors="coerce")
    ewma = pd.to_numeric(frame["ewma4"], errors="coerce")
    return frame.loc[eligible & true_class.isin([0, 1, 2]) & stuff.notna() & ewma.notna()].copy()


def _add_angle_components(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "spin_axis" in result:
        radians = np.deg2rad(pd.to_numeric(result["spin_axis"], errors="coerce"))
        result["spin_axis_sin"] = np.sin(radians)
        result["spin_axis_cos"] = np.cos(radians)
    return result


def _select_numeric_columns(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    minimum_finite: int = 10,
) -> tuple[str, ...]:
    selected: list[str] = []
    for column in candidates:
        if column not in frame:
            continue
        finite = np.isfinite(pd.to_numeric(frame[column], errors="coerce")).sum()
        if int(finite) >= int(minimum_finite):
            selected.append(column)
    return tuple(selected)


def _individual_raw_candidates(frame: pd.DataFrame) -> tuple[str, ...]:
    pitch_mix = tuple(sorted(column for column in frame if column.startswith("pitch_mix_")))
    return tuple(dict.fromkeys((*INDIVIDUAL_RAW_CANDIDATES, *pitch_mix)))


def _derived_candidates() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *getattr(dataset_contract, "FEATURES", ()),
                *getattr(dataset_contract, "COMPACT_FEATURES", ()),
                *DERIVED_EXTRA_CANDIDATES,
            )
        )
    )


def _slope(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n, _, p = values.shape
    output = np.zeros((n, p), dtype=np.float32)
    for row in range(n):
        mask = valid[row]
        if mask.sum() < 2:
            continue
        x = np.arange(mask.sum(), dtype=float)
        centered = x - x.mean()
        denominator = np.square(centered).sum()
        if denominator <= 0:
            continue
        selected = values[row, mask]
        output[row] = (centered[:, None] * (selected - selected.mean(0))).sum(0) / denominator
    return output


def sequence_summary(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return last, mean, std, and slope for each normalized sequence feature."""

    n, _, p = values.shape
    last = np.zeros((n, p), dtype=np.float32)
    mean = np.zeros((n, p), dtype=np.float32)
    std = np.zeros((n, p), dtype=np.float32)
    for row in range(n):
        mask = valid[row]
        if not mask.any():
            continue
        selected = values[row, mask]
        last[row] = selected[-1]
        mean[row] = selected.mean(0)
        std[row] = selected.std(0)
    return np.concatenate([last, mean, std, _slope(values, valid)], axis=1)


def _summary_names(features: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}_{stat}_{feature}"
        for stat in ("last", "mean", "std", "slope")
        for feature in features
    )


def prepare_hybrid_fold_data(
    fold: PreparedFold,
    *,
    sequence_length: int = 8,
    seed: int = 20260722,
) -> HybridFoldData:
    """Build fold-local common and individual inputs without target-game leakage."""

    history = _add_angle_components(fold.history)
    train_history = _add_angle_components(fold.train_history)
    train_targets = _eligible_targets(fold.train_targets).reset_index(drop=True)
    evaluation_targets = _eligible_targets(fold.evaluation_targets).reset_index(drop=True)
    _, pitcher_mapping = build_pitcher_index(train_targets["pitcher"])
    evaluation_targets = evaluation_targets.loc[
        evaluation_targets["pitcher"].isin(pitcher_mapping)
    ].reset_index(drop=True)
    if train_targets.empty or evaluation_targets.empty:
        raise ValueError("Hybrid fold requires train and evaluation targets")

    missing_common = [column for column in COMMON_SEQUENCE_FEATURES if column not in train_history]
    if missing_common:
        raise ValueError(f"Hybrid history is missing common features: {missing_common}")
    eligible_history = train_history.loc[
        train_history["history_eligible"].fillna(False).astype(bool)
    ].copy()
    individual_raw = _select_numeric_columns(
        eligible_history,
        _individual_raw_candidates(eligible_history),
    )
    sequence_features = COMMON_SEQUENCE_FEATURES + individual_raw
    preprocessor = PerPitcherTrainImputerScaler(
        sequence_features,
        binary_columns=HYBRID_BINARY_FEATURES,
    ).fit(eligible_history)
    train_batch = build_tcn_sequences(
        train_history,
        train_targets,
        feature_names=sequence_features,
        sequence_length=sequence_length,
        sequence_order="ordered",
        seed=seed,
        preprocessor=preprocessor,
        allow_all_pitch_features=True,
    )
    evaluation_batch = build_tcn_sequences(
        history,
        evaluation_targets,
        feature_names=sequence_features,
        sequence_length=sequence_length,
        sequence_order="ordered",
        seed=seed,
        preprocessor=preprocessor,
        allow_all_pitch_features=True,
    )
    train_batch.assert_strictly_prior()
    evaluation_batch.assert_strictly_prior()

    common_count = len(COMMON_SEQUENCE_FEATURES)
    train_common_sequence = train_batch.values[:, :, :common_count]
    evaluation_common_sequence = evaluation_batch.values[:, :, :common_count]
    train_individual_sequence = train_batch.values[:, :, common_count:]
    evaluation_individual_sequence = evaluation_batch.values[:, :, common_count:]
    train_common_tabular = sequence_summary(train_common_sequence, train_batch.valid_timestep)
    evaluation_common_tabular = sequence_summary(
        evaluation_common_sequence, evaluation_batch.valid_timestep
    )
    train_individual_sequence_summary = sequence_summary(
        train_individual_sequence, train_batch.valid_timestep
    )
    evaluation_individual_sequence_summary = sequence_summary(
        evaluation_individual_sequence, evaluation_batch.valid_timestep
    )

    individual_derived = _select_numeric_columns(train_targets, _derived_candidates())
    if individual_derived:
        derived_preprocessor = PerPitcherTrainImputerScaler(individual_derived).fit(
            train_targets
        )
        train_derived = derived_preprocessor.transform_features(train_targets).to_numpy(
            dtype=np.float32
        )
        evaluation_derived = derived_preprocessor.transform_features(
            evaluation_targets
        ).to_numpy(dtype=np.float32)
    else:
        train_derived = np.empty((len(train_targets), 0), dtype=np.float32)
        evaluation_derived = np.empty((len(evaluation_targets), 0), dtype=np.float32)
    train_individual_tabular = np.concatenate(
        [train_individual_sequence_summary, train_derived], axis=1
    )
    evaluation_individual_tabular = np.concatenate(
        [evaluation_individual_sequence_summary, evaluation_derived], axis=1
    )
    individual_names = (
        *_summary_names(individual_raw, "individual"),
        *individual_derived,
    )
    train_pitcher_index, _ = build_pitcher_index(train_targets["pitcher"], pitcher_mapping)
    evaluation_pitcher_index, _ = build_pitcher_index(
        evaluation_targets["pitcher"], pitcher_mapping
    )
    return HybridFoldData(
        train_targets=train_targets,
        evaluation_targets=evaluation_targets,
        train_common_sequence=train_common_sequence,
        evaluation_common_sequence=evaluation_common_sequence,
        train_valid=train_batch.valid_timestep,
        evaluation_valid=evaluation_batch.valid_timestep,
        train_primary_available=train_batch.primary_fb_available,
        evaluation_primary_available=evaluation_batch.primary_fb_available,
        train_stuff_available=train_batch.stuff_plus_available,
        evaluation_stuff_available=evaluation_batch.stuff_plus_available,
        train_common_tabular=train_common_tabular,
        evaluation_common_tabular=evaluation_common_tabular,
        train_individual_tabular=train_individual_tabular,
        evaluation_individual_tabular=evaluation_individual_tabular,
        train_pitcher_index=train_pitcher_index,
        evaluation_pitcher_index=evaluation_pitcher_index,
        pitcher_mapping=pitcher_mapping,
        common_feature_names=_summary_names(COMMON_SEQUENCE_FEATURES, "common"),
        individual_raw_feature_names=individual_raw,
        individual_derived_feature_names=individual_derived,
        individual_tabular_feature_names=tuple(individual_names),
    )


def _balanced_sample_weight(pitcher_index: np.ndarray) -> np.ndarray:
    counts = pd.Series(pitcher_index).value_counts().to_dict()
    n_pitchers = max(len(counts), 1)
    n_rows = len(pitcher_index)
    return np.asarray(
        [n_rows / (n_pitchers * counts[int(value)]) for value in pitcher_index],
        dtype=float,
    )


def _new_tabular_estimator(config: HybridTabularConfig, *, individual: bool):
    if config.model == "ridge":
        return Ridge(
            alpha=(config.individual_ridge_alpha if individual else config.global_ridge_alpha)
        )
    if config.model != "xgboost":
        raise ValueError("Hybrid tabular model must be 'ridge' or 'xgboost'")
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=(config.individual_n_estimators if individual else config.common_n_estimators),
        max_depth=(config.individual_max_depth if individual else config.common_max_depth),
        learning_rate=(
            config.individual_learning_rate if individual else config.common_learning_rate
        ),
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective="reg:squarederror",
        tree_method="hist",
        max_bin=64,
        n_jobs=1,
        random_state=config.seed,
        verbosity=0,
    )


def _prediction_frame(
    targets: pd.DataFrame,
    predicted_residual: np.ndarray,
    *,
    alpha: float,
    model: str,
    candidate_id: str,
    global_residual: np.ndarray,
    individual_correction: np.ndarray,
) -> pd.DataFrame:
    keep = [
        "row_id", "pitcher", "game_pk", "game_date", "year",
        "target_eligible", "history_eligible", "stuff_plus", "target_y",
        "ewma4", "q33", "q67", "true_class",
    ]
    result = targets[keep].copy().reset_index(drop=True)
    result["true_stuff_plus"] = pd.to_numeric(result["stuff_plus"], errors="coerce")
    result["true_residual"] = result["true_stuff_plus"] - pd.to_numeric(
        result["ewma4"], errors="coerce"
    )
    result["global_residual"] = np.asarray(global_residual, dtype=float)
    result["pitcher_correction"] = np.asarray(individual_correction, dtype=float)
    result["predicted_residual"] = np.asarray(predicted_residual, dtype=float)
    result["predicted_stuff_plus"] = (
        pd.to_numeric(result["ewma4"], errors="coerce")
        + float(alpha) * result["predicted_residual"]
    )
    result["predicted_class"] = classify_stuff_plus_rows(
        result["predicted_stuff_plus"], result["q33"], result["q67"]
    )
    result["model"] = model
    result["candidate_id"] = candidate_id
    result["feature_eligible"] = np.isfinite(result["predicted_stuff_plus"])
    result["deployable"] = np.isfinite(result["predicted_class"])
    return result


def predict_hybrid_tabular(
    fold_data: HybridFoldData,
    config: HybridTabularConfig | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config if isinstance(config, HybridTabularConfig) else HybridTabularConfig(**dict(config or {}))
    y = (
        pd.to_numeric(fold_data.train_targets["stuff_plus"], errors="coerce")
        - pd.to_numeric(fold_data.train_targets["ewma4"], errors="coerce")
    ).to_numpy(dtype=float)
    global_model = _new_tabular_estimator(cfg, individual=False)
    global_model.fit(
        fold_data.train_common_tabular,
        y,
        sample_weight=_balanced_sample_weight(fold_data.train_pitcher_index),
    )
    global_train = np.asarray(global_model.predict(fold_data.train_common_tabular), dtype=float)
    global_eval = np.asarray(
        global_model.predict(fold_data.evaluation_common_tabular), dtype=float
    )
    remaining = y - global_train
    individual_eval = np.zeros(len(fold_data.evaluation_targets), dtype=float)
    individual_models: dict[int, Any] = {}
    for pitcher_index in sorted(np.unique(fold_data.train_pitcher_index)):
        train_mask = fold_data.train_pitcher_index == pitcher_index
        eval_mask = fold_data.evaluation_pitcher_index == pitcher_index
        if train_mask.sum() < 12 or not eval_mask.any():
            continue
        model = _new_tabular_estimator(cfg, individual=True)
        model.fit(fold_data.train_individual_tabular[train_mask], remaining[train_mask])
        individual_eval[eval_mask] = model.predict(
            fold_data.evaluation_individual_tabular[eval_mask]
        )
        individual_models[int(pitcher_index)] = model
    total = global_eval + individual_eval
    candidate = (
        f"hybrid_{cfg.model}_l{cfg.sequence_length}_a{cfg.prediction_alpha:g}_"
        f"common{fold_data.train_common_tabular.shape[1]}_"
        f"individual{fold_data.train_individual_tabular.shape[1]}"
    )
    prediction = _prediction_frame(
        fold_data.evaluation_targets,
        total,
        alpha=cfg.prediction_alpha,
        model=f"hybrid_{cfg.model}",
        candidate_id=candidate,
        global_residual=global_eval,
        individual_correction=individual_eval,
    )
    return prediction, {"global_model": global_model, "individual_models": individual_models}


if torch is not None:
    from torch import nn
    from torch.nn import functional as F

    class CommonIndividualTCN(nn.Module):
        def __init__(self, num_pitchers: int, individual_dim: int, config: HybridTCNConfig) -> None:
            super().__init__()
            self.config = config
            self.encoder = SharedTCNEncoder(
                input_dim=len(COMMON_SEQUENCE_FEATURES),
                internal_channels=config.internal_channels,
                bottleneck_dim=config.bottleneck_dim,
                kernel_size=config.kernel_size,
                dilations=config.dilations,
                dropout=config.dropout,
            )
            self.global_head = nn.Linear(config.bottleneck_dim, 1)
            self.individual_weight = nn.Embedding(num_pitchers, individual_dim)
            self.pitcher_bias = nn.Embedding(num_pitchers, 1)
            nn.init.zeros_(self.individual_weight.weight)
            nn.init.zeros_(self.pitcher_bias.weight)

        def forward(
            self,
            common_values,
            individual_values,
            valid,
            pitcher_index,
            primary_available,
            stuff_available,
        ):
            availability = torch.stack([primary_available, stuff_available], dim=-1)
            representation = self.encoder(common_values, valid, availability=availability)
            global_residual = self.global_head(representation).squeeze(1)
            individual = (
                self.individual_weight(pitcher_index) * individual_values
            ).sum(1)
            individual = individual + self.pitcher_bias(pitcher_index).squeeze(1)
            return global_residual, individual, global_residual + individual


def _tensor(values: np.ndarray, *, dtype=None, device="cpu"):
    require_torch()
    return torch.as_tensor(values, dtype=dtype, device=device)


def fit_predict_hybrid_tcn(
    fold_data: HybridFoldData,
    config: HybridTCNConfig | Mapping[str, Any] | None = None,
    *,
    device: str = "cpu",
) -> tuple[pd.DataFrame, Any, pd.DataFrame]:
    require_torch()
    cfg = config if isinstance(config, HybridTCNConfig) else HybridTCNConfig(**dict(config or {}))
    set_deterministic_seed(cfg.seed)
    train_common = _tensor(fold_data.train_common_sequence, dtype=torch.float32, device=device)
    train_individual = _tensor(
        fold_data.train_individual_tabular, dtype=torch.float32, device=device
    )
    train_valid = _tensor(fold_data.train_valid, dtype=torch.bool, device=device)
    train_pitcher = _tensor(fold_data.train_pitcher_index, dtype=torch.long, device=device)
    train_primary = _tensor(
        fold_data.train_primary_available, dtype=torch.bool, device=device
    )
    train_stuff = _tensor(
        fold_data.train_stuff_available, dtype=torch.bool, device=device
    )
    true_residual = _tensor(
        (
            pd.to_numeric(fold_data.train_targets["stuff_plus"], errors="coerce")
            - pd.to_numeric(fold_data.train_targets["ewma4"], errors="coerce")
        ).to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    model = CommonIndividualTCN(
        len(fold_data.pitcher_mapping),
        fold_data.train_individual_tabular.shape[1],
        cfg,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        global_residual, individual, predicted = model(
            train_common,
            train_individual,
            train_valid,
            train_pitcher,
            train_primary,
            train_stuff,
        )
        row_loss = F.huber_loss(
            predicted, true_residual, reduction="none", delta=cfg.huber_delta
        )
        residual_loss = pitcher_balanced_mean(row_loss, train_pitcher)
        individual_penalty = cfg.individual_l2 * model.individual_weight.weight.square().sum()
        bias_penalty = cfg.pitcher_bias_l2 * model.pitcher_bias.weight.square().sum()
        loss = residual_loss + individual_penalty + bias_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()
        if epoch == 0 or epoch == cfg.epochs - 1 or (epoch + 1) % 25 == 0:
            history.append({
                "epoch": epoch + 1,
                "loss": float(loss.detach().cpu()),
                "residual_loss": float(residual_loss.detach().cpu()),
            })

    eval_common = _tensor(
        fold_data.evaluation_common_sequence, dtype=torch.float32, device=device
    )
    eval_individual = _tensor(
        fold_data.evaluation_individual_tabular, dtype=torch.float32, device=device
    )
    eval_valid = _tensor(fold_data.evaluation_valid, dtype=torch.bool, device=device)
    eval_pitcher = _tensor(
        fold_data.evaluation_pitcher_index, dtype=torch.long, device=device
    )
    eval_primary = _tensor(
        fold_data.evaluation_primary_available, dtype=torch.bool, device=device
    )
    eval_stuff = _tensor(
        fold_data.evaluation_stuff_available, dtype=torch.bool, device=device
    )
    model.eval()
    with torch.no_grad():
        global_residual, individual, predicted = model(
            eval_common,
            eval_individual,
            eval_valid,
            eval_pitcher,
            eval_primary,
            eval_stuff,
        )
    global_np = global_residual.cpu().numpy()
    individual_np = individual.cpu().numpy()
    candidate = (
        f"hybrid_tcn_l{cfg.sequence_length}_c{cfg.internal_channels}_b"
        f"{cfg.bottleneck_dim}_l2{cfg.individual_l2:g}_a{cfg.prediction_alpha:g}_"
        f"individual{fold_data.train_individual_tabular.shape[1]}"
    )
    prediction = _prediction_frame(
        fold_data.evaluation_targets,
        predicted.cpu().numpy(),
        alpha=cfg.prediction_alpha,
        model="hybrid_tcn",
        candidate_id=candidate,
        global_residual=global_np,
        individual_correction=individual_np,
    )
    return prediction, model, pd.DataFrame(history)


def with_seed(config: HybridTCNConfig, seed: int) -> HybridTCNConfig:
    return replace(config, seed=int(seed))
