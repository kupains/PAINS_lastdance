from __future__ import annotations

import numpy as np
import pytest

from lib.hybrid_common_individual import (
    COMMON_SEQUENCE_FEATURES,
    INDIVIDUAL_RAW_CANDIDATES,
    HybridTabularConfig,
    HybridTCNConfig,
    fit_predict_hybrid_tcn,
    predict_hybrid_tabular,
    prepare_hybrid_fold_data,
)
from lib.shared_tcn import torch_available
from lib.stuff_cli import load_stuff_data
from lib.stuff_experiment import prepare_fold


def _fold_data():
    bundle = load_stuff_data(
        demo=True,
        demo_pitchers=3,
        demo_outings_per_season=8,
        train_start=2020,
        demo_seed=20260722,
    )
    fold = prepare_fold(
        bundle.outings, 2024, train_start=2020, evaluation_split="validation"
    )
    return prepare_hybrid_fold_data(fold, sequence_length=4)


def test_feature_partition_matches_requested_hybrid_contract():
    assert "offspeed_share" in COMMON_SEQUENCE_FEATURES
    assert "arm_angle" in COMMON_SEQUENCE_FEATURES
    assert "release_pos_y" in INDIVIDUAL_RAW_CANDIDATES
    assert "release_extension" in INDIVIDUAL_RAW_CANDIDATES
    assert not set(COMMON_SEQUENCE_FEATURES) & set(INDIVIDUAL_RAW_CANDIDATES)


def test_hybrid_fold_sequences_are_aligned_and_finite():
    data = _fold_data()
    assert data.train_common_sequence.shape[2] == len(COMMON_SEQUENCE_FEATURES)
    assert len(data.train_targets) == len(data.train_common_sequence)
    assert len(data.evaluation_targets) == len(data.evaluation_common_sequence)
    assert np.isfinite(data.train_common_sequence).all()
    assert np.isfinite(data.train_individual_tabular).all()


@pytest.mark.parametrize("model", ["ridge", "xgboost"])
def test_hybrid_tabular_returns_every_evaluation_row(model):
    data = _fold_data()
    prediction, fitted = predict_hybrid_tabular(
        data,
        HybridTabularConfig(
            model=model,
            sequence_length=4,
            common_n_estimators=5,
            individual_n_estimators=3,
            min_child_weight=1.0,
        ),
    )
    assert len(prediction) == len(data.evaluation_targets)
    assert prediction["predicted_stuff_plus"].notna().all()
    assert fitted["global_model"] is not None


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_hybrid_tcn_returns_shared_and_individual_components():
    data = _fold_data()
    prediction, model, history = fit_predict_hybrid_tcn(
        data,
        HybridTCNConfig(
            sequence_length=4,
            internal_channels=2,
            bottleneck_dim=2,
            dilations=(1, 2),
            epochs=2,
        ),
        device="cpu",
    )
    assert len(prediction) == len(data.evaluation_targets)
    assert prediction["global_residual"].notna().all()
    assert prediction["pitcher_correction"].notna().all()
    assert len(history) == 2
    assert model.individual_weight.num_embeddings == len(data.pitcher_mapping)
