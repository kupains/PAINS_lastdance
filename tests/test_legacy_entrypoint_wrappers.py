from __future__ import annotations

from unittest.mock import patch

import pytest

import search_xgboost_fast
import stuff_mlb_temporal_final


def test_search_legacy_cli_forwards_to_unified_selection() -> None:
    arguments = [
        "--statcast-dir",
        "statcast",
        "--stuff",
        "stuff.parquet",
        "--output-dir",
        "out",
    ]
    with patch("select_models.main", return_value=0) as unified:
        assert search_xgboost_fast.main(arguments) == 0
    forwarded = unified.call_args.args[0]
    assert forwarded[: len(arguments)] == arguments
    assert forwarded[-4:] == ["--models", "ewma", "ridge", "xgboost"]


def test_final_legacy_cli_forwards_to_locked_final() -> None:
    arguments = ["--locked-config", "lock.json", "--output-dir", "out"]
    with patch("final_evaluate.main", return_value=0) as unified:
        assert stuff_mlb_temporal_final.main(arguments) == 0
    unified.assert_called_once_with(arguments)


def test_programmatic_legacy_final_requires_a_lock() -> None:
    with pytest.raises(ValueError, match="locked_config_path is required"):
        stuff_mlb_temporal_final.run([], [], None, None)
