from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from lib.build_outings import build_outings


FASTBALL = {"FF", "SI", "FC", "FA"}
BREAKING = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED = {"CH", "FS", "FO", "SC"}
PHYSICAL_RAW = [
    "release_speed", "release_spin_rate", "release_extension", "release_pos_x",
    "release_pos_z", "arm_angle", "pfx_x", "pfx_z",
]
FEATURES = [
    "prev_start_pitch_count", "rest_days", "workload_density_3starts",
    "prior_stuff_plus", "stuff_plus_mean_last5", "stuff_plus_slope_last5",
    *[value for column in PHYSICAL_RAW for value in (f"{column}_ma5", f"{column}_slope5")],
    "spin_axis_sin_ma5", "spin_axis_cos_ma5",
    "breaking_share_ma5", "breaking_share_slope5",
    "offspeed_share_ma5", "offspeed_share_slope5",
]


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[mask]
    y = values[mask]
    x -= x.mean()
    denominator = np.sum(x * x)
    return float(np.sum(x * (y - y.mean())) / denominator) if denominator else np.nan


def _add_history(data: pd.DataFrame, column: str, add_slope: bool = True) -> None:
    data[f"{column}_ma5"] = np.nan
    if add_slope:
        data[f"{column}_slope5"] = np.nan
    for _, group in data.groupby("pitcher", sort=False):
        group = group.sort_values("game_date")
        prior = pd.to_numeric(group[column], errors="coerce").shift(1)
        data.loc[group.index, f"{column}_ma5"] = prior.rolling(5, min_periods=1).mean()
        if add_slope:
            data.loc[group.index, f"{column}_slope5"] = prior.rolling(5, min_periods=2).apply(
                _slope, raw=True
            )


def _one_pitcher_outings(path: Path) -> pd.DataFrame:
    pitches = pd.read_parquet(path)
    if pitches.empty:
        return pd.DataFrame()
    pitches = pitches.loc[pitches["game_type"].astype("string").str.upper().eq("R")].copy()
    if pitches.empty:
        return pd.DataFrame()
    pitches["game_date"] = pd.to_datetime(pitches["game_date"], errors="coerce")
    pitch_type = pitches["pitch_type"].astype("string").str.upper()
    pitches["pitch_category"] = np.select(
        [pitch_type.isin(FASTBALL), pitch_type.isin(BREAKING), pitch_type.isin(OFFSPEED)],
        ["fastball", "breaking", "offspeed"], default="other",
    )
    angle = np.deg2rad(pd.to_numeric(pitches["spin_axis"], errors="coerce"))
    pitches["spin_axis_sin"] = np.sin(angle)
    pitches["spin_axis_cos"] = np.cos(angle)
    keys = ["pitcher", "game_pk", "game_date"]

    outings = build_outings(pitches)
    circular = pitches.groupby(keys, as_index=False).agg(
        spin_axis_sin=("spin_axis_sin", "mean"),
        spin_axis_cos=("spin_axis_cos", "mean"),
    )
    categorized = pitches.loc[pitches["pitch_category"].ne("other")]
    category_counts = categorized.groupby(keys + ["pitch_category"]).size().unstack(fill_value=0)
    for category in ["fastball", "breaking", "offspeed"]:
        if category not in category_counts:
            category_counts[category] = 0
    denominator = category_counts[["fastball", "breaking", "offspeed"]].sum(axis=1)
    category_counts["breaking_share"] = category_counts["breaking"] / denominator
    category_counts["offspeed_share"] = category_counts["offspeed"] / denominator
    shares = category_counts[["breaking_share", "offspeed_share"]].reset_index()
    return outings.merge(circular, on=keys, how="left").merge(shares, on=keys, how="left")


def _as_paths(value: Path | Sequence[Path]) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    return [Path(path) for path in value]


def build_merged_outings(
    statcast_dir: Path | Sequence[Path], stuff_path: Path | Sequence[Path]
) -> pd.DataFrame:
    """Join Statcast outings to FanGraphs starts and retain 50+ pitch outings."""
    logs = pd.concat(
        [pd.read_parquet(path) for path in _as_paths(stuff_path)], ignore_index=True
    )
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs.drop_duplicates(["pitcher", "game_date"], keep="last")

    statcast_paths = [
        path
        for source in _as_paths(statcast_dir)
        for path in ([source] if source.is_file() else sorted(source.glob("*.parquet")))
    ]
    frames = [
        frame
        for path in statcast_paths
        if not (frame := _one_pitcher_outings(path)).empty
    ]
    if not frames:
        raise FileNotFoundError(f"No Statcast parquet files in {_as_paths(statcast_dir)}")
    outings = pd.concat(frames, ignore_index=True)
    outings["game_date"] = pd.to_datetime(outings["game_date"])
    outings = outings.merge(
        logs[["pitcher", "game_date", "sp_stuff"]], on=["pitcher", "game_date"], how="inner"
    )
    outings = outings.loc[pd.to_numeric(outings["pitch_count"], errors="coerce").ge(50)].copy()
    outings = outings.sort_values(["pitcher", "game_date", "game_pk"]).drop_duplicates(
        ["pitcher", "game_date"], keep="last"
    ).reset_index(drop=True)

    return outings


def make_dataset(
    statcast_dir: Path | Sequence[Path], stuff_path: Path | Sequence[Path]
) -> tuple[pd.DataFrame, list[int]]:
    """Build the candidate dataset while qualifying pitchers on 2021-2025 only."""
    outings = build_merged_outings(statcast_dir, stuff_path)
    counts = outings.groupby(["pitcher", outings["game_date"].dt.year]).size().unstack(fill_value=0)
    for year in range(2021, 2026):
        if year not in counts:
            counts[year] = 0
    qualified = sorted(
        int(value)
        for value in counts.index[(counts[list(range(2021, 2026))] >= 20).all(axis=1)]
    )
    data = outings.loc[outings["pitcher"].isin(qualified)].copy()
    data = data.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    data["year"] = data["game_date"].dt.year
    grouped = data.groupby(["pitcher", "year"], sort=False)
    data["prev_start_pitch_count"] = grouped["pitch_count"].shift(1)
    data["rest_days"] = grouped["game_date"].diff().dt.days
    prior_sum = sum(grouped["pitch_count"].shift(offset) for offset in (1, 2, 3))
    oldest_date = grouped["game_date"].shift(3)
    elapsed_days = (data["game_date"] - oldest_date).dt.days
    max_gap = grouped["rest_days"].transform(lambda values: values.rolling(3, min_periods=3).max())
    data["workload_density_3starts"] = (prior_sum / elapsed_days).where(max_gap.le(21))

    for column in PHYSICAL_RAW + ["breaking_share", "offspeed_share"]:
        _add_history(data, column)
    _add_history(data, "spin_axis_sin", add_slope=False)
    _add_history(data, "spin_axis_cos", add_slope=False)

    data["prior_stuff_plus"] = np.nan
    data["stuff_plus_mean_last5"] = np.nan
    data["stuff_plus_slope_last5"] = np.nan
    for _, group in data.groupby("pitcher", sort=False):
        group = group.sort_values("game_date")
        prior = pd.to_numeric(group["sp_stuff"], errors="coerce").shift(1)
        data.loc[group.index, "prior_stuff_plus"] = prior
        data.loc[group.index, "stuff_plus_mean_last5"] = prior.rolling(5, min_periods=1).mean()
        data.loc[group.index, "stuff_plus_slope_last5"] = prior.rolling(5, min_periods=2).apply(
            _slope, raw=True
        )
    data["target_y"] = pd.to_numeric(data["sp_stuff"], errors="coerce")
    return data, qualified
