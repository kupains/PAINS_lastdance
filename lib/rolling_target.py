# %%
"""Forward rolling residual target for denoised within-pitcher prediction.

The single-outing residual is ~99% sampling noise at BF~4, so a single-outing
class label is nearly unpredictable. Averaging a pitcher's residual over the
current + next (k-1) outings cancels most of that noise while preserving the
"vs own baseline" meaning, giving a target with real predictable signal.

Design:
- Forward window [t .. t+k-1] so the label is a genuine forecast made at
  decision time t (trailing windows are already known at t, so they are
  features, not targets).
- Windows are confined to a single season (an off-season gap must not sit
  inside a "stretch").
- Only complete k-outing windows are kept; incomplete tail windows -> NaN.
- A leakage-safe debiased variant subtracts the pitcher's prior-only expanding
  mean residual (offset uses only outings strictly before t, so it never
  overlaps the forward window).
- `target_window_end_date` (date of outing t+k-1) is emitted so the training
  split can embargo rows whose window peeks past the train/test boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# %%
def rolling_columns(k: int) -> dict[str, str]:
    """Canonical column names for a given window size."""
    return {
        "raw": f"rolling_fwd{k}_residual",
        "centered": f"rolling_fwd{k}_residual_centered",
        "offset": "pitcher_offset_prior",
        "window_end": f"rolling_fwd{k}_window_end_date",
    }


# %%
def add_forward_rolling_target(
    df: pd.DataFrame,
    k: int = 5,
    residual_col: str = "residual",
    offset_shrink_k: float = 10.0,
) -> pd.DataFrame:
    """Add forward rolling residual target columns for window size k.

    Returns a copy sorted by (pitcher, game_date, game_pk) with:
      rolling_fwd{k}_residual           mean residual over [t .. t+k-1]
      rolling_fwd{k}_residual_centered  raw minus prior-only shrunk offset
      pitcher_offset_prior              prior-only shrunk expanding mean residual
      rolling_fwd{k}_window_end_date    game_date of outing t+k-1

    The offset is shrunk by n/(n+offset_shrink_k): a single outing's residual
    has sd ~0.24, so an unshrunk prior mean over few outings would inject more
    noise than the true (sd ~0.03) offset it removes. Shrinkage toward 0 (the
    league-neutral residual) keeps low-history offsets from over-subtracting.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    cols = rolling_columns(k)
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if "game_pk" not in out.columns:
        out["game_pk"] = np.arange(len(out))
    out = out.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    out["season"] = out["game_date"].dt.year
    resid = pd.to_numeric(out[residual_col], errors="coerce")

    fwd = pd.Series(np.nan, index=out.index, dtype=float)
    window_end = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    # Forward mean within (pitcher, season): reverse -> trailing rolling -> reverse.
    # Requiring the window to contain no NaN keeps only complete, valid windows.
    for _, group in out.groupby(["pitcher", "season"], sort=False):
        idx = group.index
        r = resid.loc[idx]
        finite = r.notna().astype(float)
        rev = r.iloc[::-1]
        fwd_mean = rev.rolling(window=k, min_periods=k).mean().iloc[::-1]
        finite_count = finite.iloc[::-1].rolling(window=k, min_periods=k).sum().iloc[::-1]
        fwd_mean = fwd_mean.where(finite_count >= k)  # drop windows with any NaN residual
        fwd.loc[idx] = fwd_mean.to_numpy()
        window_end.loc[idx] = out.loc[idx, "game_date"].shift(-(k - 1)).to_numpy()

    # Career prior-only shrunk offset (may span seasons; strictly before t).
    offset = pd.Series(np.nan, index=out.index, dtype=float)
    for _, group in out.groupby("pitcher", sort=False):
        idx = group.index
        prior = resid.loc[idx].shift(1)
        n = prior.expanding(min_periods=1).count()
        mean = prior.expanding(min_periods=1).mean()
        reliability = n / (n + offset_shrink_k)
        offset.loc[idx] = (reliability * mean).to_numpy()

    out[cols["raw"]] = fwd
    out[cols["window_end"]] = window_end
    out[cols["offset"]] = offset.fillna(0.0)
    out[cols["centered"]] = fwd - out[cols["offset"]]
    # Blank the window-end date wherever the target is undefined so the embargo
    # filter never keeps a row on a stale/incomplete window.
    out.loc[out[cols["raw"]].isna(), cols["window_end"]] = pd.NaT
    return out


# %%
def slice_train_test(
    df: pd.DataFrame,
    target_col: str,
    window_end_col: str,
    test_start: pd.Timestamp,
    embargo_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test slice with forward-window leakage embargo.

    Train keeps only rows whose full forward window ends strictly before the
    embargo boundary, so no training label borrows a test-period outing. Test
    keeps rows starting at test_start with a complete window. Rows in the
    embargo gap, and rows with an undefined target, are dropped from both.
    """
    game_date = pd.to_datetime(df["game_date"], errors="coerce")
    window_end = pd.to_datetime(df[window_end_col], errors="coerce")
    has_target = df[target_col].notna()

    train_mask = has_target & (window_end < embargo_start)
    test_mask = has_target & (game_date >= test_start)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()
