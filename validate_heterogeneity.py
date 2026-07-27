# %%
"""Premise validation for response-profile personalization.

Question: is the pitcher-specific workload->performance response
(a) actually heterogeneous across pitchers and (b) stable over time?

Design: per pitcher, fit a shrunk ridge regression of residual on a small set
of workload features, separately on two halves of that pitcher's outings
(odd/even split = estimation-reliability ceiling, chronological split =
temporal stability). Then measure split-half agreement of the personal
deviation from the pooled response, and whether personal coefficients from
one half predict the other half better than the pooled coefficients alone.

If split-half agreement is ~0, response-profile clustering (and any
personalized response model) would be fitting noise.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from compare import filter_pitcher_sample
from lib.data_prep import prepare_features
from lib.labeling import create_labels


PREDICTORS = [
    "ACWR",
    "rest_days_capped",
    "back_to_back",
    "standard_abuse_sum_7d",
    "release_speed_z",
]
BINARY_PREDICTORS = {"back_to_back"}
REST_DAYS_CAP = 14.0
POOLED_RIDGE_LAMBDA = 1.0

# Tracking columns actually needed by this diagnostic. Dropping the rest before
# prepare_features() skips ~75 per-column rolling loops; each column's rolling
# features are computed independently, so trimming does not change the values
# of the columns we keep.
FAST_KEEP_TRACKING = {"release_speed"}


# %%
def _load_labeled(
    input_path: Path,
    cache_path: Path | None,
    min_pitcher_ip: float,
    min_pitcher_bf: int,
    fast: bool,
) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        print(f"[cache] loading labeled data from {cache_path}")
        return pd.read_parquet(cache_path)

    raw = pd.read_parquet(input_path)
    raw = filter_pitcher_sample(raw, min_bf=min_pitcher_bf, min_ip=min_pitcher_ip)

    if fast:
        from lib.data_prep import TRACKING_COLUMNS

        drop = [c for c in raw.columns if c.startswith("pitch_mix_")]
        drop += [c for c in TRACKING_COLUMNS if c in raw.columns and c not in FAST_KEEP_TRACKING]
        raw = raw.drop(columns=drop)

    features = prepare_features(raw)
    labeled = create_labels(features)
    labeled = labeled.loc[pd.to_numeric(labeled["residual"], errors="coerce").notna()].copy()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        labeled.to_parquet(cache_path, index=False)
        print(f"[cache] wrote {cache_path}")
    return labeled


# %%
def build_analysis_frame(labeled: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = labeled.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["season"] = df["game_date"].dt.year
    df["rest_days_capped"] = pd.to_numeric(df["rest_days"], errors="coerce").clip(upper=REST_DAYS_CAP)

    total_rows = len(df)
    capped_share = float((pd.to_numeric(df["rest_days"], errors="coerce") > REST_DAYS_CAP).mean())

    needed = PREDICTORS + ["residual"]
    nan_counts = {col: int(df[col].isna().sum()) for col in needed}
    frame = df[["pitcher", "game_date", "season", "BF"] + needed].dropna(subset=needed).copy()
    frame = frame.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    coverage = {
        "rows_labeled": int(total_rows),
        "rows_complete_case": int(len(frame)),
        "complete_case_share": round(len(frame) / total_rows, 4),
        "nan_counts": nan_counts,
        "rest_days_capped_share": round(capped_share, 4),
        "pitchers": int(frame["pitcher"].nunique()),
    }
    return frame, coverage


# %%
def preflight_checks(frame: pd.DataFrame) -> dict:
    """Identifiability and collinearity checks that gate interpretation."""
    within_std = {}
    zero_var_share = {}
    for col in PREDICTORS:
        stds = frame.groupby("pitcher")[col].std(ddof=0)
        within_std[col] = {
            "median": round(float(stds.median()), 4),
            "p10": round(float(stds.quantile(0.10)), 4),
        }
        zero_var_share[col] = round(float((stds == 0).mean()), 4)

    centered = frame.copy()
    for col in PREDICTORS:
        centered[col] = centered[col] - centered.groupby("pitcher")[col].transform("mean")
    corr = centered[PREDICTORS].corr().round(3)

    return {
        "within_pitcher_std": within_std,
        "zero_variance_pitcher_share": zero_var_share,
        "within_pitcher_correlation": corr.to_dict(),
    }


# %%
def _assign_halves(frame: pd.DataFrame, design: str) -> pd.Series:
    """0/1 half assignment per pitcher, rows assumed date-sorted."""
    rank = frame.groupby("pitcher").cumcount()
    if design == "odd_even":
        return (rank % 2).rename("half")
    if design == "chrono":
        n = frame.groupby("pitcher")["pitcher"].transform("size")
        return (rank >= (n // 2)).astype(int).rename("half")
    raise ValueError(f"Unknown design: {design}")


# %%
def _standardize_within(x: pd.DataFrame) -> np.ndarray:
    """Within-group standardization; binary predictors centered only."""
    out = np.zeros((len(x), len(PREDICTORS)), dtype=float)
    for j, col in enumerate(PREDICTORS):
        values = x[col].to_numpy(dtype=float)
        centered = values - values.mean()
        if col in BINARY_PREDICTORS:
            out[:, j] = centered
            continue
        std = values.std()
        out[:, j] = centered / std if std > 0 else 0.0
    return out


# %%
def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    p = x.shape[1]
    return np.linalg.solve(x.T @ x + lam * np.eye(p), x.T @ y)


# %%
def fit_half(frame_half: pd.DataFrame, lam: float, min_rows: int) -> dict:
    """Pooled response + per-pitcher shrunk deviations for one half."""
    blocks = []
    for pitcher, group in frame_half.groupby("pitcher", sort=False):
        if len(group) < min_rows:
            continue
        xs = _standardize_within(group[PREDICTORS])
        y = group["residual"].to_numpy(dtype=float)
        blocks.append((pitcher, xs, y - y.mean(), float(y.mean()), float(y.std())))

    if not blocks:
        return {"pitchers": [], "beta_global": np.zeros(len(PREDICTORS))}

    x_all = np.vstack([b[1] for b in blocks])
    y_all = np.concatenate([b[2] for b in blocks])
    beta_global = _ridge(x_all, y_all, POOLED_RIDGE_LAMBDA)

    records = {}
    for pitcher, xs, yc, y_mean, y_std in blocks:
        delta = _ridge(xs, yc - xs @ beta_global, lam)
        active = xs.std(axis=0) > 0
        records[pitcher] = {
            "delta": delta,
            "active": active,
            "xs": xs,
            "yc": yc,
            "y_mean": y_mean,
            "y_std": y_std,
        }
    return {"pitchers": records, "beta_global": beta_global}


# %%
def _fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n < 4 or not np.isfinite(r):
        return (float("nan"), float("nan"))
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    half = 1.959964 / np.sqrt(n - 3)
    return (float(np.tanh(z - half)), float(np.tanh(z + half)))


# %%
def _sign_test_p(n_better: int, n_total: int) -> float:
    """Two-sided binomial sign test via normal approximation."""
    if n_total == 0:
        return float("nan")
    z = (n_better - n_total / 2.0) / np.sqrt(n_total / 4.0)
    from math import erf

    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / np.sqrt(2.0)))))


# %%
def evaluate_design(frame: pd.DataFrame, design: str, lam: float, min_rows: int) -> dict:
    halves = _assign_halves(frame, design)
    fit0 = fit_half(frame[halves == 0], lam, min_rows)
    fit1 = fit_half(frame[halves == 1], lam, min_rows)
    common = sorted(set(fit0["pitchers"]) & set(fit1["pitchers"]))

    result = {
        "design": design,
        "lambda": lam,
        "min_rows_per_half": min_rows,
        "n_pitchers": len(common),
        "beta_global_half0": {c: round(float(v), 5) for c, v in zip(PREDICTORS, fit0["beta_global"])},
        "beta_global_half1": {c: round(float(v), 5) for c, v in zip(PREDICTORS, fit1["beta_global"])},
    }
    if len(common) < 10:
        result["error"] = "too few pitchers with enough rows in both halves"
        return result

    # --- coefficient-level split-half agreement of personal deviations ---
    coef_stability = {}
    rs = []
    for j, col in enumerate(PREDICTORS):
        pairs = [
            (fit0["pitchers"][p]["delta"][j], fit1["pitchers"][p]["delta"][j])
            for p in common
            if fit0["pitchers"][p]["active"][j] and fit1["pitchers"][p]["active"][j]
        ]
        if len(pairs) < 10:
            coef_stability[col] = {"r": None, "n": len(pairs)}
            continue
        d0, d1 = np.array(pairs).T
        r = float(np.corrcoef(d0, d1)[0, 1])
        lo, hi = _fisher_ci(r, len(pairs))
        signal_var = max(float(np.cov(d0, d1)[0, 1]), 0.0)
        total_var = float((d0.var(ddof=1) + d1.var(ddof=1)) / 2.0)
        coef_stability[col] = {
            "r": round(r, 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "n": len(pairs),
            "signal_sd_xwoba": round(float(np.sqrt(signal_var)), 5),
            "noise_sd_xwoba": round(float(np.sqrt(max(total_var - signal_var, 0.0))), 5),
        }
        rs.append(r)
    result["coef_stability"] = coef_stability
    result["mean_coef_r"] = round(float(np.mean(rs)), 3) if rs else None

    # --- prediction transfer: do half-A personal deltas help predict half B? ---
    per_pitcher_gain = []
    sse = {"global": 0.0, "personal": 0.0, "zero": 0.0}
    for p in common:
        for src, dst in ((fit0, fit1), (fit1, fit0)):
            xs, yc = dst["pitchers"][p]["xs"], dst["pitchers"][p]["yc"]
            pred_global = xs @ src["beta_global"]
            pred_personal = xs @ (src["beta_global"] + src["pitchers"][p]["delta"])
            mse_g = float(np.mean((yc - pred_global) ** 2))
            mse_p = float(np.mean((yc - pred_personal) ** 2))
            per_pitcher_gain.append(mse_g - mse_p)
            sse["global"] += float(np.sum((yc - pred_global) ** 2))
            sse["personal"] += float(np.sum((yc - pred_personal) ** 2))
            sse["zero"] += float(np.sum(yc**2))

    gains = np.array(per_pitcher_gain)
    n_better = int((gains > 0).sum())
    result["prediction_transfer"] = {
        "pooled_r2_global_vs_zero": round(1.0 - sse["global"] / sse["zero"], 5),
        "pooled_r2_personal_vs_zero": round(1.0 - sse["personal"] / sse["zero"], 5),
        "personal_mse_improvement_vs_global": round(1.0 - sse["personal"] / sse["global"], 5),
        "share_pitcher_halves_improved": round(n_better / len(gains), 3),
        "sign_test_p": round(_sign_test_p(n_better, len(gains)), 5),
    }

    # --- label-level reliability (noise floor / baseline drift) ---
    means0 = np.array([fit0["pitchers"][p]["y_mean"] for p in common])
    means1 = np.array([fit1["pitchers"][p]["y_mean"] for p in common])
    stds0 = np.array([fit0["pitchers"][p]["y_std"] for p in common])
    stds1 = np.array([fit1["pitchers"][p]["y_std"] for p in common])
    result["label_reliability"] = {
        "mean_residual_split_half_r": round(float(np.corrcoef(means0, means1)[0, 1]), 3),
        "residual_sd_split_half_r": round(float(np.corrcoef(stds0, stds1)[0, 1]), 3),
    }
    return result


# %%
def full_sample_deltas(frame: pd.DataFrame, lam: float, min_rows: int) -> pd.DataFrame:
    """Per-pitcher response deviations on all rows, for later profile clustering."""
    fit = fit_half(frame, lam, min_rows)
    rows = []
    for pitcher, rec in fit["pitchers"].items():
        row = {"pitcher": pitcher, "n_outings": len(rec["yc"])}
        for j, col in enumerate(PREDICTORS):
            row[f"delta_{col}"] = float(rec["delta"][j])
            row[f"beta_{col}"] = float(fit["beta_global"][j] + rec["delta"][j])
        rows.append(row)
    return pd.DataFrame(rows)


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/outings_mlb_bullpen_2021_2025.parquet"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache_heterogeneity_labeled_fast.parquet"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--full", action="store_true", help="Run full prepare_features without trimming.")
    parser.add_argument("--min-pitcher-ip", type=float, default=30.0)
    parser.add_argument("--min-pitcher-bf", type=int, default=100)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[10.0, 25.0, 50.0])
    parser.add_argument("--min-rows", type=int, nargs="+", default=[25, 40])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs/response_heterogeneity_validation"))
    args = parser.parse_args()

    cache = None if args.no_cache else args.cache
    labeled = _load_labeled(args.input, cache, args.min_pitcher_ip, args.min_pitcher_bf, fast=not args.full)
    frame, coverage = build_analysis_frame(labeled)
    preflight = preflight_checks(frame)

    print("\n=== coverage ===")
    print(json.dumps(coverage, indent=2))
    print("\n=== preflight ===")
    print(json.dumps(preflight, indent=2))

    results = []
    for design in ["odd_even", "chrono"]:
        for lam in args.lambdas:
            for min_rows in args.min_rows:
                results.append(evaluate_design(frame, design, lam, min_rows))

    print("\n=== split-half results ===")
    for res in results:
        print(json.dumps(res, indent=2))

    deltas = full_sample_deltas(frame, lam=25.0, min_rows=25)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "predictors": PREDICTORS,
        "rest_days_cap": REST_DAYS_CAP,
        "coverage": coverage,
        "preflight": preflight,
        "results": results,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    deltas.to_csv(args.output_dir / "per_pitcher_response_deltas.csv", index=False, encoding="utf-8-sig")
    print(f"\nArtifacts: {args.output_dir}")


# %%
if __name__ == "__main__":
    main()
