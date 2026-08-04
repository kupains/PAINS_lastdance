# %%
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


class StarterInferenceWarning(UserWarning):
    """The observed Statcast rows cannot prove that a pitcher was the starter."""


# %%
def _read_many(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No parquet/csv files found in {input_dir}")
    frames = []
    for path in paths:
        if path.suffix == ".parquet":
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


# %%
def _batters_faced(group: pd.DataFrame) -> int:
    if "at_bat_number" in group.columns:
        return int(group["at_bat_number"].nunique())
    if {"batter", "inning"}.issubset(group.columns):
        return int(group[["batter", "inning"]].drop_duplicates().shape[0])
    return int(max(1, round(len(group) / 4)))


# %%
def _pitch_mix(pitches: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if "pitch_type" not in pitches.columns:
        return pd.DataFrame(columns=keys)
    counts = pitches.groupby(keys + ["pitch_type"]).size().rename("n").reset_index()
    totals = counts.groupby(keys)["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    mix = counts.pivot_table(index=keys, columns="pitch_type", values="share", fill_value=0)
    mix.columns = [f"pitch_mix_{col}" for col in mix.columns]
    return mix.reset_index()


# %%
def _derive_pitching_team(pitches: pd.DataFrame) -> pd.DataFrame:
    """Attach the fielding team when Statcast did not supply it directly."""

    out = pitches.copy()
    if "pitching_team" not in out.columns:
        out["pitching_team"] = pd.NA
    if {"home_team", "away_team", "inning_topbot"}.issubset(out.columns):
        half = out["inning_topbot"].astype("string").str.strip().str.lower()
        inferred = pd.Series(pd.NA, index=out.index, dtype="string")
        inferred.loc[half.str.startswith("top", na=False)] = (
            out.loc[half.str.startswith("top", na=False), "home_team"]
            .astype("string")
            .str.upper()
        )
        inferred.loc[half.str.startswith("bot", na=False)] = (
            out.loc[half.str.startswith("bot", na=False), "away_team"]
            .astype("string")
            .str.upper()
        )
        current = out["pitching_team"].astype("string")
        out["pitching_team"] = current.where(current.notna() & current.ne(""), inferred)
    return out


# %%
def _first_pitcher_by_game(pitches: pd.DataFrame) -> pd.Series:
    """Return both teams' first-inning pitchers for every game.

    A game has two starters.  Grouping only on ``game_pk`` silently selected
    whichever club happened to pitch first.  We prefer the fielding team and
    fall back to Statcast's top/bottom half marker.  Requiring the group's first
    observed inning to be inning one also keeps this inference correct for
    pitcher-specific Statcast files that contain relief appearances.
    """

    if "inning" not in pitches.columns:
        raise ValueError("Cannot identify the first pitcher because inning is missing.")

    ordered = _derive_pitching_team(pitches)
    if "pitching_team" in ordered.columns and ordered["pitching_team"].notna().any():
        side_column = "pitching_team"
    elif "inning_topbot" in ordered.columns:
        side_column = "inning_topbot"
    else:
        raise ValueError(
            "Cannot identify both starting pitchers because pitching_team and "
            "inning_topbot are missing."
        )

    sort_columns = ["game_pk", side_column, "inning"]
    ordered["inning"] = pd.to_numeric(ordered["inning"], errors="coerce")
    for column in ["at_bat_number", "pitch_number"]:
        if column in ordered.columns:
            ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
            sort_columns.append(column)
    ordered = ordered.sort_values(sort_columns, kind="stable")
    group_columns = ["game_pk", side_column]
    first_rows = ordered.groupby(group_columns, sort=False, as_index=False).first()
    first_rows = first_rows.loc[first_rows["inning"].eq(1)]
    return first_rows.set_index(group_columns)["pitcher"]


def _numeric_boolean(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    result = numeric.gt(0)
    result.loc[text.isin({"true", "t", "yes", "y", "starter", "starting"})] = True
    result.loc[text.isin({"false", "f", "no", "n", "reliever", "relief"})] = False
    return result.fillna(False)


def _explicit_starter_values(pitches: pd.DataFrame) -> tuple[pd.Series, str] | None:
    """Coalesce row-level starter aliases without letting a null alias win.

    Concatenating files with slightly different schemas commonly creates both
    ``is_starting_pitcher`` and ``GS`` columns, with one populated per source
    row.  Choosing one column globally turns the other source's starts into
    relievers.
    """

    columns = [
        column
        for column in ("is_starting_pitcher", "GS")
        if column in pitches.columns
    ]
    if not columns:
        return None
    values = pd.Series(pd.NA, index=pitches.index, dtype="object")
    parsed: list[tuple[str, pd.Series]] = []
    for column in columns:
        raw = pitches[column]
        known = raw.notna() & raw.astype("string").str.strip().ne("")
        boolean = _numeric_boolean(raw)
        parsed.append((column, boolean.where(known)))
        values = values.where(values.notna(), raw.where(known))

    parsed_frame = pd.concat(
        [boolean.rename(column) for column, boolean in parsed],
        axis=1,
    )
    saw_true = parsed_frame.eq(True).fillna(False).any(axis=1)  # noqa: E712
    saw_false = parsed_frame.eq(False).fillna(False).any(axis=1)  # noqa: E712
    if (saw_true & saw_false).any():
        raise ValueError(
            "Conflicting is_starting_pitcher/GS values for a Statcast row."
        )
    if not values.notna().any():
        return None
    return values, "/".join(columns)


# %%
def build_outings(
    pitches: pd.DataFrame,
    player_bio: pd.DataFrame | None = None,
    *,
    starter_inference: str = "warn",
) -> pd.DataFrame:
    """Aggregate Statcast pitch-level rows into pitcher-game outings."""
    inference_mode = str(starter_inference).strip().lower()
    if inference_mode not in {"warn", "allow", "strict"}:
        raise ValueError("starter_inference must be one of: warn, allow, strict")
    required = {"pitcher", "game_pk", "game_date"}
    missing = required - set(pitches.columns)
    if missing:
        raise ValueError(f"Missing required pitch columns: {sorted(missing)}")

    df = _derive_pitching_team(pitches)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    keys = ["pitcher", "game_pk", "game_date"]

    agg_spec = {
        "pitch_count": ("pitcher", "size"),
        "release_speed": ("release_speed", "mean"),
        "effective_speed": ("effective_speed", "mean"),
        "release_spin_rate": ("release_spin_rate", "mean"),
        "release_extension": ("release_extension", "mean"),
        "release_pos_x": ("release_pos_x", "mean"),
        "release_pos_y": ("release_pos_y", "mean"),
        "release_pos_z": ("release_pos_z", "mean"),
        "arm_angle": ("arm_angle", "mean"),
        "spin_axis": ("spin_axis", "mean"),
        "pfx_x": ("pfx_x", "mean"),
        "pfx_z": ("pfx_z", "mean"),
        "plate_x": ("plate_x", "mean"),
        "plate_z": ("plate_z", "mean"),
        "zone": ("zone", "mean"),
        "api_break_z_with_gravity": ("api_break_z_with_gravity", "mean"),
        "api_break_x_arm": ("api_break_x_arm", "mean"),
        "estimated_woba_using_speedangle_mean": ("estimated_woba_using_speedangle", "mean"),
        "woba_value_mean": ("woba_value", "mean"),
        "pitching_team": ("pitching_team", "first"),
        "game_type": ("game_type", "first"),
        "GS": ("GS", "max"),
    }
    existing_agg = {name: spec for name, spec in agg_spec.items() if spec[0] in df.columns}
    outings = df.groupby(keys, as_index=False).agg(**existing_agg)

    if "estimated_woba_using_speedangle_mean" in outings.columns and "woba_value_mean" in outings.columns:
        outings["outing_xwOBA"] = outings["estimated_woba_using_speedangle_mean"].fillna(outings["woba_value_mean"])
    elif "estimated_woba_using_speedangle_mean" in outings.columns:
        outings["outing_xwOBA"] = outings["estimated_woba_using_speedangle_mean"]
    elif "woba_value_mean" in outings.columns:
        outings["outing_xwOBA"] = outings["woba_value_mean"]

    explicit_starter = _explicit_starter_values(df)
    explicit_starter_column: str | None = None
    if explicit_starter is not None:
        explicit_values, explicit_starter_column = explicit_starter
        explicit = df[keys].copy()
        explicit["_explicit_start"] = _numeric_boolean(explicit_values)
        explicit = explicit.groupby(keys, as_index=False)["_explicit_start"].max()
        outings = outings.merge(explicit, on=keys, how="left")
        outings["is_starting_pitcher"] = (
            outings["_explicit_start"].fillna(False).astype(int)
        )
        outings["starter_status_source"] = explicit_starter_column
        outings = outings.drop(columns="_explicit_start")

    if "inning" in df.columns:
        inning_df = df.copy()
        inning_df["inning"] = pd.to_numeric(inning_df["inning"], errors="coerce")
        inning_meta = inning_df.groupby(keys, as_index=False).agg(
            first_inning=("inning", "min"),
            last_inning=("inning", "max"),
        )
        outings = outings.merge(inning_meta, on=keys, how="left")
        if explicit_starter_column is None:
            if inference_mode == "strict":
                raise ValueError(
                    "Starter status cannot be proven from first-observed Statcast "
                    "rows. Supply is_starting_pitcher/GS or use "
                    "starter_inference='warn'/'allow'."
                )
            if inference_mode == "warn":
                warnings.warn(
                    "Inferring starters from each team's first observed pitcher. "
                    "A pitcher-filtered extract can misclassify a bulk reliever "
                    "who enters in inning 1; reconcile with an explicit GS/status "
                    "source when available.",
                    StarterInferenceWarning,
                    stacklevel=2,
                )
            first_pitchers = _first_pitcher_by_game(df).rename("starter").reset_index()
            starter_pairs = first_pitchers[["game_pk", "starter"]].rename(
                columns={"starter": "pitcher"}
            ).drop_duplicates()
            starter_pairs["_inferred_start"] = 1
            outings = outings.merge(
                starter_pairs, on=["game_pk", "pitcher"], how="left"
            )
            outings["is_starting_pitcher"] = (
                outings["_inferred_start"].fillna(0).astype(int)
            )
            outings["starter_status_source"] = "statcast_first_inning_by_team"
            outings = outings.drop(columns="_inferred_start")

    bf = df.groupby(keys).apply(_batters_faced, include_groups=False).rename("BF").reset_index()
    outings = outings.merge(bf, on=keys, how="left")

    mix = _pitch_mix(df, keys)
    if not mix.empty:
        outings = outings.merge(mix, on=keys, how="left")

    if {"p_throws", "stand"}.issubset(df.columns):
        matchup = df.assign(
            same_hand=(df["p_throws"].astype(str).str[0] == df["stand"].astype(str).str[0]).astype(float),
            lefty_batter=(df["stand"].astype(str).str.upper().str[0] == "L").astype(float),
        ).groupby(keys, as_index=False).agg(
            same_hand_ratio=("same_hand", "mean"),
            lefty_batter_ratio=("lefty_batter", "mean"),
        )
        outings = outings.merge(matchup, on=keys, how="left")

    if "batter_prior_xwOBA" in df.columns:
        batter_quality = df.groupby(keys, as_index=False).agg(
            opponent_batter_prior_xwOBA=("batter_prior_xwOBA", "mean")
        )
        outings = outings.merge(batter_quality, on=keys, how="left")

    outings = outings.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    outings["rest_days"] = outings.groupby("pitcher")["game_date"].diff().dt.days
    outings["day_of_season"] = outings["game_date"].dt.dayofyear

    if player_bio is not None and {"pitcher", "birth_date"}.issubset(player_bio.columns):
        bio = player_bio[["pitcher", "birth_date"]].copy()
        bio["birth_date"] = pd.to_datetime(bio["birth_date"], errors="coerce")
        outings = outings.merge(bio, on="pitcher", how="left")
        outings["age"] = (outings["game_date"] - outings["birth_date"]).dt.days / 365.25
        outings = outings.drop(columns=["birth_date"])
    elif "age" not in outings.columns:
        outings["age"] = np.nan

    return outings


# %%
def filter_starter_outings(outings: pd.DataFrame, min_pitches: int = 50) -> pd.DataFrame:
    """Keep first pitchers with enough pitches, then recompute starter-only rest."""
    required = {"is_starting_pitcher", "pitch_count", "pitcher", "game_date", "game_pk"}
    missing = required - set(outings.columns)
    if missing:
        raise ValueError(f"Cannot apply starter filter; missing columns: {sorted(missing)}")
    if min_pitches < 1:
        raise ValueError("min_pitches must be at least 1.")

    selected = outings.loc[
        outings["is_starting_pitcher"].eq(1)
        & pd.to_numeric(outings["pitch_count"], errors="coerce").ge(min_pitches)
    ].copy()
    selected = selected.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    selected["rest_days"] = selected.groupby("pitcher")["game_date"].diff().dt.days
    return selected


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="Build outing-level data from Statcast pitch data.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    role_group = parser.add_mutually_exclusive_group()
    role_group.add_argument("--relief-only", action="store_true", help="Keep non-first pitchers only.")
    role_group.add_argument(
        "--starter-only",
        action="store_true",
        help="Keep each game's first pitcher when the outing reaches --min-pitches.",
    )
    parser.add_argument("--min-pitches", type=int, default=50, help="Minimum pitches for --starter-only.")
    parser.add_argument("--regular-season-only", action="store_true", help="Keep Statcast game_type R only.")
    parser.add_argument(
        "--starter-inference",
        choices=("warn", "allow", "strict"),
        default="warn",
        help=(
            "Policy when no explicit is_starting_pitcher/GS is present. "
            "'strict' refuses first-observed-pitcher inference."
        ),
    )
    args = parser.parse_args()

    pitches = _read_many(args.input_dir)
    if args.regular_season_only:
        if "game_type" not in pitches.columns:
            raise ValueError("Cannot apply --regular-season-only because game_type is missing.")
        pitches = pitches.loc[pitches["game_type"].astype("string").str.upper().eq("R")].copy()
    outings = build_outings(pitches, starter_inference=args.starter_inference)
    if args.relief_only:
        if "is_starting_pitcher" not in outings.columns:
            raise ValueError("Cannot apply --relief-only because input rows do not include inning.")
        outings = outings.loc[outings["is_starting_pitcher"] != 1].copy()
    elif args.starter_only:
        outings = filter_starter_outings(outings, min_pitches=args.min_pitches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outings.to_parquet(args.output, index=False)
    print(f"Wrote {len(outings):,} outings to {args.output}")


# %%
if __name__ == "__main__":
    main()
