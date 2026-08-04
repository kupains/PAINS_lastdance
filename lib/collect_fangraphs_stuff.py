from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


def _coalesce_columns(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
) -> pd.Series | None:
    """Coalesce schema aliases per row instead of selecting one globally."""

    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    found = False
    folded = {
        str(column).casefold(): str(column)
        for column in frame.columns
    }
    used: set[str] = set()
    for alias in aliases:
        column = (
            str(alias)
            if alias in frame.columns
            else folded.get(str(alias).casefold())
        )
        if column is None or column in used:
            continue
        used.add(column)
        found = True
        values = frame[column]
        present = values.notna()
        if pd.api.types.is_object_dtype(values.dtype) or isinstance(
            values.dtype, pd.StringDtype
        ):
            present &= values.astype("string").str.strip().ne("")
        result = result.where(result.notna(), values.where(present))
    return result if found else None


def _deduplicate_logs(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate with row-level game-id/date identity and reject ambiguity."""

    data = frame.copy()
    if "game_pk" not in data:
        data["game_pk"] = pd.NA
    game = pd.to_numeric(data["game_pk"], errors="coerce")
    data["_has_game_pk"] = game.notna()
    data["_game_join"] = game.astype("Int64").astype("string").fillna("")
    identity = pd.Series("", index=data.index, dtype="string")
    identity.loc[data["_has_game_pk"]] = (
        data.loc[data["_has_game_pk"], "pitcher"].astype("Int64").astype("string")
        + "|g|"
        + data.loc[data["_has_game_pk"], "_game_join"]
    )
    identity.loc[~data["_has_game_pk"]] = (
        data.loc[~data["_has_game_pk"], "pitcher"].astype("Int64").astype("string")
        + "|d|"
        + data.loc[~data["_has_game_pk"], "game_date"]
        .dt.strftime("%Y-%m-%d")
        .astype("string")
    )
    data["_identity"] = identity
    payload_columns = [
        column
        for column in data.columns
        if column not in {"_has_game_pk", "_game_join", "_identity"}
    ]
    distinct = data.drop_duplicates(["_identity", *payload_columns])
    ambiguous = distinct.duplicated("_identity", keep=False)
    if ambiguous.any():
        examples = (
            distinct.loc[ambiguous, ["pitcher", "game_date", "game_pk"]]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(f"Ambiguous FanGraphs game-log identities: {examples}")
    return (
        distinct.drop_duplicates("_identity", keep="last")
        .drop(columns=["_has_game_pk", "_game_join", "_identity"])
        .reset_index(drop=True)
    )


def collect_game_logs(
    outings_path: Path | pd.DataFrame,
    output_path: Path,
    cache_dir: Path,
    start_year: int = 2020,
    end_year: int = 2025,
    force: bool = False,
) -> pd.DataFrame:
    from fungo import fangraphs
    from pybaseball import playerid_reverse_lookup

    outings = (
        outings_path.copy()
        if isinstance(outings_path, pd.DataFrame)
        else pd.read_parquet(outings_path)
    )
    outings["game_date"] = pd.to_datetime(outings["game_date"])
    id_column = "pitcher" if "pitcher" in outings.columns else "player_id"
    if id_column not in outings.columns:
        raise ValueError("Input must contain a pitcher or player_id column.")
    pitcher_ids = sorted(
        int(value)
        for value in outings.loc[
            outings["game_date"].dt.year.between(start_year, end_year), id_column
        ].dropna().unique()
    )
    lookup = playerid_reverse_lookup(pitcher_ids, key_type="mlbam")
    lookup = lookup.loc[lookup["key_mlbam"].isin(pitcher_ids)].copy()
    id_map = {
        int(row.key_mlbam): int(row.key_fangraphs)
        for row in lookup.itertuples()
        if pd.notna(row.key_fangraphs) and int(row.key_fangraphs) > 0
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for mlbam_id, fangraphs_id in sorted(id_map.items()):
        for year in range(start_year, end_year + 1):
            cache_path = cache_dir / f"fg_{fangraphs_id}_{year}.json"
            if cache_path.exists() and not force:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                for attempt in range(1, 4):
                    try:
                        payload = fangraphs.get_game_log(fangraphs_id, year, position="P")
                        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2**attempt)
            rows = payload.get("mlb", [])
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame["pitcher"] = mlbam_id
            frame["fangraphs_id"] = fangraphs_id
            frames.append(frame)

    if not frames:
        raise RuntimeError("No FanGraphs game logs were collected.")
    combined = pd.concat(frames, ignore_index=True).copy()
    raw_date = _coalesce_columns(
        combined, ("game_date", "gamedate", "gameDate", "date", "Date")
    )
    if raw_date is None:
        raise ValueError("FanGraphs game logs do not contain a game date.")
    date_text = raw_date.astype("string")
    extracted_date = date_text.str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
    combined["game_date"] = pd.to_datetime(
        extracted_date.fillna(date_text), errors="coerce"
    ).dt.normalize()
    raw_gs = _coalesce_columns(
        combined,
        ("GS", "is_starting_pitcher", "is_starter", "starter"),
    )
    if raw_gs is None:
        raise ValueError("FanGraphs game logs do not contain a starter flag.")
    combined["GS"] = pd.to_numeric(raw_gs, errors="coerce")
    gs_text = raw_gs.astype("string").str.strip().str.lower()
    combined.loc[
        combined["GS"].isna()
        & gs_text.isin({"true", "t", "yes", "y", "starter", "starting"}),
        "GS",
    ] = 1
    combined.loc[
        combined["GS"].isna()
        & gs_text.isin({"false", "f", "no", "n", "reliever", "relief"}),
        "GS",
    ] = 0
    combined = combined.loc[
        combined["game_date"].dt.year.between(start_year, end_year)
        & combined["GS"].eq(1)
    ].copy()
    combined["is_starting_pitcher"] = 1

    stuff_values = _coalesce_columns(
        combined,
        ("sp_stuff", "stuff_plus", "Stuff+", "StuffPlus", "stuffplus"),
    )
    if stuff_values is None:
        raise ValueError("FanGraphs game logs do not contain a starter Stuff+ column.")
    combined["sp_stuff"] = pd.to_numeric(stuff_values, errors="coerce")
    combined["stuff_plus"] = combined["sp_stuff"]
    game_pk = _coalesce_columns(
        combined,
        ("game_pk", "gamePk", "game_id", "game_id_mlb"),
    )
    if game_pk is not None:
        combined["game_pk"] = game_pk

    keep = [
        "pitcher", "fangraphs_id", "game_date", "game_pk", "gameid",
        "GS", "is_starting_pitcher", "Team", "Opp", "Pitches", "TBF",
        "sp_stuff", "stuff_plus", "sp_location", "sp_pitching",
        "pb_stuff", "pb_command", "pb_overall",
    ]
    combined = combined[[column for column in keep if column in combined.columns]]
    combined = _deduplicate_logs(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect FanGraphs game-level Stuff+ logs.")
    parser.add_argument("--outings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path("data/fangraphs_game_logs"), type=Path)
    parser.add_argument("--start-year", default=2020, type=int)
    parser.add_argument("--end-year", default=2025, type=int)
    parser.add_argument("--force", action="store_true", help="Redownload cached API responses.")
    args = parser.parse_args()
    result = collect_game_logs(
        args.outings, args.output, args.cache_dir, args.start_year, args.end_year, args.force
    )
    print(f"Wrote {len(result):,} starter game logs to {args.output}")


if __name__ == "__main__":
    main()
