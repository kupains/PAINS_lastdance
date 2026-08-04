from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


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

    outings = outings_path.copy() if isinstance(outings_path, pd.DataFrame) else pd.read_parquet(outings_path)
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
    combined["game_date"] = pd.to_datetime(combined.get("gamedate"), errors="coerce")
    combined = combined.loc[
        combined["game_date"].dt.year.between(start_year, end_year)
        & pd.to_numeric(combined.get("GS"), errors="coerce").eq(1)
    ].copy()
    keep = [
        "pitcher", "fangraphs_id", "game_date", "Team", "Opp", "Pitches", "TBF",
        "sp_stuff", "sp_location", "sp_pitching", "pb_stuff", "pb_command", "pb_overall",
    ]
    combined = combined[[column for column in keep if column in combined.columns]]
    combined = combined.drop_duplicates(["pitcher", "game_date"], keep="last")
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
