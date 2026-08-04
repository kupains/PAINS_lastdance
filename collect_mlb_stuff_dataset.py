from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from lib.collect_fangraphs_stuff import collect_game_logs
from lib.collect_pitcher_statcast import collect_pitchers, file_record
from lib.stuff_mlb_dataset import FEATURES, build_stuff_mlb_dataset, make_dataset


MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
QUALIFY_YEARS = tuple(range(2021, 2026))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _basic_file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_official_starter_stats(
    raw_dir: Path, years: tuple[int, ...], force: bool = False, max_retries: int = 3
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for year in years:
        params = {
            "stats": "season",
            "group": "pitching",
            "season": year,
            "gameType": "R",
            "playerPool": "ALL",
            "sportIds": 1,
            "limit": 5000,
        }
        url = f"{MLB_STATS_URL}?{urllib.parse.urlencode(params)}"
        raw_path = raw_dir / f"mlb_pitching_{year}.json"
        if raw_path.exists() and not force:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            for attempt in range(1, max_retries + 1):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        payload = json.load(response)
                    break
                except Exception:
                    if attempt == max_retries:
                        raise
                    time.sleep(2**attempt)
            raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        splits = payload.get("stats", [{}])[0].get("splits", [])
        for split in splits:
            player = split.get("player", {})
            stat = split.get("stat", {})
            rows.append({
                "player_id": int(player["id"]),
                "name": player.get("fullName"),
                "year": year,
                "games_started": int(stat.get("gamesStarted", 0)),
                "games_pitched": int(stat.get("gamesPitched", 0)),
                "innings_pitched": stat.get("inningsPitched"),
                "number_of_pitches": int(stat.get("numberOfPitches", 0)),
                "source_url": url,
            })
    result = pd.DataFrame(rows)
    duplicates = result.duplicated(["player_id", "year"])
    if duplicates.any():
        raise RuntimeError("MLB Stats API returned duplicate player-season rows.")
    return result.sort_values(["player_id", "year"]).reset_index(drop=True)


def stable_candidates(stats: pd.DataFrame, min_starts: int) -> pd.DataFrame:
    table = stats.pivot(index=["player_id", "name"], columns="year", values="games_started")
    for year in QUALIFY_YEARS:
        if year not in table:
            table[year] = 0
    selected = table.loc[table[list(QUALIFY_YEARS)].ge(min_starts).all(axis=1)].reset_index()
    selected.columns = [
        "player_id", "name", *[f"games_started_{year}" for year in QUALIFY_YEARS]
    ]
    return selected.sort_values("player_id").reset_index(drop=True)


def collection_audit(
    candidates: pd.DataFrame, merged: pd.DataFrame, min_outings: int
) -> pd.DataFrame:
    counts = (
        merged.assign(year=pd.to_datetime(merged["game_date"]).dt.year)
        .groupby(["pitcher", "year"]).size().unstack(fill_value=0)
    )
    rows = []
    for candidate in candidates.itertuples(index=False):
        row: dict[str, object] = {"player_id": candidate.player_id, "name": candidate.name}
        eligible = True
        for year in QUALIFY_YEARS:
            value = int(counts.loc[candidate.player_id, year]) if candidate.player_id in counts.index and year in counts else 0
            row[f"matched_50_pitch_starts_{year}"] = value
            eligible &= value >= min_outings
        row["final_qualified"] = eligible
        row["exclusion_reason"] = "" if eligible else f"at least one season has fewer than {min_outings} matched 50-pitch starts"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["final_qualified", "name"], ascending=[False, True])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducibly collect every source used by the MLB Stuff+ model."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/reproducible_mlb_stuff"),
        help="Dedicated collection root; do not mix files from other collection runs.",
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--min-starts", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Redownload existing Statcast files.")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    official = collect_official_starter_stats(
        args.data_dir / "raw_mlb_stats", QUALIFY_YEARS, force=args.force
    )
    official_path = args.data_dir / "mlb_official_pitching_2021_2025.parquet"
    official.to_parquet(official_path, index=False)
    candidates = stable_candidates(official, args.min_starts)
    candidates_path = args.manifest_dir / "stable_starter_candidates.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    pitcher_ids = candidates["player_id"].astype(int).tolist()

    statcast_2020_dir = args.data_dir / "statcast_mlb_stable_starters_2020"
    statcast_main_dir = args.data_dir / "statcast_mlb_stable_starters_2021_2025"
    statcast_paths = [
        *collect_pitchers(pitcher_ids, "2020-03-01", "2020-10-31", statcast_2020_dir, force=args.force),
        *collect_pitchers(pitcher_ids, "2021-03-01", "2025-10-31", statcast_main_dir, force=args.force),
    ]

    id_frame = official.loc[official["player_id"].isin(pitcher_ids), ["player_id", "year"]].copy()
    id_frame["game_date"] = pd.to_datetime(id_frame["year"].astype(str) + "-07-01")
    all_stuff_path = args.data_dir / "fangraphs_stuff_mlb_stable_starters_2020_2025.parquet"
    fangraphs_cache = args.data_dir / "fangraphs_game_logs"
    all_stuff = collect_game_logs(
        id_frame, all_stuff_path, fangraphs_cache, 2020, 2025, force=args.force
    )
    stuff_paths = []
    for label, years in (("2020", [2020]), ("2021_2025", list(QUALIFY_YEARS))):
        path = args.data_dir / f"fangraphs_stuff_mlb_stable_starters_{label}.parquet"
        all_stuff.loc[all_stuff["game_date"].dt.year.isin(years)].to_parquet(path, index=False)
        stuff_paths.append(path)

    # Use the exact files returned by this run, rather than globbing unrelated files
    # that may coexist in a directory from an older collection.
    merged_all = build_stuff_mlb_dataset(statcast_paths, stuff_paths)
    merged = merged_all.loc[merged_all["target_eligible"]].copy()
    audit = collection_audit(candidates, merged, args.min_starts)
    audit_path = args.manifest_dir / "collection_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    model_data, qualified = make_dataset(statcast_paths, stuff_paths)
    from stuff_mlb_temporal_final import _prepare_data

    prepared = _prepare_data(model_data)
    model_ready = prepared.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*FEATURES, "target_y", "stuff_ewma4_prior"]
    )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("pandas", "pyarrow", "pybaseball", "fungo")
        },
        "selection": {
            "source": MLB_STATS_URL,
            "years": list(QUALIFY_YEARS),
            "minimum_official_games_started_each_year": args.min_starts,
            "candidate_count": len(candidates),
            "candidates": candidates.to_dict(orient="records"),
        },
        "statcast": {
            "collector": "pybaseball.statcast_pitcher",
            "requested_ranges": ["2020-03-01..2020-10-31", "2021-03-01..2025-10-31"],
            "files": [file_record(path) for path in statcast_paths],
        },
        "fangraphs": {
            "collector": "fungo.fangraphs.get_game_log(position=P)",
            "cache_directory": (args.data_dir / "fangraphs_game_logs").as_posix(),
            "files": [
                {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path), "rows": len(pd.read_parquet(path))}
                for path in [all_stuff_path, *stuff_paths]
            ],
        },
        "official_stats_file": {
            "path": official_path.as_posix(), "sha256": _sha256(official_path), "rows": len(official)
        },
        "audit": {
            "path": audit_path.as_posix(),
            "qualified_count": int(audit["final_qualified"].sum()),
            "qualified_player_ids": sorted(audit.loc[audit["final_qualified"], "player_id"].astype(int).tolist()),
            "excluded": audit.loc[~audit["final_qualified"], ["player_id", "name", "exclusion_reason"]].to_dict(orient="records"),
            "matched_50_pitch_starts_by_year": {
                str(year): int(count)
                for year, count in model_data.groupby("year").size().items()
            },
            "model_ready_rows_by_year": {
                str(year): int(count)
                for year, count in model_ready.groupby("year").size().items()
            },
        },
    }
    if sorted(qualified) != manifest["audit"]["qualified_player_ids"]:
        raise RuntimeError("Audit and model dataset disagree on qualified pitcher IDs.")
    manifest["selection"]["raw_responses"] = [
        _basic_file_record(path)
        for path in sorted((args.data_dir / "raw_mlb_stats").glob("mlb_pitching_*.json"))
    ]
    manifest["fangraphs"]["raw_responses"] = [
        _basic_file_record(path) for path in sorted(fangraphs_cache.glob("fg_*.json"))
    ]
    manifest["fangraphs"]["mlbam_to_fangraphs"] = (
        all_stuff[["pitcher", "fangraphs_id"]]
        .drop_duplicates()
        .sort_values("pitcher")
        .rename(columns={"pitcher": "mlbam_id"})
        .to_dict(orient="records")
    )
    manifest_path = args.manifest_dir / "mlb_stuff_collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Candidates: {len(candidates)}; final qualified: {int(audit['final_qualified'].sum())}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
