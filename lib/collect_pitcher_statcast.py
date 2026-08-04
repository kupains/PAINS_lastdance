from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def collect_pitchers(
    pitcher_ids: list[int],
    start: str,
    end: str,
    output_dir: Path,
    max_retries: int = 3,
    force: bool = False,
) -> list[Path]:
    """Collect complete Statcast histories for a small list of pitchers."""
    from pybaseball import statcast_pitcher

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pitcher_id in pitcher_ids:
        path = output_dir / f"statcast_pitcher_{pitcher_id}_{start}_{end}.parquet"
        if force or not path.exists():
            for attempt in range(1, max_retries + 1):
                try:
                    frame = statcast_pitcher(start, end, pitcher_id)
                    break
                except Exception:
                    if attempt == max_retries:
                        raise
                    time.sleep(2**attempt)
            frame.to_parquet(path, index=False)
        paths.append(path)
    return paths


def file_record(path: Path) -> dict[str, object]:
    """Return enough metadata to audit an immutable collected parquet file."""
    frame = pd.read_parquet(path)
    dates = pd.to_datetime(frame.get("game_date"), errors="coerce")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "rows": len(frame),
        "pitcher_ids": sorted(
            int(value) for value in pd.to_numeric(frame.get("pitcher"), errors="coerce").dropna().unique()
        ),
        "min_game_date": None if dates.isna().all() else dates.min().date().isoformat(),
        "max_game_date": None if dates.isna().all() else dates.max().date().isoformat(),
        "unique_games": int(frame["game_pk"].nunique()) if "game_pk" in frame else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect complete Statcast histories by pitcher.")
    parser.add_argument("--pitcher-ids", required=True, nargs="+", type=int)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Redownload existing files.")
    parser.add_argument("--manifest", type=Path, help="Write file hashes and row/date coverage.")
    args = parser.parse_args()
    paths = collect_pitchers(args.pitcher_ids, args.start, args.end, args.output_dir, force=args.force)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "collector": "pybaseball.statcast_pitcher",
            "requested_pitcher_ids": sorted(args.pitcher_ids),
            "requested_start": args.start,
            "requested_end": args.end,
            "files": [file_record(path) for path in paths],
        }
        args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Cached {len(paths)} pitcher files under {args.output_dir}")


if __name__ == "__main__":
    main()
