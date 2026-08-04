from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.stuff_mlb_dataset import FEATURES, make_dataset
from stuff_mlb_temporal_final import _prepare_data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(record: dict[str, object]) -> list[str]:
    path = Path(str(record["path"]))
    errors = []
    if not path.exists():
        return [f"missing: {path}"]
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        errors.append(f"size mismatch: {path}")
    if sha256(path) != record["sha256"]:
        errors.append(f"SHA-256 mismatch: {path}")
    if "rows" in record and path.suffix == ".parquet":
        rows = len(pd.read_parquet(path))
        if rows != int(record["rows"]):
            errors.append(f"row count mismatch: {path} ({rows} != {record['rows']})")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify collected MLB Stuff+ inputs and audit counts.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/mlb_stuff_collection_manifest.json")
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    records = [
        *manifest["selection"]["raw_responses"],
        *manifest["statcast"]["files"],
        *manifest["fangraphs"]["raw_responses"],
        *manifest["fangraphs"]["files"],
        manifest["official_stats_file"],
    ]
    errors = [error for record in records for error in verify_file(record)]

    statcast_paths = [Path(record["path"]) for record in manifest["statcast"]["files"]]
    stuff_paths = [
        Path(record["path"])
        for record in manifest["fangraphs"]["files"]
        if record["path"].endswith("_2020.parquet")
        or record["path"].endswith("_2021_2025.parquet")
    ]
    data, qualified = make_dataset(statcast_paths, stuff_paths)
    prepared = _prepare_data(data)
    ready = prepared.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*FEATURES, "target_y", "stuff_ewma4_prior"]
    )
    actual_matched = {str(year): int(count) for year, count in data.groupby("year").size().items()}
    actual_ready = {str(year): int(count) for year, count in ready.groupby("year").size().items()}
    if sorted(qualified) != manifest["audit"]["qualified_player_ids"]:
        errors.append("qualified pitcher IDs do not match manifest")
    if actual_matched != manifest["audit"]["matched_50_pitch_starts_by_year"]:
        errors.append(f"matched yearly counts differ: {actual_matched}")
    if actual_ready != manifest["audit"]["model_ready_rows_by_year"]:
        errors.append(f"model-ready yearly counts differ: {actual_ready}")

    if errors:
        raise SystemExit("Collection verification failed:\n- " + "\n- ".join(errors))
    print(f"Verified {len(records)} files, {len(qualified)} pitchers, and {len(ready):,} model-ready rows.")
    print(f"Model-ready rows by year: {actual_ready}")


if __name__ == "__main__":
    main()
