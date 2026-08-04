"""Build the canonical expanding-window Stuff+ fold manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from lib.stuff_cli import (
    FINAL_TEST_YEAR,
    SELECTION_VALIDATION_YEARS,
    StuffDataBundle,
    add_stuff_data_arguments,
    load_stuff_data_from_args,
)
from lib.temporal_splits import (
    build_fold_manifest as create_fold_manifest,
    fold_manifest_fingerprint,
    rolling_fold_specs,
)


def build_manifest(bundle: StuffDataBundle) -> tuple[pd.DataFrame, str]:
    """Create shared 2022-2024 validation folds plus the 2025 test fold.

    The published roster may have been fixed using 2025 start availability.
    That cohort decision is preserved as metadata and is not confused with
    validation scoring: thresholds and model comparisons for the selection
    folds still use only prior training years.
    """

    folds = rolling_fold_specs(
        validation_years=SELECTION_VALIDATION_YEARS,
        test_year=FINAL_TEST_YEAR,
    )
    target_column = (
        "target_y"
        if "target_y" in bundle.outings
        else (
            "stuff_plus"
            if "stuff_plus" in bundle.outings
            else "sp_stuff"
        )
    )
    manifest = create_fold_manifest(
        bundle.outings,
        folds=folds,
        target_col=target_column,
    )
    validation = manifest.loc[manifest["split"].eq("validation")]
    observed_validation_years = tuple(
        sorted(int(value) for value in validation["fold_year"].unique())
    )
    if observed_validation_years != SELECTION_VALIDATION_YEARS:
        raise AssertionError(
            "Manifest validation folds must be exactly 2022, 2023, and 2024"
        )
    if (
        validation["year"].eq(FINAL_TEST_YEAR).any()
        or validation["fold_year"].eq(FINAL_TEST_YEAR).any()
    ):
        raise AssertionError("2025 rows entered a model-selection validation fold")

    fingerprint = fold_manifest_fingerprint(manifest)
    qualification = dict(bundle.qualification_metadata)
    manifest.attrs.update(
        {
            "dataset_fingerprint": bundle.dataset_fingerprint,
            "fold_manifest_fingerprint": fingerprint,
            "qualified_pitchers": bundle.qualified_pitchers,
            "qualification_rule": bundle.qualification_rule,
            "qualification_fit_years": bundle.qualification_fit_years,
            "qualification_uses_2025_roster_availability": qualification.get(
                "uses_2025_roster_availability", False
            ),
            "selection_validation_years": SELECTION_VALIDATION_YEARS,
            "selection_uses_2025_outcomes": False,
            "final_test_year": FINAL_TEST_YEAR,
        }
    )
    return manifest, fingerprint


def write_manifest(manifest: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.suffix.lower() != ".parquet":
        raise ValueError("Fold manifest output must use a .parquet extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)
    return output


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic 2022-2024 model-selection folds and the "
            "separate 2025 final-test fold."
        )
    )
    add_stuff_data_arguments(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fold_manifest.parquet"),
    )
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser


def run_from_args(args: argparse.Namespace) -> tuple[Path, dict]:
    bundle = load_stuff_data_from_args(args)
    manifest, manifest_fingerprint = build_manifest(bundle)
    output = write_manifest(manifest, args.output)
    metadata = {
        **bundle.metadata,
        "fold_manifest_fingerprint": manifest_fingerprint,
        "manifest_rows": int(len(manifest)),
        "manifest_path": str(output),
        "selection_validation_years": list(SELECTION_VALIDATION_YEARS),
        "selection_uses_2025_outcomes": False,
        "final_test_year": FINAL_TEST_YEAR,
        "fold_years": sorted(
            int(value) for value in manifest["fold_year"].unique()
        ),
    }
    metadata_output = getattr(args, "metadata_output", None)
    if metadata_output is not None:
        metadata_path = Path(metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return output, metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    output, metadata = run_from_args(args)
    print(
        f"Wrote {metadata['manifest_rows']:,} fold rows to {output} "
        f"(dataset={metadata['dataset_fingerprint']}, "
        f"manifest={metadata['fold_manifest_fingerprint']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
