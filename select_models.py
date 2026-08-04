"""Run development-only Stuff+ model selection.

The normal profile is the reproducible research run: five TCN seeds, sequence
lengths 5 and 8, 250 epochs, and 32 staged XGBoost trials.  ``--smoke-test``
uses a deliberately small, separately labelled profile for integration checks.
Both profiles score only the 2022, 2023, and 2024 rolling validation folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lib.stuff_cli import (
    add_stuff_data_arguments,
    load_stuff_data_from_args,
)
from lib.stuff_selection import (
    DEFAULT_MODELS,
    DEFAULT_RANDOM_SEEDS,
    run_model_selection,
)


NORMAL_SEQUENCE_LENGTHS = (5, 8)
SMOKE_SEQUENCE_LENGTHS = (5,)
NORMAL_TCN_EPOCHS = 250
SMOKE_TCN_EPOCHS = 2
NORMAL_XGBOOST_TRIALS = 32
SMOKE_XGBOOST_TRIALS = 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select Stuff+ models on the 2022-2024 rolling folds. "
            "The 2025 outcomes are reserved for locked final evaluation."
        )
    )
    add_stuff_data_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_selection"),
        help="Directory for OOF predictions, metrics, manifest, and model lock.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(DEFAULT_MODELS),
        default=list(DEFAULT_MODELS),
        help="Candidate families to evaluate.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="TCN training device.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help=(
            "Number of deterministic TCN seeds. Defaults to 5 normally and "
            "1 only with --smoke-test."
        ),
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        choices=list(NORMAL_SEQUENCE_LENGTHS),
        default=None,
        help="TCN sequence lengths. Defaults to 5 8 normally and 5 in smoke mode.",
    )
    parser.add_argument(
        "--tcn-epochs",
        type=int,
        default=None,
        help="TCN epochs per candidate and seed (normal default: 250).",
    )
    parser.add_argument(
        "--xgboost-trials",
        type=int,
        default=None,
        help="Staged TPE trials on development data (normal default: 32).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run the explicitly labelled reduced integration profile: one "
            "seed, L=5, two TCN epochs, and one XGBoost trial."
        ),
    )
    return parser


def _resolved_profile(args: argparse.Namespace) -> dict[str, object]:
    smoke_test = bool(args.smoke_test)
    num_seeds = (
        int(args.num_seeds)
        if args.num_seeds is not None
        else (1 if smoke_test else len(DEFAULT_RANDOM_SEEDS))
    )
    if num_seeds < 1:
        raise ValueError("--num-seeds must be positive")
    if num_seeds > len(DEFAULT_RANDOM_SEEDS):
        raise ValueError(
            f"--num-seeds cannot exceed {len(DEFAULT_RANDOM_SEEDS)} "
            "without an explicit registered seed list"
        )
    sequence_lengths = tuple(
        int(value)
        for value in (
            args.sequence_lengths
            if args.sequence_lengths is not None
            else (
                SMOKE_SEQUENCE_LENGTHS
                if smoke_test
                else NORMAL_SEQUENCE_LENGTHS
            )
        )
    )
    tcn_epochs = (
        int(args.tcn_epochs)
        if args.tcn_epochs is not None
        else (
            SMOKE_TCN_EPOCHS
            if smoke_test
            else NORMAL_TCN_EPOCHS
        )
    )
    xgboost_trials = (
        int(args.xgboost_trials)
        if args.xgboost_trials is not None
        else (
            SMOKE_XGBOOST_TRIALS
            if smoke_test
            else NORMAL_XGBOOST_TRIALS
        )
    )
    if tcn_epochs < 1:
        raise ValueError("--tcn-epochs must be positive")
    if xgboost_trials < 1:
        raise ValueError("--xgboost-trials must be positive")
    if (
        "tcn" in args.models
        and not smoke_test
        and num_seeds < 5
    ):
        raise ValueError(
            "Normal TCN selection requires at least five seeds; "
            "use --smoke-test for a reduced integration run"
        )
    return {
        "random_seeds": DEFAULT_RANDOM_SEEDS[:num_seeds],
        "sequence_lengths": sequence_lengths,
        "tcn_epochs": tcn_epochs,
        "xgboost_trials": xgboost_trials,
        "smoke_test": smoke_test,
    }


def run_from_args(args: argparse.Namespace) -> dict:
    profile = _resolved_profile(args)
    bundle = load_stuff_data_from_args(args)
    return run_model_selection(
        bundle,
        output_dir=args.output_dir,
        models=tuple(args.models),
        device=args.device,
        random_seeds=profile["random_seeds"],
        sequence_lengths=profile["sequence_lengths"],
        tcn_epochs=profile["tcn_epochs"],
        xgboost_trials=profile["xgboost_trials"],
        smoke_test=profile["smoke_test"],
        cutter_category=args.cutter_category,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    result = run_from_args(args)
    summary = result["metrics_summary"]
    output = result["output_dir"]
    print(
        f"Wrote {len(summary):,} candidate summaries to {output}. "
        "Selection folds: 2022, 2023, 2024; 2025 outcomes scored: 0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
