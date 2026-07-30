"""Run the one-shot 2025 evaluation from an existing immutable model lock."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lib.evaluation import load_locked_model_config
from lib.stuff_cli import add_stuff_data_arguments, load_stuff_data_from_args
from lib.stuff_final import FINAL_MODEL_FAMILIES, run_locked_final_evaluation


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refit exactly the locked 2020-2024 Stuff+ models and evaluate "
            "the reserved 2025 fold once."
        )
    )
    add_stuff_data_arguments(parser)
    parser.add_argument(
        "--locked-config",
        type=Path,
        required=True,
        help="Immutable locked_model_config.json produced by select_models.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for final 2025 predictions, metrics, and diagnostics.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(FINAL_MODEL_FAMILIES),
        default=list(FINAL_MODEL_FAMILIES),
        help=(
            "Locked active model families to replay. Inactive lock entries "
            "are never fitted or emitted."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for an active locked TCN.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help=(
            "Optional assertion against the lock's seed count. This option "
            "cannot change the locked seeds."
        ),
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        choices=[5, 8],
        default=None,
        help=(
            "Optional assertion that every replayed TCN component uses one "
            "of these locked lengths. This option cannot change the lock."
        ),
    )
    return parser


def _locked_tcn_sequence_lengths(lock: dict) -> set[int]:
    shared = lock["shared_tcn"]
    candidate_ids: set[str] = set()
    if shared.get("status") == "selected" and shared.get(
        "selected_candidate_id"
    ):
        candidate_ids.add(str(shared["selected_candidate_id"]))
    ensemble = shared.get("ensemble", {})
    if ensemble.get("selected_candidate_id") and ensemble.get(
        "tcn_candidate_id"
    ):
        candidate_ids.add(str(ensemble["tcn_candidate_id"]))

    catalog = lock.get("candidate_configs", {})
    lengths: set[int] = set()
    for candidate_id in candidate_ids:
        if candidate_id not in catalog:
            raise ValueError(
                f"Locked TCN candidate is missing: {candidate_id}"
            )
        config = catalog[candidate_id]
        if "sequence_length" not in config:
            raise ValueError(
                f"Locked TCN candidate has no sequence_length: {candidate_id}"
            )
        lengths.add(int(config["sequence_length"]))
    return lengths


def run_from_args(args: argparse.Namespace) -> dict:
    lock = load_locked_model_config(args.locked_config)
    if int(args.train_start) != int(lock["train_start"]):
        raise ValueError(
            "--train-start cannot change the locked analysis: "
            f"locked={lock['train_start']}, requested={args.train_start}"
        )
    locked_span = int(lock["ewma"]["span"])
    if int(args.ewma_span) != locked_span:
        raise ValueError(
            "--ewma-span cannot change the locked analysis: "
            f"locked={locked_span}, requested={args.ewma_span}"
        )
    locked_cutter = lock["pitch_type_mapping"].get("cutter_category")
    if (
        locked_cutter is not None
        and str(args.cutter_category) != str(locked_cutter)
    ):
        raise ValueError(
            "--cutter-category cannot change the locked analysis: "
            f"locked={locked_cutter}, requested={args.cutter_category}"
        )
    if args.sequence_lengths is not None:
        allowed = {int(value) for value in args.sequence_lengths}
        locked_lengths = _locked_tcn_sequence_lengths(lock)
        if not locked_lengths.issubset(allowed):
            raise ValueError(
                "--sequence-lengths cannot alter the lock: "
                f"locked={sorted(locked_lengths)}, "
                f"asserted={sorted(allowed)}"
            )

    bundle = load_stuff_data_from_args(args)
    return run_locked_final_evaluation(
        bundle,
        lock,
        output_dir=args.output_dir,
        models=tuple(args.models),
        device=args.device,
        requested_num_seeds=args.num_seeds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    result = run_from_args(args)
    print(result["common_metrics"].to_string(index=False))
    print(f"Final 2025 artifacts: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
