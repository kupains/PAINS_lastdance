from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest import mock

import pandas as pd
import pytest

from build_fold_manifest import build_manifest, main as build_manifest_main
import lib.stuff_cli as stuff_cli
from lib.stuff_cli import (
    RESEARCH_QUALIFICATION_YEARS,
    load_stuff_data,
    qualify_pitchers,
)
from lib.stuff_demo_data import make_demo_stuff_outings


def _research_roster_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    game_pk = 1000
    for pitcher in ("qualified", "one_short"):
        for year in RESEARCH_QUALIFICATION_YEARS:
            count = 19 if pitcher == "one_short" and year == 2023 else 20
            for index in range(count):
                game_pk += 1
                rows.append(
                    {
                        "pitcher": pitcher,
                        "game_pk": game_pk,
                        "game_date": pd.Timestamp(year, 4, 1)
                        + pd.Timedelta(days=index * 7),
                        "year": year,
                        "pitch_count": 70,
                        "target_y": 95.0 + index,
                        "stuff_plus": 95.0 + index,
                        "sp_stuff": 95.0 + index,
                        "history_eligible": True,
                        "target_eligible": True,
                        "is_starting_pitcher": 1,
                        "game_type": "R",
                    }
                )
    return pd.DataFrame(rows)


def test_research_qualification_preserves_published_2021_2025_rule() -> None:
    result = qualify_pitchers(_research_roster_frame(), mode="research")

    assert result.qualified_pitchers == ("qualified",)
    assert result.fit_years == RESEARCH_QUALIFICATION_YEARS
    assert result.uses_2025_roster_availability is True
    assert result.selection_validation_years == (2022, 2023, 2024)
    assert result.selection_uses_2025_outcomes is False
    assert result.counts_by_pitcher_year["qualified"]["2025"] == 20


def test_demo_auto_mode_is_relaxed_deterministic_and_labelled() -> None:
    first = load_stuff_data(
        demo=True,
        demo_pitchers=2,
        demo_outings_per_season=3,
    )
    second = load_stuff_data(
        demo=True,
        demo_pitchers=2,
        demo_outings_per_season=3,
    )

    assert first.source_mode == "demo"
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.outings["row_id"].is_unique
    assert first.qualification_metadata["mode"] == "development"
    assert first.qualification_metadata["uses_2025_roster_availability"] is False
    assert first.metadata["selection_uses_2025_outcomes"] is False


def test_prepared_mode_supports_multiple_paths_and_name_metadata() -> None:
    frame = make_demo_stuff_outings(
        pitchers=1,
        outings_per_season=3,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "prepared_a.csv"
        second = root / "prepared_b.parquet"
        midpoint = len(frame) // 2
        frame.iloc[:midpoint].to_csv(first, index=False)
        frame.iloc[midpoint:].to_parquet(second, index=False)
        names = root / "names.csv"
        pd.DataFrame(
            {
                "player_id": ["demo_pitcher_01"],
                "name": ["Demo Starter"],
            }
        ).to_csv(names, index=False)

        bundle = load_stuff_data(
            prepared_outings_paths=[first, second],
            official_stats_paths=[names],
        )

    assert len(bundle.outings) == len(frame)
    assert bundle.pitcher_names["demo_pitcher_01"] == "Demo Starter"
    assert bundle.outings["pitcher_name"].eq("Demo Starter").all()
    assert bundle.source_metadata["official_stats_name_paths"] == [str(names)]


def test_name_metadata_is_never_forwarded_as_starter_status() -> None:
    frame = _research_roster_frame().loc[
        lambda value: value["pitcher"].eq("qualified")
    ].copy()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        names = root / "official_names.csv"
        pd.DataFrame(
            {"player_id": ["qualified"], "name": ["Qualified Pitcher"]}
        ).to_csv(names, index=False)

        with mock.patch.object(
            stuff_cli.stuff_dataset,
            "build_stuff_mlb_dataset",
            return_value=frame,
            create=True,
        ) as builder, mock.patch.object(
            stuff_cli.stuff_dataset,
            "add_history_features",
            side_effect=lambda value, **_: value,
            create=True,
        ):
            bundle = load_stuff_data(
                statcast_paths=[root / "statcast_a", root / "statcast_b"],
                stuff_paths=[root / "stuff_a.parquet", root / "stuff_b.parquet"],
                official_stats_paths=[names],
                qualification_mode="research",
            )

    assert bundle.qualified_pitchers == ("qualified",)
    call = builder.call_args
    assert "official_stats_path" not in call.kwargs
    assert "starter_status_path" not in call.kwargs


def test_explicit_row_level_starter_status_uses_separate_builder_input() -> None:
    frame = _research_roster_frame().loc[
        lambda value: value["pitcher"].eq("qualified")
    ].copy()
    captured: dict[str, object] = {}

    def builder(
        statcast_paths,
        stuff_paths,
        official_stats_path=None,
        cutter_category="fastball",
    ):
        captured["official_stats_path"] = official_stats_path
        captured["cutter_category"] = cutter_category
        return frame

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        starter = root / "starters.csv"
        pd.DataFrame(
            {
                "pitcher": ["qualified"],
                "game_date": ["2021-04-01"],
                "is_starting_pitcher": [1],
            }
        ).to_csv(starter, index=False)
        names = root / "names.csv"
        pd.DataFrame(
            {"player_id": ["qualified"], "name": ["Qualified Pitcher"]}
        ).to_csv(names, index=False)

        with mock.patch.object(
            stuff_cli.stuff_dataset,
            "build_stuff_mlb_dataset",
            new=builder,
            create=True,
        ), mock.patch.object(
            stuff_cli.stuff_dataset,
            "add_history_features",
            side_effect=lambda value, **_: value,
            create=True,
        ):
            load_stuff_data(
                statcast_paths=[root / "statcast"],
                stuff_paths=[root / "stuff.parquet"],
                official_stats_paths=[names],
                starter_status_paths=[starter],
                qualification_mode="research",
            )

    assert captured["official_stats_path"] == (starter,)
    assert captured["official_stats_path"] != (names,)


def test_ambiguous_starter_status_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "not_row_level.csv"
        pd.DataFrame(
            {"player_id": [1], "name": ["Pitcher"], "GS": [30]}
        ).to_csv(path, index=False)
        with pytest.raises(ValueError, match="row-level starter data"):
            load_stuff_data(
                statcast_paths=["statcast"],
                stuff_paths=["stuff"],
                starter_status_paths=[path],
            )


def test_manifest_allows_historical_2025_roster_but_not_2025_selection() -> None:
    frame = make_demo_stuff_outings(
        pitchers=1,
        outings_per_season=20,
    )
    frame["pitch_count"] = 70
    frame["target_eligible"] = True
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "research.parquet"
        frame.to_parquet(path, index=False)
        bundle = load_stuff_data(
            prepared_outings_paths=[path],
            qualification_mode="research",
        )
        manifest, fingerprint = build_manifest(bundle)

    assert len(fingerprint) == 64
    assert bundle.qualification_metadata["uses_2025_roster_availability"] is True
    validation = manifest.loc[manifest["split"].eq("validation")]
    assert set(validation["fold_year"].astype(int)) == {2022, 2023, 2024}
    assert not validation["year"].eq(2025).any()
    assert manifest.attrs["selection_uses_2025_outcomes"] is False
    assert manifest.attrs["qualification_uses_2025_roster_availability"] is True


def test_manifest_cli_writes_parquet_and_explicit_metadata() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "fold_manifest.parquet"
        metadata = root / "fold_manifest.json"
        result = build_manifest_main(
            [
                "--demo",
                "--demo-pitchers",
                "1",
                "--demo-outings-per-season",
                "2",
                "--output",
                str(output),
                "--metadata-output",
                str(metadata),
            ]
        )
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        output_exists = output.is_file()

    assert result == 0
    assert output_exists
    assert payload["selection_validation_years"] == [2022, 2023, 2024]
    assert payload["selection_uses_2025_outcomes"] is False
    assert payload["final_test_year"] == 2025
