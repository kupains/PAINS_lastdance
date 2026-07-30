from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from lib.build_outings import StarterInferenceWarning, build_outings
from lib.collect_fangraphs_stuff import _coalesce_columns, _deduplicate_logs
from lib.stuff_mlb_dataset import (
    COMPACT_FEATURES,
    FEATURES,
    TCN_RAW_FEATURES,
    add_history_features,
    aggregate_statcast_outings,
    build_stuff_mlb_dataset,
    default_pitch_type_mapping,
    make_dataset,
    merge_stuff_plus,
)


def _pitch_rows(
    *,
    pitcher: int,
    game_pk: int,
    game_date: str,
    pitch_types: list[str],
    half: str = "Top",
    inning: int = 1,
    explicit_starter: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, pitch_type in enumerate(pitch_types, start=1):
        row: dict[str, object] = {
            "pitcher": pitcher,
            "game_pk": game_pk,
            "game_date": game_date,
            "game_type": "R",
            "home_team": "HOM",
            "away_team": "AWY",
            "inning_topbot": half,
            "inning": inning,
            "at_bat_number": index,
            "pitch_number": 1,
            "pitch_type": pitch_type,
            "release_speed": 90.0 + index,
            "release_spin_rate": 2200.0 + 10 * index,
            "release_extension": 6.0,
            "release_pos_x": -1.7,
            "release_pos_z": 5.9,
            "arm_angle": 45.0,
            "pfx_x": -0.5 + 0.01 * index,
            "pfx_z": 1.2 + 0.01 * index,
            "spin_axis": 180.0,
        }
        if explicit_starter is not None:
            row["is_starting_pitcher"] = explicit_starter
        rows.append(row)
    return rows


def test_build_outings_marks_both_teams_starters() -> None:
    rows = []
    rows += _pitch_rows(
        pitcher=10,
        game_pk=100,
        game_date="2024-04-01",
        pitch_types=["FF", "FF"],
        half="Top",
    )
    rows += _pitch_rows(
        pitcher=20,
        game_pk=100,
        game_date="2024-04-01",
        pitch_types=["SI", "SI"],
        half="Bot",
    )
    rows += _pitch_rows(
        pitcher=30,
        game_pk=100,
        game_date="2024-04-01",
        pitch_types=["FF"],
        half="Top",
        inning=2,
    )

    with pytest.warns(StarterInferenceWarning):
        outings = build_outings(pd.DataFrame(rows)).set_index("pitcher")

    assert outings.at[10, "is_starting_pitcher"] == 1
    assert outings.at[20, "is_starting_pitcher"] == 1
    assert outings.at[30, "is_starting_pitcher"] == 0
    assert outings.at[10, "pitching_team"] == "HOM"
    assert outings.at[20, "pitching_team"] == "AWY"


def test_known_denominator_and_per_fastball_type_aggregates() -> None:
    pitches = pd.DataFrame(
        _pitch_rows(
            pitcher=10,
            game_pk=100,
            game_date="2024-04-01",
            pitch_types=["FF", "FC", "SL", "CH", "XX"],
            explicit_starter=1,
        )
    )
    row = aggregate_statcast_outings(pitches).iloc[0]

    assert row["pitch_count"] == 5
    assert row["known_pitch_count"] == 4
    assert row["known_pitch_share"] == pytest.approx(4 / 5)
    assert row["fastball_share"] == pytest.approx(2 / 4)
    assert row["breaking_share"] == pytest.approx(1 / 4)
    assert row["offspeed_share"] == pytest.approx(1 / 4)
    assert (
        row["fastball_share"]
        + row["breaking_share"]
        + row["offspeed_share"]
    ) == pytest.approx(1.0)
    assert row["ff_count"] == 1
    assert row["ff_velocity"] == pytest.approx(91.0)
    assert row["si_count"] == 0
    assert pd.isna(row["si_velocity"])
    # Legacy all-pitch values remain available only for tabular reproduction.
    assert row["release_speed"] == pytest.approx(93.0)


def test_cutter_category_is_configurable_but_never_primary() -> None:
    fastball = default_pitch_type_mapping("fastball")
    breaking = default_pitch_type_mapping("breaking")

    assert "FC" in fastball["fastball_share_types"]
    assert "FC" in breaking["breaking_types"]
    assert "FC" not in fastball["primary_fastball_candidates"]
    assert "FC" not in breaking["primary_fastball_candidates"]


def test_short_start_remains_in_cross_season_history() -> None:
    outings = pd.DataFrame(
        {
            "pitcher": [10, 10, 10],
            "game_pk": [100, 101, 102],
            "game_date": ["2023-09-30", "2024-04-05", "2024-04-10"],
            "pitch_count": [60, 40, 70],
            "stuff_plus": [100.0, 110.0, 120.0],
            "history_eligible": [True, True, True],
        }
    )
    featured = add_history_features(outings)
    short = featured.loc[featured["game_pk"].eq(101)].iloc[0]
    following = featured.loc[featured["game_pk"].eq(102)].iloc[0]

    assert not bool(short["target_eligible"])
    assert bool(short["history_eligible"])
    assert bool(short["season_start"])
    assert bool(short["long_gap"])
    assert short["rest_days_log"] == pytest.approx(np.log1p(30))
    assert bool(following["target_eligible"])
    assert following["prior_stuff_plus"] == pytest.approx(110.0)
    assert following["prev_start_pitch_count"] == pytest.approx(40.0)
    assert following["pitcher_prior_outing_count"] == 2


def test_history_is_strictly_before_target_date() -> None:
    outings = pd.DataFrame(
        {
            "pitcher": [10, 10, 10],
            "game_pk": [100, 101, 102],
            "game_date": ["2024-04-01", "2024-04-08", "2024-04-08"],
            "pitch_count": [60, 65, 70],
            "stuff_plus": [90.0, 200.0, 300.0],
            "history_eligible": [True, True, True],
        }
    )
    featured = add_history_features(outings)
    same_date = featured.loc[
        featured["game_date"].eq(pd.Timestamp("2024-04-08"))
    ]

    assert same_date["prior_stuff_plus"].tolist() == [90.0, 90.0]
    assert same_date["ewma4"].tolist() == [90.0, 90.0]
    assert same_date["pitcher_prior_outing_count"].tolist() == [1.0, 1.0]


def test_stuff_alias_merge_keeps_missing_and_short_starts() -> None:
    outings = pd.DataFrame(
        {
            "pitcher": [10, 10, 10],
            "game_pk": [100, 101, 102],
            "game_date": pd.to_datetime(
                ["2024-04-01", "2024-04-08", "2024-04-15"]
            ),
            "pitch_count": [60, 40, 70],
            "history_eligible": [True, True, True],
        }
    )
    stuff = pd.DataFrame(
        {
            "pitcher_id": [10, 10],
            "gamePk": [100, 101],
            "date": ["2024-04-01", "2024-04-08"],
            "sp_stuff": [101.0, 99.0],
            "GS": [1, 1],
        }
    )
    merged = merge_stuff_plus(outings, stuff)

    assert len(merged) == 3
    assert merged["target_eligible"].tolist() == [True, False, False]
    assert merged["stuff_plus"].equals(merged["sp_stuff"])
    assert merged["stuff_plus"].equals(merged["target_y"])
    assert merged["history_eligible"].all()
    assert merged["GS"].tolist()[:2] == [1.0, 1.0]


def test_partial_schema_stuff_rows_use_game_then_date_fallback() -> None:
    outings = pd.DataFrame(
        {
            "pitcher": [10, 10, 10],
            "game_pk": [100, 101, 102],
            "game_date": pd.to_datetime(
                ["2024-04-01", "2024-04-08", "2024-04-15"]
            ),
            "pitch_count": [60, 60, 60],
            "history_eligible": [True, True, True],
        }
    )
    # Row 1 has canonical identifiers, row 2 is date-only through aliases,
    # and row 3 has a non-matching game id that must fall back to its date.
    stuff = pd.DataFrame(
        {
            "pitcher": [10, pd.NA, 10],
            "pitcher_id": [pd.NA, 10, pd.NA],
            "game_pk": [100, pd.NA, 999],
            "game_date": [pd.NA, pd.NA, "2024-04-15"],
            "date": ["2024-04-01", "2024-04-08", pd.NA],
            "stuff_plus": [pd.NA, pd.NA, pd.NA],
            "sp_stuff": [101.0, 102.0, 103.0],
            "GS": [1, 1, 1],
        }
    )

    merged = merge_stuff_plus(outings, stuff)

    assert merged["stuff_plus"].tolist() == [101.0, 102.0, 103.0]
    assert merged["target_eligible"].all()


def test_date_fallback_rejects_ambiguous_doubleheader() -> None:
    outings = pd.DataFrame(
        {
            "pitcher": [10, 10],
            "game_pk": [100, 101],
            "game_date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "pitch_count": [60, 60],
            "history_eligible": [True, True],
        }
    )
    date_only = pd.DataFrame(
        {
            "pitcher": [10],
            "game_date": ["2024-06-01"],
            "sp_stuff": [101.0],
            "GS": [1],
        }
    )
    with pytest.raises(ValueError, match="doubleheader"):
        merge_stuff_plus(outings, date_only)

    exact = pd.DataFrame(
        {
            "pitcher": [10, 10],
            "game_pk": [100, 101],
            "game_date": ["2024-06-01", "2024-06-01"],
            "sp_stuff": [101.0, 102.0],
            "GS": [1, 1],
        }
    )
    matched = merge_stuff_plus(outings, exact)
    assert matched["stuff_plus"].tolist() == [101.0, 102.0]


def test_inferred_first_inning_bulk_reliever_is_reconciled_by_gs(
    tmp_path,
) -> None:
    rows = []
    rows += _pitch_rows(
        pitcher=10,
        game_pk=100,
        game_date="2025-05-10",
        pitch_types=["FF"] * 60,
    )
    # In a pitcher-only extract this first-inning bulk appearance is
    # indistinguishable from a start without an external GS source.
    rows += _pitch_rows(
        pitcher=10,
        game_pk=101,
        game_date="2025-05-17",
        pitch_types=["FF"] * 62,
    )
    statcast_path = tmp_path / "pitcher.csv"
    pd.DataFrame(rows).to_csv(statcast_path, index=False)
    stuff_path = tmp_path / "starter_stuff.csv"
    pd.DataFrame(
        {
            "pitcher": [10],
            "game_date": ["2025-05-10"],
            "sp_stuff": [100.0],
            "GS": [1],
        }
    ).to_csv(stuff_path, index=False)

    with pytest.warns(StarterInferenceWarning):
        dataset = build_stuff_mlb_dataset(statcast_path, stuff_path)
    starts = dataset.set_index("game_pk")
    assert bool(starts.at[100, "history_eligible"])
    assert not bool(starts.at[101, "history_eligible"])
    assert starts.at[101, "is_starting_pitcher"] == 0
    assert not bool(starts.at[101, "target_eligible"])

    with pytest.raises(ValueError, match="cannot be proven"):
        build_stuff_mlb_dataset(
            statcast_path,
            stuff_path,
            starter_inference="strict",
        )


def test_dataset_build_warns_once_across_many_statcast_files(tmp_path) -> None:
    statcast_paths = []
    stuff_rows = []
    for offset in range(3):
        game_pk = 200 + offset
        game_date = (
            pd.Timestamp("2024-04-01") + pd.Timedelta(days=7 * offset)
        ).date().isoformat()
        path = tmp_path / f"statcast_{offset}.csv"
        pd.DataFrame(
            _pitch_rows(
                pitcher=10,
                game_pk=game_pk,
                game_date=game_date,
                pitch_types=["FF"] * 5,
            )
        ).to_csv(path, index=False)
        statcast_paths.append(path)
        stuff_rows.append(
            {
                "pitcher": 10,
                "game_date": game_date,
                "sp_stuff": 100.0 + offset,
                "GS": 1,
            }
        )
    stuff_path = tmp_path / "stuff.csv"
    pd.DataFrame(stuff_rows).to_csv(stuff_path, index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", StarterInferenceWarning)
        dataset = build_stuff_mlb_dataset(statcast_paths, stuff_path)
    starter_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, StarterInferenceWarning)
    ]

    assert len(dataset) == 3
    assert len(starter_warnings) == 1

    # Direct calls intentionally keep their per-call warning behavior.
    direct_input = pd.DataFrame(
        _pitch_rows(
            pitcher=10,
            game_pk=999,
            game_date="2024-09-01",
            pitch_types=["FF"],
        )
    )
    with warnings.catch_warnings(record=True) as direct:
        warnings.simplefilter("always", StarterInferenceWarning)
        build_outings(direct_input)
        build_outings(direct_input)
    assert sum(
        issubclass(warning.category, StarterInferenceWarning)
        for warning in direct
    ) == 2


def test_partial_starter_status_schema_is_attached_rowwise(tmp_path) -> None:
    rows = []
    rows += _pitch_rows(
        pitcher=10,
        game_pk=100,
        game_date="2024-04-01",
        pitch_types=["FF"] * 5,
    )
    rows += _pitch_rows(
        pitcher=10,
        game_pk=101,
        game_date="2024-04-08",
        pitch_types=["FF"] * 5,
    )
    statcast = tmp_path / "statcast.csv"
    pd.DataFrame(rows).to_csv(statcast, index=False)

    status_game = tmp_path / "status_game.csv"
    pd.DataFrame(
        {
            "pitcher": [10],
            "game_pk": [100],
            "game_date": ["2024-04-01"],
            "is_starting_pitcher": [1],
        }
    ).to_csv(status_game, index=False)
    status_date = tmp_path / "status_date.csv"
    pd.DataFrame(
        {
            "pitcher_id": [10],
            "date": ["2024-04-08"],
            "GS": [1],
        }
    ).to_csv(status_date, index=False)
    stuff = tmp_path / "stuff.csv"
    pd.DataFrame(
        {
            "pitcher": [10, 10],
            "game_date": ["2024-04-01", "2024-04-08"],
            "sp_stuff": [100.0, 101.0],
            "GS": [1, 1],
        }
    ).to_csv(stuff, index=False)

    dataset = build_stuff_mlb_dataset(
        statcast,
        stuff,
        starter_status_paths=[status_game, status_date],
        starter_inference="strict",
    )
    assert len(dataset) == 2
    assert dataset["history_eligible"].all()


def test_fangraphs_helpers_coalesce_aliases_and_keep_partial_identities() -> None:
    raw = pd.DataFrame(
        {
            "stuff_plus": [pd.NA, 102.0],
            "sp_stuff": [101.0, pd.NA],
        }
    )
    values = _coalesce_columns(raw, ("stuff_plus", "sp_stuff"))
    assert values is not None
    assert values.tolist() == [101.0, 102.0]

    logs = pd.DataFrame(
        {
            "pitcher": [10, 10],
            "game_pk": [100, pd.NA],
            "game_date": pd.to_datetime(["2024-04-01", "2024-04-08"]),
            "stuff_plus": [101.0, 102.0],
        }
    )
    assert len(_deduplicate_logs(logs)) == 2

    ambiguous = pd.concat(
        [logs.iloc[[1]], logs.iloc[[1]].assign(stuff_plus=999.0)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="Ambiguous FanGraphs"):
        _deduplicate_logs(ambiguous)


def test_legacy_and_tcn_feature_contracts_are_separate() -> None:
    assert len(FEATURES) == 28
    assert len(COMPACT_FEATURES) == 16
    assert "release_speed_slope5" in COMPACT_FEATURES
    assert "release_speed" not in TCN_RAW_FEATURES
    assert "primary_fb_velocity" in TCN_RAW_FEATURES
    assert "fastball_share" in TCN_RAW_FEATURES
    assert "offspeed_share" not in TCN_RAW_FEATURES


def test_end_to_end_accepts_names_only_official_stats(tmp_path) -> None:
    pitch_rows = []
    for game_pk, game_date, count in [
        (100, "2024-04-01", 55),
        (101, "2024-04-08", 40),
        (102, "2024-04-15", 60),
    ]:
        pitch_rows += _pitch_rows(
            pitcher=10,
            game_pk=game_pk,
            game_date=game_date,
            pitch_types=["FF"] * count,
            explicit_starter=1,
        )
    statcast_path = tmp_path / "statcast.csv"
    pd.DataFrame(pitch_rows).to_csv(statcast_path, index=False)
    stuff_path = tmp_path / "stuff.csv"
    pd.DataFrame(
        {
            "pitcher": [10, 10, 10],
            "game_pk": [100, 101, 102],
            "game_date": ["2024-04-01", "2024-04-08", "2024-04-15"],
            "sp_stuff": [100.0, 105.0, 103.0],
            "GS": [1, 1, 1],
        }
    ).to_csv(stuff_path, index=False)
    names_path = tmp_path / "names.csv"
    pd.DataFrame({"player_id": [10], "name": ["Pitcher Ten"]}).to_csv(
        names_path, index=False
    )

    dataset = build_stuff_mlb_dataset(
        statcast_path,
        stuff_path,
        official_stats_path=names_path,
    )

    assert len(dataset) == 3
    assert dataset["target_eligible"].tolist() == [True, False, True]
    assert dataset.loc[2, "prior_stuff_plus"] == pytest.approx(105.0)
    assert dataset["pitcher_name"].unique().tolist() == ["Pitcher Ten"]
    assert set(COMPACT_FEATURES).issubset(dataset.columns)


def test_make_dataset_keeps_original_signature_and_qualification(tmp_path) -> None:
    pitches: list[dict[str, object]] = []
    stuff_rows: list[dict[str, object]] = []
    game_pk = 1000
    for year in range(2021, 2026):
        for index in range(20):
            game_pk += 1
            date = (
                pd.Timestamp(year=year, month=4, day=1)
                + pd.Timedelta(days=7 * index)
            ).date().isoformat()
            pitches += _pitch_rows(
                pitcher=10,
                game_pk=game_pk,
                game_date=date,
                pitch_types=["FF"] * 50,
                explicit_starter=1,
            )
            stuff_rows.append(
                {
                    "pitcher": 10,
                    "game_pk": game_pk,
                    "game_date": date,
                    "sp_stuff": 100.0 + index,
                    "GS": 1,
                }
            )
    # The 40-pitch start is not a target, but remains in the returned history.
    game_pk += 1
    pitches += _pitch_rows(
        pitcher=10,
        game_pk=game_pk,
        game_date="2024-09-01",
        pitch_types=["FF"] * 40,
        explicit_starter=1,
    )
    stuff_rows.append(
        {
            "pitcher": 10,
            "game_pk": game_pk,
            "game_date": "2024-09-01",
            "sp_stuff": 123.0,
            "GS": 1,
        }
    )
    statcast_dir = tmp_path / "statcast"
    statcast_dir.mkdir()
    pd.DataFrame(pitches).to_parquet(statcast_dir / "pitcher.parquet")
    stuff_path = tmp_path / "stuff.parquet"
    pd.DataFrame(stuff_rows).to_parquet(stuff_path)

    data, qualified = make_dataset(statcast_dir, stuff_path)

    assert qualified == [10]
    assert len(data) == 101
    short = data.loc[data["game_pk"].eq(game_pk)].iloc[0]
    assert bool(short["history_eligible"])
    assert not bool(short["target_eligible"])
