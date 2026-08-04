"""Shared data loading and roster metadata for the Stuff+ workflow.

The historical research roster and model selection answer different questions:

* roster qualification keeps the published rule of 20 eligible official starts
  in every season from 2021 through 2025;
* model selection scores only the expanding 2022, 2023, and 2024 validation
  folds.  The 2025 target values are reserved for the locked final evaluation.

Synthetic and prepared data can use an explicitly labelled development-mode
qualification so small integration tests do not need 100 outings per pitcher.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import lib.stuff_mlb_dataset as stuff_dataset
from lib.stuff_demo_data import make_demo_stuff_outings
from lib.temporal_splits import add_row_id, dataframe_fingerprint


RESEARCH_QUALIFICATION_YEARS = (2021, 2022, 2023, 2024, 2025)
SELECTION_VALIDATION_YEARS = (2022, 2023, 2024)
FINAL_TEST_YEAR = 2025

DATASET_FINGERPRINT_CORE_COLUMNS = (
    "row_id",
    "pitcher",
    "game_pk",
    "game_date",
    "year",
    "pitch_count",
    "stuff_plus",
    "sp_stuff",
    "target_y",
    "history_eligible",
    "target_eligible",
    "is_starting_pitcher",
    "game_type",
    "fastball_share",
    "breaking_share",
    "offspeed_share",
    "known_pitch_share",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "arm_angle",
    "pfx_x",
    "pfx_z",
    "ff_count",
    "ff_velocity",
    "ff_spin",
    "ff_pfx_x",
    "ff_pfx_z",
    "si_count",
    "si_velocity",
    "si_spin",
    "si_pfx_x",
    "si_pfx_z",
    "fa_count",
    "fa_velocity",
    "fa_spin",
    "fa_pfx_x",
    "fa_pfx_z",
    "rest_days",
    "rest_days_capped",
    "rest_days_log",
    "season_start",
    "long_gap",
    "ewma4",
)

_PITCHER_ID_COLUMNS = ("pitcher", "player_id", "pitcher_id", "mlbam_id")
_PITCHER_NAME_COLUMNS = ("pitcher_name", "name", "player_name")
_STARTER_COLUMNS = (
    "is_starting_pitcher",
    "is_starter",
    "starter",
    "starting_pitcher",
    "GS",
)


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    return value


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    exact = set(frame.columns)
    folded = {str(column).casefold(): str(column) for column in frame.columns}
    for name in names:
        if name in exact:
            return name
        match = folded.get(name.casefold())
        if match is not None:
            return match
    return None


def _as_paths(paths: Sequence[str | Path] | None) -> tuple[Path, ...]:
    return tuple(Path(path) for path in (paths or ()))


def _read_paths(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Read multiple CSV/parquet files or directories deterministically."""

    requested = _as_paths(paths)
    if not requested:
        raise ValueError("At least one input path is required")
    files: list[Path] = []
    for path in requested:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            discovered = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in {".csv", ".parquet"}
            )
            if not discovered:
                raise FileNotFoundError(f"No CSV/parquet files found under {path}")
            files.extend(discovered)
        elif path.suffix.lower() in {".csv", ".parquet"}:
            files.append(path)
        else:
            raise ValueError(f"Unsupported input file type: {path}")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve()).casefold()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    frames = [
        (
            pd.read_parquet(path)
            if path.suffix.lower() == ".parquet"
            else pd.read_csv(path, low_memory=False)
        )
        for path in unique
    ]
    if not frames:
        raise ValueError("No readable input tables were found")
    return pd.concat(frames, ignore_index=True, sort=False)


def read_pitcher_names(
    official_stats_paths: Sequence[str | Path] | None,
) -> dict[Any, str]:
    """Read player-id/name metadata without treating it as starter evidence."""

    paths = _as_paths(official_stats_paths)
    if not paths:
        return {}
    frame = _read_paths(paths)
    pitcher_col = _first_column(frame, _PITCHER_ID_COLUMNS)
    name_col = _first_column(frame, _PITCHER_NAME_COLUMNS)
    if pitcher_col is None or name_col is None:
        raise ValueError(
            "--official-stats is name metadata and must contain a pitcher/player_id "
            "column and a name/pitcher_name column"
        )
    names: dict[Any, str] = {}
    for pitcher, name in frame[[pitcher_col, name_col]].itertuples(
        index=False, name=None
    ):
        if pd.isna(pitcher) or pd.isna(name):
            continue
        text = str(name).strip()
        if text:
            names[_python_scalar(pitcher)] = text
    return names


def validate_starter_status_paths(
    starter_status_paths: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    """Validate an explicit row-level starter-status source.

    A season-level player-name file is not sufficient.  The separate input must
    contain pitcher, game date, and an unambiguous starter flag.
    """

    paths = _as_paths(starter_status_paths)
    if not paths:
        return ()
    frame = _read_paths(paths)
    pitcher_col = _first_column(frame, _PITCHER_ID_COLUMNS)
    date_col = _first_column(frame, ("game_date", "date", "gameDate"))
    starter_col = _first_column(frame, _STARTER_COLUMNS)
    missing = [
        label
        for label, value in (
            ("pitcher/player_id", pitcher_col),
            ("game_date", date_col),
            ("is_starting_pitcher/GS", starter_col),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "--starter-status must be row-level starter data; missing "
            + ", ".join(missing)
        )
    if pd.to_datetime(frame[date_col], errors="coerce").isna().any():
        raise ValueError("--starter-status contains invalid game_date values")
    return paths


@dataclass(frozen=True)
class QualificationResult:
    qualified_pitchers: tuple[Any, ...]
    qualification_rule: str
    fit_years: tuple[int, ...]
    mode: str
    counts_by_pitcher_year: Mapping[str, Mapping[str, int]]
    uses_2025_roster_availability: bool
    selection_validation_years: tuple[int, ...] = SELECTION_VALIDATION_YEARS
    selection_uses_2025_outcomes: bool = False


def qualify_pitchers(
    outings: pd.DataFrame,
    *,
    mode: str = "research",
    qualification_years: Sequence[int] = RESEARCH_QUALIFICATION_YEARS,
    min_starts_per_year: int = 20,
    train_start: int = 2020,
    development_end_year: int = 2024,
    min_development_outings: int = 1,
) -> QualificationResult:
    """Apply either the published roster rule or an explicit test-mode rule."""

    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"research", "development"}:
        raise ValueError("qualification mode must be research or development")
    required = {"pitcher", "year", "target_eligible"}
    missing = sorted(required - set(outings.columns))
    if missing:
        raise ValueError(f"Cannot qualify pitchers; missing columns: {missing}")

    years = pd.to_numeric(outings["year"], errors="coerce")
    eligible = outings["target_eligible"].fillna(False).astype(bool)
    if "history_eligible" in outings:
        eligible &= outings["history_eligible"].fillna(False).astype(bool)
    if "is_starting_pitcher" in outings:
        eligible &= pd.to_numeric(
            outings["is_starting_pitcher"], errors="coerce"
        ).eq(1)
    if "game_type" in outings:
        eligible &= (
            outings["game_type"].astype("string").str.strip().str.upper().eq("R")
        )

    if resolved_mode == "research":
        roster_years = tuple(int(year) for year in qualification_years)
        if not roster_years:
            raise ValueError("qualification_years cannot be empty")
        if int(min_starts_per_year) < 1:
            raise ValueError("min_starts_per_year must be positive")
        fit = outings.loc[
            eligible & years.isin(roster_years), ["pitcher", "year"]
        ].copy()
        counts = (
            fit.groupby(["pitcher", "year"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        for year in roster_years:
            if year not in counts:
                counts[year] = 0
        qualified_index = counts.index[
            counts[list(roster_years)].ge(int(min_starts_per_year)).all(axis=1)
        ]
        rule = (
            f">= {int(min_starts_per_year)} target_eligible official "
            f"regular-season starts in every year {roster_years[0]}-"
            f"{roster_years[-1]}; this is the historical roster rule, not a "
            "model-selection score"
        )
        fit_years = roster_years
        uses_2025 = FINAL_TEST_YEAR in roster_years
    else:
        if int(min_development_outings) < 1:
            raise ValueError("min_development_outings must be positive")
        fit = outings.loc[
            eligible & years.between(int(train_start), int(development_end_year)),
            ["pitcher", "year"],
        ].copy()
        counts = (
            fit.groupby(["pitcher", "year"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        totals = counts.sum(axis=1) if not counts.empty else pd.Series(dtype=int)
        qualified_index = totals.loc[
            totals.ge(int(min_development_outings))
        ].index
        rule = (
            f">= {int(min_development_outings)} target_eligible outing(s) in "
            f"{int(train_start)}-{int(development_end_year)}; relaxed "
            "development/demo rule"
        )
        fit_years = tuple(
            sorted(int(value) for value in fit["year"].dropna().unique())
        )
        uses_2025 = False

    qualified = tuple(
        sorted(
            (_python_scalar(value) for value in qualified_index),
            key=lambda value: (str(type(value)), str(value)),
        )
    )
    if not qualified:
        raise ValueError(f"No pitchers satisfy the {resolved_mode} qualification rule")

    count_metadata: dict[str, dict[str, int]] = {}
    for pitcher in qualified_index:
        if pitcher not in counts.index:
            continue
        count_metadata[str(_python_scalar(pitcher))] = {
            str(int(year)): int(value)
            for year, value in counts.loc[pitcher].items()
        }
    return QualificationResult(
        qualified_pitchers=qualified,
        qualification_rule=rule,
        fit_years=fit_years,
        mode=resolved_mode,
        counts_by_pitcher_year=count_metadata,
        uses_2025_roster_availability=uses_2025,
    )


@dataclass(frozen=True)
class StuffDataBundle:
    outings: pd.DataFrame
    qualified_pitchers: tuple[Any, ...]
    qualification_rule: str
    qualification_fit_years: tuple[int, ...]
    dataset_fingerprint: str
    fingerprint_columns: tuple[str, ...]
    source_mode: str
    pitcher_names: Mapping[Any, str] = field(default_factory=dict)
    qualification_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "qualified_pitchers": list(self.qualified_pitchers),
            "qualified_pitcher_count": len(self.qualified_pitchers),
            "qualification_rule": self.qualification_rule,
            "qualification_fit_years": list(self.qualification_fit_years),
            "qualification": dict(self.qualification_metadata),
            "selection_validation_years": list(SELECTION_VALIDATION_YEARS),
            "selection_uses_2025_outcomes": False,
            "final_test_year": FINAL_TEST_YEAR,
            "dataset_fingerprint": self.dataset_fingerprint,
            "fingerprint_columns": list(self.fingerprint_columns),
            "pitcher_names": {
                str(key): value for key, value in self.pitcher_names.items()
            },
            "source": dict(self.source_metadata),
            "n_rows": int(len(self.outings)),
        }


def dataset_fingerprint(
    outings: pd.DataFrame,
    *,
    core_columns: Sequence[str] = DATASET_FINGERPRINT_CORE_COLUMNS,
) -> tuple[str, tuple[str, ...]]:
    required = ("row_id", "pitcher", "game_pk", "game_date", "year")
    missing = [column for column in required if column not in outings]
    if missing:
        raise ValueError(f"Cannot fingerprint Stuff+ data; missing columns: {missing}")
    selected = [column for column in core_columns if column in outings]
    feature_contract = [
        *getattr(stuff_dataset, "FEATURES", ()),
        *getattr(stuff_dataset, "COMPACT_FEATURES", ()),
        *getattr(stuff_dataset, "TCN_RAW_FEATURES", ()),
    ]
    selected.extend(
        column
        for column in feature_contract
        if column in outings and column not in selected
    )
    columns = tuple(selected)
    return dataframe_fingerprint(outings, columns=columns), columns


def _ensure_canonical_outings(
    outings: pd.DataFrame,
    *,
    ewma_span: int,
) -> pd.DataFrame:
    out = outings.copy()
    required = {"pitcher", "game_pk", "game_date", "pitch_count"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Outing data is missing canonical columns: {missing}")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if out["game_date"].isna().any():
        raise ValueError("Outing data contains invalid game_date values")
    out["year"] = out["game_date"].dt.year.astype(int)

    target_source = next(
        (
            column
            for column in ("target_y", "stuff_plus", "sp_stuff")
            if column in out
        ),
        None,
    )
    if target_source is None:
        raise ValueError("Outing data needs target_y, stuff_plus, or sp_stuff")
    target = pd.to_numeric(out[target_source], errors="coerce")
    out["target_y"] = target
    out["stuff_plus"] = target
    out["sp_stuff"] = target

    if "history_eligible" not in out:
        if "is_starting_pitcher" not in out or "game_type" not in out:
            raise ValueError(
                "Prepared outings need history_eligible or explicit "
                "is_starting_pitcher and game_type columns"
            )
        out["history_eligible"] = (
            pd.to_numeric(out["is_starting_pitcher"], errors="coerce").eq(1)
            & out["game_type"].astype("string").str.upper().eq("R")
        )
    out["history_eligible"] = out["history_eligible"].fillna(False).astype(bool)
    if "target_eligible" not in out:
        out["target_eligible"] = (
            out["history_eligible"]
            & pd.to_numeric(out["pitch_count"], errors="coerce").ge(50)
            & target.notna()
        )
    out["target_eligible"] = out["target_eligible"].fillna(False).astype(bool)

    add_history = getattr(stuff_dataset, "add_history_features", None)
    if callable(add_history):
        parameters = inspect.signature(add_history).parameters
        kwargs = {"ewma_span": int(ewma_span)} if "ewma_span" in parameters else {}
        out = add_history(out, **kwargs)
    out = add_row_id(out, verify_existing=True, require_unique=True)
    return out.sort_values(
        ["pitcher", "game_date", "game_pk", "row_id"], kind="mergesort"
    ).reset_index(drop=True)


def _build_raw_dataset(
    statcast_paths: Sequence[Path],
    stuff_paths: Sequence[Path],
    *,
    starter_status_paths: Sequence[Path],
    cutter_category: str,
) -> pd.DataFrame:
    builder = getattr(stuff_dataset, "build_stuff_mlb_dataset", None)
    if callable(builder):
        parameters = inspect.signature(builder).parameters
        kwargs: dict[str, Any] = {}
        if "cutter_category" in parameters:
            kwargs["cutter_category"] = cutter_category
        if starter_status_paths:
            for argument in (
                "starter_status_paths",
                "starter_status_path",
                "official_starter_paths",
                "official_starter_path",
                "official_stats_path",
            ):
                if argument in parameters:
                    kwargs[argument] = tuple(starter_status_paths)
                    break
            else:
                raise TypeError(
                    "The dataset builder cannot accept explicit --starter-status data"
                )
        return builder(tuple(statcast_paths), tuple(stuff_paths), **kwargs)

    legacy_builder = getattr(stuff_dataset, "make_dataset", None)
    if not callable(legacy_builder):
        raise AttributeError("No public Stuff+ dataset builder is available")
    if starter_status_paths:
        raise TypeError(
            "The legacy dataset builder cannot accept explicit --starter-status data"
        )
    result = legacy_builder(tuple(statcast_paths), tuple(stuff_paths))
    return result[0] if isinstance(result, tuple) else result


def _path_strings(paths: Sequence[Path]) -> list[str]:
    return [str(path) for path in paths]


def load_stuff_data(
    *,
    statcast_paths: Sequence[str | Path] | None = None,
    stuff_paths: Sequence[str | Path] | None = None,
    official_stats_paths: Sequence[str | Path] | None = None,
    starter_status_paths: Sequence[str | Path] | None = None,
    prepared_outings_paths: Sequence[str | Path] | None = None,
    demo: bool = False,
    cutter_category: str = "fastball",
    ewma_span: int = 4,
    train_start: int = 2020,
    qualification_mode: str = "auto",
    qualification_years: Sequence[int] = RESEARCH_QUALIFICATION_YEARS,
    min_starts_per_year: int = 20,
    development_end_year: int = 2024,
    min_development_outings: int = 1,
    demo_pitchers: int = 3,
    demo_outings_per_season: int = 12,
    demo_seed: int = 20260722,
) -> StuffDataBundle:
    """Load exactly one of raw, prepared, or deterministic demo data."""

    statcast = _as_paths(statcast_paths)
    stuff = _as_paths(stuff_paths)
    prepared = _as_paths(prepared_outings_paths)
    starter = validate_starter_status_paths(starter_status_paths)
    raw_requested = bool(statcast or stuff)
    mode_count = int(bool(demo)) + int(bool(prepared)) + int(raw_requested)
    if mode_count != 1:
        raise ValueError(
            "Choose exactly one data mode: --demo, --prepared-outings, or "
            "--statcast-dir with --stuff"
        )

    pitcher_names = read_pitcher_names(official_stats_paths)
    if demo:
        if starter:
            raise ValueError("--starter-status is only valid with raw Statcast data")
        outings = make_demo_stuff_outings(
            pitchers=int(demo_pitchers),
            outings_per_season=int(demo_outings_per_season),
            start_year=int(train_start),
            end_year=FINAL_TEST_YEAR,
            seed=int(demo_seed),
        )
        source_mode = "demo"
    elif prepared:
        if starter:
            raise ValueError("--starter-status is only valid with raw Statcast data")
        outings = _read_paths(prepared)
        source_mode = "prepared_outings"
    else:
        if not statcast or not stuff:
            raise ValueError("Raw mode requires both --statcast-dir and --stuff")
        outings = _build_raw_dataset(
            statcast,
            stuff,
            starter_status_paths=starter,
            cutter_category=cutter_category,
        )
        source_mode = "raw"

    canonical = _ensure_canonical_outings(outings, ewma_span=int(ewma_span))
    resolved_qualification = (
        ("research" if source_mode == "raw" else "development")
        if qualification_mode == "auto"
        else qualification_mode
    )
    qualification = qualify_pitchers(
        canonical,
        mode=resolved_qualification,
        qualification_years=qualification_years,
        min_starts_per_year=int(min_starts_per_year),
        train_start=int(train_start),
        development_end_year=int(development_end_year),
        min_development_outings=int(min_development_outings),
    )
    qualified_keys = {str(value) for value in qualification.qualified_pitchers}
    filtered = canonical.loc[
        canonical["pitcher"].map(lambda value: str(_python_scalar(value))).isin(
            qualified_keys
        )
    ].copy()
    name_by_key = {str(_python_scalar(key)): value for key, value in pitcher_names.items()}
    if name_by_key:
        existing_names = (
            filtered["pitcher_name"]
            if "pitcher_name" in filtered
            else pd.Series(pd.NA, index=filtered.index, dtype="string")
        )
        mapped_names = filtered["pitcher"].map(
            lambda value: name_by_key.get(str(_python_scalar(value)))
        )
        filtered["pitcher_name"] = mapped_names.where(
            mapped_names.notna(), existing_names
        )

    fingerprint, fingerprint_columns = dataset_fingerprint(filtered)
    qualification_metadata = {
        "mode": qualification.mode,
        "fit_years": list(qualification.fit_years),
        "uses_2025_roster_availability": (
            qualification.uses_2025_roster_availability
        ),
        "selection_validation_years": list(
            qualification.selection_validation_years
        ),
        "selection_uses_2025_outcomes": (
            qualification.selection_uses_2025_outcomes
        ),
        "counts_by_pitcher_year": dict(
            qualification.counts_by_pitcher_year
        ),
    }
    source_metadata = {
        "mode": source_mode,
        "statcast_paths": _path_strings(statcast),
        "stuff_paths": _path_strings(stuff),
        "official_stats_name_paths": _path_strings(
            _as_paths(official_stats_paths)
        ),
        "starter_status_paths": _path_strings(starter),
        "prepared_outings_paths": _path_strings(prepared),
        "cutter_category": cutter_category,
    }
    filtered.attrs.update(
        {
            "dataset_fingerprint": fingerprint,
            "dataset_fingerprint_columns": fingerprint_columns,
            "qualified_pitchers": qualification.qualified_pitchers,
            "qualification_rule": qualification.qualification_rule,
            "qualification_fit_years": qualification.fit_years,
            "qualification_metadata": qualification_metadata,
            "source_metadata": source_metadata,
        }
    )
    return StuffDataBundle(
        outings=filtered.reset_index(drop=True),
        qualified_pitchers=qualification.qualified_pitchers,
        qualification_rule=qualification.qualification_rule,
        qualification_fit_years=qualification.fit_years,
        dataset_fingerprint=fingerprint,
        fingerprint_columns=fingerprint_columns,
        source_mode=source_mode,
        pitcher_names=pitcher_names,
        qualification_metadata=qualification_metadata,
        source_metadata=source_metadata,
    )


def add_stuff_data_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--statcast-dir", nargs="+", type=Path, default=None)
    parser.add_argument("--stuff", nargs="+", type=Path, default=None)
    parser.add_argument(
        "--official-stats",
        nargs="+",
        type=Path,
        default=None,
        help="Player-id/name metadata only; never used as starter evidence.",
    )
    parser.add_argument(
        "--starter-status",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Optional explicit row-level starter table with pitcher, game_date, "
            "and is_starting_pitcher/GS."
        ),
    )
    parser.add_argument("--prepared-outings", nargs="+", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--cutter-category",
        choices=["fastball", "breaking", "offspeed", "other"],
        default="fastball",
    )
    parser.add_argument("--ewma-span", type=int, default=4)
    parser.add_argument("--train-start", type=int, default=2020)
    parser.add_argument(
        "--qualification-mode",
        choices=["auto", "research", "development"],
        default="auto",
        help="auto uses research for raw data and relaxed development for demo/prepared.",
    )
    parser.add_argument(
        "--qualification-years",
        nargs="+",
        type=int,
        default=list(RESEARCH_QUALIFICATION_YEARS),
    )
    parser.add_argument("--min-starts-per-year", type=int, default=20)
    parser.add_argument("--development-end-year", type=int, default=2024)
    parser.add_argument("--min-development-outings", type=int, default=1)
    parser.add_argument("--demo-pitchers", type=int, default=3)
    parser.add_argument("--demo-outings-per-season", type=int, default=12)
    parser.add_argument("--demo-seed", type=int, default=20260722)
    return parser


add_data_arguments = add_stuff_data_arguments


def load_stuff_data_from_args(args: argparse.Namespace) -> StuffDataBundle:
    return load_stuff_data(
        statcast_paths=getattr(args, "statcast_dir", None),
        stuff_paths=getattr(args, "stuff", None),
        official_stats_paths=getattr(args, "official_stats", None),
        starter_status_paths=getattr(args, "starter_status", None),
        prepared_outings_paths=getattr(args, "prepared_outings", None),
        demo=bool(getattr(args, "demo", False)),
        cutter_category=getattr(args, "cutter_category", "fastball"),
        ewma_span=int(getattr(args, "ewma_span", 4)),
        train_start=int(getattr(args, "train_start", 2020)),
        qualification_mode=getattr(args, "qualification_mode", "auto"),
        qualification_years=getattr(
            args, "qualification_years", RESEARCH_QUALIFICATION_YEARS
        ),
        min_starts_per_year=int(getattr(args, "min_starts_per_year", 20)),
        development_end_year=int(
            getattr(args, "development_end_year", 2024)
        ),
        min_development_outings=int(
            getattr(args, "min_development_outings", 1)
        ),
        demo_pitchers=int(getattr(args, "demo_pitchers", 3)),
        demo_outings_per_season=int(
            getattr(args, "demo_outings_per_season", 12)
        ),
        demo_seed=int(getattr(args, "demo_seed", 20260722)),
    )


load_dataset_from_args = load_stuff_data_from_args
