from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.build_outings import build_outings


# FC belongs to the fastball share, but is intentionally excluded from the
# primary-fastball candidates used by fold-local mapping.
FASTBALL_SHARE_TYPES = frozenset({"FF", "SI", "FC", "FA"})
PRIMARY_FASTBALL_CANDIDATES = frozenset({"FF", "SI", "FA"})
BREAKING_TYPES = frozenset({"SL", "ST", "CU", "KC", "SV", "CS"})
OFFSPEED_TYPES = frozenset({"CH", "FS", "FO", "SC"})
PRIMARY_FASTBALL_PREFIXES = ("ff", "si", "fa")
PRIMARY_FASTBALL_MEASUREMENTS = ("velocity", "spin", "pfx_x", "pfx_z")

# Backwards-compatible names used by the original Stuff+ scripts.
FASTBALL = set(FASTBALL_SHARE_TYPES)
BREAKING = set(BREAKING_TYPES)
OFFSPEED = set(OFFSPEED_TYPES)

# These all-pitch aggregates are retained only to reproduce the existing
# EWMA/Ridge/XGBoost feature table.  They are not part of TCN_RAW_FEATURES.
PHYSICAL_RAW = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "arm_angle",
    "pfx_x",
    "pfx_z",
]
FEATURES = [
    "prev_start_pitch_count",
    "rest_days",
    "workload_density_3starts",
    "prior_stuff_plus",
    "stuff_plus_mean_last5",
    "stuff_plus_slope_last5",
    *[
        value
        for column in PHYSICAL_RAW
        for value in (f"{column}_ma5", f"{column}_slope5")
    ],
    "spin_axis_sin_ma5",
    "spin_axis_cos_ma5",
    "breaking_share_ma5",
    "breaking_share_slope5",
    "offspeed_share_ma5",
    "offspeed_share_slope5",
]

# Exact span-4 compact feature names used by the original model scripts.
COMPACT_FEATURES = (
    "prior_minus_ewma4",
    "mean5_minus_ewma4",
    "stuff_plus_slope_last5",
    "prev_start_pitch_count",
    "rest_days",
    "workload_density_3starts",
    "release_speed_slope5",
    "release_spin_rate_slope5",
    "release_extension_slope5",
    "release_pos_x_slope5",
    "release_pos_z_slope5",
    "arm_angle_slope5",
    "pfx_x_slope5",
    "pfx_z_slope5",
    "breaking_share_slope5",
    "offspeed_share_slope5",
)
COMPACT_FEATURE_NAMES = COMPACT_FEATURES
COMPACT_16_FEATURES = COMPACT_FEATURES

# Raw prior-outing values consumed by the TCN sequence builder.  Primary
# fastball values are materialized later from a training-only FF/SI/FA mapping.
TCN_RAW_FEATURES = (
    "stuff_plus",
    "pitch_count",
    "rest_days_log",
    "long_gap",
    "season_start",
    "primary_fb_velocity",
    "primary_fb_spin",
    "primary_fb_pfx_x",
    "primary_fb_pfx_z",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "arm_angle",
    "fastball_share",
    "breaking_share",
)
TCN_SEQUENCE_FEATURES = TCN_RAW_FEATURES
TCN_BINARY_FEATURES = (
    "long_gap",
    "season_start",
    "primary_fb_available",
    "valid_timestep",
)

_IDENTIFIER_ALIASES = {
    "pitcher": (
        "pitcher",
        "pitcher_id",
        "player_id",
        "mlbam_id",
        "mlbamid",
        "MLBAMID",
    ),
    # FanGraphs' ``gameid`` is not guaranteed to be MLB's numeric game_pk, so
    # it is deliberately not accepted as an identity alias.
    "game_pk": ("game_pk", "gamePk", "game_id", "game_id_mlb"),
    "game_date": ("game_date", "gameDate", "gamedate", "date", "Date"),
}
_STUFF_PLUS_ALIASES = (
    "stuff_plus",
    "sp_stuff",
    "Stuff+",
    "StuffPlus",
    "stuffplus",
    "stuff",
    "Stf+",
)


def default_pitch_type_mapping(cutter_category: str = "fastball") -> dict[str, Any]:
    """Return the canonical pitch-family mapping.

    The cutter may be moved for an explicit ablation, but never becomes a
    primary-fastball candidate.
    """

    category = str(cutter_category).strip().lower()
    category = {
        "excluded": "other",
        "exclude": "other",
        "none": "other",
    }.get(category, category)
    if category not in {"fastball", "breaking", "offspeed", "other"}:
        raise ValueError(
            "cutter_category must be one of: fastball, breaking, offspeed, other"
        )

    fastball = set(FASTBALL_SHARE_TYPES) - {"FC"}
    breaking = set(BREAKING_TYPES) - {"FC"}
    offspeed = set(OFFSPEED_TYPES) - {"FC"}
    if category == "fastball":
        fastball.add("FC")
    elif category == "breaking":
        breaking.add("FC")
    elif category == "offspeed":
        offspeed.add("FC")
    return {
        "fastball_share_types": sorted(fastball),
        "breaking_types": sorted(breaking),
        "offspeed_types": sorted(offspeed),
        "primary_fastball_candidates": sorted(PRIMARY_FASTBALL_CANDIDATES),
        "cutter_category": category,
    }


def _normalise_pitch_type_mapping(
    mapping: Mapping[str, Any] | None,
) -> dict[str, set[str]]:
    raw = default_pitch_type_mapping() if mapping is None else mapping

    def values(names: Iterable[str], default: Iterable[str]) -> set[str]:
        selected: Any = None
        for name in names:
            if name in raw:
                selected = raw[name]
                break
        if selected is None:
            selected = default
        if isinstance(selected, str):
            selected = [selected]
        return {
            str(value).strip().upper()
            for value in selected
            if str(value).strip()
        }

    result = {
        "fastball": values(
            ("fastball_share_types", "fastball_types", "fastball"),
            FASTBALL_SHARE_TYPES,
        ),
        "breaking": values(("breaking_types", "breaking"), BREAKING_TYPES),
        "offspeed": values(("offspeed_types", "offspeed"), OFFSPEED_TYPES),
        "primary": values(
            ("primary_fastball_candidates", "primary_candidates"),
            PRIMARY_FASTBALL_CANDIDATES,
        ),
    }
    overlap = (
        (result["fastball"] & result["breaking"])
        | (result["fastball"] & result["offspeed"])
        | (result["breaking"] & result["offspeed"])
    )
    if overlap:
        raise ValueError(
            f"Pitch types cannot belong to multiple categories: {sorted(overlap)}"
        )
    return result


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    exact = set(frame.columns)
    folded = {str(column).casefold(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in exact:
            return str(alias)
        found = folded.get(str(alias).casefold())
        if found is not None:
            return found
    return None


def _matching_columns(
    frame: pd.DataFrame,
    aliases: Iterable[str],
) -> list[str]:
    """Return every matching alias in preference order, without duplicates."""

    columns = [str(column) for column in frame.columns]
    result: list[str] = []
    for alias in aliases:
        exact = [column for column in columns if column == str(alias)]
        folded = [
            column
            for column in columns
            if column.casefold() == str(alias).casefold()
        ]
        for column in [*exact, *folded]:
            if column not in result:
                result.append(column)
    return result


def _coalesce_alias_values(
    frame: pd.DataFrame,
    aliases: Iterable[str],
) -> pd.Series | None:
    """Coalesce aliases row by row, treating blank strings as missing."""

    columns = _matching_columns(frame, aliases)
    if not columns:
        return None
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in columns:
        values = frame[column]
        present = values.notna()
        if pd.api.types.is_object_dtype(values.dtype) or isinstance(
            values.dtype, pd.StringDtype
        ):
            present &= values.astype("string").str.strip().ne("")
        result = result.where(result.notna(), values.where(present))
    return result


def _as_paths(
    value: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value]


def _input_files(
    paths: str | Path | Sequence[str | Path],
) -> list[Path]:
    requested = _as_paths(paths)
    if not requested:
        raise ValueError("At least one input path is required.")
    files: list[Path] = []
    for path in requested:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            found = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in {".parquet", ".csv"}
            )
            if not found:
                raise FileNotFoundError(f"No parquet/csv files found under {path}")
            files.extend(found)
        elif path.suffix.lower() in {".parquet", ".csv"}:
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
    return unique


def _read_input_file(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, low_memory=False)
    )


def read_input_paths(
    paths: str | Path | Sequence[str | Path],
) -> pd.DataFrame:
    """Read explicit parquet/CSV files or all such files below directories."""

    frames = [_read_input_file(path) for path in _input_files(paths)]
    return pd.concat(frames, ignore_index=True, sort=False)


def _canonicalize_identifiers(
    frame: pd.DataFrame,
    *,
    context: str,
    require_game_pk: bool,
) -> pd.DataFrame:
    out = frame.copy()
    required = ["pitcher", "game_date"]
    if require_game_pk:
        required.append("game_pk")
    for canonical, aliases in _IDENTIFIER_ALIASES.items():
        values = _coalesce_alias_values(out, aliases)
        if values is None:
            if canonical in required:
                raise ValueError(
                    f"{context} is missing {canonical}; accepted aliases: {aliases}"
                )
            continue
        out[canonical] = values
    parsed = pd.to_datetime(out["game_date"], errors="coerce", utc=True)
    out["game_date"] = parsed.dt.tz_convert(None).dt.normalize()
    if out[required].isna().any(axis=1).any():
        raise ValueError(f"{context} contains invalid/null identifiers: {required}")
    return out


def _row_ids(frame: pd.DataFrame) -> pd.Series:
    try:
        from lib.temporal_splits import make_row_ids

        return make_row_ids(frame)
    except ImportError:
        # Keep the fallback byte-for-byte compatible with temporal_splits so
        # this data module can also be used before the manifest layer exists.
        def identifier(value: object) -> str:
            if isinstance(value, (np.integer, int)):
                return str(int(value))
            if isinstance(value, (np.floating, float)):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError("row_id identifiers must be finite")
                return (
                    str(int(numeric))
                    if numeric.is_integer()
                    else format(numeric, ".17g")
                )
            return str(value)

        def one(row: tuple[object, object, object]) -> str:
            pitcher, game_pk, game_date = row
            payload = [
                identifier(pitcher),
                identifier(game_pk),
                pd.Timestamp(game_date).date().isoformat(),
            ]
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        return pd.Series(
            (
                one(row)
                for row in frame[
                    ["pitcher", "game_pk", "game_date"]
                ].itertuples(index=False, name=None)
            ),
            index=frame.index,
            dtype="string",
            name="row_id",
        )


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[mask]
    y = values[mask]
    x -= x.mean()
    denominator = np.sum(x * x)
    return (
        float(np.sum(x * (y - y.mean())) / denominator)
        if denominator
        else np.nan
    )


def _pitch_aggregates(
    pitches: pd.DataFrame,
    mapping: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Build known-family shares and fold-safe FF/SI/FA physical aggregates."""

    df = _canonicalize_identifiers(
        pitches, context="Statcast input", require_game_pk=True
    )
    pitch_type_column = _find_column(df, ("pitch_type", "pitchType"))
    if pitch_type_column is None:
        raise ValueError("Statcast input is missing pitch_type.")
    normalized = _normalise_pitch_type_mapping(mapping)
    keys = ["pitcher", "game_pk", "game_date"]
    pitch_type = (
        df[pitch_type_column].astype("string").str.strip().str.upper()
    )
    df["_pitch_type"] = pitch_type
    df["_fastball"] = pitch_type.isin(normalized["fastball"]).astype(int)
    df["_breaking"] = pitch_type.isin(normalized["breaking"]).astype(int)
    df["_offspeed"] = pitch_type.isin(normalized["offspeed"]).astype(int)
    counts = df.groupby(keys, as_index=False, sort=False).agg(
        fastball_count=("_fastball", "sum"),
        breaking_count=("_breaking", "sum"),
        offspeed_count=("_offspeed", "sum"),
        _all_pitch_count=("_pitch_type", "size"),
    )
    counts["known_pitch_count"] = counts[
        ["fastball_count", "breaking_count", "offspeed_count"]
    ].sum(axis=1)
    counts["other_pitch_count"] = (
        counts["_all_pitch_count"] - counts["known_pitch_count"]
    ).clip(lower=0)
    known = counts["known_pitch_count"].replace(0, np.nan)
    for category in ("fastball", "breaking", "offspeed"):
        counts[f"{category}_share"] = counts[f"{category}_count"] / known
    counts["known_pitch_share"] = (
        counts["known_pitch_count"]
        / counts["_all_pitch_count"].replace(0, np.nan)
    )
    counts = counts.drop(columns="_all_pitch_count")

    measurement_aliases = {
        "velocity": ("release_speed", "velocity"),
        "spin": ("release_spin_rate", "spin_rate"),
        "pfx_x": ("pfx_x",),
        "pfx_z": ("pfx_z",),
    }
    result = counts
    for code in ("FF", "SI", "FA"):
        prefix = code.lower()
        subset = df.loc[df["_pitch_type"].eq(code)].copy()
        if subset.empty:
            result[f"{prefix}_count"] = 0
            for suffix in measurement_aliases:
                result[f"{prefix}_{suffix}"] = np.nan
            continue
        type_counts = (
            subset.groupby(keys, as_index=False, sort=False)
            .size()
            .rename(columns={"size": f"{prefix}_count"})
        )
        result = result.merge(type_counts, on=keys, how="left")
        result[f"{prefix}_count"] = (
            result[f"{prefix}_count"].fillna(0).astype(int)
        )
        for suffix, aliases in measurement_aliases.items():
            source = _find_column(subset, aliases)
            output = f"{prefix}_{suffix}"
            if source is None:
                result[output] = np.nan
                continue
            subset["_measurement"] = pd.to_numeric(
                subset[source], errors="coerce"
            )
            aggregate = (
                subset.groupby(keys, as_index=False, sort=False)["_measurement"]
                .mean()
                .rename(columns={"_measurement": output})
            )
            result = result.merge(aggregate, on=keys, how="left")
    for prefix in ("ff", "si", "fa"):
        count = f"{prefix}_count"
        if count not in result:
            result[count] = 0
        result[count] = result[count].fillna(0).astype(int)
        for suffix in ("velocity", "spin", "pfx_x", "pfx_z"):
            column = f"{prefix}_{suffix}"
            if column not in result:
                result[column] = np.nan
    return result


def aggregate_statcast_outings(
    pitches: pd.DataFrame,
    pitch_type_mapping: Mapping[str, Any] | None = None,
    *,
    starter_inference: str = "warn",
) -> pd.DataFrame:
    """Aggregate all official regular-season starts without a 50-pitch filter."""

    df = _canonicalize_identifiers(
        pitches, context="Statcast input", require_game_pk=True
    )
    game_type = _find_column(df, ("game_type", "gameType"))
    if game_type is None:
        raise ValueError(
            "Statcast input must include game_type so regular season can be enforced."
        )
    df = df.loc[
        df[game_type].astype("string").str.strip().str.upper().eq("R")
    ].copy()
    if df.empty:
        return pd.DataFrame()

    outings = build_outings(df, starter_inference=starter_inference)
    if "is_starting_pitcher" not in outings.columns:
        raise ValueError(
            "Cannot establish official starts: supply inning/top-bottom data or "
            "an is_starting_pitcher/GS flag."
        )
    outings = outings.loc[
        pd.to_numeric(outings["is_starting_pitcher"], errors="coerce").eq(1)
    ].copy()
    if outings.empty:
        return pd.DataFrame()

    keys = ["pitcher", "game_pk", "game_date"]
    per_type = _pitch_aggregates(df, pitch_type_mapping)
    outings = outings.merge(per_type, on=keys, how="left", validate="one_to_one")

    if "spin_axis" in df.columns:
        angle = np.deg2rad(pd.to_numeric(df["spin_axis"], errors="coerce"))
        circular = (
            df.assign(
                _spin_axis_sin=np.sin(angle),
                _spin_axis_cos=np.cos(angle),
            )
            .groupby(keys, as_index=False, sort=False)
            .agg(
                spin_axis_sin=("_spin_axis_sin", "mean"),
                spin_axis_cos=("_spin_axis_cos", "mean"),
            )
        )
        outings = outings.merge(
            circular, on=keys, how="left", validate="one_to_one"
        )
    else:
        outings["spin_axis_sin"] = np.nan
        outings["spin_axis_cos"] = np.nan

    outings["game_type"] = "R"
    outings["is_starting_pitcher"] = 1
    outings["history_eligible"] = True
    outings["stuff_plus_available"] = False
    outings["target_eligible"] = False
    outings["game_date"] = pd.to_datetime(outings["game_date"]).dt.normalize()
    outings["year"] = outings["game_date"].dt.year.astype(int)
    outings["row_id"] = _row_ids(outings)
    if outings["row_id"].duplicated().any():
        raise ValueError("Duplicate pitcher/game/date rows after Statcast aggregation.")
    return outings.sort_values(
        ["pitcher", "game_date", "game_pk"], kind="stable"
    ).reset_index(drop=True)


def _one_pitcher_outings(
    path: Path,
    pitch_type_mapping: Mapping[str, Any] | None = None,
    *,
    starter_inference: str = "warn",
) -> pd.DataFrame:
    """Compatibility wrapper; files may contain one pitcher or all MLB pitchers."""

    pitches = pd.read_parquet(path)
    if pitches.empty:
        return pd.DataFrame()
    return aggregate_statcast_outings(
        pitches,
        pitch_type_mapping,
        starter_inference=starter_inference,
    )


def _canonicalize_stuff(stuff: pd.DataFrame) -> pd.DataFrame:
    out = _canonicalize_identifiers(
        stuff, context="Stuff+ input", require_game_pk=False
    )
    values = _coalesce_alias_values(out, _STUFF_PLUS_ALIASES)
    if values is None:
        raise ValueError(
            f"Stuff+ input is missing a target column; accepted aliases: "
            f"{_STUFF_PLUS_ALIASES}"
        )
    out["stuff_plus"] = pd.to_numeric(values, errors="coerce")
    starter_values = _coalesce_alias_values(
        out,
        (
            "is_starting_pitcher",
            "is_starter",
            "starter",
            "starting_pitcher",
            "GS",
        ),
    )
    if starter_values is not None:
        out["_stuff_starter"] = _nullable_boolean_values(starter_values)
    gs_values = _coalesce_alias_values(out, ("GS",))
    if gs_values is not None:
        out["GS"] = pd.to_numeric(gs_values, errors="coerce")
    return out


def _join_identifier(series: pd.Series) -> pd.Series:
    def one(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    return series.map(one).astype("string")


def _boolean_values(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    values = numeric.gt(0)
    values.loc[text.isin({"true", "t", "yes", "y", "starter", "starting"})] = True
    values.loc[text.isin({"false", "f", "no", "n", "reliever", "relief"})] = False
    return values.fillna(False)


def _nullable_boolean_values(series: pd.Series) -> pd.Series:
    """Parse starter flags while retaining unknown values as ``pd.NA``."""

    text = series.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    values = pd.Series(pd.NA, index=series.index, dtype="boolean")
    values.loc[numeric.notna()] = numeric.loc[numeric.notna()].ne(0)
    values.loc[text.isin({"true", "t", "yes", "y", "starter", "starting"})] = True
    values.loc[text.isin({"false", "f", "no", "n", "reliever", "relief"})] = False
    return values


def _with_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.copy()
    keyed["_pitcher_join"] = _join_identifier(keyed["pitcher"])
    keyed["_game_join"] = (
        _join_identifier(keyed["game_pk"])
        if "game_pk" in keyed.columns
        else pd.Series("", index=keyed.index, dtype="string")
    )
    keyed["_date_join"] = keyed["game_date"]
    return keyed


def _coalesce_match_payloads(
    candidates: pd.DataFrame,
    payload_columns: Sequence[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Coalesce duplicate source records only when every value agrees."""

    rows: list[dict[str, object]] = []
    for left_match_id, group in candidates.groupby(
        "_left_match_id", sort=False
    ):
        row: dict[str, object] = {"_left_match_id": left_match_id}
        for column in payload_columns:
            observed = group[column].dropna().drop_duplicates()
            if len(observed) > 1:
                raise ValueError(
                    f"Ambiguous {context} matches have conflicting {column} "
                    f"values for left row {left_match_id}."
                )
            row[column] = observed.iloc[0] if len(observed) else pd.NA
        rows.append(row)
    return pd.DataFrame(rows, columns=["_left_match_id", *payload_columns])


def _resolve_identity_matches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    payload_columns: Sequence[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Resolve source rows by game id first, then by an unambiguous date.

    This is intentionally row-wise.  A concatenated source may contain game
    identifiers for only some rows; choosing the join strategy once for the
    whole DataFrame silently loses every date-only row.
    """

    key_columns = ["_pitcher_join", "_game_join", "_date_join"]
    left_ids = (
        _with_join_keys(left)[key_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    left_ids["_left_match_id"] = np.arange(len(left_ids), dtype=np.int64)
    source = _with_join_keys(right)[[*key_columns, *payload_columns]].copy()
    source = source.drop_duplicates().reset_index(drop=True)
    source["_right_match_id"] = np.arange(len(source), dtype=np.int64)

    # MLB game_pk is stronger than a date, but a claimed exact match with a
    # different date is corrupt rather than something to average away.
    exact_source = source.loc[source["_game_join"].ne("")].copy()
    exact = left_ids.merge(
        exact_source,
        on=["_pitcher_join", "_game_join"],
        how="inner",
        suffixes=("_left", "_right"),
    )
    if not exact.empty:
        date_conflict = exact["_date_join_left"].ne(exact["_date_join_right"])
        if date_conflict.any():
            sample = exact.loc[
                date_conflict,
                [
                    "_pitcher_join",
                    "_game_join",
                    "_date_join_left",
                    "_date_join_right",
                ],
            ].head(5)
            raise ValueError(
                f"{context} has game-id matches with conflicting dates: "
                f"{sample.to_dict(orient='records')}"
            )
        exact_matches = _coalesce_match_payloads(
            exact,
            payload_columns,
            context=f"{context} game-id",
        )
    else:
        exact_matches = pd.DataFrame(
            columns=["_left_match_id", *payload_columns]
        )

    matched_ids = set(exact_matches["_left_match_id"].tolist())
    fallback_left = left_ids.loc[
        ~left_ids["_left_match_id"].isin(matched_ids)
    ].copy()
    fallback = fallback_left.merge(
        source,
        on=["_pitcher_join", "_date_join"],
        how="inner",
        suffixes=("_left", "_right"),
    )
    if not fallback.empty:
        # Date-only matching cannot select one half of a pitcher's doubleheader.
        date_counts = left_ids.groupby(
            ["_pitcher_join", "_date_join"], sort=False
        ).size()
        fallback_keys = pd.MultiIndex.from_frame(
            fallback[["_pitcher_join", "_date_join"]].drop_duplicates()
        )
        ambiguous_dates = date_counts.reindex(fallback_keys, fill_value=0).gt(1)
        if ambiguous_dates.any():
            examples = [
                {"pitcher": pitcher, "game_date": str(game_date.date())}
                for pitcher, game_date in fallback_keys[ambiguous_dates.to_numpy()][
                    :5
                ]
            ]
            raise ValueError(
                f"Ambiguous {context} date fallback for doubleheader rows: "
                f"{examples}"
            )

        fallback = fallback.drop_duplicates(
            ["_left_match_id", "_right_match_id"]
        )
        right_game_counts = (
            fallback.loc[fallback["_game_join_right"].ne("")]
            .groupby("_left_match_id", sort=False)["_game_join_right"]
            .nunique()
        )
        ambiguous_source_games = right_game_counts.gt(1)
        if ambiguous_source_games.any():
            ambiguous_ids = set(
                right_game_counts.index[ambiguous_source_games].tolist()
            )
            sample = fallback.loc[
                fallback["_left_match_id"].isin(ambiguous_ids),
                ["_pitcher_join", "_date_join"],
            ].drop_duplicates().head(5)
            raise ValueError(
                f"Ambiguous {context} date matches reference multiple game ids: "
                f"{sample.to_dict(orient='records')}"
            )
        fallback_matches = _coalesce_match_payloads(
            fallback,
            payload_columns,
            context=f"{context} date",
        )
    else:
        fallback_matches = pd.DataFrame(
            columns=["_left_match_id", *payload_columns]
        )

    matches = pd.concat(
        [exact_matches, fallback_matches],
        ignore_index=True,
        sort=False,
    )
    result = left_ids.merge(
        matches, on="_left_match_id", how="left", validate="one_to_one"
    )
    return result.drop(columns="_left_match_id")


def _attach_starter_status(
    pitches: pd.DataFrame,
    starter_status: pd.DataFrame,
) -> pd.DataFrame:
    """Attach an optional, explicitly row-level official starter source."""

    raw = _canonicalize_identifiers(
        pitches, context="Statcast input", require_game_pk=True
    )
    status = _canonicalize_identifiers(
        starter_status, context="Starter-status input", require_game_pk=False
    )
    starter_values = _coalesce_alias_values(
        status,
        (
            "is_starting_pitcher",
            "is_starter",
            "starter",
            "starting_pitcher",
            "GS",
        ),
    )
    if starter_values is None:
        raise ValueError(
            "Starter-status input needs an is_starting_pitcher/GS column."
        )
    status["_official_start"] = _nullable_boolean_values(starter_values)
    official = _resolve_identity_matches(
        raw,
        status,
        ["_official_start"],
        context="starter-status",
    )
    raw = _with_join_keys(raw).merge(
        official,
        on=["_pitcher_join", "_game_join", "_date_join"],
        how="left",
        validate="many_to_one",
    )
    raw["is_starting_pitcher"] = (
        raw["_official_start"].fillna(False).astype(int)
    )
    return raw.drop(
        columns=[
            "_official_start",
            "_pitcher_join",
            "_game_join",
            "_date_join",
        ],
        errors="ignore",
    )


def merge_stuff_plus(outings: pd.DataFrame, stuff: pd.DataFrame) -> pd.DataFrame:
    """Left-join Stuff+, retaining short starts and starts with missing targets."""

    left = _canonicalize_identifiers(
        outings, context="Outing input", require_game_pk=True
    )
    right = _canonicalize_stuff(stuff)
    left = left.drop(
        columns=["stuff_plus", "sp_stuff", "target_y"],
        errors="ignore",
    )
    payload = ["stuff_plus"]
    for column in ("GS", "_stuff_starter"):
        if column in right.columns:
            payload.append(column)
    targets = _resolve_identity_matches(
        left,
        right,
        payload,
        context="Stuff+",
    ).rename(
        columns={
            "GS": "_stuff_GS",
            "_stuff_starter": "_stuff_is_start",
        }
    )
    merged = _with_join_keys(left).merge(
        targets,
        on=["_pitcher_join", "_game_join", "_date_join"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=["_pitcher_join", "_game_join", "_date_join"],
        errors="ignore",
    )
    if "_stuff_GS" in merged:
        if "GS" in merged:
            merged["GS"] = pd.to_numeric(
                merged["GS"], errors="coerce"
            ).fillna(merged["_stuff_GS"])
        else:
            merged["GS"] = merged["_stuff_GS"]
    # A GS-bearing target source is authoritative for outings that Statcast
    # could only infer.  This preserves the historical FG GS=1 inner-join
    # semantics and removes first-inning bulk relievers from model history.
    history = merged.get(
        "history_eligible", pd.Series(True, index=merged.index)
    ).fillna(False).astype(bool)
    if "_stuff_is_start" in merged:
        inferred = merged.get(
            "starter_status_source",
            pd.Series("", index=merged.index, dtype="string"),
        ).astype("string").eq("statcast_first_inning_by_team")
        authoritative = right["_stuff_starter"].notna().all()
        if authoritative:
            confirmed = merged["_stuff_is_start"].fillna(False).astype(bool)
            history.loc[inferred] &= confirmed.loc[inferred]
            if "is_starting_pitcher" in merged:
                merged.loc[inferred, "is_starting_pitcher"] = (
                    confirmed.loc[inferred].astype(int)
                )
    merged["history_eligible"] = history
    merged = merged.drop(
        columns=["_stuff_GS", "_stuff_is_start"], errors="ignore"
    )

    merged["sp_stuff"] = merged["stuff_plus"]
    merged["target_y"] = merged["stuff_plus"]
    merged["stuff_plus_available"] = merged["stuff_plus"].notna()
    pitch_count = pd.to_numeric(merged["pitch_count"], errors="coerce")
    merged["target_eligible"] = (
        history & pitch_count.ge(50) & merged["stuff_plus_available"]
    )
    if not merged["stuff_plus_available"].any():
        raise ValueError(
            "Stuff+ merge matched zero outings; verify pitcher/game/date identifiers."
        )
    return merged.sort_values(
        ["pitcher", "game_date", "game_pk"], kind="stable"
    ).reset_index(drop=True)


def _strict_prior_ends(dates: pd.Series) -> np.ndarray:
    values = dates.to_numpy(dtype="datetime64[ns]")
    return np.searchsorted(values, values, side="left")


def _add_history(
    data: pd.DataFrame,
    column: str,
    add_slope: bool = True,
) -> None:
    """Add legacy MA5/slope5 columns using only earlier calendar dates."""

    data[f"{column}_ma5"] = np.nan
    if add_slope:
        data[f"{column}_slope5"] = np.nan
    for _, rows in data.groupby("pitcher", sort=False):
        group = rows.loc[
            rows["history_eligible"].fillna(False).astype(bool)
        ].sort_values(["game_date", "game_pk"], kind="stable")
        if group.empty:
            continue
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
        prior_ends = _strict_prior_ends(group["game_date"])
        means: list[float] = []
        slopes: list[float] = []
        for prior_end in prior_ends:
            recent = values[:prior_end][-5:]
            finite = recent[np.isfinite(recent)]
            means.append(float(finite.mean()) if len(finite) else np.nan)
            if add_slope:
                slopes.append(_slope(recent))
        data.loc[group.index, f"{column}_ma5"] = means
        if add_slope:
            data.loc[group.index, f"{column}_slope5"] = slopes


def add_history_features(
    outings: pd.DataFrame,
    ewma_span: int = 4,
) -> pd.DataFrame:
    """Add strictly-prior, pitcher-continuous history across season boundaries."""

    if int(ewma_span) < 1:
        raise ValueError("ewma_span must be at least 1.")
    data = _canonicalize_identifiers(
        outings, context="Outing history input", require_game_pk=True
    )
    data = data.sort_values(
        ["pitcher", "game_date", "game_pk"], kind="stable"
    ).reset_index(drop=True)
    data["history_eligible"] = data.get(
        "history_eligible", pd.Series(True, index=data.index)
    ).fillna(False).astype(bool)
    if "row_id" not in data:
        data["row_id"] = _row_ids(data)

    numeric_defaults = (
        "rest_days",
        "rest_days_capped",
        "rest_days_log",
        "prev_start_pitch_count",
        "workload_density_3starts",
        "pitcher_prior_outing_count",
        "prior_stuff_plus",
        "stuff_plus_mean_last5",
        "stuff_plus_slope_last5",
        "stuff_plus_lag1",
        "stuff_plus_prior_mean5",
        "stuff_plus_prior_std5",
        "stuff_plus_prior_slope5",
        "pitch_count_lag1",
        "pitch_count_prior_mean5",
        f"stuff_plus_ewma{int(ewma_span)}",
        f"ewma{int(ewma_span)}",
    )
    for column in numeric_defaults:
        data[column] = np.nan
    data["season_start"] = False
    data["long_gap"] = False

    prior_mean_sources = (
        "fastball_share",
        "breaking_share",
        "release_extension",
        "release_pos_x",
        "release_pos_z",
        "arm_angle",
    )
    for source in prior_mean_sources:
        if source in data:
            data[f"{source}_prior_mean5"] = np.nan

    for _, rows in data.groupby("pitcher", sort=False):
        group = rows.loc[
            rows["history_eligible"].fillna(False).astype(bool)
        ].sort_values(["game_date", "game_pk"], kind="stable")
        if group.empty:
            continue
        index = group.index
        dates = group["game_date"]
        date_array = dates.to_numpy(dtype="datetime64[ns]")
        prior_ends = _strict_prior_ends(dates)
        pitch_counts = pd.to_numeric(
            group["pitch_count"], errors="coerce"
        ).to_numpy(float)
        stuff_source = (
            group["stuff_plus"]
            if "stuff_plus" in group
            else pd.Series(np.nan, index=group.index)
        )
        stuff = pd.to_numeric(stuff_source, errors="coerce").to_numpy(float)

        for position, (row_index, prior_end) in enumerate(
            zip(index, prior_ends, strict=True)
        ):
            current_date = date_array[position]
            data.at[row_index, "pitcher_prior_outing_count"] = float(prior_end)
            if prior_end:
                previous_date = date_array[prior_end - 1]
                rest = float(
                    (current_date - previous_date) / np.timedelta64(1, "D")
                )
                data.at[row_index, "rest_days"] = rest
                capped = min(max(rest, 0.0), 30.0)
                data.at[row_index, "rest_days_capped"] = capped
                data.at[row_index, "rest_days_log"] = np.log1p(capped)
                data.at[row_index, "long_gap"] = bool(rest > 21)
                data.at[row_index, "season_start"] = bool(
                    pd.Timestamp(current_date).year
                    != pd.Timestamp(previous_date).year
                )
                data.at[row_index, "prev_start_pitch_count"] = pitch_counts[
                    prior_end - 1
                ]
                data.at[row_index, "pitch_count_lag1"] = pitch_counts[
                    prior_end - 1
                ]
            else:
                data.at[row_index, "season_start"] = True

            prior_stuff = stuff[:prior_end]
            recent_stuff = prior_stuff[-5:]
            if len(prior_stuff):
                data.at[row_index, "prior_stuff_plus"] = prior_stuff[-1]
                data.at[row_index, "stuff_plus_lag1"] = prior_stuff[-1]
                ewma = (
                    pd.Series(prior_stuff)
                    .ewm(
                        span=int(ewma_span),
                        adjust=False,
                        min_periods=1,
                        ignore_na=True,
                    )
                    .mean()
                )
                data.at[row_index, f"stuff_plus_ewma{int(ewma_span)}"] = (
                    ewma.iloc[-1]
                )
                data.at[row_index, f"ewma{int(ewma_span)}"] = ewma.iloc[-1]
            finite_stuff = recent_stuff[np.isfinite(recent_stuff)]
            if len(finite_stuff):
                mean5 = float(finite_stuff.mean())
                data.at[row_index, "stuff_plus_mean_last5"] = mean5
                data.at[row_index, "stuff_plus_prior_mean5"] = mean5
            if len(finite_stuff) >= 2:
                data.at[row_index, "stuff_plus_prior_std5"] = float(
                    np.std(finite_stuff, ddof=0)
                )
            slope = _slope(recent_stuff)
            data.at[row_index, "stuff_plus_slope_last5"] = slope
            data.at[row_index, "stuff_plus_prior_slope5"] = slope

            recent_pitch_counts = pitch_counts[:prior_end][-5:]
            finite_counts = recent_pitch_counts[
                np.isfinite(recent_pitch_counts)
            ]
            if len(finite_counts):
                data.at[row_index, "pitch_count_prior_mean5"] = float(
                    finite_counts.mean()
                )
            if prior_end >= 3:
                selected_counts = pitch_counts[prior_end - 3 : prior_end]
                selected_dates = date_array[prior_end - 3 : prior_end]
                window_dates = np.append(selected_dates, current_date)
                gaps = np.diff(window_dates) / np.timedelta64(1, "D")
                elapsed = float(
                    (current_date - selected_dates[0])
                    / np.timedelta64(1, "D")
                )
                if (
                    elapsed > 0
                    and np.isfinite(selected_counts).all()
                    and np.max(gaps) <= 21
                ):
                    data.at[row_index, "workload_density_3starts"] = float(
                        selected_counts.sum() / elapsed
                    )

            for source in prior_mean_sources:
                if source not in group:
                    continue
                values = pd.to_numeric(
                    group[source], errors="coerce"
                ).to_numpy(float)
                recent = values[:prior_end][-5:]
                finite = recent[np.isfinite(recent)]
                if len(finite):
                    data.at[row_index, f"{source}_prior_mean5"] = float(
                        finite.mean()
                    )

    for column in PHYSICAL_RAW + ["breaking_share", "offspeed_share"]:
        if column not in data:
            data[column] = np.nan
        _add_history(data, column)
    for column in ("spin_axis_sin", "spin_axis_cos"):
        if column not in data:
            data[column] = np.nan
        _add_history(data, column, add_slope=False)

    data["year"] = data["game_date"].dt.year.astype(int)
    if "stuff_plus" not in data:
        data["stuff_plus"] = np.nan
    data["sp_stuff"] = pd.to_numeric(data["stuff_plus"], errors="coerce")
    data["target_y"] = data["sp_stuff"]
    data["stuff_plus_available"] = data["sp_stuff"].notna()
    data["target_eligible"] = (
        data["history_eligible"]
        & pd.to_numeric(data["pitch_count"], errors="coerce").ge(50)
        & data["stuff_plus_available"]
    )
    ewma4 = data.get(
        "ewma4",
        data.get("stuff_plus_ewma4", pd.Series(np.nan, index=data.index)),
    )
    data["prior_minus_ewma4"] = data["prior_stuff_plus"] - ewma4
    data["mean5_minus_ewma4"] = data["stuff_plus_mean_last5"] - ewma4
    return data


def _attach_names(
    outings: pd.DataFrame,
    official_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Use a names-only official file without treating it as an outing log."""

    pitcher_column = _find_column(
        official_stats,
        ("player_id", "pitcher", "pitcher_id", "mlbam_id"),
    )
    name_column = _find_column(
        official_stats, ("name", "player_name", "pitcher_name")
    )
    if pitcher_column is None or name_column is None:
        return outings
    names = (
        official_stats[[pitcher_column, name_column]]
        .dropna(subset=[pitcher_column])
        .drop_duplicates(pitcher_column, keep="last")
        .rename(
            columns={
                pitcher_column: "_official_pitcher",
                name_column: "pitcher_name",
            }
        )
    )
    left = outings.copy()
    left["_pitcher_join"] = _join_identifier(left["pitcher"])
    names["_pitcher_join"] = _join_identifier(names["_official_pitcher"])
    names = names[["_pitcher_join", "pitcher_name"]]
    return left.merge(
        names, on="_pitcher_join", how="left", validate="many_to_one"
    ).drop(columns="_pitcher_join")


def build_stuff_mlb_dataset(
    statcast_paths: str | Path | Sequence[str | Path],
    stuff_paths: str | Path | Sequence[str | Path],
    official_stats_path: (
        str | Path | Sequence[str | Path] | None
    ) = None,
    starter_status_paths: (
        str | Path | Sequence[str | Path] | None
    ) = None,
    cutter_category: str = "fastball",
    starter_inference: str = "warn",
) -> pd.DataFrame:
    """Build the complete official-start history before target-row filtering."""

    inference_mode = str(starter_inference).strip().lower()
    if inference_mode not in {"warn", "allow", "strict"}:
        raise ValueError("starter_inference must be one of: warn, allow, strict")
    starter_status = (
        read_input_paths(starter_status_paths)
        if starter_status_paths is not None
        else None
    )
    if starter_status_paths is not None:
        if starter_status is None or starter_status.empty:
            raise ValueError("Starter-status input is empty.")
    mapping = default_pitch_type_mapping(cutter_category)
    outing_frames: list[pd.DataFrame] = []
    inference_warning_emitted = False
    for path in _input_files(statcast_paths):
        pitches = _read_input_file(path)
        if pitches.empty:
            continue
        if starter_status is not None:
            pitches = _attach_starter_status(pitches, starter_status)
        effective_inference_mode = inference_mode
        if inference_mode == "warn" and inference_warning_emitted:
            explicit = _coalesce_alias_values(
                pitches, ("is_starting_pitcher", "GS")
            )
            has_explicit_status = (
                explicit is not None and explicit.notna().any()
            )
            if not has_explicit_status:
                # ``build_outings`` intentionally warns on every direct call.
                # A dataset build fans one logical operation out over many
                # files, so suppress only repeats inside this invocation.
                effective_inference_mode = "allow"
        frame = aggregate_statcast_outings(
            pitches,
            mapping,
            starter_inference=effective_inference_mode,
        )
        if (
            inference_mode == "warn"
            and effective_inference_mode == "warn"
            and not frame.empty
            and frame.get(
                "starter_status_source",
                pd.Series("", index=frame.index, dtype="string"),
            )
            .astype("string")
            .eq("statcast_first_inning_by_team")
            .any()
        ):
            inference_warning_emitted = True
        if not frame.empty:
            outing_frames.append(frame)
    if not outing_frames:
        raise ValueError("No official regular-season Statcast starts were found.")
    outings = pd.concat(outing_frames, ignore_index=True, sort=False)
    duplicate = outings.duplicated(["pitcher", "game_pk", "game_date"], keep=False)
    if duplicate.any():
        examples = outings.loc[
            duplicate, ["pitcher", "game_pk", "game_date"]
        ].drop_duplicates().head(5)
        raise ValueError(
            "Statcast inputs contain overlapping outing rows; examples: "
            f"{examples.to_dict(orient='records')}"
        )
    if outings.empty:
        raise ValueError("No official regular-season Statcast starts were found.")
    stuff = read_input_paths(stuff_paths)
    merged = merge_stuff_plus(outings, stuff)
    if official_stats_path is not None:
        # This input is frequently only ``player_id,name``.  It must never be
        # required to contain an outing-level starter flag.
        merged = _attach_names(merged, read_input_paths(official_stats_path))
    return add_history_features(merged, ewma_span=4)


def make_dataset(
    statcast_dir: Path | Sequence[Path],
    stuff_path: Path | Sequence[Path],
) -> tuple[pd.DataFrame, list[int]]:
    """Compatibility entry point with the original 2021-2025 qualification."""

    data = build_stuff_mlb_dataset(statcast_dir, stuff_path)
    target = data.loc[data["target_eligible"]].copy()
    counts = target.groupby(
        ["pitcher", target["game_date"].dt.year]
    ).size().unstack(fill_value=0)
    for year in range(2021, 2026):
        if year not in counts:
            counts[year] = 0
    qualified = sorted(
        int(value)
        for value in counts.index[
            (counts[list(range(2021, 2026))] >= 20).all(axis=1)
        ]
    )
    return (
        data.loc[data["pitcher"].isin(qualified)]
        .sort_values(["pitcher", "game_date", "game_pk"], kind="stable")
        .reset_index(drop=True),
        qualified,
    )
