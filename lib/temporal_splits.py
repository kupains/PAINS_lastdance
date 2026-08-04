"""Deterministic row identities and expanding-window temporal fold metadata.

This module operates on an already-built outing table.  Roster qualification is
deliberately a separate, upstream concern: callers may pass a roster determined
from 2021-2025 availability without that roster decision being treated as model
selection.  Fold-local targets, thresholds, and pitch mappings still use only
the rows supplied as training data for each fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_TRAIN_START = 2020
DEFAULT_VALIDATION_YEARS = (2022, 2023, 2024)
DEFAULT_TEST_YEAR = 2025
DEFAULT_PRIMARY_FASTBALL_CANDIDATES = ("FF", "SI", "FA")


@dataclass(frozen=True)
class FoldSpec:
    """One expanding-window temporal fold."""

    fold_year: int
    train_start: int = DEFAULT_TRAIN_START
    evaluation_split: str = "validation"

    def __post_init__(self) -> None:
        if self.fold_year <= self.train_start:
            raise ValueError("fold_year must be later than train_start")
        if self.evaluation_split not in {"validation", "test"}:
            raise ValueError("evaluation_split must be 'validation' or 'test'")

    @property
    def train_end(self) -> int:
        return self.fold_year - 1

    @property
    def train_years(self) -> tuple[int, ...]:
        return tuple(range(self.train_start, self.train_end + 1))

    def split_for_year(self, year: int) -> str | None:
        if self.train_start <= year <= self.train_end:
            return "train"
        if year == self.fold_year:
            return self.evaluation_split
        return None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["train_end"] = self.train_end
        result["train_years"] = list(self.train_years)
        return result


def rolling_fold_specs(
    train_start: int = DEFAULT_TRAIN_START,
    validation_years: Sequence[int] = DEFAULT_VALIDATION_YEARS,
    test_year: int = DEFAULT_TEST_YEAR,
) -> tuple[FoldSpec, ...]:
    """Return validation folds followed by one final test fold."""

    years = tuple(int(year) for year in validation_years)
    if not years:
        raise ValueError("validation_years cannot be empty")
    if tuple(sorted(set(years))) != years:
        raise ValueError("validation_years must be unique and increasing")
    if int(test_year) <= years[-1]:
        raise ValueError("test_year must be later than every validation year")
    folds = [
        FoldSpec(year, train_start=int(train_start), evaluation_split="validation")
        for year in years
    ]
    folds.append(
        FoldSpec(int(test_year), train_start=int(train_start), evaluation_split="test")
    )
    return tuple(folds)


DEFAULT_FOLDS = rolling_fold_specs()


def _canonical_identifier(value: Any, field_name: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{field_name} cannot be missing when row_id is created")
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} must be finite when row_id is created")
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".17g")
    return str(value)


def _canonical_game_date(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("game_date cannot be missing when row_id is created")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.date().isoformat()


def stable_row_id(pitcher: Any, game_pk: Any, game_date: Any) -> str:
    """Create a stable SHA-256 outing id from pitcher, game id, and date."""

    payload = [
        _canonical_identifier(pitcher, "pitcher"),
        _canonical_identifier(game_pk, "game_pk"),
        _canonical_game_date(game_date),
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def make_row_ids(
    frame: pd.DataFrame,
    pitcher_col: str = "pitcher",
    game_pk_col: str = "game_pk",
    game_date_col: str = "game_date",
) -> pd.Series:
    """Vectorized wrapper around :func:`stable_row_id`."""

    required = [pitcher_col, game_pk_col, game_date_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot create row_id; missing columns: {missing}")
    values = (
        stable_row_id(pitcher, game_pk, game_date)
        for pitcher, game_pk, game_date in frame[required].itertuples(
            index=False, name=None
        )
    )
    return pd.Series(values, index=frame.index, dtype="string", name="row_id")


def add_row_id(
    frame: pd.DataFrame,
    *,
    row_id_col: str = "row_id",
    pitcher_col: str = "pitcher",
    game_pk_col: str = "game_pk",
    game_date_col: str = "game_date",
    verify_existing: bool = True,
    require_unique: bool = True,
) -> pd.DataFrame:
    """Return a copy with a verified deterministic row id."""

    result = frame.copy()
    calculated = make_row_ids(result, pitcher_col, game_pk_col, game_date_col)
    if row_id_col in result.columns and verify_existing:
        existing = result[row_id_col].astype("string")
        mismatch = existing.isna() | existing.ne(calculated)
        if mismatch.any():
            examples = result.index[mismatch].tolist()[:5]
            raise ValueError(
                f"Existing {row_id_col} values do not match canonical ids "
                f"at rows {examples}"
            )
    result[row_id_col] = calculated
    if require_unique and result[row_id_col].duplicated().any():
        duplicates = result.loc[
            result[row_id_col].duplicated(keep=False), row_id_col
        ].unique()[:5]
        raise ValueError(f"Duplicate outing row ids: {duplicates.tolist()}")
    return result


def _canonical_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        if math.isinf(numeric):
            return "Infinity" if numeric > 0 else "-Infinity"
        if numeric == 0:
            return 0.0
        return float(format(numeric, ".17g"))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_canonical_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def dataframe_fingerprint(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> str:
    """Hash a dataframe canonically, independent of row and column order."""

    selected = sorted(frame.columns if columns is None else tuple(columns))
    if len(set(selected)) != len(selected):
        raise ValueError("Fingerprint columns must be unique")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot fingerprint dataframe; missing columns: {missing}")
    digest = sha256()
    digest.update(
        json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")
    records: list[bytes] = []
    for row in frame[selected].itertuples(index=False, name=None):
        canonical = [_canonical_value(value) for value in row]
        records.append(
            json.dumps(
                canonical,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
    for encoded in sorted(records):
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


fingerprint_dataframe = dataframe_fingerprint


def _fit_year_metadata(
    frame: pd.DataFrame,
    *,
    year_col: str = "year",
    game_date_col: str = "game_date",
) -> dict[str, Any]:
    if year_col in frame.columns:
        years = pd.to_numeric(frame[year_col], errors="coerce").dropna().astype(int)
    elif game_date_col in frame.columns:
        years = (
            pd.to_datetime(frame[game_date_col], errors="coerce")
            .dt.year.dropna()
            .astype(int)
        )
    else:
        years = pd.Series(dtype=int)
    unique = tuple(sorted(int(year) for year in years.unique()))
    return {
        "fit_years": unique,
        "fit_year_min": min(unique) if unique else None,
        "fit_year_max": max(unique) if unique else None,
        "fit_n_rows": int(len(frame)),
    }


def compute_pitcher_tertiles(
    training: pd.DataFrame,
    *,
    pitcher_col: str = "pitcher",
    target_col: str = "target_y",
    eligibility_col: str | None = "target_eligible",
    q_low: float = 1.0 / 3.0,
    q_high: float = 2.0 / 3.0,
    year_col: str = "year",
    game_date_col: str = "game_date",
) -> pd.DataFrame:
    """Fit per-pitcher target thresholds from the supplied training rows."""

    if not 0 < q_low < q_high < 1:
        raise ValueError("Expected 0 < q_low < q_high < 1")
    required = [pitcher_col, target_col]
    missing = [column for column in required if column not in training.columns]
    if missing:
        raise ValueError(f"Cannot compute tertiles; missing columns: {missing}")
    mask = pd.to_numeric(training[target_col], errors="coerce").notna()
    if eligibility_col is not None and eligibility_col in training.columns:
        mask &= training[eligibility_col].fillna(False).astype(bool)
    fit = training.loc[mask, [pitcher_col, target_col]].copy()
    fit[target_col] = pd.to_numeric(fit[target_col], errors="coerce")
    if fit.empty:
        result = pd.DataFrame(
            columns=[pitcher_col, "q33", "q67", "n_train_targets"]
        )
    else:
        grouped = fit.groupby(pitcher_col, sort=True, dropna=False)[target_col]
        result = grouped.agg(
            q33=lambda values: values.quantile(q_low),
            q67=lambda values: values.quantile(q_high),
            n_train_targets="count",
        ).reset_index()
    metadata = _fit_year_metadata(
        training.loc[mask],
        year_col=year_col,
        game_date_col=game_date_col,
    )
    for key, value in metadata.items():
        result[key] = [value] * len(result)
    result.attrs.update(metadata)
    result.attrs.update({"q_low": float(q_low), "q_high": float(q_high)})
    return result


def assign_true_classes(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    pitcher_col: str = "pitcher",
    target_col: str = "target_y",
    class_col: str = "true_class",
) -> pd.DataFrame:
    """Attach thresholds and the legacy Low/Middle/High class definition.

    This matches ``stuff_mlb_temporal_final._classes`` exactly:
    Low is ``value <= q33``, Middle is ``q33 < value <= q67``, and High is
    ``value > q67``.  In particular, an observation tied at q67 is Middle.
    """

    required_frame = [pitcher_col, target_col]
    missing_frame = [
        column for column in required_frame if column not in frame.columns
    ]
    if missing_frame:
        raise ValueError(f"Cannot assign target classes; missing columns: {missing_frame}")
    required_thresholds = [pitcher_col, "q33", "q67"]
    missing_thresholds = [
        column for column in required_thresholds if column not in thresholds.columns
    ]
    if missing_thresholds:
        raise ValueError(f"Threshold table is missing columns: {missing_thresholds}")
    if thresholds[pitcher_col].duplicated().any():
        raise ValueError("Threshold table must contain one row per pitcher")

    result = frame.drop(columns=["q33", "q67", class_col], errors="ignore").copy()
    result = result.merge(
        thresholds[required_thresholds],
        how="left",
        on=pitcher_col,
        sort=False,
        validate="many_to_one",
    )
    target = pd.to_numeric(result[target_col], errors="coerce")
    valid = target.notna() & result["q33"].notna() & result["q67"].notna()
    classes = pd.Series(pd.NA, index=result.index, dtype="Int64")
    classes.loc[valid] = 1
    classes.loc[valid & target.gt(result["q67"])] = 2
    classes.loc[valid & target.le(result["q33"])] = 0
    result[class_col] = classes
    return result


def _boolean_column(
    frame: pd.DataFrame,
    column: str,
    *,
    fallback: pd.Series | bool,
) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(False).astype(bool)
    if isinstance(fallback, pd.Series):
        return fallback.fillna(False).astype(bool)
    return pd.Series(bool(fallback), index=frame.index, dtype=bool)


def build_fold_manifest(
    outings: pd.DataFrame,
    *,
    folds: Sequence[FoldSpec] = DEFAULT_FOLDS,
    pitcher_col: str = "pitcher",
    game_pk_col: str = "game_pk",
    game_date_col: str = "game_date",
    year_col: str = "year",
    target_col: str = "target_y",
    history_eligible_col: str = "history_eligible",
    target_eligible_col: str = "target_eligible",
) -> pd.DataFrame:
    """Create a deterministic manifest with fold-local training thresholds.

    ``outings`` may already be filtered to a research roster whose availability
    was checked through 2025.  The manifest neither recomputes nor rejects that
    qualification decision.  It only prevents 2025 outcomes from entering the
    2022-2024 validation folds and labels 2025 as the final test split.
    """

    required = [pitcher_col, game_pk_col, game_date_col, target_col]
    missing = [column for column in required if column not in outings.columns]
    if missing:
        raise ValueError(f"Cannot build fold manifest; missing columns: {missing}")
    data = add_row_id(
        outings,
        pitcher_col=pitcher_col,
        game_pk_col=game_pk_col,
        game_date_col=game_date_col,
    )
    data[game_date_col] = pd.to_datetime(data[game_date_col], errors="coerce")
    if data[game_date_col].isna().any():
        raise ValueError("game_date contains invalid values")
    derived_year = data[game_date_col].dt.year.astype(int)
    if year_col in data.columns:
        supplied_year = pd.to_numeric(data[year_col], errors="coerce")
        mismatch = supplied_year.notna() & supplied_year.astype("Int64").ne(derived_year)
        if mismatch.any():
            raise ValueError("year is inconsistent with game_date")
    data[year_col] = derived_year
    target_values = pd.to_numeric(data[target_col], errors="coerce")
    data[history_eligible_col] = _boolean_column(
        data, history_eligible_col, fallback=True
    )
    data[target_eligible_col] = _boolean_column(
        data, target_eligible_col, fallback=target_values.notna()
    )

    manifests: list[pd.DataFrame] = []
    for fold in folds:
        in_fold = data[year_col].between(fold.train_start, fold.fold_year)
        fold_data = data.loc[in_fold].copy()
        fold_data["fold_year"] = int(fold.fold_year)
        fold_data["split"] = fold_data[year_col].map(fold.split_for_year)
        train = fold_data.loc[
            fold_data["split"].eq("train") & fold_data[target_eligible_col]
        ].copy()
        thresholds = compute_pitcher_tertiles(
            train,
            pitcher_col=pitcher_col,
            target_col=target_col,
            eligibility_col=target_eligible_col,
            year_col=year_col,
            game_date_col=game_date_col,
        )
        classified = assign_true_classes(
            fold_data,
            thresholds,
            pitcher_col=pitcher_col,
            target_col=target_col,
        )
        classified.loc[
            ~classified[target_eligible_col], "true_class"
        ] = pd.NA
        manifests.append(classified)

    if not manifests:
        raise ValueError("At least one fold is required")
    manifest = pd.concat(manifests, ignore_index=True)
    split_order = pd.Categorical(
        manifest["split"],
        categories=["train", "validation", "test"],
        ordered=True,
    )
    manifest = (
        manifest.assign(_split_order=split_order)
        .sort_values(
            [
                "fold_year",
                "_split_order",
                game_date_col,
                pitcher_col,
                game_pk_col,
                "row_id",
            ],
            kind="mergesort",
        )
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )
    columns = [
        "row_id",
        pitcher_col,
        game_pk_col,
        game_date_col,
        year_col,
        "fold_year",
        "split",
        target_eligible_col,
        history_eligible_col,
        "q33",
        "q67",
        "true_class",
    ]
    extra = [column for column in manifest.columns if column not in columns]
    return manifest[columns + extra]


def fold_manifest_fingerprint(manifest: pd.DataFrame) -> str:
    """Fingerprint the analysis-defining columns of a fold manifest."""

    columns = [
        "row_id",
        "fold_year",
        "split",
        "target_eligible",
        "history_eligible",
        "q33",
        "q67",
        "true_class",
    ]
    return dataframe_fingerprint(manifest, columns=columns)


def fit_primary_fastball_mapping(
    training: pd.DataFrame,
    *,
    pitcher_col: str = "pitcher",
    candidates: Sequence[str] = DEFAULT_PRIMARY_FASTBALL_CANDIDATES,
    fold_year: int | None = None,
    year_col: str = "year",
    game_date_col: str = "game_date",
) -> pd.DataFrame:
    """Choose one primary fastball per pitcher from training-only usage."""

    if pitcher_col not in training.columns:
        raise ValueError(
            f"Cannot fit primary fastball mapping; missing column: {pitcher_col}"
        )
    normalized_candidates = tuple(str(candidate).upper() for candidate in candidates)
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ValueError("Primary fastball candidates must be unique")
    count_columns = [
        f"{candidate.lower()}_count" for candidate in normalized_candidates
    ]
    missing = [column for column in count_columns if column not in training.columns]
    if missing:
        raise ValueError(
            f"Cannot fit primary fastball mapping; missing count columns: {missing}"
        )

    counts = training[[pitcher_col, *count_columns]].copy()
    for column in count_columns:
        counts[column] = (
            pd.to_numeric(counts[column], errors="coerce").fillna(0.0).clip(lower=0)
        )
    totals = counts.groupby(
        pitcher_col, sort=True, dropna=False
    )[count_columns].sum()
    rows: list[dict[str, Any]] = []
    for pitcher, values in totals.iterrows():
        numeric = values.to_numpy(dtype=float)
        best_index = int(np.argmax(numeric))
        best_count = float(numeric[best_index])
        primary_type: Any = (
            normalized_candidates[best_index] if best_count > 0 else pd.NA
        )
        row: dict[str, Any] = {
            pitcher_col: pitcher,
            "primary_fb_type": primary_type,
            "primary_fb_training_pitch_count": best_count,
        }
        row.update(
            {
                f"{candidate.lower()}_training_count": float(numeric[index])
                for index, candidate in enumerate(normalized_candidates)
            }
        )
        if fold_year is not None:
            row["fold_year"] = int(fold_year)
        rows.append(row)
    mapping = pd.DataFrame(rows)
    metadata = _fit_year_metadata(
        training, year_col=year_col, game_date_col=game_date_col
    )
    for key, value in metadata.items():
        mapping[key] = [value] * len(mapping)
    mapping.attrs.update(metadata)
    mapping.attrs["candidates"] = normalized_candidates
    mapping.attrs["fold_year"] = fold_year
    return mapping


def apply_primary_fastball_mapping(
    outings: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    pitcher_col: str = "pitcher",
    candidates: Sequence[str] = DEFAULT_PRIMARY_FASTBALL_CANDIDATES,
) -> pd.DataFrame:
    """Apply a fixed training mapping to per-pitch-type outing aggregates."""

    keys = [pitcher_col]
    if "fold_year" in outings.columns and "fold_year" in mapping.columns:
        keys.insert(0, "fold_year")
    required_mapping = [*keys, "primary_fb_type"]
    missing = [
        column for column in required_mapping if column not in mapping.columns
    ]
    if missing:
        raise ValueError(f"Primary fastball mapping is missing columns: {missing}")
    if mapping.duplicated(keys).any():
        raise ValueError(f"Primary fastball mapping must have one row per {keys}")

    result = outings.drop(
        columns=[
            "primary_fb_type",
            "primary_fb_pitch_count",
            "primary_fb_velocity",
            "primary_fb_spin",
            "primary_fb_pfx_x",
            "primary_fb_pfx_z",
            "primary_fb_available",
        ],
        errors="ignore",
    ).copy()
    order_column = "_primary_fb_input_order"
    while order_column in result.columns:
        order_column = f"_{order_column}"
    result[order_column] = np.arange(len(result))
    result = result.merge(
        mapping[required_mapping],
        how="left",
        on=keys,
        sort=False,
        validate="many_to_one",
    )
    source_suffixes = {
        "pitch_count": "count",
        "velocity": "velocity",
        "spin": "spin",
        "pfx_x": "pfx_x",
        "pfx_z": "pfx_z",
    }
    normalized_candidates = tuple(str(candidate).upper() for candidate in candidates)
    for output_suffix, source_suffix in source_suffixes.items():
        output_column = f"primary_fb_{output_suffix}"
        values = pd.Series(np.nan, index=result.index, dtype=float)
        for candidate in normalized_candidates:
            source_column = f"{candidate.lower()}_{source_suffix}"
            if source_column not in result.columns:
                continue
            mask = result["primary_fb_type"].eq(candidate)
            # Normalize nullable Float64 extension arrays before assigning
            # into the NumPy float output. Pandas 3 otherwise rejects this
            # representable assignment with a LossySetitemError.
            source = pd.Series(
                pd.to_numeric(
                    result[source_column], errors="coerce"
                ).to_numpy(dtype=float, na_value=np.nan),
                index=result.index,
                dtype=float,
            )
            values.loc[mask] = source.loc[mask]
        result[output_column] = values
    physical_columns = [
        "primary_fb_velocity",
        "primary_fb_spin",
        "primary_fb_pfx_x",
        "primary_fb_pfx_z",
    ]
    physical_available = result[physical_columns].notna().any(axis=1)
    result["primary_fb_available"] = (
        result["primary_fb_type"].notna()
        & result["primary_fb_pitch_count"].fillna(0).gt(0)
        & physical_available
    )
    result = result.sort_values(order_column, kind="mergesort").drop(
        columns=order_column
    )
    result.index = outings.index
    result.attrs["primary_fastball_mapping_fit_years"] = mapping.attrs.get(
        "fit_years",
        tuple(mapping["fit_years"].iloc[0])
        if "fit_years" in mapping and len(mapping)
        else (),
    )
    return result


choose_primary_fastball_mapping = fit_primary_fastball_mapping
