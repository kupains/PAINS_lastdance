"""Deterministic synthetic outing data for fast integration tests.

The demo data mirrors the canonical outing contract, but it is never a
substitute for the recorded MLB results in ``FINAL_MODEL_REPORT.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_stuff_outings(
    *,
    pitchers: int = 3,
    outings_per_season: int = 12,
    start_year: int = 2020,
    end_year: int = 2025,
    seed: int = 20260722,
) -> pd.DataFrame:
    """Return a small deterministic official-start table.

    It includes both the legacy all-pitch physical columns used to reconstruct
    the original compact-16 features and the FF/SI/FA aggregates used by the
    fold-safe primary-fastball/TCN path.
    """

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    game_pk = 900_000

    for pitcher_index in range(int(pitchers)):
        pitcher = f"demo_pitcher_{pitcher_index + 1:02d}"
        primary = ("FF", "SI", "FA")[pitcher_index % 3]
        pitcher_level = 96.0 + pitcher_index * 5.0
        latent_state = rng.normal(0.0, 2.0)

        for year in range(int(start_year), int(end_year) + 1):
            first_date = pd.Timestamp(year=year, month=4, day=1)
            for outing_index in range(int(outings_per_season)):
                game_pk += 1
                game_date = first_date + pd.Timedelta(
                    days=outing_index * 13 + pitcher_index
                )
                latent_state = 0.72 * latent_state + rng.normal(0.0, 2.7)
                pitch_count = int(np.clip(rng.normal(87, 17), 32, 112))
                if outing_index == 3 and year in {2021, 2023}:
                    pitch_count = 40

                velocity = 93.0 + pitcher_index * 0.8 + 0.05 * latent_state
                spin = 2_250.0 + pitcher_index * 55.0 + 5.0 * latent_state
                pfx_x = -0.45 + pitcher_index * 0.18 + rng.normal(0.0, 0.04)
                pfx_z = 1.35 + pitcher_index * 0.05 + rng.normal(0.0, 0.04)
                stuff_plus = pitcher_level + latent_state + rng.normal(0.0, 3.2)

                type_counts = {"FF": 8, "SI": 8, "FA": 5}
                type_counts[primary] = 45 + int(rng.integers(-4, 5))
                breaking_count = 24 + int(rng.integers(-3, 4))
                offspeed_count = max(
                    4, pitch_count - sum(type_counts.values()) - breaking_count
                )
                known_count = (
                    sum(type_counts.values()) + breaking_count + offspeed_count
                )
                spin_axis = 205.0 + pitcher_index * 8.0 + rng.normal(0.0, 3.0)

                row: dict[str, object] = {
                    "pitcher": pitcher,
                    "pitcher_name": pitcher.replace("_", " ").title(),
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "year": year,
                    "game_type": "R",
                    "is_starting_pitcher": 1,
                    "starter_status_source": "demo",
                    "pitch_count": pitch_count,
                    "stuff_plus": stuff_plus,
                    "sp_stuff": stuff_plus,
                    "release_speed": velocity - 1.2,
                    "release_spin_rate": spin - 70.0,
                    "release_extension": (
                        6.1 + pitcher_index * 0.08 + rng.normal(0.0, 0.08)
                    ),
                    "release_pos_x": (
                        -1.8 + pitcher_index * 0.25 + rng.normal(0.0, 0.05)
                    ),
                    "release_pos_z": (
                        5.9 + pitcher_index * 0.1 + rng.normal(0.0, 0.05)
                    ),
                    "arm_angle": (
                        44.0 + pitcher_index * 3.0 + rng.normal(0.0, 1.0)
                    ),
                    "pfx_x": pfx_x + 0.04,
                    "pfx_z": pfx_z - 0.03,
                    "spin_axis": spin_axis,
                    "spin_axis_sin": np.sin(np.deg2rad(spin_axis)),
                    "spin_axis_cos": np.cos(np.deg2rad(spin_axis)),
                    "fastball_share": sum(type_counts.values()) / known_count,
                    "breaking_share": breaking_count / known_count,
                    "offspeed_share": offspeed_count / known_count,
                    "known_pitch_share": min(
                        1.0, known_count / max(pitch_count, 1)
                    ),
                    "history_eligible": True,
                    "stuff_plus_available": True,
                    "target_eligible": pitch_count >= 50,
                }
                for pitch_type in ("FF", "SI", "FA"):
                    prefix = pitch_type.lower()
                    is_primary = pitch_type == primary
                    row.update(
                        {
                            f"{prefix}_count": type_counts[pitch_type],
                            f"{prefix}_velocity": velocity
                            - (0.0 if is_primary else 1.4),
                            f"{prefix}_spin": spin
                            - (0.0 if is_primary else 90.0),
                            f"{prefix}_pfx_x": pfx_x
                            + (0.0 if is_primary else 0.10),
                            f"{prefix}_pfx_z": pfx_z
                            - (0.0 if is_primary else 0.08),
                        }
                    )
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame["target_eligible"] &= frame["stuff_plus"].notna()
    return frame.sort_values(
        ["pitcher", "game_date", "game_pk"], kind="mergesort"
    ).reset_index(drop=True)
