"""
Phase 1A — the feature builder.

Turns one CS2 .dem into "round snapshots": for every round we sample the game
state at regular time steps and attach the eventual winner. Each row is one
moment in a round described by side-symmetric features, labelled with who won
that round. This table is the training data for the win-probability model
(Phase 1B) and the backbone of WPA (Phase 2).

Public entry point:
    build_round_snapshots(demo_path) -> polars.DataFrame

Team encoding in demoparser2:  team_num 2 = T, 3 = CT.
Round winner in round_end:     winner   2 = T, 3 = CT.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from demoparser2 import DemoParser

# CS2 GOTV demos tick at 64/s. Exposed as a constant so it's easy to change
# if we ever meet a differently-recorded demo.
TICKRATE = 64
SAMPLE_EVERY_SEC = 1.0                       # snapshot cadence within a round
SAMPLE_STEP = int(TICKRATE * SAMPLE_EVERY_SEC)

# Standard competitive round length (freeze time excluded). Used only to derive
# a "time elapsed / remaining" feature — not for logic.
ROUND_SECONDS = 115

TICK_PROPS = [
    "health", "team_num", "is_alive",
    "balance",              # money the player holds right now
    "current_equip_value",  # value of the gear the player is carrying
]


def _events(parser: DemoParser) -> dict[str, pl.DataFrame]:
    """Parse the events we need, each as a polars frame (empty frame if absent)."""
    wanted = ["round_freeze_end", "round_end", "bomb_planted"]
    present = [e for e in wanted if e in parser.list_game_events()]
    out: dict[str, pl.DataFrame] = {}
    for name, df in parser.parse_events(present):
        out[name] = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    for name in wanted:
        out.setdefault(name, pl.DataFrame({"tick": []}))
    return out


def _round_windows(ev: dict[str, pl.DataFrame]) -> list[dict]:
    """
    Pair each round_end with the freeze-end that started that round's live action.

    Returns one dict per round: {start_tick, end_tick, winner, plant_tick|None}.
    Score going into the round is computed by the caller so it can stay ordered.
    """
    freeze_ends = sorted(int(t) for t in ev["round_freeze_end"]["tick"].to_list())
    ends = ev["round_end"].sort("tick")
    plants = sorted(int(t) for t in ev["bomb_planted"]["tick"].to_list())

    windows: list[dict] = []
    for row in ends.iter_rows(named=True):
        end_tick = int(row["tick"])
        winner = int(row["winner"])
        # start = latest freeze-end strictly before this round_end
        starts_before = [t for t in freeze_ends if t < end_tick]
        if not starts_before:
            continue  # round_end with no preceding freeze (warmup artefact) — skip
        start_tick = starts_before[-1]
        # bomb plant that falls inside this round's window, if any
        plant = next((t for t in plants if start_tick <= t <= end_tick), None)
        windows.append(
            {"start_tick": start_tick, "end_tick": end_tick,
             "winner": winner, "plant_tick": plant}
        )
    return windows


def _sample_ticks(windows: list[dict]) -> list[int]:
    """All tick indices we want per-player state for (union across rounds)."""
    ticks: set[int] = set()
    for w in windows:
        ticks.update(range(w["start_tick"], w["end_tick"], SAMPLE_STEP))
        ticks.add(w["start_tick"])  # always capture the full-buy t=0 state
    return sorted(ticks)


def build_round_snapshots(demo_path: str | Path) -> pl.DataFrame:
    demo = Path(demo_path)
    parser = DemoParser(str(demo))

    ev = _events(parser)
    windows = _round_windows(ev)
    if not windows:
        return pl.DataFrame()

    sample_ticks = _sample_ticks(windows)
    ticks = parser.parse_ticks(TICK_PROPS, ticks=sample_ticks)
    ticks = ticks if isinstance(ticks, pl.DataFrame) else pl.from_pandas(ticks)

    # Aggregate per (tick, side): alive count, total HP, team money, team equip.
    ticks = ticks.filter(pl.col("team_num").is_in([2, 3]))
    agg = (
        ticks.group_by(["tick", "team_num"])
        .agg(
            alive=pl.col("is_alive").sum(),
            hp=pl.when(pl.col("is_alive")).then(pl.col("health")).otherwise(0).sum(),
            money=pl.col("balance").sum(),
            equip=pl.col("current_equip_value").sum(),
        )
    )
    # Pivot sides into columns: *_t (team_num 2) and *_ct (team_num 3).
    t = agg.filter(pl.col("team_num") == 2).drop("team_num")
    ct = agg.filter(pl.col("team_num") == 3).drop("team_num")
    wide = (
        t.rename({c: f"{c}_t" for c in ["alive", "hp", "money", "equip"]})
        .join(
            ct.rename({c: f"{c}_ct" for c in ["alive", "hp", "money", "equip"]}),
            on="tick", how="inner",
        )
    )

    # Build one snapshot table by stitching each round's sampled ticks onto its
    # window metadata (score, timing, plant, label).
    rows: list[pl.DataFrame] = []
    ct_score = t_score = 0
    for rnd_idx, w in enumerate(windows):
        in_round = wide.filter(
            (pl.col("tick") >= w["start_tick"]) & (pl.col("tick") <= w["end_tick"])
        ).sort("tick")
        if in_round.height == 0:
            ct_score += w["winner"] == 3
            t_score += w["winner"] == 2
            continue

        planted = (
            (pl.lit(w["plant_tick"] is not None))
            & (pl.col("tick") >= (w["plant_tick"] or 0))
        )
        in_round = in_round.with_columns(
            round_idx=pl.lit(rnd_idx),
            seconds_elapsed=((pl.col("tick") - w["start_tick"]) / TICKRATE),
            # NOTE: side score is approximate — it does not yet track team
            # identity across the halftime side swap. Refine in Phase 1B.
            ct_score_pre=pl.lit(ct_score),
            t_score_pre=pl.lit(t_score),
            bomb_planted=planted.cast(pl.Int8),
            ct_win=pl.lit(int(w["winner"] == 3)),
        )
        rows.append(in_round)
        ct_score += w["winner"] == 3
        t_score += w["winner"] == 2

    if not rows:
        return pl.DataFrame()

    snaps = pl.concat(rows)
    snaps = snaps.with_columns(
        seconds_remaining=(ROUND_SECONDS - pl.col("seconds_elapsed")).clip(lower_bound=0),
        alive_diff=(pl.col("alive_ct") - pl.col("alive_t")),
        hp_diff=(pl.col("hp_ct") - pl.col("hp_t")),
        equip_diff=(pl.col("equip_ct") - pl.col("equip_t")),
        demo=pl.lit(demo.stem),
    )
    return snaps
