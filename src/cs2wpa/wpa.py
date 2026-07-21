"""
Phase 2 — Win Probability Added (WPA).

The payoff metric. The win-prob model (Phase 1B) tells us P(CT wins) at any game
state. WPA attributes the *change* in that probability across each kill to the
player who made it:

    for a kill at tick k, look at the state just before and just after it,
    score both with the model, and credit the swing to the killer's side.

A CT kill removes a T → P(CT win) rises → the CT killer earns that rise.
A T kill removes a CT → P(CT win) falls → the T killer earns the drop.
Summed over a tournament, WPA ranks players by how much win probability they
actually generated — catching clutch timing and impact that K/D can't see.

Public entry point:
    compute_wpa(demo_path, model) -> polars.DataFrame  (one row per kill)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .model import FEATURES
from .snapshots import ROUND_SECONDS, TICKRATE, TICK_PROPS, aggregate_sides, round_windows

# Ticks either side of a kill to sample the "before" and "after" state. ~0.25s
# at 64 tick — long enough for the death to register in is_alive/health.
BUF = 16


def _round_meta(windows: list[dict]) -> list[dict]:
    """Attach pre-round score to each window (same ordering as the snapshot builder)."""
    meta = []
    ct_score = t_score = 0
    for idx, w in enumerate(windows):
        meta.append({**w, "round_idx": idx, "ct_score_pre": ct_score, "t_score_pre": t_score})
        ct_score += w["ct_win"]
        t_score += not w["ct_win"]
    return meta


def _features_at(wide: pl.DataFrame, eval_rows: pl.DataFrame) -> pl.DataFrame:
    """Join side-aggregates onto eval ticks and derive the 16 model features."""
    df = eval_rows.join(wide, on="tick", how="inner")
    return df.with_columns(
        seconds_elapsed=((pl.col("tick") - pl.col("start_tick")) / TICKRATE),
        alive_diff=(pl.col("alive_ct").cast(pl.Int32) - pl.col("alive_t").cast(pl.Int32)),
        hp_diff=(pl.col("hp_ct").cast(pl.Int32) - pl.col("hp_t").cast(pl.Int32)),
        equip_diff=(pl.col("equip_ct") - pl.col("equip_t")),
        bomb_planted=(
            (pl.col("plant_tick").is_not_null()) & (pl.col("tick") >= pl.col("plant_tick").fill_null(0))
        ).cast(pl.Int8),
    ).with_columns(
        seconds_remaining=(ROUND_SECONDS - pl.col("seconds_elapsed")).clip(lower_bound=0),
    )


def extract_kill_states(demo_path: str | Path) -> pl.DataFrame:
    """
    Model-INDEPENDENT half of WPA: for every kill, the game-state feature vectors
    just before and just after it (long form, 2 rows per kill), plus killer side
    and kill metadata. Cache this at parse time so WPA can be scored later without
    the (huge) demo file. score_wpa() turns it into WPA once a model exists.
    """
    demo = Path(demo_path)
    parser, windows = round_windows(demo)
    if not windows:
        return pl.DataFrame()
    meta = _round_meta(windows)

    deaths = parser.parse_event("player_death")
    deaths = deaths if isinstance(deaths, pl.DataFrame) else pl.from_pandas(deaths)
    deaths = deaths.drop_nulls(["tick", "attacker_steamid", "user_steamid"])

    kills = []
    for row in deaths.iter_rows(named=True):
        k = int(row["tick"])
        rnd = next((m for m in meta if m["start_tick"] <= k <= m["end_tick"]), None)
        if rnd is None:
            continue
        kills.append({
            "kill_id": len(kills), "tick": k, "kill_tick": k, "round_idx": rnd["round_idx"],
            "start_tick": rnd["start_tick"], "plant_tick": rnd["plant_tick"],
            "ct_score_pre": rnd["ct_score_pre"], "t_score_pre": rnd["t_score_pre"],
            "attacker_steamid": str(row["attacker_steamid"]), "attacker_name": row["attacker_name"],
            "victim_name": row["user_name"], "weapon": row["weapon"],
        })
    if not kills:
        return pl.DataFrame()
    kdf = pl.DataFrame(kills)

    eval_rows = pl.concat([
        kdf.with_columns(role=pl.lit("before"), tick=pl.col("tick") - BUF),
        kdf.with_columns(role=pl.lit("after"), tick=pl.col("tick") + BUF),
    ])
    eval_ticks = sorted(set(eval_rows["tick"].to_list()))

    ticks = parser.parse_ticks(TICK_PROPS, ticks=eval_ticks)
    ticks = ticks if isinstance(ticks, pl.DataFrame) else pl.from_pandas(ticks)
    # killer side at the kill: look up team_num of the attacker at the 'after' tick
    team_at = {(int(t), str(s)): int(tn)
               for t, s, tn in zip(ticks["tick"], ticks["steamid"], ticks["team_num"])
               if tn is not None}

    feats = _features_at(aggregate_sides(ticks), eval_rows)
    killer_side = kdf.with_columns(
        killer_side=pl.struct(["tick", "attacker_steamid"]).map_elements(
            lambda r: team_at.get((r["tick"] + BUF, r["attacker_steamid"])),
            return_dtype=pl.Int64,
        )
    ).select(["kill_id", "killer_side"])

    keep = ["kill_id", "role", "round_idx", "kill_tick", "attacker_name",
            "attacker_steamid", "victim_name", "weapon", *FEATURES]
    return feats.select(keep).join(killer_side, on="kill_id", how="inner").with_columns(
        demo=pl.lit(demo.stem)
    )


def score_wpa(kill_states: pl.DataFrame, model) -> pl.DataFrame:
    """Model-dependent half: score before/after win-prob and attribute the swing
    to the killer's side. One row per kill."""
    if kill_states.height == 0:
        return pl.DataFrame()
    X = kill_states.select(FEATURES).to_numpy()
    scored = kill_states.with_columns(
        wp_ct=pl.Series(model.predict_proba(X)[:, 1]),
        # kill_id restarts at 0 per demo, so make a globally-unique key before
        # pivoting across many maps.
        uid=pl.col("demo") + "#" + pl.col("kill_id").cast(pl.Utf8),
    )

    piv = scored.select(["uid", "role", "wp_ct"]).pivot(
        values="wp_ct", index="uid", on="role")
    info = (scored.filter(pl.col("role") == "after")
            .select(["uid", "round_idx", "kill_tick", "attacker_name", "attacker_steamid",
                     "victim_name", "weapon", "killer_side", "demo"]))
    out = info.join(piv, on="uid", how="inner").drop_nulls(["before", "after", "killer_side"])

    # CT kill (side 3): swing_ct is the gain. T kill (side 2): killer earns the drop.
    swing_ct = pl.col("after") - pl.col("before")
    return out.with_columns(
        wp_before=pl.col("before"), wp_after=pl.col("after"),
        wpa=pl.when(pl.col("killer_side") == 3).then(swing_ct).otherwise(-swing_ct),
    ).drop(["before", "after"])


def compute_wpa(demo_path: str | Path, model) -> pl.DataFrame:
    """Convenience: extract kill states from a demo and score them in one go."""
    return score_wpa(extract_kill_states(demo_path), model)


def leaderboard(kills: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-kill WPA into a per-player leaderboard."""
    if kills.height == 0:
        return pl.DataFrame()
    return (
        kills.group_by(["attacker_steamid", "attacker_name"])
        .agg(
            total_wpa=pl.col("wpa").sum(),
            kills=pl.len(),
            wpa_per_kill=pl.col("wpa").mean(),
        )
        .sort("total_wpa", descending=True)
    )
