"""
Phase 4 — team scouting report ("Moneyball").

For each team (rosters recovered from cached kill data — see teams.py), reports
tendencies an analyst actually uses: side win rates, buy-type distribution and
success, and opening-duel impact. All from data already on disk — no re-parse.

(Bomb-site A/B preference needs a richer re-parse and is intentionally left for
a later pass; noted in the output.)

Writes results/scouting.md.

Usage:
    python scripts/scout.py
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cs2wpa.teams import ECO_MAX, FULL_MIN, build_team_rounds  # noqa: E402

SNAPS = Path("data/processed/snapshots.parquet")
KILLS = Path("data/processed/wpa_kills.parquet")
OUT = Path("results/scouting.md")
MIN_ROUNDS = 40


def buy_type(equip: int) -> str:
    if equip < ECO_MAX:
        return "eco"
    if equip >= FULL_MIN:
        return "full"
    return "force"


def main() -> None:
    side_rows, player_team = build_team_rounds()
    snaps = pl.read_parquet(SNAPS)
    kills = pl.read_parquet(KILLS)

    # round start state per (demo, round): outcome + each side's buy value
    start = (snaps.sort("seconds_elapsed")
             .group_by(["demo", "round_idx"])
             .agg(ct_win=pl.col("ct_win").first(),
                  equip_ct=pl.col("equip_ct").first(),
                  equip_t=pl.col("equip_t").first()))

    # first kill side per (demo, round) → who won the opening duel
    fk = (kills.sort("kill_tick").group_by(["demo", "round_idx"])
          .agg(fk_side=pl.col("killer_side").first()))

    tr = (side_rows.join(start, on=["demo", "round_idx"], how="inner")
          .join(fk, on=["demo", "round_idx"], how="left")
          .with_columns(
              won=pl.when(pl.col("side") == 3).then(pl.col("ct_win"))
                    .otherwise(1 - pl.col("ct_win")),
              team_equip=pl.when(pl.col("side") == 3).then(pl.col("equip_ct"))
                           .otherwise(pl.col("equip_t")),
              got_fb=(pl.col("fk_side") == pl.col("side")).cast(pl.Int8),
          )
          .with_columns(
              buy=pl.col("team_equip").map_elements(buy_type, return_dtype=pl.Utf8))
          )

    # per-team aggregates
    def wr(df):  # win rate helper
        return (100 * df["won"].mean()) if df.height else float("nan")

    teams = sorted(tr["team"].unique().to_list())
    rows = []
    for t in teams:
        d = tr.filter(pl.col("team") == t)
        if d.height < MIN_ROUNDS:
            continue
        ct = d.filter(pl.col("side") == 3)
        tt = d.filter(pl.col("side") == 2)
        fb = d.filter(pl.col("got_fb") == 1)
        nofb = d.filter(pl.col("got_fb") == 0)
        rows.append({
            "team": t, "rounds": d.height,
            "win%": wr(d), "CT%": wr(ct), "T%": wr(tt),
            "fb_rate": 100 * d["got_fb"].mean(),
            "win|fb": wr(fb), "win|nofb": wr(nofb),
        })
    board = pl.DataFrame(rows).sort("win%", descending=True)

    # buy-type success across all teams (context)
    buy = (tr.group_by("buy").agg(rounds=pl.len(), win=(100 * pl.col("won").mean()))
           .sort("rounds", descending=True))

    # --- write report -------------------------------------------------------
    def fmt(v):
        return f"{v:.0f}" if v == v else "—"

    lines = [
        "# Team Scouting — IEM Cologne Major 2026",
        "",
        f"Rosters recovered from kill data; {tr.height} team-rounds across "
        f"{tr['demo'].n_unique()} maps. Teams with ≥{MIN_ROUNDS} rounds.",
        "",
        "## Team tendencies",
        "| Team | Rounds | Win% | CT% | T% | 1st-blood% | Win if 1st blood | Win if not |",
        "|------|-------:|-----:|----:|---:|-----------:|-----------------:|-----------:|",
    ]
    for r in board.iter_rows(named=True):
        lines.append(f"| {r['team']} | {r['rounds']} | {fmt(r['win%'])} | "
                     f"{fmt(r['CT%'])} | {fmt(r['T%'])} | {fmt(r['fb_rate'])} | "
                     f"{fmt(r['win|fb'])} | {fmt(r['win|nofb'])} |")
    lines += [
        "",
        "## The opening duel decides rounds",
        "How often a team wins the round given they got first blood vs not — the "
        "single biggest swing an analyst can target.",
        "",
        "## Buy-type success (all teams)",
        "| Buy | Rounds | Win% |",
        "|-----|-------:|-----:|",
    ]
    for r in buy.iter_rows(named=True):
        lines.append(f"| {r['buy']} | {r['rounds']} | {r['win']:.0f} |")
    lines += ["", "_Bomb-site (A/B) preference needs a richer re-parse — planned._"]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")

    print(f"{tr.height} team-rounds, {len(teams)} teams.")
    with pl.Config(tbl_rows=20, tbl_hide_dataframe_shape=True):
        print(board.with_columns(pl.col(pl.Float64).round(0)))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
