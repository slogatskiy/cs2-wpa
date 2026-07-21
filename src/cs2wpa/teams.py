"""
Phase 4 (applied / "Moneyball") — team identity + tactical round table.

We purged the demos, so team names and per-round sides aren't stored anywhere.
But they're recoverable from the cached kill data:

  * Within a map, two players are OPPONENTS if they got kills on opposite sides
    in the same round. Everyone never-opposed-to an anchor is their team → a
    clean 5-man roster per map.
  * The same roster recurs across maps; merge by overlap to get a global team,
    then label it with the clan token its filenames all share (e.g. "furia").

From that we build a per-(map, round, team) table with side, result, buy type
and opening-duel outcome — the raw material for a scouting report. No re-parse.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import polars as pl

KILLS_DIR = Path("data/processed/kills")

# team equipment value at round start → buy type (team totals, ~5 players)
ECO_MAX = 5000
FULL_MIN = 18000

# Teams that only ever face each other in the dataset can't be told apart from
# filenames alone (both clan tokens appear equally). One known player anchors the
# label. Extend as needed for new events.
NAME_HINTS = {"Spinx": "mouz", "torzsi": "mouz"}


def _load_kills_after() -> pl.DataFrame:
    caches = sorted(KILLS_DIR.glob("*.parquet"))
    if not caches:
        raise SystemExit("No kill caches — run build_dataset.py first.")
    return pl.concat([pl.read_parquet(p) for p in caches]).filter(pl.col("role") == "after")


def _demo_rosters(k: pl.DataFrame) -> tuple[frozenset, frozenset] | None:
    """Split one map's players into two rosters via same-side-per-round logic."""
    players = sorted(p for p in set(k["attacker_name"].to_list()) if p)
    if len(players) < 6:
        return None
    by_rs = k.group_by(["round_idx", "killer_side"]).agg(
        pl.col("attacker_name").unique().alias("ps"))
    enemies: set[frozenset] = set()
    for rnd in k["round_idx"].unique().to_list():
        d = {r["killer_side"]: set(r["ps"])
             for r in by_rs.filter(pl.col("round_idx") == rnd).iter_rows(named=True)}
        if 2 in d and 3 in d:
            for a in d[2]:
                for b in d[3]:
                    enemies.add(frozenset((a, b)))
    anchor = players[0]
    teamA = frozenset(p for p in players
                      if p == anchor or frozenset((p, anchor)) not in enemies)
    teamB = frozenset(players) - teamA
    # only trust clean 5v5 splits — anything else is a bad inference and would
    # cross-link two real teams during the merge.
    if len(teamA) != 5 or len(teamB) != 5:
        return None
    return teamA, teamB


def _team_tokens(stem: str) -> tuple[str, str]:
    """'furia-vs-falcons-m1-mirage' -> ('furia', 'falcons');
    '9z-vs-the-mongolz-m2-overpass' -> ('9z', 'the-mongolz')."""
    import re
    left, _, right = stem.partition("-vs-")
    right = re.sub(r"-m\d.*$", "", right)  # strip -m1-mirage / -m2-nuke-p1 …
    return left, right


def build_team_rounds() -> pl.DataFrame:
    """One row per (demo, round, team): side, won, buy type, opening duel."""
    kills = _load_kills_after()

    # 1) rosters per demo + which side each roster is on each round
    demo_rosters: dict[str, tuple[frozenset, frozenset]] = {}
    for demo in kills["demo"].unique().to_list():
        r = _demo_rosters(kills.filter(pl.col("demo") == demo))
        if r:
            demo_rosters[demo] = r

    # 2) global team identity: merge recurring rosters against a FIXED core
    # (match on ≥4/5 shared so a stand-in still merges, but the core never grows
    # into a snowball that swallows other teams).
    from collections import Counter
    globals_: list[dict] = []  # {core:frozenset, members:Counter, demos:list}
    for demo, (a, b) in demo_rosters.items():
        for roster in (a, b):
            hit = next((g for g in globals_ if len(g["core"] & roster) >= 4), None)
            if hit is None:
                hit = {"core": roster, "members": Counter(), "demos": []}
                globals_.append(hit)
            hit["members"].update(roster)
            hit["demos"].append(demo)

    # label = the filename token present in the MOST of a team's own maps (its own
    # token appears in all of them; opponents vary). Resolve ties/collisions by
    # assigning the most-confident teams first and skipping taken names.
    used: set[str] = set()
    for g in sorted(globals_, key=lambda x: -len(x["demos"])):
        hint = next((NAME_HINTS[p] for p in g["members"] if p in NAME_HINTS), None)
        if hint:
            g["name"] = hint
            used.add(hint)
            continue
        toks: list[str] = []
        for d in g["demos"]:
            toks.extend(_team_tokens(d))
        ranked = sorted(set(toks), key=lambda t: -toks.count(t))
        g["name"] = next((t for t in ranked if t not in used), ranked[0])
        used.add(g["name"])

    player_team = {}
    for g in globals_:
        for p, _ in g["members"].most_common(5):  # drop rare stand-ins
            player_team[p] = g["name"]

    # 3) per (demo, round): each team's side, from its players' kill side
    side_rows = (kills.with_columns(
        team=pl.col("attacker_name").replace_strict(player_team, default=None))
        .drop_nulls("team")
        .group_by(["demo", "round_idx", "team"])
        .agg(side=pl.col("killer_side").mode().first()))

    return side_rows, player_team


if __name__ == "__main__":
    rows, pt = build_team_rounds()
    print("players mapped:", len(pt))
    teams = sorted(set(pt.values()))
    print("teams:", teams)
    for t in teams:
        roster = sorted(p for p, n in pt.items() if n == t)
        print(f"  {t:12} {roster}")
