"""
Phase 0 — prove the pipeline is alive.

Takes ONE CS2 .dem file and turns it into tables so we can literally *see*
a match as data. This is the de-risk step: if this runs on a real demo,
the rest of the project is just modeling on top.

Usage:
    python scripts/explore_demo.py path/to/match.dem

What it does:
    1. Reads the header (map, tickrate, etc.)
    2. Lists every game event present in the demo.
    3. Pulls the key events we'll build features from (kills, bomb, round ends).
    4. Samples per-tick player state (position, HP, money, team).
    5. Prints a human summary and saves everything to data/processed/<demo>/.
"""

import sys
from pathlib import Path

import polars as pl
from demoparser2 import DemoParser

# Per-tick player properties we care about for a win-probability model.
# (demoparser2 exposes many more via list_updated_fields; this is a starter set.)
TICK_PROPS = [
    "X", "Y", "Z",          # position on the map
    "health", "armor_value",
    "team_num",             # 2 = T, 3 = CT
    "balance",              # money the player is holding
    "is_alive",
    "current_equip_value",  # value of gear currently held
]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main(demo_path: str) -> None:
    demo = Path(demo_path)
    if not demo.exists():
        sys.exit(f"Demo not found: {demo}")

    out_dir = Path("data/processed") / demo.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = DemoParser(str(demo))

    # 1) Header ---------------------------------------------------------------
    section("HEADER")
    header = parser.parse_header()
    for k, v in header.items():
        print(f"  {k:<20} {v}")

    # 2) What events does this demo even contain? -----------------------------
    section("GAME EVENTS PRESENT")
    events = parser.list_game_events()
    print(f"  {len(events)} distinct events")
    print("  " + ", ".join(sorted(events)))

    # 3) Key events -----------------------------------------------------------
    section("KEY EVENTS")
    wanted = ["round_start", "round_end", "player_death", "bomb_planted", "bomb_defused"]
    present = [e for e in wanted if e in events]
    parsed = parser.parse_events(present)  # list of (event_name, DataFrame)

    for name, df in parsed:
        pdf = pl.from_pandas(df) if not isinstance(df, pl.DataFrame) else df
        print(f"\n  --- {name}: {pdf.height} rows, cols={pdf.columns}")
        print(pdf.head(3))
        pdf.write_parquet(out_dir / f"event_{name}.parquet")

    # 4) Per-tick player state (sampled) --------------------------------------
    section("PER-TICK PLAYER STATE (sample)")
    ticks = parser.parse_ticks(TICK_PROPS)
    tdf = pl.from_pandas(ticks) if not isinstance(ticks, pl.DataFrame) else ticks
    print(f"  {tdf.height} player-tick rows, cols={tdf.columns}")
    print(tdf.head(5))
    tdf.write_parquet(out_dir / "ticks.parquet")

    # 5) Summary --------------------------------------------------------------
    section("SUMMARY")
    n_ticks = tdf.select(pl.col("tick").n_unique()).item() if "tick" in tdf.columns else "?"
    print(f"  map:            {header.get('map_name', '?')}")
    print(f"  unique ticks:   {n_ticks}")
    print(f"  saved tables →  {out_dir}/")
    print("\n  Phase 0 done. We can now see a match as data.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/explore_demo.py path/to/match.dem")
    main(sys.argv[1])
