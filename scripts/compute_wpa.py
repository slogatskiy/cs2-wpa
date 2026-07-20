"""
Score WPA for every cached kill-state table and print the player leaderboard.

Reads data/processed/kills/*.parquet (produced by build_dataset.py), applies the
trained win-prob model, and ranks players by total Win Probability Added.

Usage:
    python scripts/compute_wpa.py                 # all cached kills
    python scripts/compute_wpa.py --min-kills 20  # filter noisy small samples
"""

import pickle
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cs2wpa.wpa import leaderboard, score_wpa  # noqa: E402

KILLS = Path("data/processed/kills")
MODEL = Path("models/winprob_lgbm.pkl")
OUT = Path("data/processed/wpa_kills.parquet")


def main(argv: list[str]) -> None:
    min_kills = 0
    if "--min-kills" in argv:
        min_kills = int(argv[argv.index("--min-kills") + 1])

    if not MODEL.exists():
        sys.exit("No model — run scripts/train_winprob.py first.")
    caches = sorted(KILLS.glob("*.parquet"))
    if not caches:
        sys.exit("No kill-states — run scripts/build_dataset.py on some demos first.")

    model = pickle.load(open(MODEL, "rb"))
    states = pl.concat([pl.read_parquet(p) for p in caches], how="vertical")
    kills = score_wpa(states, model)
    kills.write_parquet(OUT)

    board = leaderboard(kills)
    if min_kills:
        board = board.filter(pl.col("kills") >= min_kills)

    print(f"Scored {kills.height} kills across {len(caches)} map(s).\n")
    print("=== WPA LEADERBOARD (total win probability added via kills) ===")
    with pl.Config(tbl_rows=30):
        print(board.with_columns(
            pl.col("total_wpa").round(2), pl.col("wpa_per_kill").round(3)
        ).select(["attacker_name", "total_wpa", "kills", "wpa_per_kill"]))
    print(f"\nPer-kill detail saved → {OUT}")


if __name__ == "__main__":
    main(sys.argv)
