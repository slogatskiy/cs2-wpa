"""
Run the feature builder over every demo in data/raw/ and write one combined
snapshot table to data/processed/snapshots.parquet.

Usage:
    python scripts/build_dataset.py                # all *.dem in data/raw
    python scripts/build_dataset.py path/to.dem    # a single demo
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cs2wpa.snapshots import build_round_snapshots  # noqa: E402

OUT = Path("data/processed/snapshots.parquet")


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        demos = [Path(argv[1])]
    else:
        demos = sorted(Path("data/raw").glob("*.dem"))
    if not demos:
        sys.exit("No demos found in data/raw/ — drop a .dem there first.")

    frames = []
    for demo in demos:
        print(f"→ {demo.name} ...", end=" ", flush=True)
        try:
            snaps = build_round_snapshots(demo)
        except Exception as e:  # keep going if one demo is corrupt
            print(f"FAILED ({e})")
            continue
        if snaps.height == 0:
            print("no rounds")
            continue
        frames.append(snaps)
        print(f"{snaps.height} snapshots, {snaps['round_idx'].n_unique()} rounds")

    if not frames:
        sys.exit("No snapshots produced.")

    combined = pl.concat(frames, how="vertical")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUT)

    print(f"\nSaved {combined.height} snapshots from {len(frames)} demo(s) → {OUT}")
    ct_winrate = combined["ct_win"].mean()
    print(f"CT win rate in labels: {ct_winrate:.1%}")


if __name__ == "__main__":
    main(sys.argv)
