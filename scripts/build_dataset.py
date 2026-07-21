"""
Turn demos in data/raw/*.dem into round snapshots.

Incremental & resumable: each demo's snapshots are cached to
data/processed/snapshots/<demo>.parquet, so re-running skips demos already
done. At the end everything is merged into data/processed/snapshots.parquet.

Usage:
    python scripts/build_dataset.py                 # process all new demos
    python scripts/build_dataset.py path/to.dem     # a single demo
    python scripts/build_dataset.py --purge         # delete each .dem after it
                                                    # is successfully parsed
                                                    # (frees disk for big demos)

Typical big-batch flow: download a few 1GB demos → run with --purge → the raw
.dem files are removed once parsed, so disk never fills up. Snapshots are tiny.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cs2wpa.snapshots import build_round_snapshots  # noqa: E402
from cs2wpa.wpa import extract_kill_states  # noqa: E402

RAW = Path("data/raw")
CACHE = Path("data/processed/snapshots")
KILLS = Path("data/processed/kills")
COMBINED = Path("data/processed/snapshots.parquet")


def main(argv: list[str]) -> None:
    purge = "--purge" in argv
    paths = [a for a in argv[1:] if not a.startswith("--")]
    demos = [Path(paths[0])] if paths else sorted(RAW.glob("*.dem"))
    if not demos:
        sys.exit("No demos found in data/raw/ — drop a .dem (or .rar) there first.")

    CACHE.mkdir(parents=True, exist_ok=True)
    KILLS.mkdir(parents=True, exist_ok=True)

    for demo in demos:
        cache_file = CACHE / f"{demo.stem}.parquet"
        kills_file = KILLS / f"{demo.stem}.parquet"
        # Snapshots and kill-states are cached independently: a demo parsed before
        # kill-caching existed has snapshots but no kills, so only skip when BOTH
        # are present. Otherwise re-parse to fill whichever is missing.
        if cache_file.exists() and kills_file.exists():
            print(f"• {demo.name} — cached, skip")
            continue
        print(f"→ {demo.name} ...", end=" ", flush=True)
        try:
            snaps = build_round_snapshots(demo) if not cache_file.exists() else None
            kill_states = extract_kill_states(demo) if not kills_file.exists() else None
        except Exception as e:  # keep going if one demo is corrupt
            print(f"FAILED ({e})")
            continue
        if snaps is not None:
            if snaps.height == 0:
                print("no rounds (skipped)")
                continue
            snaps.write_parquet(cache_file)
        if kill_states is not None and kill_states.height:
            kill_states.write_parquet(kills_file)
        n_snap = snaps.height if snaps is not None else "cached"
        n_kills = (kill_states.height // 2) if kill_states is not None else "cached"
        print(f"{n_snap} snapshots, {n_kills} kills")
        if purge and demo.exists():
            demo.unlink()
            print(f"    purged raw demo ({demo.name})")

    # Merge every cached per-demo table into the combined dataset.
    cached = sorted(CACHE.glob("*.parquet"))
    if not cached:
        sys.exit("No snapshots produced.")
    combined = pl.concat([pl.read_parquet(p) for p in cached], how="vertical")
    combined.write_parquet(COMBINED)

    print(f"\nCombined: {combined.height} snapshots from {len(cached)} demo(s) "
          f"→ {COMBINED}")
    print(f"CT win rate in labels: {combined['ct_win'].mean():.1%}")


if __name__ == "__main__":
    main(sys.argv)
