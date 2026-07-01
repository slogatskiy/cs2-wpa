# CS2 Win Probability & Player Impact (WPA)

End-to-end ML pipeline for **Counter-Strike 2**: parse pro demos → model the
live probability of winning a round → derive a **Win Probability Added (WPA)**
metric that measures a player's real contribution — the stuff K/D can't see.

Extends the CSGO work of Xenopoulos et al. ("Valuing Player Actions in
Counter-Strike") to CS2's new demo format, volumetric smokes, and sub-tick.

## Pipeline

| Phase | What | Status |
|------|------|--------|
| **0** | Parse a `.dem` into event + per-tick tables | ✅ done |
| **1A** | Build round *snapshots* (game state → round winner) | ✅ done |
| **1B** | Win-probability model (LightGBM) + calibration | next |
| **2** | WPA — attribute Δ win-prob to player actions | planned |
| **3** | Streamlit dashboard + writeup | planned |

## Layout

```
scripts/explore_demo.py    # Phase 0: one demo → tables (de-risk)
scripts/build_dataset.py   # Phase 1A: data/raw/*.dem → snapshots.parquet
src/cs2wpa/snapshots.py    # feature builder (round snapshots)
data/raw/                  # .dem files (git-ignored)
data/processed/            # parsed tables & datasets (git-ignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_dataset.py        # builds snapshots from data/raw/*.dem
```

## A snapshot row

One moment in a round, described by side-symmetric features and labelled with
who won that round:

`seconds_elapsed, alive_ct/t, hp_ct/t, money_ct/t, equip_ct/t, bomb_planted,
score, alive_diff, hp_diff → ct_win`

Team encoding: `team_num` 2 = T, 3 = CT.
