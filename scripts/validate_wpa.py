"""
Validate WPA against raw fragging, and surface where they diverge.

A good impact metric should (a) correlate strongly with kills / K-D — proof it
isn't nonsense — yet (b) diverge for specific players in interpretable ways:
clutch / high-leverage players rank higher in WPA than their raw frags suggest,
stat-padders rank lower. Those divergences are the interesting findings.

Writes results/wpa_validation.md.

Usage:
    python scripts/validate_wpa.py
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cs2wpa.wpa import leaderboard  # noqa: E402

KILLS = Path("data/processed/wpa_kills.parquet")
OUT = Path("results/wpa_validation.md")
MIN_KILLS = 40


def spearman(a: pl.Series, b: pl.Series) -> float:
    """Spearman = Pearson on ranks."""
    ra = a.rank().to_numpy()
    rb = b.rank().to_numpy()
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def main() -> None:
    if not KILLS.exists():
        sys.exit("No scored kills — run scripts/compute_wpa.py first.")
    kills = pl.read_parquet(KILLS)

    # frags (as killer) and deaths (as victim), by player name
    deaths = kills.group_by("victim_name").len().rename(
        {"victim_name": "player", "len": "deaths"})
    board = leaderboard(kills).rename({"attacker_name": "player"})
    df = (board.join(deaths, on="player", how="left")
          .with_columns(pl.col("deaths").fill_null(0))
          .filter(pl.col("kills") >= MIN_KILLS)
          .with_columns(kd=(pl.col("kills") / pl.col("deaths").clip(lower_bound=1)))
          )

    # correlations: WPA vs raw impact
    r_kills = spearman(df["total_wpa"], df["kills"])
    r_kd = spearman(df["total_wpa"], df["kd"])

    # divergence: WPA rank vs kills rank (positive = WPA rates them higher)
    df = df.with_columns(
        wpa_rank=pl.col("total_wpa").rank(descending=True, method="ordinal").cast(pl.Int32),
        kills_rank=pl.col("kills").rank(descending=True, method="ordinal").cast(pl.Int32),
    ).with_columns(rank_gap=(pl.col("kills_rank") - pl.col("wpa_rank")))

    over = df.sort("rank_gap", descending=True).head(8)   # WPA loves them more than frags do
    under = df.sort("rank_gap").head(8)                   # frags flatter them vs WPA

    def tbl(d):
        rows = []
        for r in d.iter_rows(named=True):
            rows.append(f"| {r['player']} | {r['total_wpa']:.1f} | {r['kills']} | "
                        f"{r['kd']:.2f} | #{r['wpa_rank']} vs #{r['kills_rank']} | {r['rank_gap']:+d} |")
        return rows

    lines = [
        "# WPA Validation — does it measure real impact?",
        "",
        f"{df.height} players with ≥{MIN_KILLS} kills across the event.",
        "",
        "## Sanity: WPA correlates with raw fragging",
        f"- Spearman(WPA, kills) = **{r_kills:.2f}**",
        f"- Spearman(WPA, K/D)   = **{r_kd:.2f}**",
        "",
        "Strong correlation ⇒ WPA agrees with the obvious signal. The value is in "
        "the *residual* — players it rates differently than frags alone.",
        "",
        "## Impact above frags (WPA ranks them higher than kills do)",
        "| Player | WPA | Kills | K/D | WPA vs kills rank | gap |",
        "|--------|-----|-------|-----|-------------------|-----|",
        *tbl(over),
        "",
        "## Frag-flattered (kills rank them higher than WPA)",
        "| Player | WPA | Kills | K/D | WPA vs kills rank | gap |",
        "|--------|-----|-------|-----|-------------------|-----|",
        *tbl(under),
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")

    print(f"Spearman(WPA, kills)={r_kills:.2f}  Spearman(WPA, K/D)={r_kd:.2f}")
    print("Biggest 'impact above frags':")
    print(over.select(["player", "total_wpa", "kills", "kd", "rank_gap"]))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
