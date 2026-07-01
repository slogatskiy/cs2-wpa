"""
Train the win-probability model on data/processed/snapshots.parquet, print
metrics + feature importance, and save a reliability (calibration) plot and
the fitted model.

Usage:
    python scripts/train_winprob.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cs2wpa.model import feature_importance, train_winprob  # noqa: E402

SNAPS = Path("data/processed/snapshots.parquet")
REPORTS = Path("reports")
MODELS = Path("models")


def main() -> None:
    if not SNAPS.exists():
        sys.exit("No snapshots — run scripts/build_dataset.py first.")
    snaps = pl.read_parquet(SNAPS)

    n_demos = snaps["demo"].n_unique()
    print(f"Loaded {snaps.height} snapshots from {n_demos} demo(s), "
          f"{snaps['round_idx'].n_unique()} rounds each demo (approx).")
    if n_demos == 1:
        print("NOTE: single demo → splitting by round. Metrics are a smoke test "
              "only; add a batch of pro demos for a real evaluation.")

    res = train_winprob(snaps)

    print("\n=== METRICS (held-out) ===")
    for k, v in res.metrics.items():
        print(f"  {k:<12} {v:.4f}" if isinstance(v, float) else f"  {k:<12} {v}")
    print("  (log_loss & brier: lower better; auc: higher better)")

    print("\n=== FEATURE IMPORTANCE ===")
    print(feature_importance(res.model))

    # Reliability plot -------------------------------------------------------
    REPORTS.mkdir(exist_ok=True)
    n_bins = min(10, max(3, len(res.y_true) // 30))
    frac_pos, mean_pred = calibration_curve(res.y_true, res.y_prob, n_bins=n_bins,
                                            strategy="quantile")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax1.plot(mean_pred, frac_pos, "o-", label="model")
    ax1.set(xlabel="predicted P(CT win)", ylabel="actual CT win rate",
            title="Reliability (calibration)")
    ax1.legend()
    ax2.hist(res.y_prob, bins=20, color="steelblue")
    ax2.set(xlabel="predicted P(CT win)", ylabel="count",
            title="Prediction distribution")
    fig.tight_layout()
    plot_path = REPORTS / "calibration.png"
    fig.savefig(plot_path, dpi=120)
    print(f"\nSaved calibration plot → {plot_path}")

    # Persist model ----------------------------------------------------------
    MODELS.mkdir(exist_ok=True)
    model_path = MODELS / "winprob_lgbm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(res.model, f)
    print(f"Saved model → {model_path}")


if __name__ == "__main__":
    main()
