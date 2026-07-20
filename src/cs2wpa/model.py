"""
Phase 1B — the win-probability model.

Learns P(CT wins the round | game state) from the round snapshots built in
Phase 1A. The point isn't raw accuracy — it's a *calibrated* probability, so
that "70%" actually means 70%. That calibrated curve is what WPA (Phase 2)
differentiates.

Key correctness detail: snapshots from the same round (and same match) are
highly correlated, so we split by MATCH (demo) to avoid leakage. With a single
demo we fall back to splitting by round.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

FEATURES = [
    "seconds_elapsed", "seconds_remaining",
    "alive_ct", "alive_t", "alive_diff",
    "hp_ct", "hp_t", "hp_diff",
    "money_ct", "money_t", "equip_ct", "equip_t", "equip_diff",
    "bomb_planted", "ct_score_pre", "t_score_pre",
]
TARGET = "ct_win"


@dataclass
class TrainResult:
    model: LGBMClassifier
    metrics: dict[str, float]
    # test-set arrays for plotting calibration
    y_true: np.ndarray
    y_prob: np.ndarray


def _group_key(df: pl.DataFrame) -> np.ndarray:
    """Split by match if we have several demos, else by round (leakage guard)."""
    if df["demo"].n_unique() > 1:
        return df["demo"].to_numpy()
    return (df["demo"] + "#" + df["round_idx"].cast(pl.Utf8)).to_numpy()


def train_winprob(snapshots: pl.DataFrame, test_size: float = 0.25,
                  seed: int = 42) -> TrainResult:
    X = snapshots.select(FEATURES).to_numpy()
    y = snapshots[TARGET].to_numpy()
    groups = _group_key(snapshots)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X[train_idx], y[train_idx])

    prob = model.predict_proba(X[test_idx])[:, 1]
    y_test = y[test_idx]
    metrics = {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "log_loss": log_loss(y_test, prob, labels=[0, 1]),
        "brier": brier_score_loss(y_test, prob),
        # AUC needs both classes present in the test fold
        "auc": roc_auc_score(y_test, prob) if len(np.unique(y_test)) == 2 else float("nan"),
        "base_rate": float(y_test.mean()),
    }
    return TrainResult(model=model, metrics=metrics, y_true=y_test, y_prob=prob)


def feature_importance(model: LGBMClassifier) -> pl.DataFrame:
    """
    Gain-based importance (total loss reduction), normalised to %. Unlike the
    default split-count importance, gain isn't biased toward high-cardinality
    continuous features — so integer state like alive-count shows its true weight.
    """
    gain = model.booster_.feature_importance(importance_type="gain")
    return (
        pl.DataFrame({"feature": FEATURES, "gain_pct": 100 * gain / gain.sum()})
        .sort("gain_pct", descending=True)
    )
