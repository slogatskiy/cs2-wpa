"""
Phase 3 — Streamlit dashboard.

Two views on the CS2 WPA project:
  1. WPA leaderboard — rank players by win-probability added.
  2. Round explorer — the live P(CT win) curve through a chosen round, with each
     kill marked and attributed (the "broadcast win-prob bar", but transparent).

Run:
    streamlit run app/dashboard.py
"""

import pickle
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cs2wpa.model import FEATURES  # noqa: E402
from cs2wpa.snapshots import TICKRATE  # noqa: E402
from cs2wpa.wpa import leaderboard  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAPS = ROOT / "data/processed/snapshots.parquet"
KILLS = ROOT / "data/processed/wpa_kills.parquet"
MODEL = ROOT / "models/winprob_lgbm.pkl"

st.set_page_config(page_title="CS2 Win Probability & WPA", page_icon="🎯", layout="wide")


@st.cache_resource
def load_model():
    return pickle.load(open(MODEL, "rb"))


@st.cache_data
def load_snapshots() -> pl.DataFrame:
    df = pl.read_parquet(SNAPS)
    model = load_model()
    wp = model.predict_proba(df.select(FEATURES).to_numpy())[:, 1]
    return df.with_columns(wp_ct=pl.Series(wp))


@st.cache_data
def load_kills() -> pl.DataFrame:
    return pl.read_parquet(KILLS)


def pretty(demo: str) -> str:
    return demo.replace("-vs-", " vs ").replace("-", " ").title()


# --- data ------------------------------------------------------------------
if not (SNAPS.exists() and KILLS.exists() and MODEL.exists()):
    st.error("Missing data. Run build_dataset.py, train_winprob.py and compute_wpa.py first.")
    st.stop()

snaps = load_snapshots()
kills = load_kills()

st.title("🎯 CS2 Win Probability & Player Impact (WPA)")
st.caption("Live round win-probability from parsed CS2 demos, and WPA — a player's "
           "real contribution measured as the win-probability their kills add.")

tab_board, tab_round = st.tabs(["🏆 WPA Leaderboard", "📈 Round Explorer"])

# --- leaderboard -----------------------------------------------------------
with tab_board:
    n_maps = kills["demo"].n_unique()
    st.subheader(f"WPA Leaderboard · {n_maps} maps · {kills.height} kills")
    min_kills = st.slider("Minimum kills", 5, 100, 30, step=5)
    board = leaderboard(kills).filter(pl.col("kills") >= min_kills)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(
            board.with_columns(
                pl.col("total_wpa").round(2), pl.col("wpa_per_kill").round(3)
            ).select(["attacker_name", "total_wpa", "kills", "wpa_per_kill"])
            .rename({"attacker_name": "player", "total_wpa": "WPA",
                     "wpa_per_kill": "WPA/kill"}),
            height=520, use_container_width=True,
        )
    with c2:
        top = board.head(15).reverse()
        fig = go.Figure(go.Bar(
            x=top["total_wpa"], y=top["attacker_name"], orientation="h",
            marker_color="#2a9d8f",
            text=[f"{v:.1f}" for v in top["total_wpa"]], textposition="outside",
        ))
        fig.update_layout(title="Top 15 by total WPA", height=520,
                          margin=dict(l=10, r=10, t=40, b=10), xaxis_title="WPA")
        st.plotly_chart(fig, use_container_width=True)

# --- round explorer --------------------------------------------------------
with tab_round:
    demos = sorted(snaps["demo"].unique().to_list())
    demo = st.selectbox("Map", demos, format_func=pretty)
    d_snaps = snaps.filter(pl.col("demo") == demo)
    rounds = sorted(d_snaps["round_idx"].unique().to_list())
    rnd = st.selectbox("Round", rounds, format_func=lambda r: f"Round {r + 1}")

    r_snaps = d_snaps.filter(pl.col("round_idx") == rnd).sort("seconds_elapsed")
    if r_snaps.height == 0:
        st.info("No data for this round.")
        st.stop()

    # map kill ticks to seconds within this round (seconds = (tick-start)/tickrate)
    start_tick = int((r_snaps["tick"] - (r_snaps["seconds_elapsed"] * TICKRATE)).mean())
    r_kills = kills.filter((pl.col("demo") == demo) & (pl.col("round_idx") == rnd)).with_columns(
        secs=(pl.col("tick") - start_tick) / TICKRATE
    )

    fig = go.Figure()
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig.add_trace(go.Scatter(
        x=r_snaps["seconds_elapsed"], y=r_snaps["wp_ct"], mode="lines",
        line=dict(color="#264653", width=3), name="P(CT win)",
    ))
    for k in r_kills.iter_rows(named=True):
        fig.add_trace(go.Scatter(
            x=[k["secs"]], y=[k["wp_after"]], mode="markers",
            marker=dict(size=11, color="#e76f51" if k["wpa"] >= 0 else "#8d99ae",
                        line=dict(width=1, color="white")),
            name=k["attacker_name"], showlegend=False,
            hovertext=f"{k['attacker_name']} ✗ {k['victim_name']} ({k['weapon']})<br>"
                      f"WPA {k['wpa']:+.3f}", hoverinfo="text",
        ))
    winner = "CT" if r_snaps["ct_win"][0] == 1 else "T"
    fig.update_layout(
        title=f"{pretty(demo)} — Round {rnd + 1}  ·  winner: {winner}",
        xaxis_title="seconds into round", yaxis_title="P(CT win)",
        yaxis_range=[0, 1], height=460, margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Dots = kills, placed at the post-kill win probability. Orange swings "
               "the round toward the killer's side. Hover for who killed whom and the WPA.")
    st.dataframe(
        r_kills.with_columns(pl.col("secs").round(1), pl.col("wpa").round(3))
        .select(["secs", "attacker_name", "victim_name", "weapon", "wpa"])
        .rename({"secs": "t", "attacker_name": "killer", "victim_name": "victim"}),
        use_container_width=True,
    )
