"""
Tennis Predictor — Data Explorer
Run after pipeline.py to inspect feature quality, ELO leaderboard,
and Wimbledon match coverage.

Usage:
    python explore.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
ATP_DIR  = os.path.join(os.path.dirname(__file__), "..", "tennis_atp")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load outputs ──────────────────────────────────────────────────────────────
print("Loading pipeline outputs...")
matches  = pd.read_parquet(os.path.join(DATA_DIR, "matches_clean.parquet"))
elo_df   = pd.read_parquet(os.path.join(DATA_DIR, "grass_elo.parquet"))
features = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
players  = pd.read_csv(os.path.join(ATP_DIR, "atp_players.csv"), low_memory=False)
players["full_name"] = players["name_first"].fillna("") + " " + players["name_last"].fillna("")

def pid_to_name(pid):
    row = players[players["player_id"] == pid]
    return row["full_name"].values[0].strip() if len(row) else str(pid)

# ── ELO Leaderboard ───────────────────────────────────────────────────────────
print("\n── Grass ELO Leaderboard (current ratings) ──")
# Current ELO = last recorded post-match ELO for each player
latest_w = elo_df.groupby("winner_id")["winner_elo_post"].last().reset_index()
latest_w.columns = ["player_id", "elo"]
latest_l = elo_df.groupby("loser_id")["loser_elo_post"].last().reset_index()
latest_l.columns = ["player_id", "elo"]
current_elo = pd.concat([latest_w, latest_l]).groupby("player_id")["elo"].max().reset_index()
current_elo["name"] = current_elo["player_id"].apply(pid_to_name)
current_elo = current_elo.sort_values("elo", ascending=False).reset_index(drop=True)

# Filter to active players (appeared in 2024 or 2025)
recent_ids = set(
    matches[matches["year"] >= 2024]["winner_id"].tolist() +
    matches[matches["year"] >= 2024]["loser_id"].tolist()
)
active_elo = current_elo[current_elo["player_id"].isin(recent_ids)].head(20)
print(active_elo[["name","elo"]].to_string(index=False))

# ── Wimbledon coverage ────────────────────────────────────────────────────────
wimbledon = features[features["tourney_name"] == "Wimbledon"].copy()
print(f"\n── Wimbledon matches in feature matrix: {len(wimbledon)//2:,} (before symmetrize) ──")
by_year = wimbledon[wimbledon["label"]==1].groupby(wimbledon["date"].dt.year).size()
print(by_year.tail(10).to_string())

# ── Feature completeness ──────────────────────────────────────────────────────
grass_feats = features[features["surface"] == "grass"]
feature_cols = [c for c in features.columns if c not in
    ("date","tourney_name","surface","round","winner_id","loser_id",
     "winner_name","loser_name","label","is_grass","is_grand_slam","best_of")]
null_pct = grass_feats[feature_cols].isnull().mean().sort_values()
print(f"\n── Feature completeness on grass matches (top complete) ──")
print(null_pct.head(10).apply(lambda x: f"{1-x:.1%} complete").to_string())

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Tennis Prediction Platform — Phase 1 Data Report", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

colors = {"grass":"#4CAF50", "clay":"#E07B39", "hard":"#5B9BD5", "unknown":"#aaa"}

# Plot 1: matches per year by surface
ax1 = fig.add_subplot(gs[0, 0])
surf_year = matches.groupby(["year","surface"]).size().unstack(fill_value=0)
for surf in ["grass","clay","hard"]:
    if surf in surf_year.columns:
        ax1.plot(surf_year.index, surf_year[surf], label=surf, color=colors[surf], linewidth=2)
ax1.set_title("Matches per year by surface", fontsize=11)
ax1.set_xlabel("Year"); ax1.set_ylabel("Matches")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# Plot 2: Grass ELO top 15 active players
ax2 = fig.add_subplot(gs[0, 1])
top15 = active_elo.head(15)
bars = ax2.barh(top15["name"][::-1], top15["elo"][::-1], color="#4CAF50", alpha=0.8)
ax2.set_title("Grass ELO — top 15 active", fontsize=11)
ax2.set_xlabel("ELO rating")
ax2.axvline(1500, color="gray", linestyle="--", alpha=0.5, label="Baseline 1500")
for bar, val in zip(bars, top15["elo"][::-1]):
    ax2.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
             f"{val:.0f}", va="center", fontsize=8)
ax2.tick_params(axis="y", labelsize=8)

# Plot 3: ELO difference vs actual outcome (calibration check)
ax3 = fig.add_subplot(gs[0, 2])
feats_label1 = features[features["label"] == 1].copy()
feats_label1["elo_bucket"] = pd.cut(feats_label1["elo_diff"], bins=10)
calibration = feats_label1.groupby("elo_bucket", observed=True)["label"].mean()
bucket_mids = [iv.mid for iv in calibration.index.categories if iv.mid is not None]
ax3.scatter(bucket_mids, calibration.values, color="#4CAF50", zorder=5)
ax3.plot([min(bucket_mids), max(bucket_mids)], [0.5, 1.0], "k--", alpha=0.3, label="Perfect cal.")
ax3.set_title("ELO diff → win rate (calibration)", fontsize=11)
ax3.set_xlabel("ELO difference (A − B)"); ax3.set_ylabel("Actual win rate of A")
ax3.grid(alpha=0.3); ax3.set_ylim(0.3, 1.0)

# Plot 4: Feature null rate on grass
ax4 = fig.add_subplot(gs[1, 0])
null_plot = null_pct.sort_values(ascending=False).head(12)
ax4.barh(null_plot.index[::-1], null_plot.values[::-1] * 100, color="#5B9BD5", alpha=0.8)
ax4.set_title("Feature missing % on grass matches", fontsize=11)
ax4.set_xlabel("% missing"); ax4.axvline(20, color="red", linestyle="--", alpha=0.4, label="20% threshold")
ax4.tick_params(axis="y", labelsize=8); ax4.legend(fontsize=8)

# Plot 5: Wimbledon matches per year
ax5 = fig.add_subplot(gs[1, 1])
wim_year = wimbledon[wimbledon["label"]==1].copy()
wim_year["year"] = wim_year["date"].dt.year
wim_counts = wim_year.groupby("year").size()
ax5.bar(wim_counts.index, wim_counts.values, color="#4CAF50", alpha=0.8)
ax5.set_title("Wimbledon matches per year", fontsize=11)
ax5.set_xlabel("Year"); ax5.set_ylabel("Matches")
ax5.grid(alpha=0.3, axis="y")

# Plot 6: ELO win prob vs actual (grass only)
ax6 = fig.add_subplot(gs[1, 2])
grass_only = features[(features["surface"] == "grass") & (features["label"] == 1)]
grass_only = grass_only.dropna(subset=["elo_win_prob"])
bins = np.linspace(0.3, 0.9, 8)
grass_only["prob_bin"] = pd.cut(grass_only["elo_win_prob"], bins=bins)
cal2 = grass_only.groupby("prob_bin", observed=True)["label"].mean()
mids2 = [iv.mid for iv in cal2.index.categories if iv.mid is not None]
ax6.scatter(mids2, cal2.values, color="#4CAF50", zorder=5, label="Actual")
ax6.plot([0.3, 0.9], [0.3, 0.9], "k--", alpha=0.3, label="Perfect cal.")
ax6.set_title("ELO prob vs actual (grass)", fontsize=11)
ax6.set_xlabel("ELO win probability"); ax6.set_ylabel("Actual win rate")
ax6.legend(fontsize=9); ax6.grid(alpha=0.3)
ax6.set_xlim(0.3, 0.9); ax6.set_ylim(0.3, 0.9)

plt.savefig(os.path.join(OUT_DIR, "phase1_report.png"), dpi=150, bbox_inches="tight")
print(f"\nChart saved → outputs/phase1_report.png")
print("Phase 1 exploration complete.")
