#!/usr/bin/env python3
"""
Model evaluation script for Wimbledon 2026 Prediction System.

Trains the full ATP pipeline, then produces:
  metrics/calibration_curve.png  — reliability diagram
  metrics/shap_summary.png       — SHAP feature importance
  metrics/upset_analysis.png     — model errors by seed/rank gap
  metrics/feature_importance.png — XGBoost native feature importances

Run from the repo root:
    python3 metrics/evaluate_model.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# Make sure the package is importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import shap
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from tennis_predictor.duckdb_data import load_atp_data, load_atp_rankings
from tennis_predictor.elo import EloEngine
from tennis_predictor.stats import build_stats
from tennis_predictor.features import FEATURE_NAMES, build_features
from tennis_predictor.config import TRAIN_END_YEAR, CALIB_YEAR

OUT_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# 1.  Train ATP model  (reuse from pipeline)
# ---------------------------------------------------------------------------

def _train(df_train, df_calib, df_val, rankings):
    """Return (xgb_raw, calibrated, X_val_g, y_val_g, engine, stats_val)."""
    engine = EloEngine()
    engine.process_dataframe(df_train)
    snaps_tr = engine.get_snapshots_df()
    stats_tr = build_stats(df_train)
    X_tr, y_tr, w_tr = build_features(df_train, snaps_tr, rankings=rankings, **stats_tr)

    xgb = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
        use_label_encoder=False, random_state=42, verbosity=0,
    )
    xgb.fit(X_tr, y_tr, sample_weight=w_tr)

    engine.process_dataframe(df_calib)
    snaps_cal = engine.get_snapshots_df()
    stats_cal = build_stats(pd.concat([df_train, df_calib]))
    X_cal, y_cal, _ = build_features(df_calib, snaps_cal, rankings=rankings, **stats_cal)
    gi = FEATURE_NAMES.index("is_grass")
    gm = X_cal[:, gi] == 1.0
    calibrated = CalibratedClassifierCV(xgb, cv="prefit", method="isotonic")
    calibrated.fit(X_cal[gm], y_cal[gm])

    engine.process_dataframe(df_val)
    snaps_val = engine.get_snapshots_df()
    stats_val = build_stats(pd.concat([df_train, df_calib, df_val]))
    X_val, y_val, _ = build_features(df_val, snaps_val, rankings=rankings, **stats_val)
    gm_v = X_val[:, gi] == 1.0
    X_g, y_g = X_val[gm_v], y_val[gm_v]

    return xgb, calibrated, X_g, y_g, engine, stats_val


# ---------------------------------------------------------------------------
# 2.  Calibration curve  (reliability diagram)
# ---------------------------------------------------------------------------

def plot_calibration(xgb, calibrated, X_g, y_g):
    fig, ax = plt.subplots(figsize=(7, 6))

    for model, label, color in [
        (xgb,        "Uncalibrated XGBoost", "#e67e22"),
        (calibrated, "Isotonic calibration",  "#2980b9"),
    ]:
        prob = model.predict_proba(X_g)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_g, prob, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "s-", color=color, label=label, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives (actual win rate)", fontsize=12)
    ax.set_title("Calibration curve — ATP grass matches (2024–2025 validation)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "calibration_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 3.  SHAP summary plot
# ---------------------------------------------------------------------------

def plot_shap(xgb, X_g):
    print("  Computing SHAP values (this takes ~30s)...")
    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_g)

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_g,
        feature_names=FEATURE_NAMES,
        show=False,
        plot_size=None,
        max_display=14,
    )
    plt.title("SHAP feature importance — ATP XGBoost (grass validation set)", fontsize=13, pad=10)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "shap_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 4.  Upset analysis  (where the model was most wrong)
# ---------------------------------------------------------------------------

def plot_upset_analysis(xgb, calibrated, X_g, y_g, df_val):
    """
    Find matches where the model predicted ≥ 70% confidence but was wrong.
    Plot error rate binned by rank difference.
    """
    prob   = calibrated.predict_proba(X_g)[:, 1]
    errors = ((prob >= 0.5).astype(int) != y_g)

    # Rank diff feature index
    rd_idx = FEATURE_NAMES.index("rank_diff")
    rank_diff = X_g[:, rd_idx]

    bins   = [-200, -50, -20, -5, 5, 20, 50, 200]
    labels = ["<-50", "-50→-20", "-20→-5", "≈0", "5→20", "20→50", ">50"]
    bin_idx = np.digitize(rank_diff, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(labels) - 1)

    error_rates = []
    counts      = []
    for i in range(len(labels)):
        mask = bin_idx == i
        n = mask.sum()
        err = errors[mask].mean() if n > 0 else 0
        error_rates.append(err * 100)
        counts.append(n)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, error_rates, color="#c0392b", alpha=0.8, edgecolor="white")
    ax.axhline(errors.mean() * 100, color="#2c3e50", linestyle="--", lw=1.5,
               label=f"Overall error rate ({errors.mean()*100:.1f}%)")

    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"n={n:,}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Rank difference (A rank − B rank, positive = A is better ranked)", fontsize=11)
    ax.set_ylabel("Error rate (%)", fontsize=11)
    ax.set_title("Model error rate by rank gap — ATP grass (2024–2025)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "upset_analysis.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 5.  Native XGBoost feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(xgb):
    importances = xgb.feature_importances_
    order = np.argsort(importances)
    names = [FEATURE_NAMES[i] for i in order]
    vals  = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2980b9" if "grass" in n or n == "is_grass" else "#7f8c8d" for n in names]
    ax.barh(names, vals, color=colors, edgecolor="white")

    blue_patch = mpatches.Patch(color="#2980b9", label="Grass-specific feature")
    grey_patch = mpatches.Patch(color="#7f8c8d", label="All-surface feature")
    ax.legend(handles=[blue_patch, grey_patch], fontsize=10)

    ax.set_xlabel("XGBoost feature importance (gain)", fontsize=11)
    ax.set_title("Feature importances — ATP XGBoost model", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "feature_importance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 6.  ELO vs ranking accuracy comparison
# ---------------------------------------------------------------------------

def print_elo_vs_ranking(X_g, y_g, engine):
    """Compare ranking-only baseline accuracy against ELO-based model accuracy."""
    rd_idx   = FEATURE_NAMES.index("rank_diff")
    elo_idx  = FEATURE_NAMES.index("elo_all_diff")
    rank_pred = (X_g[:, rd_idx] > 0).astype(int)   # positive rank_diff → A better ranked
    elo_pred  = (X_g[:, elo_idx] > 0).astype(int)  # positive elo_diff  → A higher ELO

    print("\n  Baseline comparison (ATP grass, 2024–2025 validation):")
    print(f"    Ranking-only accuracy : {accuracy_score(y_g, rank_pred):.4f}")
    print(f"    ELO-only accuracy     : {accuracy_score(y_g, elo_pred):.4f}")
    print(f"    XGBoost (full model)  : — see calibration_curve.png for full metrics")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  WIMBLEDON MODEL EVALUATION")
    print("=" * 60)

    print("\n[1/5] Loading ATP data via DuckDB...")
    df_all   = load_atp_data(start_year=2010, end_year=2026)
    df_train = df_all[df_all["tourney_date"].dt.year <= TRAIN_END_YEAR]
    df_calib = df_all[df_all["tourney_date"].dt.year == CALIB_YEAR]
    df_val   = df_all[(df_all["tourney_date"].dt.year >= 2024) &
                      (df_all["tourney_date"].dt.year <= 2025)]
    rankings = load_atp_rankings()
    print(f"  Train={len(df_train):,}  Calib={len(df_calib):,}  Val={len(df_val):,}")

    print("\n[2/5] Training model...")
    xgb, calibrated, X_g, y_g, engine, stats_val = _train(
        df_train, df_calib, df_val, rankings
    )
    print(f"  Validation grass samples: {len(X_g):,}")

    print("\n[3/5] Plotting calibration curve...")
    plot_calibration(xgb, calibrated, X_g, y_g)

    print("\n[4/5] Plotting SHAP summary...")
    plot_shap(xgb, X_g)

    print("\n[4/5] Plotting upset analysis...")
    plot_upset_analysis(xgb, calibrated, X_g, y_g, df_val)

    print("\n[5/5] Plotting feature importance + ELO vs ranking comparison...")
    plot_feature_importance(xgb)
    print_elo_vs_ranking(X_g, y_g, engine)

    print("\nAll plots saved to metrics/")


if __name__ == "__main__":
    main()
