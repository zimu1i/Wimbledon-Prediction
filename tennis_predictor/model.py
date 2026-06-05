"""
Model training, calibration, and evaluation.

``train_model`` is a single generic function used by both the ATP and WTA
pipelines — the only differences are the ``tag`` label printed to stdout and
the ``rankings`` dict passed in from the caller.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score,
)

from .features import FEATURE_NAMES, build_features
from .stats import build_stats
from .elo import EloEngine


# ---------------------------------------------------------------------------
# Evaluation / diagnostics
# ---------------------------------------------------------------------------

def print_metrics(y_true: np.ndarray, proba: np.ndarray,
                  label: str = "", tag: str = "ATP") -> float:
    """
    Print accuracy, log-loss, Brier, AUC-ROC, precision-at-threshold table,
    and a calibration bin table.

    Returns
    -------
    log_loss value (float) — lets callers compare calibrated vs uncalibrated.
    """
    y_pred = (proba >= 0.5).astype(int)
    acc   = accuracy_score(y_true, y_pred)
    ll    = log_loss(y_true, proba)
    brier = brier_score_loss(y_true, proba)
    auc   = roc_auc_score(y_true, proba)

    print(f"\n[{tag}] {label}Evaluation metrics:")
    print(f"      Accuracy  : {acc:.4f}")
    print(f"      Log-loss  : {ll:.4f}")
    print(f"      Brier     : {brier:.4f}  (lower = better, perfect = 0)")
    print(f"      AUC-ROC   : {auc:.4f}")

    print(f"\n[{tag}] Precision at confidence thresholds:")
    print(f"      {'Threshold':<12} {'Precision':>10} {'Coverage':>10} {'n samples':>10}")
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = proba >= thresh
        n = mask.sum()
        if n < 10:
            print(f"      {thresh:.0%}         {'—':>10} {'—':>10} {n:>10}")
            continue
        prec = y_true[mask].mean()
        cov  = mask.mean()
        print(f"      {thresh:.0%}         {prec:>10.4f} {cov:>10.2%} {n:>10,}")

    print(f"\n[{tag}] Calibration (predicted prob vs actual win rate):")
    print(f"      {'Bin':<14} {'Predicted':>10} {'Actual':>10} {'n':>8}")
    bins = [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
    for lo, hi in bins:
        mask = (proba >= lo) & (proba < hi)
        n = mask.sum()
        if n < 5:
            continue
        pred_mean = proba[mask].mean()
        act_mean  = y_true[mask].mean()
        flag = " ✓" if abs(pred_mean - act_mean) < 0.05 else " ✗ (miscal.)"
        print(f"      {lo:.0%}–{hi:.0%}       {pred_mean:>10.3f} {act_mean:>10.3f} {n:>8,}{flag}")

    return ll


# ---------------------------------------------------------------------------
# Main training function  (shared by ATP and WTA)
# ---------------------------------------------------------------------------

def train_model(
    tag:       str,           # "ATP" or "WTA" — used in print statements
    df_train:  pd.DataFrame,  # matches 2010-2022
    df_calib:  pd.DataFrame,  # matches 2023  (isotonic calibration)
    df_val:    pd.DataFrame,  # matches 2024-2025  (evaluation)
    rankings:  dict,          # {player_id: rank} — current season
):
    """
    1. Compute ELO + features for df_train, fit XGBoost.
    2. Extend ELO with df_calib, fit isotonic calibration layer on grass matches.
    3. Extend ELO with df_val, evaluate both models on grass matches.
    4. Return the better model (by log-loss) plus engine / stats / rankings.

    Returns
    -------
    (final_model, engine, stats_val, rankings)
    """
    # ------------------------------------------------------------------
    # Step 1 — Train
    # ------------------------------------------------------------------
    print(f"\n[{tag}] Building ELO + features for training set (2010–{df_train['tourney_date'].dt.year.max()})...")
    engine = EloEngine()
    engine.process_dataframe(df_train)
    snaps_tr = engine.get_snapshots_df()
    stats_tr = build_stats(df_train)
    X_tr, y_tr, w_tr = build_features(df_train, snaps_tr, rankings=rankings, **stats_tr)

    grass_tr = (snaps_tr["surface"] == "Grass").sum()
    print(f"[{tag}] Training samples: {len(X_tr):,}  "
          f"(all surfaces × 2 symmetric; grass matches: {grass_tr:,})")

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr, sample_weight=w_tr)

    # ------------------------------------------------------------------
    # Step 2 — Calibrate  (isotonic on 2023 grass matches)
    # ------------------------------------------------------------------
    print(f"[{tag}] Calibrating on {df_calib['tourney_date'].dt.year.min()} data (isotonic regression)...")
    engine.process_dataframe(df_calib)
    snaps_cal = engine.get_snapshots_df()
    stats_cal = build_stats(pd.concat([df_train, df_calib]))
    X_cal, y_cal, _ = build_features(df_calib, snaps_cal, rankings=rankings, **stats_cal)

    grass_mask = X_cal[:, FEATURE_NAMES.index("is_grass")] == 1.0
    X_cal_g, y_cal_g = X_cal[grass_mask], y_cal[grass_mask]
    print(f"[{tag}] Calibration grass samples: {len(X_cal_g):,}")

    calibrated = CalibratedClassifierCV(xgb, cv="prefit", method="isotonic")
    calibrated.fit(X_cal_g, y_cal_g)

    # ------------------------------------------------------------------
    # Step 3 — Evaluate  (2024-2025 grass only)
    # ------------------------------------------------------------------
    print(f"[{tag}] Evaluating on validation set "
          f"({df_val['tourney_date'].dt.year.min()}–{df_val['tourney_date'].dt.year.max()})...")
    engine.process_dataframe(df_val)
    snaps_val = engine.get_snapshots_df()
    stats_val = build_stats(pd.concat([df_train, df_calib, df_val]))
    X_val, y_val, _ = build_features(df_val, snaps_val, rankings=rankings, **stats_val)

    g_mask = X_val[:, FEATURE_NAMES.index("is_grass")] == 1.0
    X_g, y_g = X_val[g_mask], y_val[g_mask]
    print(f"[{tag}] Validation grass samples: {len(X_g):,}")

    raw_proba = xgb.predict_proba(X_g)[:, 1]
    cal_proba = calibrated.predict_proba(X_g)[:, 1]

    raw_ll = print_metrics(y_g, raw_proba, label="Uncalibrated XGBoost — ", tag=tag)
    cal_ll = print_metrics(y_g, cal_proba, label="Calibrated model — ",     tag=tag)

    # Fall back to uncalibrated if isotonic overfit the calibration set
    if cal_ll > raw_ll:
        print(f"\n[{tag}] Calibration increased loss — using uncalibrated model for predictions.")
        final_model = xgb
    else:
        final_model = calibrated

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------
    print(f"\n[{tag}] Feature importances (XGBoost):")
    for name, imp in sorted(zip(FEATURE_NAMES, xgb.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"       {name:<24} {imp:.4f}")

    return final_model, engine, stats_val, rankings
