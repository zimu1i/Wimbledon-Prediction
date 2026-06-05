"""
Feature engineering: builds training / prediction feature vectors.

Design choices:
- Train on ALL surfaces so we have 64k+ examples instead of ~4k grass-only.
- Grass-specific features are multiplied by ``is_grass`` (masking), so they
  contribute 0 on hard/clay and their true value on grass.
- Each match generates TWO symmetric rows (winner label=1, loser label=0).
- Grass matches receive sample_weight=3.0 to compensate for class imbalance.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "elo_all_diff",        # overall ELO difference (A − B)
    "elo_grass_diff",      # grass ELO diff × is_grass
    "grass_wr_3y_diff",    # 3-year grass win-rate diff × is_grass
    "form_365d_diff",      # 1-year all-surface form diff
    "form_90d_diff",       # 90-day all-surface form diff
    "grass_form_90d_diff", # 90-day grass form diff × is_grass
    "bo5_wr_diff",         # Grand Slam (bo5) win-rate diff
    "rank_diff",           # rank diff (positive → A is better ranked)
    "age_diff",            # A age − B age
    "ace_rate_diff",       # grass ace-rate diff × is_grass
    "first_serve_diff",    # grass 1st-serve % diff × is_grass
    "bp_saved_diff",       # grass break-points-saved % diff × is_grass
    "h2h_wr_centered",     # A's H2H win-rate − 0.5
    "is_grass",            # surface indicator (1 = Grass, 0 = other)
]

# Default serve stats used when a player has insufficient grass serve data
_DEFAULT_SERVE = (0.06, 0.62, 0.62)  # (ace_rate, first_serve_pct, bp_saved_pct)


# ---------------------------------------------------------------------------
# Single-row builder  (used both for training and prediction)
# ---------------------------------------------------------------------------

def make_row(
    a_id, b_id,
    a_elo_all:  float, b_elo_all:  float,
    a_elo_g:    float, b_elo_g:    float,
    grass_wr_3y:    dict,
    form_365d:      dict,
    form_90d:       dict,
    grass_form_90d: dict,
    bo5_wr:         dict,
    rankings:       dict,
    a_age: float, b_age: float,
    serve_stats:    dict,
    a_h2h_wr:       float,   # A's pre-match H2H win-rate vs B
    is_grass:       float,   # 1.0 if Grass, else 0.0
) -> list:
    """Return a 14-element feature vector for the match-up A vs B."""
    a_srv = serve_stats.get(a_id, _DEFAULT_SERVE)
    b_srv = serve_stats.get(b_id, _DEFAULT_SERVE)
    return [
        a_elo_all  - b_elo_all,
        (a_elo_g   - b_elo_g)                                           * is_grass,
        (grass_wr_3y.get(a_id, 0.5) - grass_wr_3y.get(b_id, 0.5))     * is_grass,
        form_365d.get(a_id, 0.5)     - form_365d.get(b_id, 0.5),
        form_90d.get(a_id, 0.5)      - form_90d.get(b_id, 0.5),
        (grass_form_90d.get(a_id, 0.5) - grass_form_90d.get(b_id, 0.5)) * is_grass,
        bo5_wr.get(a_id, 0.5)        - bo5_wr.get(b_id, 0.5),
        float(rankings.get(b_id, 200) - rankings.get(a_id, 200)),
        float(a_age) - float(b_age),
        (a_srv[0] - b_srv[0]) * is_grass,
        (a_srv[1] - b_srv[1]) * is_grass,
        (a_srv[2] - b_srv[2]) * is_grass,
        a_h2h_wr - 0.5,
        is_grass,
    ]


# ---------------------------------------------------------------------------
# Batch feature builder  (training / calibration / validation)
# ---------------------------------------------------------------------------

def build_features(
    df:             pd.DataFrame,
    snapshots:      pd.DataFrame,
    grass_wr_3y:    dict,
    form_365d:      dict,
    form_90d:       dict,
    grass_form_90d: dict,
    bo5_wr:         dict,
    rankings:       dict,
    serve_stats:    dict,
):
    """
    Build symmetric training examples from a match DataFrame.

    For each match we produce:
      - one row with label=1  (winner's perspective)
      - one row with label=0  (loser's perspective)

    Grass-specific features are masked to 0 for non-grass matches.
    Sample weights: grass=3.0, other surfaces=1.0.

    Returns
    -------
    X : np.ndarray, shape (2*n_matches, 14), float32
    y : np.ndarray, shape (2*n_matches,),    int32
    w : np.ndarray, shape (2*n_matches,),    float32  (sample weights)
    """
    snap_dict = {
        (s["winner_id"], s["loser_id"], s["date"]): s
        for _, s in snapshots.iterrows()
    }

    X_rows, y_rows, w_rows = [], [], []

    for _, m in df.iterrows():
        w_id = m["winner_id"]
        l_id = m["loser_id"]
        date = m["tourney_date"]
        is_g   = 1.0 if m["surface"] == "Grass" else 0.0
        weight = 3.0 if is_g else 1.0

        snap = snap_dict.get((w_id, l_id, date))
        if snap is None:
            continue

        w_h2h = float(snap["w_h2h_wr_pre"])
        w_age  = m.get("winner_age", 26) or 26
        l_age  = m.get("loser_age",  26) or 26

        common = dict(
            grass_wr_3y=grass_wr_3y, form_365d=form_365d, form_90d=form_90d,
            grass_form_90d=grass_form_90d, bo5_wr=bo5_wr, rankings=rankings,
            serve_stats=serve_stats, is_grass=is_g,
        )

        # Winner perspective → label 1
        X_rows.append(make_row(
            w_id, l_id,
            snap["w_elo_all_pre"], snap["l_elo_all_pre"],
            snap["w_elo_grass_pre"], snap["l_elo_grass_pre"],
            a_age=w_age, b_age=l_age, a_h2h_wr=w_h2h, **common,
        ))
        y_rows.append(1)
        w_rows.append(weight)

        # Loser perspective → label 0
        X_rows.append(make_row(
            l_id, w_id,
            snap["l_elo_all_pre"], snap["w_elo_all_pre"],
            snap["l_elo_grass_pre"], snap["w_elo_grass_pre"],
            a_age=l_age, b_age=w_age, a_h2h_wr=(1 - w_h2h), **common,
        ))
        y_rows.append(0)
        w_rows.append(weight)

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.int32),
        np.array(w_rows, dtype=np.float32),
    )
