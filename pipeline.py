"""
Tennis Match Prediction Pipeline — Phase 1
Data ingestion, cleaning, ELO computation, and feature engineering
from the Jeff Sackmann ATP dataset.

Usage:
    python pipeline.py

Outputs (written to data/):
    matches_clean.parquet    — filtered, cleaned match history (2010-2025)
    grass_elo.parquet        — grass-surface ELO ratings for every player
    features.parquet         — model-ready feature matrix with labels
    feature_report.txt       — summary stats printed during build
"""

import os
import glob
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

ATP_DIR    = os.path.join(os.path.dirname(__file__), "..", "tennis_atp")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "data")
YEAR_START = 2010          # enough history for reliable rolling stats
YEAR_END   = 2025
ELO_K      = 32            # ELO learning rate — standard starting point
ELO_BASE   = 1500          # starting ELO for every new player
FORM_WINDOWS = [5, 10, 20] # match-count windows for rolling form features

os.makedirs(OUT_DIR, exist_ok=True)


# ── Step 1: Load and clean raw match data ─────────────────────────────────────

def load_matches(atp_dir: str, year_start: int, year_end: int) -> pd.DataFrame:
    """Load all yearly CSVs, concatenate, and do basic cleaning."""
    print(f"\n[1/4] Loading matches {year_start}–{year_end}...")

    # Only singles main-draw files: atp_matches_YYYY.csv (no doubles/futures/qual)
    import re
    all_files = sorted(glob.glob(os.path.join(atp_dir, "atp_matches_*.csv")))
    files = [f for f in all_files if re.fullmatch(r"atp_matches_\d{4}\.csv", os.path.basename(f))]
    dfs = []
    for f in tqdm(files, desc="  Reading CSVs"):
        year = int(os.path.basename(f).replace("atp_matches_", "").replace(".csv", ""))
        if year_start <= year <= year_end:
            df = pd.read_csv(f, low_memory=False)
            df["year"] = year
            dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)
    print(f"  Raw rows loaded: {len(raw):,}")

    # Parse date
    raw["date"] = pd.to_datetime(raw["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")

    # Drop retirements / walkovers — we need completed matches
    completed_mask = ~raw["score"].str.contains(r"RET|W/O|DEF|Def\.", na=True, case=False)
    raw = raw[completed_mask].copy()

    # Drop rows missing both players' ranks (very old/minor events)
    raw = raw.dropna(subset=["winner_rank", "loser_rank"])

    # Surface normalise
    surface_map = {"Hard": "hard", "Clay": "clay", "Grass": "grass", "Carpet": "hard"}
    raw["surface"] = raw["surface"].map(surface_map).fillna("unknown")

    # Numeric serve stats — coerce so NaN stays NaN, not string
    serve_cols = ["w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon",
                  "w_bpSaved","w_bpFaced",
                  "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon",
                  "l_bpSaved","l_bpFaced"]
    for col in serve_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Sort chronologically — essential for leak-free rolling features
    raw = raw.sort_values(["date", "match_num"], ascending=True).reset_index(drop=True)

    print(f"  Clean rows after filtering: {len(raw):,}")
    print(f"  Surfaces: {raw['surface'].value_counts().to_dict()}")
    return raw


# ── Step 2: Grass-surface ELO ─────────────────────────────────────────────────

def compute_elo(matches: pd.DataFrame, surface_filter: str = "grass",
                k: float = ELO_K, base: float = ELO_BASE) -> pd.DataFrame:
    """
    Compute ELO ratings restricted to one surface.
    Returns a DataFrame with one row per match containing pre-match ELO
    for both winner and loser, plus the updated post-match ratings.

    Key design choices:
    - We use *all* years including pre-2010 if available, because ELO needs
      burn-in time before it becomes meaningful.  We'll load 2000+ here.
    - Surface-specific ELO: a player's grass ELO only updates on grass matches.
      This is the core signal advantage over generic ATP rankings.
    """
    print(f"\n[2/4] Computing {surface_filter} ELO ratings...")

    surface_matches = matches[matches["surface"] == surface_filter].copy()
    print(f"  Grass matches in window: {len(surface_matches):,}")

    ratings: dict[int, float] = defaultdict(lambda: base)
    records = []

    for _, row in tqdm(surface_matches.iterrows(),
                       total=len(surface_matches), desc="  ELO loop"):
        w_id = int(row["winner_id"])
        l_id = int(row["loser_id"])

        r_w = ratings[w_id]
        r_l = ratings[l_id]

        # Expected win probability (standard ELO formula)
        exp_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        exp_l = 1 - exp_w

        # Update
        ratings[w_id] = r_w + k * (1 - exp_w)
        ratings[l_id] = r_l + k * (0 - exp_l)

        records.append({
            "match_id":       row.name,
            "date":           row["date"],
            "tourney_name":   row["tourney_name"],
            "winner_id":      w_id,
            "loser_id":       l_id,
            "winner_elo_pre": r_w,
            "loser_elo_pre":  r_l,
            "winner_elo_post": ratings[w_id],
            "loser_elo_post":  ratings[l_id],
            "elo_diff":       r_w - r_l,          # positive = winner was rated higher
            "winner_exp_prob": exp_w,              # ELO's implied win probability
        })

    elo_df = pd.DataFrame(records)
    top10 = sorted(ratings.items(), key=lambda x: -x[1])[:10]
    print(f"  Top 10 grass ELO (all-time peak): {top10}")
    return elo_df, dict(ratings)


# ── Step 3: Rolling player stats (serve, return, form) ────────────────────────

def _safe_pct(num, den):
    """Percentage, returns NaN when denominator is zero or either value is NaN."""
    try:
        if den is None or num is None:
            return np.nan
        den_f = float(den)
        num_f = float(num)
        if np.isnan(den_f) or np.isnan(num_f) or den_f == 0:
            return np.nan
        return num_f / den_f
    except (TypeError, ValueError):
        return np.nan


def build_player_history(matches: pd.DataFrame) -> dict[int, list]:
    """
    Iterate matches chronologically and build a per-player history list.
    Each entry records stats from the player's perspective after the match.
    We store winner and loser separately so rolling windows work correctly.
    """
    history: dict[int, list] = defaultdict(list)

    for _, row in matches.iterrows():
        date = row["date"]

        # Serve stats — winner perspective
        w_svpt = row.get("w_svpt", np.nan)
        w_1stIn = row.get("w_1stIn", np.nan)
        w_1stWon = row.get("w_1stWon", np.nan)
        w_2ndWon = row.get("w_2ndWon", np.nan)
        w_ace = row.get("w_ace", np.nan)
        w_df = row.get("w_df", np.nan)
        w_bpSaved = row.get("w_bpSaved", np.nan)
        w_bpFaced = row.get("w_bpFaced", np.nan)

        # Serve stats — loser perspective
        l_svpt = row.get("l_svpt", np.nan)
        l_1stIn = row.get("l_1stIn", np.nan)
        l_1stWon = row.get("l_1stWon", np.nan)
        l_2ndWon = row.get("l_2ndWon", np.nan)
        l_ace = row.get("l_ace", np.nan)
        l_df = row.get("l_df", np.nan)
        l_bpSaved = row.get("l_bpSaved", np.nan)
        l_bpFaced = row.get("l_bpFaced", np.nan)

        w_id = int(row["winner_id"])
        l_id = int(row["loser_id"])
        surface = row["surface"]

        history[w_id].append({
            "date": date, "won": 1, "surface": surface,
            "surface_match": 1,
            "1st_serve_pct":   _safe_pct(w_1stIn, w_svpt),
            "1st_serve_won":   _safe_pct(w_1stWon, w_1stIn),
            "2nd_serve_won":   _safe_pct(w_2ndWon, w_svpt - w_1stIn) if pd.notna(w_svpt) and pd.notna(w_1stIn) else np.nan,
            "ace_rate":        _safe_pct(w_ace, w_svpt),
            "df_rate":         _safe_pct(w_df, w_svpt),
            "bp_save_rate":    _safe_pct(w_bpSaved, w_bpFaced),
            "return_pts_won":  _safe_pct(l_svpt - l_1stWon - l_2ndWon, l_svpt) if pd.notna(l_svpt) else np.nan,
            "rank": row["winner_rank"],
        })

        history[l_id].append({
            "date": date, "won": 0, "surface": surface,
            "surface_match": 1,
            "1st_serve_pct":   _safe_pct(l_1stIn, l_svpt),
            "1st_serve_won":   _safe_pct(l_1stWon, l_1stIn),
            "2nd_serve_won":   _safe_pct(l_2ndWon, l_svpt - l_1stIn) if pd.notna(l_svpt) and pd.notna(l_1stIn) else np.nan,
            "ace_rate":        _safe_pct(l_ace, l_svpt),
            "df_rate":         _safe_pct(l_df, l_svpt),
            "bp_save_rate":    _safe_pct(l_bpSaved, l_bpFaced),
            "return_pts_won":  _safe_pct(w_svpt - w_1stWon - w_2ndWon, w_svpt) if pd.notna(w_svpt) else np.nan,
            "rank": row["loser_rank"],
        })

    return history


def rolling_stats(player_matches: list, n: int, surface: str = None) -> dict:
    """
    Given a player's match history (list of dicts, already in chronological
    order), return rolling average over the last n matches.
    Optionally filter to a specific surface.
    """
    pool = player_matches
    if surface:
        pool = [m for m in pool if m["surface"] == surface]
    window = pool[-n:] if len(pool) >= 1 else []

    if not window:
        return {}

    stat_keys = ["won","1st_serve_pct","1st_serve_won","2nd_serve_won",
                 "ace_rate","df_rate","bp_save_rate","return_pts_won"]
    result = {}
    for key in stat_keys:
        vals = [m[key] for m in window if pd.notna(m.get(key))]
        result[key] = np.mean(vals) if vals else np.nan
    result["n_matches"] = len(window)
    return result


# ── Step 4: Build feature matrix ──────────────────────────────────────────────

def build_features(matches: pd.DataFrame, elo_df: pd.DataFrame,
                   grass_only: bool = False) -> pd.DataFrame:
    """
    For each match, compute pre-match features for both players and
    combine them into a single row.  Label = 1 (player A wins).

    To avoid data leakage:
    - All rolling stats use only matches *before* the current one.
    - ELO pre-match ratings are used (not post-match).
    - We iterate chronologically, updating history as we go.
    """
    print("\n[3/4] Building feature matrix...")

    if grass_only:
        pool = matches[matches["surface"] == "grass"].copy()
    else:
        pool = matches.copy()

    pool = pool.sort_values("date").reset_index(drop=True)

    # Build ELO lookup: match_id → pre-match ELO
    elo_lookup = elo_df.set_index("match_id")[
        ["winner_elo_pre", "loser_elo_pre", "winner_exp_prob"]
    ].to_dict("index")

    # Build per-player history incrementally (ensures no leakage)
    history: dict[int, list] = defaultdict(list)
    # H2H tracker: (player_a, player_b) → [outcomes]
    h2h: dict[tuple, list] = defaultdict(list)
    h2h_grass: dict[tuple, list] = defaultdict(list)

    rows = []

    for _, row in tqdm(pool.iterrows(), total=len(pool), desc="  Feature rows"):
        w_id = int(row["winner_id"])
        l_id = int(row["loser_id"])
        surface = row["surface"]
        date = row["date"]
        mid = row.name

        # ── ELO features ──────────────────────────────────────────────────
        elo_info = elo_lookup.get(mid, {})
        w_elo = elo_info.get("winner_elo_pre", ELO_BASE)
        l_elo = elo_info.get("loser_elo_pre", ELO_BASE)
        elo_exp = elo_info.get("winner_exp_prob", 0.5)

        # ── Rolling stats (all surfaces) ──────────────────────────────────
        def get_stats(pid, n, surf=None):
            return rolling_stats(history[pid], n, surface=surf)

        w10 = get_stats(w_id, 10)
        l10 = get_stats(l_id, 10)
        w10g = get_stats(w_id, 10, surf="grass")
        l10g = get_stats(l_id, 10, surf="grass")
        w20 = get_stats(w_id, 20)
        l20 = get_stats(l_id, 20)

        # ── H2H ───────────────────────────────────────────────────────────
        pair = (min(w_id, l_id), max(w_id, l_id))
        h2h_history = h2h[pair]
        h2h_grass_history = h2h_grass[pair]

        # Encode: did player w_id win in each past h2h meeting?
        h2h_w_wins = sum(1 for (pid, outcome) in h2h_history if pid == w_id and outcome == 1)
        h2h_total  = len(h2h_history)
        h2h_win_rate = h2h_w_wins / h2h_total if h2h_total > 0 else 0.5

        h2h_g_wins  = sum(1 for (pid, outcome) in h2h_grass_history if pid == w_id and outcome == 1)
        h2h_g_total = len(h2h_grass_history)
        h2h_grass_win_rate = h2h_g_wins / h2h_g_total if h2h_g_total > 0 else 0.5

        # ── Fatigue proxy (matches in last 14 and 30 days) ─────────────────
        def matches_in_days(pid, days):
            cutoff = date - pd.Timedelta(days=days)
            return sum(1 for m in history[pid] if m["date"] >= cutoff)

        w_fatigue_14 = matches_in_days(w_id, 14)
        l_fatigue_14 = matches_in_days(l_id, 14)
        w_fatigue_30 = matches_in_days(w_id, 30)
        l_fatigue_30 = matches_in_days(l_id, 30)

        # ── Assemble feature row ───────────────────────────────────────────
        # Convention: all features are (winner_stat - loser_stat) or
        # winner_stat alone where subtraction doesn't make sense.
        # Label = 1 always (winner won).  We'll symmetrize during training.

        feature_row = {
            # Identifiers (dropped before training)
            "date":         date,
            "tourney_name": row["tourney_name"],
            "surface":      surface,
            "round":        row["round"],
            "winner_id":    w_id,
            "loser_id":     l_id,
            "winner_name":  row["winner_name"],
            "loser_name":   row["loser_name"],

            # Label
            "label": 1,  # will be symmetrized to add mirrored rows with label=0

            # ── ELO ──────────────────────────────────────────────────────
            "elo_a":        w_elo,
            "elo_b":        l_elo,
            "elo_diff":     w_elo - l_elo,
            "elo_win_prob": elo_exp,   # ELO's own implied probability

            # ── ATP ranking ───────────────────────────────────────────────
            "rank_a":       row["winner_rank"],
            "rank_b":       row["loser_rank"],
            "rank_diff":    row["loser_rank"] - row["winner_rank"],  # positive = winner ranked better

            # ── 10-match rolling (all surfaces) ──────────────────────────
            "win_rate_a_10":         w10.get("won", np.nan),
            "win_rate_b_10":         l10.get("won", np.nan),
            "win_rate_diff_10":      w10.get("won", np.nan) - l10.get("won", np.nan) if w10 and l10 else np.nan,

            "1st_serve_a_10":        w10.get("1st_serve_pct", np.nan),
            "1st_serve_b_10":        l10.get("1st_serve_pct", np.nan),
            "1st_srv_won_a_10":      w10.get("1st_serve_won", np.nan),
            "1st_srv_won_b_10":      l10.get("1st_serve_won", np.nan),
            "ace_rate_diff_10":      (w10.get("ace_rate", 0) or 0) - (l10.get("ace_rate", 0) or 0),
            "bp_save_diff_10":       (w10.get("bp_save_rate", 0) or 0) - (l10.get("bp_save_rate", 0) or 0),
            "return_diff_10":        (w10.get("return_pts_won", 0) or 0) - (l10.get("return_pts_won", 0) or 0),

            # ── 10-match rolling grass-only ───────────────────────────────
            "grass_win_rate_a_10":   w10g.get("won", np.nan),
            "grass_win_rate_b_10":   l10g.get("won", np.nan),
            "grass_win_diff_10":     (w10g.get("won") or 0) - (l10g.get("won") or 0),
            "grass_ace_diff_10":     (w10g.get("ace_rate") or 0) - (l10g.get("ace_rate") or 0),
            "grass_bp_save_diff_10": (w10g.get("bp_save_rate") or 0) - (l10g.get("bp_save_rate") or 0),
            "grass_1st_srv_a_10":    w10g.get("1st_serve_pct", np.nan),
            "grass_1st_srv_b_10":    l10g.get("1st_serve_pct", np.nan),

            # ── 20-match rolling (all surfaces) ──────────────────────────
            "win_rate_diff_20":      (w20.get("won") or 0) - (l20.get("won") or 0),
            "return_diff_20":        (w20.get("return_pts_won") or 0) - (l20.get("return_pts_won") or 0),

            # ── H2H ──────────────────────────────────────────────────────
            "h2h_win_rate_a":        h2h_win_rate,
            "h2h_total":             h2h_total,
            "h2h_grass_win_rate_a":  h2h_grass_win_rate,
            "h2h_grass_total":       h2h_g_total,

            # ── Fatigue ──────────────────────────────────────────────────
            "fatigue_diff_14":       w_fatigue_14 - l_fatigue_14,
            "fatigue_diff_30":       w_fatigue_30 - l_fatigue_30,

            # ── Match context ─────────────────────────────────────────────
            "is_grass":              int(surface == "grass"),
            "is_grand_slam":         int(row.get("tourney_level") == "G"),
            "best_of":               int(row.get("best_of", 3)),
        }

        rows.append(feature_row)

        # ── Update history AFTER recording features (no leakage) ──────────
        entry_w = {
            "date": date, "won": 1, "surface": surface,
            "1st_serve_pct":   _safe_pct(row.get("w_1stIn"), row.get("w_svpt")),
            "1st_serve_won":   _safe_pct(row.get("w_1stWon"), row.get("w_1stIn")),
            "2nd_serve_won":   np.nan,
            "ace_rate":        _safe_pct(row.get("w_ace"), row.get("w_svpt")),
            "df_rate":         _safe_pct(row.get("w_df"), row.get("w_svpt")),
            "bp_save_rate":    _safe_pct(row.get("w_bpSaved"), row.get("w_bpFaced")),
            "return_pts_won":  np.nan,
        }
        entry_l = {
            "date": date, "won": 0, "surface": surface,
            "1st_serve_pct":   _safe_pct(row.get("l_1stIn"), row.get("l_svpt")),
            "1st_serve_won":   _safe_pct(row.get("l_1stWon"), row.get("l_1stIn")),
            "2nd_serve_won":   np.nan,
            "ace_rate":        _safe_pct(row.get("l_ace"), row.get("l_svpt")),
            "df_rate":         _safe_pct(row.get("l_df"), row.get("l_svpt")),
            "bp_save_rate":    _safe_pct(row.get("l_bpSaved"), row.get("l_bpFaced")),
            "return_pts_won":  np.nan,
        }
        history[w_id].append(entry_w)
        history[l_id].append(entry_l)

        h2h[pair].append((w_id, 1))
        h2h[pair].append((l_id, 0))
        if surface == "grass":
            h2h_grass[pair].append((w_id, 1))
            h2h_grass[pair].append((l_id, 0))

    features = pd.DataFrame(rows)

    # ── Symmetrize: add mirrored rows so the model sees both sides ─────────
    # Swap player A and B, flip label to 0
    mirror = features.copy()
    mirror["label"] = 0
    swap_pairs = [
        ("elo_a","elo_b"), ("elo_diff",None), ("elo_win_prob",None),
        ("rank_a","rank_b"), ("rank_diff",None),
        ("win_rate_a_10","win_rate_b_10"), ("win_rate_diff_10",None),
        ("1st_serve_a_10","1st_serve_b_10"), ("1st_srv_won_a_10","1st_srv_won_b_10"),
        ("ace_rate_diff_10",None), ("bp_save_diff_10",None), ("return_diff_10",None),
        ("grass_win_rate_a_10","grass_win_rate_b_10"),
        ("grass_win_diff_10",None),("grass_ace_diff_10",None),
        ("grass_bp_save_diff_10",None),
        ("grass_1st_srv_a_10","grass_1st_srv_b_10"),
        ("win_rate_diff_20",None),("return_diff_20",None),
        ("h2h_win_rate_a",None), ("h2h_grass_win_rate_a",None),
        ("fatigue_diff_14",None),("fatigue_diff_30",None),
    ]
    for col_a, col_b in swap_pairs:
        if col_b:
            mirror[col_a], mirror[col_b] = features[col_b].copy(), features[col_a].copy()
        else:
            mirror[col_a] = -features[col_a]

    # Flip winner/loser identity columns
    mirror["winner_id"], mirror["loser_id"] = features["loser_id"].copy(), features["winner_id"].copy()
    mirror["winner_name"], mirror["loser_name"] = features["loser_name"].copy(), features["winner_name"].copy()
    mirror["h2h_win_rate_a"] = 1 - features["h2h_win_rate_a"]
    mirror["h2h_grass_win_rate_a"] = 1 - features["h2h_grass_win_rate_a"]
    mirror["elo_win_prob"] = 1 - features["elo_win_prob"]

    full = pd.concat([features, mirror], ignore_index=True)
    full = full.sort_values("date").reset_index(drop=True)

    print(f"  Feature rows (before symmetrize): {len(features):,}")
    print(f"  Feature rows (after symmetrize):  {len(full):,}")
    return full


# ── Step 5: Report and save ────────────────────────────────────────────────────

def report_and_save(matches: pd.DataFrame, elo_df: pd.DataFrame,
                    features: pd.DataFrame):
    print("\n[4/4] Saving outputs...")

    matches.to_parquet(os.path.join(OUT_DIR, "matches_clean.parquet"), index=False)
    elo_df.to_parquet(os.path.join(OUT_DIR, "grass_elo.parquet"), index=False)
    features.to_parquet(os.path.join(OUT_DIR, "features.parquet"), index=False)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("TENNIS PREDICTION PIPELINE — DATA REPORT")
    report_lines.append("=" * 60)

    report_lines.append(f"\nTotal matches loaded:      {len(matches):,}")
    report_lines.append(f"Date range:                {matches['date'].min().date()} → {matches['date'].max().date()}")
    report_lines.append(f"\nBy surface:")
    for s, n in matches["surface"].value_counts().items():
        report_lines.append(f"  {s:10s}: {n:,}")

    report_lines.append(f"\nGrass ELO computed for {elo_df['winner_id'].nunique() + elo_df['loser_id'].nunique()} unique player appearances")

    report_lines.append(f"\nFeature matrix shape:  {features.shape}")
    report_lines.append(f"Features columns ({len(features.columns)}): {list(features.columns)}")

    grass_feats = features[features["surface"] == "grass"]
    report_lines.append(f"\nGrass-only rows:       {len(grass_feats):,}")

    wimbledon_feats = features[features["tourney_name"] == "Wimbledon"]
    report_lines.append(f"Wimbledon-only rows:   {len(wimbledon_feats):,}")

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("label","winner_id","loser_id","is_grass","is_grand_slam","best_of")]
    null_pct = features[numeric_cols].isnull().mean().sort_values(ascending=False)
    report_lines.append(f"\nTop features by missing value %:")
    for col, pct in null_pct.head(8).items():
        report_lines.append(f"  {col:35s}: {pct:.1%} missing")

    report_lines.append(f"\nLabel balance: {features['label'].mean():.1%} wins (should be 50%)")
    report_lines.append("\nOutputs written to:")
    for fname in ["matches_clean.parquet", "grass_elo.parquet", "features.parquet"]:
        report_lines.append(f"  data/{fname}")
    report_lines.append("\n" + "=" * 60)

    report = "\n".join(report_lines)
    print(report)

    with open(os.path.join(OUT_DIR, "feature_report.txt"), "w") as f:
        f.write(report)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TENNIS PREDICTION PIPELINE — PHASE 1")
    print("=" * 60)

    # Load matches
    matches = load_matches(ATP_DIR, YEAR_START, YEAR_END)

    # Compute grass ELO (on grass matches only)
    elo_df, final_ratings = compute_elo(matches, surface_filter="grass")

    # Build feature matrix
    features = build_features(matches, elo_df, grass_only=False)

    # Save everything
    report_and_save(matches, elo_df, features)

    print("\nPhase 1 complete. Run `python explore.py` to inspect the data.")
    return matches, elo_df, features


if __name__ == "__main__":
    main()
