"""
Rolling / aggregate statistics computed from a match DataFrame.
All functions are pure (no global state) and take a context DataFrame as input.
"""
import numpy as np
import pandas as pd


def compute_form_window(df: pd.DataFrame, lookback_days: int,
                        surface: str = None, min_matches: int = 5) -> dict:
    """
    Win rate over the trailing ``lookback_days`` window.

    Parameters
    ----------
    df : DataFrame with columns winner_id, loser_id, tourney_date, surface
    lookback_days : number of days to look back from the last date in df
    surface : if given, restrict to matches on that surface (e.g. "Grass")
    min_matches : players with fewer total matches are excluded (return 0.5 default)

    Returns
    -------
    {player_id: win_rate}
    """
    cutoff = df["tourney_date"].max() - pd.Timedelta(days=lookback_days)
    sub = df[df["tourney_date"] >= cutoff]
    if surface:
        sub = sub[sub["surface"] == surface]

    wins   = sub.groupby("winner_id").size().rename("wins")
    losses = sub.groupby("loser_id").size().rename("losses")
    stats  = pd.concat([wins, losses], axis=1).fillna(0)
    stats["total"] = stats["wins"] + stats["losses"]
    stats = stats[stats["total"] >= min_matches]
    stats["wr"] = stats["wins"] / stats["total"]
    return stats["wr"].to_dict()


def compute_bo5_win_rate(df: pd.DataFrame) -> dict:
    """
    Win rate in Grand Slam matches (best-of-5 format).
    Wimbledon is always best-of-5, so this captures clutch long-match ability.

    Returns {player_id: win_rate} (players with < 5 GS matches excluded).
    """
    gs = df[df["tourney_level"] == "G"]
    wins   = gs.groupby("winner_id").size().rename("wins")
    losses = gs.groupby("loser_id").size().rename("losses")
    stats  = pd.concat([wins, losses], axis=1).fillna(0)
    stats["total"] = stats["wins"] + stats["losses"]
    stats = stats[stats["total"] >= 5]
    stats["bo5_wr"] = stats["wins"] / stats["total"]
    return stats["bo5_wr"].to_dict()


def compute_grass_serve_stats(df: pd.DataFrame) -> dict:
    """
    Aggregate grass-court serve statistics per player.

    Returns
    -------
    {player_id: (ace_rate, first_serve_pct, bp_saved_pct)}
    Players with < 100 serve points on grass are excluded (default tuple used).
    """
    grass = df[(df["surface"] == "Grass") & df["w_svpt"].notna()].copy()

    w = grass[["winner_id", "w_ace", "w_svpt", "w_1stIn", "w_bpSaved", "w_bpFaced"]].copy()
    l = grass[["loser_id",  "l_ace", "l_svpt", "l_1stIn", "l_bpSaved", "l_bpFaced"]].copy()
    w.columns = l.columns = ["pid", "ace", "svpt", "first_in", "bp_saved", "bp_faced"]

    agg = pd.concat([w, l], ignore_index=True).groupby("pid").sum()
    agg = agg[agg["svpt"] >= 100]
    agg["ace_rate"]        = agg["ace"]      / agg["svpt"]
    agg["first_serve_pct"] = agg["first_in"] / agg["svpt"]
    agg["bp_saved_pct"]    = (
        agg["bp_saved"] / agg["bp_faced"].replace(0, np.nan)
    ).fillna(0.62)

    return {
        pid: (r["ace_rate"], r["first_serve_pct"], r["bp_saved_pct"])
        for pid, r in agg.iterrows()
    }


def build_stats(df_context: pd.DataFrame) -> dict:
    """
    Convenience wrapper: compute all six stat dicts from a single context DataFrame.

    Returns a dict with keys:
        grass_wr_3y, form_365d, form_90d, grass_form_90d, bo5_wr, serve_stats
    """
    return {
        "grass_wr_3y":    compute_form_window(df_context, 3 * 365, surface="Grass", min_matches=5),
        "form_365d":      compute_form_window(df_context, 365,                       min_matches=10),
        "form_90d":       compute_form_window(df_context, 90,                        min_matches=5),
        "grass_form_90d": compute_form_window(df_context, 90,      surface="Grass",  min_matches=3),
        "bo5_wr":         compute_bo5_win_rate(df_context),
        "serve_stats":    compute_grass_serve_stats(df_context),
    }
