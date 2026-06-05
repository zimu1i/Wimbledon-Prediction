"""
High-level pipeline runner: load data → train → simulate → print results.

Entry point: call ``main()`` or run this module directly.
"""
import pandas as pd

from .config import TRAIN_END_YEAR, CALIB_YEAR, N_SIMS
from .data import (
    load_atp_data, load_atp_rankings, load_atp_player_names,
    load_wta_data, load_wta_rankings, load_wta_player_names,
)
from .model import train_model
from .simulate import get_draw_players, simulate_wimbledon
from .stats import build_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_results(label: str, df: pd.DataFrame, top_n: int = 20) -> None:
    """Pretty-print the top-n predictions from a simulation result DataFrame."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  {'Rank':<5} {'Player':<30} {'Country':<8} {'Win %':>7}")
    print(f"  {'-' * 52}")
    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        country = row.get("country", row.get("ioc", "—"))
        print(f"  {i:<5} {row['name']:<30} {country:<8} {row['win_pct']:>6.2f}%")


# ---------------------------------------------------------------------------
# Generic tour pipeline
# ---------------------------------------------------------------------------

def _run_tour(
    tour:            str,
    load_data_fn,
    load_rankings_fn,
    load_names_fn,
) -> pd.DataFrame:
    """
    End-to-end pipeline for one tour (ATP or WTA):
      1. Load all match data (2010–2026) and split into train/calib/val/2026
      2. Train + calibrate XGBoost model
      3. Update ELO with 2026 results, build full stats
      4. Construct 128-player draw, run Monte Carlo simulation
      5. Return results DataFrame

    Parameters
    ----------
    tour            : "ATP" or "WTA" (used in print labels only)
    load_data_fn    : callable() → pd.DataFrame  (e.g. load_atp_data)
    load_rankings_fn: callable() → dict           (e.g. load_atp_rankings)
    load_names_fn   : callable() → dict           (e.g. load_atp_player_names)
    """
    tag = f"[{tour}]"
    print(f"\n{tag} Loading match data (2010–2026)...")

    df_all   = load_data_fn(start_year=2010, end_year=2026)
    df_train = df_all[df_all["tourney_date"].dt.year <= TRAIN_END_YEAR]
    df_calib = df_all[df_all["tourney_date"].dt.year == CALIB_YEAR]
    df_val   = df_all[(df_all["tourney_date"].dt.year >= 2024) &
                      (df_all["tourney_date"].dt.year <= 2025)]
    df_2026  = df_all[df_all["tourney_date"].dt.year == 2026]

    print(f"{tag} Train: {len(df_train):,}  Calib: {len(df_calib):,}  "
          f"Val: {len(df_val):,}  2026: {len(df_2026):,}")

    rankings = load_rankings_fn()
    model, engine, _, rankings = train_model(
        tag=tour,
        df_train=df_train,
        df_calib=df_calib,
        df_val=df_val,
        rankings=rankings,
    )

    print(f"\n{tag} Updating ELO with 2026 results...")
    engine.process_dataframe(df_2026)
    stats_full = build_stats(df_all)

    player_names = load_names_fn()
    draw = get_draw_players(engine, rankings, player_names)

    # Attach country codes from match data
    pid_to_ioc = {}
    for _, row in df_all[["winner_id", "winner_ioc"]].drop_duplicates().iterrows():
        pid_to_ioc[row["winner_id"]] = row["winner_ioc"]
    draw["country"] = draw["player_id"].map(pid_to_ioc).fillna("—")

    print(f"\n{tag} Running {N_SIMS:,} Wimbledon 2026 simulations...")
    results = simulate_wimbledon(model, draw, stats_full, rankings)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  WIMBLEDON 2026 PREDICTION SYSTEM  (v4)")
    print("  ELO + XGBoost + Isotonic Calibration")
    print("=" * 60)

    atp_results = _run_tour("ATP", load_atp_data, load_atp_rankings, load_atp_player_names)
    print_results("ATP MEN'S WIMBLEDON 2026 — WIN PROBABILITIES", atp_results)

    wta_results = _run_tour("WTA", load_wta_data, load_wta_rankings, load_wta_player_names)
    print_results("WTA WOMEN'S WIMBLEDON 2026 — WIN PROBABILITIES", wta_results)

    atp_fav = atp_results.iloc[0]
    wta_fav = wta_results.iloc[0]
    print(f"\n{'=' * 60}")
    print(f"  PREDICTED WINNERS")
    print(f"  Men's:   {atp_fav['name']} ({atp_fav['country']})  — {atp_fav['win_pct']:.1f}%")
    print(f"  Women's: {wta_fav['name']} ({wta_fav['country']})  — {wta_fav['win_pct']:.1f}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
