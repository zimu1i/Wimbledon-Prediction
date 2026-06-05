"""
Monte Carlo simulation of the Wimbledon 128-player bracket.

The draw is constructed with standard Wimbledon seeding placement:
  - Seeds 1–8: fixed positions
  - Seeds 9–16: randomly placed within their designated quarter slots
  - Seeds 17–32: randomly placed within their designated eighth slots
  - Unseeded players: fill remaining slots in random order
"""
import numpy as np
import pandas as pd

from .features import make_row, FEATURE_NAMES
from .config import N_SIMS


# ---------------------------------------------------------------------------
# Draw construction
# ---------------------------------------------------------------------------

def get_draw_players(engine, rankings: dict, player_names: dict, n: int = 128) -> pd.DataFrame:
    """
    Return the top-n ranked players as a DataFrame ready for simulation.

    Columns: player_id, name, rank, elo_all, elo_grass
    """
    rows = []
    for pid, rank in sorted(rankings.items(), key=lambda x: x[1])[:n]:
        rows.append({
            "player_id": pid,
            "name":      player_names.get(pid, f"Player {pid}"),
            "rank":      rank,
            "elo_all":   engine.all_elo(pid),
            "elo_grass": engine.grass_elo(pid),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Head-to-head probability  (single match on grass)
# ---------------------------------------------------------------------------

def _match_prob(model, p1: dict, p2: dict, stats: dict, rankings: dict) -> float:
    """P(p1 beats p2) on grass using the trained (possibly calibrated) model."""
    row = make_row(
        p1["player_id"], p2["player_id"],
        p1["elo_all"],   p2["elo_all"],
        p1["elo_grass"], p2["elo_grass"],
        grass_wr_3y    = stats["grass_wr_3y"],
        form_365d      = stats["form_365d"],
        form_90d       = stats["form_90d"],
        grass_form_90d = stats["grass_form_90d"],
        bo5_wr         = stats["bo5_wr"],
        rankings       = rankings,
        a_age=26, b_age=26,          # age unknown at draw time; use tour average
        serve_stats    = stats["serve_stats"],
        a_h2h_wr       = 0.5,        # no draw-specific H2H data at prediction time
        is_grass       = 1.0,
    )
    return float(model.predict_proba(np.array([row], dtype=np.float32))[0][1])


# ---------------------------------------------------------------------------
# Tournament simulation
# ---------------------------------------------------------------------------

def simulate_wimbledon(
    model,
    draw:     pd.DataFrame,
    stats:    dict,
    rankings: dict,
    n_sims:   int = N_SIMS,
) -> pd.DataFrame:
    """
    Run ``n_sims`` Monte Carlo simulations of the Wimbledon 128-player bracket.

    The bracket is rebuilt from scratch each iteration with standard Wimbledon
    seeding placement (seeds 1-8 fixed, 9-16 and 17-32 randomised within their
    designated slots, unseeded players placed randomly in the remaining slots).

    Returns
    -------
    draw DataFrame with an added ``win_pct`` column, sorted descending.
    """
    players  = draw.to_dict("records")
    n        = len(players)
    wins     = np.zeros(n, dtype=np.int32)
    rng      = np.random.default_rng(42)

    seeds    = players[:32]
    unseeded = players[32:]

    # Standard Wimbledon seeding positions (0-indexed)
    seed_pos_1_8  = [0, 127, 63, 64, 31, 96, 32, 95]
    slots_9_16    = [15, 16, 47, 48, 79, 80, 111, 112]
    slots_17_32   = [8,  23, 40, 55, 72, 87, 104, 119,
                     7,  24, 39, 56, 71, 88, 103, 120]

    for _ in range(n_sims):
        rng.shuffle(unseeded)
        bracket = [None] * 128

        # Place seeds 1-8 in fixed positions
        for i, pos in enumerate(seed_pos_1_8):
            bracket[pos] = seeds[i]

        # Seeds 9-16: shuffle within their quarter slots
        s916 = seeds[8:16].copy()
        rng.shuffle(s916)
        for i, pos in enumerate(slots_9_16):
            bracket[pos] = s916[i]

        # Seeds 17-32: shuffle within their eighth slots
        s1732 = seeds[16:32].copy()
        rng.shuffle(s1732)
        for i, pos in enumerate(slots_17_32):
            bracket[pos] = s1732[i]

        # Fill remaining slots with unseeded players
        uns = iter(unseeded)
        for j in range(128):
            if bracket[j] is None:
                bracket[j] = next(uns)

        # Play out the bracket round-by-round
        curr = bracket[:]
        while len(curr) > 1:
            nxt = []
            for i in range(0, len(curr), 2):
                p = _match_prob(model, curr[i], curr[i + 1], stats, rankings)
                nxt.append(curr[i] if rng.random() < p else curr[i + 1])
            curr = nxt

        # Record the winner
        winner_id = curr[0]["player_id"]
        idx = next(i for i, p in enumerate(players) if p["player_id"] == winner_id)
        wins[idx] += 1

    result = draw.copy()
    result["win_pct"] = wins / n_sims * 100
    return result.sort_values("win_pct", ascending=False)
