"""
EloEngine — tracks per-player overall ELO, grass ELO, and head-to-head win rates.

Key design principle: snapshots are stored BEFORE each match update so they can
be used as training features with zero look-ahead leakage.
"""
import pandas as pd
from .config import INITIAL_ELO, K_ALL, K_GRASS


class EloEngine:
    """
    Tracks per-player overall ELO, grass ELO, and H2H win rates.

    Call ``process_match`` (or ``process_dataframe``) in chronological order.
    All stored snapshots capture ratings *before* the match is applied, so
    they are safe to use as features without future leakage.
    """

    def __init__(self, k_all: float = K_ALL, k_grass: float = K_GRASS):
        self.k_all   = k_all
        self.k_grass = k_grass
        self.elo_all:    dict = {}
        self.elo_grass:  dict = {}
        self._h2h:       dict = {}   # (min_id, max_id) → [min_wins, max_wins]
        self._snapshots: list = []

    # -- private helpers -------------------------------------------------------

    def _get(self, store: dict, pid) -> float:
        return store.get(pid, INITIAL_ELO)

    @staticmethod
    def _expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))

    def _h2h_wr(self, a_id, b_id) -> float:
        """Return a_id's historical H2H win-rate vs b_id (0.5 if < 2 meetings)."""
        k = (min(a_id, b_id), max(a_id, b_id))
        rec = self._h2h.get(k)
        if rec is None:
            return 0.5
        w_min, w_max = rec
        total = w_min + w_max
        if total < 2:
            return 0.5
        return (w_min if a_id == k[0] else w_max) / total

    def _update_h2h(self, winner_id, loser_id) -> None:
        k = (min(winner_id, loser_id), max(winner_id, loser_id))
        if k not in self._h2h:
            self._h2h[k] = [0, 0]
        if winner_id == k[0]:
            self._h2h[k][0] += 1
        else:
            self._h2h[k][1] += 1

    # -- public interface ------------------------------------------------------

    def process_match(self, winner_id, loser_id, surface: str, date) -> None:
        """Record one match: snapshot ratings BEFORE, update AFTER."""
        wa = self._get(self.elo_all,   winner_id)
        la = self._get(self.elo_all,   loser_id)
        wg = self._get(self.elo_grass, winner_id)
        lg = self._get(self.elo_grass, loser_id)

        exp_w = self._expected(wa, la)
        w_h2h = self._h2h_wr(winner_id, loser_id)   # snapshot BEFORE H2H update

        self._snapshots.append({
            "winner_id":       winner_id,
            "loser_id":        loser_id,
            "surface":         surface,
            "date":            date,
            "w_elo_all_pre":   wa,
            "l_elo_all_pre":   la,
            "w_elo_grass_pre": wg,
            "l_elo_grass_pre": lg,
            "w_h2h_wr_pre":    w_h2h,
        })

        # Update overall ELO for every match
        self.elo_all[winner_id] = wa + self.k_all * (1 - exp_w)
        self.elo_all[loser_id]  = la + self.k_all * (0 - (1 - exp_w))

        # Update grass ELO only for grass matches
        if surface == "Grass":
            exp_g = self._expected(wg, lg)
            self.elo_grass[winner_id] = wg + self.k_grass * (1 - exp_g)
            self.elo_grass[loser_id]  = lg + self.k_grass * (0 - (1 - exp_g))

        self._update_h2h(winner_id, loser_id)   # update H2H AFTER snapshot

    def process_dataframe(self, df: pd.DataFrame) -> None:
        """Process all rows of a match DataFrame in order."""
        for _, row in df.iterrows():
            self.process_match(
                row["winner_id"], row["loser_id"],
                row["surface"], row["tourney_date"],
            )

    def get_snapshots_df(self) -> pd.DataFrame:
        """Return all recorded pre-match snapshots as a DataFrame."""
        return pd.DataFrame(self._snapshots)

    def grass_elo(self, pid) -> float:
        """Current grass ELO for player ``pid``."""
        return self._get(self.elo_grass, pid)

    def all_elo(self, pid) -> float:
        """Current overall ELO for player ``pid``."""
        return self._get(self.elo_all, pid)
