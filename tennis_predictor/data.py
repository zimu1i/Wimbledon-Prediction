"""
Data-loading helpers for ATP and WTA match / rankings / player data.
All functions return pandas DataFrames or dicts; no side-effects.
"""
import os
import pandas as pd

from .config import ATP_DIR, WTA_DIR


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_matches(data_dir: str, prefix: str, levels: list,
                  start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    """Read yearly CSV files, filter to the given tour levels, drop retirements."""
    dfs = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(data_dir, f"{prefix}_matches_{year}.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, low_memory=False))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["tourney_level"].isin(levels)]
    invalid = df["score"].str.contains(r"W/O|RET|DEF|walkover", case=False, na=True)
    df = df[~invalid].copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df.sort_values("tourney_date", inplace=True)
    return df.reset_index(drop=True)


def _load_rankings(data_dir: str, filename: str) -> dict:
    """Return {player_id: rank} for the most-recent ranking date in the file."""
    df = pd.read_csv(os.path.join(data_dir, filename))
    latest = df["ranking_date"].max()
    subset = df[df["ranking_date"] == latest]
    return dict(zip(subset["player"], subset["rank"]))


def _load_player_names(data_dir: str, filename: str) -> dict:
    """Return {player_id: 'First Last'} from a players CSV."""
    df = pd.read_csv(os.path.join(data_dir, filename), low_memory=False)
    df["full_name"] = df["name_first"].str.strip() + " " + df["name_last"].str.strip()
    return dict(zip(df["player_id"], df["full_name"]))


# ---------------------------------------------------------------------------
# Public ATP loaders
# ---------------------------------------------------------------------------

def load_atp_data(start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    """Load ATP main-tour matches: Grand Slams, Masters, ATP 500/250, Finals."""
    return _load_matches(ATP_DIR, "atp", ["G", "M", "A", "F"], start_year, end_year)


def load_atp_rankings() -> dict:
    """Current ATP rankings: {player_id: rank}."""
    return _load_rankings(ATP_DIR, "atp_rankings_current.csv")


def load_atp_player_names() -> dict:
    """ATP player names: {player_id: 'First Last'}."""
    return _load_player_names(ATP_DIR, "atp_players.csv")


# ---------------------------------------------------------------------------
# Public WTA loaders
# ---------------------------------------------------------------------------

def load_wta_data(start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    """Load WTA main-tour matches: Grand Slams, Premier Mandatory/Premier/International, Finals."""
    return _load_matches(WTA_DIR, "wta", ["G", "PM", "P", "I", "F"], start_year, end_year)


def load_wta_rankings() -> dict:
    """Current WTA rankings: {player_id: rank}."""
    return _load_rankings(WTA_DIR, "wta_rankings_current.csv")


def load_wta_player_names() -> dict:
    """WTA player names: {player_id: 'First Last'}."""
    return _load_player_names(WTA_DIR, "wta_players.csv")
