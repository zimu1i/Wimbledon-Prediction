"""
DuckDB-based data loading layer.

Replaces the CSV-reading loop in data.py with DuckDB queries that run
directly against the raw CSV files — no ETL step needed. DuckDB reads
all yearly CSVs in a single glob scan, applies the WHERE filters in SQL,
and returns a pandas DataFrame.

This is a drop-in replacement for load_atp_data() / load_wta_data():
the returned DataFrame has the same schema and is sorted by tourney_date.
"""
import os
import duckdb
import pandas as pd

from .config import ATP_DIR, WTA_DIR

# One shared in-memory connection (thread-safe for reads)
_con = duckdb.connect()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sql_load_matches(data_dir: str, prefix: str, levels: list,
                      start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    """
    Read all {prefix}_matches_YYYY.csv files via DuckDB glob scan,
    filter to the requested tour levels and date range, drop retirements.
    Only year-based singles files are included (doubles files are excluded).
    """
    # Build explicit list of per-year file paths to avoid picking up
    # doubles files (e.g. atp_matches_doubles_2003.csv) which have a
    # different schema and would cause a glob schema-mismatch error.
    year_files = [
        os.path.join(data_dir, f"{prefix}_matches_{yr}.csv")
        for yr in range(start_year, end_year + 1)
        if os.path.exists(os.path.join(data_dir, f"{prefix}_matches_{yr}.csv"))
    ]
    if not year_files:
        raise FileNotFoundError(f"No {prefix} match CSVs found in {data_dir}")

    # DuckDB accepts a Python list of paths directly
    levels_sql = ", ".join(f"'{lv}'" for lv in levels)

    sql = f"""
        SELECT *,
            STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')::DATE AS _date_parsed
        FROM read_csv_auto({year_files!r}, union_by_name=true, ignore_errors=true)
        WHERE tourney_level IN ({levels_sql})
          AND YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d'))
              BETWEEN {start_year} AND {end_year}
          AND score NOT LIKE '%W/O%'
          AND score NOT LIKE '%RET%'
          AND score NOT LIKE '%DEF%'
          AND score NOT LIKE '%walkover%'
        ORDER BY _date_parsed
    """

    df = _con.execute(sql).df()

    # Rename parsed date column to match the existing pipeline's schema
    df["tourney_date"] = pd.to_datetime(df["_date_parsed"])
    df.drop(columns=["_date_parsed"], inplace=True)
    return df.reset_index(drop=True)


def _sql_load_rankings(data_dir: str, filename: str) -> dict:
    path = os.path.join(data_dir, filename)
    sql = f"""
        SELECT player, rank
        FROM read_csv_auto('{path}')
        WHERE ranking_date = (SELECT MAX(ranking_date) FROM read_csv_auto('{path}'))
    """
    df = _con.execute(sql).df()
    return dict(zip(df["player"], df["rank"]))


def _sql_load_player_names(data_dir: str, filename: str) -> dict:
    path = os.path.join(data_dir, filename)
    sql = f"""
        SELECT player_id,
               TRIM(name_first) || ' ' || TRIM(name_last) AS full_name
        FROM read_csv_auto('{path}', ignore_errors=true)
    """
    df = _con.execute(sql).df()
    return dict(zip(df["player_id"], df["full_name"]))


# ---------------------------------------------------------------------------
# Public ATP loaders  (identical signatures to data.py)
# ---------------------------------------------------------------------------

def load_atp_data(start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    """Load ATP main-tour matches via DuckDB (Grand Slams, Masters, 500/250, Finals)."""
    return _sql_load_matches(ATP_DIR, "atp", ["G", "M", "A", "F"],
                             start_year, end_year)


def load_atp_rankings() -> dict:
    """Current ATP rankings via DuckDB: {player_id: rank}."""
    return _sql_load_rankings(ATP_DIR, "atp_rankings_current.csv")


def load_atp_player_names() -> dict:
    """ATP player names via DuckDB: {player_id: 'First Last'}."""
    return _sql_load_player_names(ATP_DIR, "atp_players.csv")


# ---------------------------------------------------------------------------
# Public WTA loaders  (identical signatures to data.py)
# ---------------------------------------------------------------------------

def load_wta_data(start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    """Load WTA main-tour matches via DuckDB (Grand Slams, Premier, International, Finals)."""
    return _sql_load_matches(WTA_DIR, "wta", ["G", "PM", "P", "I", "F"],
                             start_year, end_year)


def load_wta_rankings() -> dict:
    """Current WTA rankings via DuckDB: {player_id: rank}."""
    return _sql_load_rankings(WTA_DIR, "wta_rankings_current.csv")


def load_wta_player_names() -> dict:
    """WTA player names via DuckDB: {player_id: 'First Last'}."""
    return _sql_load_player_names(WTA_DIR, "wta_players.csv")


# ---------------------------------------------------------------------------
# Ad-hoc query helper (used by the analysis notebook)
# ---------------------------------------------------------------------------

def sql(query: str) -> pd.DataFrame:
    """
    Run any DuckDB SQL query and return a pandas DataFrame.

    The query can reference raw files via DuckDB's read_csv_auto() or
    reference Python variables with duckdb.query(sql, df=df).

    Example
    -------
    >>> from tennis_predictor.duckdb_data import sql
    >>> sql(\"\"\"
    ...     SELECT surface, COUNT(*) AS n
    ...     FROM read_csv_auto('tennis_atp/atp_matches_*.csv', ignore_errors=true)
    ...     GROUP BY surface ORDER BY n DESC
    ... \"\"\")
    """
    return _con.execute(query).df()
