-- 01_load_matches.sql
-- Load and clean all ATP main-tour matches from 2010 onward.
-- Run with DuckDB: SELECT * FROM read_csv_auto('tennis_atp/atp_matches_*.csv')
--
-- Usage from Python:
--   import duckdb
--   df = duckdb.query(open('queries/01_load_matches.sql').read()).df()

SELECT
    CAST(tourney_date AS VARCHAR)                                  AS tourney_date_raw,
    STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')::DATE        AS tourney_date,
    YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d'))        AS match_year,
    tourney_name,
    tourney_level,
    surface,
    round,
    best_of,

    winner_id,
    winner_name,
    winner_ioc,
    winner_rank,
    winner_rank_points,
    winner_age,
    winner_ht,

    loser_id,
    loser_name,
    loser_ioc,
    loser_rank,
    loser_rank_points,
    loser_age,
    loser_ht,

    -- Serve stats (winner)
    w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
    w_SvGms, w_bpSaved, w_bpFaced,

    -- Serve stats (loser)
    l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon,
    l_SvGms, l_bpSaved, l_bpFaced,

    score,
    minutes

FROM read_csv_auto('tennis_atp/atp_matches_*.csv', ignore_errors=true, union_by_name=true)

WHERE
    -- Main tour only
    tourney_level IN ('G', 'M', 'A', 'F')
    -- 2010 onward
    AND YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')) >= 2010
    -- Drop retirements, walkovers, defaults
    AND score NOT LIKE '%W/O%'
    AND score NOT LIKE '%RET%'
    AND score NOT LIKE '%DEF%'
    AND score NOT LIKE '%walkover%'

ORDER BY tourney_date;
