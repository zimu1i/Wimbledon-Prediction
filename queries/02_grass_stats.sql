-- 02_grass_stats.sql
-- Per-player aggregate statistics on grass courts (2010–present).
-- Useful for serve dominance analysis and historical win-rate checks.

WITH all_matches AS (
    SELECT
        STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')::DATE AS tourney_date,
        YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')) AS match_year,
        surface, tourney_level,
        winner_id, winner_name, winner_ioc,
        loser_id,  loser_name,
        TRY_CAST(w_ace AS DOUBLE) AS w_ace, TRY_CAST(w_svpt AS DOUBLE) AS w_svpt,
        TRY_CAST(w_1stIn AS DOUBLE) AS w_1stIn,
        TRY_CAST(w_bpSaved AS DOUBLE) AS w_bpSaved, TRY_CAST(w_bpFaced AS DOUBLE) AS w_bpFaced,
        TRY_CAST(l_ace AS DOUBLE) AS l_ace, TRY_CAST(l_svpt AS DOUBLE) AS l_svpt,
        TRY_CAST(l_1stIn AS DOUBLE) AS l_1stIn,
        TRY_CAST(l_bpSaved AS DOUBLE) AS l_bpSaved, TRY_CAST(l_bpFaced AS DOUBLE) AS l_bpFaced
    FROM read_csv_auto('tennis_atp/atp_matches_*.csv', ignore_errors=true, union_by_name=true)
    WHERE surface = 'Grass'
      AND tourney_level IN ('G', 'M', 'A', 'F')
      AND YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')) >= 2010
      AND score NOT LIKE '%W/O%'
      AND score NOT LIKE '%RET%'
      AND score NOT LIKE '%DEF%'
),

winner_side AS (
    SELECT
        winner_id AS player_id, winner_name AS player_name, winner_ioc AS ioc,
        1          AS won,
        w_ace      AS ace,   w_svpt    AS svpt,
        w_1stIn    AS first_in,
        w_bpSaved  AS bp_saved, w_bpFaced AS bp_faced
    FROM all_matches
),

loser_side AS (
    SELECT
        loser_id   AS player_id, loser_name AS player_name, NULL AS ioc,
        0          AS won,
        l_ace      AS ace,   l_svpt    AS svpt,
        l_1stIn    AS first_in,
        l_bpSaved  AS bp_saved, l_bpFaced AS bp_faced
    FROM all_matches
),

combined AS (
    SELECT * FROM winner_side
    UNION ALL
    SELECT * FROM loser_side
)

SELECT
    player_id,
    MAX(player_name)                                        AS player_name,
    MAX(ioc)                                                AS country,
    COUNT(*)                                                AS matches,
    SUM(won)                                                AS wins,
    ROUND(100.0 * SUM(won) / COUNT(*), 1)                  AS win_pct,
    ROUND(SUM(ace)      / NULLIF(SUM(svpt), 0), 4)         AS ace_rate,
    ROUND(SUM(first_in) / NULLIF(SUM(svpt), 0), 4)         AS first_serve_pct,
    ROUND(SUM(bp_saved) / NULLIF(SUM(bp_faced), 0), 4)     AS bp_saved_pct
FROM combined
GROUP BY player_id
HAVING COUNT(*) >= 20            -- minimum 20 grass matches
ORDER BY win_pct DESC;
