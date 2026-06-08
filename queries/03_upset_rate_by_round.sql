-- 03_upset_rate_by_round.sql
-- Historical Wimbledon upset rate: % of matches won by the lower-ranked player,
-- broken down by round. Validates whether model calibration matches reality.

WITH wimbledon AS (
    SELECT
        STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')::DATE  AS tourney_date,
        YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d'))  AS match_year,
        round,
        winner_rank,
        loser_rank,
        -- Upset = lower-ranked (higher number) player won
        CASE
            WHEN winner_rank > loser_rank THEN 1
            ELSE 0
        END AS is_upset
    FROM read_csv_auto('tennis_atp/atp_matches_*.csv', ignore_errors=true, union_by_name=true)
    WHERE tourney_name LIKE '%Wimbledon%'
      AND tourney_level = 'G'
      AND YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')) BETWEEN 2010 AND 2025
      AND score NOT LIKE '%W/O%'
      AND score NOT LIKE '%RET%'
      AND score NOT LIKE '%DEF%'
      AND winner_rank IS NOT NULL
      AND loser_rank  IS NOT NULL
),

round_order AS (
    SELECT *,
        CASE round
            WHEN 'R128' THEN 1
            WHEN 'R64'  THEN 2
            WHEN 'R32'  THEN 3
            WHEN 'R16'  THEN 4
            WHEN 'QF'   THEN 5
            WHEN 'SF'   THEN 6
            WHEN 'F'    THEN 7
            ELSE 8
        END AS round_num
    FROM wimbledon
)

SELECT
    round,
    round_num,
    COUNT(*)                                      AS total_matches,
    SUM(is_upset)                                 AS upsets,
    ROUND(100.0 * SUM(is_upset) / COUNT(*), 1)   AS upset_rate_pct
FROM round_order
GROUP BY round, round_num
ORDER BY round_num;
