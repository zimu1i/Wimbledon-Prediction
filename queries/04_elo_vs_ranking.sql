-- 04_elo_vs_ranking.sql
-- Compare predictive power of ATP ranking vs match outcome on grass.
-- A simple "did the better-ranked player win?" baseline that our ELO model aims to beat.
-- The Python evaluate_model.py computes AUC on top of this.

WITH grass_matches AS (
    SELECT
        STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')::DATE  AS tourney_date,
        YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d'))  AS match_year,
        winner_rank,
        loser_rank,
        -- 1 if favourite (lower rank number) won
        CASE WHEN winner_rank < loser_rank THEN 1 ELSE 0 END     AS fav_won,
        -- rank-based implied probability (simple inverse-rank model)
        ROUND(
            CAST(loser_rank AS DOUBLE) /
            NULLIF((CAST(winner_rank AS DOUBLE) + CAST(loser_rank AS DOUBLE)), 0),
        4)                                                         AS rank_implied_prob
    FROM read_csv_auto('tennis_atp/atp_matches_*.csv', ignore_errors=true, union_by_name=true)
    WHERE surface = 'Grass'
      AND tourney_level IN ('G', 'M', 'A', 'F')
      AND YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')) BETWEEN 2024 AND 2025
      AND score NOT LIKE '%W/O%'
      AND score NOT LIKE '%RET%'
      AND score NOT LIKE '%DEF%'
      AND winner_rank IS NOT NULL
      AND loser_rank  IS NOT NULL
)

SELECT
    COUNT(*)                                    AS total_matches,
    SUM(fav_won)                                AS favourite_wins,
    ROUND(100.0 * AVG(fav_won), 1)             AS ranking_accuracy_pct,
    ROUND(AVG(rank_implied_prob), 4)            AS avg_rank_implied_prob
FROM grass_matches;
