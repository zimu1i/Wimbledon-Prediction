# Methodology & Design Decisions

This document explains the analytical choices behind the Wimbledon 2026 prediction system and the trade-offs considered at each step.

---

## 1. ELO hyperparameters: K=32 (overall) vs K=40 (grass)

**Why different K values?**  
K controls how fast ELO ratings respond to new results. A higher K means a single match shifts ratings more — which is desirable on grass because the surface rewards a specific style (big serve, net approach) that may not translate from clay or hard courts.

**What was tested:**

| K (grass) | Validation AUC | Log-loss |
|-----------|---------------|----------|
| 32 (same as overall) | 0.768 | 0.581 |
| **40** | **0.774** | **0.568** |
| 50 | 0.771 | 0.573 |
| 64 | 0.766 | 0.579 |

K=40 provided the best log-loss on the 2024–2025 grass holdout. Higher values caused over-reaction to early-season results before Wimbledon.

---

## 2. Isotonic calibration vs Platt scaling

XGBoost tends to output overconfident probabilities — it assigns 85% to matches that are actually won ~75% of the time. Calibration corrects this.

**Options compared:**

| Method | Calibrated log-loss (ATP) | Notes |
|--------|--------------------------|-------|
| None (raw XGBoost) | 0.5704 | Overconfident |
| Platt scaling (sigmoid) | 0.5698 | Very similar to raw; sigmoid too rigid for this distribution |
| **Isotonic regression** | **0.5678** | Best; fits a non-parametric monotone function |

Isotonic was chosen for ATP where calibration grass samples (n=1,166) are large enough to fit reliably.

**Fallback guard:** For WTA, the calibration sample was only 598 grass matches. Isotonic overfitted, raising log-loss from 0.621 to 0.987. The pipeline automatically falls back to the uncalibrated model whenever `calibrated_log_loss > raw_log_loss`.

---

## 3. Training on all surfaces vs grass-only

**Problem:** Training on grass only gives ~4,000 examples — not enough for XGBoost to generalise. Hard and clay surfaces dominate ATP tours (80%+ of matches).

**Solution:** Train on all surfaces (64,114 ATP / 58,694 WTA examples) with:
- Grass-specific features (`elo_grass_diff`, `grass_wr_3y_diff`, `grass_form_90d_diff`, serve stats) multiplied by the `is_grass` indicator — they contribute 0 on hard/clay and their true value on grass.
- Grass matches upweighted **3×** via `sample_weight` to compensate for surface imbalance.

This gave a 3.8% AUC improvement over the grass-only approach with the same feature set.

---

## 4. Data exclusions

| Exclusion | Reason |
|-----------|--------|
| Retirements (RET) | Match outcome does not reflect player ability |
| Walkovers (W/O) | Opponent withdrew before match — no performance data |
| Defaults (DEF) | Same as walkovers |
| Qualifiers & Challengers | ELO pool becomes noisy with lower-tier matches; not predictive for Grand Slams |
| Pre-2010 matches | Historic serve/stats columns (ace, svpt) are sparsely populated before 2010 |

---

## 5. Feature look-ahead prevention

ELO snapshots are taken **before** each match and updated **after**. Head-to-head records follow the same pattern. Rolling win rates are computed on all data up to (not including) the match date by passing a context DataFrame that excludes the match itself.

This prevents any form of future leakage in the training data.

---

## 6. Monte Carlo simulation design

**Why simulate the bracket instead of computing exact probabilities?**  
Exact bracket probability requires summing across all possible draw realisations — exponential complexity for 128 players. Monte Carlo with 10,000 iterations converges to < 0.3% standard error per player and runs in ~90 seconds.

**Seeding placement** follows the official Wimbledon rules:
- Seeds 1 and 2: opposite halves (positions 1 and 128)
- Seeds 3 and 4: randomly placed in the two semi-final positions
- Seeds 5–8: randomly placed in the four quarter-final positions
- Seeds 9–16: randomly distributed to the designated slots within each quarter
- Seeds 17–32: randomly distributed within their respective eighth slots
- Unseeded players: fill all remaining 96 slots at random

---

## 7. Head-to-head feature

`h2h_wr_centered` = (player A's H2H win rate vs player B) − 0.5.

It is set to 0.5 (centred to 0) when fewer than 2 meetings exist — insufficient data to estimate a reliable rate. This avoids encoding noise from single-match H2H records.

During Monte Carlo simulation, `h2h_wr = 0.5` is used for all match-ups since the full draw is unknown at prediction time.

---

## 8. Known limitations

| Issue | Impact | Status |
|-------|--------|--------|
| `grass_form_90d_diff` shows 0 importance | Feature is computed from Dec 2022 snapshot, where no recent grass matches exist in training data | Known; would be fixed by computing rolling stats per-match rather than globally |
| Age feature uses 26 as default | Players without age data (rare) get a neutral value | Low impact; <1% of matches |
| Serve stats not available for all players | ~15% of players lack 100+ grass serve points | Default stats used (tour average) |
| Draw not yet published | H2H feature set to 0.5 for all simulations | Will be fixed once official draw is released |
