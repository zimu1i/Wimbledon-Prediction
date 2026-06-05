#!/usr/bin/env python3
"""
Wimbledon 2026 Prediction System  (v4)

Entry point — delegates to the modular pipeline.

Usage:
    python3 tennis_predictor/wimbledon_predictor.py

Modules:
    config.py   — constants (paths, ELO params, simulation settings)
    data.py     — match / rankings / player-name loaders (ATP + WTA)
    elo.py      — EloEngine (overall ELO, grass ELO, H2H tracking)
    stats.py    — rolling win-rate, bo5, serve-stat aggregators
    features.py — FEATURE_NAMES, make_row(), build_features()
    model.py    — train_model(), print_metrics()
    simulate.py — get_draw_players(), simulate_wimbledon()
    train.py    — _run_tour(), print_results(), main()
"""

from tennis_predictor.train import main

if __name__ == "__main__":
    main()
