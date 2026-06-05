"""
Global constants shared across the Wimbledon prediction pipeline.
"""
import os

# Paths to raw match data (relative to this file)
ATP_DIR = os.path.join(os.path.dirname(__file__), "..", "tennis_atp")
WTA_DIR = os.path.join(os.path.dirname(__file__), "..", "tennis_wta")

# ELO hyperparameters
INITIAL_ELO: float = 1500.0
K_ALL:   float = 32.0    # update rate for overall ELO (all surfaces)
K_GRASS: float = 40.0    # higher K → grass ELO adapts faster to recent results

# Monte Carlo
N_SIMS: int = 10_000     # number of bracket simulations

# Train / calibrate / validate split
TRAIN_END_YEAR: int = 2022   # train on 2010–2022
CALIB_YEAR:     int = 2023   # isotonic calibration on 2023
# Validation: 2024–2025 (implicit in _run_tour)
