"""Primary steps 3/4: train only Bayesian Deep SVDD on frozen GenD features.

Uses the registered fitting-role group holdout for early stopping, separate
genuine calibration for thresholds, and an explicit lock before final tests.
NIW/Ledoit-Wolf remain a separate parallel entry point: evaluate_diagnostic.py.
"""
from evaluate_diagnostic import main


if __name__ == "__main__":
    main(profile="bayesian")
