"""
HazeNet — Conditionally-Linear Neural Operator pipeline for PM2.5 haze
forecasting + source attribution over the SEA domain (N. Thailand / Myanmar / Laos).

One config-driven package, runnable end-to-end:

    python -m hazenet.cli --config configs/local.yaml  --stage all
    python -m hazenet.cli --config configs/runpod.yaml --stage train

Stages: fetch -> grid -> datacube -> train -> eval -> attribution -> figures
"""

__version__ = "1.0.0"
