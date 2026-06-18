"""
HazeNet CLI — one entry point for the whole pipeline.

    python -m hazenet.cli --config configs/local.yaml  --stage all
    python -m hazenet.cli --config configs/runpod.yaml --stage datacube,train,eval

Stages (run in this order with 'all'):
    fetch  -> grid -> datacube -> train -> eval -> attribution
"""
from __future__ import annotations

import sys
import argparse

from .config import Config

ORDER = ["fetch", "grid", "datacube", "train", "eval", "attribution"]


def _run_stage(stage: str, cfg: Config):
    print(f"\n{'='*60}\n>> STAGE: {stage}\n{'='*60}")
    if stage == "fetch":
        from .data import fetch_all
        fetch_all(cfg)
    elif stage == "grid":
        from .features.grid import build_grid
        build_grid(cfg)
    elif stage == "datacube":
        from .features.datacube import build_datacube
        build_datacube(cfg)
    elif stage == "train":
        from .train import train
        train(cfg)
    elif stage == "eval":
        from .evaluate import evaluate
        evaluate(cfg)
    elif stage == "loyo":
        from .loyo import loyo
        loyo(cfg)
    elif stage == "attribution":
        from .attribution import Attributor
        a = Attributor(cfg)
        # default demo: most polluted test day for station 0
        import numpy as np
        day = int(np.nanargmax(np.nanmean(a.d["y_raw"], axis=1)))
        print(a.attribution(day, 0))
    else:
        raise SystemExit(f"unknown stage: {stage}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hazenet")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all",
                    help="'all' or comma list: " + ",".join(ORDER))
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    print(cfg.summary())

    stages = ORDER if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    for s in stages:
        _run_stage(s, cfg)
    print("\n✅ done:", stages)


if __name__ == "__main__":
    main(sys.argv[1:])
