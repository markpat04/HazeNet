"""
PM2.5 ground-truth (the training labels) — the binding constraint on how far
back the dataset can go. The proven OpenAQ + PCD collector lives in
src/download_pm25_m2.py; rewriting that fiddly client blind would risk silent
label corruption, so this stage reuses csvs already on disk and otherwise tells
you exactly what to run.

For the full RunPod build, extend coverage (more stations / more years) by
running the collector for the wider window, then point paths.pm25_glob here.
"""
from __future__ import annotations

import os
import glob


def fetch(cfg) -> None:
    files = glob.glob(cfg.pm25_glob)
    if files:
        print(f"[pm25] {len(files)} csv present matching {cfg.pm25_glob} -> reuse")
        return
    print(f"[pm25] no csv matched {cfg.pm25_glob}.\n"
          f"       Collect PM2.5 labels first (OpenAQ + Thai PCD), e.g.:\n"
          f"         python src/download_pm25_m2.py\n"
          f"       then place pm25_*.csv where paths.pm25_glob points.\n"
          f"       NOTE: labels are the project's binding constraint — gather as "
          f"many stations/years as available.")
