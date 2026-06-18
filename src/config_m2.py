"""
M2 shared constants — domain, dates, paths.
All M2 scripts import from here to stay in sync.
"""
import numpy as np
import pandas as pd
import os

# ── Master grid ────────────────────────────────────────────────────────
# Covers: Northern Thailand + Myanmar (Shan/Kachin) + Northern Laos
# Mandatory for transboundary source attribution
LAT  = np.round(np.linspace(14.0, 25.0, 111), 1)   # 111 cells  (14.0 .. 25.0)
LON  = np.round(np.linspace(96.0, 106.0, 101), 1)  # 101 cells  (96.0 .. 106.0)
H, W = len(LAT), len(LON)                          # 111 x 101
STEP = 0.1
BOX  = dict(minx=96.0, miny=14.0, maxx=106.0, maxy=25.0)

# ── Burning seasons ────────────────────────────────────────────────────
# Feb–Apr of each year (fire peaks Feb–Apr in SEA)
YEARS      = [2019, 2020, 2021, 2022, 2023]
TEST_YEAR  = 2023       # held-out test year

def season_dates(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-02-01", f"{year}-04-30", freq="D")

def all_dates() -> pd.DatetimeIndex:
    parts = [season_dates(y) for y in YEARS]
    return pd.DatetimeIndex(sorted(set().union(*[list(p) for p in parts])))

DATES = all_dates()     # ~449 days total

# ── Directories ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(_HERE)
RAW   = os.path.join(ROOT, "data", "raw_m2")
PROC  = os.path.join(ROOT, "data", "processed_m2")
