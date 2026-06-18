"""
NOAA CPC Oceanic Niño Index (ONI) — the seasonal ENSO index, converted to a
per-month table (year, month, oni). El Niño (positive) years drive drought →
worse burning, a prime candidate for the cross-year non-stationarity signal.

Source: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt  (public, reliable)
Each 3-month season is mapped to its centre month.
"""
from __future__ import annotations

import os
import io
import urllib.request

URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
# 3-month season code -> centre month
SEASON_CENTRE = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                 "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def fetch(cfg) -> None:
    import pandas as pd
    out_dir = os.path.join(cfg.raw_dir, "enso")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "oni.csv")
    if os.path.exists(out) and os.path.getsize(out) > 200:
        print("[enso] skip (oni.csv present)"); return
    try:
        txt = urllib.request.urlopen(URL, timeout=60).read().decode()
    except Exception as e:
        print(f"[enso] download failed: {e}"); return

    df = pd.read_csv(io.StringIO(txt), sep=r"\s+")
    # columns: SEAS YR TOTAL ANOM   (ANOM is the ONI)
    df.columns = [c.upper() for c in df.columns]
    df["month"] = df["SEAS"].map(SEASON_CENTRE)
    df = df.dropna(subset=["month"])
    out_df = df.rename(columns={"YR": "year", "ANOM": "oni"})[["year", "month", "oni"]]
    out_df["month"] = out_df["month"].astype(int)
    out_df.to_csv(out, index=False)
    print(f"[enso] -> {out}  ({len(out_df)} rows, "
          f"{out_df.year.min()}-{out_df.year.max()})")
