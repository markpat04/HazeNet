"""
Data acquisition stages. Each fetcher is resume-able (skips files already on
disk) and reads its window from the Config (years, season_months, domain).

fetch_all(cfg) runs them in order. Individual fetchers can be imported and
called directly. Network/credentials required:
  - ERA5 : ~/.cdsapirc (CDS API key)        -> blh, t850, winds, precip, t2m, d2m
  - FIRMS: FIRMS_MAP_KEY (env or src/.keys)  -> fire FRP
  - OpenAQ PM2.5, DEM, NOAA ONI : public
"""
from __future__ import annotations

import os


def get_key(name: str):
    """Read an API key from env, else from src/.keys (KEY=value lines)."""
    v = os.environ.get(name)
    if v:
        return v.strip()
    from .. config import ROOT
    kp = os.path.join(ROOT, "src", ".keys")
    if os.path.exists(kp):
        for line in open(kp, encoding="utf-8"):
            if line.startswith(name):
                return line.split("=", 1)[1].strip()
    return None


def fetch_all(cfg) -> None:
    from . import fetch_dem, fetch_era5, fetch_firms, fetch_pm25, fetch_enso
    fetch_dem.fetch(cfg)
    fetch_era5.fetch(cfg)
    fetch_firms.fetch(cfg)
    fetch_pm25.fetch(cfg)
    fetch_enso.fetch(cfg)
