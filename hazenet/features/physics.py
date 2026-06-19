"""
Physics / geography feature engineering.

Each function is pure (numpy in → numpy out) so it is trivially testable and
reusable on either the small local window or the full 9-year RunPod cube.

All features must be knowable AT prediction time (no future leakage):
  - wind_speed, ventilation, inversion : same-day diagnostics
  - tpi                                : static terrain
  - precip_accum                       : trailing window (past days only)
  - enso                               : monthly index, published with lag
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """10 m wind speed |U| = sqrt(u²+v²).  Drives horizontal dispersion."""
    return np.sqrt(u ** 2 + v ** 2).astype("float32")


def tpi(dem2d: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Topographic Position Index = elevation − local mean elevation.
    Negative = valley/basin (traps smoke), positive = ridge.
    Static (lat,lon).
    """
    from scipy.ndimage import uniform_filter
    size = 2 * radius + 1
    local_mean = uniform_filter(dem2d.astype("float32"), size=size, mode="nearest")
    return (dem2d - local_mean).astype("float32")


def precip_accum(precip: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Trailing precipitation sum over the past `window` days (inclusive of today).
    Captures multi-day wet-deposition memory. precip: (T,H,W).
    """
    T = precip.shape[0]
    out = np.zeros_like(precip, dtype="float32")
    for t in range(T):
        lo = max(0, t - window + 1)
        out[t] = precip[lo:t + 1].sum(axis=0)
    return out


def emission_accum(emission: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Trailing FRP/emission sum over the past `window` days (inclusive of today).
    Smoke from biomass burning lingers and is transported over several days, so
    a same-day FRP field under-represents accumulated haze. This is the smoke
    transport-delay feature (Sprint 2). emission: (T,H,W). Past days only — no
    future leakage.
    """
    T = emission.shape[0]
    out = np.zeros_like(emission, dtype="float32")
    for t in range(T):
        lo = max(0, t - window + 1)
        out[t] = emission[lo:t + 1].sum(axis=0)
    return out


def ventilation(blh: np.ndarray, wind_spd: np.ndarray) -> np.ndarray:
    """
    Ventilation coefficient = BLH × wind speed.
    Low value (shallow boundary layer + calm wind) ⇒ pollution accumulates.
    Classic air-quality dispersion index.
    """
    return (blh * wind_spd).astype("float32")


def inversion(t_upper: np.ndarray, t_surface: np.ndarray) -> np.ndarray:
    """
    Inversion / stability proxy = T(upper level, e.g. 850 hPa) − T(surface).
    Positive ⇒ warm-over-cold cap that suppresses vertical mixing.
    """
    return (t_upper - t_surface).astype("float32")


def enso_series(times: pd.DatetimeIndex, csv_path: str) -> np.ndarray:
    """
    Per-day ONI (Oceanic Niño Index) from a NOAA CPC table.
    Returns (T,) — broadcast to (T,H,W) by the caller.

    Expected CSV columns: year, month, oni  (month 1–12).
    El Niño years (positive ONI) → drought → worse burning; this is the most
    likely 'missing variable' behind cross-year non-stationarity.
    """
    tab = pd.read_csv(csv_path)
    tab.columns = [c.lower() for c in tab.columns]
    lut = {(int(r["year"]), int(r["month"])): float(r["oni"]) for _, r in tab.iterrows()}
    vals = np.array([lut.get((t.year, t.month), np.nan) for t in times], dtype="float32")
    # forward/back fill any gaps so we never inject NaNs into the cube
    s = pd.Series(vals).ffill().bfill()
    return s.values.astype("float32")
