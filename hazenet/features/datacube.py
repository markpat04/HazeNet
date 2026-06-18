"""
Datacube assembly — grid.nc (+ derived physics) → datacube.zarr + targets.

Output cube X has dims (time, channel, lat, lon) where channels are exactly
`cfg.channels + ['emission']`.  The encoder consumes channels[:-1]; the K@E
term consumes the 'emission' channel.  Normalisation is deferred to train time
(train-split only) to avoid leakage.
"""
from __future__ import annotations

import os
import glob
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from . import physics


def _need(grid: xr.Dataset, var: str, ch: str):
    if var not in grid:
        raise KeyError(
            f"channel '{ch}' needs grid variable '{var}' which is missing from "
            f"grid.nc. Either remove '{ch}' from config.channels or re-run the "
            f"grid stage with that source downloaded (e.g. CDS ERA5 for blh/t850)."
        )


def _resolve_channel(ch: str, grid: xr.Dataset, cfg, nt: int) -> np.ndarray:
    """Return a (T,H,W) float32 array for the requested channel name."""
    H, W = cfg.H, cfg.W

    def time_var(v):
        _need(grid, v, ch); return grid[v].values.astype("float32")

    if ch == "dem":
        _need(grid, "dem", ch)
        return np.broadcast_to(grid.dem.values.astype("float32"), (nt, H, W)).copy()
    if ch == "tpi":
        _need(grid, "dem", ch)
        t = physics.tpi(grid.dem.values, radius=cfg.tpi_radius)
        return np.broadcast_to(t, (nt, H, W)).copy()
    if ch == "wind_speed":
        return physics.wind_speed(time_var("u10"), time_var("v10"))
    if ch in ("precip_accum", f"precip_accum{cfg.precip_accum_window}"):
        return physics.precip_accum(time_var("precip"), window=cfg.precip_accum_window)
    if ch == "ventilation":
        ws = physics.wind_speed(time_var("u10"), time_var("v10"))
        return physics.ventilation(time_var("blh"), ws)
    if ch == "inversion":
        return physics.inversion(time_var("t850"), time_var("temp"))
    if ch == "enso":
        if not cfg.enso_csv or not os.path.exists(cfg.enso_csv):
            raise KeyError("channel 'enso' needs features.enso_csv to point at a NOAA ONI table")
        series = physics.enso_series(pd.DatetimeIndex(grid.time.values), cfg.enso_csv)
        return np.broadcast_to(series[:, None, None], (nt, H, W)).astype("float32").copy()
    # plain pass-through time variable (u10, v10, precip, rh, temp, blh, t850, ...)
    return time_var(ch)


def build_datacube(cfg) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    grid = xr.open_dataset(cfg.grid_nc)
    # restrict to configured dates
    gtimes = pd.DatetimeIndex(grid.time.values)
    keep = gtimes.isin(cfg.dates())
    if keep.sum() == 0:
        raise ValueError("no overlap between grid.nc times and config dates — "
                         "check time.years / time.season_months vs the grid window")
    grid = grid.isel(time=keep)
    times = pd.DatetimeIndex(grid.time.values)
    nt = len(times)
    print(f"  grid window: {nt} days  {times[0].date()}..{times[-1].date()}")

    # encoder channels + emission last
    arrays, names = [], []
    for ch in cfg.channels:
        arrays.append(_resolve_channel(ch, grid, cfg, nt)); names.append(ch)
    _need(grid, "emission", "emission")
    arrays.append(grid.emission.values.astype("float32")); names.append("emission")

    X = np.stack(arrays, axis=1).astype("float32")          # (T, C, H, W)
    print(f"  channels ({len(names)}): {names}  X={X.shape}")

    test_mask = np.array([t.year == cfg.test_year for t in times])
    train_mask = ~test_mask

    ds = xr.Dataset(
        {"X": (["time", "channel", "lat", "lon"], X),
         "train_mask": (["time"], train_mask),
         "test_mask": (["time"], test_mask)},
        coords=dict(time=times, channel=names, lat=cfg.LAT, lon=cfg.LON),
        attrs=dict(name=cfg.name, n_met=len(cfg.channels),
                   emission_index=len(names) - 1),
    )
    if os.path.exists(cfg.datacube_zarr):
        shutil.rmtree(cfg.datacube_zarr)
    ds.to_zarr(cfg.datacube_zarr, mode="w")
    print(f"[ok] datacube -> {cfg.datacube_zarr}  "
          f"train={train_mask.sum()} test={test_mask.sum()}")

    _build_targets(cfg, times)


def _build_targets(cfg, times: pd.DatetimeIndex) -> None:
    files = sorted(glob.glob(cfg.pm25_glob))
    if not files:
        raise FileNotFoundError(f"no PM2.5 csv matched {cfg.pm25_glob} — run fetch stage")
    pm = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pm["date"] = pd.to_datetime(pm["date"])
    pm = pm.drop_duplicates(["date", "locationId"])

    pm["ilat"] = np.round((pm["lat"] - cfg.LAT[0]) / cfg.step).astype(int)
    pm["ilon"] = np.round((pm["lon"] - cfg.LON[0]) / cfg.step).astype(int)
    t_map = {pd.Timestamp(t): i for i, t in enumerate(times)}
    pm["tidx"] = pm["date"].map(t_map)

    ok = (pm["ilat"].between(0, cfg.H - 1) & pm["ilon"].between(0, cfg.W - 1)
          & pm["tidx"].notna())
    pm = pm[ok].copy()
    pm["tidx"] = pm["tidx"].astype(int)

    cols = ["date", "tidx", "locationId", "location", "lat", "lon", "ilat", "ilon", "pm25"]
    pm[cols].to_csv(cfg.target_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] targets -> {cfg.target_csv}  "
          f"{len(pm)} rows  {pm['locationId'].nunique()} stations  "
          f"PM2.5 {pm.pm25.min():.0f}-{pm.pm25.max():.0f}")
