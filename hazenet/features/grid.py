"""
Grid stage — regrid raw sources onto the master grid → grid_<name>.nc.

Reads CDS ERA5 (single-levels + 850 hPa temperature), FIRMS fire, and the DEM,
producing daily fields on the H×W grid:
    dem, u10, v10, blh, precip, temp, rh, t850, emission

This is the path for the full CDS-based build. The local config keeps using the
existing Open-Meteo grid (data/processed_m2/grid_m2.nc), so this stage is only
needed when building the big datacube from CDS ERA5.
"""
from __future__ import annotations

import os
import glob

import numpy as np
import pandas as pd
import xarray as xr


# ── DEM ──
def _grid_dem(cfg) -> np.ndarray:
    import rioxarray
    from rioxarray.merge import merge_arrays
    paths = sorted(glob.glob(os.path.join(cfg.raw_dir, "dem", "*.tif")))
    if not paths:
        # fall back to the same-domain m2 DEM
        from ..config import ROOT
        paths = sorted(glob.glob(os.path.join(ROOT, "data", "raw_m2", "dem", "*.tif")))
    if not paths:
        raise FileNotFoundError("no DEM tiles (run fetch/dem)")
    tiles = []
    for f in paths:
        try:
            t = rioxarray.open_rasterio(f, masked=True).rio.clip_box(
                cfg.lon0 - 0.05, cfg.lat0 - 0.05, cfg.lon1 + 0.05, cfg.lat1 + 0.05)
            if t.size:
                _ = t.values[:, :5, :5]; tiles.append(t)
        except Exception as e:
            print(f"  [dem skip] {os.path.basename(f)}: {e}")
    dem = merge_arrays(tiles).isel(band=0).coarsen(x=30, y=30, boundary="trim").mean()
    dem = dem.interp(x=cfg.LON, y=cfg.LAT, method="linear")
    if dem.y.values[0] > dem.y.values[-1]:
        dem = dem.isel(y=slice(None, None, -1))
    df = pd.DataFrame(dem.values).ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
    return df.values.astype("float32")


def _rh_from_dewpoint(t2m_c, d2m_c):
    """Magnus relative humidity (%) from temperature and dewpoint (°C)."""
    es = np.exp(17.625 * t2m_c / (243.04 + t2m_c))
    e = np.exp(17.625 * d2m_c / (243.04 + d2m_c))
    return np.clip(100.0 * e / es, 0, 100).astype("float32")


# ── ERA5 ──
def _grid_era5(cfg, dates):
    sl_files = sorted(glob.glob(os.path.join(cfg.raw_dir, "era5", "era5_sl_*.nc")))
    pl_files = sorted(glob.glob(os.path.join(cfg.raw_dir, "era5", "era5_pl_*.nc")))
    if not sl_files:
        raise FileNotFoundError("no ERA5 single-level files (run fetch/era5)")

    def load_daily(files, how="mean"):
        parts = []
        for f in files:
            ds = xr.open_dataset(f)
            tn = "valid_time" if "valid_time" in ds.dims else "time"
            ds = ds.resample({tn: "1D"}).sum() if how == "sum" else ds.resample({tn: "1D"}).mean()
            if tn != "time":
                ds = ds.rename({tn: "time"})
            parts.append(ds)
        out = xr.concat(parts, dim="time").sortby("time")
        out = out.isel(time=pd.DatetimeIndex(out.time.values).isin(dates))
        la = "latitude" if "latitude" in out.dims else "lat"
        lo = "longitude" if "longitude" in out.dims else "lon"
        out = out.interp({la: cfg.LAT, lo: cfg.LON}, method="linear")
        if la != "lat" or lo != "lon":
            out = out.rename({la: "lat", lo: "lon"})
        return out

    sl_mean = load_daily(sl_files, "mean")
    sl_sum = load_daily(sl_files, "sum")          # for precip accumulation
    times = pd.DatetimeIndex(sl_mean.time.values)

    t2m_c = sl_mean["t2m"].values.astype("float32") - 273.15
    d2m_c = sl_mean["d2m"].values.astype("float32") - 273.15
    out = dict(
        u10=sl_mean["u10"].values.astype("float32"),
        v10=sl_mean["v10"].values.astype("float32"),
        blh=sl_mean["blh"].values.astype("float32"),
        temp=t2m_c,
        rh=_rh_from_dewpoint(t2m_c, d2m_c),
        precip=(sl_sum["tp"].values.astype("float32") * 1000.0),   # m -> mm/day
    )
    if pl_files:
        pl = load_daily(pl_files, "mean")
        tvar = "t" if "t" in pl else list(pl.data_vars)[0]
        arr = pl[tvar]
        if "pressure_level" in arr.dims:
            arr = arr.isel(pressure_level=0)
        out["t850"] = arr.values.astype("float32") - 273.15
    return out, times


# ── FIRMS ──
def _grid_firms(cfg, times):
    files = sorted(glob.glob(os.path.join(cfg.raw_dir, "firms", "firms_*.csv")))
    if not files:
        raise FileNotFoundError("no FIRMS csv (run fetch/firms)")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates()
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    emis = np.zeros((len(times), cfg.H, cfg.W), dtype="float32")
    tmap = {d.date(): i for i, d in enumerate(times)}
    for _, r in df.iterrows():
        di = tmap.get(r["acq_date"].date())
        if di is None:
            continue
        i = int(round((r["latitude"] - cfg.LAT[0]) / cfg.step))
        j = int(round((r["longitude"] - cfg.LON[0]) / cfg.step))
        if 0 <= i < cfg.H and 0 <= j < cfg.W:
            emis[di, i, j] += r["frp"]
    return emis


def build_grid(cfg) -> None:
    os.makedirs(os.path.dirname(cfg.grid_nc), exist_ok=True)
    dates = cfg.dates()
    print(f"[grid] {cfg.H}×{cfg.W}  target {len(dates)} days")
    dem = _grid_dem(cfg)
    era, times = _grid_era5(cfg, dates)
    emis = _grid_firms(cfg, times)

    data_vars = {"dem": (["lat", "lon"], dem),
                 "emission": (["time", "lat", "lon"], emis)}
    for k, v in era.items():
        data_vars[k] = (["time", "lat", "lon"], v)

    ds = xr.Dataset(data_vars, coords=dict(time=times, lat=cfg.LAT, lon=cfg.LON),
                    attrs=dict(title=f"HazeNet grid {cfg.name}", source="CDS ERA5 + FIRMS + DEM"))
    ds.to_netcdf(cfg.grid_nc)
    print(f"[grid] -> {cfg.grid_nc}  vars={list(data_vars)}  days={len(times)}")
