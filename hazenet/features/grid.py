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
import zipfile
import tempfile

import numpy as np
import pandas as pd
import xarray as xr


def _open_era5(f):
    """Open an ERA5 file that may be a CDS zip (instant + accum members)."""
    if zipfile.is_zipfile(f):
        d = tempfile.mkdtemp()
        zipfile.ZipFile(f).extractall(d)
        members = [os.path.join(d, m) for m in os.listdir(d) if m.endswith(".nc")]
        return xr.merge([xr.open_dataset(m) for m in members],
                        compat="override", join="outer")
    return xr.open_dataset(f)


# ── DEM ──
def _grid_dem(cfg) -> np.ndarray:
    import rioxarray
    from scipy.interpolate import RegularGridInterpolator
    paths = sorted(glob.glob(os.path.join(cfg.raw_dir, "dem", "*.tif")))
    if not paths:
        from ..config import ROOT
        paths = sorted(glob.glob(os.path.join(ROOT, "data", "raw_m2", "dem", "*.tif")))
    if not paths:
        raise FileNotFoundError("no DEM tiles (run fetch/dem)")

    # Collect points from each tile without GDAL VRT (merge_arrays crashes on Windows).
    # Downsample each tile by stride to keep memory under control (SRTM is 3600×3600 per °).
    STRIDE = 30  # ~0.008° native → ~0.25° after stride; fine enough to interp to 0.1°
    lon_pts, lat_pts, elev_pts = [], [], []
    for f in paths:
        try:
            t = rioxarray.open_rasterio(f, masked=True)
            x = t.x.values; y = t.y.values
            if x.max() < cfg.lon0 - 0.5 or x.min() > cfg.lon1 + 0.5:
                continue
            if y.max() < cfg.lat0 - 0.5 or y.min() > cfg.lat1 + 0.5:
                continue
            # Clip to domain + small pad, then subsample by STRIDE
            xi = np.where((x >= cfg.lon0 - 0.3) & (x <= cfg.lon1 + 0.3))[0][::STRIDE]
            yi = np.where((y >= cfg.lat0 - 0.3) & (y <= cfg.lat1 + 0.3))[0][::STRIDE]
            if xi.size == 0 or yi.size == 0:
                continue
            v = t.values[0][np.ix_(yi, xi)].astype("float32")
            mask = np.isfinite(v)
            XX, YY = np.meshgrid(x[xi], y[yi])
            lon_pts.append(XX[mask]); lat_pts.append(YY[mask]); elev_pts.append(v[mask])
        except Exception as e:
            print(f"  [dem skip] {os.path.basename(f)}: {e}")

    if not lon_pts:
        raise RuntimeError("DEM: no valid tiles loaded")
    all_lon = np.concatenate(lon_pts)
    all_lat = np.concatenate(lat_pts)
    all_elev = np.concatenate(elev_pts)

    # Bin onto a regular 0.01° grid then interpolate to target resolution
    res = 0.01
    glon = np.arange(cfg.lon0 - 0.05, cfg.lon1 + 0.05 + res, res)
    glat = np.arange(cfg.lat0 - 0.05, cfg.lat1 + 0.05 + res, res)
    acc = np.zeros((len(glat), len(glon)), dtype="float64")
    cnt = np.zeros((len(glat), len(glon)), dtype="int32")
    ji = np.round((all_lon - glon[0]) / res).astype(int)
    ii = np.round((all_lat - glat[0]) / res).astype(int)
    ok = (ii >= 0) & (ii < len(glat)) & (ji >= 0) & (ji < len(glon))
    np.add.at(acc, (ii[ok], ji[ok]), all_elev[ok].astype("float64"))
    np.add.at(cnt, (ii[ok], ji[ok]), 1)
    with np.errstate(invalid="ignore"):
        grid = np.where(cnt > 0, acc / cnt, np.nan).astype("float32")

    # Fill NaN with neighbour average
    df = pd.DataFrame(grid).ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
    grid = df.values.astype("float32")

    # Interpolate to target cfg.LAT / cfg.LON
    interp = RegularGridInterpolator(
        (glat, glon), grid, method="linear", bounds_error=False, fill_value=None)
    LONM, LATM = np.meshgrid(cfg.LON, cfg.LAT)
    dem = interp(np.stack([LATM.ravel(), LONM.ravel()], axis=1)).reshape(cfg.H, cfg.W)
    return dem.astype("float32")


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
            ds = _open_era5(f)
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

    # Soil moisture — era5_sm_*.nc files (downloaded separately via fetch_soil).
    sm_files = sorted(glob.glob(os.path.join(os.path.dirname(sl_files[0]), "era5_sm_*.nc")))
    if sm_files:
        sm = load_daily(sm_files, "mean")
        svars = [v for v in ["swvl1", "volumetric_soil_water_layer_1"] if v in sm]
        if svars:
            out["swvl1"] = sm[svars[0]].values.astype("float32")
            print(f"[grid] soil moisture (swvl1) loaded from {len(sm_files)} files")
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
