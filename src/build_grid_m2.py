"""
Stage 1 M2 — Regrid: แปลงข้อมูลทุกแหล่งลง SEA master grid (111×101 @ 0.1°)
สำหรับทุกวันในฤดูเผา ก.พ.-เม.ย. 2019-2023 (~449 วัน)

Input:  data/raw_m2/{dem,era5,firms}/
Output: data/processed_m2/grid_m2.nc

Run: conda run -n hazenet --no-capture-output python src/build_grid_m2.py
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rioxarray.merge import merge_arrays

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import LAT, LON, H, W, STEP, BOX, DATES, YEARS, ROOT, RAW, PROC


# ── DEM ────────────────────────────────────────────────────────────────
def grid_dem() -> np.ndarray:
    """รวม GeoTIFF tiles -> เฉลี่ยลง 0.1° grid  shape: (H, W)"""
    tile_paths = sorted(glob.glob(os.path.join(RAW, "dem", "*.tif")))
    if not tile_paths:
        raise FileNotFoundError(f"ไม่พบ DEM tiles ใน {RAW}/dem/  "
                                "ให้รัน download_dem_m2.py ก่อน")

    tiles = []
    n_skip = 0
    for f in tile_paths:
        try:
            t = rioxarray.open_rasterio(f, masked=True)
            t = t.rio.clip_box(BOX["minx"] - 0.05, BOX["miny"] - 0.05,
                               BOX["maxx"] + 0.05, BOX["maxy"] + 0.05)
            if t.size == 0:
                continue
            # Pre-validate: force a small read to catch corrupt tiles
            _ = t.values[:, :5, :5]
            tiles.append(t)
        except Exception as e:
            print(f"  [skip corrupt] {os.path.basename(f)}: {e}")
            n_skip += 1
            continue
    if n_skip:
        print(f"  Skipped {n_skip} corrupt tiles")

    if not tiles:
        raise RuntimeError("clip_box ทำให้ทุก tile ว่างเปล่า — ตรวจ BOX")

    dem = merge_arrays(tiles).isel(band=0)
    # coarsen ลง ~0.1° (GLO-30 = 1 arcsec → 30 cells per 0.1°)
    dem = dem.coarsen(x=30, y=30, boundary="trim").mean()
    dem = dem.interp(x=LON, y=LAT, method="linear")
    if dem.y.values[0] > dem.y.values[-1]:
        dem = dem.isel(y=slice(None, None, -1))

    raw_arr = dem.values
    n_nan   = int(np.isnan(raw_arr).sum())
    df      = pd.DataFrame(raw_arr)
    df      = df.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
    arr     = df.values.astype("float32")
    print(f"  DEM: {arr.shape}  {arr.min():.0f}-{arr.max():.0f} m  "
          f"(filled {n_nan} edge NaNs)")
    return arr


# ── ERA5 / NCEP ─────────────────────────────────────────────────────────
def grid_wind() -> tuple:
    """
    ลอง ERA5 ก่อน ถ้าไม่มีใช้ NCEP Reanalysis 2 แทน
    ทั้งคู่ interp -> daily mean (T, H, W)
    """
    # ── ERA5 ──
    era5_files = sorted(glob.glob(os.path.join(RAW, "era5", "era5_wind_*.nc")))
    if era5_files:
        return _grid_era5(era5_files)

    # ── NCEP fallback ──
    ncep_u_files = sorted(glob.glob(os.path.join(RAW, "ncep", "ncep_uwnd_*.nc")))
    ncep_v_files = sorted(glob.glob(os.path.join(RAW, "ncep", "ncep_vwnd_*.nc")))
    if ncep_u_files and ncep_v_files:
        return _grid_ncep(ncep_u_files, ncep_v_files)

    # ── Open-Meteo ERA5 coarse grid fallback ──
    om_file = os.path.join(RAW, "openmeteo", "openmeteo_wind_2019_2023.nc")
    if os.path.exists(om_file):
        return _grid_openmeteo(om_file)

    raise FileNotFoundError(
        "ไม่พบไฟล์ลม ให้รัน download_era5_m2.py / download_ncep_m2.py "
        "หรือ download_openmeteo_m2.py ก่อน"
    )


def grid_met_extra(times_used: pd.DatetimeIndex) -> dict:
    """
    ตัวแปรอุตุฯ เพิ่ม (precip, rh, temp) จาก openmeteo_met_*.nc
    คืน dict ของ (T,H,W) ที่ align กับ times_used. ถ้าไม่มีไฟล์ → คืน {} (ข้าม)
    """
    from scipy.interpolate import RegularGridInterpolator
    met_file = os.path.join(RAW, "openmeteo", "openmeteo_met_2019_2023.nc")
    if not os.path.exists(met_file):
        print("  [met-extra] ไม่พบ openmeteo_met → ข้าม (ใช้แค่ลม+dem)")
        return {}

    ds = xr.open_dataset(met_file)
    clat = ds.lat.values.astype(float)
    clon = ds.lon.values.astype(float)
    if clat[0] > clat[-1]:
        ds = ds.isel(lat=slice(None, None, -1)); clat = ds.lat.values.astype(float)

    ds_times = pd.DatetimeIndex(ds.time.values)
    tpos = {pd.Timestamp(t): i for i, t in enumerate(ds_times)}
    grid_pts = np.array([[la, lo] for la in LAT for lo in LON])

    out = {}
    for var in ["precip", "rh", "temp"]:
        if var not in ds:
            continue
        arr = np.zeros((len(times_used), H, W), dtype="float32")
        vals = ds[var].values
        for ti, t in enumerate(times_used):
            si = tpos.get(pd.Timestamp(t))
            if si is None:
                continue
            interp = RegularGridInterpolator((clat, clon), vals[si],
                                             method="linear",
                                             bounds_error=False, fill_value=None)
            arr[ti] = interp(grid_pts).reshape(H, W)
        out[var] = arr
    if out:
        msg = "  ".join(f"{k}:{v.min():.0f}..{v.max():.0f}" for k, v in out.items())
        print(f"  Met-extra: {msg}")
    return out


def _grid_era5(files: list) -> tuple:
    datasets = []
    for f in files:
        ds    = xr.open_dataset(f)
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        daily = ds.resample({tname: "1D"}).mean()
        if tname != "time":
            daily = daily.rename({tname: "time"})
        datasets.append(daily)

    combined  = xr.concat(datasets, dim="time").sortby("time")
    date_vals = pd.DatetimeIndex(combined.time.values)
    combined  = combined.isel(time=date_vals.isin(DATES))

    lat_dim = "latitude" if "latitude" in combined.dims else "lat"
    lon_dim = "longitude" if "longitude" in combined.dims else "lon"
    combined = combined.interp({lat_dim: LAT, lon_dim: LON}, method="linear")
    if lat_dim != "lat" or lon_dim != "lon":
        combined = combined.rename({lat_dim: "lat", lon_dim: "lon"})

    u = combined.u10.values.astype("float32")
    v = combined.v10.values.astype("float32")
    times_used = pd.DatetimeIndex(combined.time.values)
    print(f"  ERA5: u10 {u.shape}  wind max={np.sqrt(u**2+v**2).max():.1f} m/s"
          f"  {times_used[0].date()}..{times_used[-1].date()}")
    return u, v, times_used


def _grid_ncep(u_files: list, v_files: list) -> tuple:
    """NCEP Reanalysis 2: uwnd/vwnd 10m, daily, Gaussian grid ~1.875°"""
    def load_concat(files, varname):
        parts = []
        for f in files:
            ds = xr.open_dataset(f)
            # NCEP dims: time, lat, lon  var: uwnd/vwnd
            parts.append(ds)
        combined = xr.concat(parts, dim="time").sortby("time")
        date_vals = pd.DatetimeIndex(combined.time.values)
        combined  = combined.isel(time=date_vals.isin(DATES))
        # NCEP lat descending -> sort ascending
        if combined.lat.values[0] > combined.lat.values[-1]:
            combined = combined.isel(lat=slice(None, None, -1))
        # Clip to SEA region before interp (saves memory)
        combined = combined.sel(lat=slice(LAT[0] - 2, LAT[-1] + 2),
                                lon=slice(LON[0] - 2, LON[-1] + 2))
        combined = combined.interp(lat=LAT, lon=LON, method="linear")
        return combined[varname].values.astype("float32"), \
               pd.DatetimeIndex(combined.time.values)

    u, times_u = load_concat(u_files, "uwnd")
    v, times_v = load_concat(v_files, "vwnd")

    # align times
    common = times_u.intersection(times_v)
    u = u[times_u.isin(common)]
    v = v[times_v.isin(common)]
    times_used = common

    print(f"  NCEP: uwnd {u.shape}  wind max={np.sqrt(u**2+v**2).max():.1f} m/s"
          f"  {times_used[0].date()}..{times_used[-1].date()}")
    return u, v, times_used


def _grid_openmeteo(om_file: str) -> tuple:
    """Open-Meteo ERA5 coarse 1° grid -> interp ลง 0.1° SEA grid"""
    from scipy.interpolate import RegularGridInterpolator

    ds = xr.open_dataset(om_file)
    # align dates to DATES
    date_vals   = pd.DatetimeIndex(ds.time.values)
    mask        = date_vals.isin(DATES)
    ds          = ds.isel(time=mask)
    times_used  = pd.DatetimeIndex(ds.time.values)

    coarse_lat = ds.lat.values.astype(float)
    coarse_lon = ds.lon.values.astype(float)

    # sort ascending (needed by RegularGridInterpolator)
    if coarse_lat[0] > coarse_lat[-1]:
        ds = ds.isel(lat=slice(None, None, -1))
        coarse_lat = ds.lat.values.astype(float)

    T  = len(times_used)
    U  = np.zeros((T, H, W), dtype="float32")
    V  = np.zeros((T, H, W), dtype="float32")

    # Target grid
    grid_pts = np.array([[la, lo] for la in LAT for lo in LON])

    for t in range(T):
        for arr_out, var in ((U, "u10"), (V, "v10")):
            layer = ds[var].values[t]  # (n_lat_coarse, n_lon_coarse)
            interp = RegularGridInterpolator(
                (coarse_lat, coarse_lon), layer,
                method="linear", bounds_error=False,
                fill_value=None,
            )
            arr_out[t] = interp(grid_pts).reshape(H, W)

    print(f"  Open-Meteo: u10 {U.shape}  wind max={np.sqrt(U**2+V**2).max():.1f} m/s"
          f"  {times_used[0].date()}..{times_used[-1].date()}")
    return U, V, times_used


# ── FIRMS ────────────────────────────────────────────────────────────────
def grid_firms(times_used: pd.DatetimeIndex) -> np.ndarray:
    """Rasterize FIRMS FRP -> (T, H, W)"""
    files = sorted(glob.glob(os.path.join(RAW, "firms", "firms_*.csv")))
    if not files:
        raise FileNotFoundError(f"ไม่พบ FIRMS csv ใน {RAW}/firms/")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["acq_date"] = pd.to_datetime(df["acq_date"])
        frames.append(df)
    all_firms = pd.concat(frames, ignore_index=True).drop_duplicates()

    T     = len(times_used)
    emis  = np.zeros((T, H, W), dtype="float32")
    t_map = {d.date(): i for i, d in enumerate(times_used)}
    n_used = 0

    for _, r in all_firms.iterrows():
        di = t_map.get(r["acq_date"].date())
        if di is None:
            continue
        ilat = int(round((r["latitude"]  - LAT[0]) / STEP))
        ilon = int(round((r["longitude"] - LON[0]) / STEP))
        if 0 <= ilat < H and 0 <= ilon < W:
            emis[di, ilat, ilon] += r["frp"]
            n_used += 1

    print(f"  FIRMS: {emis.shape}  used {n_used}/{len(all_firms)} points  "
          f"FRP_total={emis.sum():.0f}")
    return emis


# ─────────────────────────────────────────────
def main():
    os.makedirs(PROC, exist_ok=True)
    print(f"Stage 1 M2 — Regrid -> SEA grid {H}×{W} @ 0.1°")
    print(f"  LAT {LAT[0]:.1f}..{LAT[-1]:.1f}  LON {LON[0]:.1f}..{LON[-1]:.1f}")
    print(f"  Target dates: {len(DATES)} days over {YEARS}")

    dem         = grid_dem()
    u, v, times = grid_wind()
    emis        = grid_firms(times)
    extra       = grid_met_extra(times)

    data_vars = dict(
        dem     =(["lat", "lon"],        dem),
        u10     =(["time", "lat", "lon"], u),
        v10     =(["time", "lat", "lon"], v),
        emission=(["time", "lat", "lon"], emis),
    )
    for var, arr in extra.items():
        data_vars[var] = (["time", "lat", "lon"], arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(time=times, lat=LAT, lon=LON),
        attrs=dict(
            title    = "HazeNet M2 master grid (SEA domain)",
            grid_step= STEP,
            window   = f"Feb-Apr {YEARS[0]}-{YEARS[-1]}",
            extra_met= ",".join(extra.keys()) if extra else "none",
            H=H, W=W,
        ),
    )

    out = os.path.join(PROC, "grid_m2.nc")
    ds.to_netcdf(out)
    print(f"\n[ok] -> {out}")
    print(ds)


if __name__ == "__main__":
    main()
