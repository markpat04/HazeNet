"""
Stage 1 — Regrid: แปลงข้อมูล 4 แหล่งให้อยู่บน master grid เดียวกัน
  master grid: lat 18.0..19.5, lon 98.0..100.0, step 0.1° -> 16 x 21
  เวลา: รายวัน 2023-03-19..31 -> 13 วัน

  - DEM (30m, static)         -> เฉลี่ยลงช่อง 0.1°            -> dem(lat,lon)
  - ERA5 (0.25°, 3-hourly)    -> เฉลี่ยรายวัน + interp 0.1°   -> u10/v10(time,lat,lon)
  - FIRMS (จุดไฟ)             -> รวม FRP ต่อช่องต่อวัน        -> emission(time,lat,lon)

รัน:  conda run -n hazenet --no-capture-output python src/build_grid.py
ออก:  data/processed/grid.nc
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

# master grid (cell centers)
LAT = np.round(np.arange(18.0, 19.5 + 1e-6, 0.1), 1)   # 16
LON = np.round(np.arange(98.0, 100.0 + 1e-6, 0.1), 1)  # 21
DATES = pd.date_range("2023-03-19", "2023-03-31", freq="D")  # 13
STEP = 0.1
BOX = dict(minx=98.0, miny=18.0, maxx=100.0, maxy=19.5)


def grid_dem():
    tiles = []
    for f in glob.glob(os.path.join(RAW, "dem", "*.tif")):
        t = rioxarray.open_rasterio(f, masked=True)
        t = t.rio.clip_box(BOX["minx"] - 0.05, BOX["miny"] - 0.05,
                           BOX["maxx"] + 0.05, BOX["maxy"] + 0.05)
        tiles.append(t)
    dem = merge_arrays(tiles).isel(band=0)
    # ลดความละเอียดหยาบก่อน (เร็วขึ้น) แล้ว interp ลงจุด grid พอดี
    dem = dem.coarsen(x=30, y=30, boundary="trim").mean()
    dem = dem.interp(x=LON, y=LAT, method="linear")
    # dem.y may be descending; reorder to match LAT (ascending)
    if dem.y.values[0] > dem.y.values[-1]:
        dem = dem.isel(y=slice(None, None, -1))
    # เติม NaN ที่ขอบ (interp เกินขอบ tile) ด้วยค่าเพื่อนบ้านใกล้สุด
    raw = dem.values
    n_nan = int(np.isnan(raw).sum())
    df = pd.DataFrame(raw)
    df = df.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
    arr = df.values.astype("float32")
    print(f"  DEM grid: {arr.shape}  elev {np.nanmin(arr):.0f}-{np.nanmax(arr):.0f} m"
          f"  (เติม NaN ขอบ {n_nan} ช่อง)")
    return arr


def grid_era5():
    f = glob.glob(os.path.join(RAW, "era5", "*.nc"))[0]
    ds = xr.open_dataset(f)
    # 3-hourly -> รายวัน (mean) ; valid_time -> time
    daily = ds.resample(valid_time="1D").mean()
    daily = daily.rename({"valid_time": "time"})
    # interp ลง master grid
    daily = daily.interp(latitude=LAT, longitude=LON, method="linear")
    daily = daily.sel(time=DATES, method="nearest")
    u = daily.u10.values.astype("float32")  # (time, lat, lon)
    v = daily.v10.values.astype("float32")
    print(f"  ERA5 grid: u10 {u.shape}  spd max {np.sqrt(u**2+v**2).max():.1f} m/s")
    return u, v


def grid_firms():
    f = sorted(glob.glob(os.path.join(RAW, "firms", "*.csv")))[-1]
    df = pd.read_csv(f)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    emis = np.zeros((len(DATES), len(LAT), len(LON)), dtype="float32")
    date_idx = {d.date(): i for i, d in enumerate(DATES)}
    n_used = 0
    for _, r in df.iterrows():
        di = date_idx.get(r["acq_date"].date())
        if di is None:
            continue
        ilat = int(round((r["latitude"] - LAT[0]) / STEP))
        ilon = int(round((r["longitude"] - LON[0]) / STEP))
        if 0 <= ilat < len(LAT) and 0 <= ilon < len(LON):
            emis[di, ilat, ilon] += r["frp"]
            n_used += 1
    print(f"  FIRMS grid: emission {emis.shape}  ใช้ {n_used}/{len(df)} จุด  "
          f"FRP รวม {emis.sum():.0f}")
    return emis


def main():
    os.makedirs(PROC, exist_ok=True)
    print("Stage 1 — Regrid ลง master grid 16x21 @ 0.1°")
    dem = grid_dem()
    u, v = grid_era5()
    emis = grid_firms()

    ds = xr.Dataset(
        data_vars=dict(
            dem=(["lat", "lon"], dem),
            u10=(["time", "lat", "lon"], u),
            v10=(["time", "lat", "lon"], v),
            emission=(["time", "lat", "lon"], emis),
        ),
        coords=dict(time=DATES, lat=LAT, lon=LON),
        attrs=dict(
            title="HazeNet master grid (Chiang Mai box)",
            grid_step=STEP, window="2023-03-19..31",
        ),
    )
    out = os.path.join(PROC, "grid.nc")
    ds.to_netcdf(out)
    print(f"\n[ok] -> {out}")
    print(ds)


if __name__ == "__main__":
    main()
