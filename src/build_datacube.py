"""
Stage 2 — Datacube: รวม grid.nc เป็น "กล่องข้อมูล" สำหรับโมเดล + จัด target

  X: channels (time, channel, lat, lon)  channel = [u10, v10, dem, emission]
     dem เป็น static -> broadcast ซ้ำทุก time step
  y: PM2.5 รายสถานี รายวัน -> map ลงช่อง grid (ilat, ilon) เพื่อใช้เทรน

รัน:  conda run -n hazenet --no-capture-output python src/build_datacube.py
ออก:  data/processed/datacube.zarr/   (X)
      data/processed/target_pm25.csv  (y + grid index)
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw")
CHANNELS = ["u10", "v10", "dem", "emission"]


def main():
    grid = xr.open_dataset(os.path.join(PROC, "grid.nc"))
    nt, nlat, nlon = grid.sizes["time"], grid.sizes["lat"], grid.sizes["lon"]
    LAT = grid.lat.values
    LON = grid.lon.values

    # ---- X: stack channels (time, channel, lat, lon) ----
    dem_b = np.broadcast_to(grid.dem.values, (nt, nlat, nlon))
    stack = np.stack([grid.u10.values, grid.v10.values, dem_b,
                      grid.emission.values], axis=1).astype("float32")
    cube = xr.DataArray(
        stack, dims=["time", "channel", "lat", "lon"],
        coords=dict(time=grid.time, channel=CHANNELS, lat=LAT, lon=LON),
        name="X",
    )
    cube = cube.to_dataset()
    out_zarr = os.path.join(PROC, "datacube.zarr")
    if os.path.exists(out_zarr):
        import shutil
        shutil.rmtree(out_zarr)
    cube.to_zarr(out_zarr, mode="w")
    print(f"[ok] datacube -> {out_zarr}")
    print(f"     shape (time,channel,lat,lon) = {stack.shape}  channels={CHANNELS}")

    # ---- y: PM2.5 station target mapped to grid cells ----
    pm = pd.read_csv(os.path.join(RAW, "pm25", "pm25_daily_2023-03-19_31.csv"))
    pm["date"] = pd.to_datetime(pm["date"])
    pm["ilat"] = np.round((pm["lat"] - LAT[0]) / 0.1).astype(int)
    pm["ilon"] = np.round((pm["lon"] - LON[0]) / 0.1).astype(int)
    pm["tidx"] = pm["date"].map({pd.Timestamp(d): i
                                 for i, d in enumerate(grid.time.values)})
    # คัดเฉพาะที่ตกอยู่ในกล่อง + มี time index
    ok = (pm["ilat"].between(0, nlat - 1) & pm["ilon"].between(0, nlon - 1)
          & pm["tidx"].notna())
    pm = pm[ok].copy()
    pm["tidx"] = pm["tidx"].astype(int)
    out_csv = os.path.join(PROC, "target_pm25.csv")
    pm[["date", "tidx", "locationId", "location", "lat", "lon",
        "ilat", "ilon", "pm25"]].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] target  -> {out_csv}")
    print(f"     {len(pm)} แถว  {pm.locationId.nunique()} สถานี  "
          f"PM2.5 {pm.pm25.min():.0f}-{pm.pm25.max():.0f} µg/m³")


if __name__ == "__main__":
    main()
