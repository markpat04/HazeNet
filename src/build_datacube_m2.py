"""
Stage 2 M2 — Datacube: รวม grid_m2.nc เป็น datacube + target PM2.5

X: (T, channel=4, H, W)   channel = [u10, v10, dem, emission]
y: PM2.5 รายสถานี รายวัน  saved as target_pm25_m2.csv

ทำ temporal mask ด้วย: บันทึก train_mask / test_mask
  train: 2019-2022, test: 2023

Run: conda run -n hazenet --no-capture-output python src/build_datacube_m2.py
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import xarray as xr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import LAT, LON, YEARS, TEST_YEAR, ROOT, RAW, PROC

def main():
    os.makedirs(PROC, exist_ok=True)

    grid = xr.open_dataset(os.path.join(PROC, "grid_m2.nc"))
    nt   = grid.sizes["time"]
    nlat = grid.sizes["lat"]
    nlon = grid.sizes["lon"]
    times = pd.DatetimeIndex(grid.time.values)

    print(f"Grid: time={nt}  lat={nlat}  lon={nlon}")

    # ── X: stack channels ──────────────────────────────────────────────
    # met channels first (encoder input), emission LAST (used by K@E)
    dem_b   = np.broadcast_to(grid.dem.values, (nt, nlat, nlon))
    met_vars   = [("u10", grid.u10.values), ("v10", grid.v10.values),
                  ("dem", dem_b)]
    for v in ["precip", "rh", "temp"]:           # add if present in grid
        if v in grid:
            met_vars.append((v, grid[v].values))
    CHANNELS = [name for name, _ in met_vars] + ["emission"]
    layers   = [arr for _, arr in met_vars] + [grid.emission.values]
    stack    = np.stack(layers, axis=1).astype("float32")
    print(f"Channels ({len(CHANNELS)}): {CHANNELS}")

    cube = xr.DataArray(
        stack, dims=["time", "channel", "lat", "lon"],
        coords=dict(time=times, channel=CHANNELS, lat=LAT, lon=LON),
        name="X",
    )
    cube_ds = cube.to_dataset()

    # เพิ่ม mask: train vs test
    test_mask  = np.array([t.year == TEST_YEAR  for t in times])
    train_mask = ~test_mask
    cube_ds["train_mask"] = xr.DataArray(train_mask, dims=["time"])
    cube_ds["test_mask"]  = xr.DataArray(test_mask,  dims=["time"])

    out_zarr = os.path.join(PROC, "datacube_m2.zarr")
    if os.path.exists(out_zarr):
        import shutil; shutil.rmtree(out_zarr)
    cube_ds.to_zarr(out_zarr, mode="w")
    print(f"[ok] datacube -> {out_zarr}  shape {stack.shape}")
    print(f"     train days: {train_mask.sum()}  test days: {test_mask.sum()}")

    # ── y: PM2.5 station targets ───────────────────────────────────────
    pm_files = sorted(glob.glob(os.path.join(RAW, "pm25", "pm25_*.csv")))
    if not pm_files:
        raise FileNotFoundError(f"ไม่พบ pm25 csv ใน {RAW}/pm25/  "
                                "รัน download_pm25_m2.py ก่อน")

    frames = [pd.read_csv(f) for f in pm_files]
    pm = pd.concat(frames, ignore_index=True)
    pm["date"] = pd.to_datetime(pm["date"])
    pm = pm.drop_duplicates(["date", "locationId"])

    # map ลง grid index
    pm["ilat"] = np.round((pm["lat"] - LAT[0]) / 0.1).astype(int)
    pm["ilon"] = np.round((pm["lon"] - LON[0]) / 0.1).astype(int)
    t_map = {pd.Timestamp(t): i for i, t in enumerate(times)}
    pm["tidx"] = pm["date"].map(t_map)

    ok = (pm["ilat"].between(0, nlat - 1) &
          pm["ilon"].between(0, nlon - 1) &
          pm["tidx"].notna())
    pm = pm[ok].copy()
    pm["tidx"] = pm["tidx"].astype(int)

    out_csv = os.path.join(PROC, "target_pm25_m2.csv")
    pm[["date", "tidx", "locationId", "location", "lat", "lon",
        "ilat", "ilon", "pm25"]].to_csv(out_csv, index=False, encoding="utf-8-sig")

    n_sta  = pm["locationId"].nunique()
    n_rows = len(pm)
    print(f"[ok] target  -> {out_csv}")
    print(f"     {n_rows} rows  {n_sta} stations  "
          f"PM2.5 {pm.pm25.min():.0f}-{pm.pm25.max():.0f} µg/m³")
    print(f"     stations by country (lat proxy):")
    thai_mask = pm["lat"] >= 14.5
    print(f"       Thailand (approx lat>=14.5): "
          f"{pm[thai_mask]['locationId'].nunique()} stations")
    print(f"       Other (Myanmar/Laos): "
          f"{pm[~thai_mask]['locationId'].nunique()} stations")


if __name__ == "__main__":
    main()
