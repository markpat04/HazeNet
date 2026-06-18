"""
ดาวน์โหลดลม 10m (u10, v10) จาก Open-Meteo ERA5 archive
ฟรี ไม่ต้องมี API key ใช้ ERA5 reanalysis data

Strategy:
  - ดึง hourly wind ที่ coarse grid 1°x1° (12×11 = 132 จุด)
  - แต่ละคำขอครอบ 2019-02-01..2023-04-30 ทั้งหมด (~449 วัน = ~10,776 records)
  - เฉลี่ยเป็นรายวัน -> บันทึกเป็น NetCDF (time, lat_coarse, lon_coarse)
  - build_grid_m2.py จะ interp ลง 0.1° grid ให้เอง

Output: data/raw_m2/openmeteo/openmeteo_wind_2019_2023.nc  (~5MB)

Run: conda run -n hazenet --no-capture-output python src/download_openmeteo_m2.py
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import requests
import xarray as xr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import DATES, ROOT

OUT_DIR = os.path.join(ROOT, "data", "raw_m2", "openmeteo")
OUT_NC  = os.path.join(OUT_DIR, "openmeteo_wind_2019_2023.nc")

# Coarse 1° grid over SEA domain (12×11 = 132 API calls)
COARSE_LAT = np.arange(14.0, 26.0, 1.0)   # 12 values
COARSE_LON = np.arange(96.0, 107.0, 1.0)  # 11 values
API_BASE   = "https://archive-api.open-meteo.com/v1/archive"
# Full span of all burning seasons
DATE_FROM  = "2019-02-01"
DATE_TO    = "2023-04-30"


def fetch_point(lat: float, lon: float, retries: int = 3) -> dict:
    """ดึง hourly wind speed + direction จาก Open-Meteo สำหรับ 1 จุด"""
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      DATE_FROM,
        "end_date":        DATE_TO,
        "hourly":          "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "models":          "era5",
        "format":          "json",
    }
    for attempt in range(retries):
        try:
            r = requests.get(API_BASE, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(30)
                continue
            print(f"  [HTTP {r.status_code}] lat={lat} lon={lon}")
            return {}
        except Exception as e:
            print(f"  [ERR ] attempt {attempt+1} lat={lat} lon={lon}: {e}")
            time.sleep(5)
    return {}


def hourly_to_daily_uv(times: list, speeds: list, dirs: list) -> tuple:
    """
    Convert hourly (speed, direction) -> daily mean (u10, v10)
    Average u/v vectors per day, not speed/direction directly.
    """
    df = pd.DataFrame({
        "time":  pd.to_datetime(times),
        "speed": speeds,
        "dir":   dirs,
    }).dropna()
    df["u"] = -df["speed"] * np.sin(np.radians(df["dir"]))
    df["v"] = -df["speed"] * np.cos(np.radians(df["dir"]))
    df["date"] = df["time"].dt.normalize()
    daily = df.groupby("date")[["u", "v"]].mean()
    return daily


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(OUT_NC) and os.path.getsize(OUT_NC) > 10_000:
        mb = os.path.getsize(OUT_NC) / 1e6
        print(f"[skip] {OUT_NC}  ({mb:.1f} MB)  (ลบไฟล์นี้ถ้าต้องการ re-download)")
        return

    n_pts = len(COARSE_LAT) * len(COARSE_LON)
    print(f"Open-Meteo ERA5: {n_pts} points  ({DATE_FROM}..{DATE_TO})")
    print(f"  Grid: lat {COARSE_LAT[0]:.0f}..{COARSE_LAT[-1]:.0f}°N  "
          f"lon {COARSE_LON[0]:.0f}..{COARSE_LON[-1]:.0f}°E  (1° spacing)")

    # Determine target dates (burning seasons only)
    target_dates = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in DATES])

    # Storage: (n_dates, n_lat, n_lon)
    T = len(target_dates)
    U = np.full((T, len(COARSE_LAT), len(COARSE_LON)), np.nan, dtype="float32")
    V = np.full_like(U, np.nan)

    date_idx = {d: i for i, d in enumerate(target_dates)}
    n_ok = 0

    for ilat, lat in enumerate(COARSE_LAT):
        for ilon, lon in enumerate(COARSE_LON):
            print(f"  [{ilat*len(COARSE_LON)+ilon+1:3d}/{n_pts}]  "
                  f"lat={lat:.0f}  lon={lon:.0f} ...", end=" ", flush=True)
            data = fetch_point(lat, lon)
            h = data.get("hourly", {})
            times  = h.get("time", [])
            speeds = h.get("wind_speed_10m", [])
            dirs   = h.get("wind_direction_10m", [])
            if not times:
                print("empty")
                continue

            daily = hourly_to_daily_uv(times, speeds, dirs)
            n_matched = 0
            for date, row in daily.iterrows():
                idx = date_idx.get(date)
                if idx is not None:
                    U[idx, ilat, ilon] = row["u"]
                    V[idx, ilat, ilon] = row["v"]
                    n_matched += 1
            print(f"{n_matched} days")
            n_ok += 1
            time.sleep(0.4)   # be polite to the API

    print(f"\nFetched {n_ok}/{n_pts} points successfully")

    # Fill remaining NaN with spatial mean (ocean grid cells near coast)
    for t in range(T):
        for arr in (U, V):
            layer = arr[t]
            if np.isnan(layer).any() and not np.isnan(layer).all():
                mean_val = np.nanmean(layer)
                layer[np.isnan(layer)] = mean_val

    # Save as NetCDF
    ds = xr.Dataset(
        data_vars=dict(
            u10=(["time", "lat", "lon"], U),
            v10=(["time", "lat", "lon"], V),
        ),
        coords=dict(
            time=target_dates,
            lat=COARSE_LAT.astype("float32"),
            lon=COARSE_LON.astype("float32"),
        ),
        attrs=dict(
            source="Open-Meteo ERA5 archive",
            coarse_resolution="1 degree",
            note="Interpolate to 0.1 degree in build_grid_m2.py",
        ),
    )
    ds.to_netcdf(OUT_NC)
    mb = os.path.getsize(OUT_NC) / 1e6
    print(f"\n[ok] {OUT_NC}  ({mb:.1f} MB)")
    print(f"     u10 range: {np.nanmin(U):.1f}..{np.nanmax(U):.1f} m/s")
    print(f"     v10 range: {np.nanmin(V):.1f}..{np.nanmax(V):.1f} m/s")


if __name__ == "__main__":
    main()
