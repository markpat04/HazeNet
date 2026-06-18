"""
ดาวน์โหลดตัวแปรอุตุฯ เพิ่มจาก Open-Meteo ERA5 archive (ฟรี ไม่ต้องมี key)

เพิ่มจากเดิม (ลม) → เพิ่ม 3 ตัวที่ควบคุม "การสะสมฝุ่น":
  - precipitation        ฝน (ชะล้างฝุ่น) → daily SUM
  - relative_humidity_2m ความชื้น (โตของละออง) → daily MEAN
  - temperature_2m       อุณหภูมิ (เสถียรภาพอากาศ) → daily MEAN
  + ลม u10/v10 (เหมือนเดิม) → daily MEAN

เหตุผล: LOYO เผยว่าโมเดลจับ "ระดับฝุ่นต่อปี" ไม่ได้ เพราะความสัมพันธ์
ไฟ→ฝุ่น ขึ้นกับสภาพอากาศที่ยังไม่ได้ใส่เข้าโมเดล

Output: data/raw_m2/openmeteo/openmeteo_met_2019_2023.nc

Run: conda run -n hazenet --no-capture-output python src/download_openmeteo_met_m2.py
"""
import os, sys, time
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
OUT_NC  = os.path.join(OUT_DIR, "openmeteo_met_2019_2023.nc")

COARSE_LAT = np.arange(14.0, 26.0, 1.0)   # 12
COARSE_LON = np.arange(96.0, 107.0, 1.0)  # 11
API_BASE   = "https://archive-api.open-meteo.com/v1/archive"
DATE_FROM  = "2019-02-01"
DATE_TO    = "2023-04-30"

HOURLY = ("wind_speed_10m,wind_direction_10m,precipitation,"
          "relative_humidity_2m,temperature_2m")


def fetch_point(lat, lon, retries=3):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": DATE_FROM, "end_date": DATE_TO,
        "hourly": HOURLY, "wind_speed_unit": "ms",
        "models": "era5", "format": "json",
    }
    for attempt in range(retries):
        try:
            r = requests.get(API_BASE, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(30); continue
            print(f"  [HTTP {r.status_code}] lat={lat} lon={lon}")
            return {}
        except Exception as e:
            print(f"  [ERR ] attempt {attempt+1} lat={lat} lon={lon}: {e}")
            time.sleep(5)
    return {}


def to_daily(h):
    """hourly dict -> daily DataFrame [u, v, precip(sum), rh(mean), temp(mean)]"""
    df = pd.DataFrame({
        "time":  pd.to_datetime(h["time"]),
        "speed": h.get("wind_speed_10m"),
        "dir":   h.get("wind_direction_10m"),
        "precip": h.get("precipitation"),
        "rh":    h.get("relative_humidity_2m"),
        "temp":  h.get("temperature_2m"),
    })
    df["u"] = -df["speed"] * np.sin(np.radians(df["dir"]))
    df["v"] = -df["speed"] * np.cos(np.radians(df["dir"]))
    df["date"] = df["time"].dt.normalize()
    g = df.groupby("date")
    daily = pd.DataFrame({
        "u":      g["u"].mean(),
        "v":      g["v"].mean(),
        "precip": g["precip"].sum(),     # ฝนรวมต่อวัน
        "rh":     g["rh"].mean(),
        "temp":   g["temp"].mean(),
    })
    return daily


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_NC) and os.path.getsize(OUT_NC) > 10_000:
        print(f"[skip] {OUT_NC} (ลบถ้าต้องการ re-download)")
        return

    target_dates = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in DATES])
    T = len(target_dates)
    nL, nO = len(COARSE_LAT), len(COARSE_LON)
    arrs = {k: np.full((T, nL, nO), np.nan, "float32")
            for k in ["u10", "v10", "precip", "rh", "temp"]}
    didx = {d: i for i, d in enumerate(target_dates)}
    n_pts = nL * nO
    print(f"Open-Meteo MET: {n_pts} points  ({DATE_FROM}..{DATE_TO})  vars={HOURLY}")

    n_ok = 0
    for ila, la in enumerate(COARSE_LAT):
        for ilo, lo in enumerate(COARSE_LON):
            k = ila * nO + ilo + 1
            print(f"  [{k:3d}/{n_pts}] lat={la:.0f} lon={lo:.0f} ...", end=" ", flush=True)
            data = fetch_point(la, lo)
            h = data.get("hourly", {})
            if not h.get("time"):
                print("empty"); continue
            daily = to_daily(h)
            nm = 0
            for date, row in daily.iterrows():
                i = didx.get(date)
                if i is not None:
                    arrs["u10"][i, ila, ilo]    = row["u"]
                    arrs["v10"][i, ila, ilo]    = row["v"]
                    arrs["precip"][i, ila, ilo] = row["precip"]
                    arrs["rh"][i, ila, ilo]     = row["rh"]
                    arrs["temp"][i, ila, ilo]   = row["temp"]
                    nm += 1
            print(f"{nm} days"); n_ok += 1
            time.sleep(0.4)

    print(f"\nFetched {n_ok}/{n_pts} points")

    # fill spatial NaN per timestep
    for k in arrs:
        for t in range(T):
            layer = arrs[k][t]
            if np.isnan(layer).any() and not np.isnan(layer).all():
                layer[np.isnan(layer)] = np.nanmean(layer)

    ds = xr.Dataset(
        {k: (["time", "lat", "lon"], v) for k, v in arrs.items()},
        coords=dict(time=target_dates,
                    lat=COARSE_LAT.astype("float32"),
                    lon=COARSE_LON.astype("float32")),
        attrs=dict(source="Open-Meteo ERA5 archive",
                   note="precip=daily sum, others=daily mean; interp to 0.1deg in build_grid"),
    )
    ds.to_netcdf(OUT_NC)
    print(f"\n[ok] {OUT_NC}  ({os.path.getsize(OUT_NC)/1e6:.1f} MB)")
    for k in arrs:
        a = arrs[k]
        print(f"     {k:7} {np.nanmin(a):8.1f} .. {np.nanmax(a):8.1f}")


if __name__ == "__main__":
    main()
