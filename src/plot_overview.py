"""
วาดแผนที่ภาพรวมกล่องเชียงใหม่ ซ้อน 4 ชั้น:
  - พื้นหลัง = DEM (ความสูงภูมิประเทศ)
  - ลูกศร    = ลม 10m จาก ERA5 (1 ช่วงเวลา)
  - จุดแดง   = จุดไฟ FIRMS (ขนาดตาม FRP)
  - สามเหลี่ยม = สถานีวัด PM2.5 (สีตาม avg PM2.5)

รัน: conda run -n hazenet --no-capture-output python src/plot_overview.py
ออก: figures/overview_chiangmai.png
"""
import os
import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
BOX = dict(minx=98.0, miny=18.0, maxx=100.0, maxy=19.5)


def load_dem():
    tiles = []
    for f in glob.glob(os.path.join(RAW, "dem", "*.tif")):
        t = rioxarray.open_rasterio(f, masked=True)
        t = t.rio.clip_box(BOX["minx"], BOX["miny"], BOX["maxx"], BOX["maxy"])
        tiles.append(t)
    dem = merge_arrays(tiles).isel(band=0)
    dem = dem.coarsen(x=12, y=12, boundary="trim").mean()
    print(f"  DEM: {dem.shape}  elev {float(dem.min()):.0f}-{float(dem.max()):.0f} m")
    return dem


def load_wind():
    f = glob.glob(os.path.join(RAW, "era5", "*.nc"))[0]
    ds = xr.open_dataset(f)
    # ERA5 CDS uses 'valid_time' dimension
    tname = "valid_time" if "valid_time" in ds.dims else "time"
    spd = np.sqrt(ds.u10**2 + ds.v10**2).mean(dim=["latitude", "longitude"])
    ti = int(spd.argmax())
    t = ds.isel({tname: ti})
    ts = str(ds[tname].values[ti])[:16]
    print(f"  wind: timestamp {ts} (avg spd {float(spd[ti]):.1f} m/s)")
    return t.longitude.values, t.latitude.values, t.u10.values, t.v10.values


def load_fire():
    candidates = glob.glob(os.path.join(RAW, "firms", "*.csv"))
    if not candidates:
        print("  fire: ไม่พบไฟล์ FIRMS")
        return pd.DataFrame(columns=["longitude", "latitude", "frp"])
    f = sorted(candidates)[-1]
    df = pd.read_csv(f)
    print(f"  fire: {len(df)} จุด  FRP รวม={df['frp'].sum():.0f}  ({os.path.basename(f)})")
    return df


def load_stations():
    p = os.path.join(RAW, "pm25", "pm25_daily_2023-03-19_31.csv")
    if not os.path.exists(p):
        print("  stations: ไม่พบ pm25_daily — ข้ามสถานี")
        return pd.DataFrame(columns=["lon", "lat", "location", "pm25"])
    df = pd.read_csv(p)
    # mean PM2.5 ต่อสถานีสำหรับ coloring
    agg = df.groupby(["locationId", "location", "lat", "lon"]).agg(
        pm25=("pm25", "mean")).reset_index()
    print(f"  stations: {len(agg)} สถานี  avg PM2.5 {agg['pm25'].mean():.1f} ug/m3")
    return agg


def main():
    print("โหลดข้อมูล...")
    dem = load_dem()
    lon, lat, u, v = load_wind()
    fire = load_fire()
    sta = load_stations()

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # 1) DEM พื้นหลัง
    im = ax.pcolormesh(dem.x, dem.y, dem.values, cmap="terrain", shading="auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Elevation (m)")

    # 2) ลม (quiver)
    LON, LAT = np.meshgrid(lon, lat)
    ax.quiver(LON, LAT, u, v, color="black", scale=120, width=0.003,
              alpha=0.8, label="ERA5 wind 10m")

    # 3) จุดไฟ FIRMS
    if len(fire) > 0:
        s = 8 + (fire["frp"] / fire["frp"].max() * 120)
        ax.scatter(fire["longitude"], fire["latitude"], s=s, c="red",
                   alpha=0.5, edgecolors="darkred", linewidths=0.3,
                   label=f"FIRMS fire ({len(fire)} pts)")

    # 4) สถานี PM2.5 (สีตาม avg PM2.5)
    if len(sta) > 0:
        norm = plt.Normalize(sta["pm25"].min(), sta["pm25"].max())
        sc = ax.scatter(sta["lon"], sta["lat"], marker="^", s=180,
                        c=sta["pm25"], cmap="YlOrRd", norm=norm,
                        edgecolors="black", linewidths=1.0, zorder=5,
                        label=f"PM2.5 station ({len(sta)})")
        cb2 = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.09, location="right")
        cb2.set_label("Avg PM2.5 (µg/m³)")
        for _, r in sta.iterrows():
            short = r["location"].split(",")[0][:12]
            ax.annotate(short, (r["lon"], r["lat"]),
                        fontsize=6.5, xytext=(4, 4), textcoords="offset points",
                        color="navy")

    ax.set_xlim(BOX["minx"], BOX["maxx"])
    ax.set_ylim(BOX["miny"], BOX["maxy"])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        "HazeNet — Chiang Mai box: DEM + wind + fire + PM2.5 stations\n"
        "Severe haze event: 2023-03-19..31  (PM2.5 avg 93.8, max 420 µg/m³)"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")

    out = os.path.join(ROOT, "figures", "overview_chiangmai.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n[ok] บันทึกรูป -> {out}")


if __name__ == "__main__":
    main()
