"""
(ก) PM2.5 จริงซ้อนบนแผนที่ "วันวิกฤต" (วันที่ค่าฝุ่นเฉลี่ยสถานีสูงสุด)
แสดง: ไฟ(FRP) เป็นพื้นหลัง + ลม + จุดสถานีระบายสีตามค่า PM2.5 จริง
Output: figures/pm25_crisis_overlay.png
Run: KMP_DUPLICATE_LIB_OK=TRUE conda run -n hazenet --no-capture-output python src/plot_pm25_crisis.py
"""
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import LAT, LON, H, W, PROC, ROOT

grid  = xr.open_dataset(os.path.join(PROC, "grid_m2.nc"))
times = pd.DatetimeIndex(grid.time.values)
extent = [LON[0], LON[-1], LAT[0], LAT[-1]]

tgt = pd.read_csv(os.path.join(PROC, "target_pm25_m2.csv"))
tgt["date"] = pd.to_datetime(tgt["date"])

# --- find crisis day = highest mean station PM2.5 (require >=10 stations) ---
g = tgt.groupby("tidx").agg(mean_pm=("pm25", "mean"), n=("pm25", "size"))
g = g[g["n"] >= 10]
di = int(g["mean_pm"].idxmax())
worst = times[di]
day = tgt[tgt["tidx"] == di]
print(f"crisis day = {worst.date()}  mean PM2.5={g.loc[di,'mean_pm']:.0f}  "
      f"max={day['pm25'].max():.0f}  n_sta={len(day)}")

dem    = grid.dem.values
u, v   = grid.u10.values[di], grid.v10.values[di]
emis_d = grid.emission.values[di]

fig, ax = plt.subplots(figsize=(12, 12))

# terrain (faint) + fire (hot)
ax.imshow(dem, origin="lower", extent=extent, cmap="Greys", alpha=0.35, aspect="auto")
em = np.where(emis_d > 0, emis_d, np.nan)
ax.imshow(em, origin="lower", extent=extent, cmap="autumn",
          norm=LogNorm(vmin=1, vmax=np.nanmax(emis_d)), alpha=0.75, aspect="auto")

# wind
step = 5
ax.quiver(LON[::step], LAT[::step], u[::step, ::step], v[::step, ::step],
          color="#3070ff", scale=130, width=0.0028, alpha=0.7)

# PM2.5 stations
sc = ax.scatter(day["lon"], day["lat"], c=day["pm25"], cmap="turbo",
                vmin=0, vmax=200, s=160, edgecolor="black", linewidth=0.8, zorder=5)
cb = plt.colorbar(sc, ax=ax, shrink=0.75)
cb.set_label("Observed station PM2.5 (µg/m³)", fontsize=12)

# guides
ax.plot(98.98, 18.79, "*", color="cyan", ms=20, mec="black", zorder=6)
ax.text(99.15, 18.79, "Chiang Mai", color="cyan", fontsize=11, fontweight="bold")
ax.text(96.4, 23.0, "MYANMAR", color="black", fontsize=10, fontweight="bold")
ax.text(103.0, 21.5, "LAOS", color="black", fontsize=10, fontweight="bold")
ax.text(98.4, 15.5, "N. THAILAND", color="black", fontsize=10, fontweight="bold")

ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
ax.set_xlabel("longitude (E)"); ax.set_ylabel("latitude (N)")
ax.set_title(f"HAZE CRISIS DAY {worst.date()}  |  mean PM2.5 = {g.loc[di,'mean_pm']:.0f}, "
             f"max = {day['pm25'].max():.0f} µg/m³\n"
             f"orange background = fire (FRP)   blue arrows = wind   dots = stations colored by observed PM2.5",
             fontsize=13, fontweight="bold")

out = os.path.join(ROOT, "figures", "pm25_crisis_overlay.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"[ok] -> {out}")
