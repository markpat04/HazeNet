"""
วาดภาพรวม datacube M2 (111x101 @ 0.1deg) ให้เห็นว่าครอบคลุมแค่ไหน + หน้าตาข้อมูลจริง
Output: figures/datacube_m2_overview.png
Run: KMP_DUPLICATE_LIB_OK=TRUE conda run -n hazenet --no-capture-output python src/plot_datacube_overview.py
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
from config_m2 import LAT, LON, H, W, PROC, ROOT, YEARS

grid = xr.open_dataset(os.path.join(PROC, "grid_m2.nc"))
times = pd.DatetimeIndex(grid.time.values)
extent = [LON[0], LON[-1], LAT[0], LAT[-1]]   # left,right,bottom,top

# --- pick worst fire day ---
emis = grid.emission.values                    # (T,H,W)
day_frp = emis.sum(axis=(1, 2))
di = int(np.argmax(day_frp))
worst = times[di]
print(f"Grid {H}x{W}  T={len(times)}  worst fire day={worst.date()} FRP={day_frp[di]:.0f}")

dem = grid.dem.values
u = grid.u10.values[di]; v = grid.v10.values[di]
emis_d = emis[di]

# --- PM2.5 stations ---
tgt = pd.read_csv(os.path.join(PROC, "target_pm25_m2.csv"))
sta = tgt.groupby("locationId").first().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(15, 14))
fig.suptitle(f"HazeNet datacube M2  —  SEA grid {H}x{W} @ 0.1 deg (~11 km)\n"
             f"coverage {LAT[0]:.0f}-{LAT[-1]:.0f} N, {LON[0]:.0f}-{LON[-1]:.0f} E  |  "
             f"Feb-Apr {YEARS[0]}-{YEARS[-1]} ({len(times)} days)",
             fontsize=15, fontweight="bold")

# approx country reference lines (visual guide only)
def country_guides(ax):
    ax.axhline(20.4, color="white", lw=0.6, ls=":", alpha=0.5)
    ax.text(96.4, 23.0, "MYANMAR", color="white", fontsize=9, alpha=0.8, fontweight="bold")
    ax.text(103.0, 21.5, "LAOS", color="white", fontsize=9, alpha=0.8, fontweight="bold")
    ax.text(98.4, 16.5, "N. THAILAND", color="white", fontsize=9, alpha=0.8, fontweight="bold")
    ax.plot(98.98, 18.79, "*", color="cyan", ms=16, mec="black")
    ax.text(99.2, 18.79, "Chiang Mai", color="cyan", fontsize=9, fontweight="bold")

# (1) DEM
ax = axes[0, 0]
im = ax.imshow(dem, origin="lower", extent=extent, cmap="terrain", aspect="auto")
ax.set_title(f"(1) Terrain / DEM  —  {np.nanmin(dem):.0f} to {np.nanmax(dem):.0f} m", fontsize=12)
plt.colorbar(im, ax=ax, label="elevation (m)", shrink=0.8)
country_guides(ax)

# (2) Emission on worst day
ax = axes[0, 1]
em_plot = np.where(emis_d > 0, emis_d, np.nan)
im = ax.imshow(em_plot, origin="lower", extent=extent, cmap="hot",
               norm=LogNorm(vmin=1, vmax=np.nanmax(emis_d)), aspect="auto")
ax.set_facecolor("#202020")
ax.set_title(f"(2) Fire emission (FRP)  —  worst day {worst.date()}", fontsize=12)
plt.colorbar(im, ax=ax, label="FRP per cell (MW, log)", shrink=0.8)
country_guides(ax)

# (3) Wind field
ax = axes[1, 0]
spd = np.sqrt(u**2 + v**2)
im = ax.imshow(spd, origin="lower", extent=extent, cmap="viridis", aspect="auto")
step = 4
ax.quiver(LON[::step], LAT[::step], u[::step, ::step], v[::step, ::step],
          color="white", scale=120, width=0.003)
ax.set_title(f"(3) Wind u/v  —  {worst.date()}  (max {spd.max():.0f} m/s)", fontsize=12)
plt.colorbar(im, ax=ax, label="wind speed (m/s)", shrink=0.8)
country_guides(ax)

# (4) Grid + PM2.5 stations
ax = axes[1, 1]
ax.imshow(dem, origin="lower", extent=extent, cmap="Greys", alpha=0.5, aspect="auto")
# draw grid lines every 1 deg to show the 0.1 cells density
for x in np.arange(LON[0], LON[-1] + 0.01, 1.0):
    ax.axvline(x, color="gray", lw=0.3, alpha=0.4)
for y in np.arange(LAT[0], LAT[-1] + 0.01, 1.0):
    ax.axhline(y, color="gray", lw=0.3, alpha=0.4)
th = sta[sta["lat"] >= 14.5]
ax.scatter(sta["lon"], sta["lat"], c="red", s=40, edgecolor="black",
           zorder=5, label=f"PM2.5 stations (n={len(sta)})")
ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
ax.set_title(f"(4) Grid coverage + {len(sta)} PM2.5 stations\n"
             f"each big square = 1 deg = 10x10 model cells", fontsize=12)
ax.legend(loc="upper right")
country_guides(ax)

for ax in axes.flat:
    ax.set_xlabel("longitude (E)"); ax.set_ylabel("latitude (N)")

out = os.path.join(ROOT, "figures", "datacube_m2_overview.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"[ok] -> {out}")
