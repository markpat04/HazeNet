"""
(ข) Animation ควัน/ฝุ่นเคลื่อนที่หลายวัน รอบช่วงวิกฤต
แต่ละเฟรม: ไฟ(FRP) พื้นหลัง + ลม + จุดสถานี PM2.5 ระบายสีตามค่าวันนั้น
Output: figures/smoke_animation.gif
Run: KMP_DUPLICATE_LIB_OK=TRUE conda run -n hazenet --no-capture-output python src/animate_smoke.py
"""
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import LAT, LON, H, W, PROC, ROOT

grid   = xr.open_dataset(os.path.join(PROC, "grid_m2.nc"))
times  = pd.DatetimeIndex(grid.time.values)
extent = [LON[0], LON[-1], LAT[0], LAT[-1]]
dem    = grid.dem.values
U      = grid.u10.values
V      = grid.v10.values
EM     = grid.emission.values

tgt = pd.read_csv(os.path.join(PROC, "target_pm25_m2.csv"))

# --- choose window: 18 days centered on crisis (max mean PM2.5, >=10 sta) ---
g  = tgt.groupby("tidx").agg(mean_pm=("pm25", "mean"), n=("pm25", "size"))
g  = g[g["n"] >= 10]
ci = int(g["mean_pm"].idxmax())
half = 9
idxs = [i for i in range(ci - half, ci + half + 1) if 0 <= i < len(times)]
# keep window within same burning season (avoid jumping across years)
yr = times[ci].year
idxs = [i for i in idxs if times[i].year == yr]
print(f"crisis {times[ci].date()}  window {times[idxs[0]].date()}..{times[idxs[-1]].date()} "
      f"({len(idxs)} frames)")

vmax_em = float(np.nanpercentile(EM[EM > 0], 99))

fig, ax = plt.subplots(figsize=(11, 11))

def draw(frame):
    ax.clear()
    di = idxs[frame]
    d  = times[di]
    ax.imshow(dem, origin="lower", extent=extent, cmap="Greys", alpha=0.3, aspect="auto")
    # smoke proxy = emission blurred (เพื่อให้เห็นการแผ่ — ป้ายชัดว่าเป็น proxy)
    smoke = gaussian_filter(EM[di], sigma=1.5)
    sm = np.where(smoke > 0.5, smoke, np.nan)
    ax.imshow(sm, origin="lower", extent=extent, cmap="hot",
              norm=LogNorm(vmin=1, vmax=vmax_em), alpha=0.7, aspect="auto")
    step = 6
    ax.quiver(LON[::step], LAT[::step], U[di, ::step, ::step], V[di, ::step, ::step],
              color="#3a86ff", scale=130, width=0.003, alpha=0.65)
    day = tgt[tgt["tidx"] == di]
    if len(day):
        ax.scatter(day["lon"], day["lat"], c=day["pm25"], cmap="turbo",
                   vmin=0, vmax=200, s=120, edgecolor="black", linewidth=0.7, zorder=5)
    ax.plot(98.98, 18.79, "*", color="cyan", ms=18, mec="black", zorder=6)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("longitude (E)"); ax.set_ylabel("latitude (N)")
    mean_pm = day["pm25"].mean() if len(day) else float("nan")
    ax.set_title(f"HazeNet — {d.date()}   |   mean PM2.5 = {mean_pm:.0f} µg/m³\n"
                 f"background = fire FRP (blurred to suggest smoke spread)   arrows = wind   dots = PM2.5 stations",
                 fontsize=12, fontweight="bold")

# static colorbar
sm0 = plt.cm.ScalarMappable(cmap="turbo",
        norm=plt.Normalize(vmin=0, vmax=200))
cb = fig.colorbar(sm0, ax=ax, shrink=0.75)
cb.set_label("Station PM2.5 (µg/m³)", fontsize=11)

anim = FuncAnimation(fig, draw, frames=len(idxs), interval=600)
out = os.path.join(ROOT, "figures", "smoke_animation.gif")
os.makedirs(os.path.dirname(out), exist_ok=True)
anim.save(out, writer=PillowWriter(fps=2))
print(f"[ok] -> {out}  ({len(idxs)} frames)")
