"""
Stage 5 — Prediction map: ใช้ XGBoost ทำนาย PM2.5 "ทั่วทั้งกล่อง" (ไม่ใช่แค่ 12 สถานี)
  สร้าง feature เดียวกัน (patch 3x3 + lat/lon/tidx) ให้ทุกช่อง grid -> ทำนาย
  ออก: แผนที่หลายวัน (panel) + animation GIF 13 วัน + จุดสถานีจริงทับไว้ตรวจ

รัน:  conda run -n hazenet --no-capture-output python src/plot_prediction.py
ออก:  figures/pm25_pred_map.png, figures/pm25_animation.gif
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
CHANNELS = ["u10", "v10", "dem", "emission"]


def predict_grid(model):
    cube = xr.open_zarr(os.path.join(PROC, "datacube.zarr"))
    X = cube.X.values                       # (T,C,lat,lon)
    T, C, H, W = X.shape
    LAT, LON = cube.lat.values, cube.lon.values
    Xpad = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")

    pred = np.zeros((T, H, W), dtype="float32")
    for t in range(T):
        feats = []
        for i in range(H):
            for j in range(W):
                patch = Xpad[t, :, i:i + 3, j:j + 3]   # (4,3,3)
                feats.append(np.concatenate([patch.flatten(),
                             [LAT[i], LON[j], t]]))
        p = model.predict(np.array(feats, dtype="float32"))
        pred[t] = p.reshape(H, W)
    pred = np.clip(pred, 0, None)
    return cube.time.values, LAT, LON, pred


def main():
    with open(os.path.join(ROOT, "models", "xgb_baseline.pkl"), "rb") as f:
        model = pickle.load(f)
    times, LAT, LON, pred = predict_grid(model)
    T = len(times)
    print(f"ทำนายทั่ว grid: {pred.shape}  PM2.5 {pred.min():.0f}-{pred.max():.0f} µg/m³")

    sta = pd.read_csv(os.path.join(PROC, "target_pm25.csv"))
    sta["date"] = pd.to_datetime(sta["date"])

    vmax = float(np.percentile(pred, 99))
    extent = [LON[0], LON[-1], LAT[0], LAT[-1]]

    # ---- panel: เลือก 6 วันกระจาย ----
    pick = np.linspace(0, T - 1, 6).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, t in zip(axes.flat, pick):
        im = ax.imshow(pred[t], origin="lower", extent=extent, cmap="YlOrRd",
                       vmin=0, vmax=vmax, aspect="auto")
        day = pd.Timestamp(times[t])
        d_sta = sta[sta["tidx"] == t]
        ax.scatter(d_sta["lon"], d_sta["lat"], c=d_sta["pm25"], cmap="YlOrRd",
                   vmin=0, vmax=vmax, s=80, edgecolors="black", linewidths=1.2)
        ax.set_title(day.strftime("%Y-%m-%d"), fontsize=11)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    fig.suptitle("HazeNet — predicted PM2.5 surface (XGBoost)\n"
                 "fill = grid prediction · circles = station observed",
                 fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.6, label="PM2.5 (µg/m³)")
    out1 = os.path.join(ROOT, "figures", "pm25_pred_map.png")
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] -> {out1}")

    # ---- animation 13 วัน ----
    figa, axa = plt.subplots(figsize=(7, 6))
    im = axa.imshow(pred[0], origin="lower", extent=extent, cmap="YlOrRd",
                    vmin=0, vmax=vmax, aspect="auto")
    sc = axa.scatter(sta[sta.tidx == 0]["lon"], sta[sta.tidx == 0]["lat"],
                     c=sta[sta.tidx == 0]["pm25"], cmap="YlOrRd",
                     vmin=0, vmax=vmax, s=90, edgecolors="black", linewidths=1.2)
    figa.colorbar(im, ax=axa, label="PM2.5 (µg/m³)")
    axa.set_xlabel("Longitude"); axa.set_ylabel("Latitude")
    ttl = axa.set_title("")

    def update(t):
        im.set_data(pred[t])
        d = sta[sta.tidx == t]
        sc.set_offsets(np.c_[d["lon"], d["lat"]])
        sc.set_array(d["pm25"].values)
        ttl.set_text(f"HazeNet predicted PM2.5 — {pd.Timestamp(times[t]):%Y-%m-%d}")
        return im, sc, ttl

    anim = FuncAnimation(figa, update, frames=T, interval=600, blit=False)
    out2 = os.path.join(ROOT, "figures", "pm25_animation.gif")
    anim.save(out2, writer=PillowWriter(fps=2))
    plt.close(figa)
    print(f"[ok] -> {out2}")


if __name__ == "__main__":
    main()
