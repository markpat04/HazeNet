"""
ดึง feature สำหรับเทรน: ต่อ 1 obs (สถานี x วัน) เอา patch 3x3 ของทุก channel
รอบช่อง grid ที่สถานีตกอยู่ -> 4 channel x 9 ช่อง = 36 + (lat, lon, tidx) = 39 features

ใช้ร่วมโดย train_baseline.py (XGBoost) และ train_mlp.py (PyTorch)
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
CHANNELS = ["u10", "v10", "dem", "emission"]


def load_xy():
    cube = xr.open_zarr(os.path.join(PROC, "datacube.zarr"))
    X = cube.X.values  # (time, channel, lat, lon)
    # pad ขอบ +1 เพื่อหยิบ patch 3x3 ได้ทุกช่อง (mode edge)
    Xpad = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")

    tgt = pd.read_csv(os.path.join(PROC, "target_pm25.csv"))
    feats, ys, meta = [], [], []
    for _, r in tgt.iterrows():
        t, il, jl = int(r["tidx"]), int(r["ilat"]), int(r["ilon"])
        patch = Xpad[t, :, il:il + 3, jl:jl + 3]   # (4,3,3)
        f = np.concatenate([patch.flatten(),
                            [r["lat"], r["lon"], t]]).astype("float32")
        feats.append(f)
        ys.append(np.float32(r["pm25"]))
        meta.append((t, r["locationId"], r["date"], r["lat"], r["lon"]))
    X_feat = np.array(feats, dtype="float32")
    y = np.array(ys, dtype="float32")
    meta = pd.DataFrame(meta, columns=["tidx", "locationId", "date", "lat", "lon"])

    # ชื่อ feature (ไว้ดู importance)
    names = [f"{c}_{p}" for c in CHANNELS for p in
             ["nw", "n", "ne", "w", "c", "e", "sw", "s", "se"]]
    names += ["lat", "lon", "tidx"]
    return X_feat, y, meta, names


def temporal_split(meta, n_test_days=3):
    """แบ่ง train/test ตามเวลา: วันท้าย ๆ เป็น test (กันรั่วอนาคต)"""
    max_t = meta["tidx"].max()
    is_test = meta["tidx"] > (max_t - n_test_days)
    return ~is_test.values, is_test.values
