"""
Attribution backend — reconstruct CLNO จาก clno_m2.pt แล้วรันรายวัน
เพื่อตอบ "ฝุ่นที่สถานี s วันนี้มาจากช่องไหน/ทิศไหน กี่%"

ใช้ clno_m2.pt (46MB) + datacube (lazy) — ไม่โหลด artifacts 397MB
cache ทุกอย่างใน global หลังโหลดครั้งแรก.
"""
import os, math, threading

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
SRC   = os.path.join(ROOT, "src")
PROC  = os.path.join(ROOT, "data", "processed_m2")
MODELS= os.path.join(ROOT, "models")

_LOCK  = threading.Lock()
_STATE = {}   # cache


def _load():
    """โหลด ckpt + model + cube (lazy) ครั้งเดียว."""
    if _STATE.get("ready"):
        return _STATE
    with _LOCK:
        if _STATE.get("ready"):
            return _STATE
        import sys, json
        import numpy as np, pandas as pd, torch, xarray as xr
        sys.path.insert(0, SRC)
        from train_operator_m2 import CLNOLowRank   # อยู่ไฟล์เดียวกับ training

        ck = torch.load(os.path.join(MODELS, "clno_m2.pt"),
                        map_location="cpu", weights_only=False)
        H, W, S = ck["H"], ck["W"], ck["S"]
        model = CLNOLowRank(H=H, W=W, n_stations=S,
                            hidden=ck["hidden"], rank=ck["rank"],
                            in_ch=ck.get("in_ch", 6),
                            dropout=ck.get("dropout", 0.1))
        model.load_state_dict(ck["state_dict"])
        model.eval()

        mu  = np.array(ck["met_mu"],  dtype="float32")   # (1,6,1,1)
        std = np.array(ck["met_std"], dtype="float32")
        e_max    = float(ck["meta"]["e_max"])
        pm25_max = float(ck["meta"]["pm25_max"])
        in_ch    = ck.get("in_ch", 6)

        # stations: dict-of-dicts {col:{idx:val}} → list ordered by idx
        st = ck["stations"]
        def col(name, i):
            c = st[name]
            return c.get(i, c.get(str(i))) if isinstance(c, dict) else c[i]
        stations = [{"idx": i,
                     "name": str(col("location", i)),
                     "lat": float(col("lat", i)),
                     "lon": float(col("lon", i)),
                     "ilat": int(col("ilat", i)),
                     "ilon": int(col("ilon", i)),
                     "locationId": int(col("locationId", i))}
                    for i in range(S)]

        cube  = xr.open_zarr(os.path.join(PROC, "datacube_m2.zarr"))
        times = pd.DatetimeIndex(cube.time.values)
        LAT   = cube.lat.values.astype(float)
        LON   = cube.lon.values.astype(float)

        # observed PM2.5 lookup (tidx, locationId) -> value  + daily mean
        tgt = pd.read_csv(os.path.join(PROC, "target_pm25_m2.csv"))
        obs = {(int(r.tidx), int(r.locationId)): float(r.pm25)
               for r in tgt.itertuples()}
        daily_mean = tgt.groupby("tidx")["pm25"].mean().to_dict()

        _STATE.update(dict(ready=True, model=model, mu=mu, std=std,
                           e_max=e_max, pm25_max=pm25_max, in_ch=in_ch,
                           stations=stations, cube=cube, times=times,
                           LAT=LAT, LON=LON, H=H, W=W, S=S,
                           obs=obs, daily_mean=daily_mean, np=np, torch=torch))
    return _STATE


def meta():
    s = _load()
    # default day = วันที่ฝุ่นเฉลี่ยสูงสุด (มีข้อมูล)
    dm = s["daily_mean"]
    best = max(dm, key=dm.get) if dm else 0
    days = [{"idx": i, "date": d.strftime("%Y-%m-%d"),
             "mean_pm": round(float(s["daily_mean"].get(i, float("nan"))), 0)
                        if i in s["daily_mean"] else None}
            for i, d in enumerate(s["times"])]
    return {"stations": s["stations"], "days": days,
            "default_day": int(best), "H": s["H"], "W": s["W"]}


def attribution(day_idx, station_idx):
    s   = _load()
    np  = s["np"]; torch = s["torch"]
    di  = int(day_idx); si = int(station_idx)
    H, W = s["H"], s["W"]
    in_ch = s["in_ch"]

    x = s["cube"].X.isel(time=di).values.astype("float32")   # (C,H,W)
    met  = x[:in_ch][None]                                    # (1,6,H,W)
    emis = x[-1][None]                                        # (1,H,W)
    met_n  = (met - s["mu"]) / s["std"]
    emis_n = emis / s["e_max"]

    with torch.no_grad():
        mt = torch.tensor(met_n, dtype=torch.float32)
        et = torch.tensor(emis_n, dtype=torch.float32)
        pred, K, b = s["model"](mt, et)
        contrib, frac = s["model"].attribution(K, et)        # (1,S,H,W)

    pred_pm = float(pred[0, si].item()) * s["pm25_max"]
    cmap = contrib[0, si].numpy()                            # (H,W) สัดส่วน (ก่อน normalize)
    total = float(cmap.sum())
    fmap = (cmap / total) if total > 1e-9 else cmap * 0      # fraction 0..1

    st  = s["stations"][si]
    slat, slon = st["lat"], st["lon"]
    LAT, LON = s["LAT"], s["LON"]

    # top contributing cells
    flat = fmap.ravel()
    order = np.argsort(flat)[::-1][:12]
    top = []
    for k in order:
        if flat[k] <= 0: continue
        i, j = divmod(int(k), W)
        clat, clon = float(LAT[i]), float(LON[j])
        dx = (clon - slon) * math.cos(math.radians(slat))
        dy = (clat - slat)
        dist = math.hypot(dx, dy) * 111.0
        brg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        top.append({"lat": round(clat, 2), "lon": round(clon, 2),
                    "pct": round(float(flat[k]) * 100, 1),
                    "km": round(dist, 0), "dir": _dir8(brg)})

    # directional breakdown (8 sectors) + near/far
    sectors = {d: 0.0 for d in ["N","NE","E","SE","S","SW","W","NW"]}
    near = far = 0.0
    II, JJ = np.nonzero(fmap > 0)
    for i, j in zip(II, JJ):
        f = float(fmap[i, j])
        clat, clon = float(LAT[i]), float(LON[j])
        dx = (clon - slon) * math.cos(math.radians(slat)); dy = (clat - slat)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            near += f; continue
        brg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        sectors[_dir8(brg)] += f
        if math.hypot(dx, dy) * 111.0 < 60: near += f
        else: far += f
    sect = [{"dir": d, "pct": round(v * 100, 1)} for d, v in sectors.items()]

    # heatmap (round, fraction*100) — ส่งทั้งกริด (≈150KB)
    grid = np.round(fmap * 100, 3).tolist()

    obs_v = s["obs"].get((di, st["locationId"]))
    return {
        "day": di, "date": s["times"][di].strftime("%Y-%m-%d"),
        "station": st, "pred_pm25": round(pred_pm, 1),
        "obs_pm25": round(obs_v, 1) if obs_v is not None else None,
        "top_cells": top, "sectors": sect,
        "near_pct": round(near * 100, 1), "far_pct": round(far * 100, 1),
        "grid": grid,
        "extent": [float(LON[0]), float(LON[-1]), float(LAT[0]), float(LAT[-1])],
    }


def _dir8(brg):
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[int((brg + 22.5) // 45) % 8]
