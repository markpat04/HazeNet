"""
Attribution backend for the dashboard — now backed by the hazenet/ package
(configs/local.yaml + models/clno_local.pt), not the old src/*_m2 scripts.

Emits the same JSON shape the frontend already consumes, so index.html/app.js
need no changes. Loads once, caches in a module global.
"""
import os, sys, math, threading

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
CONFIG = os.path.join(ROOT, "configs", "local.yaml")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_LOCK  = threading.Lock()
_STATE = {}


def _load():
    if _STATE.get("ready"):
        return _STATE
    with _LOCK:
        if _STATE.get("ready"):
            return _STATE
        # IMPORTANT: read the zarr (pandas/pyarrow) BEFORE importing torch.
        # On Windows, importing torch first then building a pandas string index
        # in open_zarr segfaults (OpenMP/pyarrow clash). Data libs first avoids it.
        import numpy as np
        from hazenet.config import Config
        from hazenet.dataset import load_dataset
        cfg = Config.load(CONFIG)
        d = load_dataset(cfg)                      # met/emis already normalised

        import torch
        from hazenet.infer import load_model
        model, ck = load_model(cfg.ckpt_path, "cpu")
        sf = ck.get("station_feats", d["station_feats"])
        sfeats_t = torch.tensor(np.asarray(sf, dtype="float32"))

        st = d["stations"]
        stations = [{"idx": i,
                     "name": str(st.iloc[i]["location"]),
                     "lat": float(st.iloc[i]["lat"]),
                     "lon": float(st.iloc[i]["lon"]),
                     "ilat": int(st.iloc[i]["ilat"]),
                     "ilon": int(st.iloc[i]["ilon"]),
                     "locationId": int(st.iloc[i]["locationId"])}
                    for i in range(len(st))]

        y_raw = d["y_raw"]
        with np.errstate(all="ignore"):
            daily_mean = np.nanmean(y_raw, axis=1)

        _STATE.update(dict(ready=True, model=model, sfeats_t=sfeats_t,
                           np=np, torch=torch,
                           met=d["met"], emis=d["emis"], y_raw=y_raw,
                           times=d["times"], pm25_max=d["meta"]["pm25_max"],
                           LAT=cfg.LAT, LON=cfg.LON, H=cfg.H, W=cfg.W,
                           S=len(st), stations=stations, daily_mean=daily_mean))
    return _STATE


def meta():
    s = _load(); np = s["np"]; dm = s["daily_mean"]
    finite = np.where(np.isfinite(dm))[0]
    best = int(finite[np.argmax(dm[finite])]) if finite.size else 0
    days = [{"idx": i, "date": d.strftime("%Y-%m-%d"),
             "mean_pm": (round(float(dm[i]), 0) if np.isfinite(dm[i]) else None)}
            for i, d in enumerate(s["times"])]
    return {"stations": s["stations"], "days": days,
            "default_day": best, "H": s["H"], "W": s["W"]}


def attribution(day_idx, station_idx):
    s = _load(); np = s["np"]; torch = s["torch"]
    di = int(day_idx); si = int(station_idx)
    H, W = s["H"], s["W"]

    met_n = s["met"][di][None]                      # (1,in_ch,H,W) normalised
    emis_n = s["emis"][di][None]                    # (1,H,W) normalised
    with torch.no_grad():
        mt = torch.tensor(met_n, dtype=torch.float32)
        et = torch.tensor(emis_n, dtype=torch.float32)
        out, K, b = s["model"](mt, et, s["sfeats_t"])
        median = s["model"].predict_median(out)
        contrib, _ = s["model"].attribution(K, et)  # (1,S,H,W)

    pred_pm = float(median[0, si].item()) * s["pm25_max"]
    cmap = contrib[0, si].numpy()
    total = float(cmap.sum())
    fmap = (cmap / total) if total > 1e-9 else cmap * 0

    st = s["stations"][si]; slat, slon = st["lat"], st["lon"]
    LAT, LON = s["LAT"], s["LON"]

    flat = fmap.ravel()
    top = []
    for k in np.argsort(flat)[::-1][:12]:
        if flat[k] <= 0:
            continue
        i, j = divmod(int(k), W)
        clat, clon = float(LAT[i]), float(LON[j])
        dx = (clon - slon) * math.cos(math.radians(slat)); dy = clat - slat
        dist = math.hypot(dx, dy) * 111.0
        brg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        top.append({"lat": round(clat, 2), "lon": round(clon, 2),
                    "pct": round(float(flat[k]) * 100, 1),
                    "km": round(dist, 0), "dir": _dir8(brg)})

    sectors = {d: 0.0 for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
    near = far = 0.0
    II, JJ = np.nonzero(fmap > 0)
    for i, j in zip(II, JJ):
        f = float(fmap[i, j])
        clat, clon = float(LAT[i]), float(LON[j])
        dx = (clon - slon) * math.cos(math.radians(slat)); dy = clat - slat
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            near += f; continue
        brg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        sectors[_dir8(brg)] += f
        if math.hypot(dx, dy) * 111.0 < 60:
            near += f
        else:
            far += f
    sect = [{"dir": d, "pct": round(v * 100, 1)} for d, v in sectors.items()]

    obs_v = s["y_raw"][di, si]
    obs_v = None if not np.isfinite(obs_v) else round(float(obs_v), 1)
    return {
        "day": di, "date": s["times"][di].strftime("%Y-%m-%d"),
        "station": st, "pred_pm25": round(pred_pm, 1), "obs_pm25": obs_v,
        "top_cells": top, "sectors": sect,
        "near_pct": round(near * 100, 1), "far_pct": round(far * 100, 1),
        "grid": np.round(fmap * 100, 3).tolist(),
        "extent": [float(LON[0]), float(LON[-1]), float(LAT[0]), float(LAT[-1])],
    }


def _dir8(brg):
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((brg + 22.5) // 45) % 8]
