"""
Source attribution for one (station, day): where does the predicted smoke
come from? Returns the 111×101 contribution map, 8-direction sectors,
near/far split, and top contributing cells.
"""
from __future__ import annotations

import numpy as np
import torch

from .dataset import load_dataset
from .infer import load_model

_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _dir8(bearing_deg: float) -> str:
    return _DIRS[int(((bearing_deg % 360) + 22.5) // 45) % 8]


class Attributor:
    """Lazy-loaded helper; reuse across many queries."""
    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg; self.dev = device
        self.d = load_dataset(cfg)
        self.model, ck = load_model(cfg.ckpt_path, device)
        sf = ck.get("station_feats", self.d["station_feats"])
        self.sfeats_t = torch.tensor(np.asarray(sf, dtype="float32")).to(device)
        self.stations = self.d["stations"]
        self.LAT, self.LON = cfg.LAT, cfg.LON

    def attribution(self, day_idx: int, station_idx: int) -> dict:
        met = torch.tensor(self.d["met"][day_idx:day_idx + 1]).to(self.dev)
        emis = torch.tensor(self.d["emis"][day_idx:day_idx + 1]).to(self.dev)
        with torch.no_grad():
            out, K, b = self.model(met, emis, self.sfeats_t)
            contrib, _ = self.model.attribution(K, emis)     # (1,S,H,W)
        cmap = contrib[0, station_idx].cpu().numpy()          # (H,W)
        total = float(cmap.sum())
        pred = float(self.model.predict_median(out)[0, station_idx].cpu()
                     ) * self.d["meta"]["pm25_max"]

        st = self.stations.iloc[station_idx]
        slat, slon = float(st["lat"]), float(st["lon"])
        obs = float(self.d["y_raw"][day_idx, station_idx])

        # sectors + near/far
        sectors = {dir_: 0.0 for dir_ in _DIRS}
        near = far = 0.0
        H, W = cmap.shape
        for i in range(H):
            for j in range(W):
                c = cmap[i, j]
                if c <= 0:
                    continue
                dlat = self.LAT[i] - slat; dlon = self.LON[j] - slon
                bearing = (np.degrees(np.arctan2(dlon, dlat))) % 360
                sectors[_dir8(bearing)] += c
                dist = np.hypot(dlat, dlon) * 111.0          # ~km
                if dist <= 60:
                    near += c
                else:
                    far += c
        frac = lambda x: round(100 * x / total, 1) if total > 0 else 0.0
        flat = cmap.flatten()
        top = []
        for idx in np.argsort(flat)[::-1][:8]:
            i, j = divmod(int(idx), W)
            if flat[idx] <= 0:
                break
            dlat = self.LAT[i] - slat; dlon = self.LON[j] - slon
            top.append(dict(lat=float(self.LAT[i]), lon=float(self.LON[j]),
                            dir=_dir8(np.degrees(np.arctan2(dlon, dlat)) % 360),
                            pct=frac(float(flat[idx]))))
        return dict(station=str(st["location"]), pred=round(pred, 1),
                    obs=round(obs, 1) if not np.isnan(obs) else None,
                    near_pct=frac(near), far_pct=frac(far),
                    sectors={k: frac(v) for k, v in sectors.items()},
                    top=top, heatmap=cmap.tolist())
