"""
Load datacube.zarr + targets into tensors, with train-only normalisation.

Returns everything train/eval need, and the meta dict that is saved into the
checkpoint so inference can reproduce the exact normalisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def load_dataset(cfg):
    cube = xr.open_zarr(cfg.datacube_zarr)
    X = cube.X.values                                  # (T, C, H, W)
    names = [str(c) for c in cube.channel.values]
    e_idx = int(cube.attrs.get("emission_index", len(names) - 1))
    times = pd.DatetimeIndex(cube.time.values)
    train_mask = cube.train_mask.values.astype(bool)
    test_mask = cube.test_mask.values.astype(bool)
    T, C, H, W = X.shape

    met_raw = np.delete(X, e_idx, axis=1).astype("float32")   # (T, n_met, H, W)
    emis_raw = X[:, e_idx].astype("float32")                  # (T, H, W)
    in_ch = met_raw.shape[1]

    # normalise met on TRAIN only
    met_mu = met_raw[train_mask].mean(axis=(0, 2, 3), keepdims=True)
    met_std = met_raw[train_mask].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    met_norm = (met_raw - met_mu) / met_std

    e_max = float(emis_raw[train_mask].max()) + 1e-6
    emis_norm = emis_raw / e_max

    # targets
    tgt = pd.read_csv(cfg.target_csv)
    tgt["date"] = pd.to_datetime(tgt["date"])
    stations = (tgt.groupby("locationId")
                .first()[["location", "lat", "lon", "ilat", "ilon"]]
                .reset_index().sort_values("locationId").reset_index(drop=True))
    S = len(stations)

    pm25_max = float(tgt[tgt["date"].dt.year != cfg.test_year]["pm25"].max()) + 1e-6
    t_map = {pd.Timestamp(t): i for i, t in enumerate(times)}
    s_map = {sid: i for i, sid in enumerate(stations["locationId"])}

    y_raw = np.full((T, S), np.nan, dtype="float32")
    for _, r in tgt.iterrows():
        ti = t_map.get(r["date"]); si = s_map.get(r["locationId"])
        if ti is not None and si is not None:
            y_raw[ti, si] = r["pm25"]
    y_norm = y_raw / pm25_max

    meta = dict(H=H, W=W, S=S, T=int(T), in_ch=in_ch, e_max=e_max,
                pm25_max=pm25_max, channels=[n for n in names if n != "emission"],
                met_mu=met_mu.tolist(), met_std=met_std.tolist())
    return dict(met=met_norm, emis=emis_norm, y_norm=y_norm, y_raw=y_raw,
                stations=stations, meta=meta, times=times,
                train_mask=train_mask, test_mask=test_mask)
