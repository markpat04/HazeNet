"""
Load datacube.zarr + targets into tensors, with train-only normalisation.

Returns everything train/eval need, and the meta dict that is saved into the
checkpoint so inference can reproduce the exact normalisation.

station_feats (S, 4): [lat_norm, lon_norm, dem_norm, tpi_norm]
  - lat/lon: linear position in domain [0..1]
  - dem/tpi: z-scored across all stations (geography is not a leakage risk)
These allow the station-agnostic CLNO to predict at any location.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def _station_feats(cfg, stations, X, channel_names):
    """Extract 4-feature station descriptor: lat_norm, lon_norm, dem_norm, tpi_norm."""
    S = len(stations)
    ilat = stations["ilat"].values.astype(int)
    ilon = stations["ilon"].values.astype(int)

    # clip indices to grid bounds (safety)
    H, W = X.shape[2], X.shape[3]
    ilat = np.clip(ilat, 0, H - 1)
    ilon = np.clip(ilon, 0, W - 1)

    def get_static(name):
        if name in channel_names:
            idx = channel_names.index(name)
            return X[0, idx, ilat, ilon].astype("float32")
        return np.zeros(S, dtype="float32")

    dem_vals = get_static("dem")
    tpi_vals = get_static("tpi")

    lat_norm = ((stations["lat"].values - cfg.lat0) / (cfg.lat1 - cfg.lat0)).astype("float32")
    lon_norm = ((stations["lon"].values - cfg.lon0) / (cfg.lon1 - cfg.lon0)).astype("float32")

    dem_norm = (dem_vals - dem_vals.mean()) / (dem_vals.std() + 1e-6)
    tpi_norm = (tpi_vals - tpi_vals.mean()) / (tpi_vals.std() + 1e-6)

    return np.stack([lat_norm, lon_norm, dem_norm, tpi_norm], axis=1)   # (S, 4)


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

    # station features for the station-agnostic model
    met_channel_names = [n for n in names if n != "emission"]
    station_feats = _station_feats(cfg, stations, X, names).astype("float32")  # (S, 4)

    meta = dict(H=H, W=W, S=S, T=int(T), in_ch=in_ch, e_max=e_max,
                pm25_max=pm25_max, channels=met_channel_names,
                met_mu=met_mu.tolist(), met_std=met_std.tolist())
    return dict(met=met_norm, emis=emis_norm, y_norm=y_norm, y_raw=y_raw,
                stations=stations, station_feats=station_feats,
                meta=meta, times=times,
                train_mask=train_mask, test_mask=test_mask)
