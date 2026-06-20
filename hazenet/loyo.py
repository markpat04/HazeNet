"""
Leave-One-Year-Out cross-validation — the honest generalisation gate.

A single held-out year (2023) is misleading: the "low-dust" years it compares
against are all in the training set, so per-year bias looks great for free.
LOYO instead holds out EACH year in turn, trains on the rest, and reports the
test error for every year. The gate is then the WORST held-out year.

Normalisation + emission/pm25 scaling are recomputed per fold (train years only)
to avoid leakage. Uses the model/regularisation from the given config.

The seen/new station split exposes the root cause: the station-agnostic CLNO
should close the gap between seen-station MAE (~19) and new-station MAE (~300)
from the old indexed model.
"""
from __future__ import annotations

import os, json
import numpy as np
import pandas as pd
import xarray as xr

# matplotlib and torch are imported lazily:
# - torch must come AFTER zarr is opened (avoids Windows OpenMP/pyarrow segfault)
# - matplotlib must come AFTER torch (avoids Windows DLL ordering crash in Agg backend)


def _compute_station_feats(cfg, stations, X, channel_names):
    """4-feature station descriptor: lat_norm, lon_norm, dem_norm, tpi_norm."""
    S = len(stations)
    H, W = X.shape[2], X.shape[3]
    ilat = np.clip(stations["ilat"].values.astype(int), 0, H - 1)
    ilon = np.clip(stations["ilon"].values.astype(int), 0, W - 1)

    def get_static(name):
        if name in channel_names:
            return X[0, channel_names.index(name), ilat, ilon].astype("float32")
        return np.zeros(S, dtype="float32")

    dem_vals = get_static("dem")
    tpi_vals = get_static("tpi")
    lat_norm = ((stations["lat"].values - cfg.lat0) / (cfg.lat1 - cfg.lat0)).astype("float32")
    lon_norm = ((stations["lon"].values - cfg.lon0) / (cfg.lon1 - cfg.lon0)).astype("float32")
    dem_norm = (dem_vals - dem_vals.mean()) / (dem_vals.std() + 1e-6)
    tpi_norm = (tpi_vals - tpi_vals.mean()) / (tpi_vals.std() + 1e-6)
    return np.stack([lat_norm, lon_norm, dem_norm, tpi_norm], axis=1).astype("float32")  # (S, 4)


def _load_raw(cfg):
    """Read cube + targets ONCE (before importing torch — avoids pyarrow segfault)."""
    cube = xr.open_zarr(cfg.datacube_zarr)
    X = cube.X.values
    names = [str(c) for c in cube.channel.values]
    e_idx = int(cube.attrs.get("emission_index", len(names) - 1))
    times = pd.DatetimeIndex(cube.time.values)
    met_raw = np.delete(X, e_idx, axis=1).astype("float32")
    emis_raw = X[:, e_idx].astype("float32")

    tgt = pd.read_csv(cfg.target_csv); tgt["date"] = pd.to_datetime(tgt["date"])
    stations = (tgt.groupby("locationId").first()[["location", "lat", "lon", "ilat", "ilon"]]
                .reset_index().sort_values("locationId").reset_index(drop=True))
    S = len(stations)
    t_map = {pd.Timestamp(t): i for i, t in enumerate(times)}
    s_map = {sid: i for i, sid in enumerate(stations["locationId"])}
    y_raw = np.full((len(times), S), np.nan, dtype="float32")
    for _, r in tgt.iterrows():
        ti = t_map.get(r["date"]); si = s_map.get(r["locationId"])
        if ti is not None and si is not None:
            y_raw[ti, si] = r["pm25"]

    station_feats = _compute_station_feats(cfg, stations, X, names)   # (S, 4)

    # ── optional wind-advection weights (for PIDGGNN) ──
    a_wind = None
    if getattr(cfg, "compute_advection_weights", False):
        met_channel_names = [n for i, n in enumerate(names) if i != e_idx]
        if "u10" in met_channel_names and "v10" in met_channel_names:
            from .model.transport import precompute_advection, row_normalize
            u10_i = met_channel_names.index("u10")
            v10_i = met_channel_names.index("v10")
            station_xy = stations[["lat", "lon"]].values.astype("float32")
            T = met_raw.shape[0]
            print(f"Precomputing advection weights ({T}×{S}×{met_raw.shape[2]*met_raw.shape[3]}) ...")
            a_wind = precompute_advection(met_raw[:, u10_i], met_raw[:, v10_i],
                                          cfg.LAT, cfg.LON, station_xy)
            a_wind = row_normalize(a_wind).astype("float32")
            print(f"  a_wind done: {a_wind.shape}")

    return met_raw, emis_raw, y_raw, times, S, station_feats, a_wind


def _fit_fold(cfg, met_raw, emis_raw, y_raw, station_feats, a_wind, train, test, S, dev):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from .model import (build_model, masked_mse, pinball_loss,
                        compute_lds_weights, weighted_pinball_loss,
                        weighted_masked_mse)
    torch.manual_seed(cfg.seed)

    mu = met_raw[train].mean(axis=(0, 2, 3), keepdims=True)
    sd = met_raw[train].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    met = (met_raw - mu) / sd
    e_max = float(emis_raw[train].max()) + 1e-6
    emis = emis_raw / e_max
    pm_max = float(np.nanmax(y_raw[train])) + 1e-6
    y = y_raw / pm_max

    in_ch = met.shape[1]
    model = build_model(cfg, in_ch=in_ch).to(dev)
    sfeats_t = torch.tensor(station_feats).to(dev)
    met_t = torch.tensor(met); emis_t = torch.tensor(emis); y_t = torch.tensor(y)
    tri = np.where(train)[0]; tei = np.where(test)[0]
    has_wind = a_wind is not None
    if has_wind:
        a_wind_t = torch.tensor(a_wind)
        tr = DataLoader(TensorDataset(met_t[tri], emis_t[tri], a_wind_t[tri], y_t[tri]),
                        batch_size=cfg.batch_size, shuffle=True)
        te = DataLoader(TensorDataset(met_t[tei], emis_t[tei], a_wind_t[tei], y_t[tei]),
                        batch_size=cfg.batch_size)
    else:
        tr = DataLoader(TensorDataset(met_t[tri], emis_t[tri], y_t[tri]),
                        batch_size=cfg.batch_size, shuffle=True)
        te = DataLoader(TensorDataset(met_t[tei], emis_t[tei], y_t[tei]),
                        batch_size=cfg.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    # Loss — optional Label Distribution Smoothing (imbalanced regression).
    # Weights are computed on TRAIN targets only (normalised scale) per fold.
    if getattr(cfg, "lds", False):
        edges, wts = compute_lds_weights(
            y[train], sigma=cfg.lds_sigma, reweight=cfg.lds_reweight,
            max_weight=cfg.lds_max_weight)
        if cfg.quantiles:
            lossf = lambda o, yb: weighted_pinball_loss(o, yb, cfg.quantiles, edges, wts)
        else:
            lossf = lambda o, yb: weighted_masked_mse(o, yb, edges, wts)
    else:
        lossf = (lambda o, yb: pinball_loss(o, yb, cfg.quantiles)) if cfg.quantiles else masked_mse

    def _unpack(mb):
        if has_wind:
            mm, me, mw, my = (x.to(dev) for x in mb)
        else:
            mm, me, my = (x.to(dev) for x in mb); mw = None
        return mm, me, mw, my

    best, bad, best_state = float("inf"), 0, None
    import copy
    for ep in range(cfg.epochs):
        model.train()
        for mb in tr:
            mm, me, mw, my = _unpack(mb)
            opt.zero_grad(); out, _, _ = model(mm, me, sfeats_t, mw)
            loss = lossf(out, my); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        sched.step()
        model.eval(); tl = 0.0
        with torch.no_grad():
            for mb in te:
                mm, me, mw, my = _unpack(mb)
                tl += lossf(model(mm, me, sfeats_t, mw)[0], my).item()
        tl /= max(1, len(te))
        if tl < best - 1e-5:
            best, bad, best_state = tl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    # predict held-out year
    model.eval(); preds = []
    with torch.no_grad():
        if has_wind:
            infer_ds = TensorDataset(met_t[tei], emis_t[tei], a_wind_t[tei])
        else:
            infer_ds = TensorDataset(met_t[tei], emis_t[tei])
        for mb in DataLoader(infer_ds, batch_size=cfg.batch_size):
            if has_wind:
                mm, me, mw = (x.to(dev) for x in mb)
            else:
                mm, me = (x.to(dev) for x in mb); mw = None
            preds.append(model.predict_median(model(mm, me, sfeats_t, mw)[0]).cpu())
    pred = torch.cat(preds).numpy() * pm_max
    g = y_raw[test]; m = ~np.isnan(g)
    err = pred - g

    # split held-out points by whether the station was SEEN in any training year
    seen_station = (~np.isnan(y_raw[train])).any(axis=0)          # (S,)
    seen_col = np.broadcast_to(seen_station, g.shape)
    ms, mn = m & seen_col, m & ~seen_col

    def sub(mask):
        if not mask.any():
            return dict(MAE=None, bias=None, n=0)
        return dict(MAE=float(np.abs(err[mask]).mean()),
                    bias=float(err[mask].mean()), n=int(mask.sum()))

    return dict(MAE=float(np.abs(err[m]).mean()),
                RMSE=float(np.sqrt((err[m] ** 2).mean())),
                bias=float(err[m].mean()),
                obs_mean=float(np.nanmean(g)), n=int(m.sum()),
                seen=sub(ms), new=sub(mn))


def loyo(cfg) -> dict:
    met_raw, emis_raw, y_raw, times, S, station_feats, a_wind = _load_raw(cfg)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yrs = np.array([t.year for t in times])
    years = sorted(set(int(y) for y in yrs))
    print(f"LOYO over {years}  model={cfg.model_kind}  dev={dev}")

    rows = []
    for Y in years:
        test = yrs == Y; train = ~test
        r = _fit_fold(cfg, met_raw, emis_raw, y_raw, station_feats, a_wind, train, test, S, dev)
        r["year"] = Y
        rows.append(r)
        sn, nw = r["seen"], r["new"]
        sn_s = f"{sn['MAE']:.1f}(n{sn['n']})" if sn["MAE"] is not None else "—"
        nw_s = f"{nw['MAE']:.1f}(n{nw['n']})" if nw["MAE"] is not None else "—"
        print(f"  holdout {Y}: MAE={r['MAE']:.1f}  bias={r['bias']:+.1f}  "
              f"obs={r['obs_mean']:.0f}  |  SEEN-station MAE={sn_s}  NEW-station MAE={nw_s}")

    df = pd.DataFrame(rows)
    worst_mae = float(df["MAE"].max())
    mean_mae = float(df["MAE"].mean())
    seen_maes = [r["seen"]["MAE"] for r in rows if r["seen"]["MAE"] is not None]
    seen_mean = float(np.mean(seen_maes)) if seen_maes else float("nan")
    new_maes = [r["new"]["MAE"] for r in rows if r["new"]["MAE"] is not None]
    new_mean = float(np.mean(new_maes)) if new_maes else float("nan")
    print(f"\nLOYO summary: ALL mean MAE={mean_mae:.1f}  worst={worst_mae:.1f}  "
          f"|  SEEN mean MAE={seen_mean:.1f}  NEW mean MAE={new_mean:.1f}  "
          f"(← spatial generalisation)")

    out = dict(folds=rows, mean_MAE=mean_mae, worst_MAE=worst_mae,
               seen_mean_MAE=seen_mean, new_mean_MAE=new_mean)
    os.makedirs(cfg.models_dir, exist_ok=True)
    json.dump(out, open(os.path.join(cfg.models_dir, f"loyo_{cfg.name}.json"), "w"),
              indent=2, default=float)

    if os.environ.get("HAZENET_NOFIG"):
        return out
    os.makedirs(cfg.figures_dir, exist_ok=True)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(df["year"].astype(str), df["MAE"], color="tab:purple")
        ax.set_ylabel("held-out MAE (µg/m³)")
        ax.set_title(f"{cfg.name} — Leave-One-Year-Out  (mean {mean_mae:.1f}, worst {worst_mae:.1f})")
        ax.grid(alpha=0.3, axis="y")
        fig.savefig(os.path.join(cfg.figures_dir, f"{cfg.name}_loyo.png"),
                    dpi=130, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[loyo] figure save failed (non-fatal): {e}")

    return out


# ─────────────────────────────────────────────────────────────────────────
# Leave-One-Station-Out (LOSO) — the pure spatial-generalisation gate.
#
# LOYO holds out a YEAR; its SEEN/NEW split hints at spatial generalisation but
# the held-out year still contains many seen stations. LOSO instead holds out a
# GROUP OF STATIONS entirely (no day at those stations is ever in training) and
# asks: can the station-agnostic CLNO predict at a location it has never seen?
# This is the field-standard test (cf. arXiv 2505.18461 on location encoders).
# ─────────────────────────────────────────────────────────────────────────
def _fit_loso_fold(cfg, met_raw, emis_raw, y_raw, station_feats, a_wind, train_st, test_st, dev):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from .model import (build_model, masked_mse, pinball_loss,
                        compute_lds_weights, weighted_pinball_loss, weighted_masked_mse)
    import copy
    torch.manual_seed(cfg.seed)

    # No temporal split → normalise met over all days (targets not involved).
    mu = met_raw.mean(axis=(0, 2, 3), keepdims=True)
    sd = met_raw.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    met = (met_raw - mu) / sd
    e_max = float(emis_raw.max()) + 1e-6
    emis = emis_raw / e_max
    pm_max = float(np.nanmax(y_raw[:, train_st])) + 1e-6      # train stations only
    y = y_raw / pm_max

    y_tr = y.copy(); y_tr[:, ~train_st] = np.nan              # mask targets by station
    y_te = y.copy(); y_te[:, ~test_st] = np.nan

    in_ch = met.shape[1]
    model = build_model(cfg, in_ch=in_ch).to(dev)
    sfeats_t = torch.tensor(station_feats).to(dev)            # ALL stations (needed to predict held-out)
    met_t = torch.tensor(met); emis_t = torch.tensor(emis)
    ytr_t = torch.tensor(y_tr); yte_t = torch.tensor(y_te)
    has_wind = a_wind is not None
    if has_wind:
        a_wind_t = torch.tensor(a_wind)
        tr = DataLoader(TensorDataset(met_t, emis_t, a_wind_t, ytr_t), batch_size=cfg.batch_size, shuffle=True)
        te = DataLoader(TensorDataset(met_t, emis_t, a_wind_t, yte_t), batch_size=cfg.batch_size)
    else:
        tr = DataLoader(TensorDataset(met_t, emis_t, ytr_t), batch_size=cfg.batch_size, shuffle=True)
        te = DataLoader(TensorDataset(met_t, emis_t, yte_t), batch_size=cfg.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    if getattr(cfg, "lds", False):
        edges, wts = compute_lds_weights(y_tr, sigma=cfg.lds_sigma,
                                         reweight=cfg.lds_reweight, max_weight=cfg.lds_max_weight)
        lossf = ((lambda o, yb: weighted_pinball_loss(o, yb, cfg.quantiles, edges, wts))
                 if cfg.quantiles else (lambda o, yb: weighted_masked_mse(o, yb, edges, wts)))
    else:
        lossf = (lambda o, yb: pinball_loss(o, yb, cfg.quantiles)) if cfg.quantiles else masked_mse

    def _unpack_loso(mb):
        if has_wind:
            mm, me, mw, my = (x.to(dev) for x in mb)
        else:
            mm, me, my = (x.to(dev) for x in mb); mw = None
        return mm, me, mw, my

    best, bad, best_state = float("inf"), 0, None
    for ep in range(cfg.epochs):
        model.train()
        for mb in tr:
            mm, me, mw, my = _unpack_loso(mb)
            opt.zero_grad(); out, _, _ = model(mm, me, sfeats_t, mw)
            loss = lossf(out, my); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        sched.step()
        model.eval(); tl = 0.0
        with torch.no_grad():
            for mb in te:
                mm, me, mw, my = _unpack_loso(mb)
                tl += lossf(model(mm, me, sfeats_t, mw)[0], my).item()
        tl /= max(1, len(te))
        if tl < best - 1e-5:
            best, bad, best_state = tl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval(); preds = []
    with torch.no_grad():
        if has_wind:
            infer_ds = TensorDataset(met_t, emis_t, a_wind_t)
        else:
            infer_ds = TensorDataset(met_t, emis_t)
        for mb in DataLoader(infer_ds, batch_size=cfg.batch_size):
            if has_wind:
                mm, me, mw = (x.to(dev) for x in mb)
            else:
                mm, me = (x.to(dev) for x in mb); mw = None
            preds.append(model.predict_median(model(mm, me, sfeats_t, mw)[0]).cpu())
    pred = torch.cat(preds).numpy() * pm_max                 # (T, S)
    g = y_raw.copy(); g[:, ~test_st] = np.nan                # eval only at held-out stations
    return pred - g, ~np.isnan(g), g


def loso(cfg, k: int = 5) -> dict:
    met_raw, emis_raw, y_raw, times, S, station_feats, a_wind = _load_raw(cfg)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed)
    has = (~np.isnan(y_raw)).any(axis=0)                     # stations with any data
    st_idx = np.where(has)[0]; rng.shuffle(st_idx)
    folds_idx = np.array_split(st_idx, k)
    yrs = np.array([t.year for t in times])
    print(f"LOSO {k}-fold over {len(st_idx)} stations  model={cfg.model_kind}  dev={dev}")

    rows = []
    for kf, held in enumerate(folds_idx):
        test_st = np.zeros(S, dtype=bool); test_st[held] = True
        train_st = has & ~test_st
        err, m, g = _fit_loso_fold(cfg, met_raw, emis_raw, y_raw, station_feats, a_wind,
                                   train_st, test_st, dev)
        mae = float(np.abs(err[m]).mean()); bias = float(err[m].mean())
        rmse = float(np.sqrt((err[m] ** 2).mean()))
        m23 = m & (yrs == 2023)[:, None]
        mae23 = float(np.abs(err[m23]).mean()) if m23.any() else None
        rows.append(dict(fold=kf, n_stations=int(test_st.sum()), MAE=mae, RMSE=rmse,
                         bias=bias, n=int(m.sum()), mae_2023=mae23))
        s23 = "—" if mae23 is None else f"{mae23:.2f}"
        print(f"  fold {kf}: {int(test_st.sum())} held  MAE={mae:.2f}  bias={bias:+.2f}  2023MAE={s23}")

    mean_mae = float(np.mean([r["MAE"] for r in rows]))
    worst = float(np.max([r["MAE"] for r in rows]))
    print(f"\nLOSO summary: mean MAE={mean_mae:.2f}  worst-fold={worst:.2f}  "
          f"(← pure spatial generalisation)")
    out = dict(folds=rows, mean_MAE=mean_mae, worst_MAE=worst, k=k)
    os.makedirs(cfg.models_dir, exist_ok=True)
    json.dump(out, open(os.path.join(cfg.models_dir, f"loso_{cfg.name}.json"), "w"),
              indent=2, default=float)
    return out
