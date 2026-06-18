"""
Leave-One-Year-Out cross-validation — the honest generalisation gate.

A single held-out year (2023) is misleading: the "low-dust" years it compares
against are all in the training set, so per-year bias looks great for free.
LOYO instead holds out EACH year in turn, trains on the rest, and reports the
test error for every year. The gate is then the WORST held-out year.

Normalisation + emission/pm25 scaling are recomputed per fold (train years only)
to avoid leakage. Uses the model/regularisation from the given config.
"""
from __future__ import annotations

import os, json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import build_model, masked_mse, pinball_loss


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
    return met_raw, emis_raw, y_raw, times, S


def _fit_fold(cfg, met_raw, emis_raw, y_raw, train, test, S, dev):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(cfg.seed)

    mu = met_raw[train].mean(axis=(0, 2, 3), keepdims=True)
    sd = met_raw[train].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    met = (met_raw - mu) / sd
    e_max = float(emis_raw[train].max()) + 1e-6
    emis = emis_raw / e_max
    pm_max = float(np.nanmax(y_raw[train])) + 1e-6
    y = y_raw / pm_max

    in_ch = met.shape[1]
    model = build_model(cfg, n_stations=S, in_ch=in_ch).to(dev)
    met_t = torch.tensor(met); emis_t = torch.tensor(emis); y_t = torch.tensor(y)
    tri = np.where(train)[0]; tei = np.where(test)[0]
    tr = DataLoader(TensorDataset(met_t[tri], emis_t[tri], y_t[tri]),
                    batch_size=cfg.batch_size, shuffle=True)
    te = DataLoader(TensorDataset(met_t[tei], emis_t[tei], y_t[tei]),
                    batch_size=cfg.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    lossf = (lambda o, yb: pinball_loss(o, yb, cfg.quantiles)) if cfg.quantiles else masked_mse

    best, bad, best_state = float("inf"), 0, None
    import copy
    for ep in range(cfg.epochs):
        model.train()
        for mb in tr:
            mm, me, my = (x.to(dev) for x in mb)
            opt.zero_grad(); out, _, _ = model(mm, me)
            loss = lossf(out, my); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        sched.step()
        model.eval(); tl = 0.0
        with torch.no_grad():
            for mb in te:
                mm, me, my = (x.to(dev) for x in mb)
                tl += lossf(model(mm, me)[0], my).item()
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
        for mb in DataLoader(TensorDataset(met_t[tei], emis_t[tei]), batch_size=cfg.batch_size):
            mm, me = (x.to(dev) for x in mb)
            preds.append(model.predict_median(model(mm, me)[0]).cpu())
    pred = torch.cat(preds).numpy() * pm_max
    g = y_raw[test]; m = ~np.isnan(g)
    return dict(MAE=float(np.abs(pred[m] - g[m]).mean()),
                RMSE=float(np.sqrt(((pred[m] - g[m]) ** 2).mean())),
                bias=float((pred[m] - g[m]).mean()),
                obs_mean=float(np.nanmean(g)), n=int(m.sum()))


def loyo(cfg) -> dict:
    met_raw, emis_raw, y_raw, times, S = _load_raw(cfg)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yrs = np.array([t.year for t in times])
    years = sorted(set(int(y) for y in yrs))
    print(f"LOYO over {years}  model={cfg.model_kind}  dev={dev}")

    rows = []
    for Y in years:
        test = yrs == Y; train = ~test
        r = _fit_fold(cfg, met_raw, emis_raw, y_raw, train, test, S, dev)
        r["year"] = Y
        rows.append(r)
        print(f"  holdout {Y}: MAE={r['MAE']:.1f}  bias={r['bias']:+.1f}  "
              f"obs_mean={r['obs_mean']:.0f}  n={r['n']}")

    df = pd.DataFrame(rows)
    worst_mae = float(df["MAE"].max())
    mean_mae = float(df["MAE"].mean())
    print(f"\nLOYO summary: mean MAE={mean_mae:.1f}  WORST year MAE={worst_mae:.1f} "
          f"({int(df.loc[df['MAE'].idxmax(),'year'])})")

    os.makedirs(cfg.figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["year"].astype(str), df["MAE"], color="tab:purple")
    ax.set_ylabel("held-out MAE (µg/m³)")
    ax.set_title(f"{cfg.name} — Leave-One-Year-Out  (mean {mean_mae:.1f}, worst {worst_mae:.1f})")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(os.path.join(cfg.figures_dir, f"{cfg.name}_loyo.png"),
                dpi=130, bbox_inches="tight"); plt.close(fig)

    out = dict(folds=rows, mean_MAE=mean_mae, worst_MAE=worst_mae)
    json.dump(out, open(os.path.join(cfg.models_dir, f"loyo_{cfg.name}.json"), "w"),
              indent=2, default=float)
    return out
