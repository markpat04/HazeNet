"""
Train the Conditionally-Linear Neural Operator (CLNO) on the Phase 0 datacube.

Input  : data/processed/datacube.zarr + target_pm25.csv
Output : models/clno.pt, models/metrics.json, figures/clno_loss.png,
         figures/clno_pred_vs_true.png

Hyperparams are tuned for Phase 0 (small data, overfit is expected).
Phase 2+ will scale H/W/S and train on multi-year SEA domain.

Run: conda run -n hazenet --no-capture-output python src/train_operator.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_operator import CLNO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────
def load_data():
    """Return tensors ready for training (normalised)."""
    cube = xr.open_zarr(os.path.join(PROC, "datacube.zarr"))
    X = cube.X.values                           # (T, 4, H, W)
    T, _, H, W = X.shape
    times = cube.time.values

    met_raw  = X[:, :3].astype("float32")      # u10, v10, dem
    emis_raw = X[:,  3].astype("float32")      # FRP

    # Normalise met per-channel (zero-mean, unit-std across T)
    met_mu  = met_raw.mean(axis=(0, 2, 3), keepdims=True)
    met_std = met_raw.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    met_norm = (met_raw - met_mu) / met_std

    # Normalise emission to [0, 1]
    e_max = float(emis_raw.max()) + 1e-6
    emis_norm = emis_raw / e_max

    # Load PM2.5 → pivot to (T, S) matrix
    tgt = pd.read_csv(os.path.join(PROC, "target_pm25.csv"))
    tgt["date"] = pd.to_datetime(tgt["date"])

    stations = (tgt.groupby("locationId")
                   .first()[["location", "lat", "lon", "ilat", "ilon"]]
                   .reset_index()
                   .sort_values("locationId")
                   .reset_index(drop=True))
    S = len(stations)

    pm25_max = float(tgt.pm25.max()) + 1e-6
    t_map = {pd.Timestamp(t): i for i, t in enumerate(times)}
    s_map = {sid: i for i, sid in enumerate(stations["locationId"])}

    y_raw  = np.full((T, S), np.nan, dtype="float32")
    for _, r in tgt.iterrows():
        ti = t_map.get(r["date"])
        si = s_map.get(r["locationId"])
        if ti is not None and si is not None:
            y_raw[ti, si] = r["pm25"]

    y_norm = y_raw / pm25_max                   # normalised targets

    meta = dict(H=H, W=W, S=S, T=T,
                e_max=e_max, pm25_max=pm25_max,
                met_mu=met_mu, met_std=met_std)
    return met_norm, emis_norm, y_norm, y_raw, stations, meta


# ─────────────────────────────────────────────
def masked_mse(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0
    return F.mse_loss(pred[mask], target[mask])


def masked_mae(pred, target):
    mask = ~torch.isnan(target)
    return (pred[mask] - target[mask]).abs().mean().item()


# ─────────────────────────────────────────────
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}"
          + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    met_n, emis_n, y_n, y_raw, stations, meta = load_data()
    H, W, S, T = meta["H"], meta["W"], meta["S"], meta["T"]
    print(f"Grid {H}x{W}, stations {S}, days {T}")
    print(f"PM2.5  mean={np.nanmean(y_raw):.1f}  max={np.nanmax(y_raw):.1f} ug/m3")

    met  = torch.tensor(met_n,  dtype=torch.float32, device=dev)   # (T,3,H,W)
    emis = torch.tensor(emis_n, dtype=torch.float32, device=dev)   # (T,H,W)
    y    = torch.tensor(y_n,    dtype=torch.float32, device=dev)   # (T,S)

    # Temporal split: last 3 days = test
    n_test = 3
    tr = slice(0, T - n_test)
    te = slice(T - n_test, T)

    model = CLNO(H=H, W=W, n_stations=S, hidden=32).to(dev)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"CLNO params: {n_p:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800)

    EPOCHS = 800
    tr_losses, te_losses = [], []

    print("\nTraining...")
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pm25_pred, _, _ = model(met[tr], emis[tr])
        loss = masked_mse(pm25_pred, y[tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            pm25_te, _, _ = model(met[te], emis[te])
            te_loss = masked_mse(pm25_te, y[te])
        te_losses.append(te_loss.item())

        if (ep + 1) % 200 == 0:
            print(f"  ep {ep+1:4d}  train={loss.item():.4f}  test={te_loss.item():.4f}")

    # ── Final evaluation ──────────────────────────────────────────────
    model.eval()
    pm25_max = meta["pm25_max"]
    with torch.no_grad():
        pred_tr, K_tr, b_tr = model(met[tr], emis[tr])
        pred_te, K_te, b_te = model(met[te], emis[te])

    # Un-normalise
    pr_tr = pred_tr.cpu().numpy() * pm25_max
    pr_te = pred_te.cpu().numpy() * pm25_max
    gt_tr = y_raw[:T - n_test]
    gt_te = y_raw[T - n_test:]

    def flat_valid(pred, gt):
        m = ~np.isnan(gt)
        return pred[m], gt[m]

    p_tr, g_tr = flat_valid(pr_tr, gt_tr)
    p_te, g_te = flat_valid(pr_te, gt_te)
    mae_tr = np.abs(p_tr - g_tr).mean()
    mae_te = np.abs(p_te - g_te).mean()
    rmse_te = np.sqrt(((p_te - g_te)**2).mean())
    print(f"\n[CLNO] train MAE={mae_tr:.1f}  test MAE={mae_te:.1f}  RMSE={rmse_te:.1f}")

    # ── Save model ────────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "H": H, "W": W, "S": S, "hidden": 32,
        "stations": stations.to_dict(),
        "meta": {k: (v.tolist() if hasattr(v, "tolist") else v)
                 for k, v in meta.items()
                 if not isinstance(v, np.ndarray)},
        "met_mu":   meta["met_mu"].tolist(),
        "met_std":  meta["met_std"].tolist(),
    }, os.path.join(ROOT, "models", "clno.pt"))

    # ── Update metrics.json ───────────────────────────────────────────
    mp = os.path.join(ROOT, "models", "metrics.json")
    d = json.load(open(mp)) if os.path.exists(mp) else {}
    d["clno"] = dict(MAE=round(float(mae_te), 2), RMSE=round(float(rmse_te), 2),
                     n_train=int((~np.isnan(gt_tr)).sum()),
                     n_test=int((~np.isnan(gt_te)).sum()))
    json.dump(d, open(mp, "w"), indent=2)

    # ── Plots ─────────────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tr_losses, label="train", alpha=0.8)
    ax.plot(te_losses, label="test",  alpha=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (normalised)")
    ax.set_title("CLNO training loss")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.savefig(os.path.join(ROOT, "figures", "clno_loss.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Pred vs true (test set)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(g_te, p_te, alpha=0.6, edgecolors="k", linewidths=0.3, c="tab:orange")
    lim = [0, max(g_te.max(), p_te.max()) * 1.1]
    ax.plot(lim, lim, "r--", lw=1, label="perfect")
    ax.set_xlabel("Observed PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.set_title(f"CLNO — test set\nMAE={mae_te:.1f}  RMSE={rmse_te:.1f} µg/m³")
    ax.legend()
    fig.savefig(os.path.join(ROOT, "figures", "clno_pred_vs_true.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("[ok] figures/clno_loss.png  clno_pred_vs_true.png")

    # Save tensors for eval_operator.py
    torch.save({
        "K_tr": K_tr.cpu(), "b_tr": b_tr.cpu(),
        "K_te": K_te.cpu(), "b_te": b_te.cpu(),
        "met_n": met.cpu(), "emis_n": emis.cpu(),
        "y_raw": torch.tensor(y_raw),
        "tr_slice": (0, T - n_test), "te_slice": (T - n_test, T),
    }, os.path.join(ROOT, "models", "clno_artifacts.pt"))
    print("[ok] models/clno_artifacts.pt  (for eval_operator.py)")


if __name__ == "__main__":
    main()
