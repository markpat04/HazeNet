"""
Evaluate CLNO M2: Attribution + Inversion + Comparison for SEA domain.

Loads models/clno_m2.pt + clno_m2_artifacts.pt and generates:
  figures/attribution_m2_worst_day.png  — worst haze day, source maps
  figures/inversion_m2_vs_firms.png     — recovered vs actual FIRMS
  figures/model_comparison_m2.png       — all model metrics updated

Run: KMP_DUPLICATE_LIB_OK=TRUE conda run -n hazenet --no-capture-output python src/eval_operator_m2.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_operator_m2 import CLNOLowRank, CLNOGlobalV
from config_m2 import ROOT, PROC, LAT, LON, TEST_YEAR

MODELS_DIR  = os.path.join(ROOT, "models")
FIGURES_DIR = os.path.join(ROOT, "figures")


def load_everything():
    ckpt = torch.load(os.path.join(MODELS_DIR, "clno_m2.pt"),
                      map_location="cpu", weights_only=False)
    art  = torch.load(os.path.join(MODELS_DIR, "clno_m2_artifacts.pt"),
                      map_location="cpu", weights_only=False)

    cls = CLNOGlobalV if ckpt.get("model_class") == "CLNOGlobalV" else CLNOLowRank
    model = cls(H=ckpt["H"], W=ckpt["W"],
                n_stations=ckpt["S"],
                hidden=ckpt["hidden"],
                rank=ckpt["rank"],
                dropout=ckpt.get("dropout", 0.1),
                in_ch=ckpt.get("in_ch", 3))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  model: {ckpt.get('model_class','CLNOLowRank')}  "
          f"rank={ckpt['rank']}  config={ckpt.get('config')}")

    stations = pd.DataFrame(ckpt["stations"])
    pm25_max = ckpt["meta"]["pm25_max"]
    return model, art, stations, pm25_max, ckpt


def plot_attribution(model, art, stations, pm25_max, ckpt):
    y_raw    = art["y_raw"].numpy()        # (T, S)
    emis_n   = art["emis_n"]              # (T, H, W) tensor

    # Consider only test-year days
    te_mask  = art["te_mask"].numpy()
    K_te     = art["K_te"]                # (T_te, S, G)
    b_te     = art["b_te"]

    y_te  = y_raw[te_mask]
    day_mean = np.nanmean(y_te, axis=1)
    local_worst = int(np.argmax(day_mean))
    worst_t = np.where(te_mask)[0][local_worst]

    cube = xr.open_zarr(os.path.join(PROC, "datacube_m2.zarr"))
    worst_date = pd.Timestamp(cube.time.values[worst_t]).strftime("%Y-%m-%d")
    print(f"  Worst haze day (test): t={worst_t}  date={worst_date}  "
          f"avg PM2.5={day_mean[local_worst]:.0f} µg/m³")

    K_t    = K_te[local_worst:local_worst+1]           # (1, S, G)
    emis_t = emis_n[worst_t:worst_t+1]                 # (1, H, W)
    contrib, fraction = model.attribution(K_t, emis_t)
    contrib  = contrib[0].numpy()   # (S, H, W)
    fraction = fraction[0].numpy()  # (S, H, W)

    total_contrib = contrib.sum(axis=0)  # (H, W)

    pm25_t  = y_raw[worst_t]
    valid_s = np.where(~np.isnan(pm25_t))[0]
    top2    = valid_s[np.argsort(-pm25_t[valid_s])[:2]]

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)

    kw = dict(origin="lower", extent=[LON[0], LON[-1], LAT[0], LAT[-1]],
              aspect="auto")

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(total_contrib, cmap="hot_r", **kw)
    emis_arr = emis_n[worst_t].numpy()
    frow, fcol = np.where(emis_arr > 0)
    if frow.size:
        ax0.scatter(LON[fcol], LAT[frow], s=2, c="lime", alpha=0.7, label="FIRMS fire")
        ax0.legend(fontsize=7)
    fig.colorbar(im0, ax=ax0, shrink=0.8, label="Contribution (norm.)")
    ax0.set_title(f"Total attribution\n{worst_date}", fontsize=10)
    ax0.set_xlabel("Lon"); ax0.set_ylabel("Lat")

    for col, si in enumerate(top2, start=1):
        ax = fig.add_subplot(gs[col])
        name = stations.iloc[si]["location"][:20]
        im = ax.imshow(fraction[si], cmap="YlOrRd", vmin=0, **kw)
        ax.scatter(stations.iloc[si]["lon"], stations.iloc[si]["lat"],
                   s=120, marker="^", c="blue", zorder=5, label=name)
        ax.legend(fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Source fraction")
        ax.set_title(f"{name}\nPM2.5={pm25_t[si]:.0f} µg/m³", fontsize=10)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

    fig.suptitle(f"CLNO M2 Attribution — {worst_date}  (test year {TEST_YEAR})",
                 fontsize=12)
    out = os.path.join(FIGURES_DIR, "attribution_m2_worst_day.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")


def plot_inversion(model, art, stations, pm25_max):
    y_raw   = art["y_raw"]
    emis_n  = art["emis_n"]
    te_mask = art["te_mask"].numpy()
    K_te    = art["K_te"]
    b_te    = art["b_te"]

    y_te_raw  = y_raw[te_mask]
    y_te_norm = (y_te_raw / pm25_max).clone() if isinstance(y_te_raw, torch.Tensor) \
        else torch.tensor(y_te_raw / pm25_max)

    E_inv = model.invert(K_te, y_te_norm, b=b_te, alpha=0.02)  # (T_te, H, W)

    emis_te = emis_n[te_mask]
    E_true  = emis_te.numpy().flatten()
    E_pred  = E_inv.numpy().flatten()
    mask    = (E_true > 0) | (E_pred > 0)
    corr    = np.corrcoef(E_true[mask], E_pred[mask])[0, 1] if mask.sum() > 2 else 0
    print(f"  Inversion vs FIRMS (test year, nonzero cells): r={corr:.3f}")

    worst_local = int(np.argmax(np.nanmean(y_raw[te_mask].numpy(), axis=1)))
    cube = xr.open_zarr(os.path.join(PROC, "datacube_m2.zarr"))
    worst_t_global = np.where(te_mask)[0][worst_local]
    wdate = pd.Timestamp(cube.time.values[worst_t_global]).strftime("%Y-%m-%d")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    kw = dict(origin="lower", extent=[LON[0], LON[-1], LAT[0], LAT[-1]],
              cmap="hot_r", aspect="auto")
    vmax = max(float(emis_te[worst_local].max()),
               float(E_inv[worst_local].max())) * 1.05 + 1e-6

    im0 = axes[0].imshow(emis_te[worst_local].numpy(), vmin=0, vmax=vmax, **kw)
    axes[0].set_title(f"Actual FIRMS FRP (norm.)\n{wdate}")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Norm. FRP")

    im1 = axes[1].imshow(E_inv[worst_local].numpy(), vmin=0, vmax=vmax, **kw)
    axes[1].set_title(f"Inverted emission (from PM2.5)\n{wdate}")
    axes[1].scatter(stations["lon"], stations["lat"], s=40, marker="^",
                    c="cyan", edgecolors="k", linewidths=0.6, zorder=5)
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Recovered emission")

    for ax in axes:
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    fig.suptitle(f"Emission inversion M2  (r={corr:.3f}, test year {TEST_YEAR})",
                 fontsize=12)

    out = os.path.join(FIGURES_DIR, "inversion_m2_vs_firms.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")


def update_comparison():
    mp = os.path.join(ROOT, "models", "metrics.json")
    d  = json.load(open(mp))
    ORDER  = ["mean_predictor", "xgboost", "mlp", "clno", "clno_m2"]
    LABELS = {"mean_predictor": "Mean\n(dumb)", "xgboost": "XGBoost",
              "mlp": "MLP (GPU)", "clno": "CLNO\n(Phase 0)",
              "clno_m2": "CLNO M2\n(SEA)"}
    models  = [m for m in ORDER if m in d]
    mae     = [d[m]["MAE"]  for m in models]
    rmse    = [d[m]["RMSE"] for m in models]
    clr_mae  = ["#f97316" if "clno" in m else "#9ca3af" for m in models]
    clr_rmse = ["#fbbf24" if "clno" in m else "#d1d5db" for m in models]

    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, mae,  w, label="MAE",  color=clr_mae)
    b2 = ax.bar(x + w/2, rmse, w, label="RMSE", color=clr_rmse)
    ax.bar_label(b1, fmt="%.0f", fontsize=9)
    ax.bar_label(b2, fmt="%.0f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in models])
    ax.set_ylabel("Error (µg/m³)  — lower is better")
    ax.set_title("HazeNet — model comparison (PM2.5, test set)\n"
                 "CLNO M2 = SEA domain 5-year training")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    out = os.path.join(FIGURES_DIR, "model_comparison_m2.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")

    print("\nAll metrics:")
    print(f"  {'model':16} {'MAE':>7} {'RMSE':>7}")
    for m in models:
        marker = " <-- ours" if "clno" in m else ""
        print(f"  {m:16} {d[m]['MAE']:>7} {d[m]['RMSE']:>7}{marker}")


def main():
    print("Loading M2 model + artifacts ...")
    model, art, stations, pm25_max, ckpt = load_everything()

    print("\nAttribution (test year):")
    plot_attribution(model, art, stations, pm25_max, ckpt)

    print("\nEmission inversion:")
    plot_inversion(model, art, stations, pm25_max)

    print("\nUpdating comparison chart:")
    update_comparison()

    print("\nDone — CLNO M2 evaluation complete.")


if __name__ == "__main__":
    main()
