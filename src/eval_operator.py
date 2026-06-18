"""
Evaluate CLNO: Attribution maps + Emission inversion + Comparison with FIRMS.

Loads trained CLNO from models/clno_artifacts.pt and generates:
  figures/attribution_worst_day.png   — for the worst haze day, which fires caused it?
  figures/inversion_vs_firms.png      — recovered emission field vs actual FIRMS
  figures/model_comparison.png        — all models updated with CLNO

Run: KMP_DUPLICATE_LIB_OK=TRUE conda run -n hazenet --no-capture-output python src/eval_operator.py
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
from model_operator import CLNO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
LAT = np.round(np.arange(18.0, 19.5 + 1e-6, 0.1), 1)
LON = np.round(np.arange(98.0, 100.0 + 1e-6, 0.1), 1)


# ── helpers ───────────────────────────────────────────────────────────
def load_model_and_artifacts():
    ckpt = torch.load(os.path.join(ROOT, "models", "clno.pt"),
                      map_location="cpu", weights_only=False)
    art  = torch.load(os.path.join(ROOT, "models", "clno_artifacts.pt"),
                      map_location="cpu", weights_only=False)

    model = CLNO(H=ckpt["H"], W=ckpt["W"],
                 n_stations=ckpt["S"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    stations = pd.DataFrame(ckpt["stations"])
    pm25_max = ckpt["meta"]["pm25_max"]

    return model, art, stations, pm25_max, ckpt


def plot_attribution(model, art, stations, pm25_max, ckpt):
    """
    Worst-haze-day attribution: for the day with highest mean PM2.5,
    show which source grid cells contributed most to each station.
    """
    y_raw = art["y_raw"].numpy()          # (T, S)
    emis  = art["emis_n"]                 # (T, H, W)
    K_all = torch.cat([art["K_tr"], art["K_te"]], dim=0)   # (T, S, G)
    b_all = torch.cat([art["b_tr"], art["b_te"]], dim=0)   # (T, S)

    # Worst day = highest mean observed PM2.5
    day_mean = np.nanmean(y_raw, axis=1)
    worst_t  = int(np.argmax(day_mean))
    print(f"  Worst haze day: t={worst_t}  avg PM2.5={day_mean[worst_t]:.0f} ug/m3")

    cube = xr.open_zarr(os.path.join(PROC, "datacube.zarr"))
    worst_date = pd.Timestamp(cube.time.values[worst_t]).strftime("%Y-%m-%d")

    K_t    = K_all[worst_t:worst_t+1]   # (1, S, G)
    emis_t = emis[worst_t:worst_t+1]    # (1, H, W)
    contrib, fraction = model.attribution(K_t, emis_t)
    contrib  = contrib[0].numpy()   # (S, H, W)
    fraction = fraction[0].numpy()  # (S, H, W)

    # Show total attribution summed over all stations → single map
    total_contrib = contrib.sum(axis=0)  # (H, W) — total contribution to all receptors

    # Also show top-2 stations by PM2.5 individually
    pm25_t = y_raw[worst_t]             # (S,)
    valid_sta = np.where(~np.isnan(pm25_t))[0]
    top2 = valid_sta[np.argsort(-pm25_t[valid_sta])[:2]]

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)

    # Panel 1: summed attribution
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(total_contrib, origin="lower",
                     extent=[LON[0], LON[-1], LAT[0], LAT[-1]],
                     cmap="hot_r", aspect="auto")
    # overlay FIRMS fire points for this day
    firms_t_mask = (art["emis_n"][worst_t].numpy() > 0)
    flon, flat = np.where(firms_t_mask)
    if flon.size:
        ax0.scatter(LON[flat], LAT[flon], s=4, c="lime", alpha=0.8,
                    label="FIRMS fire")
        ax0.legend(fontsize=7)
    fig.colorbar(im0, ax=ax0, shrink=0.8, label="Contribution (norm.)")
    ax0.set_title(f"Total attribution\n{worst_date}", fontsize=10)
    ax0.set_xlabel("Lon"); ax0.set_ylabel("Lat")

    # Panel 2 & 3: top-2 stations individually
    for col, si in enumerate(top2, start=1):
        ax = fig.add_subplot(gs[col])
        sta_name = stations.iloc[si]["location"].split(",")[0][:16]
        im = ax.imshow(fraction[si], origin="lower",
                       extent=[LON[0], LON[-1], LAT[0], LAT[-1]],
                       cmap="YlOrRd", vmin=0, aspect="auto")
        # mark the station itself
        ax.scatter(stations.iloc[si]["lon"], stations.iloc[si]["lat"],
                   s=120, marker="^", c="blue", zorder=5, label=sta_name)
        ax.legend(fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Source fraction")
        ax.set_title(f"{sta_name}\nPM2.5={pm25_t[si]:.0f} µg/m³", fontsize=10)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

    fig.suptitle(f"CLNO Attribution — {worst_date}  (worst haze day)", fontsize=12)
    out = os.path.join(ROOT, "figures", "attribution_worst_day.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")


def plot_inversion(model, art, stations, pm25_max):
    """
    Run emission inversion for every day.
    Compare recovered emission with actual FIRMS emission.
    """
    y_raw = art["y_raw"]                  # (T, S) tensor
    emis  = art["emis_n"]                 # (T, H, W) tensor  — normalised
    K_all = torch.cat([art["K_tr"], art["K_te"]], dim=0)  # (T, S, G)
    b_all = torch.cat([art["b_tr"], art["b_te"]], dim=0)  # (T, S)

    T = K_all.shape[0]

    # Invert using observed PM2.5 (normalise: divide by pm25_max)
    y_norm = (y_raw / pm25_max).clone()
    E_inv  = model.invert(K_all, y_norm, b=b_all, alpha=0.02)  # (T, H, W)

    # Compare: flatten all cells + days, mask zeros in both
    E_true = emis.numpy().flatten()  # normalised FIRMS
    E_pred = E_inv.numpy().flatten()
    mask   = (E_true > 0) | (E_pred > 0)
    corr   = np.corrcoef(E_true[mask], E_pred[mask])[0, 1]
    print(f"  Inversion vs FIRMS correlation (nonzero cells): r={corr:.3f}")

    # Pick worst day for spatial plot
    y_np    = y_raw.numpy()
    worst_t = int(np.argmax(np.nanmean(y_np, axis=1)))
    cube    = xr.open_zarr(os.path.join(PROC, "datacube.zarr"))
    wdate   = pd.Timestamp(cube.time.values[worst_t]).strftime("%Y-%m-%d")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    kw = dict(origin="lower", extent=[LON[0], LON[-1], LAT[0], LAT[-1]],
              cmap="hot_r", aspect="auto")

    vmax = max(emis[worst_t].numpy().max(), E_inv[worst_t].numpy().max()) * 1.05 + 1e-6

    im0 = axes[0].imshow(emis[worst_t].numpy(), vmin=0, vmax=vmax, **kw)
    axes[0].set_title(f"Actual FIRMS FRP (normalised)\n{wdate}")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Normalised FRP")

    im1 = axes[1].imshow(E_inv[worst_t].numpy(), vmin=0, vmax=vmax, **kw)
    axes[1].set_title(f"Inverted emission (from PM2.5 obs)\n{wdate}")
    # overlay station positions
    axes[1].scatter(stations["lon"], stations["lat"], s=60, marker="^",
                    c="cyan", edgecolors="black", linewidths=0.8, zorder=5)
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Recovered emission")

    fig.suptitle(f"Emission inversion  (Pearson r={corr:.3f} across all days)",
                 fontsize=12)
    for ax in axes:
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

    out = os.path.join(ROOT, "figures", "inversion_vs_firms.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")


def update_comparison():
    """Regenerate model_comparison.png with CLNO added."""
    mp = os.path.join(ROOT, "models", "metrics.json")
    d  = json.load(open(mp))
    ORDER  = ["mean_predictor", "xgboost", "mlp", "clno"]
    LABELS = {"mean_predictor": "Mean\n(dumb)", "xgboost": "XGBoost",
              "mlp": "MLP (GPU)", "clno": "CLNO\n(ours)"}
    models = [m for m in ORDER if m in d]
    mae    = [d[m]["MAE"]  for m in models]
    rmse   = [d[m]["RMSE"] for m in models]
    colors_mae  = ["#9ca3af" if m != "clno" else "#f97316" for m in models]
    colors_rmse = ["#d1d5db" if m != "clno" else "#fbbf24" for m in models]

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, mae,  w, label="MAE",  color=colors_mae)
    b2 = ax.bar(x + w/2, rmse, w, label="RMSE", color=colors_rmse)
    ax.bar_label(b1, fmt="%.0f", fontsize=9)
    ax.bar_label(b2, fmt="%.0f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in models])
    ax.set_ylabel("Error (µg/m³)  — lower is better")
    ax.set_title("HazeNet Phase 0 — model comparison (PM2.5, test set)\n"
                 "CLNO = Conditionally-Linear Neural Operator (our model)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    out = os.path.join(ROOT, "figures", "model_comparison.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")

    print("\nFinal metrics:")
    print(f"  {'model':16} {'MAE':>7} {'RMSE':>7}")
    for m in models:
        marker = " <-- ours" if m == "clno" else ""
        print(f"  {m:16} {d[m]['MAE']:>7} {d[m]['RMSE']:>7}{marker}")


# ─────────────────────────────────────────────
def main():
    print("Loading model + artifacts...")
    model, art, stations, pm25_max, ckpt = load_model_and_artifacts()

    print("\nAttribution:")
    plot_attribution(model, art, stations, pm25_max, ckpt)

    print("\nEmission inversion:")
    plot_inversion(model, art, stations, pm25_max)

    print("\nUpdating comparison chart:")
    update_comparison()

    print("\nDone — Phase 0 CLNO evaluation complete.")


if __name__ == "__main__":
    main()
