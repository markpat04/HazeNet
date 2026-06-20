"""
Evaluate a trained model: test MAE/RMSE, per-year bias (the W2 gate), and a
pred-vs-obs scatter. Per-year bias is the headline diagnostic for the
non-stationarity problem.
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

from .dataset import load_dataset

# matplotlib is imported lazily inside the plotting helpers — it must come AFTER
# torch on Windows (Agg-backend DLL ordering crash otherwise; see loyo.py).

# Gate W2: low-dust-year positive bias must shrink to ≤ this (µg/m³)
GATE_W2_BIAS = 25.0


def evaluate(cfg) -> dict:
    # load zarr BEFORE importing torch (Windows OpenMP/pyarrow segfault prevention)
    d = load_dataset(cfg)
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from .infer import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(cfg.ckpt_path, dev)
    pm25_max = d["meta"]["pm25_max"]

    # station features: prefer from checkpoint (matches training), fall back to dataset
    sf = ck.get("station_feats", d["station_feats"])
    sfeats_t = torch.tensor(np.asarray(sf, dtype="float32")).to(dev)

    met_t = torch.tensor(d["met"]); emis_t = torch.tensor(d["emis"])
    has_wind = d.get("a_wind") is not None
    if has_wind:
        ds = TensorDataset(met_t, emis_t, torch.tensor(d["a_wind"]))
    else:
        ds = TensorDataset(met_t, emis_t)
    preds = []
    with torch.no_grad():
        for mb in DataLoader(ds, batch_size=cfg.batch_size):
            if has_wind:
                mm, me, mw = (x.to(dev) for x in mb)
            else:
                mm, me = (x.to(dev) for x in mb); mw = None
            out, _, _ = model(mm, me, sfeats_t, mw)
            preds.append(model.predict_median(out).cpu())
    pred = torch.cat(preds).numpy() * pm25_max          # (T,S)
    y = d["y_raw"]; times = d["times"]

    def metrics(mask):
        p, g = pred[mask], y[mask]; m = ~np.isnan(g)
        if m.sum() == 0:
            return None
        return dict(MAE=float(np.abs(p[m] - g[m]).mean()),
                    RMSE=float(np.sqrt(((p[m] - g[m]) ** 2).mean())),
                    bias=float((p[m] - g[m]).mean()), n=int(m.sum()))

    te = metrics(d["test_mask"]); tr = metrics(d["train_mask"])
    print(f"train {tr}\ntest  {te}")

    # per-year bias
    years = np.array([t.year for t in times])
    rows = []
    for yv in sorted(set(years)):
        r = metrics(years == yv)
        if r:
            r["year"] = yv; r["obs_mean"] = float(np.nanmean(y[years == yv]))
            rows.append(r)
    per_year = pd.DataFrame(rows)
    print("\nper-year:\n", per_year[["year", "obs_mean", "bias", "MAE"]].to_string(index=False))

    # Gate W2: bias on the low-dust years (below median obs_mean)
    low = per_year[per_year["obs_mean"] < per_year["obs_mean"].median()]
    worst_low_bias = float(low["bias"].abs().max()) if len(low) else float("nan")
    gate_pass = worst_low_bias <= GATE_W2_BIAS
    print(f"\nGATE W2: worst low-dust-year |bias|={worst_low_bias:.1f} "
          f"(target ≤{GATE_W2_BIAS}) → {'PASS ✅' if gate_pass else 'FAIL ❌'}")

    # Save metrics FIRST (figures can crash on some Windows matplotlib builds).
    out = dict(test=te, train=tr, gate_w2_pass=bool(gate_pass),
               worst_low_bias=worst_low_bias,
               per_year=per_year.to_dict(orient="records"))
    os.makedirs(cfg.models_dir, exist_ok=True)
    json.dump(out, open(os.path.join(cfg.models_dir, f"eval_{cfg.name}.json"), "w"),
              indent=2, default=float)

    # Figures are best-effort: some Windows matplotlib builds SEGFAULT in the Agg
    # renderer (uncatchable by try/except). HAZENET_NOFIG=1 skips them so the
    # pipeline always finishes with metrics; figures can be made on Linux/RunPod.
    if not os.environ.get("HAZENET_NOFIG"):
        os.makedirs(cfg.figures_dir, exist_ok=True)
        try:
            _scatter(cfg, pred, y, d["test_mask"], te)
            _bias_bars(cfg, per_year)
        except Exception as e:
            print(f"[eval] figure save failed (non-fatal): {e}")
    return out


def _scatter(cfg, pred, y, te_mask, te):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, g = pred[te_mask], y[te_mask]; m = ~np.isnan(g)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(g[m], p[m], alpha=0.4, s=8, c="tab:orange")
    lim = [0, float(max(g[m].max(), p[m].max())) * 1.1]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Observed PM2.5 (µg/m³)"); ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.set_title(f"{cfg.name} test — MAE={te['MAE']:.1f} RMSE={te['RMSE']:.1f}")
    fig.savefig(os.path.join(cfg.figures_dir, f"{cfg.name}_pred_vs_true.png"),
                dpi=130, bbox_inches="tight"); plt.close(fig)


def _bias_bars(cfg, per_year):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    c = ["tab:red" if b > 0 else "tab:blue" for b in per_year["bias"]]
    ax.bar(per_year["year"].astype(str), per_year["bias"], color=c)
    ax.axhline(GATE_W2_BIAS, ls="--", c="k", lw=1, label=f"gate ±{GATE_W2_BIAS}")
    ax.axhline(-GATE_W2_BIAS, ls="--", c="k", lw=1)
    ax.set_ylabel("bias = pred − obs (µg/m³)"); ax.set_title(f"{cfg.name} per-year bias")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.savefig(os.path.join(cfg.figures_dir, f"{cfg.name}_year_bias.png"),
                dpi=130, bbox_inches="tight"); plt.close(fig)
