"""
Training — AMP, checkpoint/resume, masked MSE or quantile (pinball) loss,
optional live logging to the dashboard tracker.

Resumes automatically from <ckpt>.resume.pt if present (RunPod pods can drop).
"""
from __future__ import annotations

import os
import json

import numpy as np

from .dataset import load_dataset

# optional dashboard live tracking
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "hazenet_dashboard"))
    from tracker import Run
except Exception:
    Run = None


def _resume_path(cfg):
    return cfg.ckpt_path.replace(".pt", ".resume.pt")


def train(cfg) -> dict:
    # load zarr BEFORE importing torch (Windows OpenMP/pyarrow segfault prevention)
    d = load_dataset(cfg)
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from .model import build_model, masked_mse, pinball_loss
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg.amp and dev == "cuda"
    print(f"device: {dev}  amp={use_amp}")
    meta = d["meta"]; S, in_ch = meta["S"], meta["in_ch"]
    pm25_max = meta["pm25_max"]
    print(f"grid {meta['H']}×{meta['W']}  G={meta['H']*meta['W']}  "
          f"stations={S}  in_ch={in_ch}  days={meta['T']}")

    model = build_model(cfg, in_ch=in_ch).to(dev)
    sfeats_t = torch.tensor(d["station_feats"]).to(dev)    # (S, 4) — fixed per run
    n_p = sum(p.numel() for p in model.parameters())
    print(f"model={cfg.model_kind}  params={n_p:,}  quantiles={cfg.quantiles}")

    met_t = torch.tensor(d["met"]); emis_t = torch.tensor(d["emis"])
    y_t = torch.tensor(d["y_norm"])
    tr_idx = np.where(d["train_mask"])[0]; te_idx = np.where(d["test_mask"])[0]
    tr_ds = TensorDataset(met_t[tr_idx], emis_t[tr_idx], y_t[tr_idx])
    te_ds = TensorDataset(met_t[te_idx], emis_t[te_idx], y_t[te_idx])
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           pin_memory=(dev == "cuda"), num_workers=cfg.num_workers)
    te_loader = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False,
                           pin_memory=(dev == "cuda"))

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def loss_fn(out, yb):
        if cfg.quantiles:
            return pinball_loss(out, yb, cfg.quantiles)
        return masked_mse(out, yb)

    # ── resume ──
    start_ep = 0; tr_losses, te_losses = [], []
    rp = _resume_path(cfg)
    if os.path.exists(rp):
        ck = torch.load(rp, map_location=dev)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start_ep = ck["epoch"] + 1
        tr_losses, te_losses = ck["tr_losses"], ck["te_losses"]
        print(f"[resume] from epoch {start_ep}")

    run = Run(model=cfg.model_kind,
              config={"name": cfg.name, "rank": cfg.rank, "hidden": cfg.hidden,
                      "lr": cfg.lr, "epochs": cfg.epochs, "params": int(n_p),
                      "quantiles": cfg.quantiles, "curve": cfg.emission_curve}) \
        if Run else None

    os.makedirs(cfg.models_dir, exist_ok=True)
    import copy
    best_te = min(te_losses) if te_losses else float("inf")
    best_state, bad = None, 0
    print(f"training {cfg.epochs} epochs (from {start_ep}) "
          f"patience={cfg.patience or 'off'} ...")
    for ep in range(start_ep, cfg.epochs):
        model.train(); tot = 0.0
        for mb in tr_loader:
            mm, me, my = (x.to(dev) for x in mb)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out, _, _ = model(mm, me, sfeats_t); loss = loss_fn(out, my)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step(); tr_losses.append(tot / max(1, len(tr_loader)))

        model.eval(); tot = 0.0
        with torch.no_grad():
            for mb in te_loader:
                mm, me, my = (x.to(dev) for x in mb)
                out, _, _ = model(mm, me, sfeats_t); tot += loss_fn(out, my).item()
        te_losses.append(tot / max(1, len(te_loader)))

        if run:
            run.log_epoch(ep, train=tr_losses[-1], test=te_losses[-1],
                          lr=sched.get_last_lr()[0])
        if (ep + 1) % 20 == 0 or ep == start_ep:
            print(f"  ep {ep+1:3d}  train={tr_losses[-1]:.4f}  test={te_losses[-1]:.4f}")

        torch.save(dict(state_dict=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep,
                        tr_losses=tr_losses, te_losses=te_losses), rp)

        # early stopping on test loss (keep best weights)
        if te_losses[-1] < best_te - 1e-5:
            best_te, best_state, bad = te_losses[-1], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                print(f"  early stop @ ep {ep+1}  (best test={best_te:.4f}, "
                      f"no improve {bad} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[early-stop] restored best weights (test loss {best_te:.4f})")

    # ── final metrics (median path) ──
    model.eval(); preds = []
    with torch.no_grad():
        for mb in DataLoader(te_ds, batch_size=cfg.batch_size):
            mm, me, _ = (x.to(dev) for x in mb)
            out, _, _ = model(mm, me, sfeats_t)
            preds.append(model.predict_median(out).cpu())
    pred_te = torch.cat(preds).numpy() * pm25_max
    gt_te = d["y_raw"][d["test_mask"]]
    m = ~np.isnan(gt_te)
    mae = float(np.abs(pred_te[m] - gt_te[m]).mean())
    rmse = float(np.sqrt(((pred_te[m] - gt_te[m]) ** 2).mean()))
    print(f"\n[{cfg.name}] test MAE={mae:.1f}  RMSE={rmse:.1f} µg/m³")

    torch.save(dict(state_dict=model.state_dict(), cfg_name=cfg.name,
                    model_kind=cfg.model_kind, in_ch=in_ch,
                    hidden=cfg.hidden, rank=cfg.rank, quantiles=cfg.quantiles,
                    sfeat_hidden=cfg.sfeat_hidden, n_sfeats=4,
                    emission_curve=cfg.emission_curve,
                    H=meta["H"], W=meta["W"], S=S,
                    station_feats=d["station_feats"],
                    stations=d["stations"].to_dict(),
                    meta={k: v for k, v in meta.items()
                          if not isinstance(v, (np.ndarray, list))},
                    met_mu=meta["met_mu"], met_std=meta["met_std"],
                    channels=meta["channels"]), cfg.ckpt_path)
    print(f"[ok] {cfg.ckpt_path}")

    mp = os.path.join(cfg.models_dir, "metrics.json")
    dd = json.load(open(mp)) if os.path.exists(mp) else {}
    dd[f"clno_{cfg.name}"] = dict(MAE=round(mae, 2), RMSE=round(rmse, 2),
                                  model=cfg.model_kind, params=int(n_p),
                                  channels=meta["channels"])
    json.dump(dd, open(mp, "w"), indent=2)

    if run:
        run.finish(metrics={"MAE": mae, "RMSE": rmse}, status="done")
    if os.path.exists(rp):
        os.remove(rp)   # clear resume marker on clean finish
    return dict(MAE=mae, RMSE=rmse)
