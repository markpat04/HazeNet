"""
Systematic experiment for CLNO M2 — fix overfitting + proper evaluation.

แก้ตามแผน Priority 2 + 3:
  - 3-way temporal split: train 2019-2021 / val 2022 / test 2023
      (เดิมมีแค่ train/test → ทำ early stopping แบบไม่โกงข้อสอบไม่ได้)
  - early stopping จาก val loss (เก็บ weight ที่ดีที่สุด ไม่ใช่รอบสุดท้าย)
  - sweep: architecture × rank × weight_decay × dropout
      เลือกตัวที่ดีที่สุดด้วย VAL MAE แล้วค่อยรายงาน TEST MAE
  - per-station MAE: ดูว่าพลาดหนักที่สถานีไหน

Architectures:
  lowrank  = CLNOLowRank  (V predicted from met, ~23M params)   ← ของเดิม
  globalV  = CLNOGlobalV  (V เป็น global learned, ~0.6M params) ← parsimonious

Output:
  models/clno_m2.pt            (best model by val MAE, overwrite)
  models/clno_m2_artifacts.pt  (test-set K/b for eval)
  reports/sweep_results.csv
  reports/per_station_mae.csv
  figures/sweep_m2.png
  figures/per_station_mae_m2.png

Run: KMP_DUPLICATE_LIB_OK=TRUE conda run -n hazenet --no-capture-output python src/sweep_m2.py
"""
import os, sys, json, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_operator_m2 import CLNOLowRank, CLNOGlobalV
from config_m2 import ROOT, PROC

torch.manual_seed(42)
np.random.seed(42)

TRAIN_YEARS = [2019, 2020, 2021]
VAL_YEAR    = 2022
TEST_YEAR   = 2023
BATCH_SIZE  = 32
MAX_EPOCHS  = 300
PATIENCE    = 40


# ── data: 3-way temporal split ────────────────────────────────────────
def load_data_3way():
    cube  = xr.open_zarr(os.path.join(PROC, "datacube_m2.zarr"))
    X     = cube.X.values                      # (T, 4, H, W)
    T, _, H, W = X.shape
    times = pd.DatetimeIndex(cube.time.values)
    yrs   = times.year.values

    tr_mask = np.isin(yrs, TRAIN_YEARS)
    va_mask = yrs == VAL_YEAR
    te_mask = yrs == TEST_YEAR

    met_raw  = X[:, :-1].astype("float32")    # all channels except last
    emis_raw = X[:,  -1].astype("float32")    # emission = last channel
    in_ch    = met_raw.shape[1]

    # normalise on TRAIN years only (no val/test leakage)
    met_mu  = met_raw[tr_mask].mean(axis=(0, 2, 3), keepdims=True)
    met_std = met_raw[tr_mask].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    met_n   = (met_raw - met_mu) / met_std
    e_max   = float(emis_raw[tr_mask].max()) + 1e-6
    emis_n  = emis_raw / e_max

    # PM2.5 targets
    tgt = pd.read_csv(os.path.join(PROC, "target_pm25_m2.csv"))
    tgt["date"] = pd.to_datetime(tgt["date"])
    stations = (tgt.groupby("locationId")
                   .first()[["location", "lat", "lon", "ilat", "ilon"]]
                   .reset_index()
                   .sort_values("locationId")
                   .reset_index(drop=True))
    S = len(stations)

    pm25_max = float(tgt[tgt["date"].dt.year.isin(TRAIN_YEARS)]["pm25"].max()) + 1e-6
    t_map = {pd.Timestamp(t): i for i, t in enumerate(times)}
    s_map = {sid: i for i, sid in enumerate(stations["locationId"])}

    y_raw = np.full((T, S), np.nan, dtype="float32")
    for _, r in tgt.iterrows():
        ti = t_map.get(r["date"]); si = s_map.get(r["locationId"])
        if ti is not None and si is not None:
            y_raw[ti, si] = r["pm25"]
    y_n = y_raw / pm25_max

    meta = dict(H=H, W=W, S=S, T=T, in_ch=in_ch, e_max=e_max, pm25_max=pm25_max,
                met_mu=met_mu.tolist(), met_std=met_std.tolist())
    return dict(met_n=met_n, emis_n=emis_n, y_n=y_n, y_raw=y_raw,
                stations=stations, meta=meta, times=times,
                tr_mask=tr_mask, va_mask=va_mask, te_mask=te_mask)


def masked_mse(pred, target):
    m = ~torch.isnan(target)
    if m.sum() == 0:
        return pred.sum() * 0
    return F.mse_loss(pred[m], target[m])


def make_loader(d, mask, shuffle, dev):
    idx = np.where(mask)[0]
    met = torch.tensor(d["met_n"][idx])
    emi = torch.tensor(d["emis_n"][idx])
    y   = torch.tensor(d["y_n"][idx])
    ds  = TensorDataset(met, emi, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      pin_memory=(dev == "cuda"))


@torch.no_grad()
def eval_mae(model, d, mask, dev, pm25_max):
    """MAE in real µg/m³ over valid station-days under `mask`."""
    model.eval()
    idx = np.where(mask)[0]
    preds, gts = [], []
    loader = make_loader(d, mask, False, dev)
    for mb_met, mb_emis, _ in loader:
        p, _, _ = model(mb_met.to(dev), mb_emis.to(dev))
        preds.append(p.cpu().numpy())
    pred = np.concatenate(preds) * pm25_max          # (n, S)
    gt   = d["y_raw"][idx]
    m    = ~np.isnan(gt)
    mae  = np.abs(pred[m] - gt[m]).mean()
    rmse = np.sqrt(((pred[m] - gt[m]) ** 2).mean())
    return float(mae), float(rmse)


def build_model(cfg, meta):
    H, W, S, ic = meta["H"], meta["W"], meta["S"], meta["in_ch"]
    if cfg["arch"] == "lowrank":
        return CLNOLowRank(H, W, S, hidden=64, rank=cfg["rank"],
                           dropout=cfg["dropout"], in_ch=ic)
    return CLNOGlobalV(H, W, S, hidden=64, rank=cfg["rank"],
                       dropout=cfg["dropout"], in_ch=ic)


def train_one(cfg, d, dev):
    meta     = d["meta"]; pm25_max = meta["pm25_max"]
    model    = build_model(cfg, meta).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    tr_loader = make_loader(d, d["tr_mask"], True, dev)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3,
                             weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_val = float("inf"); best_state = None; best_ep = 0; wait = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for mb_met, mb_emis, mb_y in tr_loader:
            opt.zero_grad()
            pred, _, _ = model(mb_met.to(dev), mb_emis.to(dev))
            loss = masked_mse(pred, mb_y.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        val_mae, _ = eval_mae(model, d, d["va_mask"], dev, pm25_max)
        if val_mae < best_val - 1e-4:
            best_val   = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_ep    = ep
            wait       = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)        # restore best-val weights
    val_mae,  val_rmse  = eval_mae(model, d, d["va_mask"], dev, pm25_max)
    test_mae, test_rmse = eval_mae(model, d, d["te_mask"], dev, pm25_max)
    tr_mae,   _         = eval_mae(model, d, d["tr_mask"], dev, pm25_max)
    return dict(model=model, n_params=n_params, best_ep=best_ep + 1,
                tr_mae=tr_mae, val_mae=val_mae, val_rmse=val_rmse,
                test_mae=test_mae, test_rmse=test_rmse)


# ── save artifacts for eval_operator_m2 (test set only, keeps file small) ─
@torch.no_grad()
def save_best(res, cfg, d, dev):
    model = res["model"]; meta = d["meta"]
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    torch.save({
        "state_dict":  model.state_dict(),
        "model_class": "CLNOLowRank" if cfg["arch"] == "lowrank" else "CLNOGlobalV",
        "H": meta["H"], "W": meta["W"], "S": meta["S"], "in_ch": meta["in_ch"],
        "hidden": 64, "rank": cfg["rank"], "dropout": cfg["dropout"],
        "stations": d["stations"].to_dict(),
        "meta": {k: v for k, v in meta.items()
                 if not isinstance(v, (np.ndarray, list))},
        "met_mu":  meta["met_mu"], "met_std": meta["met_std"],
        "config":  cfg,
    }, os.path.join(ROOT, "models", "clno_m2.pt"))

    # test-set K/b for attribution + inversion
    model.eval()
    te_idx = np.where(d["te_mask"])[0]
    met = torch.tensor(d["met_n"][te_idx]); emi = torch.tensor(d["emis_n"][te_idx])
    Ks, bs = [], []
    for i in range(0, len(te_idx), BATCH_SIZE):
        p, K, b = model(met[i:i+BATCH_SIZE].to(dev), emi[i:i+BATCH_SIZE].to(dev))
        Ks.append(K.cpu()); bs.append(b.cpu())
    torch.save({
        "K_te":   torch.cat(Ks),
        "b_te":   torch.cat(bs),
        "emis_n": torch.tensor(d["emis_n"]),
        "y_raw":  torch.tensor(d["y_raw"]),
        "te_mask": torch.tensor(d["te_mask"]),
    }, os.path.join(ROOT, "models", "clno_m2_artifacts.pt"))


# ── per-station MAE on test set ───────────────────────────────────────
@torch.no_grad()
def per_station_mae(res, d, dev):
    model = res["model"]; meta = d["meta"]; pm25_max = meta["pm25_max"]
    model.eval()
    te_idx = np.where(d["te_mask"])[0]
    preds = []
    for mb_met, mb_emis, _ in make_loader(d, d["te_mask"], False, dev):
        p, _, _ = model(mb_met.to(dev), mb_emis.to(dev))
        preds.append(p.cpu().numpy())
    pred = np.concatenate(preds) * pm25_max     # (n_te, S)
    gt   = d["y_raw"][te_idx]                    # (n_te, S)

    rows = []
    st = d["stations"]
    for si in range(meta["S"]):
        m = ~np.isnan(gt[:, si])
        if m.sum() == 0:
            continue
        mae = np.abs(pred[m, si] - gt[m, si]).mean()
        rows.append(dict(locationId=st.iloc[si]["locationId"],
                         location=str(st.iloc[si]["location"])[:30],
                         lat=st.iloc[si]["lat"], lon=st.iloc[si]["lon"],
                         n_days=int(m.sum()),
                         obs_mean=float(gt[m, si].mean()),
                         mae=float(mae)))
    df = pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)
    return df


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}"
          + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    d = load_data_3way()
    meta = d["meta"]
    print(f"Grid {meta['H']}×{meta['W']}  G={meta['H']*meta['W']}  S={meta['S']}")
    print(f"split  train(2019-21)={d['tr_mask'].sum()}  "
          f"val(2022)={d['va_mask'].sum()}  test(2023)={d['te_mask'].sum()} days")
    print(f"PM2.5_max(train)={meta['pm25_max']:.0f}\n")

    # ── sweep grid ────────────────────────────────────────────────────
    configs = [
        dict(arch="lowrank", rank=32, wd=1e-4, dropout=0.1),   # ของเดิม (baseline)
        dict(arch="lowrank", rank=16, wd=1e-3, dropout=0.3),
        dict(arch="lowrank", rank=8,  wd=1e-3, dropout=0.3),
        dict(arch="globalV", rank=64, wd=1e-4, dropout=0.2),
        dict(arch="globalV", rank=32, wd=1e-3, dropout=0.2),
        dict(arch="globalV", rank=16, wd=1e-3, dropout=0.3),
    ]

    results = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        r  = train_one(cfg, d, dev)
        dt = time.time() - t0
        tag = f"{cfg['arch']}-r{cfg['rank']}-wd{cfg['wd']:.0e}-dp{cfg['dropout']}"
        print(f"[{i+1}/{len(configs)}] {tag:34}  params={r['n_params']/1e6:5.2f}M  "
              f"ep*={r['best_ep']:3d}  "
              f"train={r['tr_mae']:5.1f}  val={r['val_mae']:5.1f}  "
              f"test={r['test_mae']:5.1f}  ({dt:.0f}s)")
        results.append(dict(tag=tag, **cfg,
                            n_params=r["n_params"], best_ep=r["best_ep"],
                            train_mae=round(r["tr_mae"], 2),
                            val_mae=round(r["val_mae"], 2),
                            val_rmse=round(r["val_rmse"], 2),
                            test_mae=round(r["test_mae"], 2),
                            test_rmse=round(r["test_rmse"], 2),
                            _res=r))

    # ── pick best by VAL mae ──────────────────────────────────────────
    best = min(results, key=lambda x: x["val_mae"])
    print(f"\n*** BEST (by val MAE): {best['tag']}  "
          f"val={best['val_mae']}  test={best['test_mae']} µg/m³ ***")

    # save best model + artifacts
    best_cfg = dict(arch=best["arch"], rank=best["rank"],
                    wd=best["wd"], dropout=best["dropout"])
    save_best(best["_res"], best_cfg, d, dev)
    print("[ok] models/clno_m2.pt + clno_m2_artifacts.pt (best model)")

    # ── save sweep CSV ────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    sweep_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_res"}
                             for r in results])
    sweep_csv = os.path.join(ROOT, "reports", "sweep_results.csv")
    sweep_df.to_csv(sweep_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] {sweep_csv}")

    # ── per-station MAE for best ──────────────────────────────────────
    ps = per_station_mae(best["_res"], d, dev)
    ps_csv = os.path.join(ROOT, "reports", "per_station_mae.csv")
    ps.to_csv(ps_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] {ps_csv}  ({len(ps)} stations on test set)")
    print(f"     worst 5 stations (MAE µg/m³):")
    for _, r in ps.head(5).iterrows():
        print(f"       {r['mae']:6.1f}  obs_mean={r['obs_mean']:5.0f}  "
              f"n={r['n_days']:2d}  {r['location']}")
    print(f"     median per-station MAE = {ps['mae'].median():.1f}")

    # ── update metrics.json ───────────────────────────────────────────
    mp = os.path.join(ROOT, "models", "metrics.json")
    md = json.load(open(mp)) if os.path.exists(mp) else {}
    md["clno_m2"] = dict(MAE=best["test_mae"], RMSE=best["test_rmse"],
                         val_MAE=best["val_mae"],
                         domain="SEA_111x101",
                         arch=best["arch"], rank=best["rank"],
                         years="2019-21_train_2022_val_2023_test")
    json.dump(md, open(mp, "w"), indent=2)
    print("[ok] models/metrics.json updated")

    # ── figures ───────────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

    # (1) sweep comparison
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(results)); w = 0.27
    ax.bar(x - w, [r["train_mae"] for r in results], w, label="train (2019-21)", color="#86efac")
    ax.bar(x,     [r["val_mae"]   for r in results], w, label="val (2022)",      color="#fbbf24")
    ax.bar(x + w, [r["test_mae"]  for r in results], w, label="test (2023)",     color="#f97316")
    best_i = results.index(best)
    ax.axvline(best_i, color="#3b82f6", ls="--", lw=1.2, alpha=0.7)
    ax.text(best_i, ax.get_ylim()[1]*0.95, "best\n(by val)", color="#3b82f6",
            ha="center", va="top", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r["tag"] for r in results], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MAE (µg/m³) — lower better")
    ax.set_title("CLNO M2 sweep — train/val/test MAE per config")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(ROOT, "figures", "sweep_m2.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[ok] figures/sweep_m2.png")

    # (2) per-station MAE bar
    fig, ax = plt.subplots(figsize=(10, 6))
    topn = ps.head(25)
    colors = ["#ef4444" if v >= 14.5 else "#3b82f6" for v in topn["lat"]]
    ax.barh(range(len(topn)), topn["mae"], color=colors)
    ax.set_yticks(range(len(topn)))
    ax.set_yticklabels([f"{l[:22]} ({la:.1f}N)"
                        for l, la in zip(topn["location"], topn["lat"])], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE (µg/m³)")
    ax.set_title("CLNO M2 — worst 25 stations on test set (2023)\n"
                 "red ≥14.5°N (north/border) · blue lower lat")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(os.path.join(ROOT, "figures", "per_station_mae_m2.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[ok] figures/per_station_mae_m2.png")

    print("\nDone — sweep complete.")


if __name__ == "__main__":
    main()
