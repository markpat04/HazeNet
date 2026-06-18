"""
Train CLNO on M2 datacube (SEA domain, 2019-2023 burning seasons).

Key differences from Phase 0:
  - H=111, W=101, G=11211 (vs 16×21=336)
  - T~449 days across 5 years
  - Train/test split BY YEAR: 2019-2022 train, 2023 test
  - Mini-batch DataLoader (full T can overflow GPU RAM)
  - Larger CNN encoder (pool to 8×8 instead of 4×4)
  - K_head is now the largest layer: hidden → S × G (e.g. 64 → 40×11211)
    If this is too large, enable --low-rank mode (K = U @ V via two small heads)

Output: models/clno_m2.pt, models/clno_m2_artifacts.pt

Run: KMP_DUPLICATE_LIB_OK=TRUE conda run -n hazenet --no-capture-output python src/train_operator_m2.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from model_operator import CLNO
from config_m2 import ROOT, PROC, TEST_YEAR

# optional: live tracking สำหรับ dashboard (ไม่บังคับ — ถ้าไม่มีก็ข้าม)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "hazenet_dashboard"))
    from tracker import Run
except Exception:
    Run = None

torch.manual_seed(42)
np.random.seed(42)


# ── Low-rank CLNO for large G ──────────────────────────────────────────
class CLNOLowRank(nn.Module):
    """
    CLNO variant where K = softmax(U) @ V.T  with rank r << G.

    U: (B, S, r)  V: (B, G, r)  -> K: (B, S, G)
    Reduces K_head params from hidden×S×G to hidden×(S+G)×r

    For M2: G=11211, S=40, r=32 -> params = 64×(40+11211)×32 = 23M vs 64×40×11211 = 28M
    Net saving is modest, but memory saving is large (K itself is smaller in computation).
    """
    def __init__(self, H: int, W: int, n_stations: int,
                 hidden: int = 64, rank: int = 32, dropout: float = 0.1,
                 in_ch: int = 3):
        super().__init__()
        self.H, self.W  = H, W
        self.G          = H * W
        self.S          = n_stations
        self.rank       = rank
        self.in_ch      = in_ch

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),   # -> (32, 8, 8) = 2048
            nn.Flatten(),
            nn.Linear(2048, hidden), nn.GELU(),
            nn.Dropout(dropout),
        )

        # U: station basis (small)   V: spatial basis (large, shared)
        self.U_head = nn.Linear(hidden, n_stations * rank)   # (B, S*r)
        self.V_head = nn.Linear(hidden, self.G * rank)       # (B, G*r)
        self.b_head = nn.Linear(hidden, n_stations)

    def _build_K(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        U = self.U_head(h).view(B, self.S, self.rank)        # (B, S, r)
        V = self.V_head(h).view(B, self.G, self.rank)        # (B, G, r)
        # K_sg = Σ_r U_sr * V_gr  -> (B, S, G)
        K_logit = torch.bmm(U, V.transpose(1, 2))            # (B, S, G)
        return torch.softmax(K_logit, dim=-1)                 # (B, S, G)

    def forward(self, met, emission):
        B  = met.shape[0]
        h  = self.encoder(met)
        K  = self._build_K(h)                                 # (B, S, G)
        b  = F.softplus(self.b_head(h))                       # (B, S)
        E  = emission.view(B, self.G)
        pm25 = torch.bmm(K, E.unsqueeze(-1)).squeeze(-1) + b
        return pm25, K, b

    @torch.no_grad()
    def attribution(self, K, emission):
        E       = emission.view(emission.shape[0], self.G)
        contrib = K * E.unsqueeze(1)
        total   = contrib.sum(dim=-1, keepdim=True).clamp(1e-8)
        frac    = contrib / total
        shape   = (K.shape[0], self.S, self.H, self.W)
        return contrib.view(shape), frac.view(shape)

    @torch.no_grad()
    def invert(self, K, pm25_obs, b=None, alpha=0.01):
        """Dual-form inversion (S×S solve, efficient when G >> S)."""
        results = []
        for i in range(K.shape[0]):
            Ki = K[i]; yi = pm25_obs[i].clone()
            if b is not None:
                yi = yi - b[i]
            valid = ~torch.isnan(yi)
            Ki_v, yi_v = Ki[valid], yi[valid]
            Sv = valid.sum().item()
            if Sv < 2:
                results.append(torch.zeros(self.G, device=K.device))
                continue
            M     = Ki_v @ Ki_v.T + alpha * torch.eye(Sv, device=K.device)
            lam   = torch.linalg.solve(M, yi_v)
            E_hat = F.relu(Ki_v.T @ lam)
            results.append(E_hat)
        return torch.stack(results).view(-1, self.H, self.W)


# ── Global-basis CLNO: V is a SHARED learned parameter, not predicted ─────
class CLNOGlobalV(nn.Module):
    """
    Parsimonious CLNO. The spatial basis V is a single GLOBAL learned tensor
    (not regenerated from met each sample). Only the station weights U and
    background b depend on weather.

        K = softmax_G( U(met) @ V_globalᵀ )

    Params drop massively (V_head: hidden×G×r  →  V_global: G×r):
      M2 G=11211, S=99, r=32, hidden=64
        LowRank : 64×(99+11211)×32  ≈ 23.2M
        GlobalV : 64×99×32 + 11211×32 ≈ 0.56M   (≈40× smaller)

    Rationale: spatial transport "modes" are physically stable; only how each
    station weights them changes day to day. Far less prone to overfitting.
    """
    def __init__(self, H: int, W: int, n_stations: int,
                 hidden: int = 64, rank: int = 32, dropout: float = 0.2,
                 in_ch: int = 3):
        super().__init__()
        self.H, self.W = H, W
        self.G         = H * W
        self.S         = n_stations
        self.rank      = rank
        self.in_ch     = in_ch

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(2048, hidden), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.U_head   = nn.Linear(hidden, n_stations * rank)   # (B, S*r)
        self.V_global = nn.Parameter(torch.randn(self.G, rank) * 0.02)
        self.b_head   = nn.Linear(hidden, n_stations)

    def _build_K(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        U = self.U_head(h).view(B, self.S, self.rank)          # (B, S, r)
        # K_sg = Σ_r U_sr * Vg_gr  -> (B, S, G)
        K_logit = torch.einsum("bsr,gr->bsg", U, self.V_global)
        return torch.softmax(K_logit, dim=-1)

    def forward(self, met, emission):
        B  = met.shape[0]
        h  = self.encoder(met)
        K  = self._build_K(h)
        b  = F.softplus(self.b_head(h))
        E  = emission.view(B, self.G)
        pm25 = torch.bmm(K, E.unsqueeze(-1)).squeeze(-1) + b
        return pm25, K, b

    # attribution / invert are identical to CLNOLowRank
    attribution = CLNOLowRank.attribution
    invert      = CLNOLowRank.invert


# ─────────────────────────────────────────────
def masked_mse(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0
    return F.mse_loss(pred[mask], target[mask])


def load_data():
    cube = xr.open_zarr(os.path.join(PROC, "datacube_m2.zarr"))
    X    = cube.X.values                             # (T, 4, H, W)
    T, _, H, W = X.shape
    times = pd.DatetimeIndex(cube.time.values)
    train_mask = cube.train_mask.values.astype(bool) # (T,)
    test_mask  = cube.test_mask.values.astype(bool)

    met_raw  = X[:, :3].astype("float32")
    emis_raw = X[:,  3].astype("float32")

    # Normalise on TRAIN set only -> prevent test leakage
    met_mu   = met_raw[train_mask].mean(axis=(0, 2, 3), keepdims=True)
    met_std  = met_raw[train_mask].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    met_norm = (met_raw - met_mu) / met_std

    e_max    = float(emis_raw[train_mask].max()) + 1e-6
    emis_norm = emis_raw / e_max

    # PM2.5 target matrix
    tgt = pd.read_csv(os.path.join(PROC, "target_pm25_m2.csv"))
    tgt["date"] = pd.to_datetime(tgt["date"])

    stations = (tgt.groupby("locationId")
                   .first()[["location", "lat", "lon", "ilat", "ilon"]]
                   .reset_index()
                   .sort_values("locationId")
                   .reset_index(drop=True))
    S = len(stations)

    pm25_max = float(tgt[tgt["date"].dt.year != TEST_YEAR]["pm25"].max()) + 1e-6
    t_map    = {pd.Timestamp(t): i for i, t in enumerate(times)}
    s_map    = {sid: i for i, sid in enumerate(stations["locationId"])}

    y_raw  = np.full((T, S), np.nan, dtype="float32")
    for _, r in tgt.iterrows():
        ti = t_map.get(r["date"])
        si = s_map.get(r["locationId"])
        if ti is not None and si is not None:
            y_raw[ti, si] = r["pm25"]

    y_norm = y_raw / pm25_max

    meta = dict(H=H, W=W, S=S, T=T, e_max=e_max, pm25_max=pm25_max,
                met_mu=met_mu.tolist(), met_std=met_std.tolist())
    return (met_norm, emis_norm, y_norm, y_raw,
            stations, meta, train_mask, test_mask)


# ─────────────────────────────────────────────
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}"
          + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    (met_n, emis_n, y_n, y_raw,
     stations, meta, tr_mask, te_mask) = load_data()

    H, W, S, T = meta["H"], meta["W"], meta["S"], meta["T"]
    pm25_max   = meta["pm25_max"]
    print(f"Grid {H}×{W}  G={H*W}  stations={S}  days={T}")
    print(f"Train days={tr_mask.sum()}  Test days={te_mask.sum()} (year {TEST_YEAR})")
    print(f"PM2.5  mean={np.nanmean(y_raw[tr_mask]):.1f}  "
          f"max={np.nanmax(y_raw[tr_mask]):.1f} µg/m³")

    # Use low-rank CLNO for large G
    RANK   = 32
    HIDDEN = 64
    model  = CLNOLowRank(H=H, W=W, n_stations=S, hidden=HIDDEN, rank=RANK).to(dev)
    n_p    = sum(p.numel() for p in model.parameters())
    print(f"CLNOLowRank  hidden={HIDDEN}  rank={RANK}  params={n_p:,}")

    run = Run(model="CLNOLowRank",
              config={"rank": RANK, "hidden": HIDDEN, "lr": 1e-3,
                      "epochs": 200, "domain": "SEA_111x101",
                      "params": int(n_p)}) if Run else None

    # DataLoader — mini-batch by day
    met_t  = torch.tensor(met_n,  dtype=torch.float32)
    emis_t = torch.tensor(emis_n, dtype=torch.float32)
    y_t    = torch.tensor(y_n,    dtype=torch.float32)

    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]

    tr_ds = TensorDataset(met_t[tr_idx], emis_t[tr_idx], y_t[tr_idx])
    te_ds = TensorDataset(met_t[te_idx], emis_t[te_idx], y_t[te_idx])

    BATCH_SIZE = 32
    tr_loader  = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                            pin_memory=(dev == "cuda"))
    te_loader  = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=(dev == "cuda"))

    EPOCHS = 200
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    tr_losses, te_losses = [], []
    print(f"\nTraining {EPOCHS} epochs  batch={BATCH_SIZE} ...")

    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for mb_met, mb_emis, mb_y in tr_loader:
            mb_met, mb_emis, mb_y = (mb_met.to(dev),
                                     mb_emis.to(dev),
                                     mb_y.to(dev))
            opt.zero_grad()
            pred, _, _ = model(mb_met, mb_emis)
            loss = masked_mse(pred, mb_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        tr_losses.append(ep_loss / len(tr_loader))

        model.eval()
        te_loss = 0.0
        with torch.no_grad():
            for mb_met, mb_emis, mb_y in te_loader:
                mb_met, mb_emis, mb_y = (mb_met.to(dev),
                                         mb_emis.to(dev),
                                         mb_y.to(dev))
                pred, _, _ = model(mb_met, mb_emis)
                te_loss += masked_mse(pred, mb_y).item()
        te_losses.append(te_loss / len(te_loader))

        if run:
            run.log_epoch(ep, train=tr_losses[-1], test=te_losses[-1],
                          lr=sched.get_last_lr()[0])

        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:3d}  train={tr_losses[-1]:.4f}  "
                  f"test={te_losses[-1]:.4f}")

    # ── Full evaluation ───────────────────────────────────────────────
    model.eval()
    all_pred_tr, all_pred_te = [], []
    all_K_tr, all_K_te = [], []
    all_b_tr, all_b_te = [], []

    with torch.no_grad():
        for mb_met, mb_emis, mb_y in DataLoader(tr_ds, batch_size=BATCH_SIZE):
            p, K, b = model(mb_met.to(dev), mb_emis.to(dev))
            all_pred_tr.append(p.cpu()); all_K_tr.append(K.cpu()); all_b_tr.append(b.cpu())
        for mb_met, mb_emis, mb_y in DataLoader(te_ds, batch_size=BATCH_SIZE):
            p, K, b = model(mb_met.to(dev), mb_emis.to(dev))
            all_pred_te.append(p.cpu()); all_K_te.append(K.cpu()); all_b_te.append(b.cpu())

    pred_tr = torch.cat(all_pred_tr).numpy() * pm25_max
    pred_te = torch.cat(all_pred_te).numpy() * pm25_max
    gt_tr   = y_raw[tr_mask]
    gt_te   = y_raw[te_mask]

    def flat_valid(pred, gt):
        m = ~np.isnan(gt)
        return pred[m], gt[m]

    p_tr, g_tr = flat_valid(pred_tr, gt_tr)
    p_te, g_te = flat_valid(pred_te, gt_te)
    mae_tr  = np.abs(p_tr - g_tr).mean()
    mae_te  = np.abs(p_te - g_te).mean()
    rmse_te = np.sqrt(((p_te - g_te)**2).mean())
    print(f"\n[CLNO M2] train MAE={mae_tr:.1f}  test MAE={mae_te:.1f}  "
          f"RMSE={rmse_te:.1f} µg/m³")

    if run:
        run.finish(metrics={"MAE": float(mae_te), "RMSE": float(rmse_te),
                            "train_MAE": float(mae_tr)}, status="done")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_class": "CLNOLowRank",
        "H": H, "W": W, "S": S,
        "hidden": HIDDEN, "rank": RANK,
        "stations": stations.to_dict(),
        "meta": {k: v for k, v in meta.items()
                 if not isinstance(v, (np.ndarray, list))},
        "met_mu":  meta["met_mu"],
        "met_std": meta["met_std"],
    }, os.path.join(ROOT, "models", "clno_m2.pt"))
    print("[ok] models/clno_m2.pt")

    # Artifacts for eval
    K_tr = torch.cat(all_K_tr); b_tr = torch.cat(all_b_tr)
    K_te = torch.cat(all_K_te); b_te = torch.cat(all_b_te)
    torch.save({
        "K_tr": K_tr, "b_tr": b_tr,
        "K_te": K_te, "b_te": b_te,
        "met_n":  met_t,
        "emis_n": emis_t,
        "y_raw":  torch.tensor(y_raw),
        "tr_mask": torch.tensor(tr_mask),
        "te_mask": torch.tensor(te_mask),
    }, os.path.join(ROOT, "models", "clno_m2_artifacts.pt"))
    print("[ok] models/clno_m2_artifacts.pt")

    # ── Update metrics.json ───────────────────────────────────────────
    mp = os.path.join(ROOT, "models", "metrics.json")
    d  = json.load(open(mp)) if os.path.exists(mp) else {}
    d["clno_m2"] = dict(MAE=round(float(mae_te), 2),
                        RMSE=round(float(rmse_te), 2),
                        domain="SEA_111x101",
                        years=f"2019-2022_train_2023_test",
                        n_train=int((~np.isnan(gt_tr)).sum()),
                        n_test=int((~np.isnan(gt_te)).sum()))
    json.dump(d, open(mp, "w"), indent=2)
    print("[ok] models/metrics.json updated")

    # ── Loss curve ────────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tr_losses, label="train", alpha=0.8)
    ax.plot(te_losses, label="test",  alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (normalised)")
    ax.set_title("CLNO M2 training loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(ROOT, "figures", "clno_m2_loss.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Pred vs true scatter (test year)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(g_te, p_te, alpha=0.4, s=8, c="tab:orange")
    lim = [0, max(g_te.max(), p_te.max()) * 1.1]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Observed PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.set_title(f"CLNO M2 — test year {TEST_YEAR}\n"
                 f"MAE={mae_te:.1f}  RMSE={rmse_te:.1f} µg/m³")
    fig.savefig(os.path.join(ROOT, "figures", "clno_m2_pred_vs_true.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[ok] figures saved")


if __name__ == "__main__":
    main()
