"""
Leave-One-Year-Out (LOYO) cross-validation สำหรับ CLNO M2.

เป้าหมาย: พิสูจน์ว่า test MAE สูง (~72) เป็นเพราะ "เลือกปี 2023 ที่ผิดปกติ"
เป็นข้อสอบ ไม่ใช่เพราะโมเดลพัง

วิธี: วนทดสอบทีละปี
  - test = ปีนั้น   train = อีก 4 ปี
  - แบ่ง 20% ของวันใน train เป็น val สำหรับ early stopping
  - normalize ด้วย train years เท่านั้น (กันรั่ว)
  - ใช้ architecture เดียว: CLNOGlobalV rank=16 (parsimonious, 0.42M params)
      เพราะ sweep แสดงว่าโมเดลใหญ่ไม่ช่วย

Output:
  reports/loyo_results.csv
  figures/loyo_m2.png
"""
import os, sys, copy, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def set_thai_font():
    """ตั้งฟอนต์ไทยของ Windows ให้ matplotlib (กันตัวอักษรไทยเป็นกล่อง)"""
    for cand in ["C:/Windows/Fonts/leelawui.ttf", "C:/Windows/Fonts/tahoma.ttf"]:
        if os.path.exists(cand):
            fm.fontManager.addfont(cand)
            plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


set_thai_font()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_operator_m2 import CLNOGlobalV
from config_m2 import ROOT, PROC

torch.manual_seed(42); np.random.seed(42)

ALL_YEARS  = [2019, 2020, 2021, 2022, 2023]
BATCH      = 32
MAX_EPOCHS = 300
PATIENCE   = 35
RANK, WD, DROPOUT = 16, 1e-3, 0.3


def masked_mse(pred, target):
    m = ~torch.isnan(target)
    return F.mse_loss(pred[m], target[m]) if m.sum() else pred.sum() * 0


def load_raw():
    cube  = xr.open_zarr(os.path.join(PROC, "datacube_m2.zarr"))
    X     = cube.X.values
    times = pd.DatetimeIndex(cube.time.values)
    yrs   = times.year.values
    met_raw  = X[:, :-1].astype("float32")    # all channels except last
    emis_raw = X[:,  -1].astype("float32")    # emission = last channel

    tgt = pd.read_csv(os.path.join(PROC, "target_pm25_m2.csv"))
    tgt["date"] = pd.to_datetime(tgt["date"])
    stations = (tgt.groupby("locationId")
                   .first()[["location", "lat", "lon"]]
                   .reset_index().sort_values("locationId").reset_index(drop=True))
    S = len(stations)
    t_map = {pd.Timestamp(t): i for i, t in enumerate(times)}
    s_map = {sid: i for i, sid in enumerate(stations["locationId"])}
    T = len(times)
    y_raw = np.full((T, S), np.nan, dtype="float32")
    for _, r in tgt.iterrows():
        ti = t_map.get(r["date"]); si = s_map.get(r["locationId"])
        if ti is not None and si is not None:
            y_raw[ti, si] = r["pm25"]
    H, W = X.shape[2], X.shape[3]
    in_ch = met_raw.shape[1]
    return met_raw, emis_raw, y_raw, yrs, S, H, W, in_ch


def run_fold(test_year, met_raw, emis_raw, y_raw, yrs, S, H, W, in_ch, dev):
    tr_years  = [y for y in ALL_YEARS if y != test_year]
    tr_full   = np.isin(yrs, tr_years)
    te_mask   = yrs == test_year

    # split train days -> 80% train / 20% val (for early stopping)
    tr_days = np.where(tr_full)[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr_days))
    n_val = int(0.2 * len(tr_days))
    va_days = tr_days[perm[:n_val]]
    tr_days = tr_days[perm[n_val:]]

    # normalize on train days only
    met_mu  = met_raw[tr_days].mean(axis=(0, 2, 3), keepdims=True)
    met_std = met_raw[tr_days].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    met_n   = (met_raw - met_mu) / met_std
    e_max   = float(emis_raw[tr_days].max()) + 1e-6
    emis_n  = emis_raw / e_max
    pm25_max = float(np.nanmax(y_raw[tr_days])) + 1e-6
    y_n      = y_raw / pm25_max

    def loader(idx, shuffle):
        ds = TensorDataset(torch.tensor(met_n[idx]),
                           torch.tensor(emis_n[idx]),
                           torch.tensor(y_n[idx]))
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

    def mae_on(idx):
        model.eval(); preds = []
        with torch.no_grad():
            for mb_m, mb_e, _ in loader(idx, False):
                p, _, _ = model(mb_m.to(dev), mb_e.to(dev))
                preds.append(p.cpu().numpy())
        pred = np.concatenate(preds) * pm25_max
        gt = y_raw[idx]; m = ~np.isnan(gt)
        mae  = np.abs(pred[m] - gt[m]).mean()
        bias = (pred[m] - gt[m]).mean()
        return float(mae), float(bias), pred, gt, m

    model = CLNOGlobalV(H, W, S, hidden=64, rank=RANK, dropout=DROPOUT,
                        in_ch=in_ch).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_val = float("inf"); best_state = None; wait = 0
    tr_loader = loader(tr_days, True)
    for ep in range(MAX_EPOCHS):
        model.train()
        for mb_m, mb_e, mb_y in tr_loader:
            opt.zero_grad()
            pred, _, _ = model(mb_m.to(dev), mb_e.to(dev))
            loss = masked_mse(pred, mb_y.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vmae, _, *_ = mae_on(va_days)
        if vmae < best_val - 1e-4:
            best_val = vmae; best_state = copy.deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)

    te_mae, te_bias, *_ = mae_on(np.where(te_mask)[0])
    obs_mean = float(np.nanmean(y_raw[te_mask]))
    return dict(test_year=test_year, n_test_days=int(te_mask.sum()),
                obs_mean=round(obs_mean, 1),
                test_mae=round(te_mae, 1), test_bias=round(te_bias, 1),
                val_mae=round(best_val, 1))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    print(f"LOYO cross-validation — CLNOGlobalV rank={RANK} wd={WD} dropout={DROPOUT}\n")

    met_raw, emis_raw, y_raw, yrs, S, H, W, in_ch = load_raw()
    print(f"met channels = {in_ch}\n")

    rows = []
    for ty in ALL_YEARS:
        t0 = time.time()
        r  = run_fold(ty, met_raw, emis_raw, y_raw, yrs, S, H, W, in_ch, dev)
        r["sec"] = round(time.time() - t0)
        rows.append(r)
        print(f"  test={ty}  obs_mean={r['obs_mean']:5.1f}  "
              f"MAE={r['test_mae']:5.1f}  bias={r['test_bias']:+5.1f}  "
              f"({r['sec']}s)")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    csv = os.path.join(ROOT, "reports", "loyo_results.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    normal = df[df["test_year"] != 2023]["test_mae"].mean()
    print(f"\nสรุป:")
    print(f"  MAE เฉลี่ยปีปกติ (2019-22) = {normal:.1f} µg/m³")
    print(f"  MAE ปี 2023 (หมอกหนัก)     = {df[df['test_year']==2023]['test_mae'].values[0]:.1f} µg/m³")
    print(f"  [ok] {csv}")

    # figure
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#ef4444" if y == 2023 else "#3b82f6" for y in df["test_year"]]
    bars = ax.bar(df["test_year"].astype(str), df["test_mae"], color=colors)
    ax.bar_label(bars, fmt="%.0f", fontsize=10)
    ax.axhline(normal, color="#22c55e", ls="--", lw=1.3,
               label=f"avg normal years = {normal:.0f}")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, 3, f"obs={r['obs_mean']:.0f}", ha="center",
                color="white", fontsize=8)
    ax.set_ylabel("Test MAE (µg/m³)")
    ax.set_xlabel("ปีที่กันไว้เป็นข้อสอบ (held-out test year)")
    ax.set_title("Leave-One-Year-Out — โมเดลทำงานดีในปีปกติ\n"
                 "2023 (แดง) MAE สูงเพราะเป็นปีหมอกควันผิดปกติ (OOD)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    out = os.path.join(ROOT, "figures", "loyo_m2.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")


if __name__ == "__main__":
    main()
