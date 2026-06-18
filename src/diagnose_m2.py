"""
วินิจฉัย: ทำไม test MAE สูง? เป็น over-prediction จาก emission shift หรือไม่?

เช็ค:
  1. ขนาด emission (ไฟ) ต่อปี — 2023 แรงกว่า train years แค่ไหน
  2. ขนาด PM2.5 จริง ต่อปี
  3. bias ของโมเดล (mean predicted - mean observed) บน test
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import xarray as xr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_operator_m2 import CLNOLowRank, CLNOGlobalV
from config_m2 import ROOT, PROC

cube  = xr.open_zarr(os.path.join(PROC, "datacube_m2.zarr"))
times = pd.DatetimeIndex(cube.time.values)
yrs   = times.year.values
emis  = cube.X.values[:, 3]                # (T,H,W) raw FRP

print("=== 1. Emission (FIRMS FRP) ต่อปี ===")
print(f"{'year':>6} {'days':>5} {'mean/day':>10} {'max_cell':>10} {'total':>12}")
for y in [2019, 2020, 2021, 2022, 2023]:
    m = yrs == y
    e = emis[m]
    print(f"{y:>6} {m.sum():>5} {e.sum()/m.sum():>10.0f} "
          f"{e.max():>10.0f} {e.sum():>12.0f}")

train_max_cell = emis[np.isin(yrs, [2019,2020,2021])].max()
test_max_cell  = emis[yrs == 2023].max()
print(f"\ntrain(19-21) max cell = {train_max_cell:.0f}")
print(f"test(2023)   max cell = {test_max_cell:.0f}  "
      f"→ {test_max_cell/train_max_cell:.2f}× ของ train max")
print(f"  (emission ถูก normalize ด้วย train max → 2023 จะมีค่า > 1 = นอกช่วงเทรน)")

print("\n=== 2. PM2.5 จริง ต่อปี ===")
tgt = pd.read_csv(os.path.join(PROC, "target_pm25_m2.csv"))
tgt["year"] = pd.to_datetime(tgt["date"]).dt.year
for y in [2019, 2020, 2021, 2022, 2023]:
    s = tgt[tgt["year"] == y]["pm25"]
    if len(s):
        print(f"{y:>6}  n={len(s):>5}  mean={s.mean():>5.1f}  "
              f"p90={s.quantile(.9):>5.0f}  max={s.max():>5.0f}")

print("\n=== 3. โมเดล best: bias บน test (2023) ===")
ckpt = torch.load(os.path.join(ROOT, "models", "clno_m2.pt"),
                  map_location="cpu", weights_only=False)
art  = torch.load(os.path.join(ROOT, "models", "clno_m2_artifacts.pt"),
                  map_location="cpu", weights_only=False)
cls  = CLNOGlobalV if ckpt.get("model_class") == "CLNOGlobalV" else CLNOLowRank
model = cls(H=ckpt["H"], W=ckpt["W"], n_stations=ckpt["S"],
            hidden=ckpt["hidden"], rank=ckpt["rank"],
            dropout=ckpt.get("dropout", 0.1))
model.load_state_dict(ckpt["state_dict"]); model.eval()
pm25_max = ckpt["meta"]["pm25_max"]

te_mask = art["te_mask"].numpy()
met_n   = None
# recompute predictions on test using stored K/b? We have K_te,b_te,emis_n
K_te   = art["K_te"]; b_te = art["b_te"]
emis_n = art["emis_n"][te_mask]            # (n_te,H,W)
y_raw  = art["y_raw"].numpy()[te_mask]     # (n_te,S)

with torch.no_grad():
    E = emis_n.view(emis_n.shape[0], -1)
    pred = torch.bmm(K_te, E.unsqueeze(-1)).squeeze(-1) + b_te
pred = pred.numpy() * pm25_max

m = ~np.isnan(y_raw)
print(f"  observed  mean={y_raw[m].mean():6.1f}  µg/m³")
print(f"  predicted mean={pred[m].mean():6.1f}  µg/m³")
print(f"  BIAS (pred-obs)={pred[m].mean()-y_raw[m].mean():+6.1f}  µg/m³")
print(f"  predicted p99 ={np.percentile(pred[m],99):6.1f}  (มีพุ่งทะลุไหม)")
print(f"  predicted max ={pred[m].max():6.1f}")
over = (pred[m] > 2*y_raw[m]).mean()*100
print(f"  % ที่ทำนาย > 2× ค่าจริง = {over:.0f}%")
