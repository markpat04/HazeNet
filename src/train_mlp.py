"""
Stage 4b — Baseline: MLP (PyTorch, GPU) พยากรณ์ PM2.5 รายสถานีรายวัน
  features เดียวกับ XGBoost แต่ standardize ก่อน (MLP ต้องการ scale ใกล้กัน)
  เทรนบน GPU (RTX 4050) ถ้ามี ไม่งั้น CPU

รัน:  conda run -n hazenet --no-capture-output python src/train_mlp.py
ออก:  models/mlp_baseline.pt, figures/loss_curve.png, metrics ใน models/metrics.json
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import load_xy, temporal_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)
np.random.seed(42)


def save_metric(name, mae, rmse, n_train, n_test):
    p = os.path.join(ROOT, "models", "metrics.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[name] = dict(MAE=round(float(mae), 2), RMSE=round(float(rmse), 2),
                   n_train=int(n_train), n_test=int(n_test))
    json.dump(d, open(p, "w"), indent=2)


class MLP(nn.Module):
    def __init__(self, n_in, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    X, y, meta, names = load_xy()
    tr, te = temporal_split(meta, n_test_days=3)

    # standardize ด้วยสถิติของ train เท่านั้น (กันรั่ว)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    # target ก็ standardize ช่วยให้ลู่เข้าเร็ว
    ymu, ysd = y[tr].mean(), y[tr].std() + 1e-6
    ys = (y - ymu) / ysd

    Xtr = torch.tensor(Xs[tr], device=dev)
    ytr = torch.tensor(ys[tr], device=dev)
    Xte = torch.tensor(Xs[te], device=dev)

    model = MLP(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()

    losses = []
    EPOCHS = 400
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(Xtr)
        loss = lossf(out, ytr)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:4d}  train MSE(std)={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_s = model(Xte).cpu().numpy()
    pred = pred_s * ysd + ymu       # un-standardize กลับเป็น µg/m³
    mae = mean_absolute_error(y[te], pred)
    rmse = np.sqrt(mean_squared_error(y[te], pred))
    print(f"\n[MLP] MAE={mae:.1f}  RMSE={rmse:.1f} µg/m³")
    save_metric("mlp", mae, rmse, tr.sum(), te.sum())

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ROOT, "models", "mlp_baseline.pt"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, c="tab:purple")
    ax.set_xlabel("epoch"); ax.set_ylabel("train MSE (standardized)")
    ax.set_title(f"MLP training loss  (final test MAE={mae:.1f} µg/m³)")
    ax.grid(alpha=0.3)
    out = os.path.join(ROOT, "figures", "loss_curve.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[ok] -> {out}")


if __name__ == "__main__":
    main()
