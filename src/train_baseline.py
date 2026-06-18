"""
Stage 4a — Baseline: XGBoost พยากรณ์ PM2.5 รายสถานีรายวัน
  features: patch 3x3 ของ 4 channel + lat/lon/tidx (จาก featurize.py)
  split:    temporal (วันท้าย 3 วันเป็น test กันรั่วอนาคต)
  เทียบกับ: mean predictor (ทายค่าเฉลี่ย train ทุกครั้ง) เพื่อพิสูจน์โมเดลเรียนรู้จริง

รัน:  conda run -n hazenet --no-capture-output python src/train_baseline.py
ออก:  models/xgb_baseline.pkl, figures/pred_vs_true_xgb.png, metrics เก็บใน models/metrics.json
"""
import os
import sys
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import load_xy, temporal_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_metric(name, mae, rmse, n_train, n_test):
    p = os.path.join(ROOT, "models", "metrics.json")
    d = {}
    if os.path.exists(p):
        d = json.load(open(p))
    d[name] = dict(MAE=round(float(mae), 2), RMSE=round(float(rmse), 2),
                   n_train=int(n_train), n_test=int(n_test))
    json.dump(d, open(p, "w"), indent=2)


def main():
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

    X, y, meta, names = load_xy()
    tr, te = temporal_split(meta, n_test_days=3)
    print(f"ข้อมูล: {len(y)} obs  | train {tr.sum()}  test {te.sum()}")
    print(f"features: {X.shape[1]} ({len(names)} ชื่อ)")

    # --- baseline ง่าย: ทายค่าเฉลี่ย train ---
    mean_pred = np.full(te.sum(), y[tr].mean())
    mae0 = mean_absolute_error(y[te], mean_pred)
    rmse0 = np.sqrt(mean_squared_error(y[te], mean_pred))
    print(f"\n[mean predictor] MAE={mae0:.1f}  RMSE={rmse0:.1f}")
    save_metric("mean_predictor", mae0, rmse0, tr.sum(), te.sum())

    # --- XGBoost ---
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    mae = mean_absolute_error(y[te], pred)
    rmse = np.sqrt(mean_squared_error(y[te], pred))
    print(f"[XGBoost]        MAE={mae:.1f}  RMSE={rmse:.1f}  "
          f"(ดีขึ้น {(1-mae/mae0)*100:.0f}% จาก mean)")
    save_metric("xgboost", mae, rmse, tr.sum(), te.sum())

    with open(os.path.join(ROOT, "models", "xgb_baseline.pkl"), "wb") as f:
        pickle.dump(model, f)

    # --- feature importance (top 10) ---
    imp = sorted(zip(names, model.feature_importances_),
                 key=lambda x: -x[1])[:10]
    print("\n top features:")
    for n, v in imp:
        print(f"   {n:14} {v:.3f}")

    # --- plot pred vs true ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y[te], pred, alpha=0.6, c="tab:blue", edgecolors="k", linewidths=0.3)
    lim = [0, max(y[te].max(), pred.max()) * 1.1]
    ax.plot(lim, lim, "r--", lw=1, label="perfect")
    ax.set_xlabel("Observed PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.set_title(f"XGBoost baseline — test set\nMAE={mae:.1f}  RMSE={rmse:.1f} µg/m³")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    out = os.path.join(ROOT, "figures", "pred_vs_true_xgb.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n[ok] -> {out}")


if __name__ == "__main__":
    main()
