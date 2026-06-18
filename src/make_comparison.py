"""
สรุปเปรียบเทียบโมเดลทั้งหมดจาก models/metrics.json -> กราฟแท่ง + ตารางใน console
รัน:  conda run -n hazenet --no-capture-output python src/make_comparison.py
ออก:  figures/model_comparison.png
"""
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = {"mean_predictor": "Mean\n(baseline)", "xgboost": "XGBoost", "mlp": "MLP"}
ORDER = ["mean_predictor", "xgboost", "mlp"]


def main():
    d = json.load(open(os.path.join(ROOT, "models", "metrics.json")))
    models = [m for m in ORDER if m in d]
    mae = [d[m]["MAE"] for m in models]
    rmse = [d[m]["RMSE"] for m in models]

    print("เปรียบเทียบโมเดล (test set):")
    print(f"  {'model':16} {'MAE':>7} {'RMSE':>7}")
    for m in models:
        print(f"  {m:16} {d[m]['MAE']:>7} {d[m]['RMSE']:>7}")

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, mae, w, label="MAE", color="tab:blue")
    b2 = ax.bar(x + w/2, rmse, w, label="RMSE", color="tab:orange")
    ax.bar_label(b1, fmt="%.0f", fontsize=9)
    ax.bar_label(b2, fmt="%.0f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in models])
    ax.set_ylabel("Error (µg/m³)  — lower is better")
    ax.set_title("HazeNet Phase 0 — model comparison (PM2.5, test set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(ROOT, "figures", "model_comparison.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n[ok] -> {out}")


if __name__ == "__main__":
    main()
