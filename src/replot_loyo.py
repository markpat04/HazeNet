"""
สร้างกราฟ loyo_m2.png ใหม่จาก reports/loyo_results.csv
โดยใช้ฟอนต์ไทย (กันตัวอักษรไทยเป็นกล่องสี่เหลี่ยม)
ไม่ต้องรัน LOYO ใหม่

Run: conda run -n hazenet --no-capture-output python src/replot_loyo.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import ROOT


def set_thai_font():
    for cand in ["C:/Windows/Fonts/leelawui.ttf", "C:/Windows/Fonts/tahoma.ttf"]:
        if os.path.exists(cand):
            fm.fontManager.addfont(cand)
            plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
            print(f"ใช้ฟอนต์: {fm.FontProperties(fname=cand).get_name()}")
            break
    plt.rcParams["axes.unicode_minus"] = False


def main():
    set_thai_font()
    df = pd.read_csv(os.path.join(ROOT, "reports", "loyo_results.csv"))
    df = df.sort_values("test_year").reset_index(drop=True)
    normal = df[df["test_year"] != 2023]["test_mae"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#ef4444" if y == 2023 else "#3b82f6" for y in df["test_year"]]
    bars = ax.bar(df["test_year"].astype(str), df["test_mae"], color=colors)
    ax.bar_label(bars, fmt="%.0f", fontsize=11, padding=3)
    ax.axhline(normal, color="#22c55e", ls="--", lw=1.4,
               label=f"ค่าเฉลี่ยปีปกติ = {normal:.0f}")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, 3, f"ฝุ่นจริง\n{r['obs_mean']:.0f}", ha="center",
                color="white", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("ค่าทายพลาดเฉลี่ย MAE (µg/m³)")
    ax.set_xlabel("ปีที่กันไว้เป็นข้อสอบ (held-out test year)")
    ax.set_title("Leave-One-Year-Out — โมเดลทำงานดีในปีปกติ\n"
                 "ปี 2023 (แดง) สูงเพราะเป็นปีหมอกควันหนักผิดปกติ", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, df["test_mae"].max() * 1.18)

    out = os.path.join(ROOT, "figures", "loyo_m2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
