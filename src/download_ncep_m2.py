"""
ดาวน์โหลดลม 10m จาก NCEP Reanalysis 2 (NOAA) — ไม่ต้องมี API key

เป็น alternative สำหรับ ERA5 เมื่อไม่มี ~/.cdsapirc
Resolution: ~1.875° (coarser กว่า ERA5 0.25° แต่ยังดีกว่า target 0.1°)
Coverage: global, 1979-present, daily mean

Variables: uwnd.10m, vwnd.10m (m/s)

ผลลัพธ์: data/raw_m2/ncep/ncep_wind_{year}.nc  (1 ไฟล์ต่อปี ~5MB)

Run: conda run -n hazenet --no-capture-output python src/download_ncep_m2.py
"""
import urllib.request
import urllib.error
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import YEARS, ROOT

OUT_DIR  = os.path.join(ROOT, "data", "raw_m2", "ncep")
# NCEP/NCAR Reanalysis 1 — public, no API key, global daily 10m wind
# Files: uwnd.10m.gauss.{year}.nc  vwnd.10m.gauss.{year}.nc
BASE_URL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Dailies/surface"


def download_file(url: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        mb = os.path.getsize(dest) / 1e6
        print(f"[skip] {os.path.basename(dest)}  ({mb:.1f} MB)")
        return True
    print(f"[get ] {url}")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            mb = os.path.getsize(dest) / 1e6
            print(f"[ok  ] {os.path.basename(dest)}  ({mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"NCEP M2: downloading {YEARS[0]}-{YEARS[-1]} uwnd+vwnd 10m")

    for year in YEARS:
        # NOAA NCEP Reanalysis 2: daily means, global, 10m wind
        for comp in ("uwnd", "vwnd"):
            url  = f"{BASE_URL}/{comp}.10m.gauss.{year}.nc"
            dest = os.path.join(OUT_DIR, f"ncep_{comp}_{year}.nc")
            if not download_file(url, dest):
                print(f"[ERR ] failed to download {comp} {year}")

    print("\nNCEP M2 download complete.")
    print("ไฟล์ทั้งหมดใน:", OUT_DIR)


if __name__ == "__main__":
    main()
