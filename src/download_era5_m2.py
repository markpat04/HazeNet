"""
ดาวน์โหลด ERA5 ลม 10m สำหรับ domain SEA, ฤดูเผา (ก.พ.-เม.ย.) 2019-2023

ดาวน์โหลดทีละ 1 ปี 1 เดือน = 15 ไฟล์ (5 ปี × 3 เดือน)
แต่ละไฟล์: era5_wind_{year}-{mm}.nc  เก็บในไดเรกทอรี data/raw_m2/era5/

ข้อกำหนด: ~/.cdsapirc พร้อม url + key

Run: conda run -n hazenet --no-capture-output python src/download_era5_m2.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import YEARS, ROOT

OUT_DIR = os.path.join(ROOT, "data", "raw_m2", "era5")
# [North, West, South, East] — ใหญ่กว่า domain นิดนึงเผื่อ interp
AREA    = [25.5, 95.5, 13.5, 106.5]
MONTHS  = ["02", "03", "04"]   # ก.พ., มี.ค., เม.ย.
VARS    = ["10m_u_component_of_wind", "10m_v_component_of_wind"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        import cdsapi
    except ImportError:
        print("[ERR ] cdsapi ไม่ได้ติดตั้ง -> pip install 'cdsapi>=0.7'")
        return

    rc = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(rc):
        print("[ERR ] ~/.cdsapirc ไม่มี  (ต้องการ url + key จาก CDS)")
        return

    c = cdsapi.Client()
    tasks = [(y, m) for y in YEARS for m in MONTHS]
    print(f"ERA5 M2: {len(tasks)} requests  (5 years × 3 months)")

    for year, month in tasks:
        import calendar
        ndays = calendar.monthrange(year, int(month))[1]
        out = os.path.join(OUT_DIR, f"era5_wind_{year}-{month}.nc")

        if os.path.exists(out) and os.path.getsize(out) > 10_000:
            mb = os.path.getsize(out) / 1e6
            print(f"[skip] {os.path.basename(out)}  ({mb:.1f} MB)")
            continue

        print(f"[get ] {year}-{month} ({ndays} days)  -> {os.path.basename(out)} ...")
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable":     VARS,
                    "year":         str(year),
                    "month":        month,
                    "day":          [f"{d:02d}" for d in range(1, ndays + 1)],
                    "time":         [f"{h:02d}:00" for h in range(0, 24, 3)],
                    "area":         AREA,
                    "format":       "netcdf",
                },
                out,
            )
            mb = os.path.getsize(out) / 1e6
            print(f"[ok  ] {os.path.basename(out)}  ({mb:.1f} MB)")
        except Exception as e:
            print(f"[ERR ] {year}-{month}: {type(e).__name__}: {e}")

    print("\nERA5 M2 download complete.")


if __name__ == "__main__":
    main()
