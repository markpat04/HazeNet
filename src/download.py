"""
ดึงข้อมูลทุกแหล่งของ HazeNet (Lab 0 / feasibility) ในคำสั่งเดียว
รัน:  python src/download.py

- DEM (ภูเขา)        : ฟรี ไม่ต้องใช้ key  -> ดึงได้เลย
- Air4Thai (ค่าฝุ่น) : ฟรี ไม่ต้องใช้ key  -> ดึงได้เลย
- FIRMS (จุดไฟ)      : ต้องมี MAP_KEY      -> ข้ามถ้ายังไม่มี
- ERA5 (ลม)          : ต้องมีบัญชี CDS      -> ข้ามถ้ายังไม่มี
"""
import sys
import runpy
import os

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(__file__)
STEPS = [
    ("DEM (ภูเขา)        [ฟรี]", "download_dem.py"),
    ("Air4Thai (ค่าฝุ่น) [ฟรี]", "download_air4thai.py"),
    ("FIRMS (จุดไฟ)      [key]", "download_firms.py"),
    ("ERA5 (ลม)          [key]", "download_era5.py"),
]


def main():
    for title, script in STEPS:
        print("\n" + "=" * 60)
        print(f">>> {title}")
        print("=" * 60)
        try:
            runpy.run_path(os.path.join(HERE, script), run_name="__main__")
        except SystemExit:
            pass
        except Exception as e:
            print(f"[ERR ] {script}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
