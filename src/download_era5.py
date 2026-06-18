"""
ดึงลม 10 เมตร (u10, v10) จาก ERA5 ของกล่องเชียงใหม่ ผ่าน CDS API
ต้องมีบัญชี Copernicus CDS + token (ขั้นตอนใน data/README.md)

ติดตั้ง:  pip install "cdsapi>=0.7"
คอขวด: คิวประมวลผลของ CDS อาจรอเป็นชั่วโมง -> ยิง request ไว้แต่เนิ่น ๆ

ผลลัพธ์: data/raw/era5/era5_wind_2023-03.nc
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = "data/raw/era5/era5_wind_2023-03-19_31.nc"
AREA = [19.5, 98.0, 18.0, 100.0]   # [North, West, South, East]


def main():
    os.makedirs("data/raw/era5", exist_ok=True)
    try:
        import cdsapi
    except ImportError:
        print("[ข้าม] ยังไม่ได้ติดตั้ง cdsapi  ->  pip install \"cdsapi>=0.7\"")
        return

    rc = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(rc):
        print("[ข้าม] ยังไม่มีไฟล์ ~/.cdsapirc (ใส่ url + key ของ CDS)")
        print("       ดูขั้นตอนใน data/README.md")
        return

    print("[get ] ส่ง request ERA5 ลม 10m (อาจรอคิว)...")
    c = cdsapi.Client()
    try:
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": "2023",
                "month": "03",
                "day": [f"{d:02d}" for d in range(19, 32)],
                "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
                "area": AREA,
                "format": "netcdf",
            },
            OUT,
        )
        mb = os.path.getsize(OUT) / 1e6
        print(f"[ok  ] ERA5 -> {OUT}  ({mb:.2f} MB)")
    except Exception as e:
        print(f"[ERR ] {type(e).__name__}: {e}")
        print("       เช็ก: accept license ของ dataset ในเว็บแล้วหรือยัง?")


if __name__ == "__main__":
    main()
