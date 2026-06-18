"""
ตรวจสภาพแวดล้อม + เปิดข้อมูลทั้ง 4 แหล่ง (sanity check)
รัน:  conda run -n hazenet --no-capture-output python src/check_env.py
(การรันผ่าน conda run ทำให้ GDAL_DATA / PROJ_DATA ถูกตั้งให้ถูกต้อง)
"""
import sys
import os
import glob

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


def main():
    import xarray as xr
    import rioxarray  # noqa
    import rasterio
    import cartopy
    import pyproj
    import pandas as pd

    print("=== versions ===")
    print(f"python   {sys.version.split()[0]}")
    print(f"xarray   {xr.__version__}")
    print(f"rasterio {rasterio.__version__}")
    print(f"cartopy  {cartopy.__version__}")
    print(f"GDAL_DATA = {os.environ.get('GDAL_DATA')}")
    print(f"PROJ data = {pyproj.datadir.get_data_dir()}")

    print("\n=== ERA5 (ลม) ===")
    f = glob.glob(os.path.join(RAW, "era5", "*.nc"))[0]
    ds = xr.open_dataset(f)
    print(" vars", list(ds.data_vars), "dims", dict(ds.sizes))

    print("\n=== DEM (ภูเขา) + CRS op ===")
    dem = rioxarray.open_rasterio(
        os.path.join(RAW, "dem", "Copernicus_DSM_COG_10_N18_00_E098_00_DEM.tif"))
    print(" shape", tuple(dem.shape), "CRS", dem.rio.crs.to_epsg())
    print(" elev min/max =", float(dem.min()), "/", float(dem.max()), "m")
    # ทดสอบ reproject (ต้องใช้ PROJ_DATA) -> ถ้าทำได้ = env สมบูรณ์
    small = dem.isel(x=slice(0, 100), y=slice(0, 100))
    small.rio.reproject("EPSG:32647")  # UTM zone 47N
    print(" reproject EPSG:4326 -> 32647 : OK (PROJ ใช้ได้)")

    print("\n=== FIRMS (จุดไฟ) ===")
    fire = pd.read_csv(os.path.join(RAW, "firms",
                       "firms_VIIRS_SNPP_SP_2023-03-08_15.csv"))
    print(" จุดไฟ", len(fire), "| FRP รวม", round(fire.frp.sum()),
          "| วันที่", fire.acq_date.min(), "->", fire.acq_date.max())

    print("\n=== Air4Thai (PM2.5) ===")
    pm = pd.read_csv(os.path.join(RAW, "pm25", "air4thai_chiangmai_latest.csv"))
    print(" สถานีในกล่อง", len(pm), "| ตัวอย่าง pm25 =", list(pm.pm25.head(3)))

    print("\n✅ ENV + ข้อมูลทั้ง 4 แหล่ง พร้อมใช้งาน")


if __name__ == "__main__":
    main()
