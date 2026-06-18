# HazeNet — data/ (Lab 0 / feasibility)

ข้อมูลดิบของกล่องเชียงใหม่ (18–19.5°N, 98–100°E) ช่วงฤดูเผา 8–15 มี.ค. 2023
สำหรับ walking skeleton ตามแผน `Knowledge/lab-feasibility-checklist.html`

## สถานะแหล่งข้อมูล (ขั้น 0 — เข้าถึง) ✅ ครบทั้ง 4 แหล่ง

| แหล่ง | สิ่งที่ได้ | key? | สถานะ | สคริปต์ |
|---|---|---|---|---|
| **Copernicus DEM** | ความสูงภูมิประเทศ 4 tile (~180MB) | ไม่ต้อง | ✅ ดึงแล้ว | `src/download_dem.py` |
| **Air4Thai** | PM2.5 ล่าสุด 175 สถานี (8 ในกล่อง) | ไม่ต้อง | ✅ ดึงแล้ว | `src/download_air4thai.py` |
| **FIRMS** | จุดไฟ/FRP 466 จุด (8–15 มี.ค. 2023) | MAP_KEY (มีแล้ว) | ✅ ดึงแล้ว | `src/download_firms.py` |
| **ERA5** | ลม u10/v10 (64 เฟรม, กริด 7×9) | บัญชี CDS | ✅ โหลดเอง | `src/download_era5.py` |

รันทั้งหมด: `conda run -n hazenet --no-capture-output python src/download.py`

## Environment (conda — env ชื่อ `hazenet`)
สร้างจาก `environment.yml` (Python 3.12 · conda-forge: xarray/rasterio/rioxarray/cartopy/zarr/cdsapi/...)
- สร้าง env:  `conda env create -f environment.yml`
- อัปเดต:  `conda env update -f environment.yml`
- **รันสคริปต์:**  `conda run -n hazenet --no-capture-output python src/xxx.py`
- ติดตั้งเพิ่ม:  `conda install -n hazenet -c conda-forge <pkg>`
- ตรวจ env+ข้อมูล:  `conda run -n hazenet --no-capture-output python src/check_env.py`

### gotcha ที่เจอจริง (จำไว้)
- ⚠️ ต้องรันผ่าน **`conda run`** ไม่ใช่เรียก `python.exe` ตรง ๆ — ไม่งั้น GDAL_DATA/PROJ_DATA ไม่ถูกตั้ง (CRS/reproject พัง)
- ⚠️ `conda run python -c "..."` ใช้ argument **หลายบรรทัดไม่ได้** บน Windows → เขียนเป็นไฟล์ `.py` แล้วรันแทน
- ⚠️ console เด้ง error ภาษาไทย (cp1252) → ตั้ง `PYTHONUTF8=1` หรือ `sys.stdout.reconfigure(encoding="utf-8")`
- ข้อความ `Error in sys.excepthook` ตอนจบ = artifact ตอนปิดโปรแกรม (ไม่กระทบผล)

---

## สิ่งที่คุณต้องทำต่อ — สมัคร 2 key (ฟรีทั้งคู่)

### 1) FIRMS MAP_KEY (จุดไฟ) — เร็วสุด ได้ทันที
1. เปิด https://firms.modaps.eosdis.nasa.gov/api/area/
2. กรอกอีเมล → รับ MAP_KEY ทางอีเมลทันที
3. ใส่ key อย่างใดอย่างหนึ่ง:
   - สร้างไฟล์ `src/.keys` ใส่บรรทัด: `FIRMS_MAP_KEY=คีย์ของคุณ`
   - หรือ (PowerShell): `setx FIRMS_MAP_KEY "คีย์ของคุณ"` แล้วเปิด terminal ใหม่
4. รัน `python src/download_firms.py`

### 2) Copernicus CDS (ERA5 ลม) — คอขวดเวลา ทำก่อน!
> ⚠️ คิวประมวลผลของ CDS อาจรอเป็นชั่วโมง — สมัครและยิง request ไว้แต่เนิ่น ๆ

1. สมัคร: https://cds.climate.copernicus.eu/  → ยืนยันอีเมล
2. **กด accept license** ของ dataset (สำคัญ! ไม่งั้น request จะ error):
   เปิดหน้า `reanalysis-era5-single-levels` → เลื่อนลงล่าง → ติ๊ก "accept terms"
3. เอา token จากหน้า profile (https://cds.climate.copernicus.eu/profile)
4. สร้างไฟล์ `C:\Users\mark\.cdsapirc` ใส่:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-TOKEN>
   ```
5. ติดตั้ง: `pip install "cdsapi>=0.7"`
6. รัน `python src/download_era5.py`

**ถ้า CDS คิวช้ามากจนรอไม่ไหว** → ทางเลือก: ARCO-ERA5 (Zarr บน Google Cloud, ไม่ต้องรอคิว)
หรือ ERA5 ผ่าน Google Earth Engine — บอกได้ ผมช่วยเปลี่ยนสคริปต์ให้

---

## หมายเหตุ
- ไฟล์ `src/.keys` และข้อมูลใหญ่ใน `data/raw/` ไม่ควร commit ขึ้น git (ใส่ .gitignore)
- Air4Thai ที่ดึงตอนนี้เป็น "ค่าล่าสุด" (พิสูจน์ว่าเข้าถึงได้) — ข้อมูลย้อนหลังเป็นสัปดาห์
  จะทำในขั้นถัดไป (Air4Thai history หรือ OpenAQ)
- ค่า PM2.5 ตอนนี้ต่ำ (~6–12) เพราะเป็นเดือน มิ.ย. (นอกฤดูเผา) — ปกติ
