# HazeNet — PM2.5 Transboundary Haze Forecasting & Source Attribution

พยากรณ์ฝุ่น **PM2.5** จากหมอกควันข้ามพรมแดน + หาแหล่งที่มา สำหรับภาคเหนือของไทย/อาเซียน
ใช้ **CLNO (Conditionally-Linear Neural Operator)** — โมเดลเดียวให้ 3 output: พยากรณ์ + attribution + emission inversion

## ขอบเขต Phase 0

| มิติ | ค่า |
|---|---|
| พื้นที่ | กล่องเชียงใหม่ 18.0–19.5°N, 98.0–100.0°E |
| ความละเอียด | 0.1° (~11 กม.) → grid 16 × 21 |
| ช่วงเวลา | 2023-03-19 .. 03-31 (13 วัน) — เหตุการณ์หมอกควันรุนแรง |
| target | PM2.5 รายวัน 12 สถานี (avg 93.8, max 420 µg/m³) |

## ข้อมูล 4 แหล่ง (channels)

| channel | แหล่ง | บทบาท |
|---|---|---|
| `u10`, `v10` | ERA5 (CDS) | ลม 10 ม. — ควันเคลื่อนไปทางไหน |
| `dem` | Copernicus GLO-30 | ความสูงภูมิประเทศ — ควันสะสมที่ไหน |
| `emission` | NASA FIRMS (VIIRS) | FRP จากจุดไฟ — แหล่งกำเนิดควัน |

## Pipeline

```
download_*.py      ดาวน์โหลดข้อมูลดิบ -> data/raw/
   │
build_grid.py      regrid ทุกแหล่งลง master grid 16x21 -> data/processed/grid.nc
   │
build_datacube.py  stack channels -> data/processed/datacube.zarr (+ target_pm25.csv)
   │
train_baseline.py  XGBoost   ─┐
train_mlp.py       MLP (GPU)  ─┤-> models/  + metrics.json
   │
plot_prediction.py แผนที่ทำนาย PM2.5 ทั่วกล่อง + animation -> figures/
make_comparison.py สรุปเทียบโมเดล -> figures/model_comparison.png
```

## วิธีรัน

```bash
# 1. สร้าง environment (ครั้งเดียว)
conda env create -f environment.yml
conda activate hazenet

# 2. (ถ้ายังไม่มีข้อมูล) ใส่ API key ใน src/.keys และ ~/.cdsapirc แล้ว:
python src/download_dem.py
python src/download_era5.py
python src/download_firms.py
python src/download_pm25_history.py

# 3. รัน pipeline
python src/build_grid.py
python src/build_datacube.py
python src/train_baseline.py
python src/train_mlp.py
python src/plot_prediction.py
python src/make_comparison.py
```

> Windows: ถ้า GDAL/PROJ error ให้รันผ่าน `conda run -n hazenet --no-capture-output python <script>`

## ผลลัพธ์ (Phase 0)

ดู `figures/`:
- `overview_chiangmai.png` — ภาพรวมข้อมูล 4 แหล่งบนแผนที่เดียว
- `pred_vs_true_xgb.png` — ทำนาย vs จริง (test set)
- `pm25_pred_map.png` — แผนที่ PM2.5 ทำนายทั่วกล่อง 6 วัน
- `pm25_animation.gif` — animation 13 วัน
- `model_comparison.png` — เทียบ MAE/RMSE ทุกโมเดล

ตัวเลข metric ล่าสุดอยู่ใน `models/metrics.json`

## โครงสร้าง

```
config.yaml              ค่า config การทดลอง (frozen)
environment.yml          conda env "hazenet"
src/                     โค้ดทั้งหมด
data/raw/                ข้อมูลดิบ (gitignored)
data/processed/          grid.nc, datacube.zarr, target (gitignored)
models/                  โมเดล + metrics.json
figures/                 รูปผลลัพธ์
```

## M2 — SEA Scale-up (ทำต่อจาก Phase 0)

ขยายทั้ง 3 มิติพร้อมกัน pipeline เดิมทุกชิ้น แค่เปลี่ยนขนาดข้อมูล:

| มิติ | Phase 0 | M2 |
|---|---|---|
| พื้นที่ | กล่องเชียงใหม่ 16×21 | SEA 111×101 (ไทย+พม่า+ลาว) |
| ช่วงเวลา | 13 วัน (2023-03) | ~449 วัน (ก.พ.-เม.ย. 2019-2023) |
| สถานี PM2.5 | 12 | ~40+ |
| source cells G | 336 | 11,211 |
| train/test split | last 3 days | ปี 2022 vs 2023 |
| CLNO variant | standard | CLNOLowRank (rank-32 K) |

### วิธีรัน M2

```bash
# 1. ดาวน์โหลด (ทำครั้งเดียว)
python src/download_dem_m2.py        # DEM tiles 110 tiles (~2-4 GB)
python src/download_era5_m2.py       # ERA5 15 files (อาจรอคิว CDS)
python src/download_firms_m2.py      # FIRMS 5 ปี ~45 API calls
python src/download_pm25_m2.py       # OpenAQ PM2.5 multi-year

# 2. สร้าง datacube
python src/build_grid_m2.py          # -> data/processed_m2/grid_m2.nc
python src/build_datacube_m2.py      # -> datacube_m2.zarr + target_pm25_m2.csv

# 3. เทรน + ประเมิน
KMP_DUPLICATE_LIB_OK=TRUE python src/train_operator_m2.py
KMP_DUPLICATE_LIB_OK=TRUE python src/eval_operator_m2.py
```

## โครงสร้างไฟล์

```
src/
  config_m2.py          ค่า constant ทั้งหมดของ M2 (domain, dates, paths)
  download_dem_m2.py    DEM tiles สำหรับ SEA domain
  download_era5_m2.py   ERA5 ลมรายเดือน 2019-2023
  download_firms_m2.py  FIRMS FRP รายปี 2019-2023
  download_pm25_m2.py   OpenAQ PM2.5 multi-year
  build_grid_m2.py      regrid -> grid_m2.nc
  build_datacube_m2.py  datacube_m2.zarr + target_pm25_m2.csv
  train_operator_m2.py  เทรน CLNOLowRank (rank-32)
  eval_operator_m2.py   attribution + inversion + comparison chart
  model_operator.py     CLNO base class (dual-form inversion สำหรับ G >> S)
data/raw_m2/            ข้อมูลดิบ M2 (gitignored)
data/processed_m2/      grid + datacube M2 (gitignored)
models/clno_m2.pt       โมเดล M2 trained
figures/attribution_m2_worst_day.png   แผนที่แหล่งที่มาหมอกควัน
figures/inversion_m2_vs_firms.png      emission inversion vs FIRMS
figures/model_comparison_m2.png        เทียบทุกโมเดล
```

## ถัดจาก M2

เพิ่ม satellite AOD (MODIS/MAIAC) · multi-task loss (PM10, O3) ·
temporal context (เพิ่ม LSTM encoder) · submission to journal
