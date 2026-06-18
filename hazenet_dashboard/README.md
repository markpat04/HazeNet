# 🛰️ HazeNet — Mission Control (local dashboard)

เว็บไซต์ติดตามโปรเจกต์ HazeNet แบบ real-time สำหรับใช้คนเดียวบนเครื่องตัวเอง
ออกแบบให้ **ลิงก์กับโค้ด/ข้อมูลจริงในโฟลเดอร์ `internship/`** และ **อัปเดตเมื่อไฟล์เปลี่ยน**

---

## ▶️ วิธีรัน (ไม่ต้องลง dependency เพิ่ม)

ดับเบิลคลิก **`run.bat`** หรือสั่ง:

```bash
C:/Users/mark/miniconda3/Scripts/conda run -n hazenet --no-capture-output python hazenet_dashboard/serve.py
```

เปิดเบราว์เซอร์ที่ **http://localhost:8765** (เปิดให้อัตโนมัติ)

> ใช้แค่ Python stdlib + xarray/pandas/numpy ที่มีใน env `hazenet` อยู่แล้ว — **ไม่ต้อง pip install Flask/อะไรเลย**

---

## 🧭 ปรัชญาออกแบบ (ทำไมเป็นแบบนี้)

| หลักการ | การตัดสินใจ |
|---|---|
| ต้อง "รันได้แน่นอน" | zero-dependency (stdlib `http.server`) — ไม่พึ่ง framework ที่อาจลงไม่ผ่าน |
| real-time | frontend poll ทุก 3 วิ (pipeline/training/figures) + 30 วิ (EDA หนักกว่า) |
| ลิงก์โค้ดจริง | สแกน `src/`, `data/`, `models/`, `figures/` ตาม **mtime จริง** |
| รู้เมื่อโค้ดเปลี่ยน | ตรวจ **stale**: ถ้า `.py` หรืออินพุตใหม่กว่า output → ขึ้น 🟡 "ควร re-run" |
| ไม่ทำข้อมูลพัง | dashboard **อ่านอย่างเดียว** (read-only) ไม่แก้ไฟล์โปรเจกต์ |

---

## 📋 ฟีเจอร์ (v0 = ทำแล้ว / v1–v2 = roadmap)

### 1) 🗺️ Mission Control (overview) — ✅ v0
- การ์ด KPI: stage พร้อม / stale / missing
- **Pipeline DAG**: 6 stage (ingest → grid → datacube → train → eval → viz) แต่ละ node บอกสถานะ + ไฟล์ output + ขนาด + เวลาล่าสุด + สคริปต์
- กล่อง **Gate W2** (bias ปีต่ำ +45→≤25)

### 2) 📦 Data & EDA — ✅ v0
- KPI: ขนาดกริด, G, จำนวนวัน, จำนวนสถานี (ไทย/อื่น)
- กราฟ **PM2.5 เฉลี่ยรายวัน** + **FRP รวมรายวัน** (จากข้อมูลจริง)
- ตาราง **per-channel stats** (min/max/mean/NaN%) ของทุกช่อง datacube
- ตาราง **PM2.5 รายปี** (เห็น non-stationarity ตรงๆ)
- _(v1)_ histogram ราย channel · แผนที่สถานี interactive · missing-data heatmap ราย AOD/วัน

### 3) 📈 Training Monitor — ✅ v0 (ต้องต่อ `tracker.py`)
- Loss curve สด (train/test, log-scale) อัปเดตระหว่างเทรน
- KPI: สถานะ, โมเดล, epoch ปัจจุบัน, test loss
- ตาราง run history
- _(v1)_ ETA + throughput · เทียบหลาย run ซ้อนกราฟ · LR schedule · GPU mem

### 4) 🧪 Systematic Experiments — ✅ v0
- ทะเบียนการทดลอง (seed = ผลจริง Phase0/M2/LOYO + run ใหม่อัตโนมัติ)
- ตารางเทียบ + bar chart **baseline ladder (MAE)**
- _(v1)_ map กับ experiment matrix E1–E12 · เลือก 2 run มา diff · ปุ่ม export ตารางผล (สำหรับ paper) · gate auto-check

### 5) 🖼️ Visualizations — ✅ v0
- แกลเลอรีรูป/อนิเมชั่นทั้งหมดใน `figures/` (รูปใหม่ขึ้น **NEW**)
- เล่น `smoke_animation.gif` ได้เลย
- _(v1)_ slider เลือกวัน → เรนเดอร์ attribution map ของวันนั้น · เทียบ pred vs obs map

### 6) 🔧 Code & Pipeline — ✅ v0
- รายการสคริปต์ทุก stage + docstring + เวลาแก้ + ไฟล์ที่ผลิต

### 🔮 Roadmap v2 (ของเล่นใหญ่)
- ปุ่ม **"Run stage"** กดรันสคริปต์จาก dashboard (subprocess) + สตรีม log สด (SSE)
- **Attribution explorer**: คลิกสถานี → ดูว่าควันมาจากช่องไหน %
- **What-if console**: ปิดแหล่งไฟ → ดูฝุ่นเมืองลด %
- เก็บ snapshot ผลแต่ละ commit (ผูกกับ git) → ดู progress ข้ามเวลา

---

## 🔌 ต่อ tracker เพื่อดูการเทรนสด

เพิ่มในหัว `src/train_operator_m2.py` (และ `sweep_m2.py`):

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hazenet_dashboard"))
from tracker import Run

run = Run(model="CLNOLowRank", config={"rank": RANK, "hidden": HIDDEN, "lr": 1e-3})
```

ในลูป epoch:
```python
run.log_epoch(ep, train=tr_losses[-1], test=te_losses[-1], lr=sched.get_last_lr()[0])
```

หลังเทรนเสร็จ:
```python
run.finish(metrics={"MAE": mae_te, "RMSE": rmse_te}, status="done")
```

แค่นี้ Training Monitor จะเห็น loss curve ขยับสดทันที + run ขึ้นในตาราง Experiments อัตโนมัติ

---

## 🗂️ โครงสร้างโฟลเดอร์

```
hazenet_dashboard/
├── serve.py              # server (stdlib) + APIs + สแกนไฟล์
├── tracker.py            # logger ให้สคริปต์เทรนเรียก (real-time)
├── index.html            # หน้า dashboard (SPA)
├── static/
│   ├── style.css         # ธีมเข้ม
│   └── app.js            # poll APIs + วาดกราฟ (ไม่มี dep)
├── experiments_seed.json # ผลจริงที่ทราบ (Phase0/M2/LOYO)
├── runs/                 # log การเทรน (สร้างโดย tracker)
├── cache/               # cache EDA (อัตโนมัติ)
├── run.bat
└── README.md (ไฟล์นี้)
```

## 🔁 "อัปเดตเมื่อโค้ดเปลี่ยน" ทำงานยังไง
- ทุก endpoint สแกน `os.path.getmtime` ของไฟล์จริงสดๆ ตอนเรียก (ไม่ cache pipeline)
- EDA cache ผูกกับ mtime ของ datacube/grid/target → ถ้า re-build ข้อมูล cache จะ invalidate เอง
- frontend poll → เห็นการเปลี่ยนภายใน 3 วินาที
