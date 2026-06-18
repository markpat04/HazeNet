"""
ดึงจุดไฟ/FRP จาก NASA FIRMS (Area API) ของกล่องเชียงใหม่
ต้องใช้ MAP_KEY (ฟรี): https://firms.modaps.eosdis.nasa.gov/api/area/

หมายเหตุสำคัญ: Area API จำกัด day range <= 5 วัน/คำขอ
-> โค้ดนี้แบ่งช่วงเป็นก้อนละ <=5 วันแล้วรวมกันให้อัตโนมัติ

วิธีใส่ key:
  - สร้างไฟล์ src/.keys ใส่บรรทัด:  FIRMS_MAP_KEY=xxxx
  - หรือ env var FIRMS_MAP_KEY
"""
import urllib.request
import urllib.error
import os
import sys
import io
from datetime import date, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_DIR = "data/raw/firms"
AREA = "98,18,100,19.5"          # west,south,east,north (กล่องเชียงใหม่)
SOURCE = "VIIRS_SNPP_SP"         # standard product (ข้อมูลย้อนหลัง)
START_DATE = date(2023, 3, 19)   # ตรงกับ window PM2.5 ที่มีข้อมูลจริง
END_DATE = date(2023, 3, 31)     # รวมปลายทาง
MAX_RANGE = 5                    # FIRMS จำกัด <=5 วัน/คำขอ


def get_key():
    key = os.environ.get("FIRMS_MAP_KEY")
    if key:
        return key.strip()
    kp = os.path.join(os.path.dirname(__file__), ".keys")
    if os.path.exists(kp):
        for line in open(kp, encoding="utf-8"):
            if line.startswith("FIRMS_MAP_KEY"):
                return line.split("=", 1)[1].strip()
    return None


def chunks(start, end, step):
    cur = start
    while cur <= end:
        n = min(step, (end - cur).days + 1)
        yield cur, n
        cur += timedelta(days=n)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    key = get_key()
    if not key:
        print("[ข้าม] ยังไม่มี FIRMS_MAP_KEY (ดู data/README.md)")
        return

    import pandas as pd
    frames = []
    for cstart, n in chunks(START_DATE, END_DATE, MAX_RANGE):
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
               f"{key}/{SOURCE}/{AREA}/{n}/{cstart.isoformat()}")
        print(f"[get ] {cstart.isoformat()} +{n}d ...")
        try:
            txt = urllib.request.urlopen(url, timeout=120).read().decode()
            df = pd.read_csv(io.StringIO(txt))
            print(f"       -> {len(df)} จุด")
            frames.append(df)
        except urllib.error.HTTPError as e:
            print(f"[ERR ] HTTP {e.code}: {e.read().decode()[:200]}")
        except Exception as e:
            print(f"[ERR ] {type(e).__name__}: {e}")

    if not frames:
        print("[ERR ] ไม่ได้ข้อมูลเลย")
        return
    full = pd.concat(frames, ignore_index=True).drop_duplicates()
    dest = os.path.join(OUT_DIR, f"firms_{SOURCE}_2023-03-19_31.csv")
    full.to_csv(dest, index=False)
    print(f"[ok  ] รวม {len(full)} จุดไฟ -> {dest}")
    if "frp" in full.columns:
        print(f"       FRP รวม={full.frp.sum():.0f}  สูงสุด={full.frp.max():.0f}  "
              f"วันที่={full.acq_date.min()}..{full.acq_date.max()}")


if __name__ == "__main__":
    main()
