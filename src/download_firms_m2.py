"""
ดาวน์โหลดจุดไฟ FIRMS (VIIRS SNPP) สำหรับ domain SEA, ฤดูเผา 2019-2023

API จำกัด <= 10 วัน/คำขอ -> แบ่งก้อนอัตโนมัติ
ผลลัพธ์: data/raw_m2/firms/firms_VIIRS_SNPP_{year}.csv  (1 ไฟล์ต่อปี)

key: src/.keys บรรทัด FIRMS_MAP_KEY=xxxx

Run: conda run -n hazenet --no-capture-output python src/download_firms_m2.py
"""
import urllib.request
import urllib.error
import os
import sys
import io
import time
from datetime import date, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import YEARS, ROOT, BOX

OUT_DIR = os.path.join(ROOT, "data", "raw_m2", "firms")
SOURCE  = "VIIRS_SNPP_SP"
MAX_RANGE = 5    # FIRMS area API จำกัด <= 5 วัน/คำขอ (400 ถ้าเกิน)

# FIRMS area API จำกัด <= 10° ต่อด้าน -> แบ่งเป็น 2 sub-box (north/south)
# Sub-box format: "west,south,east,north"
SUB_BOXES = [
    f"{BOX['minx']},{BOX['miny']},{BOX['maxx']},20.0",   # south: 14-20°N
    f"{BOX['minx']},20.0,{BOX['maxx']},{BOX['maxy']}",   # north: 20-25°N
]


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


def chunks(start: date, end: date, step: int):
    cur = start
    while cur <= end:
        n = min(step, (end - cur).days + 1)
        yield cur, n
        cur += timedelta(days=n)


def fetch_chunk(key: str, area: str, start: date, n_days: int) -> str:
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{key}/{SOURCE}/{area}/{n_days}/{start.isoformat()}")
    for attempt in range(3):
        try:
            return urllib.request.urlopen(url, timeout=120).read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (attempt + 1))
            elif e.code == 400:
                # Print error body for debugging
                body = e.read().decode()[:300]
                print(f"  [400] area={area} n={n_days} start={start}: {body}")
                return ""
            else:
                print(f"  [HTTP {e.code}] {url}")
                return ""
        except Exception as ex:
            print(f"  [ERR ] attempt {attempt+1}: {ex}")
            time.sleep(5)
    return ""


def main():
    import pandas as pd
    os.makedirs(OUT_DIR, exist_ok=True)
    key = get_key()
    if not key:
        print("[ERR ] ไม่พบ FIRMS_MAP_KEY (src/.keys หรือ env var)")
        return

    for year in YEARS:
        out = os.path.join(OUT_DIR, f"firms_{SOURCE}_{year}.csv")
        if os.path.exists(out) and os.path.getsize(out) > 500:
            n = len(pd.read_csv(out))
            print(f"[skip] {os.path.basename(out)}  ({n} จุด)")
            continue

        season_start = date(year, 2, 1)
        season_end   = date(year, 4, 30)
        print(f"\n[year] {year}  ({season_start} .. {season_end})")

        frames = []
        for area in SUB_BOXES:
            print(f"  sub-box: {area}")
            for cstart, n in chunks(season_start, season_end, MAX_RANGE):
                txt = fetch_chunk(key, area, cstart, n)
                if not txt:
                    print(f"    [warn] {cstart} +{n}d -> ไม่ได้ข้อมูล")
                    continue
                try:
                    df = pd.read_csv(io.StringIO(txt))
                    if len(df) > 0 and "latitude" in df.columns:
                        print(f"    {cstart.isoformat()} +{n}d -> {len(df)} จุด")
                        frames.append(df)
                except Exception as e:
                    print(f"    [ERR ] parse {cstart}: {e}")
                time.sleep(0.5)

        if not frames:
            print(f"  [ERR ] ไม่ได้ข้อมูลทั้งปี {year}")
            continue

        full = pd.concat(frames, ignore_index=True).drop_duplicates()
        full.to_csv(out, index=False)
        total_frp = full["frp"].sum() if "frp" in full.columns else 0
        print(f"  [ok  ] {len(full)} จุด  FRP รวม={total_frp:.0f}  -> {os.path.basename(out)}")

    print("\nFIRMS M2 download complete.")


if __name__ == "__main__":
    main()
