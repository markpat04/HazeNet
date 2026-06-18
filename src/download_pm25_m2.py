"""
ดาวน์โหลด PM2.5 รายวันจาก OpenAQ v3 สำหรับ domain SEA (14-25N, 96-106E)
ครอบคลุม 5 ฤดูเผา ก.พ.-เม.ย. 2019-2023

Strategy:
  1. หาสถานีทั้งหมดใน bbox ที่มี PM2.5 และมีข้อมูลก่อน 2023-04-30
  2. ดึง /days ทีละสถานี กรองช่วงปี ก.พ.-เม.ย.
  3. เซฟแยกตามปี: data/raw_m2/pm25/pm25_{year}.csv

หมายเหตุ: Myanmar + Laos มีสถานีน้อยมาก (อาจแค่ 0-5 สถานี)
           Thailand จะมีสถานีหลัก (Air4Thai network) ~30-50 สถานี

key: src/.keys บรรทัด OPENAQ_API_KEY=xxxx

Run: conda run -n hazenet --no-capture-output python src/download_pm25_m2.py
"""
import os
import sys
import time
import requests
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_m2 import YEARS, ROOT, BOX

BASE    = "https://api.openaq.org/v3"
OUT_DIR = os.path.join(ROOT, "data", "raw_m2", "pm25")
BBOX    = f"{BOX['minx']},{BOX['miny']},{BOX['maxx']},{BOX['maxy']}"
PAUSE   = 1.2   # seconds between API calls


def get_key():
    kp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".keys")
    if os.path.exists(kp):
        for line in open(kp, encoding="utf-8"):
            if line.startswith("OPENAQ_API_KEY"):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("ไม่พบ OPENAQ_API_KEY ใน src/.keys")


def get_json(url, params, headers, tries=3):
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(10)
                continue
            return None
        except Exception:
            time.sleep(3)
    return None


EARLIEST_WIN = "2019-02-01"
LATEST_WIN   = "2023-04-30"

def find_stations(H: dict) -> list:
    """
    หาสถานี PM2.5 ในกล่อง SEA ที่มีข้อมูลคลุมช่วง 2019-2023
    กรอง: datetimeFirst <= 2023-04-30 (ไม่ใช่สถานีที่เพิ่งติดตั้งหลัง 2023)
          datetimeLast  >= 2019-02-01 (ยังมีข้อมูลในช่วงที่ต้องการ)
    """
    js = get_json(f"{BASE}/locations",
                  {"bbox": BBOX, "parameters_id": 2, "limit": 500}, H)
    if not js:
        return []
    locs = js["results"]
    cand = []
    for L in locs:
        c = L.get("coordinates", {})
        lat, lon = c.get("latitude"), c.get("longitude")
        if lat is None:
            continue
        if not (BOX["miny"] <= lat <= BOX["maxy"] and
                BOX["minx"] <= lon <= BOX["maxx"]):
            continue
        first = (L.get("datetimeFirst") or {}).get("utc", "")
        last  = (L.get("datetimeLast")  or {}).get("utc", "")
        # ต้องเปิดก่อนปลาย window และยังมีข้อมูลหลัง window เริ่ม
        if not first or first[:10] > LATEST_WIN:
            continue
        if not last or last[:10] < EARLIEST_WIN:
            continue
        cand.append(L)
    return cand


def parse_recs(recs: list) -> list:
    rows = []
    for rec in recs:
        period = rec.get("period") or {}
        dt_from = period.get("datetimeFrom") or {}
        day = (dt_from.get("local") or dt_from.get("utc") or
               rec.get("date") or "")[:10]
        val = rec.get("value") if rec.get("value") is not None else rec.get("average")
        if day and val is not None:
            try:
                rows.append({"date": day, "pm25": round(float(val), 1)})
            except (TypeError, ValueError):
                pass
    return rows


def fetch_station_days(sid: int, H: dict, debug: bool = False) -> pd.DataFrame:
    """ดึงข้อมูลรายวันของ sensor นี้ (limit=1000 max, paginate up to 3 pages)"""
    all_rows = []
    base_params = {
        "limit": 1000,
        "datetime_from": "2019-02-01",
        "datetime_to": "2023-04-30",
    }
    for page in range(1, 4):   # max 3000 records
        params = {**base_params, "page": page}
        data = get_json(f"{BASE}/sensors/{sid}/days", params, H)
        if not data:
            break
        recs = data.get("results", [])
        if not recs:
            break
        if debug and page == 1:
            import json as _json
            print(f"    DEBUG sensor {sid} p1 rec[0]: {_json.dumps(recs[0])[:200]}")
        all_rows.extend(parse_recs(recs))
        if len(recs) < 1000:
            break   # last page
        time.sleep(0.5)
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def is_burning_season(date_str: str) -> bool:
    """กรองเฉพาะ ก.พ.–เม.ย."""
    month = int(date_str[5:7])
    return month in (2, 3, 4)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    H = {"X-API-Key": get_key()}

    print("ค้นหาสถานีใน SEA domain ...")
    stations = find_stations(H)
    print(f"พบ {len(stations)} สถานีที่มีข้อมูล >= ก.พ. 2019")

    # เก็บข้อมูลทุกสถานี แยกตามปี
    rows_per_year = {y: [] for y in YEARS}

    for i, L in enumerate(stations):
        c   = L["coordinates"]
        lat = c["latitude"]
        lon = c["longitude"]
        loc_name = L.get("name", "")
        print(f"  [{i+1:3d}/{len(stations)}] {L['id']:>7}  "
              f"{loc_name[:28]:28}  ({lat:.2f},{lon:.2f})")

        sensors = [s for s in L.get("sensors", [])
                   if s.get("parameter", {}).get("name") == "pm25"]
        for s in sensors:
            time.sleep(PAUSE)
            debug_first = (i < 3)   # debug first 3 stations
            df = fetch_station_days(s["id"], H, debug=debug_first)
            if df.empty:
                continue
            df = df[df["date"].apply(is_burning_season)]
            if df.empty:
                continue

            df["locationId"] = L["id"]
            df["location"]   = loc_name
            df["lat"]        = lat
            df["lon"]        = lon
            df["sensor_id"]  = s["id"]

            for year in YEARS:
                sub = df[df["date"].str.startswith(str(year))]
                if not sub.empty:
                    rows_per_year[year].extend(sub.to_dict("records"))
            break   # ใช้ sensor แรกที่ได้ข้อมูล

    # บันทึกแยกตามปี
    for year in YEARS:
        rows = rows_per_year[year]
        if not rows:
            print(f"  [warn] {year}: ไม่มีข้อมูล")
            continue
        df_year = (pd.DataFrame(rows)
                   .drop_duplicates(["date", "locationId"])
                   .sort_values(["location", "date"])
                   .reset_index(drop=True))
        out = os.path.join(OUT_DIR, f"pm25_{year}.csv")
        df_year.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  [ok  ] {year}: {len(df_year)} แถว  "
              f"{df_year['locationId'].nunique()} สถานี  "
              f"PM2.5 avg={df_year['pm25'].mean():.0f} -> {os.path.basename(out)}")

    print("\nPM2.5 M2 download complete.")


if __name__ == "__main__":
    main()
