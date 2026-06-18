"""
ดึงค่า PM2.5 ล่าสุดจาก Air4Thai (กรมควบคุมมลพิษ) - endpoint สาธารณะ ไม่ต้องใช้ key
หมายเหตุ: นี่คือ snapshot "ล่าสุด" (พิสูจน์ว่าเข้าถึงได้) - ข้อมูลย้อนหลังเป็นสัปดาห์
ต้องใช้ history query แยก (จะทำในขั้นถัดไป)

กรองเฉพาะสถานีในกล่องเชียงใหม่ -> เซฟ JSON ดิบ + CSV ที่กรองแล้ว
"""
import urllib.request
import json
import csv
import os
import sys

# Windows console เป็น cp1252 -> บังคับ UTF-8 กันพิมพ์ภาษาไทยแล้ว error
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

URL = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php"
OUT_DIR = "data/raw/pm25"

# กล่องเชียงใหม่ (ตรงกับ config.yaml)
LAT_MIN, LAT_MAX = 18.0, 19.5
LON_MIN, LON_MAX = 98.0, 100.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[get ] Air4Thai getNewAQI_JSON ...")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        print(f"[ERR ] เข้าถึง Air4Thai ไม่ได้: {type(e).__name__} {e}")
        print("       -> ใช้ OpenAQ เป็นสำรอง (ดู download.py)")
        return

    data = json.loads(raw)
    # เซฟ JSON ดิบทั้งหมด
    raw_path = os.path.join(OUT_DIR, "air4thai_latest_all.json")
    with open(raw_path, "wb") as f:
        f.write(raw)
    stations = data.get("stations", [])
    print(f"[ok  ] ได้ทั้งหมด {len(stations)} สถานี -> {raw_path}")

    # กรองเฉพาะในกล่องเชียงใหม่
    rows = []
    for s in stations:
        try:
            lat = float(s.get("lat"))
            lon = float(s.get("long"))
        except (TypeError, ValueError):
            continue
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            last = s.get("AQILast", {})
            pm25 = last.get("PM25", {})
            rows.append({
                "stationID": s.get("stationID"),
                "nameTH": s.get("nameTH"),
                "areaTH": s.get("areaTH"),
                "lat": lat, "lon": lon,
                "datetime": f"{last.get('date','')} {last.get('time','')}".strip(),
                "pm25": pm25.get("value"),
            })

    csv_path = os.path.join(OUT_DIR, "air4thai_chiangmai_latest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["stationID", "nameTH", "areaTH",
                                          "lat", "lon", "datetime", "pm25"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok  ] สถานีในกล่องเชียงใหม่: {len(rows)} สถานี -> {csv_path}")
    for r in rows:
        print(f"       - {r['nameTH']}  PM2.5={r['pm25']}  ({r['datetime']})")


if __name__ == "__main__":
    main()
