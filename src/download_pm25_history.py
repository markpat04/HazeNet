"""
ดึง PM2.5 ย้อนหลัง (= target จริง) ของสถานีในกล่องเชียงใหม่ ช่วง 8-15 มี.ค. 2023
แหล่ง: OpenAQ v3 API (ต้องมี OPENAQ_API_KEY ใน src/.keys)

bug ที่เจอ + วิธีแก้:
- Air4Thai history ย้อนหลังได้แค่ ~ไม่กี่เดือน -> ใช้กับฤดูเผา 2023 ไม่ได้
- OpenAQ /days "เพิกเฉย date filter" (คืนตั้งแต่ต้น) -> ดึงมา limit ใหญ่แล้ว "กรองเองฝั่ง client"
- bbox ลากเอา sensor ชุมชน (เพิ่งติดตั้งปี 2024+) มาด้วย -> คัดด้วย datetimeFirst<=2023
- sensor 1304xxx เก็บ PM2.5 *รายวัน* ครอบ 2021-2026 (รายชั่วโมงปี 2023 ไม่พร้อม)

ออก: data/raw/pm25/pm25_daily_2023-03-08_15.csv
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

BASE = "https://api.openaq.org/v3"
OUT = "data/raw/pm25/pm25_daily_2023-03-19_31.csv"
BBOX = "98,18,100,19.5"
# เป้าเดิม 03-08..15 "ไม่มีข้อมูล" (รูข้อมูล ก.พ.-ต้น มี.ค. 2023)
# -> ขยับเป็น 03-19..31 ซึ่งมีข้อมูลครบ + เป็นเหตุการณ์หมอกควันรุนแรง (PM2.5 พุ่งถึง 170)
WIN_FROM = "2023-03-19"
WIN_TO = "2023-03-31"
PAUSE = 1.3


def get_key():
    for line in open(os.path.join(os.path.dirname(__file__), ".keys"), encoding="utf-8"):
        if line.startswith("OPENAQ_API_KEY"):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ไม่พบ OPENAQ_API_KEY")


def get_json(url, params, headers, tries=3):
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5); continue
            return None
        except Exception:
            time.sleep(2)
    return None


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    H = {"X-API-Key": get_key()}

    js = get_json(f"{BASE}/locations", {"bbox": BBOX, "parameters_id": 2,
                                        "limit": 200}, H)
    if not js:
        print("[ERR] เรียก locations ไม่ได้"); return
    locs = js["results"]

    # คัดเฉพาะสถานีที่ (ก) อยู่ในกล่องจริง (ข) มีข้อมูลตั้งแต่ก่อน 8 มี.ค. 2023
    cand = []
    for L in locs:
        c = L.get("coordinates", {})
        lat, lon = c.get("latitude"), c.get("longitude")
        if lat is None or not (18.0 <= lat <= 19.5 and 98.0 <= lon <= 100.0):
            continue
        first = (L.get("datetimeFirst") or {}).get("utc", "")
        last = (L.get("datetimeLast") or {}).get("utc", "")
        if first and first[:10] <= WIN_FROM and last and last[:10] >= WIN_TO:
            cand.append(L)
    print(f"สถานีในกล่องที่มีข้อมูลครอบ มี.ค.2023: {len(cand)} (จาก {len(locs)})")

    rows = []
    for L in cand:
        c = L["coordinates"]
        sensors = [s for s in L.get("sensors", [])
                   if s.get("parameter", {}).get("name") == "pm25"]
        n_win = 0
        for s in sensors:
            time.sleep(PAUSE)
            # ดึงรายวันก้อนใหญ่ (เรียงจากต้น) แล้วกรองช่วงเองฝั่ง client
            data = get_json(f"{BASE}/sensors/{s['id']}/days",
                            {"limit": 1000}, H)
            recs = (data or {}).get("results", [])
            for rec in recs:
                day = rec["period"]["datetimeFrom"]["local"][:10]
                if WIN_FROM <= day <= WIN_TO:
                    rows.append({"date": day, "locationId": L["id"],
                                 "location": L.get("name", ""),
                                 "lat": c["latitude"], "lon": c["longitude"],
                                 "pm25": round(rec["value"], 1)})
                    n_win += 1
            if n_win:
                break
        print(f"  {L['id']:>7} {L.get('name','')[:30]:30} -> {n_win} วันในช่วง")

    if not rows:
        print("[ERR] ไม่ได้ข้อมูลในช่วงเลย"); return
    df = pd.DataFrame(rows).drop_duplicates(["date", "locationId"]).sort_values(
        ["location", "date"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n[ok] {len(df)} แถว ({df.location.nunique()} สถานี) -> {OUT}")
    print(f"     ช่วงวันที่ {df.date.min()}..{df.date.max()}")
    print(f"     PM2.5 เฉลี่ย {df.pm25.mean():.1f}  สูงสุด {df.pm25.max():.1f} µg/m³")
    print("     ต่อสถานี:")
    for loc, g in df.groupby("location"):
        print(f"       {loc[:32]:32} {len(g)} วัน  avg {g.pm25.mean():.0f}")


if __name__ == "__main__":
    main()
