"""Debug: ทดสอบ date range filter สำหรับ /sensors/days"""
import requests, sys
sys.stdout.reconfigure(encoding="utf-8")

kp = r"C:\Users\mark\Desktop\internship\src\.keys"
key = ""
for line in open(kp):
    if line.startswith("OPENAQ_API_KEY"):
        key = line.split("=", 1)[1].strip()

H = {"X-API-Key": key}
sid = 1304242  # PM2.5 sensor ของสถานี 2328

# Test with date range filter
for params in [
    {"limit": 100, "datetime_from": "2019-02-01", "datetime_to": "2023-04-30"},
    {"limit": 100, "date_from": "2019-02-01", "date_to": "2023-04-30"},
    {"limit": 100, "period_from": "2019-02-01", "period_to": "2023-04-30"},
]:
    r = requests.get(f"https://api.openaq.org/v3/sensors/{sid}/days",
                     headers=H, params=params, timeout=60)
    data = r.json()
    recs = data.get("results", [])
    print(f"params={params}: HTTP {r.status_code}  results={len(recs)}")
    if recs:
        days = [(r.get("period",{}).get("datetimeFrom",{}).get("local","?")[:10]) for r in recs[:3]]
        print(f"  first dates: {days}")
    if r.status_code not in (200, 422):
        print("  response:", r.text[:200])
    elif r.status_code == 422:
        print("  error:", r.json().get("detail", "?"))
