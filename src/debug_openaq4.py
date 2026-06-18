"""Debug: ทดสอบ limit=2000 vs limit=1000 vs limit=100"""
import requests, sys
sys.stdout.reconfigure(encoding="utf-8")

kp = r"C:\Users\mark\Desktop\internship\src\.keys"
key = ""
for line in open(kp):
    if line.startswith("OPENAQ_API_KEY"):
        key = line.split("=", 1)[1].strip()

H = {"X-API-Key": key}
sid = 1304242  # PM2.5 sensor ของสถานี 2328

for lim in [2000, 1000, 500, 100]:
    r = requests.get(f"https://api.openaq.org/v3/sensors/{sid}/days",
                     headers=H, params={"limit": lim}, timeout=60)
    data = r.json()
    n = len(data.get("results", []))
    meta = data.get("meta", {})
    print(f"limit={lim}: HTTP {r.status_code}  results={n}  "
          f"found={meta.get('found', '?')}  total={meta.get('total','?')}")
    if r.status_code != 200:
        print("  error:", r.text[:200])
