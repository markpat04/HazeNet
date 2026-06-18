"""Quick debug: ดู OpenAQ v3 sensor structure"""
import requests, json, sys
sys.stdout.reconfigure(encoding="utf-8")

kp = r"C:\Users\mark\Desktop\internship\src\.keys"
key = ""
for line in open(kp):
    if line.startswith("OPENAQ_API_KEY"):
        key = line.split("=", 1)[1].strip()

H = {"X-API-Key": key}

# Debug station 2328
r = requests.get("https://api.openaq.org/v3/locations/2328", headers=H, timeout=30)
data = r.json()
loc = data.get("results", [{}])[0]
print("=== station 2328 sensors ===")
for s in loc.get("sensors", []):
    print(" sensor id:", s.get("id"))
    print("  parameter:", s.get("parameter"))
    print()

# Try fetching sensor days for first sensor
sensors = loc.get("sensors", [])
if sensors:
    sid = sensors[0]["id"]
    print(f"=== /sensors/{sid}/days (first 3 records) ===")
    r2 = requests.get(f"https://api.openaq.org/v3/sensors/{sid}/days",
                      headers=H, params={"limit": 3}, timeout=60)
    data2 = r2.json()
    print("HTTP", r2.status_code)
    recs = data2.get("results", [])
    print(f"results count: {len(recs)}")
    for rec in recs[:3]:
        print(json.dumps(rec, indent=2))
