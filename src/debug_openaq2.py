"""Debug: ดูว่า /v3/locations?bbox=... return sensors field หรือไม่"""
import requests, json, sys
sys.stdout.reconfigure(encoding="utf-8")

kp = r"C:\Users\mark\Desktop\internship\src\.keys"
key = ""
for line in open(kp):
    if line.startswith("OPENAQ_API_KEY"):
        key = line.split("=", 1)[1].strip()

H = {"X-API-Key": key}
BOX = "96.0,14.0,106.0,25.0"

r = requests.get("https://api.openaq.org/v3/locations",
                 headers=H,
                 params={"bbox": BOX, "parameters_id": 2, "limit": 3},
                 timeout=30)
data = r.json()
print(f"HTTP {r.status_code}, found {len(data.get('results', []))} locations")
for loc in data.get("results", [])[:2]:
    print(f"\n--- location {loc.get('id')} {loc.get('name','')[:30]} ---")
    print("  sensors field:", loc.get("sensors", "NOT PRESENT"))
    print("  top keys:", list(loc.keys()))

# Try fetching PM2.5 sensors for location 2328 via different endpoint
print("\n=== /v3/locations/2328/sensors ===")
r2 = requests.get("https://api.openaq.org/v3/locations/2328/sensors",
                  headers=H, timeout=30)
print(f"HTTP {r2.status_code}")
if r2.status_code == 200:
    data2 = r2.json()
    for s in data2.get("results", [])[:5]:
        print("  sensor id:", s.get("id"), " param:", s.get("parameter"))
else:
    print(r2.text[:200])
