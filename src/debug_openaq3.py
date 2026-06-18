"""Debug: ทดสอบ /sensors/1304242/days (PM2.5 sensor ของสถานี 2328)"""
import requests, json, sys
sys.stdout.reconfigure(encoding="utf-8")

kp = r"C:\Users\mark\Desktop\internship\src\.keys"
key = ""
for line in open(kp):
    if line.startswith("OPENAQ_API_KEY"):
        key = line.split("=", 1)[1].strip()

H = {"X-API-Key": key}

# Test PM2.5 sensor from station 2328
pm25_sensor_id = 1304242
print(f"=== /sensors/{pm25_sensor_id}/days ===")
r = requests.get(f"https://api.openaq.org/v3/sensors/{pm25_sensor_id}/days",
                 headers=H, params={"limit": 5}, timeout=60)
print(f"HTTP {r.status_code}")
data = r.json()
recs = data.get("results", [])
print(f"results count: {len(recs)}")
if recs:
    print("First record:")
    print(json.dumps(recs[0], indent=2))
else:
    print("Empty! meta:", data.get("meta"))
    # Try with date range
    print("\nTrying with date range (2019-02-01 to 2019-04-30):")
    r2 = requests.get(f"https://api.openaq.org/v3/sensors/{pm25_sensor_id}/days",
                      headers=H,
                      params={"limit": 5,
                              "datetime_from": "2019-02-01",
                              "datetime_to": "2019-04-30"},
                      timeout=60)
    print(f"HTTP {r2.status_code}")
    data2 = r2.json()
    print(f"results count: {len(data2.get('results', []))}")
    if data2.get("results"):
        print(json.dumps(data2["results"][0], indent=2))
    else:
        print("Also empty! Checking info endpoint...")
        # Check sensor info
        r3 = requests.get(f"https://api.openaq.org/v3/sensors/{pm25_sensor_id}",
                          headers=H, timeout=30)
        print(f"Sensor info HTTP {r3.status_code}")
        print(r3.text[:500])

# Also test old sensor from station 2328 that had CO data (to verify endpoint works)
print(f"\n=== /sensors/4246/days (PM10 sensor, expected to have data) ===")
r4 = requests.get("https://api.openaq.org/v3/sensors/4246/days",
                  headers=H, params={"limit": 3}, timeout=60)
print(f"HTTP {r4.status_code}")
data4 = r4.json()
recs4 = data4.get("results", [])
print(f"results count: {len(recs4)}")
if recs4:
    p = recs4[0].get("period", {})
    dt = (p.get("datetimeFrom") or {}).get("local", "?")[:10]
    val = recs4[0].get("value")
    print(f"  first record: {dt}  value={val}")
