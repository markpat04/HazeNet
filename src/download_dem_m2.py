"""
ดาวน์โหลด Copernicus GLO-30 DEM tiles ที่ครอบ domain SEA
(lat 14-25N, lon 96-106E) จาก AWS public bucket

Tiles ที่เป็น ocean/ไม่มีข้อมูลจะ 404 -> ข้ามโดยไม่ error
ผลลัพธ์: data/raw_m2/dem/*.tif  (แต่ละ tile 1°×1°)

Run: conda run -n hazenet --no-capture-output python src/download_dem_m2.py
"""
import urllib.request
import urllib.error
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BUCKET  = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "raw_m2", "dem")

# DEM tiles needed: 1° grid, round down lat/lon of domain corners
# domain: lat 14-25N, lon 96-106E -> tiles at lat 14..24, lon 96..105
LAT_TILES = list(range(14, 25))   # 11 rows
LON_TILES = list(range(96, 106))  # 10 cols  → up to 110 tiles


def tile_url(lat: int, lon: int):
    name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
    return f"{BUCKET}/{name}/{name}.tif", f"{name}.tif"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok = n_skip = n_miss = n_err = 0

    total = len(LAT_TILES) * len(LON_TILES)
    print(f"DEM M2: {total} tiles to check  (lat {LAT_TILES[0]}-{LAT_TILES[-1]}, "
          f"lon {LON_TILES[0]}-{LON_TILES[-1]})")

    for lat in LAT_TILES:
        for lon in LON_TILES:
            url, fname = tile_url(lat, lon)
            dest = os.path.join(OUT_DIR, fname)

            if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                n_skip += 1
                continue

            try:
                urllib.request.urlretrieve(url, dest)
                mb = os.path.getsize(dest) / 1e6
                print(f"[ok  ] {fname}  ({mb:.1f} MB)")
                n_ok += 1
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    n_miss += 1   # ocean/no-data tile
                else:
                    print(f"[ERR ] {fname}: HTTP {e.code}")
                    n_err += 1
            except Exception as e:
                print(f"[ERR ] {fname}: {type(e).__name__} {e}")
                n_err += 1

    print(f"\nDEM M2 done: {n_ok} downloaded, {n_skip} existed, "
          f"{n_miss} ocean/missing, {n_err} errors")
    n_total_ok = n_ok + n_skip
    print(f"Total tiles available: {n_total_ok}")


if __name__ == "__main__":
    main()
