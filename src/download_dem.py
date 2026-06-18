"""
ดึง Copernicus DEM GLO-30 (ความสูงภูมิประเทศ) ของกล่องเชียงใหม่
แหล่ง: AWS public bucket (ไม่ต้องใช้ key) -> ดาวน์โหลดผ่าน HTTPS ตรง ๆ ด้วย stdlib

แต่ละ tile = 1องศา x 1องศา (GeoTIFF/COG)
กล่อง 18-19.5N, 98-100E -> ต้องการ tile: N18/N19 x E098/E099
"""
import urllib.request
import os

BUCKET = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
OUT_DIR = "data/raw/dem"

# (lat, lon) ของมุมล่างซ้ายของแต่ละ tile ที่ครอบกล่องเชียงใหม่
TILES = [(18, 98), (18, 99), (19, 98), (19, 99)]


def tile_url(lat, lon):
    name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
    return f"{BUCKET}/{name}/{name}.tif", f"{name}.tif"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for lat, lon in TILES:
        url, fname = tile_url(lat, lon)
        dest = os.path.join(OUT_DIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[skip] {fname} (มีอยู่แล้ว)")
            continue
        print(f"[get ] {fname} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            mb = os.path.getsize(dest) / 1e6
            print(f"[ok  ] {fname}  ({mb:.1f} MB)")
        except Exception as e:
            print(f"[ERR ] {fname}: {type(e).__name__} {e}")


if __name__ == "__main__":
    main()
