"""
CDS ERA5 fetch — one netCDF per (year, month).

Single-levels: 10m winds, boundary-layer height, total precip, 2m temp, 2m dewpoint.
Pressure-level: temperature @ 850 hPa (for the inversion-strength feature).

RH is derived from t2m + d2m in the grid stage. Needs ~/.cdsapirc.
Resume-able: skips months already downloaded.
"""
from __future__ import annotations

import os
import calendar

SINGLE_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind",
               "boundary_layer_height", "total_precipitation",
               "2m_temperature", "2m_dewpoint_temperature"]


def _area(cfg):
    # [North, West, South, East], padded a touch beyond the domain for interp
    return [cfg.lat1 + 0.5, cfg.lon0 - 0.5, cfg.lat0 - 0.5, cfg.lon1 + 0.5]


def fetch(cfg) -> None:
    out_dir = os.path.join(cfg.raw_dir, "era5")
    os.makedirs(out_dir, exist_ok=True)
    try:
        import cdsapi
    except ImportError:
        print("[era5] cdsapi not installed -> pip install 'cdsapi>=0.7'"); return
    if not os.path.exists(os.path.expanduser("~/.cdsapirc")):
        print("[era5] ~/.cdsapirc missing (need CDS url+key) -> skipping"); return

    c = cdsapi.Client()
    area = _area(cfg)
    tasks = [(y, m) for y in cfg.years for m in cfg.season_months]
    print(f"[era5] {len(tasks)} (year,month) requests  area={area}")

    for year, month in tasks:
        mm = f"{month:02d}"
        ndays = calendar.monthrange(year, month)[1]
        days = [f"{d:02d}" for d in range(1, ndays + 1)]
        times = [f"{h:02d}:00" for h in range(0, 24, 3)]

        sl = os.path.join(out_dir, f"era5_sl_{year}-{mm}.nc")
        if not (os.path.exists(sl) and os.path.getsize(sl) > 10_000):
            print(f"[era5] single-levels {year}-{mm} ...")
            try:
                c.retrieve("reanalysis-era5-single-levels",
                           {"product_type": "reanalysis", "variable": SINGLE_VARS,
                            "year": str(year), "month": mm, "day": days,
                            "time": times, "area": area, "format": "netcdf"}, sl)
            except Exception as e:
                print(f"[era5] ERR sl {year}-{mm}: {e}")
        else:
            print(f"[era5] skip sl {year}-{mm}")

        pl = os.path.join(out_dir, f"era5_pl_{year}-{mm}.nc")
        if not (os.path.exists(pl) and os.path.getsize(pl) > 10_000):
            print(f"[era5] pressure-levels {year}-{mm} ...")
            try:
                c.retrieve("reanalysis-era5-pressure-levels",
                           {"product_type": "reanalysis", "variable": "temperature",
                            "pressure_level": "850", "year": str(year), "month": mm,
                            "day": days, "time": times, "area": area,
                            "format": "netcdf"}, pl)
            except Exception as e:
                print(f"[era5] ERR pl {year}-{mm}: {e}")
        else:
            print(f"[era5] skip pl {year}-{mm}")
    print("[era5] done")
