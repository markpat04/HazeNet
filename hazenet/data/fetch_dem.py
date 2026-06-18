"""
DEM (Copernicus GLO-30) — static, domain-fixed. The tiles already on disk for
the SEA domain are reused as-is (the RunPod domain is identical). If none are
present, point the user at the existing one-off downloader.
"""
from __future__ import annotations

import os
import glob


def fetch(cfg) -> None:
    dem_dir = os.path.join(cfg.raw_dir, "dem")
    tiles = glob.glob(os.path.join(dem_dir, "*.tif"))
    if tiles:
        print(f"[dem] {len(tiles)} tiles present -> reuse"); return
    # also accept the existing m2 DEM (same domain) as a fallback source
    from ..config import ROOT
    alt = glob.glob(os.path.join(ROOT, "data", "raw_m2", "dem", "*.tif"))
    if alt:
        print(f"[dem] none in {dem_dir}; {len(alt)} tiles exist in data/raw_m2/dem "
              f"(same domain) — point paths.raw_dir there or copy them.")
        return
    print("[dem] no DEM tiles found. Run the one-off downloader: "
          "python src/download_dem_m2.py  (Copernicus GLO-30, static).")
