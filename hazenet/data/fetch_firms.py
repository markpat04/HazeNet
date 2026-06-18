"""
FIRMS VIIRS fire/FRP fetch — one CSV per year (resume-able).

Area API limits: ≤10° per side and ≤10 days per request, so the domain is
split into N/S sub-boxes and the season into ≤5-day chunks.
Key: FIRMS_MAP_KEY (env or src/.keys).
"""
from __future__ import annotations

import os
import io
import time
import urllib.request
import urllib.error
from datetime import date, timedelta

from . import get_key

SOURCE = "VIIRS_SNPP_SP"
MAX_RANGE = 5


def _sub_boxes(cfg):
    mid = round((cfg.lat0 + cfg.lat1) / 2, 1)
    return [f"{cfg.lon0},{cfg.lat0},{cfg.lon1},{mid}",
            f"{cfg.lon0},{mid},{cfg.lon1},{cfg.lat1}"]


def _chunks(start: date, end: date, step: int):
    cur = start
    while cur <= end:
        n = min(step, (end - cur).days + 1)
        yield cur, n
        cur += timedelta(days=n)


def _fetch_chunk(key, area, start, n):
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{key}/{SOURCE}/{area}/{n}/{start.isoformat()}")
    for attempt in range(3):
        try:
            return urllib.request.urlopen(url, timeout=120).read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                print(f"  [HTTP {e.code}] {area} {start}+{n}"); return ""
        except Exception as ex:
            print(f"  [err] {ex}"); time.sleep(5)
    return ""


def _month_ranges(year, months):
    """Contiguous (start,end) date ranges for the season months within a year."""
    import calendar
    out = []
    for m in months:
        out.append((date(year, m, 1), date(year, m, calendar.monthrange(year, m)[1])))
    return out


def fetch(cfg) -> None:
    import pandas as pd
    out_dir = os.path.join(cfg.raw_dir, "firms")
    os.makedirs(out_dir, exist_ok=True)
    key = get_key("FIRMS_MAP_KEY")
    if not key:
        print("[firms] no FIRMS_MAP_KEY -> skipping"); return

    for year in cfg.years:
        out = os.path.join(out_dir, f"firms_{SOURCE}_{year}.csv")
        if os.path.exists(out) and os.path.getsize(out) > 500:
            print(f"[firms] skip {year}"); continue
        frames = []
        for area in _sub_boxes(cfg):
            for (s, e) in _month_ranges(year, cfg.season_months):
                for cstart, n in _chunks(s, e, MAX_RANGE):
                    txt = _fetch_chunk(key, area, cstart, n)
                    if not txt:
                        continue
                    try:
                        df = pd.read_csv(io.StringIO(txt))
                        if len(df) and "latitude" in df.columns:
                            frames.append(df)
                    except Exception:
                        pass
                    time.sleep(0.5)
        if not frames:
            print(f"[firms] no data {year}"); continue
        full = pd.concat(frames, ignore_index=True).drop_duplicates()
        full.to_csv(out, index=False)
        print(f"[firms] {year}: {len(full)} pts  FRP={full.get('frp', pd.Series()).sum():.0f}")
    print("[firms] done")
