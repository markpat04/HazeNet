"""
HazeNet Mission Control — zero-dependency dashboard server.

ใช้แค่ Python stdlib + (xarray/pandas/numpy ที่มีใน env hazenet อยู่แล้ว)
ไม่ต้อง pip install อะไรเพิ่ม.

Run:
  C:/Users/mark/miniconda3/Scripts/conda run -n hazenet --no-capture-output python hazenet_dashboard/serve.py
หรือ ดับเบิลคลิก run.bat

เปิดเบราว์เซอร์: http://localhost:8765
"""
import os, sys, json, io, ast, glob, time, threading, subprocess, http.server, socketserver, webbrowser
from urllib.parse import urlparse, parse_qs

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)                       # โฟลเดอร์ internship
SRC   = os.path.join(ROOT, "src")
RAW   = os.path.join(ROOT, "data", "raw_m2")
PROC  = os.path.join(ROOT, "data", "processed_m2")
MODELS= os.path.join(ROOT, "models")
FIGS  = os.path.join(ROOT, "figures")
RUNS  = os.path.join(HERE, "runs")                  # live training logs (จาก tracker.py)
CACHE = os.path.join(HERE, "cache")
LOGS  = os.path.join(HERE, "logs")                  # log การรันสคริปต์จาก dashboard
SEED  = os.path.join(HERE, "experiments_seed.json")
PORT  = 8765

os.makedirs(RUNS, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# subprocess ที่กำลังรัน (จากปุ่ม Run stage)
PROCS = {}   # name -> {"proc":Popen, "log":path, "started":ts}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ───────────────────────── helpers ─────────────────────────
def mtime(p):
    try: return os.path.getmtime(p)
    except OSError: return 0.0

def newest(paths):
    return max([mtime(p) for p in paths], default=0.0)

def fsize(p):
    try:
        if os.path.isdir(p):
            tot = 0
            for dp, _, fs in os.walk(p):
                for f in fs: tot += os.path.getsize(os.path.join(dp, f))
            return tot
        return os.path.getsize(p)
    except OSError: return 0

def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024: return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def ago(ts):
    if not ts: return "—"
    d = time.time() - ts
    if d < 60: return f"{d:.0f}s ago"
    if d < 3600: return f"{d/60:.0f}m ago"
    if d < 86400: return f"{d/3600:.0f}h ago"
    return f"{d/86400:.0f}d ago"

def docstring_of(pyfile):
    try:
        with open(pyfile, encoding="utf-8") as f:
            mod = ast.parse(f.read())
        ds = ast.get_docstring(mod) or ""
        for line in ds.splitlines():
            if line.strip(): return line.strip()
    except Exception: pass
    return ""

def exists_glob(pattern):
    return sorted(glob.glob(pattern))


# ───────────────────────── PIPELINE STATUS ─────────────────────────
def pipeline_status():
    """สแกนไฟล์จริง → สถานะแต่ละ stage + ตรวจ stale (โค้ด/อินพุตใหม่กว่าเอาต์พุต)."""
    def script(name):
        p = os.path.join(SRC, name)
        return {"name": name, "exists": os.path.exists(p),
                "mtime": mtime(p), "ago": ago(mtime(p)), "doc": docstring_of(p)}

    def stage(key, title, scripts, out_paths, extra_inputs=None):
        outs = []
        for op in out_paths:
            for g in exists_glob(op):
                outs.append({"path": os.path.relpath(g, ROOT).replace("\\", "/"),
                             "size": human(fsize(g)), "mtime": mtime(g), "ago": ago(mtime(g))})
        scr = [script(s) for s in scripts]
        in_mtime = newest([os.path.join(SRC, s) for s in scripts] + (extra_inputs or []))
        out_mtime = newest([os.path.join(ROOT, o["path"]) for o in outs])
        if not outs:
            status = "missing"
        elif in_mtime > out_mtime:
            status = "stale"          # โค้ดหรืออินพุตเปลี่ยนหลัง build ล่าสุด
        else:
            status = "ok"
        return {"key": key, "title": title, "status": status,
                "scripts": scr, "outputs": outs,
                "out_mtime": out_mtime, "ago": ago(out_mtime)}

    stages = [
        stage("download", "1 · Data Ingestion",
              ["download_firms_m2.py", "download_openmeteo_met_m2.py",
               "download_dem_m2.py", "download_pm25_m2.py"],
              [os.path.join(RAW, "firms", "*.csv"),
               os.path.join(RAW, "openmeteo", "*.nc"),
               os.path.join(RAW, "dem", "*.tif"),
               os.path.join(RAW, "pm25", "*.csv")]),
        stage("grid", "2 · Regrid → master grid",
              ["build_grid_m2.py"],
              [os.path.join(PROC, "grid_m2.nc")],
              extra_inputs=[RAW]),
        stage("datacube", "3 · Datacube + targets",
              ["build_datacube_m2.py"],
              [os.path.join(PROC, "datacube_m2.zarr"),
               os.path.join(PROC, "target_pm25_m2.csv")],
              extra_inputs=[os.path.join(PROC, "grid_m2.nc")]),
        stage("train", "4 · Train CLNO",
              ["train_operator_m2.py", "model_operator.py"],
              [os.path.join(MODELS, "clno_m2.pt"),
               os.path.join(MODELS, "metrics.json")],
              extra_inputs=[os.path.join(PROC, "datacube_m2.zarr")]),
        stage("eval", "5 · Evaluate + diagnose",
              ["eval_operator_m2.py", "diagnose_m2.py", "loyo_m2.py", "sweep_m2.py"],
              [os.path.join(FIGS, "model_comparison_m2.png"),
               os.path.join(FIGS, "*loyo*.png"),
               os.path.join(FIGS, "clno_m2_*.png")],
              extra_inputs=[os.path.join(MODELS, "clno_m2.pt")]),
        stage("viz", "6 · Visualizations",
              ["plot_datacube_overview.py", "plot_pm25_crisis.py", "animate_smoke.py"],
              [os.path.join(FIGS, "datacube_m2_overview.png"),
               os.path.join(FIGS, "pm25_crisis_overlay.png"),
               os.path.join(FIGS, "smoke_animation.gif")],
              extra_inputs=[os.path.join(PROC, "grid_m2.nc")]),
    ]
    return {"stages": stages, "scanned": time.strftime("%Y-%m-%d %H:%M:%S")}


# ───────────────────────── DATA / EDA (cached) ─────────────────────────
def _eda_sources_sig():
    paths = [os.path.join(PROC, "datacube_m2.zarr"),
             os.path.join(PROC, "grid_m2.nc"),
             os.path.join(PROC, "target_pm25_m2.csv")]
    return newest(paths)

def compute_eda():
    """อ่าน datacube/grid/target จริง → สถิติ EDA. หนัก → cache ไว้."""
    cache_f = os.path.join(CACHE, "eda.json")
    sig = _eda_sources_sig()
    if os.path.exists(cache_f):
        try:
            c = json.load(open(cache_f, encoding="utf-8"))
            if abs(c.get("_sig", 0) - sig) < 1e-6:
                return c
        except Exception: pass

    import numpy as np, pandas as pd, xarray as xr
    out = {"_sig": sig, "ok": True}

    try:
        cube = xr.open_zarr(os.path.join(PROC, "datacube_m2.zarr"))
        X = cube.X
        T, C, H, W = [int(s) for s in X.shape]
        chans = [str(c) for c in cube.channel.values]
        out["cube"] = {"T": T, "C": C, "H": H, "W": W, "G": H * W,
                       "channels": chans}
        per = []
        for i, name in enumerate(chans):
            arr = X.isel(channel=i)
            vals = arr.values.astype("float64")
            finite = np.isfinite(vals)
            n_nan = int((~finite).sum())
            v = vals[finite]
            per.append({"channel": name,
                        "min": float(v.min()) if v.size else None,
                        "max": float(v.max()) if v.size else None,
                        "mean": float(v.mean()) if v.size else None,
                        "nan_pct": round(100 * n_nan / vals.size, 2)})
        out["channels"] = per
    except Exception as e:
        out["cube_error"] = str(e)

    # target PM2.5
    try:
        tgt = pd.read_csv(os.path.join(PROC, "target_pm25_m2.csv"))
        tgt["date"] = pd.to_datetime(tgt["date"])
        sta = tgt.groupby("locationId").first().reset_index()
        thai = int((sta["lat"] >= 14.5).sum())
        out["pm25"] = {
            "rows": int(len(tgt)),
            "stations": int(sta["locationId"].nunique()),
            "thai": thai, "other": int(len(sta) - thai),
            "min": float(tgt["pm25"].min()), "max": float(tgt["pm25"].max()),
            "mean": float(tgt["pm25"].mean()),
        }
        # daily mean pm25 series + per-year
        g = tgt.groupby("date")["pm25"].mean()
        out["pm25_series"] = {"date": [d.strftime("%Y-%m-%d") for d in g.index],
                              "value": [round(float(x), 1) for x in g.values]}
        yr = tgt.copy(); yr["year"] = yr["date"].dt.year
        ystat = (yr.groupby("year")["pm25"].agg(["mean", "max", "count"])
                   .reset_index())
        out["pm25_by_year"] = [{"year": int(r.year), "mean": round(float(r["mean"]), 1),
                                "max": round(float(r["max"]), 1), "n": int(r["count"])}
                               for _, r in ystat.iterrows()]
        # station list (for map)
        out["stations"] = [{"lat": float(r.lat), "lon": float(r.lon)}
                           for _, r in sta.iterrows()]
    except Exception as e:
        out["pm25_error"] = str(e)

    # FRP daily total
    try:
        grid = xr.open_dataset(os.path.join(PROC, "grid_m2.nc"))
        frp = grid.emission.sum(dim=["lat", "lon"]).values
        times = pd.DatetimeIndex(grid.time.values)
        out["frp_series"] = {"date": [d.strftime("%Y-%m-%d") for d in times],
                             "value": [round(float(x), 0) for x in frp]}
    except Exception as e:
        out["frp_error"] = str(e)

    json.dump(out, open(cache_f, "w", encoding="utf-8"))
    return out


# ───────────────────────── METRICS / EXPERIMENTS ─────────────────────────
def read_metrics():
    p = os.path.join(MODELS, "metrics.json")
    if os.path.exists(p):
        try: return json.load(open(p, encoding="utf-8"))
        except Exception: return {}
    return {}

def read_experiments():
    """seed (ผลที่ทราบจริง) + runs registry จาก tracker."""
    exps = []
    if os.path.exists(SEED):
        try: exps += json.load(open(SEED, encoding="utf-8"))
        except Exception: pass
    # runs/<id>/meta.json (เสร็จแล้ว)
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        mp = os.path.join(d, "meta.json")
        if os.path.exists(mp):
            try:
                m = json.load(open(mp, encoding="utf-8"))
                m["_run"] = os.path.basename(d)
                exps.append(m)
            except Exception: pass
    return exps


# ───────────────────────── LIVE TRAINING ─────────────────────────
def list_runs():
    runs = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*")), key=mtime, reverse=True):
        meta = {}
        mp = os.path.join(d, "meta.json")
        if os.path.exists(mp):
            try: meta = json.load(open(mp, encoding="utf-8"))
            except Exception: pass
        runs.append({"id": os.path.basename(d), "mtime": mtime(d),
                     "ago": ago(mtime(d)), "status": meta.get("status", "?"),
                     "model": meta.get("model", "?")})
    return runs

def run_progress(run_id):
    d = os.path.join(RUNS, run_id)
    meta, epochs = {}, []
    mp = os.path.join(d, "meta.json")
    if os.path.exists(mp):
        try: meta = json.load(open(mp, encoding="utf-8"))
        except Exception: pass
    pj = os.path.join(d, "progress.jsonl")
    if os.path.exists(pj):
        for line in open(pj, encoding="utf-8"):
            line = line.strip()
            if line:
                try: epochs.append(json.loads(line))
                except Exception: pass
    return {"id": run_id, "meta": meta, "epochs": epochs}

def latest_run():
    rs = list_runs()
    if not rs: return {"id": None, "meta": {}, "epochs": []}
    return run_progress(rs[0]["id"])


# ───────────────────────── FIGURES ─────────────────────────
def list_figures():
    figs = []
    for ext in ("*.png", "*.gif"):
        for f in exists_glob(os.path.join(FIGS, ext)):
            figs.append({"name": os.path.basename(f),
                         "url": "/figures/" + os.path.basename(f),
                         "size": human(fsize(f)), "mtime": mtime(f), "ago": ago(mtime(f))})
    figs.sort(key=lambda x: x["mtime"], reverse=True)
    return figs


# ───────────────────────── RUN SCRIPT (subprocess) ─────────────────────────
def run_script(name):
    """รันสคริปต์ src/<name> ด้วย python ตัวเดียวกับ server (env hazenet). whitelist .py ใน src."""
    if not name.endswith(".py") or "/" in name or "\\" in name or ".." in name:
        return {"error": "bad name"}
    path = os.path.join(SRC, name)
    if not os.path.exists(path):
        return {"error": "not found", "name": name}
    cur = PROCS.get(name)
    if cur and cur["proc"].poll() is None:
        return {"status": "already-running", "name": name}
    logpath = os.path.join(LOGS, name.replace(".py", "") + ".log")
    logf = open(logpath, "w", encoding="utf-8")
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "-u", path], cwd=ROOT, env=env,
                            stdout=logf, stderr=subprocess.STDOUT)
    PROCS[name] = {"proc": proc, "log": logpath, "started": time.time()}
    return {"status": "started", "name": name, "pid": proc.pid}

def script_status(name):
    info = PROCS.get(name)
    logpath = os.path.join(LOGS, name.replace(".py", "") + ".log")
    log = ""
    if os.path.exists(logpath):
        try:
            with open(logpath, encoding="utf-8", errors="replace") as f:
                log = f.read()[-6000:]
        except Exception: pass
    if not info:
        return {"name": name, "state": "idle", "log": log}
    rc = info["proc"].poll()
    state = "running" if rc is None else ("done" if rc == 0 else f"failed({rc})")
    return {"name": name, "state": state, "log": log,
            "elapsed": round(time.time() - info["started"], 1)}

def any_running():
    return [n for n, v in PROCS.items() if v["proc"].poll() is None]


# ───────────────────────── HTTP HANDLER ─────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        if not os.path.exists(path):
            return self._send(404, {"error": "not found", "path": path})
        with open(path, "rb") as f: data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # เงียบ
        pass

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        try:
            if p == "/" or p == "/index.html":
                return self._file(os.path.join(HERE, "index.html"), "text/html; charset=utf-8")
            if p == "/static/style.css":
                return self._file(os.path.join(HERE, "static", "style.css"), "text/css; charset=utf-8")
            if p == "/static/app.js":
                return self._file(os.path.join(HERE, "static", "app.js"), "application/javascript; charset=utf-8")
            if p.startswith("/figures/"):
                name = os.path.basename(p)
                ext = name.rsplit(".", 1)[-1].lower()
                ct = {"png": "image/png", "gif": "image/gif", "jpg": "image/jpeg"}.get(ext, "application/octet-stream")
                return self._file(os.path.join(FIGS, name), ct)

            # ── APIs ──
            if p == "/api/pipeline":   return self._send(200, pipeline_status())
            if p == "/api/eda":        return self._send(200, compute_eda())
            if p == "/api/metrics":    return self._send(200, read_metrics())
            if p == "/api/experiments":return self._send(200, read_experiments())
            if p == "/api/runs":       return self._send(200, list_runs())
            if p == "/api/run/latest": return self._send(200, latest_run())
            if p == "/api/run":
                rid = parse_qs(u.query).get("id", [""])[0]
                return self._send(200, run_progress(rid))
            if p == "/api/figures":    return self._send(200, list_figures())
            if p == "/api/health":     return self._send(200, {"ok": True, "root": ROOT,
                                                              "running": any_running()})
            # run-script (subprocess)
            if p == "/api/run_script":
                name = parse_qs(u.query).get("name", [""])[0]
                return self._send(200, run_script(name))
            if p == "/api/script_status":
                name = parse_qs(u.query).get("name", [""])[0]
                return self._send(200, script_status(name))
            # attribution explorer
            if p == "/api/attrib/meta":
                import attrib
                return self._send(200, attrib.meta())
            if p == "/api/attrib":
                q = parse_qs(u.query)
                import attrib
                return self._send(200, attrib.attribution(
                    int(q.get("day", ["0"])[0]), int(q.get("station", ["0"])[0])))

            return self._send(404, {"error": "no route", "path": p})
        except Exception as e:
            import traceback
            return self._send(500, {"error": str(e), "trace": traceback.format_exc()})


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    srv = ThreadingServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print("=" * 56)
    print("  HazeNet Mission Control")
    print(f"  → {url}")
    print(f"  root: {ROOT}")
    print("  (Ctrl+C เพื่อหยุด)")
    print("=" * 56)
    try:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
