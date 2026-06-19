"""Reusable LOYO runner.  Usage:  python run_loyo.py <config.yaml>"""
import os, sys, traceback
os.chdir("C:/Users/mark/Desktop/internship")
sys.path.insert(0, "C:/Users/mark/Desktop/internship")
os.environ.setdefault("GDAL_DATA", "C:/Users/mark/miniconda3/envs/hazenet/Library/share/gdal")

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/local_cds.yaml"
print(f"Running LOYO for {cfg_path} ...", flush=True)
try:
    from hazenet.config import Config
    from hazenet.loyo import loyo
    cfg = Config.load(cfg_path)
    print(f"  lds={getattr(cfg, 'lds', False)}  reweight={getattr(cfg, 'lds_reweight', '-')}", flush=True)
    out = loyo(cfg)
    print(f"\n=== LOYO DONE [{cfg.name}] ===", flush=True)
    print(f"mean_MAE={out['mean_MAE']:.2f}  worst={out['worst_MAE']:.2f}"
          f"  SEEN={out.get('seen_mean_MAE', float('nan')):.2f}"
          f"  NEW={out.get('new_mean_MAE', float('nan')):.2f}", flush=True)
    for f in out["folds"]:
        print(f"  {f['year']}: MAE={f['MAE']:.2f}  bias={f['bias']:+.2f}", flush=True)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
