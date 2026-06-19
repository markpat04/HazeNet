"""LOSO runner — k-fold Leave-One-Station-Out spatial-generalization gate.
Usage:  python run_loso.py <config.yaml> [k]"""
import os, sys, traceback
ROOT = os.path.dirname(os.path.abspath(__file__))
_gdal = os.path.join(sys.prefix, "Library", "share", "gdal")
if os.path.isdir(_gdal):
    os.environ.setdefault("GDAL_DATA", _gdal)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/local_cds.yaml"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
print(f"Running LOSO (k={k}) for {cfg_path} ...", flush=True)
try:
    from hazenet.config import Config
    from hazenet.loyo import loso
    from hazenet.tracking import Experiment
    cfg = Config.load(cfg_path)
    out = loso(cfg, k=k)
    # log to manifest + W&B
    exp = Experiment(name=f"{cfg.name}:LOSO_k{k}", project="hazenet",
                     config={"model_kind": cfg.model_kind, "k": k, "seed": cfg.seed,
                             "validation": "LOSO"},
                     data_path=cfg.datacube_zarr, tags=[cfg.name, "loso"],
                     notes="leave-one-station-out")
    for f in out["folds"]:
        exp.log({f"LOSO_MAE/fold{f['fold']}": f["MAE"]})
    exp.finish({"mean_MAE": out["mean_MAE"], "worst_MAE": out["worst_MAE"]})
    print(f"\n=== LOSO DONE [{cfg.name}] ===  mean_MAE={out['mean_MAE']:.2f}  "
          f"worst-fold={out['worst_MAE']:.2f}", flush=True)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
