"""Full local pipeline in ONE process (warms torch before eval's torch.load,
avoiding the Windows cold-start zarr+torch segfault). Produces final figures.
Usage:  python run_pipeline.py [config.yaml] [stages]"""
import os, sys, traceback
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# repo root = this file's directory (cross-platform: Windows local + Linux server)
ROOT = os.path.dirname(os.path.abspath(__file__))
# GDAL_DATA: Windows conda needs it explicitly; on Linux conda sets it already.
_gdal = os.path.join(sys.prefix, "Library", "share", "gdal")
if os.path.isdir(_gdal):
    os.environ.setdefault("GDAL_DATA", _gdal)
# NB: do NOT import torch here — each stage opens zarr BEFORE importing torch,
# the order required to avoid the Windows OpenMP/pyarrow segfault.
os.chdir(ROOT)
sys.path.insert(0, ROOT)

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/local_cds.yaml"
stages = (sys.argv[2].split(",") if len(sys.argv) > 2
          else ["train", "eval", "loyo", "loso"])

print(f"Pipeline {cfg_path}  stages={stages}", flush=True)
try:
    from hazenet.config import Config
    from hazenet.cli import _run_stage
    cfg = Config.load(cfg_path)
    for stage in stages:
        _run_stage(stage, cfg)
    print("ALL STAGES DONE", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
